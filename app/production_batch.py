"""Resumable, SQL-free production processing for the MorningTalk recording set."""

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid

import requests
from requests.adapters import HTTPAdapter

from app.benchmark import (
    DEFAULT_RECORDING_ROOT,
    DEFAULT_SARA_ROOT,
    DEFAULT_TRANSCRIPT_ROOT,
    discover_inventory,
    extract_pdf_pages,
    extract_reference_facts,
    generate_benchmark_reports,
)
from app.config import get_azure_config, get_azure_openai_api_version
from app.file_manager import (
    resolve_meeting_output_paths,
    source_fingerprint,
)
from app.factory_context import (
    normalize_transcript_segments,
    normalization_enabled,
)
from app.meeting_analysis import GPTMeetingAnalyzer, MeetingAnalysis
from app.phase10_6 import (
    DEFAULT_PREVIEW_ROOT,
    DEFAULT_REVIEW_ROOT,
    ReviewWorkflowError,
    load_validated_transcript,
)
from app.phase10_7 import (
    PilotBatchError,
    _TrackedSession,
    _atomic_json,
    _file_hash,
    _manifest_lock,
    _set_stage_complete,
    _stage_config_fingerprints,
    _write_stage_artifact,
)
from app.pdf_export import (
    _extract_pdf_text,
    _normalized_text,
    write_meeting_summary_pdf,
    write_transcript_pdf,
)
from app.meeting_outputs import publish_staged_output_set, write_transcript_text
from app.transcript import (
    TermNormalizationRecord,
    TranscriptDocument,
    TranscriptQualityFinding,
    TranscriptSegment,
    mark_for_review,
)
from app.transcript_cleanup import TranscriptCleanup
from app.transcriber import AzureOpenAITranscriber, estimate_chunk_count
from app.workspace_paths import (
    DEFAULT_PREVIEW_ROOT,
    DEFAULT_PRODUCTION_BATCH_ROOT,
    DEFAULT_REVIEW_ROOT,
    DEFAULT_TEMP_ROOT,
    rebase_manifest_runtime_paths,
)

PRIVATE_ROOT = DEFAULT_PRODUCTION_BATCH_ROOT
SCHEMA_VERSION = 1
_STAGES = ("transcription", "cleanup", "analysis")


class ProductionBatchError(RuntimeError):
    """Safe-to-report batch preflight, execution, or resume failure."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _recording_relative_path(source: Path, recording_root: Path) -> str:
    return source.resolve().relative_to(recording_root.resolve()).as_posix()


def _validated_phase9(inventory_item, source: Path):
    valid = []
    for path in inventory_item.phase9_json_paths:
        try:
            transcript = load_validated_transcript(path, source)
        except (ReviewWorkflowError, OSError, ValueError):
            continue
        recorded_sha = transcript.transcript_payload.get("source_sha256")
        if recorded_sha is not None and recorded_sha != _sha256_file(source):
            continue
        valid.append(transcript)
    content_hashes = {
        _json_hash((item.transcript_payload["raw_transcript"],
                    item.transcript_payload["cleaned_transcript"]))
        for item in valid
    }
    if len(content_hashes) > 1:
        raise ProductionBatchError(
            f"ConflictingValidatedTranscripts:{source.name}"
        )
    return max(valid, key=lambda item: item.path.stat().st_mtime_ns) if valid else None


def _validated_analysis(inventory_item, source: Path, transcript):
    valid = []
    if transcript is None:
        return None
    for path in inventory_item.phase10_json_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            continue
        if (
            payload.get("source_fingerprint") != source_fingerprint(source)
            or payload.get("phase9_json_sha256") != transcript.sha256
        ):
            continue
        analysis_payload = payload.get("meeting_analysis")
        if not isinstance(analysis_payload, dict):
            continue
        try:
            analysis = GPTMeetingAnalyzer._parse_and_validate(
                analysis_payload,
                transcript.segments,
            )
        except (TypeError, ValueError):
            continue
        populated = sum(
            len(getattr(analysis, name))
            if isinstance(getattr(analysis, name), tuple)
            else int(getattr(analysis, name) is not None)
            for name in (
                "overview", "topics", "classifications", "key_issues",
                "actions", "dependencies", "schedules", "risks",
                "closing_activities",
            )
        )
        if analysis.status == "validated" or (
            analysis.status == "review_required"
            and analysis.review_reason == "SemanticReviewRequired"
            and populated
        ):
            valid.append((path, payload, analysis))
    signatures = {_json_hash(item[1].get("meeting_analysis")) for item in valid}
    if len(signatures) > 1:
        raise ProductionBatchError(
            f"ConflictingReusableAnalyses:{source.name}"
        )
    return max(valid, key=lambda item: item[0].stat().st_mtime_ns) if valid else None


def _configuration(config, api_version: str) -> dict[str, object]:
    config.validate_transcription()
    if not config.text_endpoint or not config.text_api_key or not config.text_deployment:
        raise ProductionBatchError("MeetingAnalysisConfigurationIncomplete")
    from app.azure_openai_http import resolve_azure_openai_endpoint
    from app.factory_context import (
        TRANSCRIPTION_PROMPT_API_VERSIONS,
        transcription_prompt_enabled,
    )
    from urllib.parse import urlparse

    endpoint = resolve_azure_openai_endpoint(
        config.text_endpoint,
        api_version,
        config.text_deployment,
    )
    return {
        "transcription_deployment": config.transcription_deployment,
        "transcription_api_version": config.transcription_api_version,
        "transcription_endpoint_host": urlparse(config.transcription_endpoint).hostname,
        "transcription_prompt_supported_by_code": (
            config.transcription_api_version.casefold()
            in TRANSCRIPTION_PROMPT_API_VERSIONS
        ),
        "transcription_prompt_enabled": transcription_prompt_enabled(
            config.transcription_api_version
        ),
        "text_deployment": config.text_deployment,
        "text_api_version": api_version,
        "text_endpoint_host": urlparse(config.text_endpoint).hostname,
        "text_endpoint_mode": endpoint.api_mode,
        "chunk_duration_seconds": config.chunk_duration_seconds,
        "chunk_overlap_seconds": config.chunk_overlap_seconds,
    }


def _upgrade_verified_prompt_configuration(
    manifest_path: Path,
    manifest: dict[str, object],
    current_configuration: dict[str, object],
    current_fingerprints: dict[str, str],
) -> None:
    previous_configuration = manifest.get("configuration")
    previous_fingerprints = manifest.get("configuration_fingerprints")
    if (
        previous_configuration == current_configuration
        and previous_fingerprints == current_fingerprints
    ):
        return
    if not isinstance(previous_configuration, dict) or not isinstance(previous_fingerprints, dict):
        raise ProductionBatchError("AzureConfigurationChangedSincePreflight")
    old_without_support = dict(previous_configuration)
    new_without_support = dict(current_configuration)
    old_support = old_without_support.pop("transcription_prompt_supported_by_code", None)
    new_support = new_without_support.pop("transcription_prompt_supported_by_code", None)
    old_fingerprints = dict(previous_fingerprints)
    new_fingerprints = dict(current_fingerprints)
    old_transcription_fingerprint = old_fingerprints.pop("transcription", None)
    new_transcription_fingerprint = new_fingerprints.pop("transcription", None)
    if (
        old_without_support != new_without_support
        or old_support is not False
        or new_support is not True
        or old_fingerprints != new_fingerprints
        or old_transcription_fingerprint == new_transcription_fingerprint
        or current_configuration.get("transcription_api_version") != "2025-03-01-preview"
    ):
        raise ProductionBatchError("AzureConfigurationChangedSincePreflight")
    manifest.setdefault("configuration_history", []).append(
        {
            "recorded_at_utc": _now(),
            "reason": "VerifiedAzureTranscriptionPromptSupport",
            "configuration": previous_configuration,
            "configuration_fingerprints": previous_fingerprints,
        }
    )
    manifest["configuration"] = current_configuration
    manifest["configuration_fingerprints"] = current_fingerprints
    manifest.setdefault("events", []).append(
        {
            "timestamp_utc": _now(),
            "event": "transcription_prompt_support_verified",
            "api_version": "2025-03-01-preview",
        }
    )
    manifest["updated_at_utc"] = _now()
    _atomic_json(manifest_path, manifest)


def prepare_production_batch(
    *,
    batch_id: str | None = None,
    recording_root: Path = DEFAULT_RECORDING_ROOT,
    transcript_root: Path = DEFAULT_TRANSCRIPT_ROOT,
    sara_root: Path = DEFAULT_SARA_ROOT,
    preview_root: Path = DEFAULT_PREVIEW_ROOT,
    review_root: Path = DEFAULT_REVIEW_ROOT,
    private_root: Path = PRIVATE_ROOT,
    config=None,
    api_version: str | None = None,
) -> tuple[Path, dict[str, object]]:
    config = config or get_azure_config()
    api_version = api_version or get_azure_openai_api_version()
    configuration = _configuration(config, api_version)
    recording_root = recording_root.resolve()
    transcript_root = transcript_root.resolve()
    sara_root = sara_root.resolve()
    preview_root = preview_root.resolve()
    review_root = review_root.resolve()
    private_root = private_root.resolve()
    batch_id = batch_id or datetime.now().strftime("batch-%Y%m%d-%H%M%S")
    if Path(batch_id).name != batch_id or batch_id in {"", ".", ".."}:
        raise ProductionBatchError("BatchIdInvalid")
    batch_root = private_root / batch_id
    if batch_root.exists():
        raise ProductionBatchError("BatchDirectoryAlreadyExists")
    stage_root = batch_root / "stages"
    response_root = batch_root / "raw_responses"
    report_path = batch_root / "final_batch_report.md"
    inventory = discover_inventory(
        recording_root,
        transcript_root,
        sara_root,
        private_root=preview_root.parent,
    )
    recordings = []
    config_hashes = _stage_config_fingerprints(
        config,
        api_version,
        os.getenv("MORNINGTALK_TERMINOLOGY", ""),
    )
    for item in inventory.meetings:
        source = item.mp3_path.resolve()
        source_sha256 = _sha256_file(source)
        duration = AzureOpenAITranscriber._probe_duration(source)
        if duration is None or duration <= 0:
            raise ProductionBatchError(f"AudioDurationUnavailable:{source.name}")
        chunk_count = estimate_chunk_count(
            duration,
            config.chunk_duration_seconds,
            config.chunk_overlap_seconds,
        )
        output_paths = resolve_meeting_output_paths(
            source,
            transcript_root=transcript_root,
        )
        phase9 = _validated_phase9(item, source)
        analysis_cache = _validated_analysis(item, source, phase9)
        relative = _recording_relative_path(source, recording_root)
        phase9_artifact = (
            {
                "path": str(phase9.path),
                "sha256": phase9.sha256,
            }
            if phase9 else None
        )
        analysis_artifact = (
            {
                "path": str(analysis_cache[0]),
                "sha256": _sha256_file(analysis_cache[0]),
            }
            if analysis_cache else None
        )
        phase9_cached = phase9 is not None
        analysis_cached = analysis_cache is not None
        stage_statuses = {
            "transcription": "reused" if phase9_cached else "pending",
            "cleanup": "reused" if phase9_cached else "pending",
            "analysis": "reused" if analysis_cached else (
                "pending" if phase9_cached else "blocked_until_cleanup"
            ),
        }
        required_requests = (
            0 if phase9_cached
            else chunk_count + 1 + (0 if analysis_cached else 1)
        )
        if phase9_cached and not analysis_cached:
            required_requests = 1
        recordings.append(
            {
                "recording_relative_path": relative,
                "source": str(source),
                "source_fingerprint": source_fingerprint(source),
                "source_sha256": source_sha256,
                "duration_seconds": round(duration, 3),
                "expected_audio_chunks": chunk_count,
                "sara_pdf": str(item.sara_pdf_path) if item.sara_pdf_path else None,
                "sara_match_status": item.sara_match_status,
                "sara_pdf_sha256": (
                    _sha256_file(item.sara_pdf_path)
                    if item.sara_pdf_path else None
                ),
                "phase9_artifact": phase9_artifact,
                "normalization_artifact": None,
                "analysis_artifact": analysis_artifact,
                "prior_phase9_artifacts": [str(path) for path in item.phase9_json_paths],
                "prior_phase10_artifacts": [str(path) for path in item.phase10_json_paths],
                "prior_analysis_request_count": sum(
                    int(
                        (json.loads(path.read_text(encoding="utf-8"))
                         .get("meeting_analysis_requests", 0))
                        or 0
                    )
                    for path in item.phase10_json_paths
                    if path.is_file()
                ),
                "stages": {
                    name: {
                        "status": stage_statuses[name],
                        "origin": (
                            "reused" if stage_statuses[name] == "reused" else "new"
                        ),
                        "artifact": (
                            phase9_artifact if name in {"transcription", "cleanup"}
                            and phase9_cached
                            else analysis_artifact if name == "analysis" and analysis_cached
                            else None
                        ),
                        "validation": (
                            analysis_cache[2].status
                            if name == "analysis" and analysis_cached
                            else None
                        ),
                        "review_reason": (
                            analysis_cache[2].review_reason
                            if name == "analysis" and analysis_cached
                            else None
                        ),
                        "config_fingerprint": (
                            config_hashes[name] if stage_statuses[name] == "reused"
                            else None
                        ),
                    }
                    for name in _STAGES
                },
                "outputs": {
                    "transcript_txt": str(output_paths.transcript_text),
                    "full_transcript_pdf": str(output_paths.full_transcript_pdf),
                    "summary_draft_pdf": str(output_paths.summary_draft_pdf),
                    "approved_summary_pdf": str(output_paths.approved_summary_pdf),
                    "approved_summary_preexisting": output_paths.approved_summary_pdf.exists(),
                    "status": "pending",
                },
                "expected_new_requests": required_requests,
                "status": "pending",
            }
        )
    recordings.sort(
        key=lambda item: (
            0 if Path(item["source"]).stem == "meeting-20261107-0817" else 1,
            item["recording_relative_path"].casefold(),
        )
    )
    planned_requests = sum(item["expected_new_requests"] for item in recordings)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "batch_id": batch_id,
        "created_at_utc": _now(),
        "updated_at_utc": _now(),
        "execution_status": "preflight_ready",
        "configuration": configuration,
        "configuration_fingerprints": config_hashes,
        "paths": {
            "recording_root": str(recording_root),
            "transcript_root": str(transcript_root),
            "sara_root": str(sara_root),
            "preview_root": str(preview_root),
            "review_root": str(review_root),
            "batch_root": str(batch_root),
            "stage_root": str(stage_root),
            "response_root": str(response_root),
            "report_path": str(report_path),
        },
        "request_budget": {
            "planned_total": planned_requests,
            "per_run_limit": max(1, planned_requests),
        },
        "recordings": recordings,
        "request_attempts": [],
        "events": [],
    }
    batch_root.mkdir(parents=True, exist_ok=False)
    _atomic_json(batch_root / "manifest.json", manifest)
    _write_preflight_report(report_path, manifest)
    return batch_root / "manifest.json", manifest


def _write_preflight_report(path: Path, manifest: dict[str, object]) -> None:
    recordings = manifest["recordings"]
    lines = [
        "# MorningTalk production batch preflight",
        "",
        f"- Batch: `{manifest['batch_id']}`",
        f"- Recording count: {len(recordings)}",
        f"- Matched SARA references: "
        f"{sum(item['sara_match_status'] == 'matched' for item in recordings)}",
        f"- Estimated new Azure requests: "
        f"{manifest['request_budget']['planned_total']}",
        "- Automatic retries: disabled",
        "- Token estimates: unavailable; actual service usage is recorded when returned",
        "- SQL writes: disabled",
        "- Approved summary PDFs: never written by this batch",
        "",
        "## Configuration",
        "",
    ]
    lines.extend(
        f"- {name}: {value}"
        for name, value in manifest["configuration"].items()
    )
    lines.extend(["", "## Recordings", ""])
    for item in recordings:
        lines.extend(
            [
                f"### {item['recording_relative_path']}",
                "",
                f"- Duration: {item['duration_seconds']} seconds",
                f"- Audio chunks: {item['expected_audio_chunks']}",
                f"- SARA match: {item['sara_match_status']}",
                f"- Cached Phase 9: "
                f"{item['stages']['cleanup']['status'] == 'reused'}",
                f"- Cached analysis: "
                f"{item['stages']['analysis']['status'] == 'reused'}",
                f"- Planned new requests: {item['expected_new_requests']}",
                f"- Existing approved PDF: "
                f"{item['outputs']['approved_summary_preexisting']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise ProductionBatchError("BatchManifestInvalid") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != SCHEMA_VERSION
        or not isinstance(payload.get("recordings"), list)
        or not isinstance(payload.get("request_attempts"), list)
    ):
        raise ProductionBatchError("BatchManifestInvalid")
    return rebase_manifest_runtime_paths(payload, path)


def _source_is_unchanged(recording: dict[str, object]) -> Path:
    source = Path(recording["source"])
    if (
        not source.is_file()
        or source_fingerprint(source) != recording["source_fingerprint"]
        or _sha256_file(source) != recording["source_sha256"]
    ):
        raise ProductionBatchError(
            f"SourceFingerprintChanged:{recording['recording_relative_path']}"
        )
    return source


def _stage_path(manifest: dict[str, object], recording: dict[str, object], name: str) -> Path:
    relative = Path(recording["recording_relative_path"])
    return (
        Path(manifest["paths"]["stage_root"])
        / relative.parent
        / relative.stem
        / f"{relative.stem}.{name}.json"
    )


def _mark_stage_requests_complete(
    manifest: dict[str, object],
    recording: dict[str, object],
    stage: str,
    run_id: str,
) -> None:
    for attempt in manifest["request_attempts"]:
        if (
            attempt.get("recording_relative_path") == recording["recording_relative_path"]
            and attempt.get("stage") == stage
            and attempt.get("run_id") == run_id
            and attempt.get("status") == "response_received"
        ):
            attempt["status"] = "completed"
            attempt["completed_at_utc"] = _now()


def _run_stage_request(
    manifest_path: Path,
    manifest: dict[str, object],
    recording: dict[str, object],
    stage: str,
    run_id: str,
    session: requests.Session,
):
    return _TrackedSession(
        manifest_path,
        manifest,
        recording,
        stage,
        run_id,
        manifest["request_budget"].get(
            "active_run_limit",
            manifest["request_budget"]["per_run_limit"],
        ),
        session,
    )


def _transcript_document(payload: dict[str, object]) -> TranscriptDocument:
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        raise ProductionBatchError("TranscriptSegmentsInvalid")
    segments = []
    for item in raw_segments:
        if not isinstance(item, dict):
            raise ProductionBatchError("TranscriptSegmentInvalid")
        findings = tuple(
            TranscriptQualityFinding(
                finding["original_phrase"],
                finding.get("proposed_correction"),
                finding.get("surrounding_context", finding.get("context", "")),
                finding.get("uncertainty", "high"),
                True,
            )
            for finding in item.get("quality_findings", [])
            if isinstance(finding, dict)
            and isinstance(finding.get("original_phrase"), str)
        )
        segments.append(
            TranscriptSegment(
                item.get("start_seconds"),
                item.get("end_seconds"),
                item.get("speaker_id"),
                item["raw_text"],
                item.get("cleaned_text"),
                item.get("review_required", False),
                findings,
            )
        )
    return TranscriptDocument(
        tuple(segments),
        payload.get("diarization_status", "unavailable"),
        payload.get("cleanup_status", "review_required"),
        payload.get("review_reason"),
        payload.get("input_tokens"),
        payload.get("output_tokens"),
        payload.get("raw_transcript"),
        payload.get("normalized_transcript"),
        tuple(
            TermNormalizationRecord(**item)
            for item in payload.get("normalization_records", [])
            if isinstance(item, dict)
        ),
        tuple(
            item
            for item in payload.get("normalization_review_findings", [])
            if isinstance(item, dict)
        ),
    )


def _analysis_from_payload(
    payload: dict[str, object],
    document: TranscriptDocument,
) -> MeetingAnalysis:
    raw_analysis = payload.get("meeting_analysis")
    if not isinstance(raw_analysis, dict):
        return MeetingAnalysis.for_review("MeetingAnalysisMissing")
    try:
        return GPTMeetingAnalyzer._parse_and_validate(raw_analysis, document.segments)
    except (TypeError, ValueError):
        return MeetingAnalysis.for_review("AnalysisParsingFailed")


def _verify_pdf(path: Path, expected_text: str | None = None, *, draft: bool = False) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ProductionBatchError(f"PdfOutputMissing:{path.name}")
    pages, text = _extract_pdf_text(path)
    if pages < 1 or not text.strip():
        raise ProductionBatchError(f"PdfUnreadable:{path.name}")
    if expected_text is not None:
        normalized_text = _normalized_text(text)
        position = 0
        for line in expected_text.splitlines():
            normalized_line = _normalized_text(line)
            if not normalized_line:
                continue
            position = normalized_text.find(normalized_line, position)
            if position < 0:
                raise ProductionBatchError(f"PdfTranscriptIncomplete:{path.name}")
            position += len(normalized_line)
    if draft and "DRAFT - NOT APPROVED" not in text.upper():
        raise ProductionBatchError(f"SummaryDraftLabelMissing:{path.name}")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "pages": pages,
        "status": "verified",
    }


def _publish_outputs(
    recording: dict[str, object],
    source: Path,
    document: TranscriptDocument,
    analysis: MeetingAnalysis,
    *,
    overwrite_draft: bool,
    review_version_suffix: str | None = None,
) -> dict[str, object]:
    paths = resolve_meeting_output_paths(
        source,
        transcript_root=Path(recording["outputs"]["transcript_txt"]).parent.parent.parent,
    )
    plant = Path(recording["recording_relative_path"]).parts[0]
    approved_exists = paths.approved_summary_pdf.exists()
    if (
        analysis.overview is None
        or not analysis.overview.text.strip()
        or not analysis.overview.evidence
    ):
        return {
            "status": "not_published_analysis_review_required",
            "transcript_txt": {"path": str(paths.transcript_text), "status": "preserved"},
            "full_transcript_pdf": {
                "path": str(paths.full_transcript_pdf),
                "status": "preserved",
            },
            "summary_draft_pdf": {
                "path": str(paths.summary_draft_pdf),
                "status": "not_generated",
            },
            "approved_summary_pdf": {
                "path": str(paths.approved_summary_pdf),
                "exists": approved_exists,
                "status": "preserved" if approved_exists else "not_generated",
            },
        }

    relative_path = Path(str(recording["recording_relative_path"]))
    stage_root = (
        DEFAULT_PREVIEW_ROOT
        / "production_batch_staging"
        / relative_path.parent
        / source.stem
        / uuid.uuid4().hex
    )
    stage_root.mkdir(parents=True)
    staged_text = stage_root / paths.transcript_text.name
    staged_full = stage_root / paths.full_transcript_pdf.name
    staged_summary = stage_root / paths.summary_draft_pdf.name

    write_transcript_text(
        staged_text,
        document.cleaned_transcript,
        overwrite=False,
    )
    write_transcript_pdf(
        staged_full,
        source,
        plant,
        document,
        overwrite=False,
    )
    replace_draft = overwrite_draft and (
        analysis.status == "validated"
        or (
            analysis.status == "review_required"
            and analysis.overview is not None
        )
    )
    write_meeting_summary_pdf(
        staged_summary,
        source,
        plant,
        analysis,
        overwrite=False,
        draft=True,
    )
    if staged_text.read_text(encoding="utf-8") != document.cleaned_transcript:
        raise ProductionBatchError("StagedTranscriptTxtMismatch")
    _verify_pdf(
        staged_full,
        document.cleaned_transcript,
    )
    draft_info = _verify_pdf(
        staged_summary,
        analysis.overview.text,
        draft=True,
    )

    staged_outputs = {
        paths.transcript_text: staged_text,
        paths.full_transcript_pdf: staged_full,
    }
    if replace_draft or not paths.summary_draft_pdf.exists():
        staged_outputs[paths.summary_draft_pdf] = staged_summary
    else:
        _verify_pdf(
            paths.summary_draft_pdf,
            analysis.overview.text,
            draft=True,
        )

    review_version_info = None
    review_version_path = None
    if review_version_suffix:
        review_version_path = paths.directory / (
            f"{source.stem}_Summary_Draft_Review_{review_version_suffix}.pdf"
        )
        if review_version_path.exists():
            _verify_pdf(
                review_version_path,
                analysis.overview.text,
                draft=True,
            )
        else:
            staged_outputs[review_version_path] = staged_summary

    backup_dir = publish_staged_output_set(
        staged_outputs,
        approved_summary=paths.approved_summary_pdf,
        backup_root=DEFAULT_TEMP_ROOT / "output_publication_backups",
    )
    if paths.summary_draft_pdf.exists():
        draft_info = _verify_pdf(
            paths.summary_draft_pdf,
            analysis.overview.text,
            draft=True,
        )
    else:
        draft_info = {
            "path": str(paths.summary_draft_pdf),
            "status": "preserved_existing",
        }
    if review_version_path is not None and review_version_path.exists():
        review_version_info = _verify_pdf(
            review_version_path,
            analysis.overview.text,
            draft=True,
        )
    full_info = _verify_pdf(
        paths.full_transcript_pdf,
        document.cleaned_transcript,
    )
    return {
        "transcript_txt": {
            "path": str(paths.transcript_text),
            "sha256": _sha256_file(paths.transcript_text),
            "status": "verified",
        },
        "full_transcript_pdf": full_info,
        "summary_draft_pdf": draft_info,
        "summary_draft_review_version_pdf": review_version_info,
        "approved_summary_pdf": {
            "path": str(paths.approved_summary_pdf),
            "exists": approved_exists,
            "status": "preserved" if approved_exists else "not_generated",
        },
        "staging_directory": str(stage_root),
        "rollback_backup_directory": str(backup_dir),
        "status": "verified",
    }


def _process_recording(
    manifest_path: Path,
    manifest: dict[str, object],
    recording: dict[str, object],
    config,
    api_version: str,
    session: requests.Session,
    run_id: str,
) -> None:
    source = _source_is_unchanged(recording)
    config_hashes = manifest["configuration_fingerprints"]
    stages = recording["stages"]
    use_terminology_pipeline = normalization_enabled()
    if (
        recording.get("phase9_artifact") is None
        and stages["cleanup"].get("status") in {"completed_validated", "review_required"}
        and stages["cleanup"].get("artifact")
    ):
        recording["phase9_artifact"] = stages["cleanup"]["artifact"]
    if (
        recording.get("analysis_artifact") is None
        and stages["analysis"].get("status") in {"completed_validated", "review_required"}
        and stages["analysis"].get("artifact")
    ):
        recording["analysis_artifact"] = stages["analysis"]["artifact"]
    if stages["transcription"]["status"] not in {"reused", "completed"}:
        stages["transcription"]["status"] = "running"
        manifest["updated_at_utc"] = _now()
        _atomic_json(manifest_path, manifest)
        tracker = _run_stage_request(
            manifest_path, manifest, recording, "transcription", run_id, session
        )
        try:
            raw_text = AzureOpenAITranscriber(
                config,
                session=tracker,
            ).transcribe(source)
            if not raw_text.strip():
                raise ProductionBatchError("EmptyTranscription")
            source = _source_is_unchanged(recording)
            payload = {
                "source_filename": source.name,
                "source_fingerprint": recording["source_fingerprint"],
                "source_sha256": recording["source_sha256"],
                "transcription_config_fingerprint": config_hashes["transcription"],
                "raw_transcript": raw_text,
                "status": "completed",
            }
            artifact = _write_stage_artifact(
                _stage_path(manifest, recording, "raw"),
                payload,
            )
            _set_stage_complete(
                manifest_path, manifest, recording, "transcription", run_id,
                config_hashes["transcription"], "completed", artifact,
            )
        except Exception as error:
            stages["transcription"]["status"] = (
                "unknown"
                if any(
                    attempt.get("status") in {"dispatching", "unknown", "response_received"}
                    for attempt in manifest["request_attempts"]
                    if attempt.get("recording_relative_path") == recording["recording_relative_path"]
                    and attempt.get("stage") == "transcription"
                    and attempt.get("run_id") == run_id
                )
                else "failed"
            )
            stages["transcription"]["reason"] = type(error).__name__
            recording["status"] = "blocked"
            manifest["updated_at_utc"] = _now()
            _atomic_json(manifest_path, manifest)
            raise ProductionBatchError(
                f"TranscriptionStageFailed:{recording['recording_relative_path']}:{type(error).__name__}"
            ) from error

    phase9_artifact = recording["phase9_artifact"]
    if (
        use_terminology_pipeline
        and phase9_artifact is not None
        and recording.get("normalization_artifact") is None
    ):
        raise ProductionBatchError(
            "CachedPhase9MissingTerminologyNormalizationArtifact"
        )
    if phase9_artifact is None and stages["cleanup"]["status"] != "completed":
        raw_info = stages["transcription"].get("artifact")
        if raw_info is None:
            raise ProductionBatchError("RawTranscriptArtifactMissing")
        raw_payload = json.loads(Path(raw_info["path"]).read_text(encoding="utf-8"))
        raw_text = raw_payload.get("raw_transcript")
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ProductionBatchError("RawTranscriptArtifactInvalid")
        raw_segment = TranscriptSegment(None, None, None, raw_text, raw_text)
        normalization = (
            normalize_transcript_segments(
                (raw_segment,),
                recording=source.name,
            )
            if use_terminology_pipeline
            else None
        )
        normalized_segments = (
            normalization.segments if normalization is not None else (raw_segment,)
        )
        normalized_text = (
            normalization.normalized_transcript
            if normalization is not None
            else raw_text
        )
        normalization_records = (
            normalization.replacements if normalization is not None else ()
        )
        normalization_findings = (
            normalization.review_findings if normalization is not None else ()
        )
        if normalization is not None:
            normalization_artifact = _write_stage_artifact(
                _stage_path(manifest, recording, "terminology"),
                {
                    "source_filename": source.name,
                    "source_fingerprint": recording["source_fingerprint"],
                    "source_sha256": recording["source_sha256"],
                    "raw_transcript": normalization.raw_transcript,
                    "normalized_transcript": normalization.normalized_transcript,
                    "normalization_records": [
                        asdict(item) for item in normalization.replacements
                    ],
                    "normalization_review_findings": list(
                        normalization.review_findings
                    ),
                },
            )
            recording["normalization_artifact"] = normalization_artifact
            manifest["updated_at_utc"] = _now()
            _atomic_json(manifest_path, manifest)

        uncertain_cleanup_attempt = None
        if normalization_findings:
            cleaned = mark_for_review(
                normalized_segments,
                "TerminologyCandidatesRequireReview",
                diarization_status="unavailable",
            )
            stages["cleanup"]["status"] = "review_required"
            stages["cleanup"]["origin"] = "terminology_review"
        else:
            stages["cleanup"]["status"] = "running"
            stages["cleanup"]["origin"] = "new"
            manifest["updated_at_utc"] = _now()
            _atomic_json(manifest_path, manifest)
            tracker = _run_stage_request(
                manifest_path, manifest, recording, "cleanup", run_id, session
            )
            cleaned = TranscriptCleanup(
                config,
                api_version=api_version,
                session=tracker,
                terminology=os.getenv("MORNINGTALK_TERMINOLOGY", ""),
            ).clean(normalized_segments)
            uncertain_cleanup_attempt = next(
                (
                    attempt
                    for attempt in manifest["request_attempts"]
                    if attempt.get("recording_relative_path")
                    == recording["recording_relative_path"]
                    and attempt.get("stage") == "cleanup"
                    and attempt.get("run_id") == run_id
                    and attempt.get("status") in {"dispatching", "unknown"}
                ),
                None,
            )
        if uncertain_cleanup_attempt:
            stages["cleanup"]["status"] = "unknown"
            stages["cleanup"]["reason"] = "RequestOutcomeNeedsReconciliation"
            recording["status"] = "blocked"
            manifest["execution_status"] = "blocked_unknown_outcome"
            manifest["blocking_error"] = "CleanupRequestOutcomeNeedsReconciliation"
            manifest["updated_at_utc"] = _now()
            _atomic_json(manifest_path, manifest)
            raise ProductionBatchError(
                f"CleanupRequestOutcomeNeedsReconciliation:"
                f"{recording['recording_relative_path']}"
            )
        cleaned = TranscriptDocument(
            cleaned.segments,
            cleaned.diarization_status,
            cleaned.cleanup_status,
            cleaned.review_reason,
            cleaned.input_tokens,
            cleaned.output_tokens,
            raw_transcript_source=raw_text,
            normalized_transcript_source=normalized_text,
            normalization_records=normalization_records,
            normalization_review_findings=normalization_findings,
        )
        source = _source_is_unchanged(recording)
        phase9_payload = {
            "source_filename": source.name,
            "source_fingerprint": recording["source_fingerprint"],
            "source_sha256": recording["source_sha256"],
            "cleanup_config_fingerprint": config_hashes["cleanup"],
            "cleanup_status": cleaned.cleanup_status,
            "review_reason": cleaned.review_reason,
            "raw_transcript": cleaned.raw_transcript,
            "normalized_transcript": cleaned.normalized_transcript,
            "cleaned_transcript": cleaned.cleaned_transcript,
            "normalization_records": [
                asdict(item) for item in cleaned.normalization_records
            ],
            "normalization_review_findings": list(
                cleaned.normalization_review_findings
            ),
            "diarization_status": cleaned.diarization_status,
            "input_tokens": cleaned.input_tokens,
            "output_tokens": cleaned.output_tokens,
            "segments": [asdict(item) for item in cleaned.segments],
        }
        artifact = _write_stage_artifact(
            _stage_path(manifest, recording, "phase9"),
            phase9_payload,
        )
        status = (
            "completed_validated"
            if cleaned.cleanup_status == "completed"
            else "review_required"
        )
        _set_stage_complete(
            manifest_path, manifest, recording, "cleanup", run_id,
            config_hashes["cleanup"], status, artifact, cleaned.review_reason,
        )
        recording["phase9_artifact"] = artifact
        if cleaned.cleanup_status != "completed":
            stages["analysis"]["status"] = "blocked_until_cleanup_review"

    phase9_artifact = recording["phase9_artifact"]
    if not phase9_artifact:
        raise ProductionBatchError("ValidatedTranscriptStageRequired")
    phase9_path = Path(phase9_artifact["path"])
    phase9_hash = _file_hash(phase9_path)
    try:
        transcript = load_validated_transcript(phase9_path, source)
    except ReviewWorkflowError:
        recording["stages"]["analysis"]["status"] = "blocked_until_cleanup_review"
        recording["output_verification"] = {
            "status": "not_published_cleanup_review_required"
        }
        recording["status"] = "review_required"
        manifest["updated_at_utc"] = _now()
        _atomic_json(manifest_path, manifest)
        return
    document_payload = transcript.transcript_payload
    document = _transcript_document(document_payload)

    analysis_cache = recording.get("analysis_artifact")
    analysis = None
    if analysis_cache:
        analysis_path = Path(analysis_cache["path"])
        if not analysis_path.is_file() or _file_hash(analysis_path) != analysis_cache["sha256"]:
            raise ProductionBatchError("CachedAnalysisArtifactHashMismatch")
        cached_payload = json.loads(analysis_path.read_text(encoding="utf-8"))
        analysis = _analysis_from_payload(cached_payload, document)
        if analysis.status not in {"validated", "review_required"}:
            raise ProductionBatchError("CachedAnalysisArtifactInvalid")
        stages["analysis"]["validation"] = analysis.status
        stages["analysis"]["review_reason"] = analysis.review_reason

    if analysis is None and stages["analysis"]["status"] not in {
        "completed_validated", "review_required"
    }:
        stages["analysis"]["status"] = "running"
        stages["analysis"]["origin"] = "new"
        manifest["updated_at_utc"] = _now()
        _atomic_json(manifest_path, manifest)
        tracker = _run_stage_request(
            manifest_path, manifest, recording, "analysis", run_id, session
        )
        try:
            analysis = GPTMeetingAnalyzer(
                config,
                api_version=api_version,
                session=tracker,
                terminology=os.getenv("MORNINGTALK_TERMINOLOGY", ""),
            ).analyze(transcript.segments)
        except Exception as error:
            stages["analysis"]["status"] = (
                "unknown"
                if any(
                    attempt.get("status") in {"dispatching", "unknown", "response_received"}
                    for attempt in manifest["request_attempts"]
                    if attempt.get("recording_relative_path") == recording["recording_relative_path"]
                    and attempt.get("stage") == "analysis"
                    and attempt.get("run_id") == run_id
                )
                else "failed"
            )
            stages["analysis"]["reason"] = type(error).__name__
            recording["status"] = "blocked"
            manifest["updated_at_utc"] = _now()
            _atomic_json(manifest_path, manifest)
            raise ProductionBatchError(
                f"AnalysisStageFailed:{recording['recording_relative_path']}:{type(error).__name__}"
            ) from error
        source = _source_is_unchanged(recording)
        analysis_payload = {
            "source_filename": source.name,
            "source_fingerprint": recording["source_fingerprint"],
            "source_sha256": recording["source_sha256"],
            "phase9_transcript_json": str(phase9_path.resolve()),
            "phase9_json_sha256": phase9_hash,
            "analysis_config_fingerprint": config_hashes["analysis"],
            "meeting_analysis_requests": 1,
            "meeting_analysis": analysis.to_dict(),
        }
        artifact = _write_stage_artifact(
            _stage_path(manifest, recording, "phase10"),
            analysis_payload,
        )
        analysis_status = (
            "completed_validated"
            if analysis.status == "validated"
            else "review_required"
        )
        _set_stage_complete(
            manifest_path, manifest, recording, "analysis", run_id,
            config_hashes["analysis"], analysis_status, artifact,
            analysis.review_reason,
        )
        recording["analysis_artifact"] = artifact
        stages["analysis"]["validation"] = analysis.status
        stages["analysis"]["review_reason"] = analysis.review_reason

    if analysis is None:
        analysis = MeetingAnalysis.for_review("MeetingAnalysisUnavailable")
    recording["output_verification"] = _publish_outputs(
        recording,
        source,
        document,
        analysis,
        overwrite_draft=stages["analysis"].get("origin") != "reused",
        review_version_suffix=(
            manifest["batch_id"]
            if source.stem == "meeting-20261107-0817"
            and analysis.status != "validated"
            and Path(recording["outputs"]["summary_draft_pdf"]).exists()
            else None
        ),
    )
    recording["status"] = (
        "completed"
        if analysis.status == "validated"
        else "review_required"
    )
    manifest["updated_at_utc"] = _now()
    _atomic_json(manifest_path, manifest)


def execute_production_batch(
    manifest_path: Path,
    *,
    config=None,
    api_version: str | None = None,
    session_factory=requests.Session,
) -> dict[str, object]:
    manifest_path = manifest_path.resolve()
    config = config or get_azure_config()
    api_version = api_version or get_azure_openai_api_version()
    configuration = _configuration(config, api_version)
    with _manifest_lock(manifest_path):
        manifest = _load_manifest(manifest_path)
        expected_hashes = _stage_config_fingerprints(
            config,
            api_version,
            os.getenv("MORNINGTALK_TERMINOLOGY", ""),
        )
        _upgrade_verified_prompt_configuration(
            manifest_path,
            manifest,
            configuration,
            expected_hashes,
        )
        blocked_recording_paths = {
            recording["recording_relative_path"]
            for recording in manifest["recordings"]
            if recording.get("status") == "blocked"
        }
        pending_attempts = [
            item for item in manifest["request_attempts"]
            if item.get("status") in {"dispatching", "response_received", "unknown"}
            and item.get("recording_relative_path") not in blocked_recording_paths
        ]
        if pending_attempts:
            raise ProductionBatchError("RequestOutcomeNeedsReconciliation")
        if expected_hashes != manifest.get("configuration_fingerprints"):
            raise ProductionBatchError("StageConfigurationChangedSincePreflight")
        remaining_budget = (
            manifest["request_budget"]["planned_total"]
            - len(manifest["request_attempts"])
        )
        if remaining_budget < 0:
            raise ProductionBatchError("RequestBudgetAlreadyExceeded")
        run_id = str(uuid.uuid4())
        manifest["active_run_id"] = run_id
        manifest["request_budget"]["active_run_limit"] = remaining_budget
        manifest["execution_status"] = (
            "running_with_isolated_blockers"
            if blocked_recording_paths else "running"
        )
        manifest["updated_at_utc"] = _now()
        _atomic_json(manifest_path, manifest)
        session = session_factory()
        if hasattr(session, "mount"):
            session.mount("https://", HTTPAdapter(max_retries=0))
            session.mount("http://", HTTPAdapter(max_retries=0))
        try:
            ordered_recordings = sorted(
                manifest["recordings"],
                key=lambda recording: (
                    0
                    if Path(recording["source"]).stem
                    == "meeting-20261107-0817"
                    else 1,
                    recording["recording_relative_path"].casefold(),
                ),
            )
            for recording in ordered_recordings:
                if recording["recording_relative_path"] in blocked_recording_paths:
                    continue
                if recording.get("status") in {"completed", "review_required"}:
                    continue
                try:
                    _process_recording(
                        manifest_path,
                        manifest,
                        recording,
                        config,
                        api_version,
                        session,
                        run_id,
                    )
                except ProductionBatchError as error:
                    recording["status"] = "blocked"
                    recording["blocking_error"] = error.reason
                    manifest["execution_status"] = "running_with_isolated_blockers"
                    manifest.setdefault("blocking_errors", []).append(
                        {
                            "recording_relative_path": recording[
                                "recording_relative_path"
                            ],
                            "reason": error.reason,
                        }
                    )
                    manifest.setdefault("events", []).append(
                        {
                            "timestamp_utc": _now(),
                            "event": "recording_blocked",
                            "recording_relative_path": recording[
                                "recording_relative_path"
                            ],
                            "reason": error.reason,
                        }
                    )
                    manifest["updated_at_utc"] = _now()
                    _atomic_json(manifest_path, manifest)
                    _write_final_report(manifest_path, manifest)
                    continue
            if any(
                item.get("status") == "blocked"
                for item in manifest["recordings"]
            ):
                manifest["execution_status"] = "completed_with_blockers"
            elif any(
                item.get("status") == "review_required"
                for item in manifest["recordings"]
            ):
                manifest["execution_status"] = "completed_with_review"
            else:
                manifest["execution_status"] = "completed"
            manifest["completed_at_utc"] = _now()
            manifest["updated_at_utc"] = _now()
            _atomic_json(manifest_path, manifest)
            _write_final_report(manifest_path, manifest)
            return manifest
        finally:
            close = getattr(session, "close", None)
            if callable(close):
                close()


def reconcile_interrupted_batch(manifest_path: Path) -> dict[str, object]:
    """Persist interrupted dispatch state and stop resumptions from duplicating requests."""
    manifest_path = manifest_path.resolve()
    with _manifest_lock(manifest_path):
        manifest = _load_manifest(manifest_path)
        unknown = [
            attempt
            for attempt in manifest["request_attempts"]
            if attempt.get("status") == "unknown"
        ]
        received = []
        for attempt in manifest["request_attempts"]:
                if attempt.get("status") == "dispatching":
                    attempt["status"] = "unknown"
                    attempt["reconciled_at_utc"] = _now()
                    attempt["reconciliation"] = "InterruptedDuringDispatch"
                    unknown.append(attempt)
                elif attempt.get("status") == "response_received":
                    raw_path = Path(attempt.get("raw_response_path", ""))
                    if (
                        not raw_path.is_file()
                        or _sha256_file(raw_path) != attempt.get("raw_response_sha256")
                    ):
                        attempt["status"] = "unknown"
                        attempt["reconciled_at_utc"] = _now()
                        attempt["reconciliation"] = "RawResponseMissingOrChanged"
                        unknown.append(attempt)
                    else:
                        received.append(attempt)
        affected = {
                attempt.get("recording_relative_path")
                for attempt in (*unknown, *received)
        }
        stage_outcomes: dict[tuple[str, str], bool] = {}
        for attempt in unknown:
                stage_outcomes[
                    (attempt["recording_relative_path"], attempt["stage"])
                ] = True
        for attempt in received:
                stage_outcomes.setdefault(
                    (attempt["recording_relative_path"], attempt["stage"]),
                    False,
                )
        for recording in manifest["recordings"]:
                if recording["recording_relative_path"] not in affected:
                    continue
                recording["status"] = "blocked"
                for (relative_path, stage), has_unknown in stage_outcomes.items():
                    if relative_path == recording["recording_relative_path"]:
                        recording["stages"][stage]["status"] = (
                            "unknown" if has_unknown else "interrupted"
                        )
        manifest["execution_status"] = (
                "blocked_unknown_outcome"
                if unknown
                else "blocked_interrupted_stage"
        )
        manifest["blocking_error"] = (
                "RequestOutcomeNeedsReconciliation"
                if unknown
                else "ResponseReceivedButStageNotCompleted"
        )
        manifest["updated_at_utc"] = _now()
        manifest.setdefault("events", []).append(
                {
                    "timestamp_utc": _now(),
                    "event": "interrupted_batch_reconciled",
                    "unknown_request_count": len(unknown),
                    "response_received_unprocessed_count": len(received),
                }
        )
        _atomic_json(manifest_path, manifest)
        _write_final_report(manifest_path, manifest)
        return manifest


def recover_saved_transcription_responses(
    manifest_path: Path,
    recording_relative_path: str,
) -> dict[str, object]:
    """Validate and preserve successfully received chunks without retrying unknown chunks."""
    manifest_path = manifest_path.resolve()
    with _manifest_lock(manifest_path):
        manifest = _load_manifest(manifest_path)
        recording = next(
            (
                item for item in manifest["recordings"]
                if item["recording_relative_path"] == recording_relative_path
            ),
            None,
        )
        if recording is None:
            raise ProductionBatchError("RecordingNotInManifest")
        known_chunks = []
        seen_indexes = set()
        for attempt in manifest["request_attempts"]:
            if (
                attempt.get("recording_relative_path") != recording_relative_path
                or attempt.get("stage") != "transcription"
                or attempt.get("status") != "response_received"
            ):
                continue
            index = attempt.get("chunk_index")
            raw_path = Path(attempt.get("raw_response_path", ""))
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 1
                or index > recording["expected_audio_chunks"]
                or index in seen_indexes
                or not raw_path.is_file()
                or _sha256_file(raw_path) != attempt.get("raw_response_sha256")
                or raw_path.stat().st_size != attempt.get("raw_response_bytes")
                or attempt.get("http_status") != 200
            ):
                raise ProductionBatchError("CapturedTranscriptionResponseInvalid")
            try:
                body = json.loads(raw_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, ValueError) as error:
                raise ProductionBatchError(
                    "CapturedTranscriptionResponseInvalid"
                ) from error
            text = body.get("text") if isinstance(body, dict) else None
            if not isinstance(text, str) or not text.strip():
                raise ProductionBatchError("CapturedTranscriptionTextMissing")
            seen_indexes.add(index)
            known_chunks.append(
                {
                    "chunk_index": index,
                    "attempt_id": attempt["attempt_id"],
                    "raw_response_sha256": attempt["raw_response_sha256"],
                    "text": text,
                }
            )
        if not known_chunks:
            raise ProductionBatchError("NoRecoverableTranscriptionResponses")
        known_chunks.sort(key=lambda item: item["chunk_index"])
        unknown_indexes = sorted(
            attempt["chunk_index"]
            for attempt in manifest["request_attempts"]
            if attempt.get("recording_relative_path") == recording_relative_path
            and attempt.get("stage") == "transcription"
            and attempt.get("status") == "unknown"
            and isinstance(attempt.get("chunk_index"), int)
        )
        missing_indexes = sorted(
            set(range(1, recording["expected_audio_chunks"] + 1))
            - seen_indexes
        )
        payload = {
            "schema_version": 1,
            "recording_relative_path": recording_relative_path,
            "source_fingerprint": recording["source_fingerprint"],
            "source_sha256": recording["source_sha256"],
            "recovery_status": "partial_unknown_outcome",
            "expected_chunks": recording["expected_audio_chunks"],
            "received_chunk_indexes": sorted(seen_indexes),
            "unknown_chunk_indexes": unknown_indexes,
            "not_received_chunk_indexes": [
                index for index in missing_indexes if index not in unknown_indexes
            ],
            "chunks": known_chunks,
        }
        source = Path(recording["source"])
        output_path = (
            Path(manifest["paths"]["batch_root"])
            / "recoveries"
            / source.parent.parent.name
            / source.parent.name
            / f"{source.stem}.partial_transcription.json"
        )
        recovery_hash = _json_hash(payload)
        if output_path.exists():
            try:
                existing = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, ValueError) as error:
                raise ProductionBatchError("RecoveryArtifactConflict") from error
            if _json_hash(existing) != recovery_hash:
                raise ProductionBatchError("RecoveryArtifactConflict")
        else:
            _atomic_json(output_path, payload)
        artifact = {
            "path": str(output_path.resolve()),
            "sha256": _sha256_file(output_path),
            "received_chunks": len(known_chunks),
            "unknown_chunk_indexes": unknown_indexes,
            "not_received_chunk_indexes": payload["not_received_chunk_indexes"],
        }
        recording["partial_recovery_artifact"] = artifact
        manifest.setdefault("events", []).append(
            {
                "timestamp_utc": _now(),
                "event": "saved_transcription_chunks_recovered",
                "recording_relative_path": recording_relative_path,
                "received_chunk_count": len(known_chunks),
                "unknown_chunk_indexes": unknown_indexes,
            }
        )
        manifest["updated_at_utc"] = _now()
        _atomic_json(manifest_path, manifest)
        _write_final_report(manifest_path, manifest)
        return artifact


def _write_final_report(manifest_path: Path, manifest: dict[str, object]) -> Path:
    paths = manifest["paths"]
    recording_root = Path(paths["recording_root"])
    transcript_root = Path(paths["transcript_root"])
    sara_root = Path(paths["sara_root"])
    preview_root = Path(paths["preview_root"])
    inventory = discover_inventory(
        recording_root,
        transcript_root,
        sara_root,
        private_root=preview_root.parent,
    )
    report_root = Path(paths["batch_root"]) / "benchmark"
    (
        _inventory_csv,
        benchmark_summary,
        discrepancy_csv,
        reference_extracts,
        _recommendations,
    ) = generate_benchmark_reports(inventory, report_root)
    discrepancy_rows = list(csv.DictReader(discrepancy_csv.open(encoding="utf-8-sig")))
    actions = {
        "Suspected transcription error": (
            "Unresolved ambiguity",
            "Human-review the audio and raw text; do not apply a guessed correction.",
        ),
        "Cleanup error": (
            "Cleanup error",
            "Compare raw and cleaned text against the audio before correcting the cleaned transcript.",
        ),
        "Summary omission": (
            "Unresolved ambiguity",
            "Check whether a supported paraphrase already covers the point; add it only with transcript evidence.",
        ),
        "Insufficient evidence": (
            "Unresolved ambiguity",
            "Review the cited transcript span and audio; do not present the claim as confirmed.",
        ),
    }
    discrepancy_path = Path(paths["batch_root"]) / "sara_comparison_discrepancies.csv"
    with discrepancy_path.open("w", encoding="utf-8-sig", newline="") as output:
        fields = [
            "plant", "meeting", "source_type", "source_path", "page",
            "category", "source_wording", "transcript_path", "summary_path",
            "classification", "evidence_status", "root_cause", "corrective_action",
        ]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in discrepancy_rows:
            classification, action = actions.get(
                row["classification"],
                ("Unresolved ambiguity", "Escalate for human review with source evidence."),
            )
            writer.writerow({**row, "classification": classification, "corrective_action": action})

    inventory_rows = []
    for item in inventory.meetings:
        record = next(
            value for value in manifest["recordings"]
            if Path(value["source"]).resolve() == item.mp3_path.resolve()
        )
        pages = extract_pdf_pages(item.sara_pdf_path) if item.sara_pdf_path else ()
        facts = extract_reference_facts(pages)
        categories: dict[str, int] = {}
        for fact in facts:
            categories[fact.category] = categories.get(fact.category, 0) + 1
        phase9_status = record["stages"]["cleanup"]["status"]
        analysis_status = (
            record["stages"]["analysis"].get("validation")
            or record["stages"]["analysis"]["status"]
        )
        inventory_rows.append(
            {
                "recording": item.recording_filename,
                "plant": item.plant,
                "month": item.month,
                "sara_reference": str(item.sara_pdf_path) if item.sara_pdf_path else "",
                "sara_pages": len(pages),
                "sara_fact_categories": json.dumps(categories, ensure_ascii=False),
                "duration_seconds": item.duration_seconds,
                "audio_chunks": record["expected_audio_chunks"],
                "phase9_status": phase9_status,
                "analysis_status": analysis_status,
                "summary_status": record.get("status"),
                "existing_transcript_txt": (
                    str(item.transcript_txt_path)
                    if item.transcript_txt_path else ""
                ),
                "existing_full_transcript_pdf": (
                    str(item.full_transcript_pdf_path)
                    if item.full_transcript_pdf_path else ""
                ),
                "existing_summary_draft_pdf": (
                    str(item.summary_draft_pdf_path)
                    if item.summary_draft_pdf_path else ""
                ),
                "existing_approved_summary_pdf": (
                    str(item.approved_summary_pdf_path)
                    if item.approved_summary_pdf_path else ""
                ),
                "comparison_result": (
                    "reference_extracted_human_review_required"
                    if item.sara_pdf_path
                    and analysis_status == "review_required"
                    else "reference_extracted_transcript_available"
                    if item.sara_pdf_path
                    and phase9_status in {"reused", "completed_validated"}
                    else "reference_extracted_transcript_unavailable"
                    if item.sara_pdf_path
                    else "no_reference"
                ),
                "corrective_action": (
                    (
                        "Review page-numbered SARA coverage against transcript evidence; "
                        "the existing analysis remains unapproved."
                    )
                    if item.sara_pdf_path and analysis_status == "review_required"
                    else (
                        "Use page-numbered SARA content only as a coverage reference; "
                        "semantic comparison is incomplete."
                    )
                    if item.sara_pdf_path
                    else "No SARA reference; assess transcript and summary solely against audio evidence."
                ),
                "summary_draft_path": record["outputs"]["summary_draft_pdf"],
                "approved_summary_exists": Path(
                    record["outputs"]["approved_summary_pdf"]
                ).exists(),
            }
        )
    comparison_path = Path(paths["batch_root"]) / "sara_comparison.csv"
    with comparison_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(inventory_rows[0]))
        writer.writeheader()
        writer.writerows(inventory_rows)

    pilot = next(
        (
            item for item in manifest["recordings"]
            if Path(item["source"]).stem == "meeting-20261107-0817"
        ),
        None,
    )
    pilot_comparison_path = None
    if pilot is not None:
        pilot_pages = (
            extract_pdf_pages(Path(pilot["sara_pdf"]))
            if pilot.get("sara_pdf") else ()
        )
        pilot_facts = extract_reference_facts(pilot_pages)
        pilot_counts: dict[str, int] = {}
        for fact in pilot_facts:
            pilot_counts[fact.category] = pilot_counts.get(fact.category, 0) + 1
        analysis_payload = {}
        if pilot.get("analysis_artifact"):
            analysis_path = Path(pilot["analysis_artifact"]["path"])
            if analysis_path.is_file():
                saved = json.loads(analysis_path.read_text(encoding="utf-8"))
                analysis_payload = saved.get("meeting_analysis", {})
        section_counts = {}
        if isinstance(analysis_payload, dict):
            for section in (
                "overview", "classifications", "topics", "key_issues",
                "actions", "dependencies", "schedules", "risks",
                "closing_activities",
            ):
                value = analysis_payload.get(section)
                section_counts[section] = (
                    len(value) if isinstance(value, list)
                    else int(isinstance(value, dict))
                )
        pilot_discrepancies = [
            row for row in discrepancy_rows
            if row.get("meeting") == Path(pilot["source"]).name
        ]
        pilot_comparison_path = (
            Path(paths["batch_root"]) / "pilot_sara_comparison.md"
        )
        pilot_lines = [
            "# Nov 7 pilot — offline SARA coverage comparison",
            "",
            f"- Recording: `{pilot['source']}`",
            f"- SARA reference: `{pilot.get('sara_pdf') or 'not available'}`",
            f"- MorningTalk transcript artifact: "
            f"`{(pilot.get('phase9_artifact') or {}).get('path', 'not available')}`",
            f"- Analysis state: `{pilot['stages']['analysis'].get('validation') or pilot['stages']['analysis']['status']}` "
            f"({pilot['stages']['analysis'].get('review_reason') or 'no review reason recorded'})",
            f"- Existing Summary Draft remains unapproved; this processing run does not approve it.",
            "",
            "SARA is used only as a presentation and coverage reference. "
            "The validated transcript and original audio are the factual sources. "
            "Counts below are structural inventories, not semantic match scores.",
            "",
            "## SARA reference facts by category",
            "",
            "| Category | Page-numbered lines |",
            "|---|---:|",
        ]
        for category, count in sorted(pilot_counts.items()):
            pilot_lines.append(f"| {category} | {count} |")
        pilot_lines.extend(
            [
                "",
                "## Existing MorningTalk analysis structure",
                "",
                "| Section | Items |",
                "|---|---:|",
            ]
        )
        for section, count in section_counts.items():
            pilot_lines.append(f"| {section} | {count} |")
        pilot_lines.extend(
            [
                "",
                "The cached analysis passes the structural evidence resolver but remains "
                "`review_required` / `SemanticReviewRequired`. Its claims are not treated "
                "as confirmed or as improved. The new review-version PDF labels every "
                "claim for review and retains segment references where present.",
                "",
                "## Discrepancy candidates",
                "",
                f"Candidates recorded: {len(pilot_discrepancies)}. Each requires human "
                "comparison; none is a verified correction.",
                "",
            ]
        )
        for row in pilot_discrepancies:
            pilot_lines.extend(
                [
                    f"- **{row.get('classification', 'Unclassified')}** — "
                    f"{row.get('category', '')}; page {row.get('page') or 'n/a'}; "
                    f"source: `{row.get('source_type', '')}`.",
                    f"  - Candidate wording: {row.get('source_wording', '')}",
                    f"  - Evidence status: {row.get('evidence_status', '')}",
                    f"  - Root-cause assessment: {row.get('root_cause', '')}",
                    "  - Corrective action: listen to the original MP3 and check the "
                    "cited transcript span before changing any wording or claim.",
                ]
            )
        pilot_lines.extend(
            [
                "",
                "## Listening required",
                "",
                "Suspected ASR phrases cannot be confirmed from SARA wording. "
                "Potential omissions, quantities, owners, deadlines, and action items "
                "must be checked against the source audio. No SARA-only statement was "
                "copied into the MorningTalk analysis.",
                "",
            ]
        )
        pilot_comparison_path.write_text(
            "\n".join(pilot_lines),
            encoding="utf-8",
        )

    attempts = manifest["request_attempts"]
    token_inputs = sum(item.get("input_tokens", 0) for item in attempts)
    token_outputs = sum(item.get("output_tokens", 0) for item in attempts)
    latest_run_id = manifest.get("active_run_id")
    latest_attempts = [
        item for item in attempts if item.get("run_id") == latest_run_id
    ]
    latest_run_recordings = {
        item.get("recording_relative_path") for item in latest_attempts
    }
    latest_stage_counts = {
        stage: sum(item.get("stage") == stage for item in latest_attempts)
        for stage in _STAGES
    }
    latest_token_inputs = sum(
        item.get("input_tokens", 0) for item in latest_attempts
    )
    latest_token_outputs = sum(
        item.get("output_tokens", 0) for item in latest_attempts
    )
    completed = sum(item.get("status") in {"completed", "review_required"} for item in manifest["recordings"])
    failed = sum(item.get("status") == "blocked" for item in manifest["recordings"])
    matched = sum(item.get("sara_match_status") == "matched" for item in manifest["recordings"])
    lines = [
        "# MorningTalk production batch report",
        "",
        f"- Batch ID: `{manifest['batch_id']}`",
        f"- Execution status: **{manifest['execution_status']}**",
        f"- Recordings discovered: {len(manifest['recordings'])}",
        f"- Completed or review-required: {completed}",
        f"- Failed/blocked: {failed}",
        f"- Exact SARA matches: {matched}; without SARA: "
        f"{len(manifest['recordings']) - matched}",
        f"- Total request attempt records in batch: {len(attempts)} "
        f"(transcription {sum(a['stage']=='transcription' for a in attempts)}, "
        f"cleanup {sum(a['stage']=='cleanup' for a in attempts)}, "
        f"analysis {sum(a['stage']=='analysis' for a in attempts)})",
        f"- Latest execution request attempts: {len(latest_attempts)} "
        f"(transcription {latest_stage_counts['transcription']}, "
        f"cleanup {latest_stage_counts['cleanup']}, "
        f"analysis {latest_stage_counts['analysis']}).",
        f"- Recorded token usage across the batch: {token_inputs} input / "
        f"{token_outputs} output; latest execution: {latest_token_inputs} input / "
        f"{latest_token_outputs} output. Audio transcription tokens may not be "
        "returned by the service.",
        f"- Request outcomes: {sum(item.get('status') == 'completed' for item in attempts)} completed, "
        f"{sum(item.get('status') == 'response_received' for item in attempts)} response received but not processed, "
        f"{sum(item.get('status') == 'unknown' for item in attempts)} unknown.",
        f"- Blocking error: {manifest.get('blocking_error', 'none')}.",
        "- Token totals include recorded usage only; unknown requests and responses "
        "without service usage data are not estimated.",
        "- Automatic retries: none; unresolved outcomes require operator reconciliation.",
        "- Approved summaries: not generated or modified.",
        "",
        "## Unresolved request details",
        "",
    ]
    unknown_attempts = [
        attempt for attempt in attempts if attempt.get("status") == "unknown"
    ]
    if unknown_attempts:
        for attempt in unknown_attempts:
            http_status = attempt.get("http_status")
            http_status_text = (
                str(http_status) if http_status is not None else "not recorded"
            )
            response_path_value = attempt.get("raw_response_path")
            if isinstance(response_path_value, str) and response_path_value:
                response_path = Path(response_path_value)
                response_text = (
                    f"retained at `{response_path}`"
                    if response_path.is_file()
                    else f"missing at `{response_path}`"
                )
            else:
                response_text = "not recorded"
            lines.append(
                f"- `{attempt['attempt_id']}` — "
                f"{attempt['recording_relative_path']}, "
                f"{attempt['stage']} chunk {attempt.get('chunk_index', 'n/a')}, "
                f"started `{attempt.get('started_at_utc')}`; "
                f"HTTP status: {http_status_text}; raw response: "
                f"{response_text}. Do not retry until reconciled."
            )
    else:
        lines.append("- None")
    lines.extend(["", "## Outputs and per-record status", ""])
    for record in manifest["recordings"]:
        outputs = record.get("outputs", {})
        output_verification = record.get("output_verification") or {}
        analysis_blocked_on_cleanup = (
            record["stages"]["analysis"]["status"]
            == "blocked_until_cleanup_review"
        )
        review_notice_published_this_run = (
            analysis_blocked_on_cleanup
            and record.get("recording_relative_path") in latest_run_recordings
            and isinstance(output_verification.get("summary_draft_pdf"), dict)
        )
        draft_status = (
            "review-required notice only; meeting analysis was not generated"
            if review_notice_published_this_run
            else "review draft"
        )
        lines.extend(
            [
                f"### {record['recording_relative_path']}",
                "",
                f"- Status: `{record.get('status')}`",
                f"- Transcription / cleanup / analysis: "
                f"{record['stages']['transcription']['status']} / "
                f"{record['stages']['cleanup']['status']} / "
                f"{record['stages']['analysis']['status']}",
                f"- SARA: `{record.get('sara_pdf') or 'no exact-stem reference'}`",
                f"- TXT: `{outputs.get('transcript_txt')}`",
                f"- Full PDF: `{outputs.get('full_transcript_pdf')}`",
                f"- Summary Draft ({draft_status}): "
                f"`{outputs.get('summary_draft_pdf')}`",
                f"- Pre-existing TXT/Full/Draft: "
                f"{Path(outputs['transcript_txt']).exists()} / "
                f"{Path(outputs['full_transcript_pdf']).exists()} / "
                f"{Path(outputs['summary_draft_pdf']).exists()}",
                f"- Approved Summary: "
                f"{'pre-existing and preserved' if Path(outputs['approved_summary_pdf']).exists() else 'not generated'}",
                f"- Review-version Summary Draft: "
                f"{((record.get('output_verification') or {}).get('summary_draft_review_version_pdf') or {}).get('path') or 'not generated'}",
                f"- Partial transcription recovery: "
                f"{(record.get('partial_recovery_artifact') or {}).get('path') or 'none'}",
                "",
            ]
        )
    lines.extend(
        [
            "## Comparison method and limits",
            "",
            "SARA PDF page text and topic-category counts are coverage references only. "
            "The transcript and audio remain the factual source. Lexical absence or "
            "paraphrase alone is not classified as an omission or unsupported statement; "
            "ambiguous candidates are routed to human review with corrective actions.",
            "",
            f"- Per-meeting comparison: `{comparison_path}`",
            f"- Discrepancy candidates: `{discrepancy_path}`",
            f"- Page-numbered SARA extracts: `{reference_extracts}`",
            f"- Benchmark summary: `{benchmark_summary}`",
            f"- Nov 7 pilot comparison: `{pilot_comparison_path or 'not generated'}`",
            f"- Targeted test result and full pytest exit code are recorded separately by the operator.",
            "",
        ]
    )
    report_path = Path(paths["report_path"])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
