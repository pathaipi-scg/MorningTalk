"""Offline preflight and explicitly authorized pilot-batch execution."""

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import logging
import msvcrt
import os
from pathlib import Path, PureWindowsPath
import tempfile
from typing import Any
import uuid

import requests
from requests.adapters import HTTPAdapter

from app.azure_openai_http import resolve_azure_openai_endpoint
from app.config import (
    AzureConfig,
    get_azure_config,
    get_azure_openai_api_version,
)
from app.file_manager import source_fingerprint
from app.meeting_analysis import GPTMeetingAnalyzer
from app.phase10_6 import (
    DEFAULT_PREVIEW_ROOT,
    DEFAULT_REVIEW_ROOT,
    _artifact_indexes,
    _load_transcript_candidates,
    _approved_state,
    load_validated_transcript,
)
from app.transcript import TranscriptSegment
from app.transcript_cleanup import TranscriptCleanup
from app.transcriber import AzureOpenAITranscriber, estimate_chunk_count

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "references" / "SaraAI" / "analysis"
DEFAULT_PAIRS_CSV = DEFAULT_ANALYSIS_DIR / "recording_pdf_pairs.csv"
DEFAULT_RECORDING_ROOT = Path(r"D:\KM\Vault\MorningTalk\Record File\CB")
DEFAULT_SARA_ROOT = PROJECT_ROOT / "references" / "SaraAI"
DEFAULT_MANIFEST_PATH = DEFAULT_ANALYSIS_DIR / "phase10_7_pilot_manifest.json"
DEFAULT_REPORT_PATH = DEFAULT_ANALYSIS_DIR / "phase10_7_pilot_dry_run.md"
DEFAULT_STAGE_ROOT = DEFAULT_PREVIEW_ROOT / "phase10_7_pilot_batch"
_MONTHS = ("Sep26", "Oct26", "Nov26")
_STAGES = ("transcription", "cleanup", "analysis")


class PilotBatchError(ValueError):
    """Safe-to-report pilot preflight or execution blocker."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


@contextmanager
def _manifest_lock(manifest_path: Path):
    lock_name = hashlib.sha256(
        str(manifest_path.resolve()).casefold().encode("utf-8")
    ).hexdigest()
    lock_path = Path(tempfile.gettempdir()) / f"morningtalk-pilot-{lock_name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise PilotBatchError("PilotBatchAlreadyRunning") from error
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    finally:
        handle.close()


def _safe_join(root: Path, relative: str) -> Path:
    parts = PureWindowsPath(relative).parts
    if not parts or PureWindowsPath(relative).is_absolute() or any(
        part in {"", ".", ".."} for part in parts
    ):
        raise PilotBatchError("InventoryPathInvalid")
    candidate = root.joinpath(*parts).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise PilotBatchError("InventoryPathOutsideRoot") from error
    return candidate


def _read_inventory(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as source:
            return list(csv.DictReader(source))
    except (OSError, csv.Error) as error:
        raise PilotBatchError("PairInventoryUnavailable") from error


def _check_artifact_status(
    source: Path,
    preview_root: Path,
    review_root: Path,
) -> tuple[list[Any], bool]:
    transcript_paths, _, review_paths = _artifact_indexes(preview_root)
    validated_transcripts, _ = _load_transcript_candidates(
        transcript_paths,
        source,
    )
    fingerprint = source_fingerprint(source)
    approved, _ = _approved_state(
        source,
        review_paths,
        fingerprint,
        review_root,
    )
    return validated_transcripts, approved


def _stage_config_fingerprints(
    config: AzureConfig,
    api_version: str,
    terminology: str,
) -> dict[str, str]:
    transcription = {
        "endpoint": config.transcription_endpoint,
        "deployment": config.transcription_deployment,
        "api_version": config.transcription_api_version,
        "chunk_duration_seconds": config.chunk_duration_seconds,
        "chunk_overlap_seconds": config.chunk_overlap_seconds,
    }
    text = {
        "endpoint": config.text_endpoint,
        "deployment": config.text_deployment,
        "api_version": api_version,
        "terminology": terminology,
    }
    return {
        "transcription": _json_hash(transcription),
        "cleanup": _json_hash(text),
        "analysis": _json_hash(text),
    }


def _configuration_preflight(
    config: AzureConfig,
    api_version: str,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    transcription_ready = True
    try:
        config.validate_transcription()
    except (ValueError, RuntimeError) as error:
        transcription_ready = False
        blockers.append(f"transcription_configuration:{type(error).__name__}")

    text_ready = bool(config.text_endpoint and config.text_api_key and config.text_deployment)
    text_mode = "unavailable"
    if not text_ready:
        blockers.append("meeting_analysis_configuration:incomplete")
    else:
        try:
            endpoint = resolve_azure_openai_endpoint(
                config.text_endpoint,
                api_version,
                config.text_deployment,
            )
            text_mode = endpoint.api_mode
        except ValueError as error:
            text_ready = False
            blockers.append(f"meeting_analysis_endpoint:{type(error).__name__}")

    return (
        {
            "transcription_ready": transcription_ready,
            "cleanup_ready": text_ready,
            "analysis_ready": text_ready,
            "text_api_version_present": bool(api_version),
            "text_endpoint_mode": text_mode,
            "chunk_duration_seconds": config.chunk_duration_seconds,
            "chunk_overlap_seconds": config.chunk_overlap_seconds,
            "credentials_stored": False,
        },
        blockers,
    )


def _candidate_records(
    *,
    recording_root: Path,
    sara_root: Path,
    pairs_csv: Path,
    preview_root: Path,
    review_root: Path,
    config: AzureConfig,
) -> list[dict[str, Any]]:
    rows = _read_inventory(pairs_csv)
    candidate_lists: dict[str, list[dict[str, Any]]] = {
        month: [] for month in _MONTHS
    }
    seen_sources: set[str] = set()
    for row in rows:
        relative = row.get("recording_relative_path", "")
        month_parts = PureWindowsPath(relative).parts
        if row.get("status") != "MATCHED" or not month_parts or month_parts[0] not in candidate_lists:
            continue
        source = _safe_join(recording_root, relative)
        key = str(source).casefold()
        if key in seen_sources:
            raise PilotBatchError("DuplicateRecordingInInventory")
        seen_sources.add(key)
        sara_pdf = _safe_join(sara_root, row.get("pdf_relative_path", ""))
        if not source.is_file() or source.stat().st_size <= 0 or not sara_pdf.is_file():
            continue
        validated_transcripts, approved = _check_artifact_status(
            source,
            preview_root,
            review_root,
        )
        if validated_transcripts or approved:
            continue
        try:
            duration = AzureOpenAITranscriber._probe_duration(source)
        except (OSError, ValueError, RuntimeError) as error:
            raise PilotBatchError(
                f"DurationProbeFailed:{month_parts[0]}:{source.name}:{type(error).__name__}"
            ) from error
        chunks = estimate_chunk_count(
            duration,
            config.chunk_duration_seconds,
            config.chunk_overlap_seconds,
        )
        candidate_lists[month_parts[0]].append(
            {
                "month": month_parts[0],
                "source": str(source.resolve()),
                "recording_relative_path": relative,
                "sara_pdf": str(sara_pdf.resolve()),
                "sara_pdf_relative_path": row["pdf_relative_path"],
                "file_size_bytes": source.stat().st_size,
                "source_fingerprint": source_fingerprint(source),
                "duration_seconds": round(duration, 3),
                "chunk_duration_seconds": config.chunk_duration_seconds,
                "chunk_overlap_seconds": config.chunk_overlap_seconds,
                "expected_audio_chunks": chunks,
                "stages": {
                    "transcription": {"status": "pending", "artifact": None},
                    "cleanup": {"status": "blocked_until_transcription", "artifact": None},
                    "analysis": {"status": "blocked_until_cleanup", "artifact": None},
                    "human_review": {"status": "pending"},
                    "approved_pdf": {"status": "blocked_until_human_approval"},
                },
            }
        )
    missing = [month for month, candidates in candidate_lists.items() if not candidates]
    if missing:
        raise PilotBatchError("NoEligibleRecordingForMonth:" + ",".join(missing))
    selected = [
        sorted(candidate_lists[month], key=lambda item: (
            item["duration_seconds"],
            item["recording_relative_path"].casefold(),
        ))[0]
        for month in _MONTHS
    ]
    for recording in selected:
        recording["source_sha256"] = _file_hash(Path(recording["source"]))
        recording["sara_pdf_sha256"] = _file_hash(Path(recording["sara_pdf"]))
    return selected


def _expected_request_counts(recording: dict[str, Any]) -> dict[str, int]:
    return {
        "transcription": recording["expected_audio_chunks"],
        "cleanup": 1,
        "analysis": 1,
        "total": recording["expected_audio_chunks"] + 2,
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise PilotBatchError("PilotManifestInvalid") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or not isinstance(payload.get("recordings"), list)
        or len(payload["recordings"]) != 3
    ):
        raise PilotBatchError("PilotManifestInvalid")
    return payload


def _validate_saved_recordings(
    recordings: list[dict[str, Any]],
    recording_root: Path,
    sara_root: Path,
    preview_root: Path,
    review_root: Path,
) -> list[str]:
    blockers = []
    months = [recording.get("month") for recording in recordings]
    if months != list(_MONTHS) or len({recording.get("source") for recording in recordings}) != 3:
        return ["manifest_selection_invalid"]
    for recording in recordings:
        source = Path(recording.get("source", ""))
        sara_pdf = Path(recording.get("sara_pdf", ""))
        if not source.is_file() or not sara_pdf.is_file():
            blockers.append(f"source_or_reference_missing:{recording.get('month')}")
            continue
        try:
            source.resolve().relative_to(recording_root.resolve())
            sara_pdf.resolve().relative_to(sara_root.resolve())
        except ValueError:
            blockers.append(f"source_or_reference_outside_root:{recording.get('month')}")
            continue
        if (
            source_fingerprint(source) != recording.get("source_fingerprint")
            or _file_hash(source) != recording.get("source_sha256")
            or _file_hash(sara_pdf) != recording.get("sara_pdf_sha256")
        ):
            blockers.append(f"source_or_reference_changed:{recording.get('month')}")
        validated_transcripts, approved = _check_artifact_status(
            source,
            preview_root,
            review_root,
        )
        cleanup_artifact = recording.get("stages", {}).get("cleanup", {}).get("artifact")
        allowed_transcript = (
            Path(cleanup_artifact["path"]).resolve()
            if isinstance(cleanup_artifact, dict) and cleanup_artifact.get("path")
            else None
        )
        if any(transcript.path.resolve() != allowed_transcript for transcript in validated_transcripts):
            blockers.append(f"validated_transcript_now_exists:{recording.get('month')}")
        if approved:
            blockers.append(f"summary_already_approved:{recording.get('month')}")
    return blockers


def _report_text(
    manifest: dict[str, Any],
    blockers: list[str],
    manifest_path: Path,
    stage_root: Path,
) -> str:
    rows = []
    total = 0
    for order, recording in enumerate(manifest["recordings"], 1):
        requests_by_stage = _expected_request_counts(recording)
        total += requests_by_stage["total"]
        rows.extend(
            [
                f"### {order}. {recording['month']}: `{Path(recording['source']).name}`",
                "",
                f"- MP3: `{recording['source']}`",
                f"- SARA PDF: `{recording['sara_pdf']}`",
                f"- Size: {recording['file_size_bytes']:,} bytes",
                f"- Duration: {recording['duration_seconds']:.3f} seconds",
                f"- Source fingerprint: `{recording['source_fingerprint']}`",
                f"- Source SHA-256: `{recording['source_sha256']}`",
                f"- Chunks: {recording['expected_audio_chunks']} "
                f"({recording['chunk_duration_seconds']}s duration, "
                f"{recording['chunk_overlap_seconds']}s overlap)",
                f"- Requests: transcription {requests_by_stage['transcription']}, "
                f"cleanup {requests_by_stage['cleanup']}, "
                f"analysis {requests_by_stage['analysis']}, "
                f"total {requests_by_stage['total']}",
                "- Existing artifacts: no validated transcript; no approved summary",
                "",
            ]
        )
    budget = manifest["request_budget"]["per_run_limit"]
    blockers = list(dict.fromkeys(blockers))
    if budget < total and f"request_budget_below_plan:{budget}<{total}" not in blockers:
        blockers.append(f"request_budget_below_plan:{budget}<{total}")
    status = "READY FOR SEPARATELY AUTHORIZED EXECUTION" if not blockers else "BLOCKED"
    lines = [
        "# MorningTalk Phase 10.7 Pilot Dry Run",
        "",
        f"- Preflight: **{status}**",
        "- Mode: offline dry run; no Azure requests or SQL writes",
        "- Selected exactly one eligible matched recording per month, shortest by ffprobe duration",
        "- Execution order: Sep26, then Oct26, then Nov26",
        "- Existing SARA PDFs are references only and are not processed or modified",
        "",
        "## Configuration preflight",
        "",
    ]
    configuration = manifest["configuration"]
    lines.extend(
        f"- {name}: {value}"
        for name, value in configuration.items()
        if name != "credentials_stored"
    )
    lines.extend(["", "## Selected recordings", "", *rows])
    lines.extend(
        [
            "## Planned requests",
            "",
            f"- Transcription requests: {sum(item['expected_audio_chunks'] for item in manifest['recordings'])}",
            f"- Cleanup requests: {len(manifest['recordings'])}",
            f"- Meeting Analysis requests: {len(manifest['recordings'])}",
            f"- Total required: {total}",
            f"- Per-run maximum: {budget}",
            "- No monetary estimate is included.",
            "",
            "## Output and approval safeguards",
            "",
            f"- Stage-artifact root: `{stage_root}`",
            "- Production MP3/PDF paths are read-only; outputs use isolated preview paths.",
            "- Meeting Analysis V2 remains review-required until Phase 10.6 explicitly approves every item.",
            "- No approved summary PDF is generated by the pilot runner.",
            "- Request attempts are persisted before dispatch; interrupted dispatches become unknown and block resumption.",
            "",
            "## Blockers",
            "",
        ]
    )
    lines.extend(f"- {blocker}" for blocker in blockers)
    if not blockers:
        lines.append("- None.")
    lines.extend(["", f"Manifest: `{manifest_path}`", ""])
    return "\n".join(lines)


def prepare_pilot_dry_run(
    *,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    stage_root: Path = DEFAULT_STAGE_ROOT,
    recording_root: Path = DEFAULT_RECORDING_ROOT,
    sara_root: Path = DEFAULT_SARA_ROOT,
    pairs_csv: Path = DEFAULT_PAIRS_CSV,
    preview_root: Path = DEFAULT_PREVIEW_ROOT,
    review_root: Path = DEFAULT_REVIEW_ROOT,
    max_requests: int | None = None,
    config: AzureConfig | None = None,
    api_version: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    manifest_path = manifest_path.resolve()
    report_path = report_path.resolve()
    stage_root = stage_root.resolve()
    recording_root = recording_root.resolve()
    sara_root = sara_root.resolve()
    preview_root = preview_root.resolve()
    review_root = review_root.resolve()
    azure_config = config or get_azure_config()
    text_api_version = api_version or get_azure_openai_api_version()
    configuration, blockers = _configuration_preflight(
        azure_config,
        text_api_version,
    )
    try:
        relative_stage_path = stage_root.relative_to(preview_root)
        if not relative_stage_path.parts or not stage_root.parent.is_dir():
            blockers.append("stage_output_path_invalid")
    except ValueError:
        blockers.append("stage_output_outside_preview_root")
    terminology = os.getenv("MORNINGTALK_TERMINOLOGY", "")
    config_hashes = _stage_config_fingerprints(
        azure_config,
        text_api_version,
        terminology,
    )
    is_new_manifest = not manifest_path.exists()
    if not is_new_manifest:
        manifest = _load_manifest(manifest_path)
        saved_paths = manifest.get("paths", {})
        if (
            Path(saved_paths.get("stage_root", stage_root)).resolve() != stage_root
            or Path(saved_paths.get("recording_root", recording_root)).resolve() != recording_root
            or Path(saved_paths.get("sara_root", sara_root)).resolve() != sara_root
        ):
            blockers.append("pilot_paths_changed")
        blockers.extend(
            _validate_saved_recordings(
                manifest["recordings"],
                recording_root,
                sara_root,
                preview_root,
                review_root,
            )
        )
        for recording in manifest["recordings"]:
            for stage in _STAGES:
                previous = recording["stages"].get(stage, {})
                if (
                    previous.get("status") in {"completed", "completed_validated", "review_required"}
                    and previous.get("config_fingerprint") != config_hashes[stage]
                ):
                    blockers.append(
                        f"stage_configuration_changed:{recording['month']}:{stage}"
                    )
        if any(
            attempt.get("status") in {"dispatching", "response_received", "unknown"}
            for attempt in manifest.get("request_attempts", [])
        ):
            blockers.append("request_outcome_requires_operator_reconciliation")
    else:
        if stage_root.exists():
            blockers.append("stage_output_directory_exists_without_manifest")
        try:
            recordings = _candidate_records(
                recording_root=recording_root,
                sara_root=sara_root,
                pairs_csv=pairs_csv,
                preview_root=preview_root,
                review_root=review_root,
                config=azure_config,
            )
        except PilotBatchError as error:
            _atomic_text(
                report_path,
                "\n".join(
                    [
                        "# MorningTalk Phase 10.7 Pilot Dry Run",
                        "",
                        "- Preflight: **BLOCKED**",
                        "- Mode: offline; no Azure requests or SQL writes",
                        f"- Blocker: `{error.reason}`",
                        "- No alternate month or recording was substituted.",
                        "",
                    ]
                ),
            )
            raise
        for recording in recordings:
            recording["expected_requests"] = _expected_request_counts(recording)
        total = sum(
            recording["expected_requests"]["total"] for recording in recordings
        )
        manifest = {
            "schema_version": 1,
            "phase": "10.7",
            "created_at_utc": _now(),
            "updated_at_utc": _now(),
            "execution_status": "dry_run",
            "configuration": configuration,
            "configuration_fingerprints": config_hashes,
            "paths": {
                "recording_root": str(recording_root),
                "sara_root": str(sara_root),
                "preview_root": str(preview_root),
                "review_root": str(review_root),
                "stage_root": str(stage_root),
            },
            "request_budget": {
                "per_run_limit": total if max_requests is None else max_requests,
                "planned_total": total,
            },
            "recordings": recordings,
            "request_attempts": [],
            "events": [],
        }
    if max_requests is not None:
        if max_requests <= 0:
            blockers.append("request_budget_must_be_positive")
        manifest["request_budget"]["per_run_limit"] = max_requests
    else:
        manifest["request_budget"]["per_run_limit"] = max(
            1,
            manifest["request_budget"].get("planned_total", 1),
        )
    manifest["configuration"] = configuration
    manifest["configuration_fingerprints"] = config_hashes
    manifest["request_budget"]["planned_total"] = sum(
        _expected_request_counts(recording)["total"]
        for recording in manifest["recordings"]
    )
    budget = manifest["request_budget"]["per_run_limit"]
    planned = manifest["request_budget"]["planned_total"]
    if isinstance(budget, int) and budget < planned:
        blockers.append(f"request_budget_below_plan:{budget}<{planned}")
    manifest["updated_at_utc"] = _now()
    if blockers:
        manifest["execution_status"] = "blocked"
    elif is_new_manifest:
        manifest["execution_status"] = "dry_run"
    if is_new_manifest and stage_root.exists():
        manifest["execution_status"] = "blocked"
    with _manifest_lock(manifest_path):
        _atomic_json(manifest_path, manifest)
        _atomic_text(
            report_path,
            _report_text(manifest, blockers, manifest_path, stage_root),
        )
    return manifest, blockers


def _stage_directory(recording: dict[str, Any], stage_root: Path) -> Path:
    return stage_root / recording["month"] / Path(recording["source"]).stem


def _artifact_is_reusable(
    stage: dict[str, Any],
    expected_config_fingerprint: str,
) -> bool:
    if stage.get("status") not in {"completed", "completed_validated", "review_required"}:
        return False
    artifact = stage.get("artifact")
    if not isinstance(artifact, dict) or stage.get("config_fingerprint") != expected_config_fingerprint:
        return False
    path = Path(artifact.get("path", ""))
    try:
        return path.is_file() and _file_hash(path) == artifact.get("sha256")
    except OSError:
        return False


def _write_stage_artifact(
    path: Path,
    payload: dict[str, Any],
) -> dict[str, str]:
    if path.exists():
        raise PilotBatchError("StageArtifactAlreadyExists")
    _atomic_json(path, payload)
    return {"path": str(path.resolve()), "sha256": _file_hash(path)}


def _set_stage_complete(
    manifest_path: Path,
    manifest: dict[str, Any],
    recording: dict[str, Any],
    stage_name: str,
    run_id: str,
    config_fingerprint: str,
    status: str,
    artifact: dict[str, str],
    reason: str | None = None,
) -> None:
    stage = recording["stages"][stage_name]
    stage.update(
        {
            "status": status,
            "artifact": artifact,
            "config_fingerprint": config_fingerprint,
            "completed_at_utc": _now(),
            "reason": reason,
        }
    )
    for attempt in manifest["request_attempts"]:
        if (
            attempt["recording_relative_path"] == recording["recording_relative_path"]
            and attempt["stage"] == stage_name
            and attempt["run_id"] == run_id
            and attempt["status"] == "response_received"
        ):
            attempt["status"] = "completed"
            attempt["completed_at_utc"] = _now()
    manifest["updated_at_utc"] = _now()
    _atomic_json(manifest_path, manifest)


class _TrackedSession:
    def __init__(
        self,
        manifest_path: Path,
        manifest: dict[str, Any],
        recording: dict[str, Any],
        stage_name: str,
        run_id: str,
        budget: int,
        session: requests.Session,
    ) -> None:
        self.manifest_path = manifest_path
        self.manifest = manifest
        self.recording = recording
        self.stage_name = stage_name
        self.run_id = run_id
        self.budget = budget
        self.session = session

    def post(self, url: str, **kwargs: Any):
        run_attempts = [
            attempt for attempt in self.manifest["request_attempts"]
            if attempt["run_id"] == self.run_id
        ]
        if len(run_attempts) >= self.budget:
            raise PilotBatchError("RequestBudgetExhausted")
        attempt = {
            "attempt_id": str(uuid.uuid4()),
            "run_id": self.run_id,
            "recording_relative_path": self.recording["recording_relative_path"],
            "stage": self.stage_name,
            "status": "dispatching",
            "started_at_utc": _now(),
        }
        self.manifest["request_attempts"].append(attempt)
        self.manifest["updated_at_utc"] = _now()
        _atomic_json(self.manifest_path, self.manifest)
        kwargs["allow_redirects"] = False
        try:
            response = self.session.post(url, **kwargs)
        except Exception as error:
            attempt["status"] = "unknown"
            attempt["error_class"] = type(error).__name__
            attempt["finished_at_utc"] = _now()
            self.manifest["updated_at_utc"] = _now()
            _atomic_json(self.manifest_path, self.manifest)
            raise
        status_code = getattr(response, "status_code", None)
        attempt["status"] = "response_received"
        attempt["http_status"] = status_code if isinstance(status_code, int) else None
        attempt["finished_at_utc"] = _now()
        self.manifest["updated_at_utc"] = _now()
        _atomic_json(self.manifest_path, self.manifest)
        return response


def _recover_unfinished_attempts(manifest: dict[str, Any]) -> list[str]:
    blockers = []
    for attempt in manifest["request_attempts"]:
        if attempt.get("status") in {"dispatching", "response_received"}:
            attempt["status"] = "unknown"
            attempt["recovered_at_utc"] = _now()
            blockers.append(
                f"request_outcome_unknown:{attempt['recording_relative_path']}:"
                f"{attempt['stage']}:{attempt['attempt_id']}"
            )
    return blockers


def authorize_stage_retry(
    manifest_path: Path,
    attempt_id: str,
    *,
    confirmed: bool,
) -> dict[str, Any]:
    if not confirmed:
        raise PilotBatchError("ExplicitRetryConfirmationRequired")
    manifest_path = manifest_path.resolve()
    with _manifest_lock(manifest_path):
        manifest = _load_manifest(manifest_path)
        attempt = next(
            (
                item for item in manifest["request_attempts"]
                if item.get("attempt_id") == attempt_id
            ),
            None,
        )
        if not isinstance(attempt, dict) or attempt.get("status") != "unknown":
            raise PilotBatchError("UnknownRequestAttemptNotFound")
        recording = next(
            item for item in manifest["recordings"]
            if item["recording_relative_path"] == attempt["recording_relative_path"]
        )
        stage = recording["stages"][attempt["stage"]]
        attempt["status"] = "retry_authorized"
        attempt["retry_authorized_at_utc"] = _now()
        stage["status"] = "pending"
        stage["retry_authorized"] = True
        stage["reason"] = "Operator authorized a possible duplicate request."
        manifest["events"].append(
            {
                "timestamp_utc": _now(),
                "event": "operator_authorized_stage_retry",
                "attempt_id": attempt_id,
                "recording_relative_path": recording["recording_relative_path"],
                "stage": attempt["stage"],
            }
        )
        manifest["updated_at_utc"] = _now()
        _atomic_json(manifest_path, manifest)
        return manifest


def _ensure_source_unchanged(recording: dict[str, Any]) -> Path:
    source = Path(recording["source"])
    if (
        not source.is_file()
        or source_fingerprint(source) != recording["source_fingerprint"]
        or _file_hash(source) != recording["source_sha256"]
    ):
        raise PilotBatchError(f"SourceFingerprintChanged:{recording['month']}")
    return source


def _unresolved_stage_attempts(
    manifest: dict[str, Any],
    recording: dict[str, Any],
    stage_name: str,
) -> list[dict[str, Any]]:
    return [
        attempt
        for attempt in manifest["request_attempts"]
        if attempt["recording_relative_path"] == recording["recording_relative_path"]
        and attempt["stage"] == stage_name
        and attempt["status"] != "retry_authorized"
        and attempt["status"] != "completed"
    ]


def _run_stage(
    manifest_path: Path,
    manifest: dict[str, Any],
    recording: dict[str, Any],
    stage_name: str,
    run_id: str,
    config: AzureConfig,
    api_version: str,
    terminology: str,
    budget: int,
    session: requests.Session,
    stage_root: Path,
) -> None:
    source = _ensure_source_unchanged(recording)
    stage = recording["stages"][stage_name]
    config_fingerprint = manifest["configuration_fingerprints"][stage_name]
    if _artifact_is_reusable(stage, config_fingerprint):
        return
    artifact_info = stage.get("artifact")
    if artifact_info:
        path = Path(artifact_info.get("path", ""))
        if not path.is_file() or _file_hash(path) != artifact_info.get("sha256"):
            stage["status"] = "artifact_invalid"
            manifest["updated_at_utc"] = _now()
            _atomic_json(manifest_path, manifest)
            raise PilotBatchError(f"StageArtifactHashMismatch:{recording['month']}:{stage_name}")
    unresolved = _unresolved_stage_attempts(manifest, recording, stage_name)
    if unresolved:
        stage["status"] = "unknown"
        raise PilotBatchError(
            f"OperatorReconciliationRequired:{recording['month']}:{stage_name}"
        )
    stage["status"] = "running"
    stage["retry_authorized"] = False
    manifest["updated_at_utc"] = _now()
    _atomic_json(manifest_path, manifest)
    tracker = _TrackedSession(
        manifest_path,
        manifest,
        recording,
        stage_name,
        run_id,
        budget,
        session,
    )
    directory = _stage_directory(recording, stage_root)
    directory.mkdir(parents=True, exist_ok=True)
    run_directory = directory / run_id
    run_directory.mkdir(parents=True, exist_ok=True)
    artifact_suffix = {
        "transcription": "raw",
        "cleanup": "phase9",
        "analysis": "phase10",
    }[stage_name]
    artifact_path = run_directory / f"{source.stem}.{artifact_suffix}.json"
    try:
        if stage_name == "transcription":
            transcript_text = AzureOpenAITranscriber(
                config,
                session=tracker,
            ).transcribe(source)
            _ensure_source_unchanged(recording)
            payload = {
                "source_filename": source.name,
                "source_fingerprint": recording["source_fingerprint"],
                "source_sha256": recording["source_sha256"],
                "transcription_config_fingerprint": config_fingerprint,
                "raw_transcript": transcript_text,
                "status": "completed",
            }
            artifact = _write_stage_artifact(artifact_path, payload)
            _set_stage_complete(
                manifest_path, manifest, recording, stage_name, run_id,
                config_fingerprint, "completed", artifact,
            )
            recording["stages"]["cleanup"]["status"] = "pending"
            _atomic_json(manifest_path, manifest)
            return

        transcription_info = recording["stages"]["transcription"].get("artifact")
        if not _artifact_is_reusable(
            recording["stages"]["transcription"],
            manifest["configuration_fingerprints"]["transcription"],
        ):
            raise PilotBatchError("ValidatedTranscriptStageRequired")
        raw_payload = json.loads(Path(transcription_info["path"]).read_text(encoding="utf-8"))
        raw_text = raw_payload.get("raw_transcript")
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise PilotBatchError("RawTranscriptArtifactInvalid")
        if stage_name == "cleanup":
            raw_segment = TranscriptSegment(None, None, None, raw_text, raw_text)
            cleaned = TranscriptCleanup(
                config,
                api_version=api_version,
                session=tracker,
                terminology=terminology,
            ).clean((raw_segment,))
            _ensure_source_unchanged(recording)
            phase9_payload = {
                "source_filename": source.name,
                "source_fingerprint": recording["source_fingerprint"],
                "source_sha256": recording["source_sha256"],
                "cleanup_config_fingerprint": config_fingerprint,
                "cleanup_status": cleaned.cleanup_status,
                "raw_transcript": cleaned.raw_transcript,
                "cleaned_transcript": cleaned.cleaned_transcript,
                "segments": [asdict(item) for item in cleaned.segments],
            }
            artifact = _write_stage_artifact(artifact_path, phase9_payload)
            status = (
                "completed_validated"
                if cleaned.cleanup_status == "completed"
                else "review_required"
            )
            _set_stage_complete(
                manifest_path, manifest, recording, stage_name, run_id,
                config_fingerprint, status, artifact, cleaned.review_reason,
            )
            if cleaned.cleanup_status == "completed":
                recording["stages"]["analysis"]["status"] = "pending"
            else:
                recording["stages"]["analysis"]["status"] = "blocked_until_cleanup"
            recording["stages"]["human_review"]["status"] = (
                "pending" if status == "completed_validated" else "blocked_until_cleanup_review"
            )
            _atomic_json(manifest_path, manifest)
            return

        cleanup_info = recording["stages"]["cleanup"].get("artifact")
        if not _artifact_is_reusable(
            recording["stages"]["cleanup"],
            manifest["configuration_fingerprints"]["cleanup"],
        ):
            raise PilotBatchError("ValidatedCleanupStageRequired")
        phase9_path = Path(cleanup_info["path"])
        transcript = load_validated_transcript(phase9_path, source)
        analysis = GPTMeetingAnalyzer(
            config,
            api_version=api_version,
            session=tracker,
            terminology=terminology,
        ).analyze(transcript.segments)
        _ensure_source_unchanged(recording)
        payload = {
            "source_filename": source.name,
            "source_fingerprint": recording["source_fingerprint"],
            "source_sha256": recording["source_sha256"],
            "phase9_transcript_json": str(phase9_path.resolve()),
            "phase9_json_sha256": _file_hash(phase9_path),
            "analysis_config_fingerprint": config_fingerprint,
            "meeting_analysis_requests": 1,
            "meeting_analysis": analysis.to_dict(),
        }
        artifact = _write_stage_artifact(artifact_path, payload)
        if analysis.status == "validated":
            status = "completed_validated"
        elif analysis.review_reason == "SemanticReviewRequired":
            status = "review_required"
        else:
            status = "review_required"
        _set_stage_complete(
            manifest_path, manifest, recording, stage_name, run_id,
            config_fingerprint, status, artifact, analysis.review_reason,
        )
        recording["stages"][stage_name]["validation"] = (
            "passed"
            if analysis.status == "validated"
            else (
                "semantic_review_required"
                if analysis.review_reason == "SemanticReviewRequired"
                else "failed"
            )
        )
        recording["stages"]["human_review"]["status"] = {
            "review_required": "pending",
            "completed_validated": "required",
            "failed": "blocked_invalid_analysis",
        }[status]
        if recording["stages"][stage_name]["validation"] == "failed":
            recording["stages"]["human_review"]["status"] = "blocked_invalid_analysis"
        recording["stages"]["approved_pdf"]["status"] = "blocked_until_human_approval"
        _atomic_json(manifest_path, manifest)
    except PilotBatchError:
        if stage.get("status") == "running":
            stage["status"] = "blocked"
            manifest["updated_at_utc"] = _now()
            _atomic_json(manifest_path, manifest)
        raise
    except Exception as error:
        stage["status"] = "unknown" if _unresolved_stage_attempts(
            manifest, recording, stage_name
        ) else "failed"
        stage["reason"] = type(error).__name__
        manifest["updated_at_utc"] = _now()
        _atomic_json(manifest_path, manifest)
        raise PilotBatchError(
            f"StageFailed:{recording['month']}:{stage_name}:{type(error).__name__}"
        ) from error


def execute_pilot_batch(
    *,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    stage_root: Path = DEFAULT_STAGE_ROOT,
    max_requests: int | None = None,
    config: AzureConfig | None = None,
    api_version: str | None = None,
    session_factory: Any = requests.Session,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    stage_root = stage_root.resolve()
    if not manifest_path.is_file():
        raise PilotBatchError("DryRunManifestRequired")
    azure_config = config or get_azure_config()
    text_api_version = api_version or get_azure_openai_api_version()
    terminology = os.getenv("MORNINGTALK_TERMINOLOGY", "")
    with _manifest_lock(manifest_path):
        manifest = _load_manifest(manifest_path)
        _, config_blockers = _configuration_preflight(
            azure_config,
            text_api_version,
        )
        if config_blockers:
            raise PilotBatchError(",".join(config_blockers))
        hashes = _stage_config_fingerprints(
            azure_config,
            text_api_version,
            terminology,
        )
        for stage in _STAGES:
            if hashes[stage] != manifest["configuration_fingerprints"].get(stage):
                raise PilotBatchError(f"ConfigurationChanged:{stage}")
        roots = manifest.get("paths", {})
        saved_stage_root = Path(roots.get("stage_root", DEFAULT_STAGE_ROOT)).resolve()
        if stage_root != saved_stage_root:
            raise PilotBatchError("StageOutputPathChanged")
        blockers = _validate_saved_recordings(
            manifest["recordings"],
            Path(roots.get("recording_root", DEFAULT_RECORDING_ROOT)),
            Path(roots.get("sara_root", DEFAULT_SARA_ROOT)),
            Path(roots.get("preview_root", DEFAULT_PREVIEW_ROOT)),
            Path(roots.get("review_root", DEFAULT_REVIEW_ROOT)),
        )
        if blockers:
            raise PilotBatchError(",".join(blockers))
        unknown = _recover_unfinished_attempts(manifest)
        if unknown:
            manifest["execution_status"] = "unknown_request_outcome"
            manifest["updated_at_utc"] = _now()
            _atomic_json(manifest_path, manifest)
            raise PilotBatchError(",".join(unknown))
        run_id = str(uuid.uuid4())
        budget = max_requests
        if budget is None:
            budget = manifest["request_budget"]["per_run_limit"]
        if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
            raise PilotBatchError("RequestBudgetInvalid")
        session = session_factory()
        if hasattr(session, "mount"):
            session.mount("https://", HTTPAdapter(max_retries=0))
            session.mount("http://", HTTPAdapter(max_retries=0))
        manifest["execution_status"] = "running"
        manifest["active_run_id"] = run_id
        manifest["updated_at_utc"] = _now()
        _atomic_json(manifest_path, manifest)
        try:
            for recording in manifest["recordings"]:
                for stage_name in _STAGES:
                    if (
                        stage_name == "cleanup"
                        and recording["stages"]["cleanup"]["status"]
                        == "blocked_until_transcription"
                    ):
                        recording["stages"]["cleanup"]["status"] = "pending"
                    if (
                        stage_name == "analysis"
                        and recording["stages"]["analysis"]["status"]
                        == "blocked_until_cleanup"
                    ):
                        recording["stages"]["analysis"]["status"] = "pending"
                    if (
                        stage_name == "analysis"
                        and recording["stages"]["cleanup"]["status"]
                        != "completed_validated"
                    ):
                        continue
                    _run_stage(
                        manifest_path,
                        manifest,
                        recording,
                        stage_name,
                        run_id,
                        azure_config,
                        text_api_version,
                        terminology,
                        budget,
                        session,
                        saved_stage_root,
                    )
            if any(
                item["stages"]["cleanup"]["status"] == "review_required"
                for item in manifest["recordings"]
            ):
                manifest["execution_status"] = "cleanup_review_required"
            elif any(
                item["stages"]["analysis"].get("validation") == "failed"
                for item in manifest["recordings"]
            ):
                manifest["execution_status"] = "analysis_validation_failed"
            else:
                manifest["execution_status"] = "awaiting_human_review"
            manifest["updated_at_utc"] = _now()
            _atomic_json(manifest_path, manifest)
            return manifest
        except PilotBatchError:
            manifest["execution_status"] = "blocked"
            manifest["updated_at_utc"] = _now()
            _atomic_json(manifest_path, manifest)
            raise
        finally:
            close = getattr(session, "close", None)
            if callable(close):
                close()


def build_dry_run_report(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    stage_root: Path = DEFAULT_STAGE_ROOT,
) -> tuple[Path, Path, list[str]]:
    _, blockers = prepare_pilot_dry_run(
        manifest_path=manifest_path,
        report_path=report_path,
        stage_root=stage_root,
    )
    return manifest_path, report_path, blockers
