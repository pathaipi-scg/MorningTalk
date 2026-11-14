"""Analysis-only resume from a validated Phase 9 transcript."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
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
from app.file_manager import resolve_meeting_output_paths, source_fingerprint
from app.meeting_analysis import (
    GPTMeetingAnalyzer,
    MeetingAnalysis,
    MeetingAnalysisError,
)
from app.pdf_export import write_meeting_summary_pdf
from app.transcript import (
    TranscriptSegment,
    preserves_numbers_and_codes,
    validate_segment_times,
)
from app.workspace_paths import get_workspace_root

logger = logging.getLogger(__name__)
EXPECTED_RECORDING = "meeting-20261107-0817.mp3"
EXPECTED_ENDPOINT = (
    "https://smartagent-gummo.openai.azure.com/openai/responses"
    "?api-version=2025-04-01-preview"
)
EXPECTED_DEPLOYMENT = "gpt-5.4-mini"


class AnalysisResumePreflightError(ValueError):
    """Safe-to-report preflight failure; no Azure request has been made."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class AnalysisResumePreflight:
    source: Path
    phase9_json: Path
    output_dir: Path
    segments: tuple[TranscriptSegment, ...] = field(repr=False)
    config: AzureConfig = field(repr=False)
    api_version: str = field(repr=False)
    source_fingerprint_value: str = field(repr=False)
    phase9_json_sha256: str = field(repr=False)
    endpoint_url: str
    api_mode: str
    deployment: str
    raw_characters: int
    cleaned_characters: int
    fingerprint_matches: bool
    requests_planned: int = 1
    retries_enabled: bool = False


@dataclass(frozen=True)
class AnalysisResumeResult:
    analysis: MeetingAnalysis
    json_path: Path
    summary_pdf: Path | None


@dataclass(frozen=True)
class PreparedAnalysisAttempt:
    attempt_id: str
    output_dir: Path
    request_path: Path
    raw_response_path: Path
    metadata: dict[str, Any] = field(repr=False)


def _write_attempt_metadata(path: Path, payload: dict[str, Any]) -> None:
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


def prepare_analysis_attempt(
    phase9_json_path: Path,
    source: Path,
    output_dir: Path,
    *,
    config: AzureConfig | None = None,
    api_version: str | None = None,
) -> PreparedAnalysisAttempt:
    preflight = preflight_analysis_resume(
        phase9_json_path,
        source,
        output_dir,
        config=config,
        api_version=api_version,
    )
    attempt_id = uuid.uuid4().hex
    request_path = preflight.output_dir / "request.json"
    raw_response_path = preflight.output_dir / "raw_response.json"
    metadata: dict[str, Any] = {
        "attempt_id": attempt_id,
        "status": "prepared",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(preflight.source),
        "source_fingerprint": preflight.source_fingerprint_value,
        "phase9_transcript_json": str(preflight.phase9_json),
        "phase9_json_sha256": preflight.phase9_json_sha256,
        "endpoint_url": preflight.endpoint_url,
        "api_mode": preflight.api_mode,
        "deployment": preflight.deployment,
        "api_version": preflight.api_version,
        "expected_requests": 1,
        "automatic_retries": False,
        "raw_response_path": str(raw_response_path),
        "transcript_characters": preflight.cleaned_characters,
        "estimated_input_tokens_from_prior_same_transcript_request": 4587,
        "prior_same_transcript_output_tokens": 3428,
        "max_output_tokens": 12000,
    }
    preflight.output_dir.parent.mkdir(parents=True, exist_ok=True)
    preflight.output_dir.mkdir(exist_ok=False)
    try:
        with request_path.open("x", encoding="utf-8") as output:
            json.dump(metadata, output, ensure_ascii=False, indent=2)
            output.flush()
            os.fsync(output.fileno())
    except Exception:
        request_path.unlink(missing_ok=True)
        preflight.output_dir.rmdir()
        raise
    return PreparedAnalysisAttempt(
        attempt_id,
        preflight.output_dir,
        request_path,
        raw_response_path,
        metadata,
    )


def preflight_analysis_resume(
    phase9_json_path: Path,
    source: Path,
    output_dir: Path,
    *,
    config: AzureConfig | None = None,
    api_version: str | None = None,
    attempt_id: str | None = None,
) -> AnalysisResumePreflight:
    phase9_json_path = phase9_json_path.resolve()
    source = source.resolve()
    output_dir = output_dir.resolve()
    preview_root = (get_workspace_root() / "previews").resolve()
    try:
        relative_output = output_dir.relative_to(preview_root)
    except ValueError as error:
        raise AnalysisResumePreflightError("OutputOutsidePreviewRoot") from error
    if not relative_output.parts:
        raise AnalysisResumePreflightError("OutputMustBeIsolatedSubdirectory")
    prepared_metadata: dict[str, Any] | None = None
    if output_dir.exists():
        if attempt_id is None:
            raise AnalysisResumePreflightError("PreviewDirectoryAlreadyExists")
        try:
            candidate = json.loads(
                (output_dir / "request.json").read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, ValueError) as error:
            raise AnalysisResumePreflightError("PreparedAttemptUnavailable") from error
        if (
            not isinstance(candidate, dict)
            or candidate.get("attempt_id") != attempt_id
            or candidate.get("status") != "prepared"
            or candidate.get("source") != str(source)
            or candidate.get("phase9_transcript_json") != str(phase9_json_path)
        ):
            raise AnalysisResumePreflightError("PreparedAttemptMismatchOrAlreadyUsed")
        prepared_metadata = candidate
    elif attempt_id is not None:
        raise AnalysisResumePreflightError("PreparedAttemptUnavailable")
    if source.name.casefold() != EXPECTED_RECORDING.casefold():
        raise AnalysisResumePreflightError("UnexpectedSourceRecording")
    if not source.is_file():
        raise AnalysisResumePreflightError("SourceRecordingUnavailable")
    if not phase9_json_path.is_file():
        raise AnalysisResumePreflightError("Phase9JsonUnavailable")

    try:
        phase9_json_bytes = phase9_json_path.read_bytes()
        payload: Any = json.loads(phase9_json_bytes.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise AnalysisResumePreflightError("Phase9JsonInvalid") from error
    if not isinstance(payload, dict):
        raise AnalysisResumePreflightError("Phase9JsonInvalid")
    if payload.get("cleanup_status") != "completed":
        raise AnalysisResumePreflightError("TranscriptCleanupNotValidated")
    existing_analysis = payload.get("meeting_analysis")
    if not isinstance(existing_analysis, dict) or existing_analysis.get("status") != "review_required":
        raise AnalysisResumePreflightError("ExistingAnalysisNotReviewRequired")

    raw_transcript = payload.get("raw_transcript")
    cleaned_transcript = payload.get("cleaned_transcript")
    raw_segments = payload.get("segments")
    if (
        not isinstance(raw_transcript, str)
        or not raw_transcript
        or not isinstance(cleaned_transcript, str)
        or not cleaned_transcript
        or not isinstance(raw_segments, list)
        or not raw_segments
    ):
        raise AnalysisResumePreflightError("TranscriptDataMissing")

    segments = []
    for item in raw_segments:
        if not isinstance(item, dict):
            raise AnalysisResumePreflightError("TranscriptSegmentInvalid")
        raw_text = item.get("raw_text")
        cleaned_text = item.get("cleaned_text")
        speaker_id = item.get("speaker_id")
        start = item.get("start_seconds")
        end = item.get("end_seconds")
        if (
            not isinstance(raw_text, str)
            or not isinstance(cleaned_text, str)
            or item.get("review_required") is not False
            or speaker_id is not None and not isinstance(speaker_id, str)
            or start is not None and (isinstance(start, bool) or not isinstance(start, (int, float)))
            or end is not None and (isinstance(end, bool) or not isinstance(end, (int, float)))
        ):
            raise AnalysisResumePreflightError("TranscriptSegmentInvalid")
        segments.append(TranscriptSegment(start, end, speaker_id, raw_text, cleaned_text))

    transcript_segments = tuple(segments)
    try:
        validate_segment_times(transcript_segments)
    except ValueError as error:
        raise AnalysisResumePreflightError("TranscriptSegmentsInvalid") from error
    if "\n".join(segment.raw_text for segment in transcript_segments) != raw_transcript:
        raise AnalysisResumePreflightError("RawTranscriptInconsistent")
    if "\n".join(segment.display_text for segment in transcript_segments) != cleaned_transcript:
        raise AnalysisResumePreflightError("CleanedTranscriptInconsistent")
    if not preserves_numbers_and_codes(raw_transcript, cleaned_transcript):
        raise AnalysisResumePreflightError("NumberOrCodeMismatch")

    expected_fingerprint = payload.get("source_fingerprint")
    if (
        not isinstance(expected_fingerprint, str)
        or not expected_fingerprint
        or source_fingerprint(source).casefold() != expected_fingerprint.casefold()
    ):
        raise AnalysisResumePreflightError("SourceFingerprintMismatch")
    phase9_json_sha256 = hashlib.sha256(phase9_json_bytes).hexdigest()
    if prepared_metadata is not None and (
        prepared_metadata.get("source_fingerprint") != expected_fingerprint
        or prepared_metadata.get("phase9_json_sha256") != phase9_json_sha256
    ):
        raise AnalysisResumePreflightError("PreparedAttemptSourcesStale")

    azure_config = config or get_azure_config()
    version = api_version or get_azure_openai_api_version()
    if (
        not azure_config.text_endpoint
        or not azure_config.text_api_key
        or not azure_config.text_deployment
    ):
        raise AnalysisResumePreflightError("AzureOpenAIConfigurationIncomplete")
    try:
        endpoint = resolve_azure_openai_endpoint(
            azure_config.text_endpoint,
            version,
            azure_config.text_deployment,
        )
    except ValueError as error:
        raise AnalysisResumePreflightError("AzureEndpointInvalid") from error
    if (
        endpoint.api_mode != "responses"
        or endpoint.url != EXPECTED_ENDPOINT
        or azure_config.text_deployment != EXPECTED_DEPLOYMENT
    ):
        raise AnalysisResumePreflightError("UnexpectedAzureEndpointOrDeployment")
    if prepared_metadata is not None and (
        prepared_metadata.get("endpoint_url") != endpoint.url
        or prepared_metadata.get("deployment") != azure_config.text_deployment
        or prepared_metadata.get("api_version") != version
    ):
        raise AnalysisResumePreflightError("PreparedAttemptConfigurationStale")

    return AnalysisResumePreflight(
        source=source,
        phase9_json=phase9_json_path,
        output_dir=output_dir,
        segments=transcript_segments,
        config=azure_config,
        api_version=version,
        source_fingerprint_value=expected_fingerprint,
        phase9_json_sha256=phase9_json_sha256,
        endpoint_url=endpoint.url,
        api_mode=endpoint.api_mode,
        deployment=azure_config.text_deployment,
        raw_characters=len(raw_transcript),
        cleaned_characters=len(cleaned_transcript),
        fingerprint_matches=True,
    )


def run_analysis_resume(
    phase9_json_path: Path,
    source: Path,
    output_dir: Path,
    *,
    session: Any = None,
    attempt_id: str | None = None,
    confirm_azure_request: bool = False,
) -> AnalysisResumeResult:
    if attempt_id is not None and not confirm_azure_request:
        raise AnalysisResumePreflightError("ExplicitAzureAuthorizationRequired")
    preflight = preflight_analysis_resume(
        phase9_json_path,
        source,
        output_dir,
        attempt_id=attempt_id,
    )
    request_path = preflight.output_dir / "request.json"
    request_metadata: dict[str, Any] | None = None
    raw_response_path = preflight.output_dir / "raw_response.json"
    if attempt_id is not None:
        request_metadata = json.loads(request_path.read_text(encoding="utf-8"))
        request_metadata.update({
            "status": "dispatching",
            "dispatched_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        _write_attempt_metadata(request_path, request_metadata)

    def preserve_response(response: Any) -> None:
        if request_metadata is None:
            return
        raw_bytes = response.content
        if not isinstance(raw_bytes, bytes):
            raise TypeError("Azure response body must be bytes for raw preservation.")
        with raw_response_path.open("xb") as raw_file:
            raw_file.write(raw_bytes)
            raw_file.flush()
            os.fsync(raw_file.fileno())
        response_headers = getattr(response, "headers", {})
        safe_headers = {}
        if hasattr(response_headers, "get"):
            for name in (
                "apim-request-id",
                "x-ms-request-id",
                "x-request-id",
                "content-type",
            ):
                value = response_headers.get(name)
                if isinstance(value, str):
                    safe_headers[name] = value
        request_metadata.update({
            "status": "response_received",
            "response_received_at_utc": datetime.now(timezone.utc).isoformat(),
            "http_status": response.status_code,
            "response_headers": safe_headers,
            "raw_response_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "raw_response_bytes": len(raw_bytes),
        })
        _write_attempt_metadata(request_path, request_metadata)

    owned_session = session is None
    http_session = session if session is not None else requests.Session()
    if owned_session:
        no_retry_adapter = HTTPAdapter(max_retries=0)
        http_session.mount("https://", no_retry_adapter)

    try:
        analysis = GPTMeetingAnalyzer(
            preflight.config,
            api_version=preflight.api_version,
            session=http_session,
            terminology=os.getenv("MORNINGTALK_TERMINOLOGY", ""),
        ).analyze(
            preflight.segments,
            on_response=preserve_response if attempt_id is not None else None,
        )
    except MeetingAnalysisError:
        if request_metadata is not None and request_metadata["status"] == "dispatching":
            request_metadata.update({
                "status": "unknown",
                "failure_class": "MeetingAnalysisError",
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            _write_attempt_metadata(request_path, request_metadata)
        raise
    except Exception as error:
        if request_metadata is not None and request_metadata["status"] == "dispatching":
            request_metadata.update({
                "status": "unknown",
                "failure_class": type(error).__name__,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            _write_attempt_metadata(request_path, request_metadata)
        raise
    finally:
        if owned_session:
            http_session.close()

    if analysis.status == "validated" and (
        analysis.overview is None or not analysis.topics or not analysis.key_issues
    ):
        analysis = MeetingAnalysis.for_review(
            "InsufficientStructuredSummary",
            analysis.input_tokens,
            analysis.output_tokens,
        )
    if source_fingerprint(preflight.source).casefold() != preflight.source_fingerprint_value.casefold():
        raise AnalysisResumePreflightError("SourceFingerprintChangedDuringAnalysis")
    try:
        current_phase9_hash = hashlib.sha256(
            preflight.phase9_json.read_bytes()
        ).hexdigest()
    except OSError as error:
        raise AnalysisResumePreflightError("Phase9JsonChangedDuringAnalysis") from error
    if current_phase9_hash != preflight.phase9_json_sha256:
        raise AnalysisResumePreflightError("Phase9JsonChangedDuringAnalysis")

    preflight.output_dir.parent.mkdir(parents=True, exist_ok=True)
    if attempt_id is None:
        preflight.output_dir.mkdir(exist_ok=False)
    json_path = preflight.output_dir / f"{preflight.source.stem}.phase10.json"
    summary_path = resolve_meeting_output_paths(
        preflight.source
    ).summary_draft_pdf
    payload = {
        "source_filename": preflight.source.name,
        "source_fingerprint": source_fingerprint(preflight.source),
        "phase9_transcript_json": str(preflight.phase9_json),
        "phase9_json_sha256": preflight.phase9_json_sha256,
        "evidence_offset_unit": (
            "zero-based Python Unicode code points; start inclusive, end exclusive"
        ),
        "cleanup_status": "completed",
        "transcription_requests": 0,
        "cleanup_requests": 0,
        "meeting_analysis_requests": 1,
        "attempt_id": attempt_id,
        "raw_response_path": str(raw_response_path) if attempt_id is not None else None,
        "meeting_analysis": analysis.to_dict(),
    }
    with json_path.open("x", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
    if request_metadata is not None:
        request_metadata.update({
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "meeting_analysis_status": analysis.status,
            "meeting_analysis_review_reason": analysis.review_reason,
            "input_tokens": analysis.input_tokens,
            "output_tokens": analysis.output_tokens,
            "analysis_json_path": str(json_path),
        })
        _write_attempt_metadata(request_path, request_metadata)

    if analysis.status != "validated":
        logger.warning(
            "Meeting analysis did not validate; rule=%s; no summary PDF created.",
            analysis.review_reason or "UnknownValidationFailure",
        )
        return AnalysisResumeResult(analysis, json_path, None)

    write_meeting_summary_pdf(
        summary_path,
        preflight.source,
        preflight.source.parent.parent.name,
        analysis,
        overwrite=True,
        draft=True,
    )
    return AnalysisResumeResult(analysis, json_path, summary_path)
