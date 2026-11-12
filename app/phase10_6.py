"""Offline corpus evaluation and approval-gated summary publishing."""

from copy import deepcopy
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path, PureWindowsPath
import tempfile
from typing import Any
import uuid

from app.file_manager import source_fingerprint
from app.meeting_analysis import (
    AnalysisItem,
    GPTMeetingAnalyzer,
    MeetingAnalysis,
)
from app.pdf_export import write_meeting_summary_pdf
from app.transcript import (
    TranscriptSegment,
    preserves_numbers_and_codes,
    validate_segment_times,
)

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "references" / "SaraAI" / "analysis"
DEFAULT_RECORDING_ROOT = Path(r"D:\KM\Vault\MorningTalk\Record File\CB")
DEFAULT_SARA_ROOT = PROJECT_ROOT / "references" / "SaraAI"
DEFAULT_PREVIEW_ROOT = (
    Path(os.getenv("LOCALAPPDATA", str(Path.home())))
    / "MorningTalk"
    / "previews"
)
DEFAULT_REVIEW_ROOT = (
    Path(os.getenv("LOCALAPPDATA", str(Path.home())))
    / "MorningTalk"
    / "reviews"
)
_SECTIONS = (
    "overview",
    "classifications",
    "topics",
    "key_issues",
    "actions",
    "dependencies",
    "schedules",
    "risks",
    "closing_activities",
)
_ZERO_HASH = "0" * 64


class ReviewWorkflowError(ValueError):
    """Safe-to-report review or batch-evaluation failure."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class ValidatedTranscript:
    path: Path
    sha256: str
    source_fingerprint_value: str
    segments: tuple[TranscriptSegment, ...]
    transcript_payload: dict[str, Any]


@dataclass(frozen=True)
class ReviewBundle:
    source: Path
    transcript: ValidatedTranscript
    analysis_path: Path
    analysis_sha256: str
    analysis_payload: dict[str, Any]
    analysis: MeetingAnalysis
    review_state_path: Path


@dataclass(frozen=True)
class BatchEvaluation:
    csv_path: Path
    markdown_path: Path
    rows: tuple[dict[str, str], ...]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path, *, error_reason: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise ReviewWorkflowError(error_reason) from error
    if not isinstance(payload, dict):
        raise ReviewWorkflowError(error_reason)
    return payload, raw


def load_validated_transcript(
    transcript_path: Path,
    source: Path,
    *,
    expected_sha256: str | None = None,
) -> ValidatedTranscript:
    transcript_path = transcript_path.resolve()
    source = source.resolve()
    payload, raw = _read_json(transcript_path, error_reason="TranscriptJsonInvalid")
    digest = _sha256(raw)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ReviewWorkflowError("TranscriptHashMismatch")
    if payload.get("cleanup_status") != "completed":
        raise ReviewWorkflowError("TranscriptCleanupNotValidated")
    expected_fingerprint = payload.get("source_fingerprint")
    if (
        not isinstance(expected_fingerprint, str)
        or not expected_fingerprint
        or source_fingerprint(source).casefold() != expected_fingerprint.casefold()
    ):
        raise ReviewWorkflowError("SourceFingerprintMismatch")

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
        raise ReviewWorkflowError("TranscriptDataMissing")

    segments: list[TranscriptSegment] = []
    for item in raw_segments:
        if not isinstance(item, dict):
            raise ReviewWorkflowError("TranscriptSegmentInvalid")
        start = item.get("start_seconds")
        end = item.get("end_seconds")
        speaker_id = item.get("speaker_id")
        raw_text = item.get("raw_text")
        cleaned_text = item.get("cleaned_text")
        if (
            not isinstance(raw_text, str)
            or not isinstance(cleaned_text, str)
            or item.get("review_required") is not False
            or speaker_id is not None and not isinstance(speaker_id, str)
            or start is not None
            and (isinstance(start, bool) or not isinstance(start, (int, float)))
            or end is not None
            and (isinstance(end, bool) or not isinstance(end, (int, float)))
        ):
            raise ReviewWorkflowError("TranscriptSegmentInvalid")
        segments.append(
            TranscriptSegment(start, end, speaker_id, raw_text, cleaned_text)
        )

    segment_tuple = tuple(segments)
    try:
        validate_segment_times(segment_tuple)
    except ValueError as error:
        raise ReviewWorkflowError("TranscriptSegmentsInvalid") from error
    if "\n".join(segment.raw_text for segment in segment_tuple) != raw_transcript:
        raise ReviewWorkflowError("RawTranscriptInconsistent")
    if "\n".join(segment.display_text for segment in segment_tuple) != cleaned_transcript:
        raise ReviewWorkflowError("CleanedTranscriptInconsistent")
    if not preserves_numbers_and_codes(raw_transcript, cleaned_transcript):
        raise ReviewWorkflowError("TranscriptNumberOrCodeMismatch")

    return ValidatedTranscript(
        transcript_path,
        digest,
        expected_fingerprint,
        segment_tuple,
        payload,
    )


def _analysis_item_count(payload: Any) -> int:
    if not isinstance(payload, dict):
        return 0
    count = int(isinstance(payload.get("overview"), dict))
    for section in _SECTIONS[1:]:
        value = payload.get(section, [])
        if isinstance(value, list):
            count += len(value)
    return count


def _parse_analysis(
    analysis_payload: dict[str, Any],
    transcript: ValidatedTranscript,
) -> MeetingAnalysis:
    analysis = GPTMeetingAnalyzer._parse_and_validate(
        analysis_payload,
        transcript.segments,
    )
    if analysis.status == "validated":
        return analysis
    if (
        analysis.status == "review_required"
        and analysis.review_reason == "SemanticReviewRequired"
    ):
        return analysis
    raise ReviewWorkflowError(
        analysis.review_reason or "EvidenceValidationFailed"
    )


def _review_id(source: Path) -> str:
    value = os.path.normcase(str(source.resolve())).encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:12]


def _review_state_path(source: Path, review_dir: Path) -> Path:
    return review_dir / f"{source.stem}-{_review_id(source)}.review.json"


def load_review_bundle(
    analysis_path: Path,
    transcript_path: Path,
    source: Path,
    review_dir: Path = DEFAULT_REVIEW_ROOT,
) -> ReviewBundle:
    analysis_path = analysis_path.resolve()
    source = source.resolve()
    analysis_payload, analysis_bytes = _read_json(
        analysis_path,
        error_reason="AnalysisJsonInvalid",
    )
    analysis_hash = _sha256(analysis_bytes)
    transcript_path = transcript_path.resolve()
    if analysis_path == transcript_path:
        expected_transcript_hash = analysis_hash
    else:
        analysis_transcript_path = analysis_payload.get("phase9_transcript_json")
        expected_transcript_hash = analysis_payload.get("phase9_json_sha256")
        if (
            not isinstance(analysis_transcript_path, str)
            or Path(analysis_transcript_path).resolve() != transcript_path
            or not isinstance(expected_transcript_hash, str)
        ):
            raise ReviewWorkflowError("AnalysisTranscriptLinkMissingOrStale")
    transcript = load_validated_transcript(
        transcript_path,
        source,
        expected_sha256=expected_transcript_hash,
    )
    if analysis_payload.get("source_filename") not in (None, source.name):
        raise ReviewWorkflowError("AnalysisSourceFilenameMismatch")
    expected_fingerprint = analysis_payload.get("source_fingerprint")
    if (
        expected_fingerprint is not None
        and expected_fingerprint.casefold()
        != transcript.source_fingerprint_value.casefold()
    ):
        raise ReviewWorkflowError("AnalysisSourceFingerprintMismatch")
    raw_analysis = analysis_payload.get("meeting_analysis")
    if not isinstance(raw_analysis, dict) or _analysis_item_count(raw_analysis) == 0:
        raise ReviewWorkflowError("AnalysisItemsUnavailable")
    analysis = _parse_analysis(raw_analysis, transcript)
    return ReviewBundle(
        source,
        transcript,
        analysis_path,
        analysis_hash,
        raw_analysis,
        analysis,
        _review_state_path(source, review_dir),
    )


def _iter_analysis_items(payload: dict[str, Any]):
    overview = payload.get("overview")
    if isinstance(overview, dict):
        yield "overview", 1, overview
    for section in _SECTIONS[1:]:
        values = payload.get(section, [])
        if isinstance(values, list):
            for index, item in enumerate(values, 1):
                if isinstance(item, dict):
                    yield section, index, item


def _item_key(section: str, index: int) -> str:
    return f"{section}:{index}"


def _item_sources(
    item: dict[str, Any],
    segments: tuple[TranscriptSegment, ...],
) -> list[str]:
    refs = item.get("evidence")
    if not isinstance(refs, list) or not refs:
        raise ReviewWorkflowError("EvidenceReferencesMissing")
    result: list[str] = []
    for reference in refs:
        if not isinstance(reference, dict):
            raise ReviewWorkflowError("EvidenceReferenceInvalid")
        segment_id = reference.get("segment_id")
        if not isinstance(segment_id, int) or isinstance(segment_id, bool):
            raise ReviewWorkflowError("EvidenceSegmentIdInvalid")
        if segment_id < 1 or segment_id > len(segments):
            raise ReviewWorkflowError("EvidenceSegmentIdNotFound")
        if "start_char" in reference or "end_char" in reference:
            start = reference.get("start_char")
            end = reference.get("end_char")
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(end, bool)
                or not isinstance(end, int)
                or start < 0
                or start >= end
                or end > len(segments[segment_id - 1].display_text)
            ):
                raise ReviewWorkflowError("EvidenceOffsetInvalid")
            quote = segments[segment_id - 1].display_text[start:end]
            if not quote.strip():
                raise ReviewWorkflowError("EvidenceSpanWhitespaceOnly")
        else:
            quote = reference.get("quote")
            if not isinstance(quote, str) or not quote.strip():
                raise ReviewWorkflowError("LegacyEvidenceQuoteMissing")
        result.append(quote)
    return result


def _new_review_state(bundle: ReviewBundle, reviewer: str) -> dict[str, Any]:
    items = {}
    for section, index, item in _iter_analysis_items(bundle.analysis_payload):
        items[_item_key(section, index)] = {
            "section": section,
            "index": index,
            "original": deepcopy(item),
            "edited": None,
            "decision": "pending",
            "reviewer": None,
            "reviewed_at": None,
            "rejection_reason": None,
        }
    state: dict[str, Any] = {
        "version": 1,
        "source": str(bundle.source),
        "source_fingerprint": bundle.transcript.source_fingerprint_value,
        "analysis_json": str(bundle.analysis_path),
        "analysis_sha256": bundle.analysis_sha256,
        "transcript_json": str(bundle.transcript.path),
        "transcript_sha256": bundle.transcript.sha256,
        "status": "review_pending",
        "approved_pdf": None,
        "items": items,
        "audit": [],
    }
    _append_audit(
        state,
        reviewer,
        "review_started",
        None,
        {"item_count": len(items)},
    )
    return state


def _audit_hash(event: dict[str, Any]) -> str:
    canonical = json.dumps(
        event,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256(canonical)


def _append_audit(
    state: dict[str, Any],
    reviewer: str,
    action: str,
    item_id: str | None,
    details: dict[str, Any],
) -> None:
    audit = state.setdefault("audit", [])
    previous_hash = audit[-1]["event_hash"] if audit else _ZERO_HASH
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer": reviewer.strip(),
        "action": action,
        "item_id": item_id,
        "details": details,
        "previous_hash": previous_hash,
    }
    event["event_hash"] = _audit_hash(event)
    audit.append(event)


def _verify_audit(state: dict[str, Any]) -> None:
    previous_hash = _ZERO_HASH
    audit = state.get("audit")
    if not isinstance(audit, list):
        raise ReviewWorkflowError("ReviewAuditInvalid")
    for event in audit:
        if not isinstance(event, dict):
            raise ReviewWorkflowError("ReviewAuditInvalid")
        event_body = {key: value for key, value in event.items() if key != "event_hash"}
        if (
            event.get("previous_hash") != previous_hash
            or event.get("event_hash") != _audit_hash(event_body)
        ):
            raise ReviewWorkflowError("ReviewAuditTampered")
        previous_hash = event["event_hash"]


def _write_review_state(path: Path, state: dict[str, Any], *, create: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = json.dumps(state, ensure_ascii=False, indent=2)
    if create:
        try:
            with path.open("x", encoding="utf-8") as output:
                output.write(contents)
                output.flush()
                os.fsync(output.fileno())
        except FileExistsError as error:
            raise ReviewWorkflowError("ReviewAlreadyStarted") from error
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_or_start_review(
    bundle: ReviewBundle,
    reviewer: str,
    *,
    start_if_missing: bool,
) -> dict[str, Any]:
    if not reviewer.strip():
        raise ReviewWorkflowError("ReviewerRequired")
    try:
        if source_fingerprint(bundle.source).casefold() != (
            bundle.transcript.source_fingerprint_value.casefold()
        ):
            raise ReviewWorkflowError("SourceFingerprintMismatch")
        if _sha256(bundle.transcript.path.read_bytes()) != bundle.transcript.sha256:
            raise ReviewWorkflowError("TranscriptHashMismatch")
        if _sha256(bundle.analysis_path.read_bytes()) != bundle.analysis_sha256:
            raise ReviewWorkflowError("AnalysisHashMismatch")
    except OSError as error:
        raise ReviewWorkflowError("ReviewSourceUnavailable") from error
    path = bundle.review_state_path
    if not path.exists():
        if not start_if_missing:
            raise ReviewWorkflowError("ReviewNotStarted")
        state = _new_review_state(bundle, reviewer)
        _write_review_state(path, state, create=True)
        return state
    state, _raw = _read_json(path, error_reason="ReviewStateInvalid")
    _verify_audit(state)
    expected = {
        "source": str(bundle.source),
        "source_fingerprint": bundle.transcript.source_fingerprint_value,
        "analysis_json": str(bundle.analysis_path),
        "analysis_sha256": bundle.analysis_sha256,
        "transcript_json": str(bundle.transcript.path),
        "transcript_sha256": bundle.transcript.sha256,
    }
    if any(state.get(key) != value for key, value in expected.items()):
        raise ReviewWorkflowError("ReviewSourcesStale")
    return state


def _validated_candidate(
    bundle: ReviewBundle,
    state: dict[str, Any],
    edited_item: tuple[str, dict[str, Any]] | None = None,
) -> MeetingAnalysis:
    payload = deepcopy(bundle.analysis_payload)
    for item_id, record in state["items"].items():
        section = record["section"]
        index = record["index"]
        value = payload.get(section)
        replacement = edited_item[1] if edited_item and edited_item[0] == item_id else record.get("edited")
        if replacement is None:
            continue
        if section == "overview":
            payload[section] = replacement
        elif isinstance(value, list) and 1 <= index <= len(value):
            value[index - 1] = replacement
        else:
            raise ReviewWorkflowError("ReviewItemLocationInvalid")
    analysis = GPTMeetingAnalyzer._parse_and_validate(
        payload,
        bundle.transcript.segments,
    )
    if analysis.status == "validated":
        return analysis
    if (
        analysis.status == "review_required"
        and analysis.review_reason == "SemanticReviewRequired"
    ):
        return analysis
    raise ReviewWorkflowError(
        analysis.review_reason or "EvidenceRevalidationFailed"
    )


def list_review_items(bundle: ReviewBundle, state: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    state_items = state.get("items", {}) if state else {}
    result = []
    for section, index, original in _iter_analysis_items(bundle.analysis_payload):
        item_id = _item_key(section, index)
        record = state_items.get(item_id, {})
        item = record.get("edited") or original
        result.append(
            {
                "item_id": item_id,
                "category": section,
                "text": item.get("text", ""),
                "owner": item.get("owner"),
                "deadline": item.get("deadline"),
                "evidence": _item_sources(item, bundle.transcript.segments),
                "decision": record.get("decision", "not_started"),
                "numeric_code_validation": "passed",
            }
        )
    return result


def record_review_decision(
    bundle: ReviewBundle,
    *,
    reviewer: str,
    item_id: str,
    decision: str,
    confirmed: bool,
    reason: str | None = None,
    edits: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not confirmed:
        raise ReviewWorkflowError("ExplicitHumanConfirmationRequired")
    state = _load_or_start_review(bundle, reviewer, start_if_missing=True)
    record = state["items"].get(item_id)
    if not isinstance(record, dict):
        raise ReviewWorkflowError("ReviewItemNotFound")
    if decision == "edit":
        if not isinstance(edits, dict):
            raise ReviewWorkflowError("ReviewEditRequired")
        candidate = deepcopy(record.get("edited") or record["original"])
        for key in ("text", "owner", "deadline"):
            if key in edits:
                candidate[key] = edits[key]
        if not isinstance(candidate.get("text"), str) or not candidate["text"].strip():
            raise ReviewWorkflowError("EditedSummaryTextRequired")
        _validated_candidate(bundle, state, (item_id, candidate))
        before = {
            key: (record.get("edited") or record["original"]).get(key)
            for key in ("text", "owner", "deadline")
        }
        after = {key: candidate.get(key) for key in ("text", "owner", "deadline")}
        record["edited"] = candidate
        record["decision"] = "pending"
        record["reviewer"] = reviewer.strip()
        record["reviewed_at"] = datetime.now(timezone.utc).isoformat()
        record["rejection_reason"] = None
        _append_audit(
            state,
            reviewer,
            "edit_revalidated",
            item_id,
            {"before": before, "after": after, "decision_reset": True},
        )
    elif decision in {"approve", "reject"}:
        if decision == "reject" and not (reason and reason.strip()):
            raise ReviewWorkflowError("RejectionReasonRequired")
        _validated_candidate(bundle, state)
        record["decision"] = "approved" if decision == "approve" else "rejected"
        record["reviewer"] = reviewer.strip()
        record["reviewed_at"] = datetime.now(timezone.utc).isoformat()
        record["rejection_reason"] = reason.strip() if decision == "reject" else None
        _append_audit(
            state,
            reviewer,
            decision,
            item_id,
            {"reason": record["rejection_reason"]},
        )
    else:
        raise ReviewWorkflowError("ReviewDecisionInvalid")
    decisions = [item.get("decision") for item in state["items"].values()]
    approved_sections = {
    record["section"]
    for record in state["items"].values()
    if record.get("decision") == "approved"
    }
    state["status"] = (
        "review_approved"
        if decisions
        and all(value != "pending" for value in decisions)
        and {"overview", "topics", "key_issues"}.issubset(approved_sections)
        else "review_pending"
    )
    _write_review_state(bundle.review_state_path, state, create=False)
    return state


def _approved_analysis(
    bundle: ReviewBundle,
    state: dict[str, Any],
) -> MeetingAnalysis:
    _verify_audit(state)
    _load_or_start_review(bundle, "approval-check", start_if_missing=False)
    if any(record.get("decision") == "pending" for record in state["items"].values()):
        raise ReviewWorkflowError("AllItemsMustBeReviewed")
    if state.get("status") not in {"review_approved", "approved"}:
        raise ReviewWorkflowError("AllRequiredItemsMustBeApproved")
    approved_items: dict[str, list[AnalysisItem]] = {section: [] for section in _SECTIONS}
    for item_id, record in state["items"].items():
        if record.get("decision") != "approved":
            continue
        section = record["section"]
        value = record.get("edited") or record["original"]
        evidence = _item_sources(value, bundle.transcript.segments)
        refs = value["evidence"]
        evidence_refs = []
        for reference, quote in zip(refs, evidence):
            evidence_refs.append(
                # The resolver's validated source text is preserved for PDF trace links.
                _evidence_reference(reference, quote)
            )
        approved_items[section].append(
            AnalysisItem(
                value["text"].strip(),
                tuple(evidence_refs),
                value.get("owner"),
                value.get("deadline"),
            )
        )
    if len(approved_items["overview"]) != 1:
        raise ReviewWorkflowError("ApprovedOverviewRequired")
    if not approved_items["topics"] or not approved_items["key_issues"]:
        raise ReviewWorkflowError("ApprovedSubstantiveSectionsRequired")
    return MeetingAnalysis(
        overview=approved_items["overview"][0],
        classifications=tuple(approved_items["classifications"]),
        topics=tuple(approved_items["topics"]),
        key_issues=tuple(approved_items["key_issues"]),
        actions=tuple(approved_items["actions"]),
        dependencies=tuple(approved_items["dependencies"]),
        schedules=tuple(approved_items["schedules"]),
        risks=tuple(approved_items["risks"]),
        closing_activities=tuple(approved_items["closing_activities"]),
        status="validated",
    )


def _evidence_reference(reference: dict[str, Any], quote: str):
    from app.meeting_analysis import EvidenceReference

    if "start_char" in reference:
        return EvidenceReference(
            reference["segment_id"],
            quote,
            reference["start_char"],
            reference["end_char"],
        )
    return EvidenceReference(reference["segment_id"], quote)


def generate_approved_pdf(
    bundle: ReviewBundle,
    state: dict[str, Any],
    output_dir: Path,
    *,
    confirmed: bool = False,
) -> Path:
    if not confirmed:
        raise ReviewWorkflowError("ExplicitHumanConfirmationRequired")
    output_dir = output_dir.resolve()
    preview_root = DEFAULT_PREVIEW_ROOT.resolve()
    try:
        relative = output_dir.relative_to(preview_root)
    except ValueError as error:
        raise ReviewWorkflowError("ApprovedPdfOutsidePreviewRoot") from error
    if not relative.parts:
        raise ReviewWorkflowError("ApprovedPdfRequiresNewPreviewDirectory")
    if output_dir.exists():
        raise ReviewWorkflowError("ApprovedPdfDirectoryAlreadyExists")
    current_bundle = load_review_bundle(
        bundle.analysis_path,
        bundle.transcript.path,
        bundle.source,
        bundle.review_state_path.parent,
    )
    if (
        current_bundle.analysis_sha256 != bundle.analysis_sha256
        or current_bundle.transcript.sha256 != bundle.transcript.sha256
    ):
        raise ReviewWorkflowError("ReviewSourcesStale")
    current_state = _load_or_start_review(
        current_bundle,
        "approval-check",
        start_if_missing=False,
    )
    if current_state != state:
        state = current_state
    analysis = _approved_analysis(current_bundle, state)
    output_dir.mkdir(parents=True, exist_ok=False)
    pdf_path = output_dir / f"{bundle.source.stem}_SARA_Approved.pdf"
    write_meeting_summary_pdf(
        pdf_path,
        bundle.source,
        bundle.source.parent.parent.name,
        analysis,
    )
    state["approved_pdf"] = {
        "path": str(pdf_path),
        "sha256": _sha256(pdf_path.read_bytes()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    state["status"] = "approved"
    _append_audit(
        state,
        "approval-check",
        "approved_pdf_generated",
        None,
        {"pdf_path": str(pdf_path), "sha256": state["approved_pdf"]["sha256"]},
    )
    _write_review_state(bundle.review_state_path, state, create=False)
    return pdf_path


def _safe_join(root: Path, relative: str) -> Path:
    pure = PureWindowsPath(relative)
    if not pure.parts or pure.is_absolute() or any(part in {"..", ""} for part in pure.parts):
        raise ReviewWorkflowError("PairPathInvalid")
    candidate = root.joinpath(*pure.parts).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ReviewWorkflowError("PairPathOutsideRoot") from error
    return candidate


def _artifact_indexes(preview_root: Path) -> tuple[list[Path], list[Path], list[Path]]:
    if not preview_root.is_dir():
        return [], [], []
    transcript_jsons = sorted(preview_root.rglob("*.phase9.json"))
    analysis_jsons = sorted(preview_root.rglob("*.phase10.json"))
    review_jsons = sorted(preview_root.rglob("*.review.json"))
    return transcript_jsons, analysis_jsons, review_jsons


def _load_transcript_candidates(
    paths: list[Path],
    source: Path,
) -> tuple[list[ValidatedTranscript], bool]:
    validated: list[ValidatedTranscript] = []
    has_raw_candidate = False
    for path in paths:
        try:
            raw, _bytes = _read_json(path, error_reason="TranscriptJsonInvalid")
        except ReviewWorkflowError:
            logger.warning("Ignoring malformed local transcript artifact; filename=%s", path.name)
            continue
        if raw.get("source_fingerprint") != source_fingerprint(source):
            continue
        if raw.get("raw_transcript"):
            has_raw_candidate = True
        try:
            validated.append(load_validated_transcript(path, source))
        except ReviewWorkflowError as error:
            logger.info(
                "Transcript artifact is not validated; filename=%s reason=%s",
                path.name,
                error.reason,
            )
    unique: dict[str, ValidatedTranscript] = {}
    for transcript in validated:
        unique.setdefault(transcript.sha256, transcript)
    return list(unique.values()), has_raw_candidate


def _candidate_analysis(
    transcript: ValidatedTranscript,
    phase10_paths: list[Path],
    source: Path,
) -> tuple[Path | None, dict[str, Any] | None, str]:
    transcript_payload = transcript.transcript_payload
    original_analysis = transcript_payload.get("meeting_analysis")
    best: tuple[Path, dict[str, Any], str] | None = None
    if isinstance(original_analysis, dict) and _analysis_item_count(original_analysis):
        try:
            parsed = _parse_analysis(original_analysis, transcript)
            quality = "passed" if parsed.status == "validated" else "semantic_review_pending"
        except ReviewWorkflowError:
            quality = "failed"
        best = (transcript.path, original_analysis, quality)

    for path in phase10_paths:
        try:
            artifact, raw = _read_json(path, error_reason="AnalysisJsonInvalid")
        except ReviewWorkflowError:
            continue
        if (
            artifact.get("source_filename") != source.name
            or artifact.get("source_fingerprint")
            != transcript.source_fingerprint_value
        ):
            continue
        analysis_payload = artifact.get("meeting_analysis")
        if not isinstance(analysis_payload, dict):
            continue
        linked_path = artifact.get("phase9_transcript_json")
        linked_hash = artifact.get("phase9_json_sha256")
        linked = (
            isinstance(linked_path, str)
            and Path(linked_path).resolve() == transcript.path
            and linked_hash == transcript.sha256
        )
        if not linked or not _analysis_item_count(analysis_payload):
            quality = "failed"
        else:
            try:
                parsed = _parse_analysis(analysis_payload, transcript)
                quality = (
                    "passed"
                    if parsed.status == "validated"
                    else "semantic_review_pending"
                )
            except ReviewWorkflowError:
                quality = "failed"
        best = (path, analysis_payload, quality)
    if best is None:
        return None, None, "missing"
    return best


def _approved_state(
    source: Path,
    review_paths: list[Path],
    fingerprint: str,
    review_root: Path,
) -> tuple[bool, bool]:
    state_path = _review_state_path(source, review_root)
    if not state_path.is_file():
        for candidate in review_paths:
            try:
                state, _raw = _read_json(candidate, error_reason="ReviewStateInvalid")
            except ReviewWorkflowError:
                continue
            if state.get("source") == str(source.resolve()):
                state_path = candidate
                break
    if not state_path.is_file():
        return False, False
    try:
        state, _raw = _read_json(state_path, error_reason="ReviewStateInvalid")
        _verify_audit(state)
    except ReviewWorkflowError:
        return False, False
    if (
        state.get("source_fingerprint") != fingerprint
        or state.get("source") != str(source.resolve())
    ):
        return False, False
    pdf_record = state.get("approved_pdf")
    if not isinstance(pdf_record, dict):
        return state.get("status") == "review_approved", False
    pdf_path = Path(pdf_record.get("path", ""))
    if not pdf_path.is_file():
        return state.get("status") == "approved", False
    try:
        pdf_valid = _sha256(pdf_path.read_bytes()) == pdf_record.get("sha256")
    except OSError:
        pdf_valid = False
    return (
        state.get("status") in {"review_approved", "approved"},
        pdf_valid and state.get("status") == "approved",
    )


def evaluate_batch(
    *,
    recording_root: Path = DEFAULT_RECORDING_ROOT,
    sara_root: Path = DEFAULT_SARA_ROOT,
    pairs_csv: Path = DEFAULT_ANALYSIS_DIR / "recording_pdf_pairs.csv",
    preview_root: Path = DEFAULT_PREVIEW_ROOT,
    review_root: Path = DEFAULT_REVIEW_ROOT,
    output_dir: Path = DEFAULT_ANALYSIS_DIR,
) -> BatchEvaluation:
    if not recording_root.is_dir() or not sara_root.is_dir():
        raise ReviewWorkflowError("DatasetRootUnavailable")
    try:
        with pairs_csv.open(encoding="utf-8-sig", newline="") as source_file:
            pair_rows = list(csv.DictReader(source_file))
    except (OSError, csv.Error) as error:
        raise ReviewWorkflowError("PairInventoryUnavailable") from error

    transcript_paths, analysis_paths, review_paths = _artifact_indexes(preview_root)
    rows: list[dict[str, str]] = []
    for pair in pair_rows:
        if pair.get("status") != "MATCHED":
            continue
        recording_rel = pair.get("recording_relative_path", "")
        pdf_rel = pair.get("pdf_relative_path", "")
        recording = _safe_join(recording_root, recording_rel)
        reference_pdf = _safe_join(sara_root, pdf_rel)
        mp3_available = recording.is_file()
        sara_available = reference_pdf.is_file()
        transcripts: list[ValidatedTranscript] = []
        has_raw_transcript = False
        if mp3_available:
            matching_artifacts = [
                path for path in transcript_paths
                if path.stem.casefold() == f"{recording.stem}.phase9".casefold()
            ]
            transcripts, has_raw_transcript = _load_transcript_candidates(
                matching_artifacts, recording
            )
        transcript = transcripts[0] if len(transcripts) == 1 else None
        if not mp3_available:
            transcript_state = "source_missing"
        elif transcript is not None:
            transcript_state = "validated"
        elif has_raw_transcript:
            transcript_state = "cleanup_or_validation_required"
        else:
            transcript_state = "transcription_required"

        analysis_path: Path | None = None
        analysis_payload: dict[str, Any] | None = None
        evidence_status = "not_available"
        analysis_status = "missing"
        ready_for_review = False
        if transcript is not None:
            analysis_path, analysis_payload, evidence_status = _candidate_analysis(
                transcript,
                analysis_paths,
                recording,
            )
            if analysis_payload is None and isinstance(
                transcript.transcript_payload.get("meeting_analysis"), dict
            ) and _analysis_item_count(
                transcript.transcript_payload["meeting_analysis"]
            ):
                analysis_path = transcript.path
                analysis_payload = transcript.transcript_payload["meeting_analysis"]
                analysis_status = str(analysis_payload.get("status", "unknown"))
                evidence_status = "failed"
            elif analysis_payload is not None:
                analysis_status = str(analysis_payload.get("status", "unknown"))
                ready_for_review = (
                    evidence_status in {"passed", "semantic_review_pending"}
                    and _analysis_item_count(analysis_payload) > 0
                )
        approved, pdf_generated = (
            _approved_state(
                recording,
                review_paths,
                transcript.source_fingerprint_value,
                review_root,
            )
            if transcript is not None
            else (False, False)
        )
        if approved:
            ready_for_review = False
        if transcript is None:
            next_step = (
                "transcription_required"
                if transcript_state == "transcription_required"
                else "transcript_cleanup_or_validation"
            )
        elif analysis_payload is None or evidence_status == "failed":
            next_step = "meeting_analysis_required"
        elif ready_for_review:
            next_step = "human_review_required"
        elif approved and pdf_generated:
            next_step = "complete"
        else:
            next_step = "review_or_approval_required"
        rows.append(
            {
                "recording_relative_path": recording_rel,
                "sara_pdf_relative_path": pdf_rel,
                "matching_evidence": pair.get("matching_evidence", ""),
                "mp3_available": str(mp3_available).lower(),
                "sara_pdf_available": str(sara_available).lower(),
                "validated_transcript_available": str(transcript is not None).lower(),
                "transcript_status": transcript_state,
                "transcript_json": str(transcript.path) if transcript else "",
                "meeting_analysis_available": str(analysis_payload is not None).lower(),
                "analysis_status": analysis_status,
                "analysis_json": str(analysis_path) if analysis_path else "",
                "evidence_validation": evidence_status,
                "semantic_review_pending": str(
                    ready_for_review and evidence_status == "semantic_review_pending"
                ).lower(),
                "ready_for_human_review": str(ready_for_review).lower(),
                "approved": str(approved).lower(),
                "pdf_generated": str(pdf_generated).lower(),
                "next_step": next_step,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "phase10_6_batch_evaluation.csv"
    markdown_path = output_dir / "phase10_6_batch_evaluation.md"
    columns = list(rows[0]) if rows else [
        "recording_relative_path", "sara_pdf_relative_path", "matching_evidence",
        "mp3_available", "sara_pdf_available", "validated_transcript_available",
        "transcript_status", "transcript_json", "meeting_analysis_available",
        "analysis_status", "analysis_json", "evidence_validation",
        "semantic_review_pending", "ready_for_human_review", "approved",
        "pdf_generated", "next_step",
    ]
    _write_csv(csv_path, columns, rows)
    markdown = _build_report(rows, pair_rows)
    _write_text_atomic(markdown_path, markdown)
    return BatchEvaluation(csv_path, markdown_path, tuple(rows))


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _write_text_atomic(path: Path, text: str) -> None:
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


def _build_report(
    rows: list[dict[str, str]],
    pair_rows: list[dict[str, str]],
) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        key = row["next_step"]
        counts[key] = counts.get(key, 0) + 1
    matched_count = sum(row.get("status") == "MATCHED" for row in pair_rows)
    audio_only = sum(row.get("status") == "AUDIO_ONLY" for row in pair_rows)
    pdf_only = sum(row.get("status") == "PDF_ONLY" for row in pair_rows)
    lines = [
        "# Phase 10.6 Batch Evaluation",
        "",
        f"- Matched pairs evaluated: {len(rows)} (inventory reports {matched_count})",
        f"- Audio-only pairs in inventory: {audio_only}",
        f"- PDF-only pairs in inventory: {pdf_only}",
        "- SARA PDFs are references only; they do not establish transcript or approval status.",
        "",
        "## Processing status",
        "",
        "| Next step | Count |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {status} | {count} |" for status, count in sorted(counts.items())
    )
    ready = [row for row in rows if row["ready_for_human_review"] == "true"]
    lines.extend(["", "## Ready for human review", ""])
    lines.extend(
        f"- `{row['recording_relative_path']}` — {row['analysis_status']}"
        for row in ready
    )
    if not ready:
        lines.append("- None.")
    lines.extend(["", "## Requires transcription or analysis", ""])
    next_required = [
        row for row in rows
        if row["next_step"] in {
            "transcription_required",
            "transcript_cleanup_or_validation",
            "meeting_analysis_required",
        }
    ]
    lines.extend(
        f"- `{row['recording_relative_path']}` — {row['next_step']}"
        for row in next_required
    )
    if not next_required:
        lines.append("- None.")
    lines.extend(["", "## Approved comparison status", ""])
    approved = [row for row in rows if row["approved"] == "true" and row["pdf_generated"] == "true"]
    if not approved:
        lines.append("No approved MorningTalk summaries exist; SARA comparison is pending.")
    else:
        lines.append(
            "Approved summaries are available for comparison. Compare each only "
            "against its paired reference PDF; transcript evidence remains authoritative."
        )
    lines.append("")
    return "\n".join(lines)


def _load_review_state(bundle: ReviewBundle, reviewer: str) -> dict[str, Any]:
    return _load_or_start_review(bundle, reviewer, start_if_missing=False)


def render_review_items(
    bundle: ReviewBundle,
    state: dict[str, Any] | None = None,
) -> str:
    rendered = []
    state_items = state.get("items", {}) if state else {}
    for item in list_review_items(bundle, state):
        record = state_items.get(item["item_id"], {})
        rendered.extend(
            [
                f"[{item['item_id']}] category={item['category']}",
                f"Decision: {item['decision']}",
                f"Summary: {item['text']}",
                f"Numeric/code/date validation: {item['numeric_code_validation']}",
                f"Owner: {item['owner'] or '(none)'}",
                f"Deadline: {item['deadline'] or '(none)'}",
            ]
        )
        rendered.extend(
            f"Evidence segment {index}: {text}"
            for index, text in enumerate(item["evidence"], 1)
        )
        if record.get("rejection_reason"):
            rendered.append(f"Rejection reason: {record['rejection_reason']}")
        rendered.append("")
    return "\n".join(rendered)


def review_and_maybe_publish(
    *,
    analysis_path: Path,
    transcript_path: Path,
    source: Path,
    reviewer: str,
    review_dir: Path = DEFAULT_REVIEW_ROOT,
    item_id: str | None = None,
    decision: str | None = None,
    reason: str | None = None,
    edits: dict[str, Any] | None = None,
    confirmed: bool = False,
    generate_pdf: bool = False,
    output_dir: Path | None = None,
) -> tuple[str, Path | None]:
    bundle = load_review_bundle(
        analysis_path,
        transcript_path,
        source,
        review_dir,
    )
    state = _load_or_start_review(
        bundle,
        reviewer,
        start_if_missing=True,
    )
    if decision is not None:
        if item_id is None:
            raise ReviewWorkflowError("ReviewItemIdRequired")
        state = record_review_decision(
            bundle,
            reviewer=reviewer,
            item_id=item_id,
            decision=decision,
            confirmed=confirmed,
            reason=reason,
            edits=edits,
        )
    report = render_review_items(bundle, state)
    pdf_path = None
    if generate_pdf:
        if not confirmed:
            raise ReviewWorkflowError("ExplicitHumanConfirmationRequired")
        if output_dir is None:
            raise ReviewWorkflowError("ApprovedPdfOutputDirectoryRequired")
        pdf_path = generate_approved_pdf(
            bundle,
            state,
            output_dir,
            confirmed=confirmed,
        )
        state = _load_review_state(bundle, reviewer)
        report += f"\nApproved PDF: {pdf_path}\n"
    return report, pdf_path
