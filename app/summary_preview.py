"""Offline generation of summaries from fully validated Phase 9 evidence."""

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from app.file_manager import source_fingerprint
from app.meeting_analysis import EvidenceReference, GPTMeetingAnalyzer, MeetingAnalysis
from app.pdf_export import write_meeting_summary_pdf
from app.transcript import TranscriptSegment, validate_segment_times


class SummaryPreviewNotReady(ValueError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def write_sara_summary_preview(
    phase9_json_path: Path,
    source: Path,
    output_dir: Path,
) -> Path:
    try:
        payload = json.loads(phase9_json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SummaryPreviewNotReady("Phase9JsonUnavailableOrInvalid") from error
    if not isinstance(payload, dict):
        raise SummaryPreviewNotReady("Phase9JsonInvalid")
    if payload.get("cleanup_status") != "completed":
        raise SummaryPreviewNotReady("TranscriptCleanupNotValidated")

    raw_transcript = payload.get("raw_transcript")
    cleaned_transcript = payload.get("cleaned_transcript")
    raw_segments = payload.get("segments")
    if (
        not isinstance(raw_transcript, str)
        or not isinstance(cleaned_transcript, str)
        or not isinstance(raw_segments, list)
        or not raw_segments
    ):
        raise SummaryPreviewNotReady("TranscriptDataMissing")

    segments: list[TranscriptSegment] = []
    for item in raw_segments:
        if not isinstance(item, dict):
            raise SummaryPreviewNotReady("TranscriptSegmentInvalid")
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
            raise SummaryPreviewNotReady("TranscriptSegmentInvalid")
        segments.append(
            TranscriptSegment(start, end, speaker_id, raw_text, cleaned_text)
        )

    transcript_segments = tuple(segments)
    try:
        validate_segment_times(transcript_segments)
    except ValueError as error:
        raise SummaryPreviewNotReady("TranscriptSegmentsInvalid") from error
    if "\n".join(segment.raw_text for segment in transcript_segments) != raw_transcript:
        raise SummaryPreviewNotReady("RawTranscriptInconsistent")
    if "\n".join(segment.display_text for segment in transcript_segments) != cleaned_transcript:
        raise SummaryPreviewNotReady("CleanedTranscriptInconsistent")

    analysis_payload: Any = payload.get("meeting_analysis")
    if not isinstance(analysis_payload, dict):
        raise SummaryPreviewNotReady("MeetingAnalysisMissing")
    if analysis_payload.get("status") != "validated":
        reason = analysis_payload.get("review_reason")
        raise SummaryPreviewNotReady(
            reason if isinstance(reason, str) and reason else "MeetingAnalysisNotValidated"
        )

    analysis = GPTMeetingAnalyzer._parse_and_validate(
        analysis_payload, transcript_segments
    )
    if analysis.status != "validated":
        raise SummaryPreviewNotReady(
            analysis.review_reason or "MeetingAnalysisEvidenceInvalid"
        )
    if (
        analysis.overview is None
        or not analysis.classifications
        or not analysis.key_issues
    ):
        raise SummaryPreviewNotReady("InsufficientStructuredSummary")
    analysis = replace(
        analysis,
        input_tokens=_nonnegative_int(analysis_payload.get("input_tokens")),
        output_tokens=_nonnegative_int(analysis_payload.get("output_tokens")),
    )

    source = source.resolve()
    if not source.is_file():
        raise SummaryPreviewNotReady("SourceRecordingUnavailable")
    expected_fingerprint = payload.get("source_fingerprint")
    if (
        not isinstance(expected_fingerprint, str)
        or not expected_fingerprint
        or source_fingerprint(source).casefold() != expected_fingerprint.casefold()
    ):
        raise SummaryPreviewNotReady("SourceFingerprintMismatch")

    output_path = output_dir / f"{source.stem}_SARA_Preview.pdf"
    if output_path.exists():
        raise FileExistsError("Summary preview already exists; choose a new output directory.")
    write_meeting_summary_pdf(
        output_path,
        source,
        source.parent.parent.name,
        analysis,
    )
    return output_path


def write_sara_layout_demo(
    phase9_json_path: Path,
    source: Path,
    output_dir: Path,
) -> Path:
    try:
        payload = json.loads(phase9_json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SummaryPreviewNotReady("Phase9JsonUnavailableOrInvalid") from error
    if not isinstance(payload, dict) or payload.get("cleanup_status") != "completed":
        raise SummaryPreviewNotReady("TranscriptCleanupNotValidated")
    cleaned = payload.get("cleaned_transcript")
    raw_segments = payload.get("segments")
    if not isinstance(cleaned, str) or not cleaned or not isinstance(raw_segments, list):
        raise SummaryPreviewNotReady("TranscriptDataMissing")

    segments = []
    for item in raw_segments:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("raw_text"), str)
            or not isinstance(item.get("cleaned_text"), str)
            or item.get("review_required") is not False
        ):
            raise SummaryPreviewNotReady("TranscriptSegmentInvalid")
        segments.append(TranscriptSegment(
            item.get("start_seconds"),
            item.get("end_seconds"),
            item.get("speaker_id"),
            item["raw_text"],
            item["cleaned_text"],
        ))
    transcript_segments = tuple(segments)
    try:
        validate_segment_times(transcript_segments)
    except ValueError as error:
        raise SummaryPreviewNotReady("TranscriptSegmentsInvalid") from error
    if "\n".join(segment.raw_text for segment in transcript_segments) != payload.get("raw_transcript"):
        raise SummaryPreviewNotReady("RawTranscriptInconsistent")
    if "\n".join(segment.display_text for segment in transcript_segments) != cleaned:
        raise SummaryPreviewNotReady("CleanedTranscriptInconsistent")

    source = source.resolve()
    if not source.is_file():
        raise SummaryPreviewNotReady("SourceRecordingUnavailable")
    expected_fingerprint = payload.get("source_fingerprint")
    if (
        not isinstance(expected_fingerprint, str)
        or not expected_fingerprint
        or source_fingerprint(source).casefold() != expected_fingerprint.casefold()
    ):
        raise SummaryPreviewNotReady("SourceFingerprintMismatch")

    text = transcript_segments[0].display_text if len(transcript_segments) == 1 else ""
    if not text:
        raise SummaryPreviewNotReady("LayoutDemoRequiresSingleSourceSegment")
    passages = _select_layout_demo_passages(text, transcript_segments[0].display_text)
    if len(passages) < 2:
        raise SummaryPreviewNotReady("InsufficientTranscriptForLayoutDemo")

    output_path = output_dir / f"{source.stem}_SARA_Preview.pdf"
    if output_path.exists():
        raise FileExistsError("Summary preview already exists; choose a new output directory.")
    write_meeting_summary_pdf(
        output_path,
        source,
        source.parent.parent.name,
        MeetingAnalysis.for_review("LayoutDemonstrationNotValidated"),
        fixture_preview=True,
        demo_passages=passages,
    )
    return output_path


def _select_layout_demo_passages(
    transcript: str,
    source_segment: str,
) -> tuple[EvidenceReference, ...]:
    passages = []
    seen = set()
    for ratio in (0.10, 0.35, 0.60, 0.85):
        center = int(len(transcript) * ratio)
        start = max(0, center - 85)
        end = min(len(transcript), center + 85)
        if start:
            boundary = transcript.rfind(" ", 0, start)
            if boundary >= start - 60:
                start = boundary + 1
        if end < len(transcript):
            boundary = transcript.find(" ", end)
            if boundary != -1 and boundary <= end + 60:
                end = boundary
        quote = transcript[start:end].strip()
        if quote and quote not in seen and quote in source_segment:
            seen.add(quote)
            passages.append(EvidenceReference(1, quote))
    return tuple(passages)


def _nonnegative_int(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None
