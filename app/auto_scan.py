"""Database synchronization and opt-in processing for recursive audio scans."""

import logging
import os
from pathlib import Path
import uuid

from app.azure_speech import AzureSpeechDiarizer, DiarizationUnavailable
from app.config import AzureConfig, get_azure_openai_api_version
from app.database import SqlRepository, retry_delay_seconds
from app.file_manager import source_fingerprint, source_processing_lock
from app.meeting_analysis import GPTMeetingAnalyzer, MeetingAnalysis, MeetingAnalysisError
from app.meeting_outputs import meeting_pdf_paths, write_meeting_pdfs
from app.scanner import ScannedAudio
from app.transcript import TranscriptDocument, TranscriptSegment, mark_for_review, validate_segment_times
from app.transcript_cleanup import TranscriptCleanup
from app.transcriber import AzureOpenAITranscriber, ChunkProgress

logger = logging.getLogger(__name__)


def synchronize_scan(items: list[ScannedAudio], repository: SqlRepository) -> list[tuple[ScannedAudio, int | None, str]]:
    results = []
    for item in items:
        if not item.stable:
            results.append((item, None, "unstable"))
            continue
        audio_file_id, status = repository.upsert_scan(
            item.path,
            item.fingerprint,
            item.size_bytes,
            item.modified_utc,
            item.pdf_path,
            item.pdf_exists,
        )
        repository.log_event(audio_file_id, "scan", status)
        results.append((item, audio_file_id, status))
    return results


def process_scanned_file(
    item: ScannedAudio,
    input_dir: Path,
    audio_file_id: int,
    repository: SqlRepository,
    config: AzureConfig,
    *,
    allow_review_required: bool = False,
) -> str:
    if not item.stable:
        return "unstable"
    summary_path, full_path = meeting_pdf_paths(item.path)
    if summary_path.exists() and full_path.exists():
        return "review_required"
    if not repository.phase9_schema_ready():
        raise RuntimeError(
            "Phase 9 MSSQL migrations 002 and 003 must be applied before Azure processing."
        )
    text_api_version = get_azure_openai_api_version()

    lease_token = uuid.uuid4()
    attempt = repository.claim_file(
        audio_file_id,
        lease_token,
        lease_seconds=7200,
        allow_review_required=allow_review_required,
    )
    if attempt is None:
        return "not_claimed"

    try:
        with source_processing_lock(item.path):
            if summary_path.exists() and full_path.exists():
                return "review_required"

            fingerprint = source_fingerprint(item.path)

            def save_chunk(progress: ChunkProgress) -> None:
                if not repository.renew_lease(audio_file_id, lease_token):
                    raise RuntimeError("Database processing lease expired.")
                repository.record_chunk(
                    audio_file_id,
                    progress.index - 1,
                    progress.start_seconds,
                    progress.end_seconds,
                    progress.status,
                    transcript=progress.transcript,
                    error_class=progress.error_class,
                )

            speech_endpoint = os.getenv("AZURE_SPEECH_ENDPOINT", "").strip()
            speech_key = os.getenv("AZURE_SPEECH_KEY", "").strip()
            speech_api_version = os.getenv("AZURE_SPEECH_API_VERSION", "2025-10-15").strip()
            diarization_status = "available"
            try:
                if not speech_endpoint or not speech_key:
                    raise DiarizationUnavailable("Azure Speech is not configured.")
                segments = AzureSpeechDiarizer(
                    speech_endpoint,
                    speech_key,
                    api_version=speech_api_version,
                ).transcribe(item.path)
                validate_segment_times(segments)
            except (DiarizationUnavailable, ValueError) as error:
                logger.warning(
                    "Diarization unavailable; using unlabeled transcription; error_class=%s",
                    type(error).__name__,
                )
                diarization_status = "unavailable"
                transcript_text = AzureOpenAITranscriber(
                    config, chunk_callback=save_chunk
                ).transcribe(item.path)
                segments = (TranscriptSegment(None, None, None, transcript_text),)
            if not any(segment.raw_text.strip() for segment in segments):
                raise ValueError("Transcription returned no text.")
            if source_fingerprint(item.path) != fingerprint:
                raise RuntimeError("Source MP3 changed during processing.")

            try:
                cleaned = TranscriptCleanup(
                    config,
                    api_version=text_api_version,
                ).clean(segments)
                transcript = TranscriptDocument(
                    cleaned.segments,
                    diarization_status,
                    cleaned.cleanup_status,
                    cleaned.review_reason,
                    cleaned.input_tokens,
                    cleaned.output_tokens,
                )
            except (ValueError, RuntimeError) as error:
                logger.error("Transcript cleanup unavailable; error_class=%s", type(error).__name__)
                transcript = mark_for_review(
                    segments,
                    "CleanupUnavailable",
                    diarization_status=diarization_status,
                )

            if transcript.cleanup_status != "completed":
                analysis = MeetingAnalysis.for_review("TranscriptCleanupRequiresReview")
            else:
                try:
                    analysis = GPTMeetingAnalyzer(
                        config,
                        api_version=text_api_version,
                        terminology=os.getenv("MORNINGTALK_TERMINOLOGY", ""),
                    ).analyze(transcript.segments)
                except MeetingAnalysisError as error:
                    logger.error("Meeting analysis unavailable; error_class=%s", type(error).__name__)
                    analysis = MeetingAnalysis.for_review(type(error).__name__)

            if source_fingerprint(item.path) != fingerprint:
                raise RuntimeError("Source MP3 changed during transcript analysis.")
            plant = item.path.relative_to(input_dir).parts[0]
            outputs = write_meeting_pdfs(
                item.path,
                plant,
                transcript,
                analysis,
            )
            repository.save_phase9_outputs(
                audio_file_id,
                transcript,
                analysis,
                outputs,
            )
            duration = AzureOpenAITranscriber._probe_duration(item.path)
            low_quality = len(transcript.cleaned_transcript) / max(duration / 60, 1 / 60) < (
                config.quality_warning_chars_per_minute
            )
            outputs_validated = all(
                result.status == "created"
                for result in (outputs.summary, outputs.full_transcript)
            )
            needs_review = (
                low_quality
                or transcript.cleanup_status != "completed"
                or analysis.status != "validated"
                or not outputs_validated
            )
            if not repository.complete_file(
                audio_file_id,
                lease_token,
                outputs.summary.path,
                review_required=needs_review,
            ):
                raise RuntimeError("Database processing claim was lost before completion.")
        final_status = "review_required" if needs_review else "completed"
        repository.log_event(audio_file_id, "processing", final_status)
        return final_status
    except Exception as error:
        repository.fail_file(
            audio_file_id,
            lease_token,
            type(error).__name__,
            retry_delay_seconds=retry_delay_seconds(attempt),
        )
        repository.log_event(
            audio_file_id,
            "processing",
            "failed",
            error_class=type(error).__name__,
        )
        raise
