"""Database synchronization and opt-in processing for recursive audio scans."""

import logging
from pathlib import Path
import uuid

from app.config import AzureConfig
from app.database import SqlRepository, retry_delay_seconds
from app.file_manager import source_fingerprint, source_processing_lock
from app.pdf_export import write_pdf
from app.scanner import ScannedAudio
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
) -> str:
    if not item.stable:
        return "unstable"
    if item.pdf_exists:
        return "review_required"

    lease_token = uuid.uuid4()
    attempt = repository.claim_file(audio_file_id, lease_token)
    if attempt is None:
        return "not_claimed"

    try:
        with source_processing_lock(item.path):
            if item.pdf_path.exists():
                if not repository.complete_file(
                    audio_file_id, lease_token, item.pdf_path, review_required=True
                ):
                    raise RuntimeError("Database claim was lost before recording existing PDF.")
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

            transcriber = AzureOpenAITranscriber(config, chunk_callback=save_chunk)
            transcript = transcriber.transcribe(item.path)
            if not transcript.strip():
                raise ValueError("Transcription returned no text.")
            if source_fingerprint(item.path) != fingerprint:
                raise RuntimeError("Source MP3 changed during processing.")

            plant = item.path.relative_to(input_dir).parts[0]
            write_pdf(item.pdf_path, item.path, plant, transcript)
            duration = AzureOpenAITranscriber._probe_duration(item.path)
            low_quality = len(transcript) / max(duration / 60, 1 / 60) < (
                config.quality_warning_chars_per_minute
            )
            if not repository.complete_file(
                audio_file_id, lease_token, item.pdf_path, review_required=low_quality
            ):
                raise RuntimeError("Database processing claim was lost before completion.")
        final_status = "review_required" if low_quality else "completed"
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
