"""MSSQL connectivity and persistence for scanned audio files."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import uuid

from app.config import SqlConfig
from app.meeting_analysis import MeetingAnalysis
from app.meeting_outputs import MeetingPdfOutputs
from app.transcript import TranscriptDocument


def _odbc_value(value: str) -> str:
    return "{" + value.replace("}", "}}") + "}"


def connect_sql(config: SqlConfig, connector: Any = None) -> Any:
    if connector is None:
        import pyodbc

        connector = pyodbc.connect
        drivers = set(pyodbc.drivers())
    else:
        drivers = None
    config.validate(drivers)
    server = config.server
    port = os.getenv("SQL_EXPECTED_TCP_PORT", "").strip()
    if port and "," not in server and "\\" not in server:
        server = f"{server},{port}"
    connection_string = ";".join((
        f"DRIVER={_odbc_value(config.driver)}",
        f"SERVER={_odbc_value(server)}",
        f"DATABASE={_odbc_value(config.database)}",
        f"UID={_odbc_value(config.username)}",
        f"PWD={_odbc_value(config.password)}",
        f"Encrypt={'yes' if config.encrypt else 'no'}",
        f"TrustServerCertificate={'yes' if config.trust_server_certificate else 'no'}",
        f"Connection Timeout={config.connection_timeout}",
    )) + ";"
    connection = connector(connection_string, timeout=config.connection_timeout, autocommit=False)
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT DB_NAME(), SUSER_SNAME()")
        database, login = cursor.fetchone()
        if config.expected_database and database.casefold() != config.expected_database.casefold():
            raise RuntimeError("Connected SQL database does not match SQL_EXPECTED_DATABASE.")
        if config.expected_login and login.casefold() != config.expected_login.casefold():
            raise RuntimeError("Connected SQL login does not match SQL_EXPECTED_LOGIN.")
    except Exception:
        connection.close()
        raise
    return connection


def source_path_hash(source_path: str | Path) -> bytes:
    normalized = os.path.normcase(str(Path(source_path).resolve()))
    return hashlib.sha256(normalized.encode("utf-8")).digest()


def _next_result_row(cursor: Any) -> Any:
    while cursor.description is None:
        if not cursor.nextset():
            return None
    return cursor.fetchone()


@dataclass(frozen=True)
class AudioRecord:
    audio_file_id: int
    source_path: str
    fingerprint: str
    status: str
    pdf_path: str | None


@dataclass(frozen=True)
class SavedTranscriptRecord:
    audio_file_id: int
    source_path: str
    fingerprint: str
    status: str
    raw_transcript: str | None
    transcript_segments_json: str | None
    diarization_status: str | None
    input_tokens: int | None
    output_tokens: int | None


class SqlRepository:
    """Uses only DML and SELECT; schema creation is deliberately external."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def list_audio_files(self) -> list[AudioRecord]:
        cursor = self._connection.cursor()
        cursor.execute(
            "SELECT AudioFileId, SourcePath, SourceFingerprint, Status, PdfPath "
            "FROM dbo.AudioFile ORDER BY SourcePath"
        )
        return [AudioRecord(*row) for row in cursor.fetchall()]

    def load_saved_transcript(self, audio_file_id: int) -> SavedTranscriptRecord | None:
        cursor = self._connection.cursor()
        cursor.execute(
            "SELECT AudioFileId, SourcePath, SourceFingerprint, Status, RawTranscript, "
            "TranscriptSegmentsJson, DiarizationStatus, InputTokens, OutputTokens "
            "FROM dbo.AudioFile WHERE AudioFileId = ?",
            audio_file_id,
        )
        row = cursor.fetchone()
        return SavedTranscriptRecord(*row) if row is not None else None

    def upsert_scan(
        self,
        source_path: str | Path,
        fingerprint: str,
        size_bytes: int,
        modified_utc: datetime,
        pdf_path: str | Path,
        pdf_exists: bool,
    ) -> tuple[int, str]:
        path = str(Path(source_path).resolve())
        path_hash = source_path_hash(path)
        cursor = self._connection.cursor()
        cursor.execute("BEGIN TRANSACTION")
        try:
            cursor.execute(
                "SELECT AudioFileId, SourceFingerprint, Status FROM dbo.AudioFile "
                "WITH (UPDLOCK, HOLDLOCK) WHERE SourcePathHash = ?",
                path_hash,
            )
            existing = cursor.fetchone()
            if existing is None:
                status = "review_required" if pdf_exists else "discovered"
                cursor.execute(
                    "INSERT INTO dbo.AudioFile "
                    "(SourcePath, SourcePathHash, SourceFingerprint, SourceSizeBytes, "
                    "SourceModifiedUtc, Status, PdfPath) "
                    "OUTPUT inserted.AudioFileId VALUES (?, ?, ?, ?, ?, ?, ?)",
                    path, path_hash, fingerprint, size_bytes, modified_utc, status,
                    str(Path(pdf_path).resolve()) if pdf_exists else None,
                )
                audio_file_id = int(cursor.fetchone()[0])
            else:
                audio_file_id, old_fingerprint, old_status = existing
                audio_file_id = int(audio_file_id)
                status = old_status
                if old_fingerprint != fingerprint:
                    status = "review_required" if pdf_exists else "modified"
                    cursor.execute(
                        "UPDATE dbo.AudioFile SET SourcePath = ?, SourceFingerprint = ?, "
                        "SourceSizeBytes = ?, SourceModifiedUtc = ?, Status = ?, "
                        "AttemptCount = 0, LeaseToken = NULL, LeaseExpiresUtc = NULL, "
                        "NextRetryUtc = NULL, LastErrorClass = NULL, "
                        "PdfPath = ?, UpdatedUtc = SYSUTCDATETIME() WHERE AudioFileId = ?",
                        path, fingerprint, size_bytes, modified_utc, status,
                        str(Path(pdf_path).resolve()) if pdf_exists else None, audio_file_id,
                    )
                elif old_status == "completed" and not pdf_exists:
                    status = "review_required"
                    cursor.execute(
                        "UPDATE dbo.AudioFile SET Status = ?, UpdatedUtc = SYSUTCDATETIME() "
                        "WHERE AudioFileId = ?",
                        status, audio_file_id,
                    )
            self._connection.commit()
            return audio_file_id, status
        except Exception:
            self._connection.rollback()
            raise

    def claim_file(
        self,
        audio_file_id: int,
        lease_token: uuid.UUID,
        *,
        lease_seconds: int = 1800,
        max_attempts: int = 3,
        allow_review_required: bool = False,
    ) -> int | None:
        cursor = self._connection.cursor()
        cursor.execute(
            """
            UPDATE dbo.AudioFile WITH (UPDLOCK, ROWLOCK)
            SET Status = 'processing', LeaseToken = ?,
                LeaseExpiresUtc = DATEADD(second, ?, SYSUTCDATETIME()),
                AttemptCount = AttemptCount + 1, UpdatedUtc = SYSUTCDATETIME()
            OUTPUT inserted.AttemptCount
            WHERE AudioFileId = ? AND AttemptCount < ? AND (
                Status IN ('discovered', 'modified') OR
                (Status = 'failed' AND (NextRetryUtc IS NULL OR NextRetryUtc <= SYSUTCDATETIME())) OR
                (Status = 'processing' AND LeaseExpiresUtc <= SYSUTCDATETIME()) OR
                (? = 1 AND Status = 'review_required')
            )
            """,
            str(lease_token), lease_seconds, audio_file_id, max_attempts,
            1 if allow_review_required else 0,
        )
        row = cursor.fetchone()
        self._connection.commit()
        return int(row[0]) if row else None

    def renew_lease(
        self, audio_file_id: int, lease_token: uuid.UUID, *, lease_seconds: int = 3600
    ) -> bool:
        cursor = self._connection.cursor()
        cursor.execute(
            "UPDATE dbo.AudioFile SET LeaseExpiresUtc = DATEADD(second, ?, SYSUTCDATETIME()), "
            "UpdatedUtc = SYSUTCDATETIME() WHERE AudioFileId = ? AND LeaseToken = ? "
            "AND Status = 'processing'",
            lease_seconds, audio_file_id, str(lease_token),
        )
        renewed = cursor.rowcount == 1
        self._connection.commit()
        return renewed

    def complete_file(
        self,
        audio_file_id: int,
        lease_token: uuid.UUID,
        pdf_path: str | Path,
        *,
        review_required: bool = False,
    ) -> bool:
        cursor = self._connection.cursor()
        cursor.execute(
            "UPDATE dbo.AudioFile SET Status = ?, PdfPath = ?, LeaseToken = NULL, "
            "LeaseExpiresUtc = NULL, NextRetryUtc = NULL, LastErrorClass = NULL, "
            "UpdatedUtc = SYSUTCDATETIME() WHERE AudioFileId = ? AND LeaseToken = ? "
            "AND Status = 'processing'",
            "review_required" if review_required else "completed",
            str(Path(pdf_path).resolve()), audio_file_id, str(lease_token),
        )
        completed = cursor.rowcount == 1
        self._connection.commit()
        return completed

    def fail_file(
        self,
        audio_file_id: int,
        lease_token: uuid.UUID,
        error_class: str,
        *,
        retry_delay_seconds: int,
    ) -> bool:
        cursor = self._connection.cursor()
        cursor.execute(
            "UPDATE dbo.AudioFile SET Status = 'failed', LastErrorClass = ?, "
            "NextRetryUtc = DATEADD(second, ?, SYSUTCDATETIME()), LeaseToken = NULL, "
            "LeaseExpiresUtc = NULL, UpdatedUtc = SYSUTCDATETIME() "
            "WHERE AudioFileId = ? AND LeaseToken = ? AND Status = 'processing'",
            error_class[:128], retry_delay_seconds, audio_file_id, str(lease_token),
        )
        failed = cursor.rowcount == 1
        self._connection.commit()
        return failed

    def release_claim_for_review(
        self, audio_file_id: int, lease_token: uuid.UUID, error_class: str
    ) -> bool:
        cursor = self._connection.cursor()
        cursor.execute(
            "UPDATE dbo.AudioFile SET Status = 'review_required', LastErrorClass = ?, "
            "LeaseToken = NULL, LeaseExpiresUtc = NULL, NextRetryUtc = NULL, "
            "UpdatedUtc = SYSUTCDATETIME() WHERE AudioFileId = ? AND LeaseToken = ? "
            "AND Status = 'processing'",
            error_class[:128], audio_file_id, str(lease_token),
        )
        released = cursor.rowcount == 1
        self._connection.commit()
        return released

    def record_chunk(
        self,
        audio_file_id: int,
        chunk_index: int,
        start_seconds: float,
        end_seconds: float,
        status: str,
        *,
        transcript: str | None = None,
        error_class: str | None = None,
    ) -> None:
        character_count = len(transcript) if transcript is not None else 0
        cursor = self._connection.cursor()
        cursor.execute(
            """
            IF EXISTS (
                SELECT 1 FROM dbo.TranscriptionChunk
                WHERE AudioFileId = ? AND ChunkIndex = ?
            )
                UPDATE dbo.TranscriptionChunk
                SET StartSeconds = ?, EndSeconds = ?, Status = ?, Transcript = ?,
                    CharacterCount = ?, ErrorClass = ?, CompletedUtc = SYSUTCDATETIME()
                WHERE AudioFileId = ? AND ChunkIndex = ?
            ELSE
                INSERT INTO dbo.TranscriptionChunk
                    (AudioFileId, ChunkIndex, StartSeconds, EndSeconds, Status,
                     Transcript, CharacterCount, ErrorClass, StartedUtc, CompletedUtc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, SYSUTCDATETIME(), SYSUTCDATETIME())
            """,
            audio_file_id, chunk_index,
            start_seconds, end_seconds, status, transcript, character_count, error_class,
            audio_file_id, chunk_index,
            audio_file_id, chunk_index, start_seconds, end_seconds, status, transcript,
            character_count, error_class,
        )
        self._connection.commit()

    def log_event(
        self,
        audio_file_id: int | None,
        event_type: str,
        status: str,
        *,
        error_class: str | None = None,
        details: str | None = None,
    ) -> None:
        cursor = self._connection.cursor()
        cursor.execute(
            "INSERT INTO dbo.ProcessingLog "
            "(AudioFileId, EventType, Status, ErrorClass, Details) VALUES (?, ?, ?, ?, ?)",
            audio_file_id, event_type[:48], status[:24], error_class[:128] if error_class else None,
            details[:1000] if details else None,
        )
        self._connection.commit()

    def status_counts(self) -> dict[str, int]:
        cursor = self._connection.cursor()
        cursor.execute("SELECT Status, COUNT_BIG(*) FROM dbo.AudioFile GROUP BY Status")
        return {str(status): int(count) for status, count in cursor.fetchall()}

    def phase9_schema_ready(self) -> bool:
        cursor = self._connection.cursor()
        cursor.execute(
            "SELECT CASE WHEN "
            "COL_LENGTH('dbo.AudioFile', 'RawTranscript') IS NOT NULL AND "
            "COL_LENGTH('dbo.AudioFile', 'CleanedTranscript') IS NOT NULL AND "
            "COL_LENGTH('dbo.AudioFile', 'TranscriptSegmentsJson') IS NOT NULL AND "
            "COL_LENGTH('dbo.AudioFile', 'DiarizationStatus') IS NOT NULL AND "
            "COL_LENGTH('dbo.AudioFile', 'CleanupStatus') IS NOT NULL AND "
            "COL_LENGTH('dbo.AudioFile', 'ReviewReason') IS NOT NULL AND "
            "COL_LENGTH('dbo.AudioFile', 'InputTokens') IS NOT NULL AND "
            "COL_LENGTH('dbo.AudioFile', 'OutputTokens') IS NOT NULL AND "
            "COL_LENGTH('dbo.AudioFile', 'MeetingAnalysisJson') IS NOT NULL AND "
            "COL_LENGTH('dbo.AudioFile', 'MeetingAnalysisStatus') IS NOT NULL AND "
            "COL_LENGTH('dbo.AudioFile', 'SummaryPdfPath') IS NOT NULL AND "
            "COL_LENGTH('dbo.AudioFile', 'SummaryPdfStatus') IS NOT NULL AND "
            "COL_LENGTH('dbo.AudioFile', 'FullTranscriptPdfPath') IS NOT NULL AND "
            "COL_LENGTH('dbo.AudioFile', 'FullTranscriptPdfStatus') IS NOT NULL "
            "THEN 1 ELSE 0 END"
        )
        return bool(cursor.fetchone()[0])

    def save_phase9_transcript(
        self, audio_file_id: int, document: TranscriptDocument
    ) -> None:
        segments = json.dumps(
            [asdict(segment) for segment in document.segments],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        cursor = self._connection.cursor()
        cursor.execute(
            "UPDATE dbo.AudioFile SET RawTranscript = ?, CleanedTranscript = ?, "
            "TranscriptSegmentsJson = ?, DiarizationStatus = ?, CleanupStatus = ?, "
            "ReviewReason = ?, InputTokens = ?, OutputTokens = ?, "
            "UpdatedUtc = SYSUTCDATETIME() WHERE AudioFileId = ?",
            document.raw_transcript,
            document.cleaned_transcript,
            segments,
            document.diarization_status,
            document.cleanup_status,
            document.review_reason,
            document.input_tokens,
            document.output_tokens,
            audio_file_id,
        )
        if cursor.rowcount != 1:
            self._connection.rollback()
            raise RuntimeError("Unable to persist Phase 9 transcript for the selected audio record.")
        cursor.execute(
            "INSERT INTO dbo.ProcessingLog "
            "(AudioFileId, EventType, Status, ErrorClass, Details) "
            "VALUES (?, ?, ?, ?, ?)",
            audio_file_id,
            "phase9",
            document.cleanup_status,
            document.review_reason,
            f"segments={len(document.segments)}; "
            f"input_tokens={document.input_tokens}; output_tokens={document.output_tokens}",
        )
        self._connection.commit()

    def save_phase9_outputs(
        self,
        audio_file_id: int,
        transcript: TranscriptDocument,
        analysis: MeetingAnalysis,
        outputs: MeetingPdfOutputs,
    ) -> None:
        transcript_review_required = (
            transcript.cleanup_status != "completed"
            or any(segment.review_required for segment in transcript.segments)
        )
        summary_status = outputs.summary.status
        if summary_status != "failed" and (
            transcript_review_required or analysis.status != "validated"
        ):
            summary_status = "review_required"
        full_status = outputs.full_transcript.status
        if full_status != "failed" and transcript_review_required:
            full_status = "review_required"

        cursor = self._connection.cursor()
        cursor.execute(
            "UPDATE dbo.AudioFile SET RawTranscript = ?, CleanedTranscript = ?, "
            "TranscriptSegmentsJson = ?, DiarizationStatus = ?, CleanupStatus = ?, "
            "ReviewReason = ?, InputTokens = ?, OutputTokens = ?, "
            "MeetingAnalysisJson = ?, MeetingAnalysisStatus = ?, "
            "SummaryPdfPath = ?, SummaryPdfStatus = ?, "
            "FullTranscriptPdfPath = ?, FullTranscriptPdfStatus = ?, "
            "PdfPath = ?, UpdatedUtc = SYSUTCDATETIME() WHERE AudioFileId = ?",
            transcript.raw_transcript,
            transcript.cleaned_transcript,
            json.dumps(
                [asdict(segment) for segment in transcript.segments],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            transcript.diarization_status,
            transcript.cleanup_status,
            transcript.review_reason,
            transcript.input_tokens,
            transcript.output_tokens,
            json.dumps(analysis.to_dict(), ensure_ascii=False, separators=(",", ":")),
            analysis.status,
            str(outputs.summary.path),
            summary_status,
            str(outputs.full_transcript.path),
            full_status,
            str(outputs.summary.path),
            audio_file_id,
        )
        if cursor.rowcount != 1:
            self._connection.rollback()
            raise RuntimeError("Unable to persist Phase 9 outputs for the selected audio record.")
        for event_type, output, status in (
            ("summary_pdf", outputs.summary, summary_status),
            ("full_transcript_pdf", outputs.full_transcript, full_status),
        ):
            cursor.execute(
                "INSERT INTO dbo.ProcessingLog "
                "(AudioFileId, EventType, Status, ErrorClass, Details) "
                "VALUES (?, ?, ?, ?, ?)",
                audio_file_id,
                event_type,
                status,
                output.error_class,
                f"output_path={output.path}",
            )
        cursor.execute(
            "INSERT INTO dbo.ProcessingLog "
            "(AudioFileId, EventType, Status, ErrorClass, Details) "
            "VALUES (?, ?, ?, ?, ?)",
            audio_file_id,
            "meeting_analysis",
            analysis.status,
            analysis.review_reason,
            f"input_tokens={analysis.input_tokens}; output_tokens={analysis.output_tokens}",
        )
        self._connection.commit()

    def recover_expired_leases(self, *, max_attempts: int = 3) -> int:
        cursor = self._connection.cursor()
        cursor.execute(
            "UPDATE dbo.AudioFile SET Status = 'failed', LastErrorClass = 'LeaseExpired', "
            "LeaseToken = NULL, LeaseExpiresUtc = NULL, NextRetryUtc = NULL, "
            "UpdatedUtc = SYSUTCDATETIME() WHERE Status = 'processing' "
            "AND LeaseExpiresUtc <= SYSUTCDATETIME() AND AttemptCount >= ?",
            max_attempts,
        )
        recovered = max(0, cursor.rowcount)
        self._connection.commit()
        return recovered


def retry_delay_seconds(attempt_count: int, *, base_seconds: int = 60, maximum_seconds: int = 3600) -> int:
    if attempt_count < 1:
        raise ValueError("attempt_count must be positive.")
    return min(base_seconds * (2 ** (attempt_count - 1)), maximum_seconds)
