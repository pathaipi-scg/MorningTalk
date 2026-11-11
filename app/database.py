"""MSSQL connectivity and persistence for scanned audio files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import os
from pathlib import Path
from typing import Any
import uuid

from app.config import SqlConfig


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
                (Status = 'processing' AND LeaseExpiresUtc <= SYSUTCDATETIME())
            )
            """,
            str(lease_token), lease_seconds, audio_file_id, max_attempts,
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
