"""Stable recursive file scanning and persisted-state classification."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time
from collections.abc import Sequence

from app.file_manager import discover_mp3_files, pdf_path_for_source, source_fingerprint
from app.database import AudioRecord


@dataclass(frozen=True)
class ScannedAudio:
    path: Path
    fingerprint: str
    size_bytes: int
    modified_utc: datetime
    pdf_path: Path
    pdf_exists: bool
    stable: bool
    status: str


def is_file_stable(path: Path, interval_seconds: float = 1.0) -> bool:
    if interval_seconds < 0:
        raise ValueError("Stability interval cannot be negative.")
    before = path.stat()
    if interval_seconds:
        time.sleep(interval_seconds)
    after = path.stat()
    return before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns


def classify_scanned_file(
    fingerprint: str,
    pdf_exists: bool,
    record: AudioRecord | None,
    *,
    stable: bool = True,
) -> str:
    if not stable:
        return "unstable"
    if record is None:
        return "review_required" if pdf_exists else "new"
    if record.fingerprint != fingerprint:
        return "review_required" if pdf_exists else "modified"
    if record.status == "completed":
        return "completed" if pdf_exists else "review_required"
    if pdf_exists:
        return "review_required"
    return record.status


def scan_directory(
    input_dir: Path,
    *,
    files: Sequence[Path] | None = None,
    records: dict[str, AudioRecord] | None = None,
    stability_interval_seconds: float = 1.0,
) -> list[ScannedAudio]:
    records = records or {}
    items = []
    sources = discover_mp3_files(input_dir) if files is None else sorted(files)
    for path in sources:
        path.relative_to(input_dir)
        stat = path.stat()
        stable = is_file_stable(path, stability_interval_seconds)
        fingerprint = source_fingerprint(path)
        pdf_path = pdf_path_for_source(path)
        key = str(path.resolve()).casefold()
        items.append(ScannedAudio(
            path=path,
            fingerprint=fingerprint,
            size_bytes=stat.st_size,
            modified_utc=datetime.fromtimestamp(stat.st_mtime, timezone.utc),
            pdf_path=pdf_path,
            pdf_exists=pdf_path.is_file(),
            stable=stable,
            status=classify_scanned_file(
                fingerprint, pdf_path.is_file(), records.get(key), stable=stable,
            ),
        ))
    return items
