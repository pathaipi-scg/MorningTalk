"""Input discovery, output mapping, and completion-state management."""

import hashlib
import os
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime
import re
from typing import Iterator

def discover_mp3_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    return sorted(path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".mp3")


def pdf_path_for_source(source: Path) -> Path:
    return source.with_suffix(".pdf")


def meeting_datetime(filename: str) -> datetime | None:
    match = re.search(r"(\d{8})-(\d{4})", Path(filename).stem)
    if not match:
        return None
    try:
        return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M")
    except ValueError:
        return None


def source_fingerprint(source: Path) -> str:
    stat = source.stat()
    value = f"{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(value.encode("ascii")).hexdigest()


@contextmanager
def source_processing_lock(source: Path) -> Iterator[None]:
    lock_path = source.with_name(f".{source.name}.morningtalk.lock")
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"Another process is already handling {source.name}.") from error
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as lock_file:
            lock_file.write(str(os.getpid()))
        yield
    finally:
        lock_path.unlink(missing_ok=True)
