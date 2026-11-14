"""Input discovery, output mapping, and completion-state management."""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime
import re
from typing import Iterator

DEFAULT_RECORDING_ROOT = Path(r"D:\KM\Vault\MorningTalk\Record File")
DEFAULT_TRANSCRIPT_ROOT = DEFAULT_RECORDING_ROOT.parent / "Transcript"


@dataclass(frozen=True)
class MeetingOutputPaths:
    directory: Path
    transcript_text: Path
    full_transcript_pdf: Path
    summary_draft_pdf: Path
    approved_summary_pdf: Path


def resolve_meeting_output_paths(
    source: Path,
    *,
    transcript_root: Path | None = None,
) -> MeetingOutputPaths:
    source = source.resolve()
    transcript_root = transcript_root or DEFAULT_TRANSCRIPT_ROOT
    if source.suffix.casefold() != ".mp3":
        raise ValueError("Meeting output paths require an MP3 source.")
    month_pattern = re.compile(r"[A-Za-z]{3}\d{2}")
    month_directory = next(
        (parent for parent in source.parents if month_pattern.fullmatch(parent.name)),
        None,
    )
    if month_directory is not None:
        month = month_directory.name
        plant = month_directory.parent.name
    elif (
        source.parent.name.casefold()[:3] in {
            "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec",
        }
        and re.fullmatch(r"\d{4}", source.parent.parent.name)
    ):
        month = f"{source.parent.name[:3]}{source.parent.parent.name[-2:]}"
        plant = source.parent.parent.parent.name
    else:
        month = ""
        plant = ""
    if not month or not plant:
        raise ValueError(
            "Source must be under Record File\\<Plant>\\<MonthYY>."
        )
    directory = transcript_root / plant / month
    stem = source.stem
    return MeetingOutputPaths(
        directory=directory,
        transcript_text=directory / f"{stem}.txt",
        full_transcript_pdf=directory / f"{stem}_Full.pdf",
        summary_draft_pdf=directory / f"{stem}_Summary_Draft.pdf",
        approved_summary_pdf=directory / f"{stem}_Summary.pdf",
    )


def discover_mp3_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    return sorted(path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".mp3")


def pdf_path_for_source(source: Path) -> Path:
    return resolve_meeting_output_paths(source).full_transcript_pdf


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
