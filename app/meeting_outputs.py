"""Publish meeting outputs with validation and atomic replacement."""

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import tempfile

from app.file_manager import resolve_meeting_output_paths
from app.meeting_analysis import MeetingAnalysis
from app.pdf_export import write_meeting_summary_pdf, write_transcript_pdf
from app.transcript import TranscriptDocument

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PdfOutputStatus:
    path: Path
    status: str
    error_class: str | None = None


@dataclass(frozen=True)
class MeetingPdfOutputs:
    summary: PdfOutputStatus
    full_transcript: PdfOutputStatus
    transcript_text: PdfOutputStatus | None = None


def meeting_pdf_paths(
    source: Path, output_dir: Path | None = None
) -> tuple[Path, Path]:
    if output_dir is None:
        paths = resolve_meeting_output_paths(source)
        return paths.summary_draft_pdf, paths.full_transcript_pdf
    directory = output_dir
    return (
        directory / f"{source.stem}_Summary_Draft.pdf",
        directory / f"{source.stem}_Full.pdf",
    )


def _create_or_preserve(
    path: Path,
    writer,
    *args,
    overwrite: bool = False,
    **kwargs,
) -> PdfOutputStatus:
    existed = path.exists()
    if existed and not overwrite:
        return PdfOutputStatus(path, "existing")
    try:
        writer(path, *args, overwrite=overwrite, **kwargs)
    except FileExistsError:
        return PdfOutputStatus(path, "existing")
    except Exception as error:
        logger.error(
            "PDF output failed; path=%s error_class=%s",
            path,
            type(error).__name__,
        )
        return PdfOutputStatus(path, "failed", type(error).__name__)
    return PdfOutputStatus(path, "overwritten" if existed else "created")


def write_meeting_pdfs(
    source: Path,
    plant: str,
    transcript: TranscriptDocument,
    analysis: MeetingAnalysis,
    *,
    output_dir: Path | None = None,
    fixture_preview: bool = False,
) -> MeetingPdfOutputs:
    summary_path, full_path = meeting_pdf_paths(source, output_dir)
    text_path = (
        output_dir / f"{source.stem}.txt"
        if output_dir is not None
        else resolve_meeting_output_paths(source).transcript_text
    )
    text_status = _create_or_preserve(
        text_path,
        write_transcript_text,
        transcript.cleaned_transcript,
        overwrite=True,
    )
    summary = _create_or_preserve(
        summary_path,
        write_meeting_summary_pdf,
        source,
        plant,
        analysis,
        overwrite=analysis.status == "validated",
        fixture_preview=fixture_preview,
        draft=True,
    )
    full_transcript = _create_or_preserve(
        full_path,
        write_transcript_pdf,
        source,
        plant,
        transcript,
        overwrite=True,
        fixture_preview=fixture_preview,
    )
    return MeetingPdfOutputs(summary, full_transcript, text_status)


def write_transcript_text(
    path: Path,
    text: str,
    *,
    overwrite: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Transcript already exists: {path.name}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".tmp.txt",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    encoded = text.encode("utf-8")
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        if temporary.read_bytes() != encoded:
            raise OSError("Temporary transcript validation failed.")
        if path.exists() and not overwrite:
            raise FileExistsError(f"Transcript already exists: {path.name}")
        if overwrite:
            os.replace(temporary, path)
        else:
            temporary.rename(path)
    finally:
        temporary.unlink(missing_ok=True)
