"""Publish summary and full-transcript PDFs without replacing existing files."""

from dataclasses import dataclass
import logging
from pathlib import Path

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


def meeting_pdf_paths(
    source: Path, output_dir: Path | None = None
) -> tuple[Path, Path]:
    directory = output_dir if output_dir is not None else source.parent
    return (
        directory / f"{source.stem}.pdf",
        directory / f"{source.stem}_Full.pdf",
    )


def _create_or_preserve(path: Path, writer, *args, **kwargs) -> PdfOutputStatus:
    if path.exists():
        return PdfOutputStatus(path, "existing")
    try:
        writer(path, *args, **kwargs)
    except FileExistsError:
        return PdfOutputStatus(path, "existing")
    except Exception as error:
        logger.error(
            "PDF output failed; path=%s error_class=%s",
            path,
            type(error).__name__,
        )
        return PdfOutputStatus(path, "failed", type(error).__name__)
    return PdfOutputStatus(path, "created")


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
    summary = _create_or_preserve(
        summary_path,
        write_meeting_summary_pdf,
        source,
        plant,
        analysis,
        fixture_preview=fixture_preview,
    )
    full_transcript = _create_or_preserve(
        full_path,
        write_transcript_pdf,
        source,
        plant,
        transcript,
        fixture_preview=fixture_preview,
    )
    return MeetingPdfOutputs(summary, full_transcript)
