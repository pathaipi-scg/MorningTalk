from pathlib import Path

import pytest

import app.meeting_outputs as meeting_outputs
import app.pdf_export as pdf_export
import app.file_manager as file_manager
from app.meeting_analysis import AnalysisItem, EvidenceReference, MeetingAnalysis
from app.meeting_outputs import write_meeting_pdfs
from app.transcript import TranscriptDocument, TranscriptSegment


def _document() -> TranscriptDocument:
    return TranscriptDocument(
        (
            TranscriptSegment(1, 2, "Speaker A", "ตรวจสอบ Encoder"),
            TranscriptSegment(2, 3, "Speaker B", "ยืนยัน Lot 123"),
        ),
        "available",
        "completed",
    )


def test_transcript_text_overwrite_is_exact(tmp_path: Path):
    path = tmp_path / "meeting.txt"
    path.write_text("stale transcript", encoding="utf-8")
    expected = _document().cleaned_transcript

    meeting_outputs.write_transcript_text(path, expected, overwrite=True)

    assert path.read_bytes() == expected.encode("utf-8")


def test_transcript_text_replace_failure_preserves_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = tmp_path / "meeting.txt"
    path.write_bytes(b"original")

    def fail_replace(_temporary: Path, _destination: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(meeting_outputs.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        meeting_outputs.write_transcript_text(
            path,
            "replacement",
            overwrite=True,
        )

    assert path.read_bytes() == b"original"
    assert not list(tmp_path.glob(".*.tmp.txt"))


def test_full_transcript_pdf_overwrite_contains_every_segment(tmp_path: Path):
    source = tmp_path / "meeting.mp3"
    source.write_bytes(b"source MP3")
    original_source = source.read_bytes()
    pdf = tmp_path / "meeting_Full.pdf"
    pdf.write_bytes(b"stale PDF")

    pdf_export.write_transcript_pdf(
        pdf,
        source,
        "CB",
        _document(),
        overwrite=True,
    )

    _page_count, extracted = pdf_export._extract_pdf_text(pdf)
    assert "ตรวจสอบ Encoder" in extracted
    assert "ยืนยัน Lot 123" in extracted
    assert source.read_bytes() == original_source
    assert not list(tmp_path.glob(".*.tmp.pdf"))


def test_pdf_validation_failure_preserves_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "meeting.mp3"
    source.write_bytes(b"source MP3")
    pdf = tmp_path / "meeting_Full.pdf"
    pdf.write_bytes(b"original PDF")

    def fail_validation(_path: Path, _expected_texts: tuple[str, ...] = ()) -> None:
        raise ValueError("simulated PDF validation failure")

    monkeypatch.setattr(pdf_export, "_validate_pdf", fail_validation)
    with pytest.raises(ValueError, match="simulated PDF validation failure"):
        pdf_export.write_transcript_pdf(
            pdf,
            source,
            "CB",
            _document(),
            overwrite=True,
        )

    assert pdf.read_bytes() == b"original PDF"
    assert not list(tmp_path.glob(".*.tmp.pdf"))


def test_meeting_outputs_overwrite_transcripts_but_preserve_unvalidated_draft(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "Record File" / "CB" / "Nov26" / "meeting.mp3"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source MP3")
    monkeypatch.setattr(file_manager, "DEFAULT_TRANSCRIPT_ROOT", tmp_path / "Transcript")
    paths = file_manager.resolve_meeting_output_paths(source)
    paths.directory.mkdir(parents=True)
    paths.transcript_text.write_bytes(b"stale transcript")
    paths.full_transcript_pdf.write_bytes(b"stale full PDF")
    paths.summary_draft_pdf.write_bytes(b"preserve unvalidated draft")
    draft_before = paths.summary_draft_pdf.read_bytes()

    outputs = write_meeting_pdfs(
        source,
        "CB",
        _document(),
        MeetingAnalysis.for_review("Needs review"),
    )

    assert paths.transcript_text.read_bytes() == (
        _document().cleaned_transcript.encode("utf-8")
    )
    _page_count, extracted = pdf_export._extract_pdf_text(paths.full_transcript_pdf)
    assert "ตรวจสอบ Encoder" in extracted
    assert "ยืนยัน Lot 123" in extracted
    assert paths.summary_draft_pdf.read_bytes() == draft_before
    assert outputs.transcript_text.status == "overwritten"
    assert outputs.full_transcript.status == "overwritten"
    assert outputs.summary.status == "existing"


def test_meeting_outputs_replace_draft_only_with_validated_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "Record File" / "CB" / "Nov26" / "meeting.mp3"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source MP3")
    monkeypatch.setattr(file_manager, "DEFAULT_TRANSCRIPT_ROOT", tmp_path / "Transcript")
    paths = file_manager.resolve_meeting_output_paths(source)
    paths.directory.mkdir(parents=True)
    paths.summary_draft_pdf.write_bytes(b"stale draft")
    analysis = MeetingAnalysis(
        AnalysisItem(
            "ตรวจสอบ Encoder",
            (EvidenceReference(1, "ตรวจสอบ Encoder"),),
        ),
        (), (), (), (), (), (), "validated",
    )

    outputs = write_meeting_pdfs(source, "CB", _document(), analysis)

    _page_count, extracted = pdf_export._extract_pdf_text(
        paths.summary_draft_pdf
    )
    assert "DRAFT - NOT APPROVED" in extracted
    assert outputs.summary.status == "overwritten"
