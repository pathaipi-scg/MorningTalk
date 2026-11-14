from pathlib import Path

import pytest

import app.meeting_outputs as meeting_outputs
import app.pdf_export as pdf_export
import app.file_manager as file_manager
from app.meeting_analysis import AnalysisItem, EvidenceReference, MeetingAnalysis
from app.meeting_outputs import (
    OutputPublicationError,
    publish_staged_output_set,
    write_meeting_pdfs,
)
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
    monkeypatch.setattr(meeting_outputs, "DEFAULT_PREVIEW_ROOT", tmp_path / "previews")
    monkeypatch.setattr(meeting_outputs, "DEFAULT_TEMP_ROOT", tmp_path / "temp")
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
    monkeypatch.setattr(meeting_outputs, "DEFAULT_PREVIEW_ROOT", tmp_path / "previews")
    monkeypatch.setattr(meeting_outputs, "DEFAULT_TEMP_ROOT", tmp_path / "temp")
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


@pytest.mark.parametrize(
    ("stem", "month", "reason"),
    (
        ("meeting-20261029-0814", "Oct26", "ReviewFindingNotSourceGrounded"),
        ("meeting-20261030-0805", "Oct26", "ReviewFindingNotSourceGrounded"),
        ("meeting-20260924-0822", "Sep26", "CleanedTextTooShort"),
        ("meeting-20260925-0822", "Sep26", "UncertainSpeechRequiresReview"),
    ),
)
def test_cleanup_review_preserves_affected_recording_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stem: str,
    month: str,
    reason: str,
):
    source = tmp_path / "Record File" / "CB" / month / f"{stem}.mp3"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"fixture MP3")
    output_root = tmp_path / "Transcript"
    monkeypatch.setattr(file_manager, "DEFAULT_TRANSCRIPT_ROOT", output_root)
    monkeypatch.setattr(meeting_outputs, "DEFAULT_PREVIEW_ROOT", tmp_path / "previews")
    monkeypatch.setattr(meeting_outputs, "DEFAULT_TEMP_ROOT", tmp_path / "temp")
    paths = file_manager.resolve_meeting_output_paths(
        source,
        transcript_root=output_root,
    )
    paths.directory.mkdir(parents=True)
    originals = {
        paths.transcript_text: b"original transcript",
        paths.full_transcript_pdf: b"original full PDF",
        paths.summary_draft_pdf: b"original summary draft",
    }
    for path, content in originals.items():
        path.write_bytes(content)
    document = TranscriptDocument(
        (TranscriptSegment(None, None, None, "ถอดเสียงที่ต้องตรวจ"),),
        "unavailable",
        "review_required",
        reason,
    )

    outputs = write_meeting_pdfs(
        source,
        "CB",
        document,
        MeetingAnalysis.for_review("TranscriptCleanupRequiresReview"),
    )

    assert outputs.transcript_text.status == "not_published"
    assert outputs.full_transcript.status == "not_published"
    assert outputs.summary.status == "not_published"
    assert {path: path.read_bytes() for path in originals} == originals
    assert not (tmp_path / "previews").exists()


def test_staged_validation_failure_preserves_existing_output_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "Record File" / "CB" / "Nov26" / "meeting.mp3"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source MP3")
    monkeypatch.setattr(file_manager, "DEFAULT_TRANSCRIPT_ROOT", tmp_path / "Transcript")
    monkeypatch.setattr(meeting_outputs, "DEFAULT_PREVIEW_ROOT", tmp_path / "previews")
    monkeypatch.setattr(meeting_outputs, "DEFAULT_TEMP_ROOT", tmp_path / "temp")
    paths = file_manager.resolve_meeting_output_paths(
        source,
        transcript_root=tmp_path / "Transcript",
    )
    paths.directory.mkdir(parents=True)
    originals = {
        paths.transcript_text: b"original transcript",
        paths.full_transcript_pdf: b"original full PDF",
        paths.summary_draft_pdf: b"original summary draft",
    }
    for path, content in originals.items():
        path.write_bytes(content)
    analysis = MeetingAnalysis(
        AnalysisItem(
            "ตรวจสอบ Encoder",
            (EvidenceReference(1, "ตรวจสอบ Encoder"),),
        ),
        (), (), (), (), (), (), "validated",
    )

    def fail_validation(*_args, **_kwargs):
        raise ValueError("simulated staged validation failure")

    monkeypatch.setattr(meeting_outputs, "_validate_staged_pdf", fail_validation)
    with pytest.raises(ValueError, match="simulated staged validation failure"):
        write_meeting_pdfs(source, "CB", _document(), analysis)

    assert {path: path.read_bytes() for path in originals} == originals


def test_multi_file_publication_rolls_back_after_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    staged_dir = tmp_path / "staged"
    output_dir = tmp_path / "Transcript"
    backup_root = tmp_path / "backups"
    staged_dir.mkdir()
    output_dir.mkdir()
    targets = (
        output_dir / "meeting.txt",
        output_dir / "meeting_Full.pdf",
        output_dir / "meeting_Summary_Draft.pdf",
    )
    candidates = (
        staged_dir / "meeting.txt",
        staged_dir / "meeting_Full.pdf",
        staged_dir / "meeting_Summary_Draft.pdf",
    )
    originals = {}
    staged_outputs = {}
    for index, (candidate, target) in enumerate(zip(candidates, targets, strict=True)):
        candidate.write_bytes(f"replacement-{index}".encode())
        originals[target] = f"original-{index}".encode()
        target.write_bytes(originals[target])
        staged_outputs[target] = candidate
    approved_summary = output_dir / "meeting_Summary.pdf"
    approved_summary.write_bytes(b"approved summary")
    original_replace = meeting_outputs.os.replace
    failed = False

    def fail_second_replace_once(source_path, target_path):
        nonlocal failed
        if Path(target_path) == targets[1] and not failed:
            failed = True
            raise OSError("simulated second-file replace failure")
        original_replace(source_path, target_path)

    monkeypatch.setattr(meeting_outputs.os, "replace", fail_second_replace_once)
    with pytest.raises(
        OutputPublicationError,
        match="OutputPublicationFailedAndRolledBack",
    ):
        publish_staged_output_set(
            staged_outputs,
            approved_summary=approved_summary,
            backup_root=backup_root,
        )

    assert failed
    assert {path: path.read_bytes() for path in originals} == originals
    assert approved_summary.read_bytes() == b"approved summary"
    assert list(backup_root.glob("*/transaction.json"))


def test_multi_file_publication_removes_new_outputs_after_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    staged_dir = tmp_path / "staged"
    output_dir = tmp_path / "Transcript"
    backup_root = tmp_path / "backups"
    staged_dir.mkdir()
    output_dir.mkdir()
    first = output_dir / "meeting.txt"
    second = output_dir / "meeting_Full.pdf"
    staged_first = staged_dir / first.name
    staged_second = staged_dir / second.name
    staged_first.write_bytes(b"new transcript")
    staged_second.write_bytes(b"new full PDF")
    original_replace = meeting_outputs.os.replace

    def fail_second_replace(source_path, target_path):
        if Path(target_path) == second:
            raise OSError("simulated replace failure")
        original_replace(source_path, target_path)

    monkeypatch.setattr(meeting_outputs.os, "replace", fail_second_replace)
    with pytest.raises(OutputPublicationError, match="OutputPublicationFailedAndRolledBack"):
        publish_staged_output_set(
            {first: staged_first, second: staged_second},
            approved_summary=output_dir / "meeting_Summary.pdf",
            backup_root=backup_root,
        )

    assert not first.exists()
    assert not second.exists()
