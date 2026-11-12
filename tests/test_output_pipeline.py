from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import app.auto_scan as auto_scan
from app.meeting_analysis import AnalysisItem, EvidenceReference, MeetingAnalysis
from app.meeting_outputs import MeetingPdfOutputs, PdfOutputStatus, meeting_pdf_paths
from app.scanner import ScannedAudio
from app.transcript import TranscriptDocument, TranscriptSegment
from app.file_manager import source_fingerprint


def _scanner_item(source: Path, *, status: str, pdf_exists: bool) -> ScannedAudio:
    stat = source.stat()
    return ScannedAudio(
        path=source,
        fingerprint=source_fingerprint(source),
        size_bytes=stat.st_size,
        modified_utc=datetime.fromtimestamp(stat.st_mtime, timezone.utc),
        pdf_path=source.with_suffix(".pdf"),
        pdf_exists=pdf_exists,
        stable=True,
        status=status,
    )


def _mock_pipeline(monkeypatch, source: Path, *, summary_status="created"):
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "test-version")
    monkeypatch.delenv("AZURE_SPEECH_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_SPEECH_KEY", raising=False)

    class FakeTranscriber:
        def __init__(self, _config, *, chunk_callback=None):
            self.chunk_callback = chunk_callback

        def transcribe(self, _path):
            return "ตรวจ Encoder Lot 123"

        @staticmethod
        def _probe_duration(_path):
            return 60.0

    transcript = TranscriptDocument(
        (
            TranscriptSegment(
                None,
                None,
                None,
                "ตรวจ Encoder Lot 123",
                "ตรวจ Encoder Lot 123",
            ),
        ),
        "unavailable",
        "completed",
    )

    class FakeCleanup:
        def __init__(self, *_args, **_kwargs):
            pass

        def clean(self, _segments):
            return transcript

    analysis = MeetingAnalysis(
        AnalysisItem(
            "ตรวจ Encoder Lot 123",
            (EvidenceReference(1, "ตรวจ Encoder Lot 123"),),
        ),
        (), (), (), (), (), (), "validated",
    )

    class FakeAnalyzer:
        def __init__(self, *_args, **_kwargs):
            pass

        def analyze(self, _segments):
            return analysis

    summary, full = meeting_pdf_paths(source)
    outputs = MeetingPdfOutputs(
        PdfOutputStatus(summary, summary_status),
        PdfOutputStatus(full, "created"),
    )
    monkeypatch.setattr(auto_scan, "AzureOpenAITranscriber", FakeTranscriber)
    monkeypatch.setattr(auto_scan, "TranscriptCleanup", FakeCleanup)
    monkeypatch.setattr(auto_scan, "GPTMeetingAnalyzer", FakeAnalyzer)
    monkeypatch.setattr(auto_scan, "write_meeting_pdfs", lambda *_a, **_kw: outputs)
    return transcript, analysis, outputs


def test_processing_creates_exact_summary_and_full_output_records(tmp_path: Path, monkeypatch):
    root = tmp_path / "input"
    source = root / "CB" / "Nov26" / "meeting-20261107-0817.mp3"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"fixture audio")
    item = _scanner_item(source, status="discovered", pdf_exists=False)
    transcript, analysis, outputs = _mock_pipeline(monkeypatch, source)
    repository = Mock()
    repository.phase9_schema_ready.return_value = True
    repository.claim_file.return_value = 1
    repository.complete_file.return_value = True

    status = auto_scan.process_scanned_file(
        item,
        root,
        21,
        repository,
        SimpleNamespace(quality_warning_chars_per_minute=0.0),
    )

    assert status == "completed"
    assert outputs.summary.path.name == "meeting-20261107-0817.pdf"
    assert outputs.full_transcript.path.name == "meeting-20261107-0817_Full.pdf"
    repository.save_phase9_outputs.assert_called_once_with(21, transcript, analysis, outputs)
    assert repository.complete_file.call_args.kwargs["review_required"] is False


def test_existing_summary_is_preserved_while_full_pdf_is_tracked_independently(
    tmp_path: Path, monkeypatch,
):
    root = tmp_path / "input"
    source = root / "CB" / "Nov26" / "meeting.mp3"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"fixture audio")
    summary, _full = meeting_pdf_paths(source)
    summary.write_bytes(b"existing summary")
    original = summary.read_bytes()
    item = _scanner_item(source, status="review_required", pdf_exists=True)
    transcript, analysis, outputs = _mock_pipeline(
        monkeypatch, source, summary_status="existing"
    )
    repository = Mock()
    repository.phase9_schema_ready.return_value = True
    repository.claim_file.return_value = 1
    repository.complete_file.return_value = True

    status = auto_scan.process_scanned_file(
        item,
        root,
        22,
        repository,
        SimpleNamespace(quality_warning_chars_per_minute=0.0),
        allow_review_required=True,
    )

    assert status == "review_required"
    assert summary.read_bytes() == original
    repository.save_phase9_outputs.assert_called_once_with(22, transcript, analysis, outputs)
    assert repository.complete_file.call_args.kwargs["review_required"] is True
    repository.claim_file.assert_called_once()
    assert repository.claim_file.call_args.kwargs["allow_review_required"] is True
