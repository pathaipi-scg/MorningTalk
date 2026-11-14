from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import app.phase9 as phase9
import app.file_manager as file_manager
from app.database import SavedTranscriptRecord
from app.meeting_analysis import AnalysisItem, EvidenceReference, MeetingAnalysis
from app.meeting_outputs import MeetingPdfOutputs, PdfOutputStatus, meeting_pdf_paths
from app.transcript import TranscriptDocument, TranscriptSegment, mark_for_review
from app.file_manager import source_fingerprint


def _setup_resume(tmp_path: Path, monkeypatch, *, fingerprint=None, cleanup_raises=False):
    input_dir = tmp_path / "input"
    source = input_dir / "CB" / "Nov26" / phase9.TARGET_RECORDING
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source audio is not read for transcription")
    monkeypatch.setattr(phase9, "is_file_stable", lambda _path: True)

    segment = TranscriptSegment(
        None, None, None, "preserved raw transcript", review_required=True
    )
    raw_transcript = segment.raw_text
    record = SavedTranscriptRecord(
        phase9.RESUME_AUDIO_FILE_ID,
        str(source.resolve()),
        fingerprint or source_fingerprint(source),
        "review_required",
        raw_transcript,
        json.dumps([asdict(segment)], ensure_ascii=False),
        "unavailable",
        None,
        None,
    )
    preview_root = tmp_path / "previews"
    monkeypatch.setattr(phase9, "PREVIEW_DIRECTORY", preview_root)
    monkeypatch.setattr(
        file_manager,
        "DEFAULT_TRANSCRIPT_ROOT",
        tmp_path / "Transcript",
    )
    output_dir = preview_root / "resume-v1"
    monkeypatch.setattr(
        phase9,
        "get_azure_config",
        lambda: SimpleNamespace(
            text_endpoint="https://text.example.test",
            text_api_key="not-real",
            text_deployment="gpt-5.4-mini",
            quality_warning_chars_per_minute=0.0,
        ),
    )
    monkeypatch.setattr(phase9, "get_azure_openai_api_version", lambda: "v1")
    connection = Mock()
    repository = Mock()
    repository.phase9_schema_ready.return_value = True
    repository.load_saved_transcript.return_value = record
    repository.claim_file.return_value = 1
    repository.renew_lease.return_value = True
    repository.complete_file.return_value = True
    monkeypatch.setattr(phase9, "connect_sql", lambda _config: connection)
    monkeypatch.setattr(phase9, "SqlRepository", lambda _connection: repository)

    class FakeCleanup:
        def __init__(self, _config, *, api_version, **_kwargs):
            assert api_version == "v1"

        def clean(self, segments):
            if cleanup_raises:
                raise RuntimeError("mock cleanup failure")
            return TranscriptDocument(
                tuple(
                    TranscriptSegment(
                        item.start_seconds,
                        item.end_seconds,
                        item.speaker_id,
                        item.raw_text,
                        item.raw_text,
                    )
                    for item in segments
                ),
                "available",
                "completed",
                input_tokens=12,
                output_tokens=4,
            )

    analysis = MeetingAnalysis(
        AnalysisItem(
            "ตรวจสอบประเด็นที่บันทึกไว้",
            (EvidenceReference(1, raw_transcript),),
        ),
        (), (), (), (), (), (), "validated",
    )

    class FakeAnalyzer:
        def __init__(self, *_args, **_kwargs):
            pass

        def analyze(self, _segments):
            return analysis

    output_calls = []

    def fake_write(source_path, _plant, transcript, meeting_analysis, *, output_dir=None):
        paths = meeting_pdf_paths(source_path, output_dir)
        output_calls.append((transcript, meeting_analysis, paths))
        return MeetingPdfOutputs(
            PdfOutputStatus(paths[0], "created"),
            PdfOutputStatus(paths[1], "created"),
        )

    monkeypatch.setattr(phase9, "TranscriptCleanup", FakeCleanup)
    monkeypatch.setattr(phase9, "GPTMeetingAnalyzer", FakeAnalyzer)
    monkeypatch.setattr(phase9, "write_meeting_pdfs", fake_write)
    transcriber = Mock()
    transcriber._probe_duration.return_value = 60.0
    monkeypatch.setattr(phase9, "AzureOpenAITranscriber", transcriber)
    return input_dir, source, output_dir, record, connection, repository, output_calls, transcriber


def _run(input_dir, output_dir):
    return phase9.run_controlled_transcript_resume(
        input_dir=input_dir,
        output_dir=output_dir,
        confirm_azure_billing=True,
        confirm_mssql_write=True,
    )


def test_resume_uses_saved_transcript_and_never_calls_transcriber(tmp_path, monkeypatch):
    (
        input_dir, source, output_dir, record, connection, repository,
        output_calls, transcriber,
    ) = _setup_resume(tmp_path, monkeypatch)
    original_fingerprint = source_fingerprint(source)

    summary, full, json_path, transcript, analysis, outputs = _run(input_dir, output_dir)

    assert source_fingerprint(source) == original_fingerprint == record.fingerprint
    assert transcript.raw_transcript == record.raw_transcript
    assert transcript.segments[0].speaker_id is None
    assert analysis.status == "validated"
    assert summary.name == "meeting-20261107-0817_Summary_Draft.pdf"
    assert full.name == "meeting-20261107-0817_Full.pdf"
    assert summary.parent == full.parent == tmp_path / "Transcript" / "CB" / "Nov26"
    assert outputs.summary.status == outputs.full_transcript.status == "created"
    assert len(output_calls) == 1
    transcriber.transcribe.assert_not_called()
    repository.claim_file.assert_called_once_with(
        phase9.RESUME_AUDIO_FILE_ID,
        repository.claim_file.call_args.args[1],
        lease_seconds=7200,
        allow_review_required=True,
    )
    repository.save_phase9_outputs.assert_called_once_with(
        phase9.RESUME_AUDIO_FILE_ID, transcript, analysis, outputs
    )
    repository.complete_file.assert_called_once()
    assert repository.complete_file.call_args.kwargs["review_required"] is False
    assert connection.close.called
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["source_fingerprint"] == record.fingerprint
    assert payload["raw_transcript"] == record.raw_transcript


def test_resume_allows_replacing_existing_transcript_folder_pdfs(tmp_path, monkeypatch):
    (
        input_dir, source, output_dir, _record, _connection, _repository,
        output_calls, _transcriber,
    ) = _setup_resume(tmp_path, monkeypatch)
    paths = file_manager.resolve_meeting_output_paths(source)
    paths.directory.mkdir(parents=True)
    paths.summary_draft_pdf.write_bytes(b"old draft")
    paths.full_transcript_pdf.write_bytes(b"old full transcript")

    summary, full, *_ = _run(input_dir, output_dir)

    assert summary == paths.summary_draft_pdf
    assert full == paths.full_transcript_pdf
    assert len(output_calls) == 1


def test_resume_rejects_source_fingerprint_mismatch_before_claim_or_api(
    tmp_path, monkeypatch,
):
    (
        input_dir, _source, output_dir, _record, connection, repository,
        _output_calls, _transcriber,
    ) = _setup_resume(tmp_path, monkeypatch, fingerprint="wrong-fingerprint")

    with pytest.raises(RuntimeError, match="fingerprint"):
        _run(input_dir, output_dir)

    repository.claim_file.assert_not_called()
    repository.save_phase9_outputs.assert_not_called()
    assert not output_dir.exists()
    assert connection.close.called


def test_resume_refuses_existing_versioned_preview_before_sql_or_api(
    tmp_path, monkeypatch,
):
    (
        input_dir, _source, output_dir, _record, _connection, _repository,
        _output_calls, _transcriber,
    ) = _setup_resume(tmp_path, monkeypatch)
    output_dir.mkdir(parents=True)
    (output_dir / "keep.pdf").write_bytes(b"existing preview")
    connect = phase9.connect_sql

    with pytest.raises(FileExistsError, match="already exists"):
        _run(input_dir, output_dir)

    assert phase9.connect_sql is connect
    assert (output_dir / "keep.pdf").read_bytes() == b"existing preview"


def test_resume_failure_releases_claim_for_review_without_retrying_or_transcribing(
    tmp_path, monkeypatch,
):
    (
        input_dir, _source, output_dir, _record, _connection, repository,
        output_calls, transcriber,
    ) = _setup_resume(tmp_path, monkeypatch, cleanup_raises=True)

    with pytest.raises(RuntimeError, match="mock cleanup failure"):
        _run(input_dir, output_dir)

    repository.release_claim_for_review.assert_called_once()
    repository.fail_file.assert_not_called()
    repository.save_phase9_outputs.assert_not_called()
    repository.complete_file.assert_not_called()
    transcriber.transcribe.assert_not_called()
    assert output_calls == []


def test_resume_requires_both_explicit_authorizations(tmp_path, monkeypatch):
    input_dir, _source, output_dir, *_rest = _setup_resume(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="explicit"):
        phase9.run_controlled_transcript_resume(
            input_dir=input_dir,
            output_dir=output_dir,
            confirm_azure_billing=True,
            confirm_mssql_write=False,
        )
