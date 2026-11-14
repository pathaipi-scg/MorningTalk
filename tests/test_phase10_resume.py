from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

import app.phase10 as phase10
import app.file_manager as file_manager
from app.config import AzureConfig
from app.file_manager import source_fingerprint
from app.main import main
from app.transcript import TranscriptSegment


ENDPOINT = (
    "https://smartagent-gummo.openai.azure.com/openai/responses"
    "?api-version=2025-04-01-preview"
)


def _fixture(tmp_path: Path, monkeypatch):
    source = (
        tmp_path / "Record File" / "CB" / "Nov26"
        / "meeting-20261107-0817.mp3"
    )
    source.parent.mkdir(parents=True)
    source.write_bytes(b"fixture recording")
    monkeypatch.setattr(file_manager, "DEFAULT_TRANSCRIPT_ROOT", tmp_path / "Transcript")
    segment = TranscriptSegment(
        1.0, 2.0, None, "ตรวจ Encoder Lot 123", "ตรวจ Encoder Lot 123"
    )
    phase9_json = tmp_path / "meeting.phase9.json"
    phase9_json.write_text(
        json.dumps(
            {
                "cleanup_status": "completed",
                "meeting_analysis": {"status": "review_required"},
                "raw_transcript": segment.raw_text,
                "cleaned_transcript": segment.display_text,
                "segments": [asdict(segment)],
                "source_fingerprint": source_fingerprint(source),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    workspace_root = tmp_path / "workspace"
    monkeypatch.setenv("MORNINGTALK_WORKSPACE_ROOT", str(workspace_root))
    output_dir = workspace_root / "previews" / "phase10-test"
    config = AzureConfig(
        text_endpoint=ENDPOINT,
        text_api_key="not-a-real-key",
        text_deployment=phase10.EXPECTED_DEPLOYMENT,
        transcription_endpoint="",
        transcription_api_key="",
        transcription_deployment="",
        transcription_api_version="",
    )
    monkeypatch.setattr(phase10, "get_azure_config", lambda: config)
    monkeypatch.setattr(phase10, "get_azure_openai_api_version", lambda: "v1")
    return phase9_json, source, output_dir


def _analysis_response(*, valid=True):
    source_text = "ตรวจ Encoder Lot 123"

    def evidence(segment_id=1):
        return {
            "segment_id": segment_id,
            "start_char": 0,
            "end_char": len(source_text),
        }

    if valid:
        analysis = {
            "overview": {
                "text": "มีการตรวจ Encoder Lot 123",
                "owner": None,
                "deadline": None,
                "evidence": [evidence()],
            },
            "classifications": [],
            "topics": [{
                "text": "เครื่องจักร — ตรวจ Encoder Lot 123",
                "owner": None,
                "deadline": None,
                "evidence": [evidence()],
            }],
            "key_issues": [{
                "text": "ตรวจ Encoder Lot 123",
                "owner": None,
                "deadline": None,
                "evidence": [evidence()],
            }],
            "actions": [],
            "dependencies": [],
            "schedules": [],
            "risks": [],
            "closing_activities": [],
        }
    else:
        analysis = {
            "overview": {
                "text": "Unsupported statement",
                "owner": None,
                "deadline": None,
                "evidence": [evidence(99)],
            },
            "classifications": [],
            "topics": [],
            "key_issues": [],
            "actions": [],
            "dependencies": [],
            "schedules": [],
            "risks": [],
            "closing_activities": [],
        }
    return Mock(
        status_code=200,
        json=Mock(return_value={
            "status": "completed",
            "output": [{
                "type": "message",
                "content": [{
                    "type": "output_text",
                    "text": json.dumps(analysis, ensure_ascii=False),
                }],
            }],
            "usage": {"input_tokens": 10, "output_tokens": 8},
        }),
    )


def test_preflight_validates_source_and_isolated_one_request_plan(tmp_path, monkeypatch):
    phase9_json, source, output_dir = _fixture(tmp_path, monkeypatch)

    preflight = phase10.preflight_analysis_resume(phase9_json, source, output_dir)

    assert preflight.endpoint_url == ENDPOINT
    assert preflight.deployment == phase10.EXPECTED_DEPLOYMENT
    assert preflight.api_mode == "responses"
    assert preflight.requests_planned == 1
    assert preflight.retries_enabled is False
    assert preflight.fingerprint_matches is True
    assert not output_dir.exists()


def test_resume_makes_one_analysis_request_and_withholds_unreviewed_summary(
    tmp_path, monkeypatch
):
    phase9_json, source, output_dir = _fixture(tmp_path, monkeypatch)
    original_phase9_json = phase9_json.read_bytes()
    session = Mock()
    session.post.return_value = _analysis_response()
    pdf_write = Mock()
    monkeypatch.setattr(phase10, "write_meeting_summary_pdf", pdf_write)
    import app.transcriber as transcriber_module
    import app.transcript_cleanup as cleanup_module

    transcribe = Mock(side_effect=AssertionError("transcription must not run"))
    clean = Mock(side_effect=AssertionError("cleanup must not run"))
    monkeypatch.setattr(
        transcriber_module.AzureOpenAITranscriber, "transcribe", transcribe
    )
    monkeypatch.setattr(cleanup_module.TranscriptCleanup, "clean", clean)

    result = phase10.run_analysis_resume(
        phase9_json, source, output_dir, session=session
    )

    session.post.assert_called_once()
    assert session.post.call_args.kwargs["allow_redirects"] is False
    assert result.analysis.status == "review_required"
    assert result.analysis.review_reason == "SemanticReviewRequired"
    assert result.summary_pdf is None
    saved = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert saved["meeting_analysis_requests"] == 1
    assert saved["transcription_requests"] == 0
    assert saved["cleanup_requests"] == 0
    assert saved["phase9_transcript_json"] == str(phase9_json.resolve())
    assert saved["evidence_offset_unit"].startswith("zero-based Python Unicode")
    evidence_ref = saved["meeting_analysis"]["overview"]["evidence"][0]
    assert evidence_ref == {
        "segment_id": 1,
        "start_char": 0,
        "end_char": len("ตรวจ Encoder Lot 123"),
    }
    assert "quote" not in evidence_ref
    assert phase9_json.read_bytes() == original_phase9_json
    pdf_write.assert_not_called()
    transcribe.assert_not_called()
    clean.assert_not_called()


def test_resume_validation_failure_writes_no_pdf_and_does_not_retry(
    tmp_path, monkeypatch
):
    phase9_json, source, output_dir = _fixture(tmp_path, monkeypatch)
    session = Mock()
    session.post.return_value = _analysis_response(valid=False)
    pdf_write = Mock()
    monkeypatch.setattr(phase10, "write_meeting_summary_pdf", pdf_write)

    result = phase10.run_analysis_resume(
        phase9_json, source, output_dir, session=session
    )

    session.post.assert_called_once()
    assert result.analysis.review_reason == "OverviewEvidenceInvalid"
    assert result.summary_pdf is None
    pdf_write.assert_not_called()
    assert result.json_path.is_file()


def test_resume_preserves_existing_preview_without_request(tmp_path, monkeypatch):
    phase9_json, source, output_dir = _fixture(tmp_path, monkeypatch)
    output_dir.mkdir(parents=True)
    sentinel = output_dir / "existing.pdf"
    sentinel.write_bytes(b"preserve")
    session = Mock()

    with pytest.raises(phase10.AnalysisResumePreflightError, match="PreviewDirectoryAlreadyExists"):
        phase10.run_analysis_resume(
            phase9_json, source, output_dir, session=session
        )

    session.post.assert_not_called()
    assert sentinel.read_bytes() == b"preserve"


def test_preflight_failure_prevents_any_analysis_request(tmp_path, monkeypatch):
    phase9_json, source, output_dir = _fixture(tmp_path, monkeypatch)
    payload = json.loads(phase9_json.read_text(encoding="utf-8"))
    payload["source_fingerprint"] = "mismatch"
    phase9_json.write_text(json.dumps(payload), encoding="utf-8")
    session = Mock()

    with pytest.raises(phase10.AnalysisResumePreflightError, match="SourceFingerprintMismatch"):
        phase10.run_analysis_resume(phase9_json, source, output_dir, session=session)

    session.post.assert_not_called()
    assert not output_dir.exists()


def test_prepared_attempt_saves_raw_response_before_validation(tmp_path, monkeypatch):
    phase9_json, source, output_dir = _fixture(tmp_path, monkeypatch)
    prepared = phase10.prepare_analysis_attempt(
        phase9_json, source, output_dir
    )
    request = json.loads(prepared.request_path.read_text(encoding="utf-8"))
    assert request["status"] == "prepared"
    assert request["expected_requests"] == 1
    assert request["automatic_retries"] is False
    response = _analysis_response()
    response.content = b'{"retained":"raw response"}'

    def parse_after_raw_saved():
        assert prepared.raw_response_path.read_bytes() == response.content
        return response.json.return_value

    response.json.side_effect = parse_after_raw_saved
    session = Mock()
    session.post.return_value = response
    original_phase9 = phase9_json.read_bytes()

    result = phase10.run_analysis_resume(
        phase9_json,
        source,
        prepared.output_dir,
        session=session,
        attempt_id=prepared.attempt_id,
        confirm_azure_request=True,
    )

    session.post.assert_called_once()
    assert result.analysis.review_reason == "SemanticReviewRequired"
    assert result.summary_pdf is None
    assert phase9_json.read_bytes() == original_phase9
    saved_request = json.loads(prepared.request_path.read_text(encoding="utf-8"))
    assert saved_request["status"] == "completed"
    assert saved_request["input_tokens"] == 10
    assert saved_request["output_tokens"] == 8
    saved_analysis = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert saved_analysis["attempt_id"] == prepared.attempt_id
    assert saved_analysis["meeting_analysis_requests"] == 1


def test_prepared_attempt_requires_explicit_live_authorization(tmp_path, monkeypatch):
    phase9_json, source, output_dir = _fixture(tmp_path, monkeypatch)
    prepared = phase10.prepare_analysis_attempt(
        phase9_json, source, output_dir
    )
    session = Mock()

    with pytest.raises(
        phase10.AnalysisResumePreflightError,
        match="ExplicitAzureAuthorizationRequired",
    ):
        phase10.run_analysis_resume(
            phase9_json,
            source,
            prepared.output_dir,
            session=session,
            attempt_id=prepared.attempt_id,
        )

    session.post.assert_not_called()
    request = json.loads(prepared.request_path.read_text(encoding="utf-8"))
    assert request["status"] == "prepared"


def test_prepared_attempt_preserves_invalid_raw_response_and_blocks_reuse(
    tmp_path, monkeypatch
):
    phase9_json, source, output_dir = _fixture(tmp_path, monkeypatch)
    prepared = phase10.prepare_analysis_attempt(
        phase9_json, source, output_dir
    )
    response = _analysis_response(valid=False)
    response.content = b'{"original":"invalid claims preserved"}'
    session = Mock()
    session.post.return_value = response

    result = phase10.run_analysis_resume(
        phase9_json,
        source,
        prepared.output_dir,
        session=session,
        attempt_id=prepared.attempt_id,
        confirm_azure_request=True,
    )

    assert result.analysis.review_reason == "OverviewEvidenceInvalid"
    assert prepared.raw_response_path.read_bytes() == response.content
    metadata = json.loads(prepared.request_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    second_session = Mock()
    with pytest.raises(
        phase10.AnalysisResumePreflightError,
        match="PreparedAttemptMismatchOrAlreadyUsed",
    ):
        phase10.run_analysis_resume(
            phase9_json,
            source,
            prepared.output_dir,
            session=second_session,
            attempt_id=prepared.attempt_id,
            confirm_azure_request=True,
        )
    second_session.post.assert_not_called()


def test_prepared_attempt_marks_uncertain_dispatch_and_never_retries(
    tmp_path, monkeypatch
):
    phase9_json, source, output_dir = _fixture(tmp_path, monkeypatch)
    prepared = phase10.prepare_analysis_attempt(
        phase9_json, source, output_dir
    )
    session = Mock()
    session.post.side_effect = phase10.requests.exceptions.Timeout

    with pytest.raises(phase10.MeetingAnalysisError):
        phase10.run_analysis_resume(
            phase9_json,
            source,
            prepared.output_dir,
            session=session,
            attempt_id=prepared.attempt_id,
            confirm_azure_request=True,
        )

    session.post.assert_called_once()
    metadata = json.loads(prepared.request_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "unknown"
    retry_session = Mock()
    with pytest.raises(
        phase10.AnalysisResumePreflightError,
        match="PreparedAttemptMismatchOrAlreadyUsed",
    ):
        phase10.run_analysis_resume(
            phase9_json,
            source,
            prepared.output_dir,
            session=retry_session,
            attempt_id=prepared.attempt_id,
            confirm_azure_request=True,
        )
    retry_session.post.assert_not_called()


def test_cli_requires_explicit_azure_authorization(tmp_path, monkeypatch):
    phase9_json, source, output_dir = _fixture(tmp_path, monkeypatch)
    preflight = Mock()
    runner = Mock()
    monkeypatch.setattr(phase10, "preflight_analysis_resume", preflight)
    monkeypatch.setattr(phase10, "run_analysis_resume", runner)

    result = main([
        "phase10-analysis-resume",
        "--phase9-json", str(phase9_json),
        "--summary-source", str(source),
        "--summary-output-dir", str(output_dir),
    ])

    assert result == 2
    preflight.assert_not_called()
    runner.assert_not_called()
