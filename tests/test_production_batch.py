import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

import app.benchmark as benchmark
import app.production_batch as batch
from app.config import AzureConfig
from app.meeting_analysis import AnalysisItem, EvidenceReference, MeetingAnalysis
from app.transcript import TranscriptDocument, TranscriptSegment


def _config():
    return AzureConfig(
        text_endpoint=(
            "https://unit-test.openai.azure.com/openai/responses"
            "?api-version=2025-04-01-preview"
        ),
        text_api_key="not-a-real-key",
        text_deployment="gpt-5.4-mini",
        transcription_endpoint="https://unit-test.openai.azure.com",
        transcription_api_key="not-a-real-key",
        transcription_deployment="gpt-4o-transcribe",
        transcription_api_version="2025-03-01-preview",
        chunk_duration_seconds=180,
        chunk_overlap_seconds=10,
    )


def _batch(
    tmp_path,
    monkeypatch,
    *,
    batch_id="offline",
    include_earlier=False,
    include_other=False,
):
    recording_root = tmp_path / "Record File"
    source = recording_root / "CB" / "Nov26" / "meeting-20261107-0817.mp3"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"fixture audio bytes")
    if include_earlier:
        (source.parent / "meeting-20261102-0838.mp3").write_bytes(
            b"earlier fixture audio"
        )
    if include_other:
        (source.parent / "meeting-20261103-0813.mp3").write_bytes(
            b"other fixture audio"
        )
    transcript_root = tmp_path / "Transcript"
    transcript_root.mkdir()
    sara_root = tmp_path / "SaraAI"
    sara_root.mkdir()
    preview_root = tmp_path / "private" / "previews"
    review_root = tmp_path / "private" / "reviews"
    private_root = tmp_path / "private" / "batches"
    original_discover = benchmark.discover_inventory

    def discover(*args, **kwargs):
        kwargs["duration_probe"] = lambda _source: 60.0
        return original_discover(*args, **kwargs)

    monkeypatch.setattr(batch, "discover_inventory", discover)
    monkeypatch.setattr(
        batch.AzureOpenAITranscriber,
        "_probe_duration",
        staticmethod(lambda _source: 60.0),
    )
    manifest_path, manifest = batch.prepare_production_batch(
        batch_id=batch_id,
        recording_root=recording_root,
        transcript_root=transcript_root,
        sara_root=sara_root,
        preview_root=preview_root,
        review_root=review_root,
        private_root=private_root,
        config=_config(),
        api_version="2025-04-01-preview",
    )
    return manifest_path, manifest, source


def test_preflight_is_offline_and_records_exact_request_plan(tmp_path, monkeypatch):
    manifest_path, manifest, source = _batch(tmp_path, monkeypatch)

    assert manifest["execution_status"] == "preflight_ready"
    assert len(manifest["recordings"]) == 1
    recording = manifest["recordings"][0]
    assert recording["source"] == str(source.resolve())
    assert recording["expected_audio_chunks"] == 1
    assert recording["expected_new_requests"] == 3
    assert manifest["request_budget"]["planned_total"] == 3
    assert manifest["request_attempts"] == []
    assert not Path(recording["outputs"]["transcript_txt"]).exists()
    assert not Path(recording["outputs"]["full_transcript_pdf"]).exists()
    assert not Path(recording["outputs"]["summary_draft_pdf"]).exists()
    assert manifest_path.is_file()
    assert _config().text_api_key not in manifest_path.read_text(encoding="utf-8")
    assert manifest["configuration"]["transcription_prompt_enabled"] is False


def test_prompt_fingerprint_tracks_explicit_opt_in(monkeypatch):
    config = replace(
        _config(),
        transcription_api_version="2025-04-01-preview",
    )
    monkeypatch.delenv("MORNINGTALK_ENABLE_TRANSCRIPTION_PROMPT", raising=False)
    prompt_disabled = batch._stage_config_fingerprints(
        config, "2025-04-01-preview", ""
    )
    monkeypatch.setenv("MORNINGTALK_ENABLE_TRANSCRIPTION_PROMPT", "true")
    prompt_enabled = batch._stage_config_fingerprints(
        config, "2025-04-01-preview", ""
    )

    assert prompt_enabled["transcription"] != prompt_disabled["transcription"]


def test_unsupported_prompt_opt_in_fails_preflight_before_network(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("MORNINGTALK_ENABLE_TRANSCRIPTION_PROMPT", "true")

    with pytest.raises(ValueError, match="not enabled for this documented API version"):
        _batch(tmp_path, monkeypatch)


def test_normalization_feature_changes_cleanup_fingerprint(monkeypatch):
    monkeypatch.delenv("MORNINGTALK_ENABLE_TERMINOLOGY_NORMALIZATION", raising=False)
    disabled = batch._stage_config_fingerprints(_config(), "2025-04-01-preview", "")
    monkeypatch.setenv("MORNINGTALK_ENABLE_TERMINOLOGY_NORMALIZATION", "true")
    enabled = batch._stage_config_fingerprints(_config(), "2025-04-01-preview", "")

    assert enabled["cleanup"] != disabled["cleanup"]


def test_pilot_recording_is_first_in_the_execution_manifest(tmp_path, monkeypatch):
    _manifest_path, manifest, _source = _batch(
        tmp_path, monkeypatch, include_earlier=True
    )

    assert manifest["recordings"][0]["source"].endswith(
        "meeting-20261107-0817.mp3"
    )


def test_uncertain_prior_request_blocks_resume_before_network(tmp_path, monkeypatch):
    manifest_path, manifest, _source = _batch(tmp_path, monkeypatch)
    manifest["request_attempts"].append(
        {
            "attempt_id": "uncertain",
            "run_id": "previous",
            "recording_relative_path": "CB/Nov26/meeting-20261107-0817.mp3",
            "stage": "transcription",
            "status": "unknown",
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    class NoNetwork:
        def __init__(self):
            raise AssertionError("network session must not be constructed")

    with pytest.raises(batch.ProductionBatchError, match="RequestOutcomeNeedsReconciliation"):
        batch.execute_production_batch(
            manifest_path,
            config=_config(),
            api_version="2025-04-01-preview",
            session_factory=NoNetwork,
        )


def test_interrupted_request_is_marked_unknown_and_reported(tmp_path, monkeypatch):
    manifest_path, manifest, _source = _batch(
        tmp_path, monkeypatch, batch_id="interrupted"
    )
    recording = manifest["recordings"][0]
    response_path = tmp_path / "response.bin"
    response_path.write_bytes(b'{"id":"response"}')
    manifest["paths"]["response_root"] = str(tmp_path / "responses")
    manifest["request_attempts"] = [
        {
            "attempt_id": "received",
            "run_id": "interrupted-run",
            "recording_relative_path": recording["recording_relative_path"],
            "stage": "transcription",
            "status": "response_received",
            "raw_response_path": str(response_path),
            "raw_response_sha256": batch._sha256_file(response_path),
        },
        {
            "attempt_id": "dispatching",
            "run_id": "interrupted-run",
            "recording_relative_path": recording["recording_relative_path"],
            "stage": "transcription",
            "status": "dispatching",
        },
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    reconciled = batch.reconcile_interrupted_batch(manifest_path)

    assert reconciled["execution_status"] == "blocked_unknown_outcome"
    assert reconciled["request_attempts"][0]["status"] == "response_received"
    assert reconciled["request_attempts"][1]["status"] == "unknown"
    assert reconciled["recordings"][0]["status"] == "blocked"
    assert reconciled["recordings"][0]["stages"]["transcription"]["status"] == "unknown"
    assert Path(reconciled["paths"]["report_path"]).is_file()


def test_unknown_request_report_handles_missing_response_metadata(
    tmp_path, monkeypatch
):
    manifest_path, manifest, _source = _batch(tmp_path, monkeypatch)
    recording = manifest["recordings"][0]
    attempt = {
        "attempt_id": "unknown-without-response",
        "run_id": "prior-run",
        "recording_relative_path": recording["recording_relative_path"],
        "stage": "transcription",
        "status": "unknown",
    }
    manifest["request_attempts"] = [attempt]

    report_path = batch._write_final_report(manifest_path, manifest)
    report = report_path.read_text(encoding="utf-8")

    assert "HTTP status: not recorded; raw response: not recorded." in report
    assert "no HTTP status or raw response was retained" not in report
    assert manifest["request_attempts"][0] == attempt


def test_unknown_request_report_includes_retained_response_information(
    tmp_path, monkeypatch
):
    manifest_path, manifest, _source = _batch(tmp_path, monkeypatch)
    recording = manifest["recordings"][0]
    response_path = tmp_path / "retained-response.json"
    response_path.write_text('{"result":"saved"}', encoding="utf-8")
    attempt = {
        "attempt_id": "unknown-with-response",
        "run_id": "prior-run",
        "recording_relative_path": recording["recording_relative_path"],
        "stage": "transcription",
        "status": "unknown",
        "http_status": 502,
        "raw_response_path": str(response_path),
    }
    manifest["request_attempts"] = [attempt]

    report_path = batch._write_final_report(manifest_path, manifest)
    report = report_path.read_text(encoding="utf-8")

    assert f"HTTP status: 502; raw response: retained at `{response_path}`." in report
    assert manifest["request_attempts"][0] == attempt


def test_saved_responses_are_recovered_without_retrying_unknown_chunk(
    tmp_path, monkeypatch
):
    manifest_path, manifest, _source = _batch(tmp_path, monkeypatch)
    recording = manifest["recordings"][0]
    recording["expected_audio_chunks"] = 7
    response_attempts = []
    for index in (1, 2):
        body = json.dumps(
            {"text": f"chunk {index}", "usage": {"input_tokens": 2}},
        ).encode("utf-8")
        raw_path = tmp_path / f"response-{index}.bin"
        raw_path.write_bytes(body)
        response_attempts.append(
            {
                "attempt_id": f"response-{index}",
                "recording_relative_path": recording["recording_relative_path"],
                "stage": "transcription",
                "chunk_index": index,
                "status": "response_received",
                "http_status": 200,
                "raw_response_path": str(raw_path),
                "raw_response_sha256": batch._sha256_file(raw_path),
                "raw_response_bytes": len(body),
            }
        )
    unknown_attempt = {
        "attempt_id": "unknown-3",
        "recording_relative_path": recording["recording_relative_path"],
        "stage": "transcription",
        "chunk_index": 3,
        "status": "unknown",
    }
    manifest["request_attempts"] = [*response_attempts, unknown_attempt]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(batch, "_write_final_report", lambda *_args: None)

    artifact = batch.recover_saved_transcription_responses(
        manifest_path,
        recording["recording_relative_path"],
    )
    recovered = json.loads(Path(artifact["path"]).read_text(encoding="utf-8"))
    saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert recovered["received_chunk_indexes"] == [1, 2]
    assert recovered["unknown_chunk_indexes"] == [3]
    assert recovered["not_received_chunk_indexes"] == [4, 5, 6, 7]
    assert [item["text"] for item in recovered["chunks"]] == ["chunk 1", "chunk 2"]
    assert saved_manifest["request_attempts"][-1]["status"] == "unknown"


def test_unknown_record_is_isolated_and_pilot_runs_first(
    tmp_path, monkeypatch
):
    manifest_path, manifest, _source = _batch(
        tmp_path,
        monkeypatch,
        batch_id="independent",
        include_earlier=True,
        include_other=True,
    )
    earlier = next(
        item for item in manifest["recordings"]
        if item["source"].endswith("meeting-20261102-0838.mp3")
    )
    earlier["status"] = "blocked"
    earlier["stages"]["transcription"]["status"] = "unknown"
    manifest["request_attempts"] = [
        {
            "attempt_id": "unknown-prior",
            "run_id": "previous",
            "recording_relative_path": earlier["recording_relative_path"],
            "stage": "transcription",
            "status": "unknown",
        }
    ]
    manifest["recordings"].reverse()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    run_order = []

    def fake_process(_path, _manifest, recording, *_args):
        run_order.append(Path(recording["source"]).stem)
        recording["status"] = "completed"

    class NoNetwork:
        def mount(self, *_args, **_kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(batch, "_process_recording", fake_process)
    monkeypatch.setattr(batch, "_write_final_report", lambda *_args: None)

    result = batch.execute_production_batch(
        manifest_path,
        config=_config(),
        api_version="2025-04-01-preview",
        session_factory=NoNetwork,
    )

    assert run_order == ["meeting-20261107-0817", "meeting-20261103-0813"]
    nov_2 = next(
        item for item in result["recordings"]
        if Path(item["source"]).name == "meeting-20261102-0838.mp3"
    )
    nov_7 = next(
        item for item in result["recordings"]
        if Path(item["source"]).name == "meeting-20261107-0817.mp3"
    )
    assert nov_2["status"] == "blocked"
    assert nov_7["status"] == "completed"
    assert result["execution_status"] == "completed_with_blockers"
    assert len(result["request_attempts"]) == 1
    assert result["request_attempts"][0]["status"] == "unknown"
    assert result["configuration"]["transcription_prompt_supported_by_code"] is False
    assert "configuration_history" not in result
    assert result["request_budget"]["active_run_limit"] == (
        result["request_budget"]["planned_total"] - 1
    )


def test_runtime_blocker_does_not_stop_independent_recordings(
    tmp_path, monkeypatch
):
    manifest_path, _manifest, _source = _batch(
        tmp_path, monkeypatch, batch_id="isolated-failure", include_other=True
    )
    run_order = []

    def fake_process(_path, _manifest, recording, *_args):
        name = Path(recording["source"]).stem
        run_order.append(name)
        if name == "meeting-20261107-0817":
            raise batch.ProductionBatchError("AnalysisStageFailed:pilot")
        recording["status"] = "completed"

    class NoNetwork:
        def mount(self, *_args, **_kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(batch, "_process_recording", fake_process)
    monkeypatch.setattr(batch, "_write_final_report", lambda *_args: None)

    result = batch.execute_production_batch(
        manifest_path,
        config=_config(),
        api_version="2025-04-01-preview",
        session_factory=NoNetwork,
    )

    assert run_order == ["meeting-20261107-0817", "meeting-20261103-0813"]
    pilot = next(
        item for item in result["recordings"]
        if Path(item["source"]).stem == "meeting-20261107-0817"
    )
    other = next(
        item for item in result["recordings"]
        if Path(item["source"]).stem == "meeting-20261103-0813"
    )
    assert pilot["status"] == "blocked"
    assert pilot["blocking_error"] == "AnalysisStageFailed:pilot"
    assert other["status"] == "completed"
    assert result["execution_status"] == "completed_with_blockers"
    assert result["blocking_errors"] == [
        {
            "recording_relative_path": pilot["recording_relative_path"],
            "reason": "AnalysisStageFailed:pilot",
        }
    ]


def test_cleanup_review_does_not_replace_existing_user_outputs(
    tmp_path, monkeypatch
):
    manifest_path, manifest, _source = _batch(tmp_path, monkeypatch)
    recording = manifest["recordings"][0]
    original_outputs = {}
    for key in ("transcript_txt", "full_transcript_pdf", "summary_draft_pdf"):
        output = Path(recording["outputs"][key])
        output.parent.mkdir(parents=True, exist_ok=True)
        original_outputs[output] = f"original-{key}".encode()
        output.write_bytes(original_outputs[output])

    phase9_path = tmp_path / "phase9.json"
    phase9_path.write_text("{}", encoding="utf-8")
    recording["phase9_artifact"] = {
        "path": str(phase9_path),
        "sha256": batch._file_hash(phase9_path),
    }
    recording["stages"]["transcription"]["status"] = "completed"
    recording["stages"]["cleanup"]["status"] = "review_required"

    def review_required(*_args):
        raise batch.ReviewWorkflowError("CleanupRequiresReview")

    monkeypatch.setattr(batch, "load_validated_transcript", review_required)
    monkeypatch.setattr(
        batch,
        "_publish_outputs",
        lambda *_args, **_kwargs: pytest.fail(
            "User-facing outputs must not be published before cleanup validation."
        ),
    )

    batch._process_recording(
        manifest_path,
        manifest,
        recording,
        _config(),
        "2025-04-01-preview",
        session=None,
        run_id="review-run",
    )

    assert recording["status"] == "review_required"
    assert recording["output_verification"]["status"] == (
        "not_published_cleanup_review_required"
    )
    assert {
        path: path.read_bytes() for path in original_outputs
    } == original_outputs


def test_unconfirmed_terminology_blocks_cleanup_and_never_dispatches_request(
    tmp_path, monkeypatch,
):
    manifest_path, manifest, source = _batch(tmp_path, monkeypatch)
    recording = manifest["recordings"][0]
    monkeypatch.setenv("MORNINGTALK_ENABLE_TERMINOLOGY_NORMALIZATION", "true")
    raw_text = "คำที่ยังยืนยันไม่ได้ สีแปลนแผ่นข้าว"
    raw_path = batch._stage_path(manifest, recording, "raw")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        json.dumps(
            {
                "source_filename": source.name,
                "source_fingerprint": recording["source_fingerprint"],
                "source_sha256": recording["source_sha256"],
                "raw_transcript": raw_text,
                "status": "completed",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    raw_artifact = {
        "path": str(raw_path),
        "sha256": batch._file_hash(raw_path),
    }
    recording["stages"]["transcription"]["status"] = "completed"
    recording["stages"]["transcription"]["artifact"] = raw_artifact
    session = Mock()

    batch._process_recording(
        manifest_path,
        manifest,
        recording,
        _config(),
        "2025-04-01-preview",
        session=session,
        run_id="offline-terminology-test",
    )

    normalization = json.loads(
        Path(recording["normalization_artifact"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    phase9 = json.loads(
        Path(recording["phase9_artifact"]["path"]).read_text(encoding="utf-8")
    )
    assert normalization["raw_transcript"] == raw_text
    assert normalization["normalized_transcript"] == raw_text
    assert normalization["normalization_review_findings"]
    assert phase9["raw_transcript"] == raw_text
    assert phase9["normalized_transcript"] == raw_text
    assert phase9["cleaned_transcript"] == raw_text
    assert phase9["cleanup_status"] == "review_required"
    assert recording["stages"]["analysis"]["status"] == (
        "blocked_until_cleanup_review"
    )
    assert recording["output_verification"]["status"] == (
        "not_published_cleanup_review_required"
    )
    assert manifest["request_attempts"] == []
    session.post.assert_not_called()
    for key in ("transcript_txt", "full_transcript_pdf", "summary_draft_pdf"):
        assert not Path(recording["outputs"][key]).exists()


def test_batch_publication_stages_all_outputs_and_preserves_approved_summary(
    tmp_path,
    monkeypatch,
):
    _manifest_path, manifest, source = _batch(tmp_path, monkeypatch)
    recording = manifest["recordings"][0]
    monkeypatch.setattr(batch, "DEFAULT_PREVIEW_ROOT", tmp_path / "private" / "previews")
    monkeypatch.setattr(batch, "DEFAULT_TEMP_ROOT", tmp_path / "private" / "temp")
    outputs = {
        key: Path(recording["outputs"][key])
        for key in ("transcript_txt", "full_transcript_pdf", "summary_draft_pdf")
    }
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"existing output")
    approved = Path(recording["outputs"]["approved_summary_pdf"])
    approved.write_bytes(b"approved summary")
    document = TranscriptDocument(
        (
            TranscriptSegment(1, 2, "Speaker A", "ตรวจสอบ Encoder"),
            TranscriptSegment(2, 3, "Speaker B", "ยืนยัน Lot 123"),
        ),
        "available",
        "completed",
    )
    analysis = MeetingAnalysis(
        AnalysisItem(
            "ตรวจสอบ Encoder",
            (EvidenceReference(1, "ตรวจสอบ Encoder"),),
        ),
        (), (), (), (), (), (), "validated",
    )

    result = batch._publish_outputs(
        recording,
        source,
        document,
        analysis,
        overwrite_draft=True,
    )

    assert result["status"] == "verified"
    assert all(path.read_bytes() != b"existing output" for path in outputs.values())
    assert approved.read_bytes() == b"approved summary"
    staging = Path(result["staging_directory"])
    assert (staging / outputs["transcript_txt"].name).is_file()
    assert (staging / outputs["full_transcript_pdf"].name).is_file()
    assert (staging / outputs["summary_draft_pdf"].name).is_file()
    assert Path(result["rollback_backup_directory"]).is_dir()


def test_batch_staged_validation_failure_preserves_existing_output_set(
    tmp_path,
    monkeypatch,
):
    _manifest_path, manifest, source = _batch(tmp_path, monkeypatch)
    recording = manifest["recordings"][0]
    preview_root = tmp_path / "private" / "previews"
    monkeypatch.setattr(batch, "DEFAULT_PREVIEW_ROOT", preview_root)
    monkeypatch.setattr(batch, "DEFAULT_TEMP_ROOT", tmp_path / "private" / "temp")
    output_paths = {
        Path(recording["outputs"][key]): b"original-" + key.encode()
        for key in ("transcript_txt", "full_transcript_pdf", "summary_draft_pdf")
    }
    for path, content in output_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    document = TranscriptDocument(
        (TranscriptSegment(1, 2, "Speaker A", "ตรวจสอบ Encoder"),),
        "available",
        "completed",
    )
    analysis = MeetingAnalysis(
        AnalysisItem(
            "ตรวจสอบ Encoder",
            (EvidenceReference(1, "ตรวจสอบ Encoder"),),
        ),
        (), (), (), (), (), (), "validated",
    )
    verify_pdf = batch._verify_pdf

    def fail_staged_summary(path, *args, **kwargs):
        if path.is_relative_to(preview_root) and path.name.endswith("_Summary_Draft.pdf"):
            raise batch.ProductionBatchError("SimulatedSummaryValidationFailure")
        return verify_pdf(path, *args, **kwargs)

    monkeypatch.setattr(batch, "_verify_pdf", fail_staged_summary)
    with pytest.raises(
        batch.ProductionBatchError,
        match="SimulatedSummaryValidationFailure",
    ):
        batch._publish_outputs(
            recording,
            source,
            document,
            analysis,
            overwrite_draft=True,
        )

    assert {path: path.read_bytes() for path in output_paths} == output_paths


def test_final_report_separates_batch_totals_from_latest_run(
    tmp_path, monkeypatch
):
    manifest_path, manifest, _source = _batch(tmp_path, monkeypatch)
    relative_path = manifest["recordings"][0]["recording_relative_path"]
    manifest["active_run_id"] = "latest-run"
    manifest["recordings"][0]["stages"]["analysis"]["status"] = (
        "blocked_until_cleanup_review"
    )
    manifest["recordings"][0]["output_verification"] = {
        "summary_draft_pdf": {"path": "review-notice.pdf"}
    }
    manifest["request_attempts"] = [
        {
            "attempt_id": "old-transcription",
            "run_id": "prior-run",
            "recording_relative_path": relative_path,
            "stage": "transcription",
            "status": "completed",
        },
        {
            "attempt_id": "new-transcription",
            "run_id": "latest-run",
            "recording_relative_path": relative_path,
            "stage": "transcription",
            "status": "completed",
        },
        {
            "attempt_id": "new-cleanup",
            "run_id": "latest-run",
            "recording_relative_path": relative_path,
            "stage": "cleanup",
            "status": "completed",
        },
    ]

    report_path = batch._write_final_report(manifest_path, manifest)
    report = report_path.read_text(encoding="utf-8")

    assert "Total request attempt records in batch: 3" in report
    assert "Latest execution request attempts: 2 (transcription 1, cleanup 1, analysis 0)." in report
    assert "Summary Draft (review-required notice only; meeting analysis was not generated)" in report
