import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

import app.main as main_module
import app.phase10_7 as pilot
from app.config import AzureConfig
from app.meeting_analysis import (
    AnalysisItem,
    EvidenceReference,
    MeetingAnalysis,
)
from app.phase10_6 import load_review_bundle, review_and_maybe_publish
from app.phase10_7 import PilotBatchError
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
        transcription_deployment="transcribe",
        transcription_api_version="v1",
        chunk_duration_seconds=180,
        chunk_overlap_seconds=10,
    )


def _fixture(tmp_path: Path, *, alternates=True):
    recording_root = tmp_path / "Record File" / "CB"
    sara_root = tmp_path / "SaraAI"
    preview_root = tmp_path / "previews"
    review_root = tmp_path / "reviews"
    stage_root = preview_root / "phase10_7"
    analysis_root = tmp_path / "analysis"
    for path in (recording_root, sara_root, preview_root, review_root, analysis_root):
        path.mkdir(parents=True, exist_ok=True)
    rows = []
    for month in ("Sep26", "Oct26", "Nov26"):
        month_recordings = recording_root / month
        month_pdfs = sara_root / month
        month_recordings.mkdir()
        month_pdfs.mkdir()
        names = [f"meeting-{month}-short.mp3"]
        if alternates:
            names.append(f"meeting-{month}-long.mp3")
        for name in names:
            recording = month_recordings / name
            recording.write_bytes(
                f"fixture {month} {name} {'x' * (len(rows) + 1)}".encode("ascii")
            )
            pdf = month_pdfs / f"{Path(name).stem}.pdf"
            pdf.write_bytes(f"reference {name}".encode("ascii"))
            rows.append(
                {
                    "status": "MATCHED",
                    "recording_relative_path": f"{month}\\{name}",
                    "pdf_relative_path": f"{month}\\{pdf.name}",
                    "matching_evidence": "test fixture",
                }
            )
    pairs_csv = analysis_root / "recording_pdf_pairs.csv"
    with pairs_csv.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "status", "recording_relative_path", "pdf_relative_path",
                "matching_evidence",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    manifest_path = analysis_root / "phase10_7_pilot_manifest.json"
    report_path = analysis_root / "phase10_7_pilot_dry_run.md"
    return {
        "recording_root": recording_root,
        "sara_root": sara_root,
        "preview_root": preview_root,
        "review_root": review_root,
        "stage_root": stage_root,
        "analysis_root": analysis_root,
        "pairs_csv": pairs_csv,
        "manifest_path": manifest_path,
        "report_path": report_path,
    }


def _prepare(paths, monkeypatch, *, durations=None, max_requests=None):
    durations = durations or {}
    monkeypatch.setattr(
        pilot.AzureOpenAITranscriber,
        "_probe_duration",
        staticmethod(lambda path: durations.get(path.name, 60.0)),
    )
    return pilot.prepare_pilot_dry_run(
        manifest_path=paths["manifest_path"],
        report_path=paths["report_path"],
        stage_root=paths["stage_root"],
        recording_root=paths["recording_root"],
        sara_root=paths["sara_root"],
        pairs_csv=paths["pairs_csv"],
        preview_root=paths["preview_root"],
        review_root=paths["review_root"],
        max_requests=max_requests,
        config=_config(),
        api_version="2025-04-01-preview",
    )


def test_selection_uses_shortest_eligible_recording_in_each_month(tmp_path, monkeypatch):
    paths = _fixture(tmp_path)
    durations = {
        "meeting-Sep26-short.mp3": 351,
        "meeting-Sep26-long.mp3": 800,
        "meeting-Oct26-short.mp3": 476.7,
        "meeting-Oct26-long.mp3": 900,
        "meeting-Nov26-short.mp3": 1080.42,
        "meeting-Nov26-long.mp3": 1300,
    }

    manifest, blockers = _prepare(paths, monkeypatch, durations=durations)

    assert blockers == []
    assert [item["month"] for item in manifest["recordings"]] == [
        "Sep26", "Oct26", "Nov26"
    ]
    assert [Path(item["source"]).name for item in manifest["recordings"]] == [
        "meeting-Sep26-short.mp3",
        "meeting-Oct26-short.mp3",
        "meeting-Nov26-short.mp3",
    ]
    assert manifest["request_budget"] == {
        "per_run_limit": 19,
        "planned_total": 19,
    }


@pytest.mark.parametrize("reason", ["validated_transcript", "approved_summary"])
def test_selection_skips_recordings_with_prior_transcript_or_approval(
    tmp_path, monkeypatch, reason
):
    paths = _fixture(tmp_path)
    monkeypatch.setattr(
        pilot.AzureOpenAITranscriber,
        "_probe_duration",
        staticmethod(lambda _path: 60.0),
    )

    def artifact_status(source, _preview_root, _review_root):
        if source.name.endswith("-short.mp3"):
            return (
                [object()] if reason == "validated_transcript" else [],
                reason == "approved_summary",
            )
        return [], False

    monkeypatch.setattr(pilot, "_check_artifact_status", artifact_status)

    manifest, blockers = _prepare(paths, monkeypatch)

    assert blockers == []
    assert all(
        Path(recording["source"]).name.endswith("-long.mp3")
        for recording in manifest["recordings"]
    )


@pytest.mark.parametrize("missing", ["mp3", "pdf"])
def test_missing_month_pair_member_blocks_without_substitution(
    tmp_path, monkeypatch, missing
):
    paths = _fixture(tmp_path, alternates=False)
    recording = paths["recording_root"] / "Sep26" / "meeting-Sep26-short.mp3"
    pdf = paths["sara_root"] / "Sep26" / "meeting-Sep26-short.pdf"
    (recording if missing == "mp3" else pdf).unlink()
    monkeypatch.setattr(
        pilot.AzureOpenAITranscriber,
        "_probe_duration",
        staticmethod(lambda _path: 60.0),
    )

    with pytest.raises(PilotBatchError, match="NoEligibleRecordingForMonth:Sep26"):
        pilot.prepare_pilot_dry_run(
            manifest_path=paths["manifest_path"],
            report_path=paths["report_path"],
            stage_root=paths["stage_root"],
            recording_root=paths["recording_root"],
            sara_root=paths["sara_root"],
            pairs_csv=paths["pairs_csv"],
            preview_root=paths["preview_root"],
            review_root=paths["review_root"],
            config=_config(),
            api_version="2025-04-01-preview",
        )

    assert "No alternate month" in paths["report_path"].read_text(encoding="utf-8")
    assert not paths["manifest_path"].exists()


def test_duplicate_inventory_selection_is_rejected(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, alternates=False)
    with paths["pairs_csv"].open("a", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow([
            "MATCHED",
            "Sep26\\meeting-Sep26-short.mp3",
            "Sep26\\meeting-Sep26-short.pdf",
            "duplicate",
        ])
    monkeypatch.setattr(
        pilot.AzureOpenAITranscriber,
        "_probe_duration",
        staticmethod(lambda _path: 60.0),
    )

    with pytest.raises(PilotBatchError, match="DuplicateRecordingInInventory"):
        _prepare(paths, monkeypatch)


@pytest.mark.parametrize(
    ("duration", "expected"),
    [(180, 1), (181, 2), (350, 2), (351, 3), (895.56, 6), (1080.42, 7)],
)
def test_overlap_aware_chunk_count_matches_transcriber_algorithm(duration, expected):
    assert pilot.estimate_chunk_count(duration, 180, 10) == expected


def test_request_budget_is_checked_before_network_dispatch(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest = {"request_attempts": []}
    calls = []

    class Session:
        def post(self, url, **kwargs):
            calls.append((url, kwargs))
            return SimpleNamespace(status_code=200)

    recording = {"recording_relative_path": "Sep26\\meeting.mp3"}
    tracker = pilot._TrackedSession(
        manifest_path,
        manifest,
        recording,
        "transcription",
        "run-1",
        1,
        Session(),
    )

    tracker.post("https://offline.invalid/one")
    with pytest.raises(PilotBatchError, match="RequestBudgetExhausted"):
        tracker.post("https://offline.invalid/two")

    assert len(calls) == 1
    assert calls[0][1]["allow_redirects"] is False
    assert manifest["request_attempts"][0]["status"] == "response_received"


def test_interrupted_dispatches_become_unknown_and_require_reconciliation(
    tmp_path, monkeypatch
):
    paths = _fixture(tmp_path, alternates=False)
    manifest, _ = _prepare(paths, monkeypatch)
    attempt = {
        "attempt_id": "attempt-1",
        "run_id": "interrupted",
        "recording_relative_path": manifest["recordings"][0]["recording_relative_path"],
        "stage": "transcription",
        "status": "dispatching",
    }
    manifest["request_attempts"].append(attempt)

    blockers = pilot._recover_unfinished_attempts(manifest)

    assert attempt["status"] == "unknown"
    assert blockers and "request_outcome_unknown" in blockers[0]
    with pytest.raises(PilotBatchError, match="ExplicitRetryConfirmationRequired"):
        pilot.authorize_stage_retry(
            paths["manifest_path"],
            "attempt-1",
            confirmed=False,
        )


def _install_fake_pipeline(
    monkeypatch,
    *,
    fail_transcription=False,
    analysis_reason="SemanticReviewRequired",
):
    call_log = []

    class FakeSession:
        def mount(self, *_args, **_kwargs):
            pass

        def post(self, url, **kwargs):
            call_log.append(url)
            if fail_transcription:
                raise requests.exceptions.ReadTimeout("simulated timeout")
            return SimpleNamespace(status_code=200)

        def close(self):
            pass

    def fake_transcribe(self, source):
        self._session.post("https://offline.invalid/transcribe")
        return "ตรวจ Encoder Lot 123"

    def fake_clean(self, segments):
        self._session.post("https://offline.invalid/cleanup")
        text = segments[0].display_text
        segment = TranscriptSegment(None, None, None, text, text)
        return TranscriptDocument(
            (segment,),
            diarization_status="unavailable",
            cleanup_status="completed",
        )

    def fake_analyze(self, segments):
        self._session.post("https://offline.invalid/analysis")
        text = segments[0].display_text
        evidence = (EvidenceReference(1, text, 0, len(text)),)
        item = AnalysisItem(text, evidence)
        return MeetingAnalysis(
            overview=item,
            classifications=(),
            topics=(item,),
            key_issues=(item,),
            actions=(),
            dependencies=(),
            schedules=(),
            risks=(),
            closing_activities=(),
            status="review_required",
            review_reason=analysis_reason,
        )

    monkeypatch.setattr(
        pilot.AzureOpenAITranscriber,
        "transcribe",
        fake_transcribe,
    )
    monkeypatch.setattr(pilot.TranscriptCleanup, "clean", fake_clean)
    monkeypatch.setattr(pilot.GPTMeetingAnalyzer, "analyze", fake_analyze)
    return FakeSession, call_log


def test_resume_skips_completed_stages_and_analysis_stays_unapproved(
    tmp_path, monkeypatch
):
    paths = _fixture(tmp_path, alternates=False)
    manifest, blockers = _prepare(paths, monkeypatch)
    assert blockers == []
    monkeypatch.setattr(
        pilot.AzureOpenAITranscriber,
        "_probe_duration",
        staticmethod(lambda _path: 60.0),
    )
    fake_session, call_log = _install_fake_pipeline(monkeypatch)
    original_mp3_hashes = {
        item["source"]: hashlib.sha256(Path(item["source"]).read_bytes()).hexdigest()
        for item in manifest["recordings"]
    }
    original_pdf_hashes = {
        item["sara_pdf"]: hashlib.sha256(Path(item["sara_pdf"]).read_bytes()).hexdigest()
        for item in manifest["recordings"]
    }

    result = pilot.execute_pilot_batch(
        manifest_path=paths["manifest_path"],
        stage_root=paths["stage_root"],
        config=_config(),
        api_version="2025-04-01-preview",
        session_factory=fake_session,
    )
    first_count = len(call_log)
    resumed = pilot.execute_pilot_batch(
        manifest_path=paths["manifest_path"],
        stage_root=paths["stage_root"],
        config=_config(),
        api_version="2025-04-01-preview",
        session_factory=fake_session,
    )

    assert result["execution_status"] == "awaiting_human_review"
    assert first_count == 9
    assert len(call_log) == first_count
    assert resumed["execution_status"] == "awaiting_human_review"
    assert all(
        item["stages"]["analysis"]["status"] == "review_required"
        for item in resumed["recordings"]
    )
    assert all(
        item["stages"]["human_review"]["status"] == "pending"
        and item["stages"]["approved_pdf"]["status"] == "blocked_until_human_approval"
        for item in resumed["recordings"]
    )
    assert not list(paths["stage_root"].rglob("*_SARA_Approved.pdf"))
    assert all(
        hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
        for path, digest in original_mp3_hashes.items()
    )
    assert all(
        hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
        for path, digest in original_pdf_hashes.items()
    )

    selected = result["recordings"][0]
    analysis_path = Path(selected["stages"]["analysis"]["artifact"]["path"])
    transcript_path = Path(selected["stages"]["cleanup"]["artifact"]["path"])
    bundle = load_review_bundle(
        analysis_path,
        transcript_path,
        Path(selected["source"]),
        paths["review_root"],
    )
    report, pdf_path = review_and_maybe_publish(
        analysis_path=analysis_path,
        transcript_path=transcript_path,
        source=Path(selected["source"]),
        reviewer="pilot-reviewer",
        review_dir=paths["review_root"],
    )
    assert bundle.analysis.status == "review_required"
    assert "Decision: pending" in report
    assert pdf_path is None


def test_source_fingerprint_change_blocks_before_network_call(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, alternates=False)
    manifest, _ = _prepare(paths, monkeypatch)
    selected = Path(manifest["recordings"][0]["source"])
    selected.write_bytes(b"changed")
    fake_session, calls = _install_fake_pipeline(monkeypatch)

    with pytest.raises(PilotBatchError, match="source_or_reference_changed"):
        pilot.execute_pilot_batch(
            manifest_path=paths["manifest_path"],
            stage_root=paths["stage_root"],
            config=_config(),
            api_version="2025-04-01-preview",
            session_factory=fake_session,
        )
    assert calls == []


def test_uncertain_request_is_not_repeated_automatically(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, alternates=False)
    _prepare(paths, monkeypatch)
    monkeypatch.setattr(
        pilot.AzureOpenAITranscriber,
        "_probe_duration",
        staticmethod(lambda _path: 60.0),
    )
    fake_session, calls = _install_fake_pipeline(monkeypatch, fail_transcription=True)

    with pytest.raises(PilotBatchError, match="StageFailed:Sep26:transcription"):
        pilot.execute_pilot_batch(
            manifest_path=paths["manifest_path"],
            stage_root=paths["stage_root"],
            config=_config(),
            api_version="2025-04-01-preview",
            session_factory=fake_session,
        )
    assert len(calls) == 1
    manifest = json.loads(paths["manifest_path"].read_text(encoding="utf-8"))
    attempt = manifest["request_attempts"][0]
    assert attempt["status"] == "unknown"
    with pytest.raises(PilotBatchError, match="OperatorReconciliationRequired"):
        pilot.execute_pilot_batch(
            manifest_path=paths["manifest_path"],
            stage_root=paths["stage_root"],
            config=_config(),
            api_version="2025-04-01-preview",
            session_factory=fake_session,
        )
    assert len(calls) == 1
    recovered = pilot.authorize_stage_retry(
        paths["manifest_path"],
        attempt["attempt_id"],
        confirmed=True,
    )
    assert recovered["request_attempts"][0]["status"] == "retry_authorized"
    assert recovered["recordings"][0]["stages"]["transcription"]["retry_authorized"] is True


def test_stage_artifact_hash_change_blocks_without_network_call(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, alternates=False)
    _prepare(paths, monkeypatch)
    monkeypatch.setattr(
        pilot.AzureOpenAITranscriber,
        "_probe_duration",
        staticmethod(lambda _path: 60.0),
    )
    fake_session, calls = _install_fake_pipeline(monkeypatch)
    result = pilot.execute_pilot_batch(
        manifest_path=paths["manifest_path"],
        stage_root=paths["stage_root"],
        config=_config(),
        api_version="2025-04-01-preview",
        session_factory=fake_session,
    )
    raw_path = Path(
        result["recordings"][0]["stages"]["transcription"]["artifact"]["path"]
    )
    raw_path.write_text("changed", encoding="utf-8")
    prior_calls = len(calls)

    with pytest.raises(PilotBatchError, match="StageArtifactHashMismatch"):
        pilot.execute_pilot_batch(
            manifest_path=paths["manifest_path"],
            stage_root=paths["stage_root"],
            config=_config(),
            api_version="2025-04-01-preview",
            session_factory=fake_session,
        )
    assert len(calls) == prior_calls


def test_invalid_analysis_is_never_validated_or_automatically_retried(
    tmp_path, monkeypatch
):
    paths = _fixture(tmp_path, alternates=False)
    _prepare(paths, monkeypatch)
    monkeypatch.setattr(
        pilot.AzureOpenAITranscriber,
        "_probe_duration",
        staticmethod(lambda _path: 60.0),
    )
    fake_session, calls = _install_fake_pipeline(
        monkeypatch,
        analysis_reason="OverviewEvidenceInvalid",
    )
    result = pilot.execute_pilot_batch(
        manifest_path=paths["manifest_path"],
        stage_root=paths["stage_root"],
        config=_config(),
        api_version="2025-04-01-preview",
        session_factory=fake_session,
    )
    count = len(calls)

    resumed = pilot.execute_pilot_batch(
        manifest_path=paths["manifest_path"],
        stage_root=paths["stage_root"],
        config=_config(),
        api_version="2025-04-01-preview",
        session_factory=fake_session,
    )

    assert result["execution_status"] == "analysis_validation_failed"
    assert all(
        item["stages"]["analysis"]["status"] == "review_required"
        and item["stages"]["analysis"]["validation"] == "failed"
        and item["stages"]["human_review"]["status"] == "blocked_invalid_analysis"
        for item in resumed["recordings"]
    )
    assert len(calls) == count


def test_request_budget_below_plan_is_a_dry_run_blocker(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, alternates=False)
    manifest, blockers = _prepare(paths, monkeypatch, max_requests=1)

    assert manifest["request_budget"]["per_run_limit"] == 1
    assert any(blocker.startswith("request_budget_below_plan") for blocker in blockers)
    assert "request_budget_below_plan" in paths["report_path"].read_text(encoding="utf-8")


def test_cli_dry_run_makes_no_azure_or_sql_requests(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, alternates=False)
    monkeypatch.setattr(
        pilot.AzureOpenAITranscriber,
        "_probe_duration",
        staticmethod(lambda _path: 60.0),
    )
    monkeypatch.setattr(pilot, "get_azure_config", _config)
    monkeypatch.setattr(
        pilot,
        "get_azure_openai_api_version",
        lambda: "2025-04-01-preview",
    )
    monkeypatch.setattr(
        "requests.Session.post",
        lambda *_args, **_kwargs: pytest.fail("Azure request in dry-run"),
    )
    monkeypatch.setattr(
        main_module,
        "connect_sql",
        lambda *_args, **_kwargs: pytest.fail("SQL write/connection in dry-run"),
    )

    result = main_module.main(
        [
            "phase10-pilot-batch",
            "--dry-run",
            "--pilot-manifest", str(paths["manifest_path"]),
            "--pilot-report", str(paths["report_path"]),
            "--pilot-stage-root", str(paths["stage_root"]),
            "--pilot-recording-root", str(paths["recording_root"]),
            "--pilot-sara-root", str(paths["sara_root"]),
            "--pilot-pairs-csv", str(paths["pairs_csv"]),
            "--pilot-preview-root", str(paths["preview_root"]),
            "--pilot-review-root", str(paths["review_root"]),
        ]
    )

    assert result == 0
    assert paths["manifest_path"].is_file()
    assert paths["report_path"].is_file()
    assert not paths["stage_root"].exists()
