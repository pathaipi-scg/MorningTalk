from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path

import pytest
from pypdf import PdfReader

import app.phase10_6 as phase10_6
from app.file_manager import source_fingerprint
from app.main import main
from app.phase10_6 import (
    ReviewWorkflowError,
    evaluate_batch,
    generate_approved_pdf,
    load_review_bundle,
    record_review_decision,
)
from app.transcript import TranscriptSegment


@pytest.fixture(autouse=True)
def quiet_pypdf_legacy_warning(monkeypatch):
    import pypdf.filters

    monkeypatch.setattr(pypdf.filters, "JBIG2DEC_BINARY", None)


def _write_review_fixture(tmp_path: Path):
    source = tmp_path / "CB" / "Nov26" / "meeting.mp3"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"fixture recording")
    text = "ตรวจ Encoder Lot 123 และส่งอะไหล่"
    segment = TranscriptSegment(1, 2, None, text, text)
    transcript_path = tmp_path / "meeting.phase9.json"
    transcript = {
        "cleanup_status": "completed",
        "source_fingerprint": source_fingerprint(source),
        "raw_transcript": text,
        "cleaned_transcript": text,
        "segments": [asdict(segment)],
        "meeting_analysis": {"status": "review_required"},
    }
    transcript_path.write_text(
        json.dumps(transcript, ensure_ascii=False), encoding="utf-8"
    )
    evidence = [{"segment_id": 1, "start_char": 0, "end_char": len(text)}]
    analysis = {
        "overview": {"text": "ตรวจ Encoder Lot 123", "evidence": evidence},
        "classifications": [],
        "topics": [{"text": "ตรวจ Encoder Lot 123", "evidence": evidence}],
        "key_issues": [{"text": "ตรวจ Encoder Lot 123", "evidence": evidence}],
        "actions": [{"text": "ส่งอะไหล่", "evidence": evidence}],
        "dependencies": [],
        "schedules": [],
        "risks": [],
        "closing_activities": [],
    }
    analysis_path = tmp_path / "meeting.phase10.json"
    analysis_path.write_text(
        json.dumps(
            {
                "source_filename": source.name,
                "source_fingerprint": source_fingerprint(source),
                "phase9_transcript_json": str(transcript_path.resolve()),
                "phase9_json_sha256": hashlib.sha256(
                    transcript_path.read_bytes()
                ).hexdigest(),
                "meeting_analysis": analysis,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return source, transcript_path, analysis_path


def _bundle(tmp_path: Path):
    source, transcript, analysis = _write_review_fixture(tmp_path)
    return load_review_bundle(analysis, transcript, source, tmp_path / "reviews")


def _decide(bundle, item_id, decision="approve", **kwargs):
    return record_review_decision(
        bundle,
        reviewer="tester",
        item_id=item_id,
        decision=decision,
        confirmed=True,
        **kwargs,
    )


def test_batch_evaluation_pairs_inventory_and_reports_missing_transcript(tmp_path):
    recording_root = tmp_path / "recordings"
    sara_root = tmp_path / "sara"
    (recording_root / "CB" / "Sep26").mkdir(parents=True)
    sara_root.mkdir()
    (recording_root / "CB" / "Sep26" / "m.mp3").write_bytes(b"mp3")
    (sara_root / "m.pdf").write_bytes(b"reference")
    pairs = tmp_path / "pairs.csv"
    with pairs.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "status", "recording_relative_path", "pdf_relative_path",
                "matching_evidence",
            ),
        )
        writer.writeheader()
        writer.writerow({
            "status": "MATCHED",
            "recording_relative_path": "CB\\Sep26\\m.mp3",
            "pdf_relative_path": "m.pdf",
            "matching_evidence": "same basename",
        })
    result = evaluate_batch(
        recording_root=recording_root,
        sara_root=sara_root,
        pairs_csv=pairs,
        preview_root=tmp_path / "previews",
        review_root=tmp_path / "reviews",
        output_dir=tmp_path / "reports",
    )

    assert len(result.rows) == 1
    assert result.rows[0]["transcript_status"] == "transcription_required"
    assert result.rows[0]["next_step"] == "transcription_required"
    assert result.csv_path.is_file()
    assert "transcription_required" in result.markdown_path.read_text(encoding="utf-8")


def test_batch_evaluation_reports_analysis_needed_for_validated_transcript(tmp_path):
    source, transcript_path, _analysis_path = _write_review_fixture(tmp_path)
    recording_root = tmp_path / "corpus"
    sara_root = tmp_path / "references"
    recording = recording_root / "Nov26" / source.name
    recording.parent.mkdir(parents=True)
    recording.write_bytes(source.read_bytes())
    transcript_payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    transcript_payload["source_fingerprint"] = source_fingerprint(recording)
    transcript_path.write_text(
        json.dumps(transcript_payload, ensure_ascii=False), encoding="utf-8"
    )
    sara_root.mkdir()
    (sara_root / "meeting.pdf").write_bytes(b"reference")
    pairs = tmp_path / "pairs.csv"
    pairs.write_text(
        "status,recording_relative_path,pdf_relative_path,matching_evidence\n"
        "MATCHED,Nov26\\meeting.mp3,meeting.pdf,fixture\n",
        encoding="utf-8",
    )
    preview_root = tmp_path / "previews"
    preview_root.mkdir()
    (preview_root / transcript_path.name).write_bytes(transcript_path.read_bytes())

    result = evaluate_batch(
        recording_root=recording_root,
        sara_root=sara_root,
        pairs_csv=pairs,
        preview_root=preview_root,
        review_root=tmp_path / "reviews",
        output_dir=tmp_path / "reports",
    )

    row = result.rows[0]
    assert row["validated_transcript_available"] == "true"
    assert row["meeting_analysis_available"] == "false"
    assert row["next_step"] == "meeting_analysis_required"


def test_review_requires_explicit_confirmation_and_logs_decisions(tmp_path):
    bundle = _bundle(tmp_path)
    with pytest.raises(ReviewWorkflowError, match="ExplicitHumanConfirmationRequired"):
        record_review_decision(
            bundle,
            reviewer="tester",
            item_id="overview:1",
            decision="approve",
            confirmed=False,
        )

    state = _decide(bundle, "overview:1")
    assert state["items"]["overview:1"]["decision"] == "approved"
    assert state["audit"][-1]["action"] == "approve"
    assert state["audit"][-1]["reviewer"] == "tester"


def test_edit_is_revalidated_and_returns_to_pending(tmp_path):
    bundle = _bundle(tmp_path)
    with pytest.raises(ReviewWorkflowError):
        _decide(
            bundle,
            "topics:1",
            "edit",
            edits={"text": "พบปัญหา 999"},
        )
    state = _decide(
        bundle,
        "topics:1",
        "edit",
        edits={"text": "ตรวจ Encoder Lot 123"},
    )
    assert state["items"]["topics:1"]["decision"] == "pending"
    assert state["items"]["topics:1"]["edited"]["text"] == "ตรวจ Encoder Lot 123"
    assert state["audit"][-1]["action"] == "edit_revalidated"


def test_stale_source_or_transcript_is_rejected_before_review(tmp_path):
    bundle = _bundle(tmp_path)
    bundle.source.write_bytes(b"changed recording")
    with pytest.raises(ReviewWorkflowError, match="SourceFingerprintMismatch"):
        _decide(bundle, "overview:1")


def test_transcript_hash_change_is_rejected_before_review(tmp_path):
    bundle = _bundle(tmp_path)
    bundle.transcript.path.write_text("{}", encoding="utf-8")
    with pytest.raises(ReviewWorkflowError, match="TranscriptHashMismatch"):
        _decide(bundle, "overview:1")


def test_approved_pdf_requires_every_decision_and_excludes_rejected_items(
    tmp_path, monkeypatch
):
    bundle = _bundle(tmp_path)
    state = _decide(bundle, "overview:1")
    state = _decide(bundle, "topics:1")
    state = _decide(bundle, "key_issues:1")
    state = _decide(
        bundle, "actions:1", "reject", reason="Not sufficiently clear."
    )
    assert state["status"] == "review_approved"

    preview_root = tmp_path / "previews"
    monkeypatch.setattr(phase10_6, "DEFAULT_PREVIEW_ROOT", preview_root)
    output_dir = preview_root / "approved-meeting"
    pdf = generate_approved_pdf(
        bundle,
        state,
        output_dir,
        confirmed=True,
    )
    reader = PdfReader(str(pdf))
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)

    assert pdf.is_file()
    assert len(reader.pages) >= 1
    assert "ตรวจ Encoder Lot 123" in extracted
    assert "ส่งอะไหล่" not in extracted
    assert state["items"]["actions:1"]["decision"] == "rejected"

    with pytest.raises(ReviewWorkflowError, match="ApprovedPdfDirectoryAlreadyExists"):
        generate_approved_pdf(bundle, state, output_dir, confirmed=True)


def test_approved_pdf_cannot_be_written_beside_production_source(tmp_path):
    bundle = _bundle(tmp_path)
    state = _decide(bundle, "overview:1")
    state = _decide(bundle, "topics:1")
    state = _decide(bundle, "key_issues:1")
    _decide(bundle, "actions:1", "reject", reason="Not sufficiently clear.")

    with pytest.raises(ReviewWorkflowError, match="ApprovedPdfOutsidePreviewRoot"):
        generate_approved_pdf(
            bundle,
            state,
            bundle.source.parent / "approved",
            confirmed=True,
        )
    assert not (bundle.source.parent / "approved").exists()


def test_review_audit_tampering_is_rejected(tmp_path):
    bundle = _bundle(tmp_path)
    state = _decide(bundle, "overview:1")
    state["audit"][0]["reviewer"] = "modified"
    bundle.review_state_path.write_text(
        json.dumps(state, ensure_ascii=False), encoding="utf-8"
    )

    with pytest.raises(ReviewWorkflowError, match="ReviewAuditTampered"):
        _decide(bundle, "topics:1")


def test_batch_cli_stays_offline(tmp_path, monkeypatch):
    recording_root = tmp_path / "recordings"
    sara_root = tmp_path / "sara"
    recording_root.mkdir()
    sara_root.mkdir()
    pairs = tmp_path / "pairs.csv"
    pairs.write_text(
        "status,recording_relative_path,pdf_relative_path,matching_evidence\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "app.main.connect_sql",
        lambda *_args, **_kwargs: pytest.fail("SQL must not be used"),
    )
    monkeypatch.setattr(
        "app.meeting_analysis.GPTMeetingAnalyzer.analyze",
        lambda *_args, **_kwargs: pytest.fail("Azure must not be used"),
    )
    monkeypatch.setattr(
        "app.transcriber.AzureOpenAITranscriber.transcribe",
        lambda *_args, **_kwargs: pytest.fail("Transcription must not be used"),
    )
    monkeypatch.setattr(
        "app.transcript_cleanup.TranscriptCleanup.clean",
        lambda *_args, **_kwargs: pytest.fail("Cleanup must not be used"),
    )

    result = main([
        "phase10-batch-evaluate",
        "--recording-root", str(recording_root),
        "--sara-root", str(sara_root),
        "--pairs-csv", str(pairs),
        "--evaluation-dir", str(tmp_path / "reports"),
        "--preview-root", str(tmp_path / "previews"),
        "--review-root", str(tmp_path / "reviews"),
    ])

    assert result == 0
    assert (tmp_path / "reports" / "phase10_6_batch_evaluation.csv").is_file()
