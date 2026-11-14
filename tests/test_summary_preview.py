import json
from dataclasses import asdict
from pathlib import Path

import pytest
from pypdf import PdfReader

from app.file_manager import source_fingerprint
import app.file_manager as file_manager
from app.meeting_analysis import AnalysisItem, EvidenceReference, MeetingAnalysis
from app.main import main
from app.pdf_export import write_meeting_summary_pdf
from app.summary_preview import (
    SummaryPreviewNotReady,
    write_sara_layout_demo,
    write_sara_summary_preview,
)
from app.transcript import TranscriptSegment


@pytest.fixture(autouse=True)
def quiet_pypdf_legacy_warning(monkeypatch):
    import pypdf.filters

    monkeypatch.setattr(pypdf.filters, "JBIG2DEC_BINARY", None)


def _write_phase9_json(tmp_path: Path, source: Path, *, analysis_status="validated"):
    segment = TranscriptSegment(
        1.0,
        2.0,
        "Speaker A",
        "ตรวจสอบ Encoder Lot 123",
        "ตรวจสอบ Encoder Lot 123",
    )
    payload = {
        "raw_transcript": segment.raw_text,
        "cleaned_transcript": segment.cleaned_text,
        "cleanup_status": "completed",
        "segments": [asdict(segment)],
        "source_fingerprint": source_fingerprint(source),
        "meeting_analysis": {
            "overview": {
                "text": "มีการตรวจสอบ Encoder Lot 123",
                "evidence": [{
                    "segment_id": 1,
                    "quote": "ตรวจสอบ Encoder Lot 123",
                }],
            },
            "classifications": [{
                "text": "การตรวจสอบเครื่องจักร",
                "evidence": [{
                    "segment_id": 1,
                    "quote": "ตรวจสอบ Encoder Lot 123",
                }],
            }],
            "key_issues": [{
                "text": "มีการระบุ Encoder Lot 123 ที่ต้องตรวจสอบ",
                "evidence": [{
                    "segment_id": 1,
                    "quote": "ตรวจสอบ Encoder Lot 123",
                }],
            }],
            "actions": [],
            "dependencies": [],
            "schedules": [],
            "risks": [],
            "status": analysis_status,
            "input_tokens": 12,
            "output_tokens": 8,
            "review_reason": (
                "OverviewEvidenceInvalid" if analysis_status != "validated" else None
            ),
        },
    }
    path = tmp_path / "meeting.phase9.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path, payload


def test_offline_summary_preview_renders_only_validated_evidence(tmp_path: Path, monkeypatch):
    source = tmp_path / "CB" / "Nov26" / "meeting-20261107-0817.mp3"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"fixture source")
    output_root = tmp_path / "Transcript"
    monkeypatch.setattr(file_manager, "DEFAULT_TRANSCRIPT_ROOT", output_root)
    json_path, _payload = _write_phase9_json(tmp_path, source)
    output_dir = output_root / "CB" / "Nov26"

    output = write_sara_summary_preview(json_path, source, output_dir)

    assert output.name == "meeting-20261107-0817_Summary_Draft.pdf"
    assert output.is_file()
    reader = PdfReader(str(output))
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert len(reader.pages) >= 1
    assert "ภาพรวมการประชุม" in extracted
    assert "หมวดหมู่การประชุม" in extracted
    assert "ตรวจสอบ Encoder Lot 123" in extracted
    assert "DRAFT - NOT APPROVED" in extracted


def test_review_required_draft_labels_claims_and_owners_for_review(tmp_path: Path):
    source = tmp_path / "meeting-20261107-0817.mp3"
    source.write_bytes(b"fixture source")
    claim = AnalysisItem(
        "ตรวจสอบ Encoder",
        (EvidenceReference(1, quote="ตรวจสอบ Encoder"),),
        owner="ผู้พูด",
        deadline="พรุ่งนี้",
    )
    analysis = MeetingAnalysis(
        overview=claim,
        classifications=(),
        key_issues=(),
        actions=(claim,),
        dependencies=(),
        schedules=(),
        risks=(),
        status="review_required",
        review_reason="SemanticReviewRequired",
    )
    output = tmp_path / "meeting_Summary_Draft.pdf"

    write_meeting_summary_pdf(
        output,
        source,
        "CB",
        analysis,
        draft=True,
    )

    extracted = "\n".join(
        page.extract_text() or "" for page in PdfReader(str(output)).pages
    )
    assert "DRAFT - NOT APPROVED" in extracted
    assert extracted.count("รอตรวจสอบ") >= 3
    assert "หลักฐานอ้างอิงเพื่อการตรวจทาน" in extracted
    assert "ผู้รับผิดชอบ" in extracted
    assert "กำหนดเวลา" in extracted


def test_offline_summary_preview_refuses_unvalidated_analysis_without_placeholder(
    tmp_path: Path,
):
    source = tmp_path / "meeting.mp3"
    source.write_bytes(b"fixture source")
    json_path, _payload = _write_phase9_json(
        tmp_path, source, analysis_status="review_required"
    )
    output_dir = tmp_path / "summary-preview"

    with pytest.raises(SummaryPreviewNotReady, match="OverviewEvidenceInvalid"):
        write_sara_summary_preview(json_path, source, output_dir)

    assert not output_dir.exists()


def test_offline_summary_preview_revalidates_saved_evidence(tmp_path: Path):
    source = tmp_path / "meeting.mp3"
    source.write_bytes(b"fixture source")
    json_path, payload = _write_phase9_json(tmp_path, source)
    payload["meeting_analysis"]["overview"]["evidence"][0]["quote"] = (
        "Unsupported wording Lot 123"
    )
    json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    output_dir = tmp_path / "summary-preview"

    with pytest.raises(SummaryPreviewNotReady, match="OverviewEvidenceInvalid"):
        write_sara_summary_preview(json_path, source, output_dir)

    assert not output_dir.exists()


def test_offline_summary_preview_refuses_unsubstantive_validated_analysis(
    tmp_path: Path,
):
    source = tmp_path / "meeting.mp3"
    source.write_bytes(b"fixture source")
    json_path, payload = _write_phase9_json(tmp_path, source)
    payload["meeting_analysis"]["classifications"] = []
    payload["meeting_analysis"]["key_issues"] = []
    json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    output_dir = tmp_path / "preview"

    with pytest.raises(SummaryPreviewNotReady, match="InsufficientStructuredSummary"):
        write_sara_summary_preview(json_path, source, output_dir)

    assert not output_dir.exists()


def test_offline_summary_preview_refuses_inconsistent_transcript(tmp_path: Path):
    source = tmp_path / "meeting.mp3"
    source.write_bytes(b"fixture source")
    json_path, payload = _write_phase9_json(tmp_path, source)
    payload["cleaned_transcript"] = "different transcript"
    json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(SummaryPreviewNotReady, match="CleanedTranscriptInconsistent"):
        write_sara_summary_preview(json_path, source, tmp_path / "preview")


def test_offline_summary_preview_atomically_replaces_existing_draft(tmp_path: Path, monkeypatch):
    source = tmp_path / "CB" / "Nov26" / "meeting.mp3"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"fixture source")
    output_root = tmp_path / "Transcript"
    monkeypatch.setattr(file_manager, "DEFAULT_TRANSCRIPT_ROOT", output_root)
    json_path, _payload = _write_phase9_json(tmp_path, source)
    output_dir = output_root / "CB" / "Nov26"
    output_dir.mkdir(parents=True)
    output = output_dir / "meeting_Summary_Draft.pdf"
    output.write_bytes(b"stale draft")

    generated = write_sara_summary_preview(json_path, source, output_dir)

    assert generated == output
    assert output.read_bytes().startswith(b"%PDF")
    extracted = "\n".join(
        page.extract_text() or "" for page in PdfReader(str(output)).pages
    )
    assert "DRAFT - NOT APPROVED" in extracted


def test_cli_blocks_v3_placeholder_without_creating_summary(tmp_path: Path):
    source = tmp_path / "meeting.mp3"
    source.write_bytes(b"fixture source")
    json_path, _payload = _write_phase9_json(
        tmp_path, source, analysis_status="review_required"
    )
    output_dir = tmp_path / "preview"

    result = main([
        "phase9-summary-preview",
        "--phase9-json", str(json_path),
        "--summary-source", str(source),
        "--summary-output-dir", str(output_dir),
    ])

    assert result == 2
    assert not output_dir.exists()


def test_offline_layout_demo_uses_only_exact_transcript_passages_and_is_isolated(
    tmp_path: Path,
):
    source = tmp_path / "CB" / "Nov26" / "meeting-20261107-0817.mp3"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"fixture source")
    json_path, payload = _write_phase9_json(tmp_path, source)
    text = " ".join(
        f"บันทึกข้อมูลหมายเลข {index} เกี่ยวกับเครื่องจักรและคุณภาพ"
        for index in range(120)
    )
    segment = TranscriptSegment(1.0, 2.0, "Speaker A", text, text)
    payload["raw_transcript"] = text
    payload["cleaned_transcript"] = text
    payload["segments"] = [asdict(segment)]
    payload["meeting_analysis"]["status"] = "review_required"
    json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    output_dir = tmp_path / "isolated-preview"

    output = write_sara_layout_demo(json_path, source, output_dir)

    assert output == output_dir / "meeting-20261107-0817_SARA_Preview.pdf"
    assert output.is_file()
    assert output.parent != source.parent
    assert not (source.parent / output.name).exists()
    extracted = "\n".join(page.extract_text() or "" for page in PdfReader(str(output)).pages)
    assert "ตัวอย่างจัดหน้าเท่านั้น" in extracted
    assert "ข้อความคัดเลือกเพื่อทดสอบรูปแบบ" in extracted
    assert "เกี่ยวกับเครื่องจักรและคุณภาพ" in extracted
    assert "ต้องตรวจทาน: การวิเคราะห์ข้อเท็จจริงยังไม่ผ่านการตรวจสอบ" not in extracted
    assert "ภาพรวมการประชุม" not in extracted


def test_offline_layout_demo_requires_validated_cleanup_and_preserves_existing_pdf(
    tmp_path: Path,
):
    source = tmp_path / "meeting.mp3"
    source.write_bytes(b"fixture source")
    json_path, payload = _write_phase9_json(tmp_path, source)
    output_dir = tmp_path / "preview"
    payload["cleanup_status"] = "review_required"
    json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(SummaryPreviewNotReady, match="TranscriptCleanupNotValidated"):
        write_sara_layout_demo(json_path, source, output_dir)

    assert not output_dir.exists()

    payload["cleanup_status"] = "completed"
    text = " ".join(
        f"บันทึกข้อมูลหมายเลข {index} เกี่ยวกับเครื่องจักรและคุณภาพ"
        for index in range(120)
    )
    segment = TranscriptSegment(1.0, 2.0, "Speaker A", text, text)
    payload["raw_transcript"] = text
    payload["cleaned_transcript"] = text
    payload["segments"] = [asdict(segment)]
    json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    output_dir.mkdir()
    output = output_dir / "meeting_SARA_Preview.pdf"
    output.write_bytes(b"preserve")

    with pytest.raises(FileExistsError):
        write_sara_layout_demo(json_path, source, output_dir)

    assert output.read_bytes() == b"preserve"
