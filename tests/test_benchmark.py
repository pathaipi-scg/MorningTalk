import csv
from pathlib import Path

from reportlab.pdfgen import canvas

from app.benchmark import (
    _analysis_claims,
    classify_discrepancy,
    discover_inventory,
    extract_pdf_pages,
    extract_reference_facts,
    generate_benchmark_reports,
)


def _touch(path: Path, content: bytes = b"fixture") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_recursive_inventory_uses_exact_filename_stems_and_output_paths(tmp_path):
    recording_root = tmp_path / "Record File"
    transcript_root = tmp_path / "Transcript"
    sara_root = tmp_path / "SaraAI"
    source = _touch(
        recording_root / "PlantA" / "Nov26" / "meeting-20261107-0817.mp3"
    )
    sara = _touch(sara_root / "Nov26" / "meeting-20261107-0817.pdf")
    transcript = _touch(
        transcript_root / "PlantA" / "Nov26" / "meeting-20261107-0817.txt",
        b"validated transcript",
    )
    full = _touch(
        transcript_root / "PlantA" / "Nov26"
        / "meeting-20261107-0817_Full.pdf"
    )
    draft = _touch(
        transcript_root / "PlantA" / "Nov26"
        / "meeting-20261107-0817_Summary_Draft.pdf"
    )

    result = discover_inventory(
        recording_root,
        transcript_root,
        sara_root,
        duration_probe=lambda _path: 123.5,
    )

    assert len(result.meetings) == 1
    meeting = result.meetings[0]
    assert meeting.mp3_path == source
    assert meeting.sara_pdf_path == sara
    assert meeting.transcript_txt_path == transcript
    assert meeting.full_transcript_pdf_path == full
    assert meeting.summary_draft_pdf_path == draft
    assert meeting.approved_summary_pdf_path is None
    assert meeting.duration_seconds == 123.5
    assert meeting.missing_files == ()


def test_same_stem_duplicate_recordings_without_plant_reference_are_ambiguous(
    tmp_path,
):
    recording_root = tmp_path / "Record File"
    transcript_root = tmp_path / "Transcript"
    sara_root = tmp_path / "SaraAI"
    transcript_root.mkdir()
    _touch(recording_root / "PlantA" / "Nov26" / "meeting-20261107-0817.mp3")
    _touch(recording_root / "PlantB" / "Nov26" / "meeting-20261107-0817.mp3")
    sara = _touch(sara_root / "Nov26" / "meeting-20261107-0817.pdf")

    result = discover_inventory(recording_root, transcript_root, sara_root)

    assert all(meeting.sara_match_status == "ambiguous" for meeting in result.meetings)
    assert result.ambiguous_sara_pdfs == (sara,)
    assert result.unmatched_sara_pdfs == ()


def test_source_adjacent_pdfs_are_associated_only_by_exact_recording_stem(tmp_path):
    recording_root = tmp_path / "Record File"
    transcript_root = tmp_path / "Transcript"
    sara_root = tmp_path / "SaraAI"
    first = _touch(
        recording_root / "PlantA" / "Nov26" / "meeting-20261107-0817.mp3"
    )
    _touch(recording_root / "PlantA" / "Nov26" / "meeting-20261108-0817.mp3")
    adjacent = _touch(first.with_name("meeting-20261107-0817_Full.pdf"))
    _touch(first.with_name("meeting_Summary_Draft.pdf"))
    transcript_root.mkdir()
    sara_root.mkdir()

    result = discover_inventory(recording_root, transcript_root, sara_root)

    assert result.meetings[0].source_adjacent_pdf_paths == (adjacent,)
    assert result.meetings[1].source_adjacent_pdf_paths == ()


def test_missing_and_nearby_date_reference_are_not_fuzzy_matched(tmp_path):
    recording_root = tmp_path / "Record File"
    transcript_root = tmp_path / "Transcript"
    sara_root = tmp_path / "SaraAI"
    transcript_root.mkdir()
    _touch(recording_root / "PlantA" / "Nov26" / "meeting-20261107-0817.mp3")
    nearby = _touch(sara_root / "Nov26" / "meeting-20261108-0817.pdf")

    result = discover_inventory(recording_root, transcript_root, sara_root)

    assert result.meetings[0].sara_match_status == "missing"
    assert "SARA reference PDF" in result.meetings[0].missing_files
    assert result.unmatched_sara_pdfs == (nearby,)


def test_pdf_extraction_preserves_pages_and_reference_wording(tmp_path):
    pdf_path = tmp_path / "sara.pdf"
    document = canvas.Canvas(str(pdf_path))
    document.drawString(40, 800, "Twin Loader Encoder issue Strength 109")
    document.showPage()
    document.drawString(40, 800, "Maintenance action due Friday")
    document.save()

    pages = extract_pdf_pages(pdf_path)
    facts = extract_reference_facts(pages)

    assert len(pages) == 2
    assert "Twin Loader Encoder issue Strength 109" in pages[0].text
    assert any(fact.page_number == 1 and fact.category == "equipment" for fact in facts)
    assert any(
        fact.page_number == 2 and fact.category == "decision_action"
        for fact in facts
    )


def test_discrepancy_classifier_uses_conservative_evidence_labels():
    statement = "Twin Loader encoder issue"
    assert classify_discrepancy(
        evidence_kind="sara_reference",
        source_wording=statement,
        raw_transcript=f"Discussed {statement}",
        cleaned_transcript=f"Discussed {statement}",
        summary_text="Different summary",
    ) == "Summary omission"
    assert classify_discrepancy(
        evidence_kind="sara_reference",
        source_wording=statement,
        raw_transcript="No matching source text",
        cleaned_transcript="No matching source text",
        summary_text="",
    ) == "Insufficient evidence"
    assert classify_discrepancy(
        evidence_kind="morningtalk_summary",
        source_wording="unsupported item",
        raw_transcript="",
        cleaned_transcript="",
        summary_text="unsupported item",
    ) == "Insufficient evidence"
    assert classify_discrepancy(
        evidence_kind="suspicious_phrase",
        source_wording="พ่อครับ",
        raw_transcript="พ่อครับ",
        cleaned_transcript="พ่อครับ",
        summary_text=None,
        suspicious_phrase=True,
    ) == "Suspected transcription error"
    assert classify_discrepancy(
        evidence_kind="sara_reference",
        source_wording=statement,
        raw_transcript="Twin Loader",
        cleaned_transcript=f"Discussed {statement}",
        summary_text=None,
    ) == "Cleanup error"


def test_phase10_evidence_offsets_must_point_to_nonempty_segment_text():
    payload = {
        "meeting_analysis": {
            "status": "review_required",
            "overview": [{
                "text": "Encoder was discussed",
                "evidence": [{
                    "segment_id": 1,
                    "start_char": 6,
                    "end_char": 13,
                }],
            }],
        }
    }

    claims = _analysis_claims(
        payload,
        source_segments=[{
            "raw_text": "Check Encoder",
            "cleaned_text": "Check Encoder",
        }],
    )

    assert claims == [("overview", "Encoder was discussed", True)]
    invalid_claim = {
        "meeting_analysis": {
            "status": "review_required",
            "overview": [{
                "text": "Unsupported span",
                "evidence": [{
                    "segment_id": 1,
                    "start_char": 1,
                    "end_char": 99,
                }],
            }],
        }
    }
    assert not _analysis_claims(
        invalid_claim,
        source_segments=[{"raw_text": "Check Encoder", "cleaned_text": "Check Encoder"}],
    )[0][2]


def test_benchmark_report_generation_writes_inventory_and_discrepancies(tmp_path):
    recording_root = tmp_path / "Record File"
    transcript_root = tmp_path / "Transcript"
    sara_root = tmp_path / "SaraAI"
    _touch(recording_root / "PlantA" / "Nov26" / "meeting-20261107-0817.mp3")
    reference = sara_root / "Nov26" / "meeting-20261107-0817.pdf"
    reference.parent.mkdir(parents=True)
    document = canvas.Canvas(str(reference))
    document.drawString(40, 800, "Twin Loader maintenance action")
    document.save()
    _touch(
        transcript_root / "PlantA" / "Nov26" / "meeting-20261107-0817.txt",
        "Discussed Twin Loader พ่อครับ".encode("utf-8"),
    )

    result = discover_inventory(
        recording_root,
        transcript_root,
        sara_root,
        duration_probe=lambda _path: None,
    )
    paths = generate_benchmark_reports(result, tmp_path / "reports" / "benchmark")

    assert all(path.is_file() for path in paths)
    with paths[0].open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    assert rows[0]["sara_match_status"] == "matched"
    with paths[2].open(encoding="utf-8-sig", newline="") as source:
        discrepancies = list(csv.DictReader(source))
    assert any(row["category"] == "suspicious_phrase" for row in discrepancies)
    with paths[3].open(encoding="utf-8-sig", newline="") as source:
        extracts = list(csv.DictReader(source))
    assert extracts
    assert all(row["page"] == "1" for row in extracts)
