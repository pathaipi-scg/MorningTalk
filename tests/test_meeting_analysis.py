import json
import logging
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests
from pypdf import PdfReader

from app.config import AzureConfig
from app.meeting_analysis import (
    AnalysisItem,
    EvidenceReference,
    GPTMeetingAnalyzer,
    MeetingAnalysis,
    MeetingAnalysisError,
)
from app.meeting_outputs import meeting_pdf_paths, write_meeting_pdfs
from app.pdf_export import write_meeting_summary_pdf
from app.transcript import TranscriptDocument, TranscriptSegment


@pytest.fixture(autouse=True)
def quiet_pypdf_legacy_warning(monkeypatch):
    import pypdf.filters

    monkeypatch.setattr(pypdf.filters, "JBIG2DEC_BINARY", None)


def azure_config(deployment="configured-deployment"):
    return AzureConfig(
        text_endpoint="https://text.example.test",
        text_api_key="test-key",
        text_deployment=deployment,
        transcription_endpoint="https://transcription.example.test",
        transcription_api_key="test-transcription-key",
        transcription_deployment="gpt-4o-transcribe",
        transcription_api_version="2025-03-01-preview",
    )


def valid_payload():
    return {
        "overview": {
            "text": "The meeting discussed Encoder Lot 123.",
            "evidence": [{"segment_id": 1, "quote": "ตรวจ Encoder Lot 123"}],
        },
        "classifications": [{
            "text": "Encoder และเครื่องจักร",
            "evidence": [{"segment_id": 1, "quote": "ตรวจ Encoder Lot 123"}],
        }],
        "topics": [{
            "text": "เครื่องจักร — ตรวจ Encoder Lot 123",
            "evidence": [{"segment_id": 1, "quote": "ตรวจ Encoder Lot 123"}],
        }],
        "key_issues": [{
            "text": "ตรวจสอบ Encoder Lot 123",
            "evidence": [{"segment_id": 1, "quote": "ตรวจ Encoder Lot 123"}],
        }],
        "actions": [],
        "dependencies": [],
        "schedules": [],
        "risks": [],
        "closing_activities": [],
    }


def response_for(payload):
    response = Mock(status_code=200)
    response.json.return_value = {
        "choices": [{"message": {"content": json.dumps(payload, ensure_ascii=False)}}],
        "usage": {"prompt_tokens": 500, "completion_tokens": 100},
    }
    return response


def responses_api_result(payload, *, status="completed", usage=None):
    response = Mock(status_code=200)
    response.json.return_value = {
        "status": status,
        "output": [{
            "type": "message",
            "content": [{
                "type": "output_text",
                "text": json.dumps(payload, ensure_ascii=False),
            }],
        }],
        "usage": usage or {"input_tokens": 500, "output_tokens": 100},
    }
    return response


def test_analysis_uses_configured_model_and_prompts_for_source_offsets():
    segments = (TranscriptSegment(1, 2, "Speaker A", "ตรวจ Encoder Lot 123"),)
    session = Mock()
    session.post.return_value = response_for(valid_payload())

    analysis = GPTMeetingAnalyzer(
        azure_config("customer-deployment-name"),
        api_version="test-version",
        session=session,
    ).analyze(segments)

    assert analysis.status == "validated"
    assert analysis.overview is not None
    assert analysis.overview.evidence[0].segment_id == 1
    assert len(analysis.topics) == 1
    assert analysis.closing_activities == ()
    assert analysis.input_tokens == 500
    assert analysis.output_tokens == 100
    assert "customer-deployment-name" in session.post.call_args.args[0]
    assert session.post.call_args.kwargs["params"] == {"api-version": "test-version"}
    request_text = session.post.call_args.kwargs["json"]["messages"][1]["content"]
    assert "ตรวจ Encoder Lot 123" in request_text
    assert "start_char/end_char offsets" in request_text
    assert "Python extracts the cited text directly from the source" in request_text
    assert "Python Unicode string character indexes" in request_text
    assert "SARA" not in request_text


def test_analysis_uses_complete_responses_url_and_structured_json():
    url = "https://text.example.test/openai/responses?api-version=2025-04-01-preview"
    session = Mock()
    offset_payload = valid_payload()
    for section in (
        "overview", "classifications", "topics", "key_issues", "actions",
        "dependencies", "schedules", "risks", "closing_activities",
    ):
        items = offset_payload[section]
        if isinstance(items, dict):
            items = [items]
        for item in items:
            quote = item["evidence"][0]["quote"]
            start = "ตรวจ Encoder Lot 123".index(quote)
            item["evidence"] = [{
                "segment_id": 1,
                "start_char": start,
                "end_char": start + len(quote),
            }]
    session.post.return_value = responses_api_result(offset_payload)
    config = azure_config("gpt-5.4-mini")
    config = AzureConfig(
        **{
            **config.__dict__,
            "text_endpoint": url,
        }
    )
    analyzer = GPTMeetingAnalyzer(
        config, api_version="v1", session=session
    )

    analysis = analyzer.analyze((TranscriptSegment(1, 2, "Speaker A", "ตรวจ Encoder Lot 123"),))

    request = session.post.call_args
    body = request.kwargs["json"]
    assert request.args[0] == url
    assert request.kwargs["params"] is None
    assert body["model"] == "gpt-5.4-mini"
    assert body["input"][0]["role"] == "user"
    assert body["input"][0]["content"][0]["type"] == "input_text"
    assert "ตรวจ Encoder Lot 123" in body["input"][0]["content"][0]["text"]
    assert body["max_output_tokens"] == 12000
    assert body["text"]["format"]["type"] == "json_schema"
    item_schema = body["text"]["format"]["schema"]["properties"]["overview"]
    assert item_schema["required"] == ["text", "owner", "deadline", "evidence"]
    assert item_schema["properties"]["evidence"]["items"]["required"] == [
        "segment_id", "start_char", "end_char"
    ]
    assert "quote" not in item_schema["properties"]["evidence"]["items"]["properties"]
    assert body["text"]["format"]["schema"]["required"] == [
        "overview",
        "classifications",
        "topics",
        "key_issues",
        "actions",
        "dependencies",
        "schedules",
        "risks",
        "closing_activities",
    ]
    assert "messages" not in body
    assert analysis.status == "review_required"
    assert analysis.review_reason == "SemanticReviewRequired"
    assert (analysis.input_tokens, analysis.output_tokens) == (500, 100)


def _parse_analysis_payload(payload, source_text):
    return GPTMeetingAnalyzer._parse_and_validate(
        payload,
        (TranscriptSegment(1, 2, None, source_text),),
    )


def _overview_payload(quote, *, segment_id=1, text="Discussed Lot 123"):
    return {
        "overview": {
            "text": text,
            "evidence": [{"segment_id": segment_id, "quote": quote}],
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


def test_analysis_accepts_only_nfc_and_whitespace_equivalent_verbatim_evidence():
    source = "Cafe\u0301 ตรวจสอบ   เครื่องจักร Lot 123"
    quote = "Café ตรวจสอบ เครื่องจักร Lot 123"

    analysis = _parse_analysis_payload(_overview_payload(quote), source)

    assert analysis.status == "validated"
    assert analysis.overview is not None
    assert analysis.overview.evidence[0].quote == quote


def test_analysis_evidence_matches_cleaned_display_text():
    segment = TranscriptSegment(
        1, 2, None, "raw source wording", "cleaned wording Lot 123"
    )
    analysis = GPTMeetingAnalyzer._parse_and_validate(
        _overview_payload("cleaned wording Lot 123"),
        (segment,),
    )

    assert analysis.status == "validated"


@pytest.mark.parametrize(
    ("source", "quote", "segment_id", "text", "failure"),
    [
        ("ตรวจ Encoder Lot 123", "ตรวจสอบ Encoder Lot 123", 1,
         "Discussed Lot 123", "quote_not_verbatim"),
        ("ตรวจ Encoder Lot 123", "อนุมัติ Lot 123", 1,
         "Discussed Lot 123", "quote_not_verbatim"),
        ("ตรวจ Encoder Lot 123", "ตรวจ Encoder Lot 123", 2,
         "Discussed Lot 123", "segment_id_not_found"),
        ("ตรวจ Encoder, Lot 123", "ตรวจ Encoder Lot 123", 1,
         "Discussed Lot 123", "quote_not_verbatim"),
        ("ตรวจ Encoder Lot 123", "ตรวจ Encoder Lot 123", 1,
         "Approved Lot 124", "claim_number_not_in_evidence"),
    ],
)
def test_analysis_rejects_nonverbatim_evidence_with_safe_diagnostics(
    source, quote, segment_id, text, failure, caplog
):
    payload = _overview_payload(
        quote, segment_id=segment_id, text=text
    )

    with caplog.at_level(logging.WARNING, logger="app.meeting_analysis"):
        analysis = _parse_analysis_payload(payload, source)

    assert analysis.status == "review_required"
    assert analysis.review_reason == "OverviewEvidenceInvalid"
    assert "rule=OverviewEvidenceInvalid" in caplog.text
    assert "section=overview" in caplog.text
    assert f"match_failure={failure}" in caplog.text
    assert f"quote_chars={len(quote)}" in caplog.text
    assert "ตรวจ" not in caplog.text
    if segment_id == 2:
        assert "segment_id=2" in caplog.text


def test_analysis_rejects_missing_evidence_and_logs_no_quote_content(caplog):
    payload = _overview_payload("private transcript sentinel")
    payload["overview"]["evidence"] = []

    with caplog.at_level(logging.WARNING, logger="app.meeting_analysis"):
        analysis = _parse_analysis_payload(
            payload, "private transcript sentinel"
        )

    assert analysis.status == "review_required"
    assert analysis.review_reason == "OverviewEvidenceInvalid"
    assert "match_failure=missing_evidence" in caplog.text
    assert "private transcript sentinel" not in caplog.text


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (
            {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
            "IncompleteResponse",
        ),
        (
            {
                "status": "completed",
                "output": [{
                    "type": "message",
                    "content": [{"type": "refusal", "refusal": "refused"}],
                }],
            },
            "Refusal",
        ),
        ({"status": "completed", "output": []}, "MalformedOutput"),
    ],
)
def test_analysis_rejects_responses_api_failure_states(body, reason):
    session = Mock()
    response = Mock(status_code=200)
    response.json.return_value = {
        **body,
        "usage": {"input_tokens": 15, "output_tokens": 6},
    }
    session.post.return_value = response
    config = azure_config()
    config = AzureConfig(
        **{
            **config.__dict__,
            "text_endpoint": (
                "https://text.example.test/openai/responses"
                "?api-version=2025-04-01-preview"
            ),
        }
    )

    analysis = GPTMeetingAnalyzer(
        config, api_version="v1", session=session
    ).analyze((TranscriptSegment(1, 2, None, "source"),))

    assert analysis.status == "review_required"
    assert analysis.review_reason == reason
    assert (analysis.input_tokens, analysis.output_tokens) == (15, 6)


def test_analysis_does_not_retry_responses_404():
    session = Mock()
    response = Mock(status_code=404, headers={"apim-request-id": "req-2"})
    response.json.return_value = {"error": {"code": "DeploymentNotFound"}}
    session.post.return_value = response
    config = AzureConfig(
        **{
            **azure_config().__dict__,
            "text_endpoint": (
                "https://text.example.test/openai/responses"
                "?api-version=2025-04-01-preview"
            ),
        }
    )

    with pytest.raises(MeetingAnalysisError, match="HTTP 404"):
        GPTMeetingAnalyzer(
            config, api_version="v1", session=session
        ).analyze((TranscriptSegment(1, 2, None, "source"),))

    session.post.assert_called_once()


def test_analysis_v1_uses_deployment_model_without_dated_api_version():
    session = Mock()
    session.post.return_value = response_for(valid_payload())
    analyzer = GPTMeetingAnalyzer(
        azure_config("configured-gpt-5.4-mini"),
        api_version="v1",
        session=session,
    )

    analysis = analyzer.analyze((TranscriptSegment(1, 2, "Speaker A", "ตรวจ Encoder Lot 123"),))

    request = session.post.call_args
    assert analysis.status == "validated"
    assert request.args[0] == "https://text.example.test/openai/v1/chat/completions"
    assert request.kwargs["params"] is None
    assert request.kwargs["json"]["model"] == "configured-gpt-5.4-mini"


@pytest.mark.parametrize(
    ("field", "replace"),
    [
        ("overview", {"text": "Unsupported", "evidence": [{"segment_id": 2, "quote": "ตรวจ Encoder Lot 123"}]}),
        ("overview", {"text": "Unsupported", "evidence": [{"segment_id": 1, "quote": "ตรวจระบบ"}]}),
        ("key_issues", [{"text": "Lot 124 was approved", "evidence": [{"segment_id": 1, "quote": "ตรวจ Encoder Lot 123"}]}]),
        ("topics", [{"text": "เครื่องจักร — Lot 124 was approved", "evidence": [{"segment_id": 1, "quote": "ตรวจ Encoder Lot 123"}]}]),
    ],
)
def test_analysis_rejects_unsupported_or_numeric_claims(field, replace):
    segments = (TranscriptSegment(1, 2, None, "ตรวจ Encoder Lot 123"),)
    payload = valid_payload()
    payload[field] = replace
    session = Mock()
    session.post.return_value = response_for(payload)

    analysis = GPTMeetingAnalyzer(
        azure_config(), api_version="test-version", session=session
    ).analyze(segments)

    assert analysis.status == "review_required"
    assert analysis.overview is None
    assert analysis.key_issues == ()


def test_analysis_api_failure_is_reported_without_request_details():
    session = Mock()
    session.post.side_effect = requests.exceptions.Timeout("secret request contents")
    analyzer = GPTMeetingAnalyzer(
        azure_config(), api_version="test-version", session=session
    )
    with pytest.raises(MeetingAnalysisError) as error:
        analyzer.analyze((TranscriptSegment(None, None, None, "source"),))
    assert "secret" not in str(error.value)


def test_summary_pdf_contains_topic_structure_without_repeating_evidence_quotes(tmp_path: Path):
    source = tmp_path / "meeting-20261107-0817.mp3"
    source.write_bytes(b"fixture")
    evidence = (EvidenceReference(1, "ตรวจ Encoder Lot 123"),)
    analysis = MeetingAnalysis(
        overview=AnalysisItem("ตรวจสอบ Encoder Lot 123", evidence),
        classifications=(AnalysisItem("เครื่องจักร", evidence),),
        key_issues=(AnalysisItem("ต้องตรวจ Encoder", evidence),),
        actions=(),
        dependencies=(),
        schedules=(),
        risks=(),
        status="validated",
        topics=(
            AnalysisItem(
                "เครื่องจักรและการผลิต — ตรวจสอบ Encoder รุ่น 123",
                evidence,
            ),
        ),
        closing_activities=(
            AnalysisItem("ปิดประชุม", evidence),
        ),
    )
    pdf = tmp_path / "meeting-20261107-0817.pdf"

    write_meeting_summary_pdf(pdf, source, "CB", analysis)

    text = "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf)).pages)
    for section in (
        "สรุปการประชุมภาคเช้า",
        "ภาพรวมการประชุม",
        "สรุปแยกตามหัวข้อ",
        "ประเด็นสำคัญ",
        "กิจกรรมปิดการประชุม",
        "หลักฐาน: ช่วงที่ 1",
    ):
        assert section in text
    assert "ตรวจ Encoder Lot 123" not in text
    assert "รายการดำเนินการและผู้รับผิดชอบ" not in text
    assert "กำหนดการ" not in text
    assert "ไม่มีข้อมูลที่ยืนยันได้" not in text
    assert len(PdfReader(str(pdf)).pages) == 1
    page = PdfReader(str(pdf)).pages[0]
    assert round(float(page.mediabox.width), 1) == 595.3
    assert round(float(page.mediabox.height), 1) == 841.9


def test_summary_pdf_displays_explicit_owner_and_deadline(tmp_path: Path):
    source = tmp_path / "meeting-20261107-0817.mp3"
    source.write_bytes(b"fixture")
    evidence = (EvidenceReference(1, "สมชายตรวจ Encoder Lot 123 ภายในพรุ่งนี้"),)
    analysis = MeetingAnalysis(
        overview=AnalysisItem("ตรวจ Encoder Lot 123", evidence),
        classifications=(),
        key_issues=(),
        actions=(
            AnalysisItem(
                "ตรวจ Encoder Lot 123",
                evidence,
                owner="สมชาย",
                deadline="พรุ่งนี้",
            ),
        ),
        dependencies=(),
        schedules=(),
        risks=(),
        status="validated",
    )
    pdf = tmp_path / "meeting-summary.pdf"

    write_meeting_summary_pdf(pdf, source, "CB", analysis)

    extracted = "\n".join(
        page.extract_text() or "" for page in PdfReader(str(pdf)).pages
    )
    assert "ผู้รับผิดชอบ: สมชาย" in extracted
    assert "กำหนดเวลา: พรุ่งนี้" in extracted


def test_summary_pdf_paginates_long_thai_text_and_keeps_layout_readable(tmp_path: Path):
    source = tmp_path / "long-meeting.mp3"
    source.write_bytes(b"fixture")
    excerpt = "ตรวจสอบประเด็นการผลิตและคุณภาพอย่างต่อเนื่อง "
    long_text = excerpt * 420
    evidence = (EvidenceReference(1, excerpt.strip()),)
    analysis = MeetingAnalysis(
        overview=AnalysisItem("สรุปภาพรวมการประชุม", evidence),
        classifications=(),
        key_issues=(),
        actions=(),
        dependencies=(),
        schedules=(),
        risks=(),
        status="validated",
        topics=(AnalysisItem("การผลิต — " + long_text, evidence),),
    )
    pdf = tmp_path / "long-meeting.pdf"

    write_meeting_summary_pdf(pdf, source, "CB", analysis)

    reader = PdfReader(str(pdf))
    assert len(reader.pages) > 1
    for page in reader.pages:
        assert round(float(page.mediabox.width), 1) == 595.3
        assert round(float(page.mediabox.height), 1) == 841.9
        assert "MorningTalk" in (page.extract_text() or "")
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "ตรวจสอบประเด็นการผลิต" in extracted
    assert "ข้อมูลที่ยืนยันได้" not in extracted
    font_names = {
        str(font.get_object().get("/BaseFont", ""))
        for page in reader.pages
        for font in page.get("/Resources", {}).get("/Font", {}).values()
    }
    assert any("Tahoma" in name or "Sarabun" in name or "Noto" in name for name in font_names)


def test_output_paths_follow_exact_basename_rules(tmp_path: Path):
    source = tmp_path / "meeting-20261107-0817.mp3"

    summary, full = meeting_pdf_paths(source)

    assert summary == tmp_path / "meeting-20261107-0817.pdf"
    assert full == tmp_path / "meeting-20261107-0817_Full.pdf"


def test_summary_and_full_outputs_are_independent_and_existing_files_are_preserved(
    tmp_path: Path,
):
    source = tmp_path / "meeting.mp3"
    source.write_bytes(b"fixture")
    summary, full = meeting_pdf_paths(source)
    summary.write_bytes(b"existing summary")
    before = summary.read_bytes()
    transcript = TranscriptDocument(
        (TranscriptSegment(1, 2, None, "full source text"),),
        "unavailable",
        "review_required",
    )
    analysis = MeetingAnalysis.for_review("No validated analysis fixture")

    outputs = write_meeting_pdfs(source, "CB", transcript, analysis)

    assert outputs.summary.status == "existing"
    assert outputs.full_transcript.status == "created"
    assert summary.read_bytes() == before
    assert full.is_file()


def test_fixture_pdf_pair_is_marked_as_fixture_and_never_uses_azure(tmp_path: Path):
    from app.phase9 import run_fixture_preview

    summary = run_fixture_preview(tmp_path)
    full = summary.with_name(f"{summary.stem}_Full.pdf")
    summary_text = "\n".join(
        page.extract_text() or "" for page in PdfReader(str(summary)).pages
    )
    full_text = "\n".join(
        page.extract_text() or "" for page in PdfReader(str(full)).pages
    )

    assert "ตัวอย่างจัดหน้าเท่านั้น" in summary_text
    assert "FIXTURE TEXT, NOT AN ACTUAL TRANSCRIPTION" in full_text
    assert "หมวดหมู่การประชุม" in summary_text
    assert "Speaker A" in full_text
