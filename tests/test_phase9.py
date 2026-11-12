import json
import logging
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests
from pypdf import PdfReader

from app.azure_speech import AzureSpeechDiarizer, DiarizationUnavailable
from app.config import AzureConfig
from app.pdf_export import write_transcript_pdf
from app.transcript import (
    TranscriptDocument,
    TranscriptSegment,
    canonical_speaker_labels,
    number_code_mismatch,
    preserves_numbers_and_codes,
    validate_segment_times,
)
from app.transcript_cleanup import TranscriptCleanup
from app.transcript_cleanup import _split_cleanup_text
import app.transcriber as transcriber_module


@pytest.fixture(autouse=True)
def quiet_pypdf_legacy_warning(monkeypatch):
    import pypdf.filters

    monkeypatch.setattr(pypdf.filters, "JBIG2DEC_BINARY", None)


def azure_config():
    return AzureConfig(
        text_endpoint="https://text.example.test",
        text_api_key="test-key",
        text_deployment="gpt-5.4-mini",
        transcription_endpoint="https://transcription.example.test",
        transcription_api_key="test-transcription-key",
        transcription_deployment="gpt-4o-transcribe",
        transcription_api_version="2025-03-01-preview",
    )


def speech_response(phrases):
    response = Mock(status_code=200)
    response.json.return_value = {"phrases": phrases}
    return response


def test_speech_diarization_labels_two_speakers_and_preserves_timestamps(
    tmp_path: Path, monkeypatch,
):
    source = tmp_path / "meeting.mp3"
    source.write_bytes(b"fixture audio")
    monkeypatch.setattr(transcriber_module.AzureOpenAITranscriber, "_probe_duration", lambda _p: 60)
    session = Mock()
    session.post.return_value = speech_response([
        {"offsetMilliseconds": 1200, "durationMilliseconds": 800, "speaker": "guest-1", "text": "เริ่มประชุม"},
        {"offsetMilliseconds": 2100, "durationMilliseconds": 900, "speaker": "guest-2", "text": "ครับ"},
        {"offsetMilliseconds": 3100, "durationMilliseconds": 400, "speaker": "guest-1", "text": "ขอบคุณ"},
    ])

    segments = AzureSpeechDiarizer(
        "https://speech.example.test", "secret", session=session
    ).transcribe(source)

    assert [segment.speaker_id for segment in segments] == [
        "Speaker A", "Speaker B", "Speaker A",
    ]
    assert [(segment.start_seconds, segment.end_seconds) for segment in segments] == [
        (1.2, 2.0), (2.1, 3.0), (3.1, 3.5),
    ]
    request = session.post.call_args
    definition = json.loads(request.kwargs["data"]["definition"])
    assert definition["locales"] == ["th-TH"]
    assert definition["diarization"]["enabled"] is True
    assert request.kwargs["headers"] == {"Ocp-Apim-Subscription-Key": "secret"}


def test_canonical_labels_support_three_speakers_and_unknown():
    assert canonical_speaker_labels(["x", "y", "z", None, "x"]) == [
        "Speaker A", "Speaker B", "Speaker C", None, "Speaker A",
    ]


def test_diarization_marks_an_uncertain_phrase_as_unknown(
    tmp_path: Path, monkeypatch,
):
    source = tmp_path / "meeting.mp3"
    source.write_bytes(b"fixture audio")
    monkeypatch.setattr(transcriber_module.AzureOpenAITranscriber, "_probe_duration", lambda _p: 60)
    session = Mock()
    session.post.return_value = speech_response([
        {"offsetMilliseconds": 0, "durationMilliseconds": 400, "text": "ไม่แน่ใจ"},
    ])

    segments = AzureSpeechDiarizer(
        "https://speech.example.test", "key", session=session
    ).transcribe(source)

    assert segments[0].speaker_id == "Speaker unknown"


def test_speech_diarization_preserves_overlapping_segments(
    tmp_path: Path, monkeypatch,
):
    source = tmp_path / "meeting.mp3"
    source.write_bytes(b"audio")
    monkeypatch.setattr(transcriber_module.AzureOpenAITranscriber, "_probe_duration", lambda _p: 60)
    session = Mock()
    session.post.return_value = speech_response([
        {"offsetMilliseconds": 1000, "durationMilliseconds": 2000, "speaker": "a", "text": "first"},
        {"offsetMilliseconds": 1800, "durationMilliseconds": 1000, "speaker": "b", "text": "overlap"},
    ])

    segments = AzureSpeechDiarizer(
        "https://speech.example.test", "key", session=session
    ).transcribe(source)

    assert segments[0].end_seconds > segments[1].start_seconds
    assert segments[1].raw_text == "overlap"


def test_speech_diarization_rejects_missing_deployment_configuration():
    with pytest.raises(ValueError, match="HTTPS URL"):
        AzureSpeechDiarizer("", "key")


def test_speech_diarization_api_failure_is_safe(tmp_path: Path, monkeypatch):
    source = tmp_path / "meeting.mp3"
    source.write_bytes(b"audio")
    monkeypatch.setattr(transcriber_module.AzureOpenAITranscriber, "_probe_duration", lambda _p: 60)
    response = Mock(status_code=429)
    session = Mock()
    session.post.return_value = response
    with pytest.raises(DiarizationUnavailable, match="HTTP 429"):
        AzureSpeechDiarizer(
            "https://speech.example.test", "key", session=session
        ).transcribe(source)


def test_cleanup_corrects_thai_while_preserving_speaker_timestamp_numbers():
    segments = (
        TranscriptSegment(1.0, 3.0, "Speaker A", "ตรวจ Encoder Lot 123 ที่ LINE1"),
    )
    response = Mock(status_code=200)
    response.json.return_value = {
        "choices": [{"message": {"content": json.dumps(
            {"segments": [{"index": 0, "text": "ตรวจ Encoder Lot 123 ที่ LINE1"}]},
            ensure_ascii=False,
        )}}],
        "usage": {"prompt_tokens": 70, "completion_tokens": 15},
    }
    session = Mock()
    session.post.return_value = response

    document = TranscriptCleanup(
        azure_config(), api_version="2025-04-01-preview", session=session
    ).clean(segments)

    assert document.cleanup_status == "completed"
    assert document.segments[0].speaker_id == "Speaker A"
    assert (document.segments[0].start_seconds, document.segments[0].end_seconds) == (1, 3)
    assert document.input_tokens == 70
    assert document.output_tokens == 15
    assert session.post.call_args.kwargs["params"] == {"api-version": "2025-04-01-preview"}


def test_cleanup_v1_uses_deployment_model_without_dated_api_version():
    segments = (TranscriptSegment(1.0, 3.0, "Speaker A", "ตรวจ Encoder Lot 123"),)
    response = Mock(status_code=200)
    response.json.return_value = {
        "choices": [{"message": {"content": json.dumps(
            {"segments": [{"index": 0, "text": "ตรวจ Encoder Lot 123"}]},
            ensure_ascii=False,
        )}}],
        "usage": {},
    }
    session = Mock()
    session.post.return_value = response

    result = TranscriptCleanup(
        azure_config(), api_version="v1", session=session
    ).clean(segments)

    request = session.post.call_args
    assert result.cleanup_status == "completed"
    assert request.args[0] == "https://text.example.test/openai/v1/chat/completions"
    assert request.kwargs["params"] is None
    assert request.kwargs["json"]["model"] == "gpt-5.4-mini"
    assert request.kwargs["headers"]["api-key"] == "test-key"


def test_cleanup_uses_complete_responses_url_and_structured_json():
    url = "https://text.example.test/openai/responses?api-version=2025-04-01-preview"
    segment = TranscriptSegment(1.0, 3.0, "Speaker A", "ตรวจ Encoder Lot 123")
    response = Mock(status_code=200)
    response.json.return_value = {
        "status": "completed",
        "output": [{
            "type": "message",
            "content": [{
                "type": "output_text",
                "text": json.dumps(
                    {"segments": [{"index": 0, "text": "ตรวจ Encoder Lot 123"}]},
                    ensure_ascii=False,
                ),
            }],
        }],
        "usage": {"input_tokens": 70, "output_tokens": 15},
    }
    session = Mock()
    session.post.return_value = response
    config = azure_config()
    config = AzureConfig(
        **{
            **config.__dict__,
            "text_endpoint": url,
            "text_deployment": "gpt-5.4-mini",
        }
    )

    document = TranscriptCleanup(
        config, api_version="v1", session=session
    ).clean((segment,))

    request = session.post.call_args
    body = request.kwargs["json"]
    assert request.args[0] == url
    assert request.kwargs["params"] is None
    assert request.kwargs["headers"] == {
        "api-key": "test-key",
        "Content-Type": "application/json",
    }
    assert body["model"] == "gpt-5.4-mini"
    assert body["input"][0]["role"] == "user"
    assert body["input"][0]["content"][0]["type"] == "input_text"
    assert "ตรวจ Encoder Lot 123" in body["input"][0]["content"][0]["text"]
    assert "Do not add, remove, convert, round, or reformat numbers or codes" in (
        body["input"][0]["content"][0]["text"]
    )
    assert body["max_output_tokens"] == 16000
    assert body["text"]["format"]["type"] == "json_schema"
    assert body["text"]["format"]["schema"]["required"] == ["segments"]
    assert "messages" not in body
    assert document.cleanup_status == "completed"
    assert (document.input_tokens, document.output_tokens) == (70, 15)


@pytest.mark.parametrize(
    ("raw", "cleaned", "expected"),
    [
        ("Lot 123", "Lot 124", False),
        ("Lot 123", "Lot 123 456", False),
        ("Lot 123 456", "Lot 123", False),
        ("ตรวจ ๑๒๓", "ตรวจ 123", True),
        ("ตรวจ 123", "ตรวจ ๑๒๓", True),
        ("ตรวจ LINE๑๒", "ตรวจ LINE12", True),
        ("ตรวจ LINE12", "ตรวจ LINE13", False),
        ("ลด 5%", "ลด 5 %", True),
        ("น้ำหนัก 140 kg", "น้ำหนัก 140kg", True),
        ("น้ำหนัก 140 kg", "น้ำหนัก 140 ton", False),
        ("น้ำหนัก 140 kg", "น้ำหนัก 140", False),
        ("เวลา 12:30", "เวลา 12:30", True),
        ("ค่า 1.5", "ค่า 1,5", False),
    ],
)
def test_number_code_preservation_canonicalizes_only_digit_scripts(raw, cleaned, expected):
    assert preserves_numbers_and_codes(raw, cleaned) is expected


@pytest.mark.parametrize(
    ("raw", "cleaned", "expected"),
    [
        ("Lot 123 line 456", "Lot 123 line 456 789", ("inserted", 3)),
        ("Lot 123 line 456", "Lot 123 line", ("deleted", 2)),
        ("Lot 123 line 456", "Lot 456 line 123", ("reordered", 1)),
        ("Lot 123 line 456", "Lot 124 line 456", ("changed", 1)),
        ("Lot ๑๒๓", "Lot 123", None),
    ],
)
def test_number_code_mismatch_reports_category_and_one_based_position(
    raw, cleaned, expected
):
    assert number_code_mismatch(raw, cleaned) == expected


def test_cleanup_splits_long_source_without_losing_offsets_or_text():
    raw_text = " ".join(
        f"ตรวจ Lot {index} น้ำหนัก 140 kg" for index in range(900)
    )
    chunks = _split_cleanup_text(raw_text)

    assert len(chunks) > 1
    cursor = 0
    reconstructed = []
    for start, end, text, separator in chunks:
        assert start == cursor
        assert raw_text[start:end] == text
        assert not (
            start > 0
            and raw_text[start - 1].isdigit()
            and raw_text[start].isdigit()
        )
        if separator:
            left = raw_text[:end].split()[-1]
            right = raw_text[end + len(separator):].split()[0]
            assert not (left.casefold() == "lot" and right.isdigit())
            assert not (left.isdigit() and right.casefold() == "kg")
        reconstructed.extend((text, separator))
        cursor = end + len(separator)
    assert cursor == len(raw_text)
    assert "".join(reconstructed) == raw_text


def test_cleanup_reconstructs_long_segment_after_chunked_response():
    raw_text = " ".join(
        f"ตรวจ Lot {index} น้ำหนัก 140 kg" for index in range(900)
    )
    segment = TranscriptSegment(10.0, 40.0, "Speaker A", raw_text)
    session = Mock()

    def echo_chunks(_url, *, json, **_kwargs):
        source_payload = json_module.loads(
            json["messages"][1]["content"].rsplit("Input JSON:\n", 1)[1]
        )
        response = Mock(status_code=200)
        response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json_module.dumps(
                        {"segments": source_payload["segments"]},
                        ensure_ascii=False,
                    )
                }
            }]
        }
        return response

    import json as json_module

    session.post.side_effect = echo_chunks
    document = TranscriptCleanup(
        azure_config(), api_version="v1", session=session
    ).clean((segment,))

    submitted = session.post.call_args.kwargs["json"]["messages"][1]["content"]
    assert len(_split_cleanup_text(raw_text)) > 1
    assert document.cleanup_status == "completed"
    assert len(document.segments) == 1
    assert document.segments[0].raw_text == raw_text
    assert document.segments[0].cleaned_text == raw_text
    assert (document.segments[0].start_seconds, document.segments[0].end_seconds) == (10, 40)
    assert "neighboring excerpts" in submitted


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ({"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}, "IncompleteResponse"),
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
def test_cleanup_rejects_responses_api_refusal_incomplete_or_malformed(body, reason):
    session = Mock()
    response = Mock(status_code=200)
    response.json.return_value = {
        **body,
        "usage": {"input_tokens": 11, "output_tokens": 3},
    }
    session.post.return_value = response

    result = TranscriptCleanup(
        AzureConfig(
            **{
                **azure_config().__dict__,
                "text_endpoint": (
                    "https://text.example.test/openai/responses"
                    "?api-version=2025-04-01-preview"
                ),
            }
        ),
        api_version="v1",
        session=session,
    ).clean((TranscriptSegment(1, 2, None, "source"),))

    assert result.cleanup_status == "review_required"
    assert result.review_reason == reason
    assert (result.input_tokens, result.output_tokens) == (11, 3)


def test_cleanup_does_not_retry_responses_404():
    session = Mock()
    response = Mock(status_code=404, headers={"apim-request-id": "req-1"})
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

    result = TranscriptCleanup(
        config, api_version="v1", session=session
    ).clean((TranscriptSegment(1, 2, None, "source"),))

    assert result.cleanup_status == "review_required"
    session.post.assert_called_once()


@pytest.mark.parametrize(
    ("cleaned", "expected_rule"),
    [
        ({"segments": [{"index": 0, "text": "ตรวจ"}]}, "CleanedTextTooShort"),
        (
            {"segments": [{"index": 1, "text": "ตรวจ Encoder Lot 123 ที่ LINE1"}]},
            "SegmentIndexMismatch",
        ),
        (
            {"segments": [{"index": 0, "text": "ตรวจ Encoder Lot 124 ที่ LINE1"}]},
            "NumberOrCodeMismatch",
        ),
    ],
)
def test_cleanup_reports_precise_content_free_validation_rule(
    cleaned, expected_rule, caplog
):
    segments = (
        TranscriptSegment(1.0, 3.0, "Speaker A", "ตรวจ Encoder Lot 123 ที่ LINE1"),
    )
    response = Mock(status_code=200)
    response.json.return_value = {
        "choices": [{"message": {"content": json.dumps(cleaned, ensure_ascii=False)}}],
    }
    session = Mock()
    session.post.return_value = response

    with caplog.at_level(logging.WARNING, logger="app.transcript_cleanup"):
        document = TranscriptCleanup(
            azure_config(), api_version="2025-04-01-preview", session=session
        ).clean(segments)

    assert document.cleanup_status == "review_required"
    assert document.review_reason == f"ValidationFailed:{expected_rule}"
    assert document.segments[0].raw_text == segments[0].raw_text
    assert document.segments[0].cleaned_text is None
    assert document.segments[0].review_required
    assert f"rule={expected_rule}" in caplog.text
    assert segments[0].raw_text not in caplog.text
    assert "123" not in caplog.text
    assert "124" not in caplog.text
    if expected_rule == "NumberOrCodeMismatch":
        assert "input_number_code_count=2" in caplog.text
        assert "output_number_code_count=2" in caplog.text
        assert "numeric_mismatch_category=changed" in caplog.text
        assert "numeric_mismatch_token_position=1" in caplog.text


@pytest.mark.parametrize(
    ("payload", "expected_rule"),
    [
        ([], "PayloadNotObject"),
        ({"segments": "invalid"}, "SegmentsFieldNotArray"),
        ({"segments": []}, "SegmentCountMismatch"),
        ({"segments": [None]}, "SegmentItemNotObject"),
        ({"segments": [{"index": 0, "text": None}]}, "CleanedTextNotString"),
        ({"segments": [{"index": 0, "text": ""}]}, "CleanedTextEmpty"),
    ],
)
def test_cleanup_reports_shape_and_count_validation_rules(payload, expected_rule, caplog):
    response = Mock(status_code=200)
    response.json.return_value = {
        "choices": [{
            "message": {
                "content": json.dumps(payload, ensure_ascii=False),
            }
        }]
    }
    session = Mock()
    session.post.return_value = response
    segment = TranscriptSegment(1, 2, None, "sentinel transcript")

    with caplog.at_level(logging.WARNING, logger="app.transcript_cleanup"):
        document = TranscriptCleanup(
            azure_config(), api_version="v1", session=session
        ).clean((segment,))

    assert document.review_reason == f"ValidationFailed:{expected_rule}"
    assert f"rule={expected_rule}" in caplog.text
    assert "sentinel transcript" not in caplog.text


def test_cleanup_short_output_diagnostic_reports_counts_without_text(caplog):
    original = "x" * 8267
    cleaned = "x" * 6613
    response = Mock(status_code=200)
    response.json.return_value = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "segments": [{"index": 0, "text": cleaned}],
                }),
            }
        }]
    }
    session = Mock()
    session.post.return_value = response

    with caplog.at_level(logging.WARNING, logger="app.transcript_cleanup"):
        document = TranscriptCleanup(
            azure_config(), api_version="v1", session=session
        ).clean((TranscriptSegment(1, 2, None, original),))

    assert document.review_reason == "ValidationFailed:CleanedTextTooShort"
    assert "input_chars=8267" in caplog.text
    assert "output_chars=6613" in caplog.text
    assert "minimum_output_chars=6614" in caplog.text
    assert original not in caplog.text
    assert cleaned not in caplog.text


def test_cleanup_failure_preserves_raw_transcript_for_review():
    segment = TranscriptSegment(1, 2, "Speaker A", "บทสนทนา")
    session = Mock()
    session.post.side_effect = requests.exceptions.Timeout(
        "do not log request content"
    )

    document = TranscriptCleanup(
        azure_config(), api_version="2025-04-01-preview", session=session
    ).clean((segment,))

    assert document.cleanup_status == "review_required"
    assert document.raw_transcript == "บทสนทนา"
    assert document.segments[0].review_required


def test_timestamp_validation_rejects_out_of_order_segments():
    segments = (
        TranscriptSegment(2.0, 3.0, "Speaker A", "later"),
        TranscriptSegment(1.0, 2.0, "Speaker B", "earlier"),
    )
    with pytest.raises(ValueError, match="chronological"):
        validate_segment_times(segments)


def test_segmented_pdf_renders_unknown_or_actual_speaker_without_fabrication(
    tmp_path: Path,
):
    source = tmp_path / "meeting-20261107-0817.mp3"
    source.write_bytes(b"fixture audio")
    pdf = tmp_path / "preview.pdf"
    document = TranscriptDocument(
        (
            TranscriptSegment(80, 87, None, "ผู้พูดไม่ทราบ"),
            TranscriptSegment(87, 95, "Speaker A", "ตรวจ Encoder"),
        ),
        "available",
        "completed",
    )

    write_transcript_pdf(pdf, source, "Site", document)
    text = "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf)).pages)

    assert "Morning Meeting - Full Transcript" in text
    assert "[00:01:20]" in text
    assert "Speaker unknown" not in text
    assert "Speaker A" in text
    assert "[segment 2]" in text
    assert "[00:01:20]" in text
    assert "ผู้พูดไม่ทราบ" in text


def test_existing_segmented_pdf_is_not_overwritten(tmp_path: Path):
    source = tmp_path / "meeting.mp3"
    source.write_bytes(b"audio")
    pdf = tmp_path / "existing.pdf"
    pdf.write_bytes(b"preserve")
    document = TranscriptDocument((), "unavailable", "review_required")

    with pytest.raises(FileExistsError):
        write_transcript_pdf(pdf, source, "", document)
    assert pdf.read_bytes() == b"preserve"


def test_fixture_preview_does_not_make_azure_calls(tmp_path: Path, monkeypatch):
    import app.phase9 as phase9

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Fixture preview must be fully offline.")

    monkeypatch.setattr(phase9, "AzureSpeechDiarizer", forbidden)
    monkeypatch.setattr(phase9, "AzureOpenAITranscriber", forbidden)
    monkeypatch.setattr(phase9, "TranscriptCleanup", forbidden)

    summary = phase9.run_fixture_preview(tmp_path)
    full = summary.with_name(f"{summary.stem}_Full.pdf")
    summary_text = "\n".join(
        page.extract_text() or "" for page in PdfReader(str(summary)).pages
    )
    full_text = "\n".join(
        page.extract_text() or "" for page in PdfReader(str(full)).pages
    )
    assert "ตัวอย่างจัดหน้าเท่านั้น" in summary_text
    assert "FIXTURE TEXT, NOT AN ACTUAL TRANSCRIPTION" in full_text
    assert summary.is_file() and full.is_file()


def test_controlled_live_test_requires_both_explicit_authorizations(tmp_path: Path):
    import app.phase9 as phase9

    with pytest.raises(ValueError, match="Azure billing and MSSQL write"):
        phase9.run_controlled_live_test(
            input_dir=tmp_path,
            confirm_azure_billing=True,
        )


def test_controlled_target_is_found_recursively_and_requires_unique_stable_file(
    tmp_path: Path, monkeypatch,
):
    import app.phase9 as phase9

    source = tmp_path / "CB" / "Nov26" / "meeting-20261107-0817.mp3"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"fixture audio")
    monkeypatch.setattr(phase9, "is_file_stable", lambda _path: True)

    assert phase9.locate_controlled_target(tmp_path) == source

    duplicate = tmp_path / "another" / source.name
    duplicate.parent.mkdir()
    duplicate.write_bytes(b"second file")
    with pytest.raises(FileNotFoundError, match="found 2"):
        phase9.locate_controlled_target(tmp_path)


def test_live_test_checks_database_migration_before_any_azure_request(
    tmp_path: Path, monkeypatch,
):
    import app.phase9 as phase9

    source = tmp_path / "CB" / "Nov26" / "meeting-20261107-0817.mp3"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"fixture audio")
    monkeypatch.setattr(phase9, "is_file_stable", lambda _path: True)
    monkeypatch.setattr(phase9, "get_azure_config", azure_config)
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "test-version")
    connection = Mock()
    repository = Mock()
    repository.phase9_schema_ready.return_value = False
    monkeypatch.setattr(phase9, "connect_sql", lambda _config: connection)
    monkeypatch.setattr(phase9, "SqlRepository", lambda _connection: repository)
    monkeypatch.setattr(
        phase9,
        "AzureSpeechDiarizer",
        Mock(side_effect=AssertionError("Azure must not be called before schema preflight.")),
    )

    with pytest.raises(RuntimeError, match="no Azure request was made"):
        phase9.run_controlled_live_test(
            input_dir=tmp_path,
            output_dir=tmp_path / "preview",
            confirm_azure_billing=True,
            confirm_mssql_write=True,
        )

    connection.close.assert_called_once()
