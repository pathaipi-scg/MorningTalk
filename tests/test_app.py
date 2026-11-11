from contextlib import contextmanager
import logging
from pathlib import Path
import subprocess
from unittest.mock import Mock
import hashlib

import pytest
import requests
from reportlab.pdfbase.ttfonts import TTFont as ReportLabTTFont
from pypdf import PdfReader

from app.config import AzureConfig
from app.file_manager import discover_mp3_files, meeting_datetime, pdf_path_for_source, source_processing_lock
from app.main import process_file, main
from app.pdf_export import write_pdf
import app.config as config_module
import app.transcriber as transcriber_module
from app.transcriber import (
    AzureOpenAITranscriber, AzureTranscriptionError, _merge_overlapping_transcripts,
    estimate_chunk_count,
)


@pytest.fixture(autouse=True)
def fake_audio_tools(monkeypatch):
    monkeypatch.setattr(config_module.shutil, "which", lambda name: f"/test-bin/{name}")


def make_azure_config(
    *,
    endpoint="https://example.openai.azure.com",
    api_key="test-transcription-key",
    deployment="gpt-4o-transcribe",
    api_version="2025-03-01-preview",
    chunk_duration_seconds=180,
    chunk_overlap_seconds=10,
    quality_warning_chars_per_minute=10.0,
):
    return AzureConfig(
        text_endpoint="https://text-resource.openai.azure.com",
        text_api_key="test-text-key",
        text_deployment="gpt-5.4-mini",
        transcription_endpoint=endpoint,
        transcription_api_key=api_key,
        transcription_deployment=deployment,
        transcription_api_version=api_version,
        chunk_duration_seconds=chunk_duration_seconds,
        chunk_overlap_seconds=chunk_overlap_seconds,
        quality_warning_chars_per_minute=quality_warning_chars_per_minute,
    )


@pytest.fixture
def single_audio_chunk(monkeypatch):
    @contextmanager
    def fake_chunks(_self, audio_path):
        yield [audio_path]
    monkeypatch.setattr(AzureOpenAITranscriber, "_audio_chunks", fake_chunks)
    monkeypatch.setattr(AzureOpenAITranscriber, "_probe_duration", staticmethod(lambda _path: 60.0))


def test_recursive_discovery_and_adjacent_output_mapping(tmp_path: Path):
    root = tmp_path / "input"
    source = root / "Site" / "Month" / "meeting-20261104-0815.MP3"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"sample")
    assert discover_mp3_files(root) == [source]
    assert pdf_path_for_source(source) == source.with_suffix(".pdf")
    assert pdf_path_for_source(source).name == "meeting-20261104-0815.pdf"


def test_datetime_parsing():
    assert meeting_datetime("meeting-20261104-0815.mp3").strftime("%Y-%m-%d %H:%M") == "2026-11-04 08:15"
    assert meeting_datetime("bad.mp3") is None


def test_existing_pdf_is_skipped_without_transcription(tmp_path: Path):
    root = tmp_path / "input"
    source = root / "Plant" / "Month" / "m.mp3"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"audio")
    pdf = source.with_suffix(".pdf")
    pdf.write_bytes(b"existing")
    transcriber = Mock()
    assert not process_file(source, root, transcriber)
    transcriber.transcribe.assert_not_called()
    assert pdf.read_bytes() == b"existing"


def test_failed_processing_can_retry(tmp_path: Path):
    root = tmp_path / "input"
    source = root / "Plant" / "m.mp3"
    source.parent.mkdir(parents=True)
    original = b"original"
    source.write_bytes(original)
    transcriber = Mock()
    transcriber.transcribe.side_effect = [RuntimeError("temporary"), "สวัสดี"]
    with pytest.raises(RuntimeError):
        process_file(source, root, transcriber)
    assert process_file(source, root, transcriber)
    assert source.read_bytes() == original
    assert source.with_suffix(".pdf").is_file()
    assert not source.with_suffix(".txt").exists()

def test_force_overwrites_existing_pdf_only(tmp_path: Path):
    root = tmp_path / "input"
    source = root / "Plant" / "meeting.mp3"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"audio")
    pdf = source.with_suffix(".pdf")
    pdf.write_bytes(b"old pdf")
    transcriber = Mock()
    transcriber.transcribe.return_value = "ข้อความใหม่"
    assert process_file(source, root, transcriber, force=True)
    assert pdf.read_bytes().startswith(b"%PDF")
    assert not source.with_suffix(".txt").exists()


def test_azure_transcription_uses_mock_and_thai_language(tmp_path: Path, single_audio_chunk):
    source = tmp_path / "meeting.mp3"
    source.write_bytes(b"mock audio")
    response = Mock(status_code=200)
    response.json.return_value = {"text": "ประชุมเริ่มแล้ว"}
    session = Mock()
    uploaded_audio = []

    def fake_post(url, **kwargs):
        uploaded_audio.append(kwargs["files"]["file"][1].read())
        return response

    session.post.side_effect = fake_post
    config = make_azure_config()
    assert AzureOpenAITranscriber(config, session).transcribe(source) == "ประชุมเริ่มแล้ว"
    call = session.post.call_args
    assert call.args[0] == (
        "https://example.openai.azure.com/openai/deployments/"
        "gpt-4o-transcribe/audio/transcriptions"
    )
    assert call.kwargs["params"] == {"api-version": "2025-03-01-preview"}
    assert call.kwargs["headers"]["api-key"] == "test-transcription-key"
    assert call.kwargs["data"] == {"language": "th"}
    assert call.kwargs["timeout"] == (30, 1800)
    assert call.kwargs["files"]["file"][0] == "audio.mp3"
    assert call.kwargs["files"]["file"][2] == "audio/mpeg"
    assert uploaded_audio == [b"mock audio"]


def test_v1_api_uses_transcription_deployment_as_model(tmp_path: Path, single_audio_chunk):
    source = tmp_path / "meeting.mp3"
    source.write_bytes(b"mock audio")
    response = Mock(status_code=200)
    response.json.return_value = {"text": "ประชุมเริ่มแล้ว"}
    session = Mock()
    session.post.return_value = response
    config = make_azure_config(deployment="audio-deployment", api_version="v1")
    assert AzureOpenAITranscriber(config, session).transcribe(source) == "ประชุมเริ่มแล้ว"
    call = session.post.call_args
    assert call.args[0] == "https://example.openai.azure.com/openai/v1/audio/transcriptions"
    assert call.kwargs["params"] is None
    assert call.kwargs["data"]["model"] == "audio-deployment"
    assert call.kwargs["data"]["language"] == "th"


def test_large_audio_is_split_with_boundary_overlap(tmp_path: Path, monkeypatch):
    source = tmp_path / "large.mp3"
    source.write_bytes(b"large")
    monkeypatch.setattr(transcriber_module.shutil, "which", lambda command: command)
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if command[0] == "ffprobe":
            return subprocess.CompletedProcess(command, 0, stdout="130\n", stderr="")
        Path(command[-1]).write_bytes(b"chunk")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(transcriber_module.subprocess, "run", fake_run)
    config = make_azure_config(chunk_duration_seconds=60, chunk_overlap_seconds=10)
    adapter = AzureOpenAITranscriber(config, Mock())
    with adapter._audio_chunks(source) as chunks:
        assert len(chunks) == 3
        starts = [command[command.index("-ss") + 1] for command in calls if command[0] == "ffmpeg"]
        assert starts == ["0.000", "50.000", "100.000"]


def test_chunk_transcriptions_are_ordered_and_overlap_is_removed(tmp_path: Path, monkeypatch):
    source = tmp_path / "large.mp3"
    source.write_bytes(b"large")
    first = tmp_path / "first.mp3"
    second = tmp_path / "second.mp3"
    first.write_bytes(b"1")
    second.write_bytes(b"2")

    @contextmanager
    def fake_chunks(_self, _audio_path):
        yield [first, second]

    monkeypatch.setattr(AzureOpenAITranscriber, "_audio_chunks", fake_chunks)
    monkeypatch.setattr(AzureOpenAITranscriber, "_probe_duration", staticmethod(lambda _path: 180.0))
    responses = []
    for text in ("สวัสดีประชุมวันนี้", "ประชุมวันนี้เริ่มแล้ว"):
        response = Mock(status_code=200)
        response.json.return_value = {"text": text}
        responses.append(response)
    session = Mock()
    session.post.side_effect = responses
    config = make_azure_config()
    assert AzureOpenAITranscriber(config, session).transcribe(source) == "สวัสดีประชุมวันนี้เริ่มแล้ว"
    assert [call.kwargs["files"]["file"][0] for call in session.post.call_args_list] == [
        "audio.mp3", "audio.mp3"
    ]


@pytest.mark.parametrize("status_code", [401, 403, 404, 429, 500])
def test_azure_http_errors_are_safe_to_report(tmp_path: Path, status_code: int, single_audio_chunk):
    source = tmp_path / "meeting.mp3"
    source.write_bytes(b"mock audio")
    response = Mock(status_code=status_code)
    session = Mock()
    session.post.return_value = response
    config = make_azure_config(api_key="secret-test-value")
    with pytest.raises(AzureTranscriptionError, match=f"HTTP {status_code}") as error:
        AzureOpenAITranscriber(config, session).transcribe(source)
    assert "secret-test-value" not in str(error.value)


@pytest.mark.parametrize(
    "exception_type",
    [
        requests.exceptions.ConnectTimeout,
        requests.exceptions.ReadTimeout,
        requests.exceptions.ConnectionError,
        requests.exceptions.ProxyError,
        requests.exceptions.SSLError,
        requests.exceptions.RequestException,
    ],
)
def test_request_exception_logs_only_safe_class_chain(
    tmp_path: Path,
    caplog,
    exception_type,
    single_audio_chunk,
):
    source = tmp_path / "meeting.mp3"
    source.write_bytes(b"mock audio")
    session = Mock()
    request_error = exception_type("secret URL and key should never appear")
    request_error.__cause__ = OSError("secret nested cause")
    session.post.side_effect = request_error
    config = make_azure_config(api_key="secret-test-value")
    with caplog.at_level(logging.ERROR, logger="app.transcriber"):
        with pytest.raises(AzureTranscriptionError) as raised:
            AzureOpenAITranscriber(config, session).transcribe(source)
    assert type(request_error).__name__ in caplog.text
    assert "OSError" in caplog.text
    assert "secret" not in caplog.text
    assert "secret" not in str(raised.value)


def test_invalid_azure_response_is_rejected(tmp_path: Path, single_audio_chunk):
    source = tmp_path / "meeting.mp3"
    source.write_bytes(b"mock audio")
    response = Mock(status_code=200)
    response.json.return_value = {"not_text": "unexpected"}
    session = Mock()
    session.post.return_value = response
    with pytest.raises(ValueError, match="did not contain transcript text"):
        AzureOpenAITranscriber(make_azure_config(), session).transcribe(source)


def test_overlap_without_match_retains_both_texts(caplog):
    assert _merge_overlapping_transcripts("คำพูดจบ", "เริ่มใหม่") == "คำพูดจบ\nเริ่มใหม่"
    assert "Could not align transcript overlap" in caplog.text


def test_thai_pdf_atomic_and_extractable(tmp_path: Path, monkeypatch):
    source = tmp_path / "meeting-20261104-0815.mp3"
    source.write_bytes(b"audio")
    pdf = source.with_suffix(".pdf")
    text = "สวัสดีครับ\nรายงานการประชุม"
    write_pdf(pdf, source, "Site", text)
    assert pdf.read_bytes().startswith(b"%PDF")
    import pypdf.filters
    monkeypatch.setattr(pypdf.filters, "JBIG2DEC_BINARY", None)
    extracted = "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf)).pages)
    assert "สวัสดีครับ" in extracted
    assert "รายงานการประชุม" in extracted
    font = ReportLabTTFont("ThaiCoverageTest", r"C:\Windows\Fonts\tahoma.ttf")
    assert all(font.face.charToGlyph[ord(char)] for char in "สวัสดีครับ")


def test_atomic_pdf_write_does_not_leave_partial_file(tmp_path: Path):
    source = tmp_path / "meeting.mp3"
    source.write_bytes(b"audio")
    pdf = source.with_suffix(".pdf")
    write_pdf(pdf, source, "Site", "สวัสดี")
    first_contents = pdf.read_bytes()
    with pytest.raises(FileExistsError):
        write_pdf(pdf, source, "Site", "ใหม่")
    assert pdf.read_bytes() == first_contents
    assert not list(tmp_path.glob("*.tmp.pdf"))


def test_dry_run_does_not_call_azure(tmp_path: Path, capsys, monkeypatch):
    root = tmp_path / "input"
    source = root / "Site" / "meeting.mp3"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"audio")
    monkeypatch.setattr(AzureOpenAITranscriber, "_probe_duration", staticmethod(lambda _path: 200.0))
    assert main(["--input-dir", str(root), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert f"{source} -> {source.with_suffix('.pdf')}" in output
    assert "chunks=2" in output


def test_chunk_estimation_and_covering_overlapped_boundaries():
    assert estimate_chunk_count(180) == 1
    assert estimate_chunk_count(181) == 2
    assert estimate_chunk_count(350) == 2
    assert estimate_chunk_count(351) == 3


def test_empty_chunk_fails_instead_of_being_omitted(tmp_path: Path, monkeypatch, caplog):
    source = tmp_path / "large.mp3"
    source.write_bytes(b"audio")
    first, second = tmp_path / "first.mp3", tmp_path / "second.mp3"
    first.write_bytes(b"1")
    second.write_bytes(b"2")

    @contextmanager
    def fake_chunks(_self, _audio_path):
        yield [first, second]

    monkeypatch.setattr(AzureOpenAITranscriber, "_audio_chunks", fake_chunks)
    monkeypatch.setattr(AzureOpenAITranscriber, "_probe_duration", staticmethod(lambda _path: 180.0))
    ok = Mock(status_code=200)
    ok.json.return_value = {"text": "ได้ยินข้อความ"}
    empty = Mock(status_code=200)
    empty.json.return_value = {"text": " "}
    session = Mock()
    session.post.side_effect = [ok, empty]
    with pytest.raises(ValueError, match="did not contain transcript text"):
        AzureOpenAITranscriber(make_azure_config(), session).transcribe(source)
    assert "status=failed" in caplog.text
    assert session.post.call_count == 2


def test_source_integrity_preserved_during_pdf_processing(tmp_path: Path):
    root = tmp_path / "input"
    source = root / "Plant" / "Month" / "m.mp3"
    source.parent.mkdir(parents=True)
    original = b"immutable source bytes"
    source.write_bytes(original)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    transcriber = Mock()
    transcriber.transcribe.return_value = "บทสนทนา"
    assert process_file(source, root, transcriber)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest


def test_duplicate_processing_is_blocked_by_source_lock(tmp_path: Path):
    root = tmp_path / "input"
    source = root / "Plant" / "Month" / "meeting.mp3"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"audio")
    transcriber = Mock()
    with source_processing_lock(source):
        with pytest.raises(RuntimeError, match="already handling"):
            process_file(source, root, transcriber)
    transcriber.transcribe.assert_not_called()


def test_short_transcript_logs_quality_warning(tmp_path: Path, single_audio_chunk, caplog):
    source = tmp_path / "short.mp3"
    source.write_bytes(b"audio")
    response = Mock(status_code=200)
    response.json.return_value = {"text": "ใช่"}
    session = Mock()
    session.post.return_value = response
    assert AzureOpenAITranscriber(make_azure_config(), session).transcribe(source) == "ใช่"
    assert "Transcript quality warning" in caplog.text


def test_configuration_requires_transcription_deployment():
    config = make_azure_config(deployment="")
    with pytest.raises(ValueError, match="AZURE_TRANSCRIPTION_DEPLOYMENT"):
        config.validate_transcription()


def test_configuration_rejects_unsupported_transcription_api_version():
    config = make_azure_config(api_version="2024-06-01")
    with pytest.raises(ValueError, match="Unsupported AZURE_TRANSCRIPTION_API_VERSION"):
        config.validate_transcription()


def test_configuration_rejects_non_https_endpoint():
    with pytest.raises(ValueError, match="must be an HTTPS URL"):
        make_azure_config(endpoint="http://example.openai.azure.com").validate_transcription()


def test_configuration_requires_ffmpeg_and_ffprobe(monkeypatch):
    monkeypatch.setattr(config_module.shutil, "which", lambda name: None if name == "ffprobe" else "ffmpeg")
    with pytest.raises(RuntimeError, match="ffprobe"):
        make_azure_config().validate_transcription()


def test_configuration_reads_transcription_and_text_settings_separately(monkeypatch):
    values = {
        "AZURE_OPENAI_ENDPOINT": "https://text.openai.azure.com",
        "AZURE_OPENAI_API_KEY": "text-secret",
        "AZURE_OPENAI_DEPLOYMENT": "gpt-5.4-mini",
        "AZURE_TRANSCRIPTION_ENDPOINT": "https://speech.openai.azure.com",
        "AZURE_TRANSCRIPTION_API_KEY": "transcription-secret",
        "AZURE_TRANSCRIPTION_DEPLOYMENT": "gpt-4o-transcribe",
        "AZURE_TRANSCRIPTION_API_VERSION": "2025-03-01-preview",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    config = config_module.get_azure_config()
    assert config.text_endpoint == values["AZURE_OPENAI_ENDPOINT"]
    assert config.text_api_key == values["AZURE_OPENAI_API_KEY"]
    assert config.text_deployment == values["AZURE_OPENAI_DEPLOYMENT"]
    assert config.transcription_endpoint == values["AZURE_TRANSCRIPTION_ENDPOINT"]
    assert config.transcription_api_key == values["AZURE_TRANSCRIPTION_API_KEY"]
    assert config.transcription_deployment == values["AZURE_TRANSCRIPTION_DEPLOYMENT"]
    assert config.transcription_api_version == values["AZURE_TRANSCRIPTION_API_VERSION"]
