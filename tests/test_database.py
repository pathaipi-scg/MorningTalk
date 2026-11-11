from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock
import uuid

import pytest

from app.config import SqlConfig
from app.database import SqlRepository, connect_sql, retry_delay_seconds
from app.scanner import classify_scanned_file, scan_directory
from app.database import AudioRecord
from app.auto_scan import process_scanned_file
from app.scanner import ScannedAudio
import app.auto_scan as auto_scan_module


def make_sql_config(**overrides):
    values = {
        "driver": "ODBC Driver 18 for SQL Server",
        "server": "sql.example,1433",
        "database": "MorningTalk",
        "username": "app-login",
        "password": "test-secret",
        "encrypt": True,
        "trust_server_certificate": False,
        "connection_timeout": 15,
        "expected_database": "MorningTalk",
        "expected_login": "app-login",
    }
    values.update(overrides)
    return SqlConfig(**values)


def test_sql_config_requires_installed_driver_without_showing_values():
    config = make_sql_config()
    with pytest.raises(ValueError, match="driver is not installed"):
        config.validate({"SQL Server"})
    config.validate({"ODBC Driver 18 for SQL Server"})


def test_connect_sql_validates_database_and_login():
    connection = Mock()
    cursor = connection.cursor.return_value
    cursor.fetchone.return_value = ("MorningTalk", "app-login")
    connector = Mock(return_value=connection)
    result = connect_sql(make_sql_config(), connector=connector)
    assert result is connection
    assert "DATABASE={MorningTalk}" in connector.call_args.args[0]
    assert "PWD={test-secret}" in connector.call_args.args[0]
    cursor.execute.assert_called_once_with("SELECT DB_NAME(), SUSER_SNAME()")


def test_connect_sql_closes_mismatched_database():
    connection = Mock()
    connection.cursor.return_value.fetchone.return_value = ("OtherDb", "app-login")
    with pytest.raises(RuntimeError, match="does not match"):
        connect_sql(make_sql_config(), connector=Mock(return_value=connection))
    connection.close.assert_called_once()


def test_atomic_claim_uses_conditional_update_and_returns_attempt():
    connection = Mock()
    cursor = connection.cursor.return_value
    cursor.fetchone.return_value = (2,)
    repository = SqlRepository(connection)
    token = uuid.uuid4()
    assert repository.claim_file(17, token) == 2
    query = cursor.execute.call_args.args[0]
    assert "UPDATE dbo.AudioFile WITH (UPDLOCK, ROWLOCK)" in query
    assert "LeaseExpiresUtc <= SYSUTCDATETIME()" in query
    assert "AttemptCount < ?" in query
    connection.commit.assert_called_once()


def test_claim_without_eligible_record_returns_none():
    connection = Mock()
    cursor = connection.cursor.return_value
    cursor.fetchone.return_value = None
    repository = SqlRepository(connection)
    assert repository.claim_file(17, uuid.uuid4()) is None


def test_retry_delay_is_exponential_and_capped():
    assert retry_delay_seconds(1) == 60
    assert retry_delay_seconds(2) == 120
    assert retry_delay_seconds(10) == 3600
    with pytest.raises(ValueError):
        retry_delay_seconds(0)


@pytest.mark.parametrize(
    ("record", "fingerprint", "pdf_exists", "expected"),
    [
        (None, "a", False, "new"),
        (None, "a", True, "review_required"),
        (AudioRecord(1, "x", "a", "completed", "x.pdf"), "a", True, "completed"),
        (AudioRecord(1, "x", "a", "completed", "x.pdf"), "b", False, "modified"),
        (AudioRecord(1, "x", "a", "failed", None), "a", False, "failed"),
        (AudioRecord(1, "x", "a", "review_required", None), "a", False, "review_required"),
    ],
)
def test_file_status_classification(record, fingerprint, pdf_exists, expected):
    assert classify_scanned_file(fingerprint, pdf_exists, record) == expected


def test_scanner_discovers_recursively_and_reports_file_stability(tmp_path: Path, monkeypatch):
    root = tmp_path / "input"
    source = root / "Site" / "Month" / "meeting.mp3"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"audio")
    monkeypatch.setattr("app.scanner.is_file_stable", lambda *_args: True)
    items = scan_directory(root, stability_interval_seconds=0)
    assert len(items) == 1
    assert items[0].status == "new"
    assert items[0].stable
    assert items[0].pdf_path == source.with_suffix(".pdf")


def test_unstable_file_is_never_claimed_for_processing():
    assert classify_scanned_file("a", False, None, stable=False) == "unstable"


def test_failed_processing_does_not_publish_pdf(tmp_path: Path, monkeypatch):
    root = tmp_path / "input"
    source = root / "Site" / "Month" / "meeting.mp3"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"stable audio")
    item = ScannedAudio(
        path=source,
        fingerprint="fingerprint",
        size_bytes=source.stat().st_size,
        modified_utc=datetime.now(timezone.utc),
        pdf_path=source.with_suffix(".pdf"),
        pdf_exists=False,
        stable=True,
        status="new",
    )
    repository = Mock()
    repository.claim_file.return_value = 1
    transcriber = Mock()
    transcriber.transcribe.side_effect = RuntimeError("failure")
    monkeypatch.setattr(
        auto_scan_module,
        "AzureOpenAITranscriber",
        Mock(return_value=transcriber),
    )
    with pytest.raises(RuntimeError):
        process_scanned_file(item, root, 5, repository, object())
    assert not item.pdf_path.exists()
    repository.fail_file.assert_called_once()
    assert repository.fail_file.call_args.args[2] == "RuntimeError"
