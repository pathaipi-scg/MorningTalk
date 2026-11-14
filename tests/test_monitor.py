from pathlib import Path
import logging

import pytest

import app.main as main_module
from app.main import build_parser, main
from app.monitor import (
    MonitorAlreadyRunning,
    monitor_instance_lock,
    run_monitor,
)
from app.scanner import scan_directory
from app.file_manager import pdf_path_for_source
import app.file_manager as file_manager
from app.auto_scan import synchronize_scan
from unittest.mock import Mock


def test_monitor_defaults_to_database_scan_only_with_300_second_interval():
    args = build_parser().parse_args(["monitor"])
    assert args.interval == 300
    assert not args.process
    assert not args.local_only
    assert not args.once


def test_single_monitor_cycle_syncs_database_without_enabling_azure(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MORNINGTALK_WORKSPACE_ROOT", str(tmp_path))
    calls = []

    def scan_cycle(input_dir, **options):
        calls.append((input_dir, options))
        return 0

    monkeypatch.setattr(main_module, "_scan_once", scan_cycle)
    monkeypatch.setattr(
        main_module,
        "get_azure_config",
        lambda: (_ for _ in ()).throw(AssertionError("Azure config must not be loaded.")),
    )
    assert main(["monitor", "--once", "--input-dir", str(tmp_path)]) == 0
    assert calls == [(tmp_path.resolve(), {"sync_db": True, "process_enabled": False})]


def test_monitor_retries_failed_scan_on_next_cycle_and_shuts_down_gracefully(
    tmp_path: Path, monkeypatch,
):
    monkeypatch.setenv("MORNINGTALK_WORKSPACE_ROOT", str(tmp_path))
    calls = []
    results = iter((2, 0))

    def scan_cycle(_input_dir, **_options):
        calls.append(1)
        return next(results)

    sleep_calls = []

    def stop_after_second_cycle(seconds):
        sleep_calls.append(seconds)
        if len(sleep_calls) == 1:
            return
        raise KeyboardInterrupt

    monkeypatch.setattr("app.monitor.time.sleep", stop_after_second_cycle)
    try:
        assert run_monitor(scan_cycle, tmp_path, interval_seconds=17) == 0
    finally:
        for handler in list(logging.getLogger().handlers):
            if getattr(handler, "_morningtalk_file_handler", False):
                logging.getLogger().removeHandler(handler)
                handler.close()
            if getattr(handler, "_morningtalk_console_handler", False):
                logging.getLogger().removeHandler(handler)
                handler.close()
    assert len(calls) == 2
    assert sleep_calls == [17, 17]
    log_text = (tmp_path / "logs" / "monitor.log").read_text(encoding="utf-8")
    assert "will retry next cycle" in log_text
    assert "Monitor stopped by user" in log_text


def test_monitor_lock_prevents_a_second_live_instance(tmp_path: Path):
    lock = tmp_path / "monitor.lock"
    with monitor_instance_lock(lock):
        with pytest.raises(MonitorAlreadyRunning):
            with monitor_instance_lock(lock):
                pass
    assert not lock.exists()


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("CB") / "Dec26" / "meeting.mp3",
        Path("CB") / "Jan27" / "meeting.mp3",
        Path("CB") / "2027" / "Jan" / "meeting.mp3",
        Path("CB") / "2027" / "Feb" / "meeting.mp3",
        Path("CB") / "Dec26" / "Archive" / "meeting.mp3",
    ],
)
def test_scanner_discovers_calendar_and_arbitrarily_nested_folders(
    tmp_path: Path, monkeypatch, relative_path: Path,
):
    root = tmp_path / "Record File"
    source = root / relative_path
    source.parent.mkdir(parents=True)
    source.write_bytes(b"audio")
    monkeypatch.setattr("app.scanner.is_file_stable", lambda *_args: True)

    items = scan_directory(root, stability_interval_seconds=0)

    assert [item.path for item in items] == [source]
    assert items[0].pdf_path == pdf_path_for_source(source)


def test_same_mp3_basename_in_different_folders_is_tracked_by_full_path(
    tmp_path: Path, monkeypatch,
):
    root = tmp_path / "Record File"
    sources = [
        root / folder / "Nov26" / "meeting.mp3"
        for folder in ("first", "second")
    ]
    for source in sources:
        source.parent.mkdir(parents=True)
        source.write_bytes(b"audio")
    monkeypatch.setattr("app.scanner.is_file_stable", lambda *_args: True)

    items = scan_directory(root, stability_interval_seconds=0)

    assert [item.path for item in items] == sorted(sources)
    assert len({str(item.path.resolve()).casefold() for item in items}) == 2
    assert all(item.pdf_path == pdf_path_for_source(item.path) for item in items)


def test_monitor_scan_discovers_folder_created_after_monitor_start(
    tmp_path: Path, monkeypatch,
):
    root = tmp_path / "Record File"
    root.mkdir()
    monkeypatch.setattr("app.scanner.is_file_stable", lambda *_args: True)

    assert scan_directory(root, stability_interval_seconds=0) == []

    source = root / "CB" / "Dec26" / "new.mp3"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"new audio")
    assert [item.path for item in scan_directory(root, stability_interval_seconds=0)] == [source]


def test_partially_copied_mp3_is_marked_unstable_and_not_synchronized(
    tmp_path: Path, monkeypatch,
):
    root = tmp_path / "Record File"
    source = root / "CB" / "Nov26" / "nested" / "copy.mp3"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"partial")
    monkeypatch.setattr("app.scanner.is_file_stable", lambda *_args: False)
    repository = Mock()

    items = scan_directory(root, stability_interval_seconds=0)
    results = synchronize_scan(items, repository)

    assert results[0][2] == "unstable"
    repository.upsert_scan.assert_not_called()


def test_missing_input_root_fails_instead_of_appearing_empty(tmp_path: Path):
    missing_root = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        scan_directory(missing_root)


def test_monitor_retries_after_temporary_database_failure(
    tmp_path: Path, monkeypatch,
):
    monkeypatch.setenv("MORNINGTALK_WORKSPACE_ROOT", str(tmp_path))
    calls = []

    def scan_cycle(_input_dir, **_options):
        calls.append(None)
        if len(calls) == 1:
            raise ConnectionError("temporary database outage")
        raise KeyboardInterrupt

    monkeypatch.setattr("app.monitor.time.sleep", lambda _seconds: None)
    assert run_monitor(scan_cycle, tmp_path, interval_seconds=1) == 0
    assert len(calls) == 2
    log_text = (tmp_path / "logs" / "monitor.log").read_text(
        encoding="utf-8"
    )
    assert "error_class=ConnectionError" in log_text
    assert "temporary database outage" not in log_text


def test_new_mp3_is_queued_in_database_and_existing_pdf_is_untouched(
    tmp_path: Path, monkeypatch,
):
    monkeypatch.setattr(
        file_manager,
        "DEFAULT_TRANSCRIPT_ROOT",
        tmp_path / "Transcript",
    )
    root = tmp_path / "input"
    new_source = root / "Site" / "Nov26" / "new.mp3"
    old_source = root / "Site" / "Nov26" / "review.mp3"
    new_source.parent.mkdir(parents=True)
    new_source.write_bytes(b"new audio")
    old_source.write_bytes(b"old audio")
    old_pdf = pdf_path_for_source(old_source)
    old_pdf.parent.mkdir(parents=True)
    old_pdf.write_bytes(b"preserve this pdf")
    pdf_before = old_pdf.read_bytes()
    monkeypatch.setattr("app.scanner.is_file_stable", lambda *_args: True)
    items = scan_directory(root, stability_interval_seconds=0)
    repository = Mock()
    repository.upsert_scan.side_effect = [(41, "discovered"), (42, "review_required")]
    results = synchronize_scan(items, repository)
    assert [status for _item, _audio_id, status in results] == [
        "discovered", "review_required"
    ]
    assert repository.upsert_scan.call_count == 2
    first_scan_args = repository.upsert_scan.call_args_list[0].args
    assert Path(first_scan_args[0]) == new_source
    assert first_scan_args[1] == items[0].fingerprint
    assert first_scan_args[2] == new_source.stat().st_size
    assert old_pdf.read_bytes() == pdf_before
    assert not pdf_path_for_source(new_source).exists()
