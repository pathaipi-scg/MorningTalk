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
from app.auto_scan import synchronize_scan
from unittest.mock import Mock


def test_monitor_defaults_to_database_scan_only_with_300_second_interval():
    args = build_parser().parse_args(["monitor"])
    assert args.interval == 300
    assert not args.process
    assert not args.local_only
    assert not args.once


def test_single_monitor_cycle_syncs_database_without_enabling_azure(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
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
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
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
    log_text = (tmp_path / "MorningTalk" / "logs" / "monitor.log").read_text(encoding="utf-8")
    assert "will retry next cycle" in log_text
    assert "Monitor stopped by user" in log_text


def test_monitor_lock_prevents_a_second_live_instance(tmp_path: Path):
    lock = tmp_path / "monitor.lock"
    with monitor_instance_lock(lock):
        with pytest.raises(MonitorAlreadyRunning):
            with monitor_instance_lock(lock):
                pass
    assert not lock.exists()


def test_new_mp3_is_queued_in_database_and_existing_pdf_is_untouched(
    tmp_path: Path, monkeypatch,
):
    root = tmp_path / "input"
    new_source = root / "Site" / "Month" / "new.mp3"
    old_source = root / "Site" / "Month" / "review.mp3"
    new_source.parent.mkdir(parents=True)
    new_source.write_bytes(b"new audio")
    old_source.write_bytes(b"old audio")
    old_pdf = old_source.with_suffix(".pdf")
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
    assert not new_source.with_suffix(".pdf").exists()
