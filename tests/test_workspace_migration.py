import os
from pathlib import Path

import pytest

from scripts.migrate_runtime_workspace import (
    MigrationError,
    build_plan,
    copy_verified,
    default_sources,
)


def test_verified_copy_preserves_contents_and_timestamp(tmp_path):
    source_root = tmp_path / "old"
    destination_root = tmp_path / "workspace"
    source = source_root / "production_batches" / "batch-one" / "response.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"private response bytes")
    timestamp_ns = 1_700_000_000_123_456_700
    os.utime(source, ns=(timestamp_ns, timestamp_ns))

    plan = build_plan([(source_root, destination_root)])
    result = copy_verified(plan)
    destination = destination_root / "production_batches" / "batch-one" / "response.bin"

    assert result == {
        "copied_files": 1,
        "already_verified_files": 0,
        "total_files": 1,
        "total_bytes": len(b"private response bytes"),
    }
    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_mtime_ns == source.stat().st_mtime_ns
    assert result["total_files"] == len(plan)


def test_single_file_mapping_uses_exact_destination_path(tmp_path):
    source = tmp_path / "source.json"
    destination = tmp_path / "workspace" / "reports" / "copied.json"
    source.write_text("manifest", encoding="utf-8")

    plan = build_plan([(source, destination)])

    assert len(plan) == 1
    assert plan[0].destination == destination.resolve()


def test_default_sources_include_app_temp_artifacts_but_not_test_runner_logs(
    tmp_path,
):
    project = tmp_path / "project"
    legacy = tmp_path / "old-runtime"
    temp_root = tmp_path / "temp"
    scratch = temp_root / "morningtalk-chunks"
    scratch.mkdir(parents=True)
    (scratch / "chunk.mp3").write_bytes(b"audio")
    (temp_root / "morningtalk-pilot-one.lock").write_bytes(b"\0")
    (temp_root / "morningtalk-full-pytest.out").write_text("test output")

    mappings = default_sources(project, legacy, temp_root)
    mapped_sources = {source for source, _destination in mappings}

    assert legacy in mapped_sources
    assert scratch in mapped_sources
    assert temp_root / "morningtalk-pilot-one.lock" in mapped_sources
    assert temp_root / "morningtalk-full-pytest.out" not in mapped_sources


def test_verified_copy_is_idempotent_for_matching_destination(tmp_path):
    source_root = tmp_path / "old"
    destination_root = tmp_path / "workspace"
    source = source_root / "cache.json"
    destination = destination_root / "cache.json"
    source.parent.mkdir(parents=True)
    source.write_text('{"status":"unknown"}', encoding="utf-8")
    plan = build_plan([(source_root, destination_root)])

    first = copy_verified(plan)
    second = copy_verified(plan)

    assert first["copied_files"] == 1
    assert second["copied_files"] == 0
    assert second["already_verified_files"] == 1
    assert destination.read_bytes() == source.read_bytes()


def test_conflicting_destination_is_not_overwritten_or_partially_copied(tmp_path):
    source_root = tmp_path / "old"
    destination_root = tmp_path / "workspace"
    source_root.mkdir()
    (source_root / "first.json").write_text("source", encoding="utf-8")
    (source_root / "conflict.json").write_text("source conflict", encoding="utf-8")
    destination_root.mkdir()
    conflict = destination_root / "conflict.json"
    conflict.write_text("keep destination", encoding="utf-8")
    plan = build_plan([(source_root, destination_root)])

    with pytest.raises(MigrationError, match="DestinationContentConflict"):
        copy_verified(plan)

    assert conflict.read_text(encoding="utf-8") == "keep destination"
    assert not (destination_root / "first.json").exists()
