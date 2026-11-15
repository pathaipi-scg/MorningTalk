import json
from pathlib import Path
import subprocess

import pytest

from app import vad_transcription_chunks as chunks


def optimized_segment(index, start_ms, end_ms, *, source_id=None):
    source_id = source_id or f"seg-{index:04d}"
    return {
        "segment_id": f"review-{index:04d}",
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_ms": end_ms - start_ms,
        "core_start_ms": start_ms + 100,
        "core_end_ms": end_ms - 100,
        "source_segment_ids": [source_id],
        "source_segments": [{
            "segment_id": source_id,
            "original_start_ms": start_ms,
            "original_end_ms": end_ms,
        }],
    }


def manifest_with(segments, duration_ms=None):
    return {
        "recording_id": "recording-1",
        "source_mp3": "meeting.mp3",
        "source_manifest": "optimized-review.json",
        "audio_duration_ms": duration_ms or segments[-1]["end_ms"],
        "segments": segments,
    }


def test_chunks_are_chronological_unsplit_and_prefer_natural_silence():
    segments = [
        optimized_segment(index, (index - 1) * 5_100, (index - 1) * 5_100 + 5_000)
        for index in range(1, 13)
    ]
    # A wider pause at a valid target boundary should win over narrower pauses.
    segments[6]["start_ms"] += 1_000
    segments[6]["end_ms"] += 1_000
    segments[6]["core_start_ms"] += 1_000
    segments[6]["core_end_ms"] += 1_000
    segments[6]["source_segments"][0]["original_start_ms"] += 1_000
    segments[6]["source_segments"][0]["original_end_ms"] += 1_000

    result = chunks.build_chunk_manifest_data(
        manifest_with(segments, duration_ms=70_000),
        deployment="gpt-4o-transcribe",
        api_version="2025-03-01-preview",
    )
    repeated = chunks.build_chunk_manifest_data(
        manifest_with(segments, duration_ms=70_000),
        deployment="gpt-4o-transcribe",
        api_version="2025-03-01-preview",
    )
    assigned = [
        segment_id
        for chunk in result["chunks"]
        for segment_id in chunk["optimized_segment_ids"]
    ]

    assert assigned == [item["segment_id"] for item in segments]
    assert result["summary"]["duplicated_source_segment_count"] == 0
    assert result["summary"]["unassigned_source_segment_count"] == 0
    assert result["summary"]["speech_coverage_percentage"] == 100
    assert result == repeated
    assert any(chunk["duration_ms"] >= 30_000 for chunk in result["chunks"])
    assert all(
        chunk["duration_ms"] <= 45_000
        for chunk in result["chunks"]
        if len(chunk["optimized_segment_ids"]) > 1
    )
    assert all(
        left["end_ms"] <= right["start_ms"]
        for left, right in zip(result["chunks"], result["chunks"][1:])
    )
    assert result["chunks"][0]["end_ms"] < segments[-1]["start_ms"]


def test_plan_preserves_original_lineage_and_separates_transcripts():
    segment = optimized_segment(1, 0, 40_000, source_id="seg-0001")
    segment["segment_id"] = "review-0001"
    segment["source_segment_ids"] = ["seg-0001", "seg-0002"]
    segment["source_segments"].append({
        "segment_id": "seg-0002",
        "original_start_ms": 10_000,
        "original_end_ms": 40_000,
    })
    result = chunks.build_chunk_manifest_data(
        manifest_with([segment]),
        deployment="gpt-4o-transcribe",
        api_version="2025-03-01-preview",
    )
    chunk = result["chunks"][0]

    assert chunk["source_segment_ids"] == ["seg-0001", "seg-0002"]
    assert [item["segment_id"] for item in chunk["segment_lineage"][0]["source_intervals"]] == [
        "seg-0001", "seg-0002"
    ]
    assert chunk["transcript_text"] is None
    assert chunk["transcription"]["status"] == "not_requested"
    assert chunk["transcription"]["timestamp_granularity"] == "chunk_only"
    assert result["chunking"]["segments_split"] is False


def test_short_final_chunk_and_indivisible_oversized_segment_are_reported():
    segments = [
        optimized_segment(1, 0, 50_000),
        optimized_segment(2, 55_000, 65_000),
    ]
    result = chunks.build_chunk_manifest_data(
        manifest_with(segments),
        deployment="deployment",
        api_version="v1",
    )

    assert [chunk["duration_ms"] for chunk in result["chunks"]] == [50_000, 10_000]
    assert [item["reason"] for item in result["summary"]["chunks_outside_target"]] == [
        "indivisible_vad_segment_exceeds_maximum",
        "recording_boundary_or_maximum_constraint",
    ]


def test_rejects_bad_order_timestamps_and_duplicate_source_lineage():
    valid = optimized_segment(1, 0, 10_000)
    later = optimized_segment(2, 9_000, 15_000, source_id="seg-0001")
    with pytest.raises(chunks.ChunkPlanError, match="chronological"):
        chunks.build_chunk_manifest_data(
            manifest_with([later, valid], duration_ms=20_000),
            deployment="d",
            api_version="v1",
        )
    with pytest.raises(chunks.ChunkPlanError, match="multiple optimized segments"):
        chunks.build_chunk_manifest_data(
            manifest_with([valid, later], duration_ms=20_000),
            deployment="d",
            api_version="v1",
        )


def test_plan_writer_refuses_to_overwrite(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    source.write_text(json.dumps(manifest_with([optimized_segment(1, 0, 10_000)])))
    output = tmp_path / "plan.json"
    output.write_text("preserve")

    monkeypatch.setattr(chunks, "get_azure_config", lambda: type(
        "Config", (), {
            "transcription_deployment": "deployment",
            "transcription_api_version": "v1",
        },
    )())
    with pytest.raises(FileExistsError):
        chunks.write_chunk_manifest(source, output_path=output)
    assert output.read_text() == "preserve"


def test_exporter_rejects_paths_outside_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(chunks, "DEFAULT_OUTPUT_DIR", workspace)

    with pytest.raises(ValueError, match="under workspace"):
        chunks.export_audio_clips(tmp_path / "plan.json", output_dir=tmp_path / "outside")


def test_exporter_rejects_clip_filename_traversal(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "source.mp3"
    source.write_bytes(b"source")
    plan_path = workspace / "plan.json"
    plan_path.write_text(json.dumps({
        "source_mp3": str(source),
        "chunks": [{
            "clip_filename": "..\\outside.mp3",
            "start_ms": 0,
            "duration_ms": 1_000,
        }],
    }))
    monkeypatch.setattr(chunks, "DEFAULT_OUTPUT_DIR", workspace)
    monkeypatch.setattr(chunks.shutil, "which", lambda name: "ffmpeg.exe")

    with pytest.raises(ValueError, match="plain MP3 filenames"):
        chunks.export_audio_clips(plan_path, output_dir=workspace / "clips")


def test_exporter_uses_only_workspace_clips_and_ffmpeg(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "source.mp3"
    source.write_bytes(b"source")
    plan_path = workspace / "plan.json"
    plan_path.write_text(json.dumps({
        "source_mp3": str(source),
        "chunks": [{
            "clip_filename": "recording-1-chunk-0001.mp3",
            "start_ms": 1_250,
            "duration_ms": 30_000,
        }],
    }))
    monkeypatch.setattr(chunks, "DEFAULT_OUTPUT_DIR", workspace)
    monkeypatch.setattr(chunks.shutil, "which", lambda name: "ffmpeg.exe" if name == "ffmpeg" else None)

    def fake_run(command, **kwargs):
        assert command[0] == "ffmpeg.exe"
        assert command[command.index("-ss") + 1] == "1.250"
        Path(command[-1]).write_bytes(b"clip")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(chunks.subprocess, "run", fake_run)
    outputs = chunks.export_audio_clips(plan_path, output_dir=workspace / "clips")

    assert outputs == [workspace / "clips" / "recording-1-chunk-0001.mp3"]
    assert outputs[0].read_bytes() == b"clip"
