import json
from pathlib import Path

import pytest

from app.vad_segment_optimizer import (
    OptimizationError,
    optimize_manifest_data,
    optimize_manifest,
)


def segment(segment_id, start, end, *, core_start=None, core_end=None):
    return {
        "segment_id": segment_id,
        "start_ms": start,
        "end_ms": end,
        "duration_ms": end - start,
        "core_start_ms": start if core_start is None else core_start,
        "core_end_ms": end if core_end is None else core_end,
        "alignment_status": "not_aligned",
        "transcript_text": None,
    }


def manifest(*segments, duration=30_000):
    return {
        "recording_id": "a" * 16,
        "source_mp3": "Record File/CB/Nov26/meeting.mp3",
        "audio_duration_ms": duration,
        "vad_engine": {"name": "WebRTC VAD"},
        "segments": list(segments),
    }


def test_short_speech_is_preserved_and_flagged_when_no_merge_is_safe():
    source = manifest(
        segment("seg-short", 1000, 1800),
        segment("seg-far", 6000, 9000),
    )

    result = optimize_manifest_data(source, max_merge_gap_ms=500)

    short = result["segments"][0]
    assert short["needs_review"] is True
    assert short["source_segment_ids"] == ["seg-short"]
    assert short["source_segments"][0]["original_start_ms"] == 1000
    assert short["source_segments"][0]["original_end_ms"] == 1800
    assert result["summary"]["unmerged_short_segments"][0]["segment_id"] == "seg-short"


def test_small_core_gap_merges_and_preserves_entire_source_intervals_and_gap():
    source = manifest(
        segment("seg-prev", 0, 1050, core_start=50, core_end=800),
        segment("seg-short", 950, 1800, core_start=1200, core_end=1550),
    )

    result = optimize_manifest_data(source, max_merge_gap_ms=400)

    merged = result["segments"][0]
    assert merged["start_ms"] == 0
    assert merged["end_ms"] == 1800
    assert merged["duration_ms"] == 1800
    assert merged["source_segment_ids"] == ["seg-prev", "seg-short"]
    assert merged["merged_gaps"] == [{
        "left_segment_id": "seg-prev",
        "right_segment_id": "seg-short",
        "speech_core_gap_ms": 400,
    }]
    assert result["summary"]["merged_group_count"] == 1


def test_short_segment_considers_both_neighbors_and_chooses_smaller_gap():
    source = manifest(
        segment("seg-prev", 0, 2000),
        segment("seg-short", 2100, 2600),
        segment("seg-next", 2650, 4650),
    )

    result = optimize_manifest_data(source, max_merge_gap_ms=200)

    assert result["segments"][0]["source_segment_ids"] == ["seg-prev"]
    assert result["segments"][1]["source_segment_ids"] == ["seg-short", "seg-next"]
    assert result["segments"][1]["merged_gaps"] == [{
        "left_segment_id": "seg-short",
        "right_segment_id": "seg-next",
        "speech_core_gap_ms": 50,
    }]


def test_gap_over_threshold_does_not_merge():
    source = manifest(
        segment("seg-prev", 0, 1000),
        segment("seg-short", 1501, 2201),
        segment("seg-next", 6000, 9000),
    )

    result = optimize_manifest_data(source, max_merge_gap_ms=500)

    assert result["summary"]["merged_group_count"] == 0
    assert result["segments"][1]["source_segment_ids"] == ["seg-short"]
    assert result["segments"][1]["needs_review"] is True


def test_merge_that_would_exceed_maximum_duration_is_rejected():
    source = manifest(
        segment("seg-long", 0, 14_500),
        segment("seg-short", 14_500, 15_100),
        segment("seg-far", 20_000, 22_000),
    )

    result = optimize_manifest_data(
        source,
        max_merge_gap_ms=1000,
        max_combined_duration_ms=15_000,
    )

    short = next(item for item in result["segments"] if "seg-short" in item["source_segment_ids"])
    assert short["source_segment_ids"] == ["seg-short"]
    assert short["needs_review"] is True
    assert max(item["duration_ms"] for item in result["segments"]) <= 15_000


def test_optimizer_is_deterministic_and_reports_before_after_distribution():
    source = manifest(
        segment("seg-a", 0, 500),
        segment("seg-b", 700, 1600),
        segment("seg-c", 3000, 5000),
    )

    first = optimize_manifest_data(source)
    second = optimize_manifest_data(source)

    assert first == second
    assert first["summary"]["before"] == {
        "total_segments": 3,
        "shorter_than_1_second": 2,
        "shorter_than_2_seconds": 2,
        "minimum_duration_ms": 500,
        "average_duration_ms": 1133,
        "maximum_duration_ms": 2000,
    }
    assert first["summary"]["after"]["total_segments"] == 2
    assert first["summary"]["merged_group_count"] == 1


def test_existing_adjustment_sidecar_is_untouched_and_not_remapped(tmp_path):
    manifest_path = tmp_path / "meeting.vad.json"
    payload = manifest(
        segment("seg-short", 0, 600),
        segment("seg-neighbor", 650, 2000),
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    sidecar = tmp_path / f"{'a' * 16}.adjustments.json"
    sidecar_content = '{"manually_adjusted": true}\n'
    sidecar.write_text(sidecar_content, encoding="utf-8")
    original_manifest = manifest_path.read_bytes()

    output_path = optimize_manifest(manifest_path, output_dir=tmp_path / "optimized")
    optimized = json.loads(output_path.read_text(encoding="utf-8"))

    assert manifest_path.read_bytes() == original_manifest
    assert sidecar.read_text(encoding="utf-8") == sidecar_content
    assert optimized["optimization"]["manual_adjustments"] == {
        "applied": False,
        "status": "not_remapped",
        "note": "Existing timing adjustments were not applied or remapped; review source IDs manually.",
        "sidecar": sidecar.name,
        "sidecar_present": True,
    }
    assert optimized["segments"][0]["source_segment_ids"] == [
        "seg-short",
        "seg-neighbor",
    ]


def test_optimized_manifest_never_overwrites_source_or_existing_output(tmp_path):
    manifest_path = tmp_path / "meeting.vad.json"
    payload = manifest(segment("seg-short", 0, 500))
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    source_bytes = manifest_path.read_bytes()
    output_dir = tmp_path / "optimized"
    output_dir.mkdir()
    output_path = output_dir / "meeting.vad.optimized-review.json"
    output_path.write_text("keep existing output", encoding="utf-8")

    with pytest.raises(FileExistsError):
        optimize_manifest(manifest_path, output_dir=output_dir)

    assert manifest_path.read_bytes() == source_bytes
    assert output_path.read_text(encoding="utf-8") == "keep existing output"


def test_invalid_or_duplicate_source_intervals_are_rejected():
    with pytest.raises(OptimizationError):
        optimize_manifest_data(manifest(segment("bad", 10, 10)))
    with pytest.raises(OptimizationError):
        optimize_manifest_data(manifest(
            segment("same", 0, 500),
            segment("same", 600, 900),
        ))
