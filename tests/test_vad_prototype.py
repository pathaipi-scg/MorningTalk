import json
from pathlib import Path
import struct

import pytest

from app import vad_prototype


class PatternDetector:
    def __init__(self, active_frames):
        self.active_frames = set(active_frames)
        self.calls = 0

    def is_speech(self, frame, sample_rate):
        assert len(frame) == 960
        assert sample_rate == 16_000
        active = self.calls in self.active_frames
        self.calls += 1
        return active


def pcm_for_frames(count):
    return struct.pack("<h", 0) * (count * 480)


def manifest_for(active_frames, frame_count, **config_values):
    return vad_prototype.build_manifest(
        Path("fixture.mp3"),
        pcm_for_frames(frame_count),
        frame_count * 0.03,
        detector=PatternDetector(active_frames),
        config=vad_prototype.VadConfig(**config_values),
    )


def test_silence_produces_no_regions_or_segments():
    manifest = manifest_for([], 100)

    assert manifest["summary"]["speech_region_count"] == 0
    assert manifest["segments"] == []
    assert manifest["summary"]["speech_percentage"] == 0


def test_short_speech_and_recording_boundary_are_kept():
    manifest = manifest_for([0, 1, 2, 98, 99], 100)

    assert len(manifest["segments"]) == 2
    assert manifest["segments"][0]["start_ms"] == 0
    assert manifest["segments"][-1]["end_ms"] == 3000
    assert manifest["segments"][0]["alignment_status"] == "not_aligned"
    assert manifest["segments"][0]["transcript_text"] is None


def test_long_continuous_speech_splits_into_bounded_contiguous_core_ranges():
    manifest = manifest_for(
        range(1000),
        1000,
        target_segment_ms=10_000,
        maximum_segment_ms=12_000,
        padding_ms=200,
    )
    segments = manifest["segments"]

    assert len(segments) == 3
    assert all(item["duration_ms"] <= 12_400 for item in segments)
    assert segments[0]["core_start_ms"] == 0
    assert segments[-1]["core_end_ms"] == 30_000
    assert all(
        left["core_end_ms"] == right["core_start_ms"]
        for left, right in zip(segments, segments[1:])
    )
    assert sum(
        item["speech_detection"]["active_frame_count"] for item in segments
    ) == 1000
    assert manifest["summary"]["duration_based_split_count"] == 2
    assert any("duration-based splits" in warning for warning in manifest["summary"]["warnings"])


def test_vad_pauses_are_merged_but_long_silence_separates_regions():
    manifest = manifest_for(
        [*range(100), *range(115, 215), *range(300, 400)],
        400,
        merge_silence_ms=450,
    )

    assert manifest["summary"]["speech_region_count"] == 2
    assert manifest["summary"]["inter_region_silence_gaps"] == [
        {"start_ms": 6450, "end_ms": 9000, "duration_ms": 2550}
    ]
    assert manifest["segments"][0]["speech_detection"]["region_id"] == "region-0001"
    assert manifest["segments"][1]["speech_detection"]["region_id"] == "region-0002"


def test_boundary_validation_and_deterministic_manifest_shape():
    first = manifest_for([*range(10, 40), *range(60, 90)], 100)
    second = manifest_for([*range(10, 40), *range(60, 90)], 100)
    required_top_level = {
        "recording_id", "source_mp3", "audio_duration_ms",
        "vad_engine", "summary", "segments",
    }

    assert set(first) == required_top_level
    assert first == second
    for segment in first["segments"]:
        assert segment["start_ms"] >= 0
        assert segment["end_ms"] > segment["start_ms"]
        assert segment["duration_ms"] == segment["end_ms"] - segment["start_ms"]
        assert segment["alignment_status"] == "not_aligned"
        assert segment["transcript_text"] is None
    assert json.loads(json.dumps(first)) == first


def test_manifest_writer_does_not_overwrite_existing_file(tmp_path, monkeypatch):
    source = tmp_path / "meeting.mp3"
    source.write_bytes(b"source")
    manifest_path = tmp_path / "out" / "meeting.vad.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text("keep me", encoding="utf-8")

    def fake_duration(path):
        assert path == source.resolve()
        return 3.0

    def fake_decode(path):
        assert path == source.resolve()
        return pcm_for_frames(100)

    def fake_detector(aggressiveness):
        assert aggressiveness == 2
        return PatternDetector(range(10))

    monkeypatch.setattr(vad_prototype, "_audio_duration", fake_duration)
    monkeypatch.setattr(vad_prototype, "_decode_pcm", fake_decode)
    monkeypatch.setattr(vad_prototype, "_load_webrtc_vad", fake_detector)

    with pytest.raises(FileExistsError):
        vad_prototype.analyze_recording(source, output_dir=manifest_path.parent)
    assert manifest_path.read_text(encoding="utf-8") == "keep me"
