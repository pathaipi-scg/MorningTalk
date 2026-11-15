"""Opt-in local VAD analysis and timestamped playback for review prototypes."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2
DEFAULT_RECORDING = Path(
    r"D:\KM\Vault\MorningTalk\Record File\CB\Nov26\meeting-20261107-0817.mp3"
)
DEFAULT_OUTPUT_DIR = Path(r"D:\AI\MorningTalk\workspace\vad_prototype")


@dataclass(frozen=True)
class VadConfig:
    frame_ms: int = 30
    aggressiveness: int = 2
    merge_silence_ms: int = 450
    minimum_speech_ms: int = 60
    target_segment_ms: int = 10_000
    maximum_segment_ms: int = 15_000
    padding_ms: int = 250

    def validate(self) -> None:
        if self.frame_ms not in {10, 20, 30}:
            raise ValueError("WebRTC VAD frame size must be 10, 20, or 30 ms.")
        if not 0 <= self.aggressiveness <= 3:
            raise ValueError("WebRTC VAD aggressiveness must be from 0 to 3.")
        if self.merge_silence_ms < 0 or self.minimum_speech_ms < 0 or self.padding_ms < 0:
            raise ValueError("Silence, minimum speech, and padding durations cannot be negative.")
        if self.target_segment_ms <= 0 or self.maximum_segment_ms <= 0:
            raise ValueError("Segment durations must be positive.")
        if self.target_segment_ms < self.frame_ms or self.maximum_segment_ms < self.frame_ms:
            raise ValueError("Target and maximum segment durations must be at least one VAD frame.")
        if self.target_segment_ms > self.maximum_segment_ms:
            raise ValueError("Target segment duration cannot exceed the maximum.")


def _require_tool(name: str) -> str:
    executable = shutil.which(name)
    if not executable:
        raise RuntimeError(f"{name} is not available on PATH.")
    return executable


def _audio_duration(audio_path: Path) -> float:
    result = subprocess.run(
        [
            _require_tool("ffprobe"), "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    duration = float(result.stdout.strip())
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Audio has an invalid duration: {audio_path.name}")
    return duration


def _decode_pcm(audio_path: Path) -> bytes:
    result = subprocess.run(
        [
            _require_tool("ffmpeg"), "-hide_banner", "-loglevel", "error",
            "-i", str(audio_path), "-map", "0:a:0", "-vn", "-ac", "1",
            "-ar", str(SAMPLE_RATE), "-f", "s16le", "pipe:1",
        ],
        check=True,
        capture_output=True,
        timeout=3600,
    )
    if not result.stdout:
        raise ValueError(f"FFmpeg decoded no audio samples: {audio_path.name}")
    return result.stdout


def _load_webrtc_vad(aggressiveness: int) -> Any:
    try:
        import webrtcvad
    except ImportError as error:
        raise RuntimeError(
            "WebRTC VAD is unavailable. Install the approved dependency "
            "'webrtcvad-wheels' before running this prototype."
        ) from error
    return webrtcvad.Vad(aggressiveness)


def _speech_frame_runs(
    pcm: bytes,
    config: VadConfig,
    detector: Any,
) -> tuple[list[tuple[int, int]], list[bool]]:
    frame_bytes = SAMPLE_RATE * config.frame_ms // 1000 * BYTES_PER_SAMPLE
    frame_count = len(pcm) // frame_bytes
    speech_frames: list[bool] = []
    for frame_index in range(frame_count):
        begin = frame_index * frame_bytes
        speech_frames.append(detector.is_speech(pcm[begin:begin + frame_bytes], SAMPLE_RATE))

    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for index, is_speech in enumerate(speech_frames):
        if is_speech and run_start is None:
            run_start = index
        elif not is_speech and run_start is not None:
            runs.append((run_start * config.frame_ms, index * config.frame_ms))
            run_start = None
    if run_start is not None:
        runs.append((run_start * config.frame_ms, frame_count * config.frame_ms))

    minimum_frames = math.ceil(config.minimum_speech_ms / config.frame_ms)
    runs = [
        run for run in runs
        if math.ceil((run[1] - run[0]) / config.frame_ms) >= minimum_frames
    ]
    return runs, speech_frames


def _merge_runs(
    runs: list[tuple[int, int]],
    merge_silence_ms: int,
) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start_ms, end_ms in runs:
        if merged and start_ms - merged[-1][1] <= merge_silence_ms:
            merged[-1] = (merged[-1][0], end_ms)
        else:
            merged.append((start_ms, end_ms))
    return merged


def _split_region(
    start_ms: int,
    end_ms: int,
    raw_runs: list[tuple[int, int]],
    config: VadConfig,
) -> list[tuple[int, int, str]]:
    segments: list[tuple[int, int, str]] = []
    cursor = start_ms
    while end_ms - cursor > config.maximum_segment_ms:
        target = cursor + config.target_segment_ms
        search_start = max(cursor + config.target_segment_ms // 2, target - 2_000)
        search_end = min(cursor + config.maximum_segment_ms, target + 2_000)
        candidates = [
            (left, right)
            for left, right in zip(raw_runs, raw_runs[1:])
            if left[1] >= search_start and right[0] <= search_end
            and right[0] - left[1] > 0
        ]
        if candidates:
            left, right = max(candidates, key=lambda gap: gap[1][0] - gap[0][1])
            cut = (left[1] + right[0]) // 2
            reason = "vad_pause"
        else:
            cut = target
            reason = "maximum_duration"
        cut = round(cut / config.frame_ms) * config.frame_ms
        cut = min(max(cut, cursor + config.frame_ms), end_ms - config.frame_ms)
        segments.append((cursor, cut, reason))
        cursor = cut
    if cursor < end_ms:
        segments.append((cursor, end_ms, "vad_region_end"))
    return segments


def build_manifest(
    audio_path: Path,
    pcm: bytes,
    duration_seconds: float,
    *,
    detector: Any,
    config: VadConfig = VadConfig(),
) -> dict[str, Any]:
    config.validate()
    raw_runs, speech_frames = _speech_frame_runs(pcm, config, detector)
    regions = _merge_runs(raw_runs, config.merge_silence_ms)
    duration_ms = round(duration_seconds * 1000)
    manifest_segments: list[dict[str, Any]] = []

    for region_index, (region_start, region_end) in enumerate(regions, 1):
        for core_start, core_end, split_reason in _split_region(
            region_start, region_end, raw_runs, config
        ):
            start_ms = max(0, core_start - config.padding_ms)
            end_ms = min(duration_ms, core_end + config.padding_ms)
            covered_frames = [
                index for index, active in enumerate(speech_frames)
                if active
                and index * config.frame_ms < core_end
                and (index + 1) * config.frame_ms > core_start
            ]
            manifest_segments.append({
                "segment_id": f"seg-{len(manifest_segments) + 1:04d}",
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": end_ms - start_ms,
                "core_start_ms": core_start,
                "core_end_ms": core_end,
                "speech_detection": {
                    "region_id": f"region-{region_index:04d}",
                    "split_reason": split_reason,
                    "active_frame_count": len(covered_frames),
                    "frame_count": max(
                        1,
                        math.ceil((core_end - core_start) / config.frame_ms),
                    ),
                    "active_frame_ratio": round(
                        len(covered_frames)
                        / max(1, math.ceil((core_end - core_start) / config.frame_ms)),
                        4,
                    ),
                },
                "alignment_status": "not_aligned",
                "transcript_text": None,
            })

    # Use frame-level activity for the speech ratio; merged pauses are not speech.
    core_speech_ms = sum(speech_frames) * config.frame_ms
    inter_region_gaps = [
        {
            "start_ms": previous_end,
            "end_ms": next_start,
            "duration_ms": next_start - previous_end,
        }
        for (_, previous_end), (next_start, _) in zip(regions, regions[1:])
    ]
    hard_split_count = sum(
        item["speech_detection"]["split_reason"] == "maximum_duration"
        for item in manifest_segments
    )
    warnings = [
        "VAD regions indicate speech activity, not sentences or speaker identity."
    ]
    if hard_split_count:
        warnings.append(
            "Some long speech regions required duration-based splits; review their boundaries."
        )
    if regions and (regions[0][0] == 0 or regions[-1][1] >= duration_ms):
        warnings.append(
            "Speech activity touches a recording boundary; listen for possible clipped words."
        )
    return {
        "recording_id": hashlib.sha256(
            str(audio_path.resolve()).encode("utf-8")
        ).hexdigest()[:16],
        "source_mp3": str(audio_path.resolve()),
        "audio_duration_ms": duration_ms,
        "vad_engine": {
            "name": "WebRTC VAD",
            "package": "webrtcvad-wheels",
            "sample_rate_hz": SAMPLE_RATE,
            "configuration": asdict(config),
        },
        "summary": {
            "speech_region_count": len(regions),
            "review_segment_count": len(manifest_segments),
            "average_segment_duration_ms": (
                round(sum(item["duration_ms"] for item in manifest_segments) / len(manifest_segments))
                if manifest_segments else 0
            ),
            "minimum_segment_duration_ms": min(
                (item["duration_ms"] for item in manifest_segments), default=0
            ),
            "maximum_segment_duration_ms": max(
                (item["duration_ms"] for item in manifest_segments), default=0
            ),
            "speech_percentage": round(
                min(100.0, core_speech_ms / max(duration_ms, 1) * 100), 2
            ),
            "segments_over_15_seconds": sum(
                item["duration_ms"] > 15_000 for item in manifest_segments
            ),
            "duration_based_split_count": hard_split_count,
            "inter_region_silence_gaps": inter_region_gaps,
            "trailing_partial_frame_ignored": len(pcm) % (
                SAMPLE_RATE * config.frame_ms // 1000 * BYTES_PER_SAMPLE
            ) != 0,
            "warnings": warnings,
        },
        "segments": manifest_segments,
    }


def analyze_recording(
    audio_path: Path,
    *,
    config: VadConfig = VadConfig(),
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    audio_path = audio_path.resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Recording not found: {audio_path}")
    config.validate()
    duration = _audio_duration(audio_path)
    pcm = _decode_pcm(audio_path)
    detector = _load_webrtc_vad(config.aggressiveness)
    manifest = build_manifest(audio_path, pcm, duration, detector=detector, config=config)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{audio_path.stem}.vad.json"
    with output_path.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return output_path


def preview_segment(
    manifest_path: Path,
    segment_id: str,
    *,
    context_before_seconds: float = 0,
    context_after_seconds: float = 0,
    playback_rate: float = 1,
) -> None:
    if context_before_seconds < 0 or context_after_seconds < 0:
        raise ValueError("Playback context cannot be negative.")
    if not 0.5 <= playback_rate <= 1.5:
        raise ValueError("Playback rate must be between 0.5 and 1.5.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    segment = next(
        (item for item in manifest["segments"] if item["segment_id"] == segment_id),
        None,
    )
    if segment is None:
        raise ValueError(f"Unknown segment ID: {segment_id}")
    duration = manifest["audio_duration_ms"] / 1000
    start = max(0, segment["start_ms"] / 1000 - context_before_seconds)
    end = min(duration, segment["end_ms"] / 1000 + context_after_seconds)
    command = [
        _require_tool("ffplay"), "-hide_banner", "-loglevel", "error",
        "-nodisp", "-autoexit", "-ss", f"{start:.3f}",
        "-i", manifest["source_mp3"], "-t", f"{end - start:.3f}",
    ]
    if playback_rate != 1:
        command.extend(["-af", f"atempo={playback_rate:g}"])
    subprocess.run(command, check=True)


def _print_summary(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = manifest["summary"]
    print(f"Manifest: {manifest_path}")
    for label, key in (
        ("Speech regions", "speech_region_count"),
        ("Review segments", "review_segment_count"),
        ("Average duration (ms)", "average_segment_duration_ms"),
        ("Minimum duration (ms)", "minimum_segment_duration_ms"),
        ("Maximum duration (ms)", "maximum_segment_duration_ms"),
        ("Speech percentage", "speech_percentage"),
        ("Segments over 15 seconds", "segments_over_15_seconds"),
    ):
        print(f"{label}: {summary[key]}")
    for warning in summary["warnings"]:
        print(f"Warning: {warning}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze", help="Create an offline VAD manifest.")
    analyze.add_argument("recording", nargs="?", type=Path, default=DEFAULT_RECORDING)
    preview = commands.add_parser("preview", help="Play one manifest segment.")
    preview.add_argument("manifest", type=Path)
    preview.add_argument("segment_id")
    preview.add_argument("--context-before", type=float, default=0)
    preview.add_argument("--context-after", type=float, default=0)
    preview.add_argument("--rate", type=float, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "analyze":
        try:
            manifest_path = analyze_recording(args.recording)
        except (FileNotFoundError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            print(str(error), file=sys.stderr)
            return 1
        _print_summary(manifest_path)
    else:
        try:
            preview_segment(
                args.manifest,
                args.segment_id,
                context_before_seconds=args.context_before,
                context_after_seconds=args.context_after,
                playback_rate=args.rate,
            )
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            print(str(error), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
