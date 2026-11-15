"""Plan Azure transcription chunks from an immutable optimized VAD manifest."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from app.config import get_azure_config
from app.vad_prototype import DEFAULT_OUTPUT_DIR

DEFAULT_SOURCE_MANIFEST = (
    DEFAULT_OUTPUT_DIR / "meeting-20261107-0817.vad.optimized-review.json"
)
DEFAULT_PLAN_PATH = (
    DEFAULT_OUTPUT_DIR / "meeting-20261107-0817.azure-chunk-plan.json"
)
DEFAULT_CLIP_DIR = DEFAULT_OUTPUT_DIR / "azure_transcription_chunks"
TARGET_MIN_MS = 30_000
TARGET_MAX_MS = 45_000


class ChunkPlanError(ValueError):
    """Raised when an optimized VAD manifest cannot be safely chunked."""


def _integer(value: object, field: str) -> int:
    if type(value) is not int:
        raise ChunkPlanError(f"{field} must be an integer.")
    return value


def _validate_payload(payload: object) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(payload, dict):
        raise ChunkPlanError("Optimized VAD manifest must be a JSON object.")
    duration_ms = _integer(payload.get("audio_duration_ms"), "audio_duration_ms")
    segments = payload.get("segments")
    if duration_ms <= 0 or not isinstance(segments, list) or not segments:
        raise ChunkPlanError("Optimized VAD manifest has invalid duration or no segments.")
    if not isinstance(payload.get("recording_id"), str) or not payload["recording_id"]:
        raise ChunkPlanError("Optimized VAD manifest has no recording_id.")

    seen_ids: set[str] = set()
    seen_source_ids: set[str] = set()
    previous_start = -1
    previous_end = -1
    validated = []
    for segment in segments:
        if not isinstance(segment, dict):
            raise ChunkPlanError("Each optimized VAD segment must be an object.")
        segment_id = segment.get("segment_id")
        start_ms = _integer(segment.get("start_ms"), "start_ms")
        end_ms = _integer(segment.get("end_ms"), "end_ms")
        segment_duration = _integer(segment.get("duration_ms"), "duration_ms")
        core_start = _integer(segment.get("core_start_ms"), "core_start_ms")
        core_end = _integer(segment.get("core_end_ms"), "core_end_ms")
        source_ids = segment.get("source_segment_ids")
        lineage = segment.get("source_segments")
        if not isinstance(segment_id, str) or not segment_id or segment_id in seen_ids:
            raise ChunkPlanError("Optimized segment IDs must be nonempty and unique.")
        if (
            not 0 <= start_ms < end_ms <= duration_ms
            or segment_duration != end_ms - start_ms
        ):
            raise ChunkPlanError(f"Invalid timestamps for optimized segment {segment_id}.")
        if not start_ms <= core_start < core_end <= end_ms:
            raise ChunkPlanError(f"Invalid speech-core interval for {segment_id}.")
        if start_ms < previous_start or end_ms < previous_end:
            raise ChunkPlanError(
                "Optimized VAD segment boundaries must be chronological."
            )
        if (
            not isinstance(source_ids, list)
            or not source_ids
            or any(not isinstance(item, str) or not item for item in source_ids)
            or len(source_ids) != len(set(source_ids))
        ):
            raise ChunkPlanError(f"Invalid source_segment_ids for {segment_id}.")
        if not isinstance(lineage, list) or not lineage:
            raise ChunkPlanError(f"Missing source lineage for {segment_id}.")
        lineage_ids = [item.get("segment_id") for item in lineage if isinstance(item, dict)]
        if lineage_ids != source_ids:
            raise ChunkPlanError(f"Source lineage does not match IDs for {segment_id}.")
        duplicate_sources = seen_source_ids.intersection(source_ids)
        if duplicate_sources:
            raise ChunkPlanError(
                "Original VAD source IDs occur in multiple optimized segments: "
                + ", ".join(sorted(duplicate_sources))
            )
        for item in lineage:
            original_start = _integer(item.get("original_start_ms"), "original_start_ms")
            original_end = _integer(item.get("original_end_ms"), "original_end_ms")
            if not 0 <= original_start < original_end <= duration_ms:
                raise ChunkPlanError(f"Invalid original source interval in {segment_id}.")

        seen_ids.add(segment_id)
        seen_source_ids.update(source_ids)
        previous_start = start_ms
        previous_end = end_ms
        validated.append(segment)
    return payload, validated


def _chunk_groups(
    segments: list[dict[str, Any]],
    *,
    target_min_ms: int,
    target_max_ms: int,
) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    first = 0
    while first < len(segments):
        candidates: list[tuple[int, int, int, int]] = []
        for last in range(first, len(segments)):
            duration = segments[last]["end_ms"] - segments[first]["start_ms"]
            if last + 1 == len(segments):
                if target_min_ms <= duration <= target_max_ms:
                    candidates.append((last, duration, 0, 0))
                continue
            if duration > target_max_ms:
                continue
            following = segments[last + 1]
            audio_gap = following["start_ms"] - segments[last]["end_ms"]
            if audio_gap < 0:
                continue
            speech_gap = following["core_start_ms"] - segments[last]["core_end_ms"]
            if duration >= target_min_ms:
                candidates.append((last, duration, audio_gap, speech_gap))

        if candidates:
            # Only split where complete playback windows do not overlap.
            last, _, _, _ = max(
                candidates,
                key=lambda item: (
                    item[2],
                    item[3],
                    -abs(item[1] - (target_min_ms + target_max_ms) // 2),
                    -item[0],
                ),
            )
        else:
            safe_short = []
            safe_long = []
            for last in range(first, len(segments) - 1):
                duration = segments[last]["end_ms"] - segments[first]["start_ms"]
                audio_gap = segments[last + 1]["start_ms"] - segments[last]["end_ms"]
                if audio_gap < 0:
                    continue
                if duration < target_min_ms:
                    safe_short.append((last, duration, audio_gap))
                elif duration > target_max_ms:
                    safe_long.append((last, duration, audio_gap))
            if safe_short:
                # A short chunk is preferable to a boundary that repeats audio.
                last, _, _ = max(safe_short, key=lambda item: (item[1], item[2], -item[0]))
            elif safe_long:
                # Preserve whole VAD segments when a safe boundary only exists past the cap.
                last, _, _ = min(safe_long, key=lambda item: (item[1], -item[2], item[0]))
            else:
                last = first
                while last + 1 < len(segments):
                    if segments[last + 1]["start_ms"] >= segments[last]["end_ms"]:
                        break
                    last += 1
                # A single indivisible segment or an overlapping chain may exceed the cap.
        groups.append((first, last))
        first = last + 1
    return groups


def _interval_union_ms(intervals: list[tuple[int, int]]) -> int:
    total = 0
    current_start: int | None = None
    current_end: int | None = None
    for start, end in sorted(intervals):
        if current_end is None or start > current_end:
            if current_start is not None and current_end is not None:
                total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_start is not None and current_end is not None:
        total += current_end - current_start
    return total


def build_chunk_manifest_data(
    payload: object,
    *,
    deployment: str,
    api_version: str,
    target_min_ms: int = TARGET_MIN_MS,
    target_max_ms: int = TARGET_MAX_MS,
) -> dict[str, Any]:
    for field, value in (
        ("target_min_ms", target_min_ms),
        ("target_max_ms", target_max_ms),
    ):
        if type(value) is not int or value <= 0:
            raise ChunkPlanError(f"{field} must be a positive integer.")
    if target_min_ms > target_max_ms:
        raise ChunkPlanError("target_min_ms cannot exceed target_max_ms.")
    if not deployment or not api_version:
        raise ChunkPlanError("Azure deployment and API version metadata are required.")

    original, segments = _validate_payload(payload)
    groups = _chunk_groups(
        segments,
        target_min_ms=target_min_ms,
        target_max_ms=target_max_ms,
    )
    azure_metadata = {
        "provider": "azure_openai",
        "deployment": deployment,
        "api_version": api_version,
        "language": "th",
        "response_content": "text",
        "timestamp_granularity": "chunk_only",
    }

    chunks = []
    for index, (first, last) in enumerate(groups, 1):
        group = segments[first:last + 1]
        start_ms = group[0]["start_ms"]
        end_ms = group[-1]["end_ms"]
        optimized_ids = [segment["segment_id"] for segment in group]
        source_ids = [
            source_id
            for segment in group
            for source_id in segment["source_segment_ids"]
        ]
        lineage = []
        speech_intervals = []
        for segment in group:
            source_intervals = [
                {
                    "segment_id": source["segment_id"],
                    "start_ms": source["original_start_ms"],
                    "end_ms": source["original_end_ms"],
                }
                for source in segment["source_segments"]
            ]
            lineage.append({
                "optimized_segment_id": segment["segment_id"],
                "source_segment_ids": list(segment["source_segment_ids"]),
                "source_intervals": source_intervals,
            })
            speech_intervals.append({
                "segment_id": segment["segment_id"],
                "start_ms": segment["core_start_ms"],
                "end_ms": segment["core_end_ms"],
            })
        chunks.append({
            "chunk_id": f"chunk-{index:04d}",
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": end_ms - start_ms,
            "clip_filename": (
                f"{original['recording_id']}-chunk-{index:04d}.mp3"
            ),
            "optimized_segment_ids": optimized_ids,
            "source_segment_ids": source_ids,
            "segment_lineage": lineage,
            "speech_intervals": speech_intervals,
            "transcript_text": None,
            "transcription": {
                **azure_metadata,
                "status": "not_requested",
                "transcript_text": None,
            },
        })

    chunk_intervals = [(item["start_ms"], item["end_ms"]) for item in chunks]
    speech_intervals = [
        (segment["core_start_ms"], segment["core_end_ms"])
        for segment in segments
    ]
    covered_speech_intervals = [
        (interval["start_ms"], interval["end_ms"])
        for chunk in chunks
        for interval in chunk["speech_intervals"]
    ]
    source_ids = [
        source_id
        for chunk in chunks
        for source_id in chunk["source_segment_ids"]
    ]
    if len(source_ids) != len(set(source_ids)):
        raise ChunkPlanError("Chunk assignment duplicated an original VAD source ID.")
    if _interval_union_ms(speech_intervals) != _interval_union_ms(covered_speech_intervals):
        raise ChunkPlanError("Chunk assignment lost or duplicated detected speech intervals.")

    boundary_gaps = [
        chunks[index + 1]["start_ms"] - chunks[index]["end_ms"]
        for index in range(len(chunks) - 1)
    ]
    if any(gap < 0 for gap in boundary_gaps):
        raise ChunkPlanError("Chunk audio windows overlap; detected speech could be duplicated.")
    durations = [item["duration_ms"] for item in chunks]
    total_audio_coverage_ms = _interval_union_ms(chunk_intervals)
    speech_duration_ms = _interval_union_ms(speech_intervals)
    out_of_target = [
        {
            "chunk_id": item["chunk_id"],
            "duration_ms": item["duration_ms"],
            "reason": (
                "indivisible_vad_segment_exceeds_maximum"
                if item["duration_ms"] > target_max_ms
                and len(item["optimized_segment_ids"]) == 1
                else (
                    "overlapping_segment_chain_requires_whole_segments"
                    if item["duration_ms"] > target_max_ms
                    else "recording_boundary_or_maximum_constraint"
                )
            ),
        }
        for item in chunks
        if item["duration_ms"] < target_min_ms or item["duration_ms"] > target_max_ms
    ]
    return {
        "manifest_type": "azure_transcription_chunk_plan",
        "recording_id": original["recording_id"],
        "source_mp3": original.get("source_mp3"),
        "source_manifest": original.get("source_manifest"),
        "audio_duration_ms": original["audio_duration_ms"],
        "chunking": {
            "algorithm": "chronological-vad-grouping-with-silence-preference",
            "target_min_ms": target_min_ms,
            "target_max_ms": target_max_ms,
            "duration_basis": "first segment start_ms to last segment end_ms, including intervening audio",
            "speech_coverage_basis": "optimized VAD core_start_ms/core_end_ms intervals",
            "segments_split": False,
            "dry_run": True,
            "audio_clips_generated": False,
        },
        "transcription": {
            **azure_metadata,
            "status": "not_requested",
            "chunk_timestamps_are_word_or_sentence_timestamps": False,
        },
        "summary": {
            "source_segment_count": len(segments),
            "chunk_count": len(chunks),
            "minimum_duration_ms": min(durations, default=0),
            "average_duration_ms": round(sum(durations) / len(durations)) if durations else 0,
            "maximum_duration_ms": max(durations, default=0),
            "chunks_outside_target": out_of_target,
            "source_speech_core_ms": speech_duration_ms,
            "chunked_speech_core_ms": _interval_union_ms(covered_speech_intervals),
            "speech_coverage_percentage": (
                round(_interval_union_ms(covered_speech_intervals) / speech_duration_ms * 100, 4)
                if speech_duration_ms else 100.0
            ),
            "chunk_audio_coverage_ms": total_audio_coverage_ms,
            "recording_audio_coverage_percentage": round(
                total_audio_coverage_ms / original["audio_duration_ms"] * 100, 4
            ),
            "inter_chunk_gap_count": sum(gap > 0 for gap in boundary_gaps),
            "inter_chunk_gap_total_ms": sum(gap for gap in boundary_gaps if gap > 0),
            "inter_chunk_gap_min_ms": min((gap for gap in boundary_gaps if gap > 0), default=0),
            "inter_chunk_gap_average_ms": (
                round(sum(gap for gap in boundary_gaps if gap > 0) / sum(gap > 0 for gap in boundary_gaps))
                if any(gap > 0 for gap in boundary_gaps) else 0
            ),
            "inter_chunk_gap_max_ms": max((gap for gap in boundary_gaps if gap > 0), default=0),
            "inter_chunk_overlap_count": 0,
            "inter_chunk_overlap_total_ms": 0,
            "unassigned_source_segment_count": 0,
            "duplicated_source_segment_count": 0,
        },
        "chunks": chunks,
    }


def write_chunk_manifest(
    source_manifest: Path = DEFAULT_SOURCE_MANIFEST,
    *,
    output_path: Path = DEFAULT_PLAN_PATH,
) -> Path:
    source_manifest = source_manifest.resolve()
    output_path = output_path.resolve()
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    config = get_azure_config()
    result = build_chunk_manifest_data(
        payload,
        deployment=config.transcription_deployment,
        api_version=config.transcription_api_version,
    )
    result["source_manifest"] = str(source_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return output_path


def _workspace_path(path: Path) -> Path:
    resolved = path.resolve()
    workspace = DEFAULT_OUTPUT_DIR.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as error:
        raise ValueError("Generated audio clips must remain under workspace/vad_prototype.") from error
    return resolved


def export_audio_clips(
    plan_path: Path,
    *,
    output_dir: Path = DEFAULT_CLIP_DIR,
) -> list[Path]:
    output_dir = _workspace_path(output_dir)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    source = Path(plan["source_mp3"]).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source recording not found: {source}")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is not available on PATH.")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for chunk in plan["chunks"]:
        clip_filename = chunk.get("clip_filename")
        if (
            not isinstance(clip_filename, str)
            or Path(clip_filename).name != clip_filename
            or Path(clip_filename).suffix.casefold() != ".mp3"
        ):
            raise ValueError("Chunk clip filenames must be plain MP3 filenames.")
        output_path = _workspace_path(output_dir / clip_filename)
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite generated clip: {output_path}")
        subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-n",
                "-ss", f"{chunk['start_ms'] / 1000:.3f}",
                "-i", str(source), "-t", f"{chunk['duration_ms'] / 1000:.3f}",
                "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "libmp3lame", "-b:a", "64k", "-f", "mp3",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            timeout=chunk["duration_ms"] / 1000 + 120,
        )
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise ValueError(f"FFmpeg produced an empty clip: {output_path.name}")
        outputs.append(output_path)
    return outputs


def _print_summary(path: Path) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    summary = manifest["summary"]
    print(f"Chunk plan: {path}")
    for key in (
        "source_segment_count",
        "chunk_count",
        "minimum_duration_ms",
        "average_duration_ms",
        "maximum_duration_ms",
        "speech_coverage_percentage",
        "recording_audio_coverage_percentage",
        "inter_chunk_gap_count",
        "inter_chunk_gap_total_ms",
        "inter_chunk_overlap_count",
        "inter_chunk_overlap_total_ms",
    ):
        print(f"{key}: {summary[key]}")
    print("chunks_outside_target:")
    print(json.dumps(summary["chunks_outside_target"], ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan", help="Write a dry-run chunk manifest.")
    plan_parser.add_argument("--manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    plan_parser.add_argument("--output", type=Path, default=DEFAULT_PLAN_PATH)
    export_parser = commands.add_parser("export-audio", help="Export planned clips under workspace.")
    export_parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    export_parser.add_argument("--output-dir", type=Path, default=DEFAULT_CLIP_DIR)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            output_path = write_chunk_manifest(args.manifest, output_path=args.output)
            _print_summary(output_path)
        else:
            clips = export_audio_clips(args.plan, output_dir=args.output_dir)
            print(f"Exported {len(clips)} clips under {args.output_dir.resolve()}")
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
