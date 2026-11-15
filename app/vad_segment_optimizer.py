"""Derive compact review segments from an immutable VAD manifest."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

from app.vad_prototype import DEFAULT_OUTPUT_DIR

DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "meeting-20261107-0817.vad.json"
SHORT_SEGMENT_MS = 1_000
MAX_MERGE_GAP_MS = 1_000
MAX_COMBINED_DURATION_MS = 15_000


class OptimizationError(ValueError):
    """Raised when source VAD metadata cannot be safely optimized."""


def _integer(value: object, field: str) -> int:
    if type(value) is not int:
        raise OptimizationError(f"{field} must be an integer.")
    return value


@dataclass
class _ReviewGroup:
    start_ms: int
    end_ms: int
    speech_start_ms: int
    speech_end_ms: int
    source_segments: list[dict[str, Any]]
    merged_gaps: list[dict[str, Any]]

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


def _validated_sources(payload: object) -> tuple[dict[str, Any], list[_ReviewGroup]]:
    if not isinstance(payload, dict):
        raise OptimizationError("VAD manifest must be a JSON object.")
    duration_ms = _integer(payload.get("audio_duration_ms"), "audio_duration_ms")
    segments = payload.get("segments")
    recording_id = payload.get("recording_id")
    if duration_ms <= 0 or not isinstance(segments, list):
        raise OptimizationError("VAD manifest has invalid duration or segments.")
    if not isinstance(recording_id, str) or not recording_id:
        raise OptimizationError("VAD manifest has no recording_id.")

    groups = []
    seen_ids: set[str] = set()
    for segment in segments:
        if not isinstance(segment, dict):
            raise OptimizationError("Each VAD segment must be an object.")
        segment_id = segment.get("segment_id")
        start_ms = _integer(segment.get("start_ms"), "start_ms")
        end_ms = _integer(segment.get("end_ms"), "end_ms")
        duration = _integer(segment.get("duration_ms"), "duration_ms")
        if not isinstance(segment_id, str) or not segment_id or segment_id in seen_ids:
            raise OptimizationError("VAD segment IDs must be nonempty and unique.")
        seen_ids.add(segment_id)
        if not 0 <= start_ms < end_ms <= duration_ms or duration != end_ms - start_ms:
            raise OptimizationError(f"VAD segment {segment_id} has invalid timestamps.")

        core_start = segment.get("core_start_ms", start_ms)
        core_end = segment.get("core_end_ms", end_ms)
        core_start = _integer(core_start, "core_start_ms")
        core_end = _integer(core_end, "core_end_ms")
        if not 0 <= core_start < core_end <= duration_ms:
            raise OptimizationError(f"VAD core interval is invalid for {segment_id}.")
        source = {
            "segment_id": segment_id,
            "original_start_ms": start_ms,
            "original_end_ms": end_ms,
            "original_duration_ms": duration,
            "core_start_ms": core_start,
            "core_end_ms": core_end,
            "alignment_status": segment.get("alignment_status", "not_aligned"),
            "transcript_text": segment.get("transcript_text"),
            "speech_detection": segment.get("speech_detection"),
            "needs_review": bool(segment.get("needs_review", False)),
        }
        groups.append(_ReviewGroup(
            start_ms,
            end_ms,
            core_start,
            core_end,
            [source],
            [],
        ))
    groups.sort(key=lambda group: (
        group.start_ms,
        group.end_ms,
        group.source_segments[0]["segment_id"],
    ))
    return payload, groups


def _stats(segments: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [item["duration_ms"] for item in segments]
    return {
        "total_segments": len(durations),
        "shorter_than_1_second": sum(duration < 1_000 for duration in durations),
        "shorter_than_2_seconds": sum(duration < 2_000 for duration in durations),
        "minimum_duration_ms": min(durations, default=0),
        "average_duration_ms": (
            round(sum(durations) / len(durations)) if durations else 0
        ),
        "maximum_duration_ms": max(durations, default=0),
    }


def optimize_manifest_data(
    payload: object,
    *,
    max_merge_gap_ms: int = MAX_MERGE_GAP_MS,
    max_combined_duration_ms: int = MAX_COMBINED_DURATION_MS,
    short_segment_ms: int = SHORT_SEGMENT_MS,
) -> dict[str, Any]:
    for name, value in (
        ("max_merge_gap_ms", max_merge_gap_ms),
        ("max_combined_duration_ms", max_combined_duration_ms),
        ("short_segment_ms", short_segment_ms),
    ):
        if type(value) is not int or value < 0:
            raise OptimizationError(f"{name} must be a nonnegative integer.")
    if max_combined_duration_ms == 0 or short_segment_ms == 0:
        raise OptimizationError("Combined and short segment limits must be positive.")

    original, groups = _validated_sources(payload)
    before_segments = [
        {
            "duration_ms": source["original_duration_ms"],
        }
        for group in groups
        for source in group.source_segments
    ]

    while True:
        selected: tuple[int, int, int] | None = None
        for index, group in enumerate(groups):
            if group.duration_ms >= short_segment_ms:
                continue
            candidates = []
            if index > 0:
                previous = groups[index - 1]
                gap = max(0, group.speech_start_ms - previous.speech_end_ms)
                combined_duration = group.end_ms - previous.start_ms
                if (
                    gap <= max_merge_gap_ms
                    and combined_duration <= max_combined_duration_ms
                ):
                    candidates.append((gap, index - 1))
            if index + 1 < len(groups):
                following = groups[index + 1]
                gap = max(0, following.speech_start_ms - group.speech_end_ms)
                combined_duration = following.end_ms - group.start_ms
                if (
                    gap <= max_merge_gap_ms
                    and combined_duration <= max_combined_duration_ms
                ):
                    candidates.append((gap, index + 1))
            if candidates:
                gap, neighbor_index = min(candidates, key=lambda item: (item[0], item[1]))
                selected = (index, neighbor_index, gap)
                break
        if selected is None:
            break

        short_index, neighbor_index, selected_gap = selected
        left_index, right_index = sorted((short_index, neighbor_index))
        left, right = groups[left_index], groups[right_index]
        merged = _ReviewGroup(
            start_ms=min(left.start_ms, right.start_ms),
            end_ms=max(left.end_ms, right.end_ms),
            speech_start_ms=min(left.speech_start_ms, right.speech_start_ms),
            speech_end_ms=max(left.speech_end_ms, right.speech_end_ms),
            source_segments=left.source_segments + right.source_segments,
            merged_gaps=left.merged_gaps + right.merged_gaps + [{
                "left_segment_id": left.source_segments[-1]["segment_id"],
                "right_segment_id": right.source_segments[0]["segment_id"],
                "speech_core_gap_ms": selected_gap,
            }],
        )
        groups[left_index:right_index + 1] = [merged]

    segments: list[dict[str, Any]] = []
    for index, group in enumerate(groups, 1):
        source_ids = [item["segment_id"] for item in group.source_segments]
        merged = len(source_ids) > 1
        needs_review = (
            group.duration_ms < short_segment_ms
            or any(item["needs_review"] for item in group.source_segments)
        )
        segments.append({
            "segment_id": (
                f"review-{index:04d}" if merged else source_ids[0]
            ),
            "start_ms": group.start_ms,
            "end_ms": group.end_ms,
            "duration_ms": group.duration_ms,
            "core_start_ms": group.speech_start_ms,
            "core_end_ms": group.speech_end_ms,
            "source_segment_ids": source_ids,
            "source_segments": group.source_segments,
            "merged_gaps": group.merged_gaps,
            "merge_status": "merged" if merged else "unchanged",
            "needs_review": needs_review,
            "alignment_status": "not_aligned",
            "transcript_text": None,
        })

    unmerged_short = [
        {
            "segment_id": item["segment_id"],
            "source_segment_ids": item["source_segment_ids"],
            "duration_ms": item["duration_ms"],
        }
        for item in segments
        if item["duration_ms"] < short_segment_ms
    ]
    merged_groups = sum(item["merge_status"] == "merged" for item in segments)
    return {
        "manifest_type": "optimized_review_segments",
        "recording_id": original["recording_id"],
        "source_mp3": original.get("source_mp3"),
        "source_manifest": original.get("source_manifest"),
        "audio_duration_ms": original["audio_duration_ms"],
        "optimization": {
            "algorithm": "greedy-short-segment-adjacent-merge",
            "short_segment_threshold_ms": short_segment_ms,
            "maximum_adjacent_gap_ms": max_merge_gap_ms,
            "maximum_combined_duration_ms": max_combined_duration_ms,
            "gap_basis": "speech core intervals when present; otherwise original segment intervals",
            "manual_adjustments": {
                "applied": False,
                "status": "not_remapped",
                "note": "Existing timing adjustments were not applied or remapped; review source IDs manually.",
            },
        },
        "summary": {
            "before": _stats(before_segments),
            "after": _stats(segments),
            "merged_group_count": merged_groups,
            "unmerged_short_segments": unmerged_short,
        },
        "segments": segments,
    }


def optimize_manifest(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_merge_gap_ms: int = MAX_MERGE_GAP_MS,
    max_combined_duration_ms: int = MAX_COMBINED_DURATION_MS,
    short_segment_ms: int = SHORT_SEGMENT_MS,
) -> Path:
    manifest_path = manifest_path.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = optimize_manifest_data(
        payload,
        max_merge_gap_ms=max_merge_gap_ms,
        max_combined_duration_ms=max_combined_duration_ms,
        short_segment_ms=short_segment_ms,
    )
    adjustment_path = manifest_path.with_name(
        f"{result['recording_id']}.adjustments.json"
    )
    result["optimization"]["manual_adjustments"].update({
        "sidecar": adjustment_path.name if adjustment_path.is_file() else None,
        "sidecar_present": adjustment_path.is_file(),
    })
    result["source_manifest"] = str(manifest_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{manifest_path.stem}.optimized-review.json"
    with output_path.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--max-merge-gap-ms", type=int, default=MAX_MERGE_GAP_MS)
    args = parser.parse_args(argv)
    try:
        output_path = optimize_manifest(
            args.manifest,
            max_merge_gap_ms=args.max_merge_gap_ms,
        )
        result = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(f"Optimized manifest: {output_path}")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
