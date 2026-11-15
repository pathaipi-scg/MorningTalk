"""Loopback-only browser review for local VAD manifests."""

from datetime import datetime, timezone
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import tempfile
from threading import RLock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from app.vad_prototype import DEFAULT_OUTPUT_DIR

DEFAULT_RECORDING_ROOT = Path(r"D:\KM\Vault\MorningTalk\Record File")
MANIFEST_DIR = DEFAULT_OUTPUT_DIR
HTML_PATH = Path(__file__).with_name("vad_review.html")
RECORDING_ID_PATTERN = re.compile(r"[a-f0-9]{16}\Z")
SEGMENT_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,80}\Z")
MAX_CONTEXT_MS = 60_000
MAX_UPDATE_BYTES = 4096
STREAM_BLOCK_BYTES = 64 * 1024


class ReviewError(ValueError):
    """Raised when a review manifest or timing value is invalid."""


class RangeNotSatisfiable(ValueError):
    """Raised for invalid or unsatisfiable HTTP byte ranges."""


def _integer(value: object, field: str) -> int:
    if type(value) is not int:
        raise ReviewError(f"{field} must be an integer.")
    return value


def _recording_relative_parts(source: Path, recording_root: Path) -> tuple[str, str, str]:
    try:
        relative = source.resolve().relative_to(recording_root.resolve())
    except ValueError as error:
        raise ReviewError("Manifest audio source is outside the configured recording root.") from error
    parts = relative.parts
    if len(parts) != 3 or source.suffix.casefold() != ".mp3":
        raise ReviewError("Recording must be under Record File\\<Plant>\\<Month>\\<file>.mp3.")
    return parts[0], parts[1], parts[2]


def playback_bounds(
    start_ms: int,
    end_ms: int,
    audio_duration_ms: int,
    *,
    context_before_ms: int = 0,
    context_after_ms: int = 0,
) -> tuple[int, int]:
    values = (start_ms, end_ms, audio_duration_ms, context_before_ms, context_after_ms)
    if any(type(value) is not int for value in values):
        raise ReviewError("Playback times must be integer milliseconds.")
    if (
        audio_duration_ms <= 0
        or start_ms < 0
        or end_ms <= start_ms
        or end_ms > audio_duration_ms
        or context_before_ms < 0
        or context_after_ms < 0
        or context_before_ms > MAX_CONTEXT_MS
        or context_after_ms > MAX_CONTEXT_MS
    ):
        raise ReviewError("Playback boundaries or context are outside valid limits.")
    return max(0, start_ms - context_before_ms), min(
        audio_duration_ms, end_ms + context_after_ms
    )


def parse_byte_range(value: str | None, file_size: int) -> tuple[int, int] | None:
    if file_size < 0:
        raise ValueError("File size cannot be negative.")
    if value is None:
        return None
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value.strip())
    if not match or "," in value:
        raise RangeNotSatisfiable("Only one valid byte range is supported.")
    first, last = match.groups()
    if not first and not last:
        raise RangeNotSatisfiable("The byte range is empty.")
    if file_size == 0:
        raise RangeNotSatisfiable("An empty file has no satisfiable byte ranges.")
    if not first:
        suffix_length = int(last)
        if suffix_length <= 0:
            raise RangeNotSatisfiable("The suffix range must be positive.")
        return max(0, file_size - suffix_length), file_size - 1
    start = int(first)
    end = int(last) if last else file_size - 1
    if start >= file_size or end < start:
        raise RangeNotSatisfiable("The byte range is outside the audio file.")
    return start, min(end, file_size - 1)


class ReviewCatalog:
    """Validated, fixed-root manifest catalog and isolated timing adjustment store."""

    def __init__(
        self,
        manifest_dir: Path = MANIFEST_DIR,
        recording_root: Path = DEFAULT_RECORDING_ROOT,
        adjustment_dir: Path | None = None,
    ) -> None:
        self.manifest_dir = manifest_dir.resolve()
        self.recording_root = recording_root.resolve()
        self.adjustment_dir = (adjustment_dir or manifest_dir).resolve()
        self.adjustment_dir.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._recordings: dict[tuple[str, str], dict[str, Any]] = {}
        self._load_manifests()

    def _load_manifests(self) -> None:
        if not self.manifest_dir.is_dir():
            raise FileNotFoundError(f"VAD manifest directory does not exist: {self.manifest_dir}")
        manifest_paths = [
            *self.manifest_dir.glob("*.vad.json"),
            *self.manifest_dir.glob("*.optimized-review.json"),
        ]
        for manifest_path in sorted(set(manifest_paths)):
            if manifest_path.resolve().parent != self.manifest_dir:
                raise ReviewError("Manifest path escaped the configured manifest directory.")
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            recording = self._validate_manifest(manifest_path, payload)
            key = (recording["mode"], recording["recording_id"])
            if key in self._recordings:
                raise ReviewError(f"Duplicate {key[0]} recording ID in VAD manifests: {key[1]}")
            self._apply_saved_adjustments(recording, manifest_path, key[0])
            self._recordings[key] = recording

    def _validate_manifest(self, manifest_path: Path, payload: object) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ReviewError(f"Manifest must contain a JSON object: {manifest_path.name}")
        mode = "optimized" if payload.get("manifest_type") == "optimized_review_segments" else "original"
        if (
            (mode == "optimized" and not manifest_path.name.endswith(".optimized-review.json"))
            or (mode == "original" and not manifest_path.name.endswith(".vad.json"))
        ):
            raise ReviewError(f"Manifest type does not match its filename: {manifest_path.name}")
        recording_id = payload.get("recording_id")
        source_value = payload.get("source_mp3")
        duration_ms = payload.get("audio_duration_ms")
        segments = payload.get("segments")
        if not isinstance(recording_id, str) or not RECORDING_ID_PATTERN.fullmatch(recording_id):
            raise ReviewError(f"Invalid recording ID in manifest: {manifest_path.name}")
        if not isinstance(source_value, str) or not source_value:
            raise ReviewError(f"Manifest has no audio source: {manifest_path.name}")
        if type(duration_ms) is not int or duration_ms <= 0 or not isinstance(segments, list):
            raise ReviewError(f"Invalid duration or segment list: {manifest_path.name}")

        source = Path(source_value).resolve()
        plant, month, filename = _recording_relative_parts(source, self.recording_root)
        if not source.is_file():
            raise FileNotFoundError(f"Manifest audio source is missing: {source}")
        seen_ids: set[str] = set()
        validated_segments = []
        for segment in segments:
            if not isinstance(segment, dict):
                raise ReviewError("Manifest segment must be a JSON object.")
            segment_id = segment.get("segment_id")
            start_ms = _integer(segment.get("start_ms"), "start_ms")
            end_ms = _integer(segment.get("end_ms"), "end_ms")
            duration = _integer(segment.get("duration_ms"), "duration_ms")
            if not isinstance(segment_id, str) or not SEGMENT_ID_PATTERN.fullmatch(segment_id):
                raise ReviewError("Manifest contains an invalid segment ID.")
            if segment_id in seen_ids:
                raise ReviewError(f"Duplicate segment ID: {segment_id}")
            seen_ids.add(segment_id)
            if not 0 <= start_ms < end_ms <= duration_ms or duration != end_ms - start_ms:
                raise ReviewError(f"Invalid timestamp range for segment {segment_id}.")
            alignment = segment.get("alignment_status")
            transcript = segment.get("transcript_text")
            if alignment != "not_aligned" or transcript is not None:
                raise ReviewError(
                    f"Segment {segment_id} has unsupported transcript alignment data."
                )
            if mode == "optimized":
                source_ids = segment.get("source_segment_ids")
                source_segments = segment.get("source_segments")
                merge_status = segment.get("merge_status")
                if (
                    not isinstance(source_ids, list)
                    or not source_ids
                    or not all(
                        isinstance(source_id, str)
                        and SEGMENT_ID_PATTERN.fullmatch(source_id)
                        for source_id in source_ids
                    )
                    or len(set(source_ids)) != len(source_ids)
                    or not isinstance(source_segments, list)
                    or not all(isinstance(item, dict) for item in source_segments)
                    or [item.get("segment_id") for item in source_segments] != source_ids
                    or not isinstance(merge_status, str)
                    or merge_status not in {"merged", "unchanged"}
                ):
                    raise ReviewError(f"Invalid source lineage for optimized segment {segment_id}.")
            validated_segments.append({
                **segment,
                "original_start_ms": start_ms,
                "original_end_ms": end_ms,
                "start_ms": start_ms,
                "end_ms": end_ms,
            })

        return {
            "mode": mode,
            "recording_id": recording_id,
            "manifest_path": manifest_path,
            "source_path": source,
            "source_filename": filename,
            "plant": plant,
            "month": month,
            "audio_duration_ms": duration_ms,
            "segments": validated_segments,
        }

    def _adjustment_path_for_mode(self, recording_id: str, mode: str) -> Path:
        if mode not in {"original", "optimized"} or not RECORDING_ID_PATTERN.fullmatch(recording_id):
            raise ReviewError("Invalid recording ID or VAD mode.")
        suffix = ".adjustments.json" if mode == "original" else ".optimized.adjustments.json"
        path = (self.adjustment_dir / f"{recording_id}{suffix}").resolve()
        if path.parent != self.adjustment_dir:
            raise ReviewError("Adjustment path escaped the configured workspace.")
        return path

    def _apply_saved_adjustments(
        self, recording: dict[str, Any], manifest_path: Path, mode: str
    ) -> None:
        path = self._adjustment_path_for_mode(recording["recording_id"], mode)
        if not path.exists():
            return
        saved = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(saved, dict)
            or saved.get("recording_id") != recording["recording_id"]
            or saved.get("source_manifest") != manifest_path.name
            or (mode == "optimized" and saved.get("mode") != "optimized")
            or saved.get("mode", "original") != mode
            or not isinstance(saved.get("segments"), dict)
        ):
            raise ReviewError(f"Invalid timing adjustment file: {path.name}")
        by_id = {item["segment_id"]: item for item in recording["segments"]}
        for segment_id, adjustment in saved["segments"].items():
            item = by_id.get(segment_id)
            if not isinstance(adjustment, dict) or item is None:
                raise ReviewError(f"Adjustment references an unknown segment: {segment_id}")
            original_start = _integer(
                adjustment.get("original_start_ms"), "original_start_ms"
            )
            original_end = _integer(
                adjustment.get("original_end_ms"), "original_end_ms"
            )
            start_ms = _integer(adjustment.get("start_ms"), "adjusted start_ms")
            end_ms = _integer(adjustment.get("end_ms"), "adjusted end_ms")
            if (
                original_start != item["original_start_ms"]
                or original_end != item["original_end_ms"]
                or not 0 <= start_ms < end_ms <= recording["audio_duration_ms"]
            ):
                raise ReviewError(f"Invalid saved timing adjustment for {segment_id}.")
            item["start_ms"] = start_ms
            item["end_ms"] = end_ms

    @staticmethod
    def _validate_mode(mode: str) -> str:
        if mode not in {"original", "optimized"}:
            raise ReviewError("mode must be original or optimized.")
        return mode

    def list_recordings(self, mode: str = "original") -> list[dict[str, Any]]:
        mode = self._validate_mode(mode)
        return [
            {
                "recording_id": recording_id,
                "mode": mode,
                "plant": item["plant"],
                "month": item["month"],
                "recording": item["source_filename"],
                "audio_duration_ms": item["audio_duration_ms"],
                "segment_count": len(item["segments"]),
            }
            for (recording_mode, recording_id), item in sorted(self._recordings.items())
            if recording_mode == mode
        ]

    def get_recording(
        self, recording_id: str, mode: str = "original"
    ) -> dict[str, Any] | None:
        mode = self._validate_mode(mode)
        recording = self._recordings.get((mode, recording_id))
        if recording is None:
            return None
        return {
            **{
                key: value for key, value in recording.items()
                if key not in {"source_path", "manifest_path"}
            },
            "source_filename": recording["source_filename"],
            "segments": [dict(segment) for segment in recording["segments"]],
        }

    def get_audio_path(
        self, recording_id: str, mode: str = "original"
    ) -> Path | None:
        mode = self._validate_mode(mode)
        item = self._recordings.get((mode, recording_id))
        if item is None:
            return None
        source = item["source_path"].resolve()
        try:
            source.relative_to(self.recording_root)
        except ValueError as error:
            raise ReviewError("Audio source escaped the configured recording root.") from error
        if source.suffix.casefold() != ".mp3" or not source.is_file():
            raise FileNotFoundError("The configured MP3 source is not available.")
        return source

    def update_segment(
        self,
        recording_id: str,
        segment_id: str,
        start_ms: int,
        end_ms: int,
        mode: str = "original",
    ) -> dict[str, Any] | None:
        mode = self._validate_mode(mode)
        with self._lock:
            recording = self._recordings.get((mode, recording_id))
            if recording is None:
                return None
            segment = next(
                (item for item in recording["segments"] if item["segment_id"] == segment_id),
                None,
            )
            if segment is None:
                return None
            start_ms = _integer(start_ms, "start_ms")
            end_ms = _integer(end_ms, "end_ms")
            if not 0 <= start_ms < end_ms <= recording["audio_duration_ms"]:
                raise ReviewError("Adjusted boundaries must be within the original audio duration.")
            changed = dict(segment)
            changed.update(start_ms=start_ms, end_ms=end_ms)
            segments = {
                item["segment_id"]: {
                    "original_start_ms": item["original_start_ms"],
                    "original_end_ms": item["original_end_ms"],
                    "start_ms": start_ms if item["segment_id"] == segment_id else item["start_ms"],
                    "end_ms": end_ms if item["segment_id"] == segment_id else item["end_ms"],
                }
                for item in recording["segments"]
                if (
                    item["start_ms"] != item["original_start_ms"]
                    or item["end_ms"] != item["original_end_ms"]
                    or item["segment_id"] == segment_id
                )
            }
            path = self._adjustment_path_for_mode(recording_id, mode)
            payload = {
                "schema_version": 1,
                "recording_id": recording_id,
                "mode": mode,
                "source_manifest": recording["manifest_path"].name,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "segments": segments,
            }
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{recording_id}.", suffix=".tmp", dir=self.adjustment_dir
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
                    json.dump(payload, output, ensure_ascii=False, indent=2)
                    output.write("\n")
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary_name, path)
            except Exception:
                Path(temporary_name).unlink(missing_ok=True)
                raise
            segment["start_ms"] = start_ms
            segment["end_ms"] = end_ms
            return dict(segment)


def _parse_context(query: str, name: str) -> int:
    values = parse_qs(query, keep_blank_values=True).get(name, ["0"])
    if len(values) != 1:
        raise ReviewError(f"{name} must be specified once.")
    try:
        value = float(values[0])
    except ValueError as error:
        raise ReviewError(f"{name} must be a number of seconds.") from error
    if not math.isfinite(value) or value < 0 or value > MAX_CONTEXT_MS / 1000:
        raise ReviewError(f"{name} must be between 0 and 60 seconds.")
    return round(value * 1000)


def _parse_mode(query: str) -> str:
    values = parse_qs(query, keep_blank_values=True).get("mode", ["original"])
    if len(values) != 1 or values[0] not in {"original", "optimized"}:
        raise ReviewError("mode must be original or optimized.")
    return values[0]


def make_handler(catalog: ReviewCatalog) -> type[BaseHTTPRequestHandler]:
    class ReviewHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            print(f"vad-review: {self.address_string()} {format % args}")

        def _loopback_host(self) -> bool:
            host = self.headers.get("Host", "")
            try:
                hostname = urlsplit(f"//{host}").hostname
            except ValueError:
                return False
            return hostname in {"127.0.0.1", "localhost", "::1"}

        def _json(self, status: int, payload: object) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

        def _error(self, status: int, message: str) -> None:
            self._json(status, {"error": message})

        def _audio(
            self, recording_id: str, mode: str, *, head_only: bool = False
        ) -> None:
            try:
                path = catalog.get_audio_path(recording_id, mode)
                if path is None:
                    self._error(404, "Recording not found.")
                    return
                size = path.stat().st_size
                byte_range = parse_byte_range(self.headers.get("Range"), size)
            except RangeNotSatisfiable:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{path.stat().st_size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            except (OSError, ReviewError) as error:
                self._error(404, str(error))
                return
            if byte_range is None:
                start, end = 0, size - 1
                status = 200
            else:
                start, end = byte_range
                status = 206
            content_length = max(0, end - start + 1)
            self.send_response(status)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(content_length))
            self.send_header("Cache-Control", "no-store")
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if head_only:
                return
            with path.open("rb") as audio:
                audio.seek(start)
                remaining = content_length
                while remaining:
                    block = audio.read(min(STREAM_BLOCK_BYTES, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)

        def do_HEAD(self) -> None:
            if not self._loopback_host():
                self._error(403, "Only loopback Host headers are accepted.")
                return
            parsed = urlsplit(self.path)
            route = unquote(parsed.path)
            match = re.fullmatch(r"/api/recordings/([a-f0-9]{16})/audio", route)
            if match:
                try:
                    mode = _parse_mode(parsed.query)
                except ReviewError as error:
                    self._error(400, str(error))
                    return
                self._audio(match.group(1), mode, head_only=True)
            else:
                self._error(404, "Not found.")

        def do_GET(self) -> None:
            if not self._loopback_host():
                self._error(403, "Only loopback Host headers are accepted.")
                return
            parsed = urlsplit(self.path)
            route = unquote(parsed.path)
            if route == "/":
                try:
                    body = HTML_PATH.read_bytes()
                except OSError:
                    self._error(500, "Review interface is unavailable.")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if route == "/api/recordings":
                try:
                    mode = _parse_mode(parsed.query)
                    recordings = catalog.list_recordings(mode)
                except ReviewError as error:
                    self._error(400, str(error))
                    return
                self._json(200, {"recordings": recordings})
                return
            match = re.fullmatch(r"/api/recordings/([a-f0-9]{16})", route)
            if match:
                try:
                    mode = _parse_mode(parsed.query)
                except ReviewError as error:
                    self._error(400, str(error))
                    return
                recording = catalog.get_recording(match.group(1), mode)
                self._json(200, recording) if recording else self._error(404, "Recording not found.")
                return
            match = re.fullmatch(
                r"/api/recordings/([a-f0-9]{16})/audio", route
            )
            if match:
                try:
                    mode = _parse_mode(parsed.query)
                except ReviewError as error:
                    self._error(400, str(error))
                    return
                self._audio(match.group(1), mode)
                return
            match = re.fullmatch(
                r"/api/recordings/([a-f0-9]{16})/segments/([A-Za-z0-9_-]{1,80})/playback",
                route,
            )
            if match:
                try:
                    mode = _parse_mode(parsed.query)
                except ReviewError as error:
                    self._error(400, str(error))
                    return
                recording = catalog.get_recording(match.group(1), mode)
                if recording is None:
                    self._error(404, "Recording not found.")
                    return
                segment = next(
                    (item for item in recording["segments"]
                     if item["segment_id"] == match.group(2)),
                    None,
                )
                if segment is None:
                    self._error(404, "Segment not found.")
                    return
                try:
                    start_ms, end_ms = playback_bounds(
                        segment["start_ms"],
                        segment["end_ms"],
                        recording["audio_duration_ms"],
                        context_before_ms=_parse_context(parsed.query, "before"),
                        context_after_ms=_parse_context(parsed.query, "after"),
                    )
                except ReviewError as error:
                    self._error(400, str(error))
                    return
                self._json(200, {"start_ms": start_ms, "end_ms": end_ms})
                return
            self._error(404, "Not found.")

        def do_PATCH(self) -> None:
            if not self._loopback_host():
                self._error(403, "Only loopback Host headers are accepted.")
                return
            route = unquote(urlsplit(self.path).path)
            match = re.fullmatch(
                r"/api/recordings/([a-f0-9]{16})/segments/([A-Za-z0-9_-]{1,80})",
                route,
            )
            if not match:
                self._error(404, "Not found.")
                return
            try:
                mode = _parse_mode(urlsplit(self.path).query)
                if self.headers.get_content_type() != "application/json":
                    raise ReviewError("Content-Type must be application/json.")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_UPDATE_BYTES:
                    raise ReviewError("Request body size is invalid.")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict) or set(payload) != {"start_ms", "end_ms"}:
                    raise ReviewError("Request must contain start_ms and end_ms only.")
                updated = catalog.update_segment(
                    match.group(1),
                    match.group(2),
                    payload["start_ms"],
                    payload["end_ms"],
                    mode,
                )
                if updated is None:
                    self._error(404, "Recording or segment not found.")
                    return
                self._json(200, updated)
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
                self._error(400, str(error))

    return ReviewHandler


def create_server(
    catalog: ReviewCatalog | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    if not ipaddress.ip_address(host).is_loopback:
        raise ValueError("The audio review server must bind to a loopback address.")
    if not 0 <= port <= 65535:
        raise ValueError("Port must be between 0 and 65535.")
    return ThreadingHTTPServer(
        (host, port),
        make_handler(catalog or ReviewCatalog()),
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    server = create_server(host=args.host, port=args.port)
    print(f"Local VAD audio review: http://{server.server_address[0]}:{server.server_address[1]}/")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping local VAD audio review.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
