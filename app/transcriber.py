"""Transcription provider interfaces and Azure OpenAI adapter."""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
import logging
import math
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Generator, NoReturn
from collections.abc import Sequence
import unicodedata

import requests

from app.config import AzureConfig
from app.factory_context import (
    build_transcription_prompt,
    load_factory_vocabulary,
    transcription_prompt_enabled,
)
from app.workspace_paths import get_workspace_root

logger = logging.getLogger(__name__)
CHUNK_AUDIO_BYTES = 20 * 1024 * 1024


class AzureTranscriptionError(RuntimeError):
    """Safe-to-log Azure transcription failure without request credentials or body."""


@dataclass(frozen=True)
class ChunkProgress:
    index: int
    start_seconds: float
    end_seconds: float
    status: str
    character_count: int = 0
    transcript: str | None = None
    error_class: str | None = None


def _raise_safe_request_error(error: requests.exceptions.RequestException) -> NoReturn:
    chain: list[str] = []
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(type(current).__name__)
        current = current.__cause__ or current.__context__

    safe_chain = " -> ".join(chain)
    logger.error("Azure transcription request failed; exception_chain=%s", safe_chain)
    raise AzureTranscriptionError(
        f"Azure transcription request failed; exception_chain={safe_chain}."
    ) from None


class Transcriber(ABC):
    @abstractmethod
    def transcribe(self, audio_path: Path) -> str:
        """Return the complete recognized speech without summarizing or translating."""


class AzureOpenAITranscriber(Transcriber):
    def __init__(
        self,
        config: AzureConfig,
        session: Any = requests,
        chunk_callback: Callable[[ChunkProgress], None] | None = None,
    ) -> None:
        config.validate_transcription()
        self._config = config
        self._session = session
        self._chunk_duration = config.chunk_duration_seconds
        self._chunk_overlap = config.chunk_overlap_seconds
        self._chunk_callback = chunk_callback
        self._factory_vocabulary = (
            load_factory_vocabulary()
            if transcription_prompt_enabled(config.transcription_api_version)
            else None
        )

    def transcribe(self, audio_path: Path) -> str:
        with self._audio_chunks(audio_path) as chunks:
            transcripts = []
            durations = []
            for index, chunk in enumerate(chunks, 1):
                duration = self._probe_duration(chunk)
                durations.append(duration)
                start_seconds = (index - 1) * (self._chunk_duration - self._chunk_overlap)
                logger.info("Chunk %d/%d duration=%.1fs status=started", index, len(chunks), duration)
                try:
                    text = self._transcribe_chunk(chunk)
                except Exception as error:
                    self._notify_chunk(ChunkProgress(
                        index, start_seconds, start_seconds + duration, "failed",
                        error_class=type(error).__name__,
                    ))
                    logger.error(
                        "Chunk %d/%d duration=%.1fs status=failed error_class=%s",
                        index, len(chunks), duration, type(error).__name__,
                    )
                    raise
                if not text.strip():
                    self._notify_chunk(ChunkProgress(
                        index, start_seconds, start_seconds + duration, "failed",
                        error_class="EmptyTranscription",
                    ))
                    logger.error("Chunk %d/%d duration=%.1fs status=empty characters=0", index, len(chunks), duration)
                    raise ValueError(f"Transcription returned an empty result for chunk {index}/{len(chunks)}.")
                transcripts.append(text)
                self._notify_chunk(ChunkProgress(
                    index, start_seconds, start_seconds + duration, "completed",
                    character_count=len(text), transcript=text,
                ))
                logger.info(
                    "Chunk %d/%d duration=%.1fs status=completed characters=%d",
                    index, len(chunks), duration, len(text),
                )
        transcript = transcripts[0]
        for following in transcripts[1:]:
            transcript = _merge_overlapping_transcripts(transcript, following)
        transcript = transcript.strip()
        total_duration = sum(durations) - self._chunk_overlap * max(0, len(durations) - 1)
        characters_per_minute = len(transcript) / max(total_duration / 60, 1 / 60)
        if characters_per_minute < self._config.quality_warning_chars_per_minute:
            logger.warning(
                "Transcript quality warning: characters_per_minute=%.2f below configured threshold=%.2f; "
                "review completeness before relying on this transcript.",
                characters_per_minute, self._config.quality_warning_chars_per_minute,
            )
        return transcript

    def _notify_chunk(self, progress: ChunkProgress) -> None:
        if self._chunk_callback is not None:
            self._chunk_callback(progress)

    @staticmethod
    def _probe_duration(audio_path: Path) -> float:
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            raise RuntimeError("ffprobe is not accessible on PATH.")
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
            check=True, capture_output=True, text=True, timeout=60,
        )
        try:
            duration = float(result.stdout.strip())
        except ValueError as error:
            raise ValueError(f"Unable to read audio duration: {audio_path.name}") from error
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Audio has an invalid duration: {audio_path.name}")
        return duration

    @contextmanager
    def _audio_chunks(self, audio_path: Path) -> Generator[Sequence[Path], None, None]:
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        if not ffmpeg or not ffprobe:
            raise RuntimeError(
                "Install FFmpeg and ffprobe and ensure both are accessible on PATH."
            )
        temp_root = get_workspace_root() / "temp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="morningtalk-", dir=temp_root
        ) as directory:
            total_duration = self._probe_duration(audio_path)
            chunks: list[Path] = []
            starts = [0.0]
            while starts[-1] + self._chunk_duration < total_duration:
                starts.append(starts[-1] + self._chunk_duration - self._chunk_overlap)
            for index, start in enumerate(starts):
                duration = min(self._chunk_duration, total_duration - start)
                chunk_path = Path(directory) / f"chunk-{index:04d}.mp3"
                subprocess.run(
                    [
                        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                        "-ss", f"{start:.3f}", "-i", str(audio_path),
                        "-t", f"{duration:.3f}", "-map", "0:a:0", "-vn",
                        "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame",
                        "-b:a", "64k", "-f", "mp3", str(chunk_path),
                    ],
                    check=True,
                    capture_output=True,
                    timeout=self._chunk_duration + 120,
                )
                if not chunk_path.is_file() or chunk_path.stat().st_size == 0:
                    raise ValueError(f"Generated audio chunk is empty: {audio_path.name}")
                if chunk_path.stat().st_size > CHUNK_AUDIO_BYTES:
                    raise ValueError(f"Generated audio chunk exceeds the safe request limit: {audio_path.name}")
                chunks.append(chunk_path)
            yield chunks

    def _transcribe_chunk(self, audio_path: Path) -> str:
        api_version = self._config.transcription_api_version.casefold()
        if api_version == "v1":
            url = f"{self._config.transcription_endpoint}/openai/v1/audio/transcriptions"
            params = None
            data = {
                "model": self._config.transcription_deployment,
                "language": "th",
            }
        else:
            url = (
                f"{self._config.transcription_endpoint}/openai/deployments/"
                f"{self._config.transcription_deployment}/audio/transcriptions"
            )
            params = {"api-version": self._config.transcription_api_version}
            data = {"language": "th"}
        if self._factory_vocabulary is not None:
            data["prompt"] = build_transcription_prompt(self._factory_vocabulary)
        with audio_path.open("rb") as audio_file:
            try:
                response = self._session.post(
                    url,
                    params=params,
                    headers={"api-key": self._config.transcription_api_key},
                    files={"file": ("audio.mp3", audio_file, "audio/mpeg")},
                    data=data,
                    timeout=(30, 1800),
                )
            except requests.exceptions.ConnectTimeout as error:
                _raise_safe_request_error(error)
            except requests.exceptions.ReadTimeout as error:
                _raise_safe_request_error(error)
            except requests.exceptions.ProxyError as error:
                _raise_safe_request_error(error)
            except requests.exceptions.SSLError as error:
                _raise_safe_request_error(error)
            except requests.exceptions.ConnectionError as error:
                _raise_safe_request_error(error)
            except requests.exceptions.RequestException as error:
                _raise_safe_request_error(error)

        status_code = getattr(response, "status_code", None)
        if not isinstance(status_code, int):
            raise AzureTranscriptionError("Azure transcription returned an invalid HTTP response.")
        if not 200 <= status_code < 300:
            raise AzureTranscriptionError(
                f"Azure transcription request failed (HTTP {status_code}).",
                )
        try:
            result = response.json()
        except (ValueError, TypeError):
            raise AzureTranscriptionError("Azure transcription returned invalid JSON.") from None
        if not isinstance(result, dict):
            raise AzureTranscriptionError("Azure transcription returned an invalid response shape.")
        text = result.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Azure transcription response did not contain transcript text.")
        return text.strip()


def _merge_overlapping_transcripts(previous: str, following: str) -> str:
    """Drop only a verified common suffix/prefix from the ten-second audio overlap."""
    def significant(char: str) -> bool:
        return char.isalnum() or unicodedata.category(char).startswith("M")

    previous_chars = [(char.casefold(), index) for index, char in enumerate(previous) if significant(char)]
    following_chars = [(char.casefold(), index) for index, char in enumerate(following) if significant(char)]
    previous_normalized = "".join(char for char, _ in previous_chars)[-512:]
    following_normalized = "".join(char for char, _ in following_chars)[:512]
    match = SequenceMatcher(None, previous_normalized, following_normalized, autojunk=False).find_longest_match(
        0, len(previous_normalized), 0, len(following_normalized)
    )
    if match.size >= 4 and match.a + match.size == len(previous_normalized) and match.b == 0:
        cutoff = following_chars[match.b + match.size - 1][1] + 1
        return previous.rstrip() + following[cutoff:]
    logger.warning("Could not align transcript overlap; retaining both chunk transcripts.")
    return previous.rstrip() + "\n" + following.lstrip()


def estimate_chunk_count(duration: float, chunk_duration: int = 180, overlap: int = 10) -> int:
    if duration <= 0 or chunk_duration <= 0 or not 0 <= overlap < chunk_duration:
        raise ValueError("Duration and chunk settings must be positive, with overlap shorter than chunk duration.")
    if duration <= chunk_duration:
        return 1
    return 1 + math.ceil((duration - chunk_duration) / (chunk_duration - overlap))


class FasterWhisperTranscriber(Transcriber):
    def transcribe(self, audio_path: Path) -> str:
        raise NotImplementedError("Offline faster-whisper support is reserved for a future phase.")
