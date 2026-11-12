"""Optional Azure Speech fast transcription with diarization and timestamps."""

import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from app.transcript import TranscriptSegment, canonical_speaker_labels, validate_segment_times

logger = logging.getLogger(__name__)
MAX_AUDIO_BYTES = 300 * 1024 * 1024
MAX_AUDIO_SECONDS = 2 * 60 * 60
SPEECH_API_VERSION = "2025-10-15"


class DiarizationUnavailable(RuntimeError):
    """Raised when diarization is not configured or cannot safely complete."""


class AzureSpeechDiarizer:
    def __init__(
        self,
        endpoint: str,
        key: str,
        *,
        api_version: str = SPEECH_API_VERSION,
        max_speakers: int = 8,
        session: Any = requests,
    ) -> None:
        parsed = urlparse(endpoint)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Azure Speech endpoint must be an HTTPS URL without embedded credentials.")
        if not key:
            raise ValueError("Azure Speech key is not configured.")
        if not api_version:
            raise ValueError("Azure Speech API version is not configured.")
        if max_speakers < 1:
            raise ValueError("Maximum speakers must be positive.")
        self._endpoint = endpoint.rstrip("/")
        self._key = key
        self._api_version = api_version
        self._max_speakers = max_speakers
        self._session = session

    def transcribe(self, audio_path: Path) -> tuple[TranscriptSegment, ...]:
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
        if audio_path.stat().st_size > MAX_AUDIO_BYTES:
            raise DiarizationUnavailable("Audio exceeds Azure Speech fast transcription size limits.")
        from app.transcriber import AzureOpenAITranscriber

        if AzureOpenAITranscriber._probe_duration(audio_path) > MAX_AUDIO_SECONDS:
            raise DiarizationUnavailable("Audio exceeds Azure Speech fast transcription duration limits.")

        url = f"{self._endpoint}/speechtotext/transcriptions:transcribe"
        definition = {
            "locales": ["th-TH"],
            "diarization": {"enabled": True, "maxSpeakers": self._max_speakers},
        }
        try:
            with audio_path.open("rb") as audio_file:
                response = self._session.post(
                    url,
                    params={"api-version": self._api_version},
                    headers={"Ocp-Apim-Subscription-Key": self._key},
                    files={"audio": (audio_path.name, audio_file, "audio/mpeg")},
                    data={"definition": json.dumps(definition)},
                    timeout=(30, 1800),
                )
        except requests.exceptions.RequestException as error:
            logger.error("Azure Speech diarization failed; error_class=%s", type(error).__name__)
            raise DiarizationUnavailable(
                f"Azure Speech diarization failed; error_class={type(error).__name__}."
            ) from None

        if not 200 <= response.status_code < 300:
            logger.error("Azure Speech diarization failed; http_status=%d", response.status_code)
            raise DiarizationUnavailable(
                f"Azure Speech diarization failed; HTTP {response.status_code}."
            )
        try:
            payload = response.json()
        except (ValueError, TypeError):
            raise DiarizationUnavailable("Azure Speech returned invalid JSON.") from None
        if not isinstance(payload, dict) or not isinstance(payload.get("phrases"), list):
            raise DiarizationUnavailable("Azure Speech response has no phrase segments.")

        segments = []
        for phrase in payload["phrases"]:
            if not isinstance(phrase, dict):
                raise DiarizationUnavailable("Azure Speech returned a malformed phrase.")
            text = phrase.get("text")
            offset = phrase.get("offsetMilliseconds")
            duration = phrase.get("durationMilliseconds")
            if (
                not isinstance(text, str) or not text.strip()
                or not isinstance(offset, (int, float))
                or not isinstance(duration, (int, float))
            ):
                raise DiarizationUnavailable("Azure Speech returned a phrase without valid text or timing.")
            speaker = phrase.get("speaker")
            if speaker is not None and not isinstance(speaker, (str, int)):
                raise DiarizationUnavailable("Azure Speech returned an invalid speaker label.")
            segments.append(TranscriptSegment(
                start_seconds=offset / 1000,
                end_seconds=(offset + duration) / 1000,
                speaker_id=str(speaker) if speaker is not None else None,
                raw_text=text.strip(),
            ))
        if not segments:
            raise DiarizationUnavailable("Azure Speech returned no speech segments.")

        labels = [
            label or "Speaker unknown"
            for label in canonical_speaker_labels(
                [segment.speaker_id for segment in segments]
            )
        ]
        normalized = tuple(
            TranscriptSegment(
                segment.start_seconds,
                segment.end_seconds,
                label,
                segment.raw_text,
            )
            for segment, label in zip(segments, labels, strict=True)
        )
        try:
            validate_segment_times(normalized)
        except ValueError as error:
            raise DiarizationUnavailable(str(error)) from None
        return normalized
