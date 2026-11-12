"""Structured transcript segments and conservative transcript validation."""

from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True)
class TranscriptSegment:
    start_seconds: float | None
    end_seconds: float | None
    speaker_id: str | None
    raw_text: str
    cleaned_text: str | None = None
    review_required: bool = False

    @property
    def display_text(self) -> str:
        return self.cleaned_text if self.cleaned_text is not None else self.raw_text


@dataclass(frozen=True)
class TranscriptDocument:
    segments: tuple[TranscriptSegment, ...]
    diarization_status: str
    cleanup_status: str
    review_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def raw_transcript(self) -> str:
        return "\n".join(segment.raw_text for segment in self.segments)

    @property
    def cleaned_transcript(self) -> str:
        return "\n".join(segment.display_text for segment in self.segments)


_NUMERIC_VALUE = r"\d+(?:[.,/-]\d+)*"
_NUMERIC_UNIT = (
    r"(?:kilograms?|grams?|milligrams?|tons?|kg|mg|g|mm|cm|km|kw|mw|lb|lbs|m|"
    r"บาท|ชิ้น|ตัน|กิโลกรัม|กรัม|ลิตร|เมตร|วัน|เดือน|ปี|ชั่วโมง|นาที|วินาที|"
    r"คน|แผ่น|เปอร์เซ็นต์|%)"
)
_NUMBER_OR_CODE = re.compile(
    rf"(?<!\w){_NUMERIC_VALUE}(?:[ \t]*{_NUMERIC_UNIT})(?!\w)"
    rf"|(?<!\w){_NUMERIC_VALUE}(?!\w)"
    r"|\b[A-Za-z]{2,}[-_]?\d(?:[A-Za-z0-9_-]|\d)*\b"
)


def validate_segment_times(segments: tuple[TranscriptSegment, ...]) -> None:
    previous_start = -1.0
    for segment in segments:
        start, end = segment.start_seconds, segment.end_seconds
        if start is None and end is None:
            continue
        if start is None or end is None or start < 0 or end < start:
            raise ValueError("Transcript segment has invalid timestamps.")
        if start < previous_start:
            raise ValueError("Transcript segments are not in chronological order.")
        previous_start = start


def number_code_count(text: str) -> int:
    return len(_NUMBER_OR_CODE.findall(text))


def _canonicalize_digits(token: str) -> str:
    canonical = "".join(
        str(unicodedata.decimal(character))
        if unicodedata.category(character) == "Nd"
        else character
        for character in token
    )
    return re.sub(r"[ \t]+", "", canonical)


def number_code_mismatch(
    raw_text: str,
    cleaned_text: str,
) -> tuple[str, int] | None:
    expected = tuple(
        _canonicalize_digits(token) for token in _NUMBER_OR_CODE.findall(raw_text)
    )
    actual = tuple(
        _canonicalize_digits(token) for token in _NUMBER_OR_CODE.findall(cleaned_text)
    )
    if expected == actual:
        return None

    position = next(
        (
            index + 1
            for index, (expected_token, actual_token) in enumerate(zip(expected, actual))
            if expected_token != actual_token
        ),
        min(len(expected), len(actual)) + 1,
    )
    if len(expected) == len(actual) and sorted(expected) == sorted(actual):
        return "reordered", position

    def is_subsequence(shorter: tuple[str, ...], longer: tuple[str, ...]) -> bool:
        cursor = 0
        for token in longer:
            if cursor < len(shorter) and shorter[cursor] == token:
                cursor += 1
        return cursor == len(shorter)

    if len(actual) > len(expected) and is_subsequence(expected, actual):
        return "inserted", position
    if len(expected) > len(actual) and is_subsequence(actual, expected):
        return "deleted", position
    return "changed", position


def preserves_numbers_and_codes(raw_text: str, cleaned_text: str) -> bool:
    return number_code_mismatch(raw_text, cleaned_text) is None


def mark_for_review(
    segments: tuple[TranscriptSegment, ...],
    reason: str,
    *,
    diarization_status: str,
) -> TranscriptDocument:
    return TranscriptDocument(
        tuple(
            TranscriptSegment(
                segment.start_seconds,
                segment.end_seconds,
                segment.speaker_id,
                segment.raw_text,
                review_required=True,
            )
            for segment in segments
        ),
        diarization_status,
        "review_required",
        reason,
    )


def canonical_speaker_labels(
    speaker_ids: list[str | None],
) -> list[str | None]:
    labels: dict[str, str] = {}
    next_label = 0
    result = []
    for speaker_id in speaker_ids:
        if speaker_id is None or not speaker_id.strip():
            result.append(None)
            continue
        if speaker_id not in labels:
            label = ""
            value = next_label
            while True:
                value, remainder = divmod(value, 26)
                label = chr(65 + remainder) + label
                if value == 0:
                    break
                value -= 1
            labels[speaker_id] = f"Speaker {label}"
            next_label += 1
        result.append(labels[speaker_id])
    return result
