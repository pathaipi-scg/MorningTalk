"""Vocabulary master loading and conservative terminology normalization."""

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import re
from typing import Any

from app.transcript import TermNormalizationRecord, TranscriptSegment

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FACTORY_VOCABULARY_PATH = PROJECT_ROOT / "config" / "factory_vocabulary.json"
TRANSCRIPTION_PROMPT_API_VERSIONS = frozenset(
    {"2024-10-21", "2025-04-01-preview"}
)
TERM_STATUSES = frozenset(
    {"USER_CONFIRMED", "STRONG_CANDIDATE", "UNCERTAIN", "CONFLICTING_EVIDENCE"}
)


@dataclass(frozen=True)
class VocabularyEntry:
    master_id: str
    canonical_english: str | None
    thai_phonetic_variants: tuple[str, ...]
    verified_alternative_spellings: tuple[str, ...]
    candidate_alternative_spellings: tuple[str, ...]
    spelling_observations: tuple[dict[str, Any], ...]
    category: str
    confirmation_status: str
    evidence_sources: tuple[str, ...]
    recording_references: tuple[str, ...]
    chunk_references: tuple[str, ...]
    occurrence_count: int
    first_observed_recording: str | None
    last_observed_recording: str | None
    context_cues: tuple[str, ...]


@dataclass(frozen=True)
class FactoryVocabulary:
    context_th: str
    entries: tuple[VocabularyEntry, ...]

    @property
    def confirmed_terms(self) -> tuple[str, ...]:
        return tuple(
            entry.canonical_english
            for entry in self.entries
            if entry.confirmation_status == "USER_CONFIRMED"
            and entry.canonical_english is not None
        )

    @property
    def terms(self) -> tuple[str, ...]:
        return self.confirmed_terms

    @property
    def suspicious_phrases(self) -> tuple[str, ...]:
        return tuple(
            phrase
            for entry in self.entries
            if entry.confirmation_status != "USER_CONFIRMED"
            for phrase in entry.thai_phonetic_variants
        )


@dataclass(frozen=True)
class NormalizationResult:
    normalized_text: str
    replacements: tuple[dict[str, Any], ...]
    review_findings: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class SegmentNormalizationResult:
    raw_transcript: str
    normalized_transcript: str
    segments: tuple[TranscriptSegment, ...]
    replacements: tuple[TermNormalizationRecord, ...]
    review_findings: tuple[dict[str, Any], ...]


def load_factory_vocabulary(path: Path | None = None) -> FactoryVocabulary:
    configured_value = os.getenv("MORNINGTALK_FACTORY_VOCABULARY_PATH", "").strip()
    configured_path = path or (
        Path(configured_value) if configured_value else DEFAULT_FACTORY_VOCABULARY_PATH
    )
    try:
        payload: Any = json.loads(configured_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Unable to load Technical Vocabulary Master: {configured_path}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("Technical Vocabulary Master must be a JSON object.")

    context = payload.get("context_th")
    raw_entries = payload.get("entries")
    if not isinstance(context, str) or not context.strip():
        raise ValueError(
            "Technical Vocabulary Master requires context_th and an entries array."
        )
    if raw_entries is None:
        raw_entries = _legacy_entries(
            payload.get("vocabulary"), payload.get("suspicious_phrases")
        )
    if not isinstance(raw_entries, list):
        raise ValueError(
            "Technical Vocabulary Master requires context_th and an entries array."
        )
    entries = tuple(_parse_entry(item) for item in raw_entries)
    if len({entry.master_id for entry in entries}) != len(entries):
        raise ValueError("Technical Vocabulary Master IDs must be unique.")
    return FactoryVocabulary(context.strip(), entries)


def build_transcription_prompt(vocabulary: FactoryVocabulary) -> str:
    terms = ", ".join(vocabulary.confirmed_terms)
    return (
        f"{vocabulary.context_th} "
        "ถอดเสียงภาษาไทยตามที่ได้ยินจริง ใช้คำศัพท์ที่ยืนยันแล้วต่อไปนี้เป็นเพียงคำใบ้ "
        "ห้ามแทนคำพูดด้วยคำเหล่านี้หากเสียงและบริบทไม่รองรับ: "
        f"{terms}"
    )


def normalization_enabled() -> bool:
    return _feature_flag("MORNINGTALK_ENABLE_TERMINOLOGY_NORMALIZATION")


def transcription_prompt_enabled(api_version: str) -> bool:
    if not _feature_flag("MORNINGTALK_ENABLE_TRANSCRIPTION_PROMPT"):
        return False
    if api_version.casefold() not in TRANSCRIPTION_PROMPT_API_VERSIONS:
        supported = ", ".join(sorted(TRANSCRIPTION_PROMPT_API_VERSIONS))
        raise ValueError(
            "Transcription prompting is not enabled for this documented API version; "
            f"supported versions: {supported}."
        )
    return True


def normalize_technical_terms(
    text: str,
    *,
    recording: str,
    context_text: str | None = None,
    context_text_offset: int | None = None,
    chunk_reference: str | None = None,
    vocabulary: FactoryVocabulary | None = None,
) -> NormalizationResult:
    master = vocabulary or load_factory_vocabulary()
    context = text if context_text is None else context_text
    replacements: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    accepted_replacements: list[tuple[int, int, str, dict[str, Any]]] = []
    claimed_spans: list[tuple[int, int]] = []

    candidates = sorted(
        (
            (entry, variant)
            for entry in master.entries
            if entry.confirmation_status == "USER_CONFIRMED"
            for variant in entry.thai_phonetic_variants
        ),
        key=lambda item: len(item[1]),
        reverse=True,
    )
    for entry, variant in candidates:
        if entry.canonical_english is None:
            continue
        for match in re.finditer(re.escape(variant), text, flags=re.IGNORECASE):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in claimed_spans):
                continue
            evidence_context = _bounded_context(text, span[0], span[1])
            context_start = (
                context_text_offset + span[0]
                if context_text_offset is not None
                else span[0] if context_text is None or context_text == text else None
            )
            if not _context_matches(
                context,
                variant,
                entry.context_cues,
                phrase_start=context_start,
            ):
                findings.append(
                    _review_finding(
                        entry,
                        variant,
                        evidence_context,
                        recording,
                        chunk_reference,
                        "ConfirmedAliasWithoutEquipmentContext",
                    )
                )
                continue
            claimed_spans.append(span)
            accepted_replacements.append(
                (
                    span[0],
                    span[1],
                    entry.canonical_english,
                    {
                        "original_phrase": variant,
                        "replacement": entry.canonical_english,
                        "vocabulary_master_id": entry.master_id,
                        "recording": recording,
                        "chunk_reference": chunk_reference,
                        "evidence_context": evidence_context,
                        "confirmation_status": entry.confirmation_status,
                    },
                )
            )

    for entry in master.entries:
        if entry.confirmation_status == "USER_CONFIRMED":
            continue
        for phrase in entry.thai_phonetic_variants:
            for match in re.finditer(re.escape(phrase), text, flags=re.IGNORECASE):
                findings.append(
                    _review_finding(
                        entry,
                        phrase,
                        _bounded_context(text, *match.span()),
                        recording,
                        chunk_reference,
                        "UnconfirmedTerminologyCandidate",
                    )
                )

    normalized = text
    for replacement in sorted(
        accepted_replacements,
        key=lambda item: item[0],
        reverse=True,
    ):
        normalized = (
            normalized[: replacement[0]]
            + replacement[2]
            + normalized[replacement[1] :]
        )
    replacements = tuple(
        replacement[3]
        for replacement in sorted(
            accepted_replacements,
            key=lambda item: item[0],
        )
    )
    return NormalizationResult(normalized, replacements, tuple(findings))


def normalize_transcript_segments(
    segments: tuple[TranscriptSegment, ...],
    *,
    recording: str,
    vocabulary: FactoryVocabulary | None = None,
) -> SegmentNormalizationResult:
    master = vocabulary or load_factory_vocabulary()
    normalized_segments: list[TranscriptSegment] = []
    records: list[TermNormalizationRecord] = []
    findings: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        context_parts = [
            item.raw_text
            for item in segments[max(0, index - 1) : min(len(segments), index + 2)]
        ]
        context_offset = (
            len(context_parts[0]) + 1
            if index > 0
            else 0
        )
        chunk_reference = (
            f"{segment.start_seconds:.3f}-{segment.end_seconds:.3f}s"
            if segment.start_seconds is not None and segment.end_seconds is not None
            else f"segment:{index + 1}"
        )
        result = normalize_technical_terms(
            segment.raw_text,
            recording=recording,
            context_text=" ".join(context_parts),
            context_text_offset=context_offset,
            chunk_reference=chunk_reference,
            vocabulary=master,
        )
        normalized_segments.append(
            replace(segment, raw_text=result.normalized_text, cleaned_text=None)
        )
        records.extend(TermNormalizationRecord(**item) for item in result.replacements)
        findings.extend(result.review_findings)
    return SegmentNormalizationResult(
        "\n".join(segment.raw_text for segment in segments),
        "\n".join(segment.raw_text for segment in normalized_segments),
        tuple(normalized_segments),
        tuple(records),
        tuple(findings),
    )


def _parse_entry(value: Any) -> VocabularyEntry:
    if not isinstance(value, dict):
        raise ValueError("Each vocabulary entry must be an object.")
    master_id = value.get("master_id")
    canonical = value.get("canonical_english")
    if not isinstance(master_id, str) or not master_id.strip():
        raise ValueError("Vocabulary entries require a non-empty master_id.")
    if canonical is not None and (not isinstance(canonical, str) or not canonical.strip()):
        raise ValueError(f"Invalid canonical spelling for {master_id}.")
    status = value.get("confirmation_status")
    if status not in TERM_STATUSES:
        raise ValueError(f"Invalid confirmation status for {master_id}.")
    category = value.get("category")
    if not isinstance(category, str) or not category.strip():
        raise ValueError(f"Vocabulary entry {master_id} requires a category.")
    arrays = {}
    for key in (
        "thai_phonetic_variants",
        "verified_alternative_spellings",
        "candidate_alternative_spellings",
        "evidence_sources",
        "recording_references",
        "chunk_references",
        "context_cues",
    ):
        array = value.get(key)
        if not _is_string_list(array):
            raise ValueError(f"Vocabulary entry {master_id} requires {key} as strings.")
        arrays[key] = tuple(_unique_nonempty(array))
    spelling_observations = value.get("spelling_observations", [])
    if not isinstance(spelling_observations, list) or not all(
        isinstance(item, dict)
        and isinstance(item.get("spelling"), str)
        and isinstance(item.get("occurrences"), int)
        and item["occurrences"] >= 0
        and _is_string_list(item.get("recordings"))
        for item in spelling_observations
    ):
        raise ValueError(f"Invalid spelling_observations for {master_id}.")
    count = value.get("occurrence_count")
    if not isinstance(count, int) or count < 0:
        raise ValueError(f"Vocabulary entry {master_id} requires a non-negative occurrence_count.")
    first = value.get("first_observed_recording")
    last = value.get("last_observed_recording")
    if first is not None and not isinstance(first, str):
        raise ValueError(f"Invalid first_observed_recording for {master_id}.")
    if last is not None and not isinstance(last, str):
        raise ValueError(f"Invalid last_observed_recording for {master_id}.")
    return VocabularyEntry(
        master_id.strip(),
        canonical.strip() if isinstance(canonical, str) else None,
        arrays["thai_phonetic_variants"],
        arrays["verified_alternative_spellings"],
        arrays["candidate_alternative_spellings"],
        tuple(spelling_observations),
        category.strip(),
        status,
        arrays["evidence_sources"],
        arrays["recording_references"],
        arrays["chunk_references"],
        count,
        first,
        last,
        arrays["context_cues"],
    )


def _context_matches(
    text: str,
    phrase: str,
    cues: tuple[str, ...],
    *,
    phrase_start: int | None = None,
) -> bool:
    if not cues:
        return False
    normalized = text.casefold()
    phrase_index = (
        phrase_start
        if phrase_start is not None
        and normalized[phrase_start : phrase_start + len(phrase)].casefold()
        == phrase.casefold()
        else normalized.find(phrase.casefold())
    )
    if phrase_index < 0:
        return False
    window = normalized[max(0, phrase_index - 100) : phrase_index + len(phrase) + 100]
    return any(cue.casefold() in window for cue in cues)


def _bounded_context(text: str, start: int, end: int) -> str:
    return text[max(0, start - 80) : min(len(text), end + 80)]


def _review_finding(
    entry: VocabularyEntry,
    phrase: str,
    evidence_context: str,
    recording: str,
    chunk_reference: str | None,
    reason: str,
) -> dict[str, Any]:
    return {
        "original_phrase": phrase,
        "vocabulary_master_id": entry.master_id,
        "recording": recording,
        "chunk_reference": chunk_reference,
        "evidence_context": evidence_context,
        "confirmation_status": entry.confirmation_status,
        "reason": reason,
    }


def _feature_flag(name: str) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return False
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value.")


def _legacy_entries(vocabulary: Any, suspicious: Any) -> list[dict[str, Any]]:
    if not _is_string_list(vocabulary) or not _is_string_list(suspicious):
        raise ValueError(
            "Legacy vocabulary configuration requires string arrays for vocabulary "
            "and suspicious_phrases."
        )
    entries = [
        _entry_payload(
            f"legacy-{index:03d}",
            term,
            "legacy vocabulary hint",
            "STRONG_CANDIDATE",
        )
        for index, term in enumerate(_unique_nonempty(vocabulary), 1)
    ]
    entries.extend(
        _entry_payload(
            f"legacy-phrase-{index:03d}",
            phrase,
            "legacy suspicious phrase; no canonical spelling inferred",
            "UNCERTAIN",
            canonical=None,
            use_term_canonical=False,
            variants=[phrase],
            category="unresolved phrase",
        )
        for index, phrase in enumerate(_unique_nonempty(suspicious), 1)
    )
    return entries


def _entry_payload(
    master_id: str,
    term: str,
    evidence: str,
    status: str,
    *,
    canonical: str | None = None,
    use_term_canonical: bool = True,
    variants: list[str] | None = None,
    category: str = "technical terminology",
) -> dict[str, Any]:
    return {
        "master_id": master_id,
        "canonical_english": term if use_term_canonical else canonical,
        "thai_phonetic_variants": variants or [],
        "verified_alternative_spellings": [],
        "candidate_alternative_spellings": [],
        "spelling_observations": [],
        "category": category,
        "confirmation_status": status,
        "evidence_sources": [evidence],
        "recording_references": [],
        "chunk_references": [],
        "occurrence_count": 0,
        "first_observed_recording": None,
        "last_observed_recording": None,
        "context_cues": [],
    }


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, str) and item.strip() for item in value
    )


def _unique_nonempty(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        if normalized.casefold() not in seen:
            seen.add(normalized.casefold())
            result.append(normalized)
    return result
