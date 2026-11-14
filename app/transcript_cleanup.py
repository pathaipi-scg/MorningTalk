"""GPT transcript cleanup that treats speaker IDs and timestamps as immutable."""

import json
import logging
import math
import os
import re
from typing import Any
import requests

from app.azure_openai_http import (
    AzureResponseValidationError,
    extract_generated_text,
    log_azure_error,
    resolve_azure_openai_endpoint,
    response_token_usage,
)
from app.config import AzureConfig
from app.factory_context import load_factory_vocabulary
from app.transcript import (
    TranscriptDocument,
    TranscriptQualityFinding,
    TranscriptSegment,
    mark_for_review,
    number_code_count,
    number_code_mismatch,
)

logger = logging.getLogger(__name__)
MAX_CLEANUP_CHARS = 1800
_NUMERIC_UNITS = frozenset(
    {
        "%", "บาท", "ชิ้น", "ตัน", "กิโลกรัม", "กรัม", "ลิตร", "เมตร",
        "วัน", "เดือน", "ปี", "ชั่วโมง", "นาที", "วินาที", "คน", "แผ่น",
        "kg", "g", "mg", "mm", "cm", "km", "kw", "mw", "ton", "tons",
    }
)
_CODE_PREFIXES = frozenset({"lot", "line", "no", "serial"})
_UNCERTAINTY_LEVELS = frozenset({"low", "medium", "high"})


class TranscriptCleanup:
    def __init__(
        self,
        config: AzureConfig,
        *,
        api_version: str,
        session: Any = requests,
        terminology: str | None = None,
    ) -> None:
        if not config.text_endpoint or not config.text_api_key or not config.text_deployment:
            raise ValueError("Azure OpenAI text cleanup configuration is incomplete.")
        if not api_version:
            raise ValueError("AZURE_OPENAI_API_VERSION is required for transcript cleanup.")
        self._config = config
        self._api_version = api_version
        self._session = session
        self._factory_vocabulary = load_factory_vocabulary()
        custom_terminology = (
            terminology
            if terminology is not None
            else os.getenv("MORNINGTALK_TERMINOLOGY", "")
        )
        all_terms = [
            *self._factory_vocabulary.terms,
            *(term.strip() for term in custom_terminology.split(",") if term.strip()),
        ]
        self._terminology = ", ".join(dict.fromkeys(all_terms))

    def clean(self, segments: tuple[TranscriptSegment, ...]) -> TranscriptDocument:
        chunks: list[tuple[int, int, int, str, str]] = []
        for segment_index, segment in enumerate(segments):
            for start, end, text, separator in _split_cleanup_text(segment.raw_text):
                chunks.append((segment_index, start, end, text, separator))
        payload = {
            "segments": [
                {
                    "index": index,
                    "source_segment_index": segment_index,
                    "text": text,
                }
                for index, (_segment_index, _start, _end, text, _separator)
                in enumerate(chunks)
            ],
            "factory_context_th": self._factory_vocabulary.context_th,
            "terminology": self._terminology,
        }
        prompt = (
            "บริบท: พนักงานโรงงานผลิตกระเบื้องหลังคาคอนกรีตกำลังสนทนาเรื่องการผลิต "
            "เครื่องจักร คุณภาพ ซ่อมบำรุง และความปลอดภัย. "
            "แก้คำถอดเสียงที่ผิดได้เฉพาะเมื่อเสียงและบริบทรอบข้างสนับสนุนอย่างชัดเจน. "
            "คำศัพท์โรงงานเป็นเพียงคำใบ้ ห้ามแทนคำที่ไม่คุ้นด้วยคำโรงงานที่ฟังดูเป็นไปได้ "
            "ห้ามแต่งชื่อเครื่องจักร บุคคล จำนวน กำหนดเวลา หรือการกระทำ. "
            "คงคำศัพท์เทคนิคและตัวเลขตามต้นฉบับ. หากคำไม่ชัดหรือดูไม่สมเหตุผล "
            "ให้คงข้อความเดิมและเพิ่มรายการ review_findings เพื่อให้คนตรวจ. "
            "รักษาเจตนาและความหมายการสนทนา ห้ามสรุปหรือเขียนใหม่เป็นรายงานประชุม. "
            "Return JSON with segments and review_findings. "
            "Each finding has index (input excerpt index), original_phrase, "
            "proposed_correction (string or null), context copied exactly from input, "
            "and uncertainty (low, medium, or high). "
            "Return JSON exactly as {\"segments\":[{\"index\":0,\"text\":\"...\"}],"
            "\"review_findings\":[]}. "
            "Keep every utterance, conversational turn, technical term, number, and Lot ID. "
            "Preserve every numeric value and code exactly and in the same order. Do not "
            "add, remove, convert, round, or reformat numbers or codes; preserve their "
            "original digits, separators, units, and attached text verbatim. "
            "Each supplied segment is a consecutive source excerpt. Do not combine "
            "neighboring excerpts or move content between them. "
            "Do not summarize, add, infer, or omit speech. If unclear, copy the original text. "
            "The supplied terminology list is spelling guidance only. Preserve every segment "
            "index and return exactly one text for each input segment. Input JSON:\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        endpoint = resolve_azure_openai_endpoint(
            self._config.text_endpoint,
            self._api_version,
            self._config.text_deployment,
        )
        if endpoint.api_mode == "responses":
            request_payload = {
                "model": self._config.text_deployment,
                "instructions": (
                    "You are a conservative Thai transcript proofreader for concrete roof tile "
                    "manufacturing. Never guess uncertain words."
                ),
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": prompt}],
                    }
                ],
                "max_output_tokens": 16000,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "transcript_cleanup",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "segments": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "index": {"type": "integer"},
                                            "text": {"type": "string"},
                                        },
                                        "required": ["index", "text"],
                                        "additionalProperties": False,
                                    },
                                },
                                "review_findings": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "index": {"type": "integer"},
                                            "original_phrase": {"type": "string"},
                                            "proposed_correction": {
                                                "anyOf": [
                                                    {"type": "string"},
                                                    {"type": "null"},
                                                ]
                                            },
                                            "context": {"type": "string"},
                                            "uncertainty": {
                                                "type": "string",
                                                "enum": ["low", "medium", "high"],
                                            },
                                        },
                                        "required": [
                                            "index",
                                            "original_phrase",
                                            "proposed_correction",
                                            "context",
                                            "uncertainty",
                                        ],
                                        "additionalProperties": False,
                                    },
                                },
                            },
                            "required": ["segments", "review_findings"],
                            "additionalProperties": False,
                        },
                    }
                },
            }
        else:
            request_payload = {
                "model": self._config.text_deployment,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a conservative Thai transcript proofreader for concrete "
                            "roof tile manufacturing. Never guess uncertain words."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_completion_tokens": 16000,
                "response_format": {"type": "json_object"},
            }
        try:
            response = self._session.post(
                endpoint.url,
                params=endpoint.params,
                headers={
                    "api-key": self._config.text_api_key,
                    "Content-Type": "application/json",
                },
                json=request_payload,
                timeout=(30, 1800),
            )
        except requests.exceptions.RequestException as error:
            logger.error("Transcript cleanup request failed; error_class=%s", type(error).__name__)
            return self._review_fallback(segments, type(error).__name__)
        if not 200 <= response.status_code < 300:
            log_azure_error(
                logger, response, endpoint.url, self._config.text_deployment, endpoint.api_mode
            )
            return self._review_fallback(segments, f"HTTP{response.status_code}")
        try:
            result = response.json()
        except (ValueError, TypeError):
            logger.error("Transcript cleanup returned malformed output.")
            return self._review_fallback(segments, "MalformedOutput")
        input_tokens, output_tokens = response_token_usage(result, endpoint.api_mode)
        try:
            content = extract_generated_text(result, endpoint.api_mode)
            parsed = json.loads(content)
        except AzureResponseValidationError as error:
            logger.error("Transcript cleanup response validation failed; reason=%s", error.reason)
            return self._review_fallback(
                segments, error.reason, input_tokens, output_tokens
            )
        except (ValueError, TypeError):
            logger.error("Transcript cleanup returned malformed output.")
            return self._review_fallback(
                segments, "MalformedOutput", input_tokens, output_tokens
            )

        def validation_failure(
            rule: str,
            *,
            index: int | None = None,
            original_text: str | None = None,
            cleaned_text: str | None = None,
            field_type: str | None = None,
            expected_index: int | None = None,
            actual_index: Any = None,
            expected_count: int | None = None,
            actual_count: int | None = None,
            mismatch_category: str | None = None,
            mismatch_position: int | None = None,
        ) -> TranscriptDocument:
            input_chars = len(original_text.strip()) if original_text is not None else None
            output_chars = len(cleaned_text.strip()) if cleaned_text is not None else None
            minimum_output_chars = (
                math.ceil(input_chars * 0.8) if input_chars is not None else None
            )
            input_codes = number_code_count(original_text) if original_text is not None else None
            output_codes = number_code_count(cleaned_text) if cleaned_text is not None else None
            safe_actual_index = (
                actual_index
                if type(actual_index) is int and abs(actual_index) <= 1_000_000
                else "out_of_range"
                if type(actual_index) is int
                else type(actual_index).__name__ if actual_index is not None else None
            )
            logger.warning(
                "Transcript cleanup validation failed; rule=%s segment_index=%s "
                "expected_index=%s actual_index=%s input_chars=%s output_chars=%s "
                "minimum_output_chars=%s input_number_code_count=%s "
                "output_number_code_count=%s expected_segment_count=%s "
                "actual_segment_count=%s field_type=%s numeric_mismatch_category=%s "
                "numeric_mismatch_token_position=%s",
                rule,
                index,
                expected_index,
                safe_actual_index,
                input_chars,
                output_chars,
                minimum_output_chars,
                input_codes,
                output_codes,
                expected_count,
                actual_count,
                field_type,
                mismatch_category,
                mismatch_position,
            )
            return self._review_fallback(
                segments,
                f"ValidationFailed:{rule}",
                input_tokens,
                output_tokens,
            )

        if not isinstance(parsed, dict):
            return validation_failure("PayloadNotObject", field_type=type(parsed).__name__)
        cleaned_items = parsed.get("segments")
        if not isinstance(cleaned_items, list):
            return validation_failure(
                "SegmentsFieldNotArray",
                field_type=type(cleaned_items).__name__,
            )
        if len(cleaned_items) != len(chunks):
            return validation_failure(
                "SegmentCountMismatch",
                expected_count=len(chunks),
                actual_count=len(cleaned_items),
            )
        raw_findings = parsed.get("review_findings", [])
        if not isinstance(raw_findings, list):
            return validation_failure("ReviewFindingsNotArray")

        cleaned_chunks: list[str] = []
        for index, ((_segment_index, _start, _end, original_text, _separator), item) in enumerate(
            zip(chunks, cleaned_items, strict=True)
        ):
            if not isinstance(item, dict):
                return validation_failure(
                    "SegmentItemNotObject",
                    index=index,
                    original_text=original_text,
                    field_type=type(item).__name__,
                )
            actual_index = item.get("index")
            if type(actual_index) is not int or actual_index != index:
                return validation_failure(
                    "SegmentIndexMismatch",
                    index=index,
                    original_text=original_text,
                    expected_index=index,
                    actual_index=actual_index,
                )
            text = item.get("text")
            if not isinstance(text, str):
                return validation_failure(
                    "CleanedTextNotString",
                    index=index,
                    original_text=original_text,
                    field_type=type(text).__name__,
                )
            if not text.strip():
                return validation_failure(
                    "CleanedTextEmpty",
                    index=index,
                    original_text=original_text,
                    cleaned_text=text,
                )
            if len(text.strip()) < len(original_text.strip()) * 0.8:
                return validation_failure(
                    "CleanedTextTooShort",
                    index=index,
                    original_text=original_text,
                    cleaned_text=text,
                )
            mismatch = number_code_mismatch(original_text, text)
            if mismatch is not None:
                return validation_failure(
                    "NumberOrCodeMismatch",
                    index=index,
                    original_text=original_text,
                    cleaned_text=text,
                    mismatch_category=mismatch[0],
                    mismatch_position=mismatch[1],
                )
            for entry in self._factory_vocabulary.entries:
                canonical = entry.canonical_english
                if (
                    entry.confirmation_status != "USER_CONFIRMED"
                    or canonical is None
                    or canonical.casefold() not in original_text.casefold()
                ):
                    continue
                if (
                    canonical.casefold() not in text.casefold()
                    or any(
                        variant.casefold() in text.casefold()
                        for variant in entry.thai_phonetic_variants
                    )
                ):
                    return validation_failure(
                        "ConfirmedTerminologyChanged",
                        index=index,
                        original_text=original_text,
                        cleaned_text=text,
                    )
            cleaned_chunks.append(text.strip())

        findings_by_segment: list[list[TranscriptQualityFinding]] = [
            [] for _ in segments
        ]
        for finding in raw_findings:
            if not isinstance(finding, dict):
                return validation_failure("ReviewFindingNotObject")
            index = finding.get("index")
            phrase = finding.get("original_phrase")
            correction = finding.get("proposed_correction")
            context = finding.get("context")
            uncertainty = finding.get("uncertainty")
            if (
                type(index) is not int
                or not 0 <= index < len(chunks)
                or not isinstance(phrase, str)
                or not phrase.strip()
                or not isinstance(context, str)
                or not context.strip()
                or not isinstance(uncertainty, str)
                or uncertainty not in _UNCERTAINTY_LEVELS
                or correction is not None and not isinstance(correction, str)
            ):
                return validation_failure("ReviewFindingFieldsInvalid")
            source_segment_index, _start, _end, source_excerpt, _separator = chunks[index]
            if (
                phrase not in source_excerpt
                or phrase not in context
                or context not in source_excerpt
            ):
                return validation_failure("ReviewFindingNotSourceGrounded")
            if correction is not None and not correction.strip():
                return validation_failure("ReviewCorrectionEmpty")
            findings_by_segment[source_segment_index].append(
                TranscriptQualityFinding(
                    phrase,
                    correction,
                    context,
                    uncertainty,
                )
            )

        for segment_index, original in enumerate(segments):
            for suspicious_phrase in self._factory_vocabulary.suspicious_phrases:
                if suspicious_phrase.casefold() not in original.raw_text.casefold():
                    continue
                if any(
                    item.original_phrase.casefold() == suspicious_phrase.casefold()
                    for item in findings_by_segment[segment_index]
                ):
                    continue
                start = original.raw_text.casefold().find(suspicious_phrase.casefold())
                context_start = max(0, start - 60)
                context_end = min(
                    len(original.raw_text), start + len(suspicious_phrase) + 60
                )
                findings_by_segment[segment_index].append(
                    TranscriptQualityFinding(
                        suspicious_phrase,
                        None,
                        original.raw_text[context_start:context_end],
                        "high",
                    )
                )

        reconstructed: list[list[str]] = [[] for _ in segments]
        for chunk_index, (segment_index, _start, _end, _text, separator) in enumerate(chunks):
            reconstructed[segment_index].append(cleaned_chunks[chunk_index])
            if separator:
                reconstructed[segment_index].append(separator)
        cleaned = [
            TranscriptSegment(
                original.start_seconds,
                original.end_seconds,
                original.speaker_id,
                original.raw_text,
                original.raw_text
                if findings_by_segment[index]
                else "".join(reconstructed[index]).strip(),
                original.review_required or bool(findings_by_segment[index]),
                tuple(findings_by_segment[index]),
            )
            for index, original in enumerate(segments)
        ]

        has_findings = any(findings_by_segment)
        return TranscriptDocument(
            tuple(cleaned),
            "available",
            "review_required" if has_findings else "completed",
            "UncertainSpeechRequiresReview" if has_findings else None,
            input_tokens=input_tokens if isinstance(input_tokens, int) else None,
            output_tokens=output_tokens if isinstance(output_tokens, int) else None,
        )

    @staticmethod
    def _review_fallback(
        segments: tuple[TranscriptSegment, ...],
        reason: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> TranscriptDocument:
        reviewed = mark_for_review(segments, reason, diarization_status="available")
        return TranscriptDocument(
            reviewed.segments,
            reviewed.diarization_status,
            reviewed.cleanup_status,
            reviewed.review_reason,
            input_tokens,
            output_tokens,
        )


def _split_cleanup_text(text: str) -> list[tuple[int, int, str, str]]:
    if len(text) <= MAX_CLEANUP_CHARS:
        return [(0, len(text), text, "")]

    chunks: list[tuple[int, int, str, str]] = []
    start = 0
    while len(text) - start > MAX_CLEANUP_CHARS:
        limit = start + MAX_CLEANUP_CHARS
        boundary = max(
            (
                index
                for index in range(start + MAX_CLEANUP_CHARS // 2, limit)
                if _is_cleanup_boundary_safe(text, index)
            ),
            default=-1,
        )
        if boundary < 0:
            boundary = next(
                (
                    index
                    for index in range(limit, len(text))
                    if _is_cleanup_boundary_safe(text, index)
                ),
                -1,
            )
        if boundary < 0:
            break
        separator_end = boundary
        while separator_end < len(text) and text[separator_end].isspace():
            separator_end += 1
        if separator_end == len(text):
            break
        chunks.append(
            (start, boundary, text[start:boundary], text[boundary:separator_end])
        )
        start = separator_end
    chunks.append((start, len(text), text[start:], ""))
    return chunks


def _is_cleanup_boundary_safe(text: str, index: int) -> bool:
    if not text[index].isspace():
        return False
    before = text[:index].split()
    after = text[index:].split()
    if not before or not after:
        return False
    left, right = before[-1], after[0]
    if left.isdigit() and right.isdigit():
        return False

    if re.search(r"\d$", left) and right.casefold().rstrip(".,;:") in _NUMERIC_UNITS:
        return False
    if left.casefold().rstrip(".") in _CODE_PREFIXES and re.match(
        r"[\w-]*\d", right, re.IGNORECASE
    ):
        return False
    return True
