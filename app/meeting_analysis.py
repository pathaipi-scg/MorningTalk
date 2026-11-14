"""Evidence-linked GPT meeting analysis and summary validation."""

from collections import Counter
from dataclasses import asdict, dataclass
import json
import logging
import re
from typing import Any, Callable
import unicodedata

import requests

from app.azure_openai_http import (
    AzureResponseValidationError,
    extract_generated_text,
    log_azure_error,
    resolve_azure_openai_endpoint,
    response_token_usage,
)
from app.config import AzureConfig
from app.transcript import TranscriptSegment

logger = logging.getLogger(__name__)
_UNIT = (
    r"kilograms?|grams?|milligrams?|tons?|kg|mg|g|mm|cm|km|kw|mw|lb|lbs|m|"
    r"บาท|ชิ้น|ตัน|กิโลกรัม|กรัม|ลิตร|เมตร|วัน|เดือน|ปี|ชั่วโมง|นาที|วินาที|"
    r"คน|แผ่น|เปอร์เซ็นต์|%"
)
_NUMBER_WITH_UNIT = re.compile(
    rf"(?<!\w)(\d+(?:[.,/-]\d+)*)(?:[ \t]*({_UNIT}))?(?!\w)",
    re.IGNORECASE,
)
_CODE = re.compile(
    r"(?<![A-Za-z0-9])(?=[A-Za-z0-9_-]*\d)[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*"
)
_TEMPORAL_WORDS = (
    "วันนี้", "พรุ่งนี้", "มะรืน", "เมื่อวาน", "สัปดาห์หน้า", "เดือนหน้า",
    "ปีหน้า", "ภายใน", "กำหนดส่ง", "กำหนด", "วันจันทร์", "วันอังคาร",
    "วันพุธ", "วันพฤหัสบดี", "วันศุกร์", "วันเสาร์", "วันอาทิตย์",
    "วันที่", "ก่อน", "หลัง",
    "มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
    "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov",
    "dec", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "today", "tomorrow", "yesterday", "next week", "next month",
    "before", "after", "within", "by",
    "deadline", "due",
)
_OWNER_MARKERS = ("ผู้รับผิดชอบ", "มอบหมาย", "เจ้าของงาน", "assigned to", "owner:")


@dataclass(frozen=True)
class EvidenceReference:
    segment_id: int
    quote: str | None = None
    start_char: int | None = None
    end_char: int | None = None

    def to_dict(self) -> dict[str, Any]:
        if self.start_char is not None and self.end_char is not None:
            return {
                "segment_id": self.segment_id,
                "start_char": self.start_char,
                "end_char": self.end_char,
            }
        return {"segment_id": self.segment_id, "quote": self.quote}


@dataclass(frozen=True)
class AnalysisItem:
    text: str
    evidence: tuple[EvidenceReference, ...]
    owner: str | None = None
    deadline: str | None = None


@dataclass(frozen=True)
class MeetingAnalysis:
    overview: AnalysisItem | None
    classifications: tuple[AnalysisItem, ...]
    key_issues: tuple[AnalysisItem, ...]
    actions: tuple[AnalysisItem, ...]
    dependencies: tuple[AnalysisItem, ...]
    schedules: tuple[AnalysisItem, ...]
    risks: tuple[AnalysisItem, ...]
    status: str
    review_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    topics: tuple[AnalysisItem, ...] = ()
    closing_activities: tuple[AnalysisItem, ...] = ()

    @classmethod
    def for_review(
        cls,
        reason: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> "MeetingAnalysis":
        return cls(
            overview=None,
            classifications=(),
            key_issues=(),
            actions=(),
            dependencies=(),
            schedules=(),
            risks=(),
            status="review_required",
            review_reason=reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for field_name in (
            "overview", "classifications", "topics", "key_issues", "actions",
            "dependencies", "schedules", "risks", "closing_activities",
        ):
            source_items = getattr(self, field_name)
            if source_items is None:
                continue
            source_items = (source_items,) if isinstance(source_items, AnalysisItem) else source_items
            payload_items = payload[field_name]
            payload_items = [payload_items] if isinstance(payload_items, dict) else payload_items
            for source_item, payload_item in zip(source_items, payload_items):
                payload_item["evidence"] = [
                    reference.to_dict() for reference in source_item.evidence
                ]
        return payload


class MeetingAnalysisError(RuntimeError):
    """Safe-to-report failure to analyze or validate a meeting transcript."""


class GPTMeetingAnalyzer:
    def __init__(
        self,
        config: AzureConfig,
        *,
        api_version: str,
        session: Any = requests,
        terminology: str = "",
    ) -> None:
        if not config.text_endpoint or not config.text_api_key or not config.text_deployment:
            raise ValueError("Azure OpenAI meeting-analysis configuration is incomplete.")
        if not api_version:
            raise ValueError("AZURE_OPENAI_API_VERSION is required for meeting analysis.")
        self._config = config
        self._api_version = api_version
        self._session = session
        self._terminology = terminology

    def analyze(
        self,
        segments: tuple[TranscriptSegment, ...],
        *,
        on_response: Callable[[Any], None] | None = None,
    ) -> MeetingAnalysis:
        source = {
            "segments": [
                {
                    "segment_id": index,
                    "speaker": segment.speaker_id,
                    "start_seconds": segment.start_seconds,
                    "end_seconds": segment.end_seconds,
                    "text": segment.display_text,
                }
                for index, segment in enumerate(segments, 1)
            ],
            "factory_terminology": self._terminology,
        }
        instructions = (
            "Create a professional, concise but sufficiently detailed Thai morning-meeting "
            "summary. The transcript is the sole factual source; reference/style material "
            "must never provide facts. First identify the actual topics, then group related "
            "discussion into coherent topic summaries. For each topic use one item whose "
            "text begins with a short topic label, then ' — ', then a concise summary of "
            "the condition, operational impact, status, and supported response. Do not "
            "force a fixed taxonomy or fill empty sections. "
            "Preserve material equipment/process names, numbers, codes, units, ranges, "
            "negation, and technical terminology as stated. Separate an observed issue "
            "from a decision or action. Identify owners, dependencies, dates, schedules, "
            "causes, outcomes, and commitments only when explicitly established. Include "
            "a closing activity only when one is present. Avoid repetition and filler. "
            "Do not invent facts, resolve conflicting values, or present uncertain speech "
            "as confirmed; preserve uncertainty and cite both sides of contradictions. "
            "Every factual item must cite one or more source spans. For each span, return "
            "the exact segment_id and start_char/end_char offsets into that segment's "
            "cleaned text. Offsets are zero-based Python Unicode string character indexes; "
            "start_char is inclusive and end_char is exclusive. Do not return or invent a "
            "quote; Python extracts the cited text directly from the source. Never estimate "
            "an offset across segments, and omit an item if its source span is uncertain. "
            "Evidence is internal traceability, not text to repeat in the visible summary. "
            "Numbers, codes, units, dates, owners, and deadlines must be supported by the "
            "cited spans. Set owner and deadline only when explicitly stated; otherwise use "
            "null. Omit unsupported items and return empty arrays when a category is not "
            "supported. Factory terminology is spelling guidance only. "
            "Return only the JSON object with fields overview, classifications, topics, "
            "key_issues, actions, dependencies, schedules, risks, and closing_activities. "
            "Each item has text, nullable owner/deadline, and "
            "evidence[{segment_id, start_char, end_char}]. Provide exactly one "
            "supported overview item; if no overview can be supported, use an empty "
            "evidence array so validation fails closed. All other fields are arrays. "
            "Input JSON:\n" + json.dumps(source, ensure_ascii=False)
        )
        endpoint = resolve_azure_openai_endpoint(
            self._config.text_endpoint,
            self._api_version,
            self._config.text_deployment,
        )
        if endpoint.api_mode == "responses":
            item_schema = {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "owner": {"type": ["string", "null"]},
                    "deadline": {"type": ["string", "null"]},
                    "evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "segment_id": {"type": "integer"},
                                "start_char": {"type": "integer"},
                                "end_char": {"type": "integer"},
                            },
                            "required": ["segment_id", "start_char", "end_char"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["text", "owner", "deadline", "evidence"],
                "additionalProperties": False,
            }
            schema = {
                "type": "object",
                "properties": {
                    "overview": item_schema,
                    "classifications": {"type": "array", "items": item_schema},
                    "topics": {"type": "array", "items": item_schema},
                    "key_issues": {"type": "array", "items": item_schema},
                    "actions": {"type": "array", "items": item_schema},
                    "dependencies": {"type": "array", "items": item_schema},
                    "schedules": {"type": "array", "items": item_schema},
                    "risks": {"type": "array", "items": item_schema},
                    "closing_activities": {"type": "array", "items": item_schema},
                },
                "required": [
                    "overview",
                    "classifications",
                    "topics",
                    "key_issues",
                    "actions",
                    "dependencies",
                    "schedules",
                    "risks",
                    "closing_activities",
                ],
                "additionalProperties": False,
            }
            request_payload = {
                "model": self._config.text_deployment,
                "instructions": "You create evidence-grounded meeting analyses.",
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": instructions}],
                    }
                ],
                "max_output_tokens": 12000,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "meeting_analysis",
                        "strict": True,
                        "schema": schema,
                    }
                },
            }
        else:
            request_payload = {
                "model": self._config.text_deployment,
                "messages": [
                    {"role": "system", "content": "You create evidence-grounded meeting analyses."},
                    {"role": "user", "content": instructions},
                ],
                "max_completion_tokens": 12000,
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
                allow_redirects=False,
            )
        except requests.exceptions.RequestException as error:
            logger.error("Meeting analysis request failed; error_class=%s", type(error).__name__)
            raise MeetingAnalysisError(
                f"Meeting analysis request failed; error_class={type(error).__name__}."
            ) from None
        if on_response is not None:
            on_response(response)
        if not 200 <= response.status_code < 300:
            log_azure_error(
                logger, response, endpoint.url, self._config.text_deployment, endpoint.api_mode
            )
            raise MeetingAnalysisError(
                f"Meeting analysis failed; HTTP {response.status_code}."
            )
        try:
            result = response.json()
        except (ValueError, TypeError):
            logger.error("Meeting analysis returned malformed output.")
            return MeetingAnalysis.for_review("MalformedOutput")
        input_tokens, output_tokens = response_token_usage(result, endpoint.api_mode)
        try:
            content = extract_generated_text(result, endpoint.api_mode)
            payload = json.loads(content)
        except AzureResponseValidationError as error:
            logger.error("Meeting analysis response validation failed; reason=%s", error.reason)
            return MeetingAnalysis.for_review(error.reason, input_tokens, output_tokens)
        except (ValueError, TypeError):
            logger.error("Meeting analysis returned malformed output.")
            return MeetingAnalysis.for_review("MalformedOutput", input_tokens, output_tokens)
        analysis = self._parse_and_validate(payload, segments)
        return MeetingAnalysis(
            overview=analysis.overview,
            classifications=analysis.classifications,
            key_issues=analysis.key_issues,
            actions=analysis.actions,
            dependencies=analysis.dependencies,
            schedules=analysis.schedules,
            risks=analysis.risks,
            status=analysis.status,
            review_reason=analysis.review_reason,
            input_tokens=input_tokens if isinstance(input_tokens, int) else None,
            output_tokens=output_tokens if isinstance(output_tokens, int) else None,
            topics=analysis.topics,
            closing_activities=analysis.closing_activities,
        )

    @staticmethod
    def _parse_and_validate(
        payload: Any,
        segments: tuple[TranscriptSegment, ...],
    ) -> MeetingAnalysis:
        if not isinstance(payload, dict):
            logger.warning(
                "Meeting analysis validation failed; rule=InvalidAnalysisShape "
                "section=payload segment_id=none quote_chars=0 match_failure=payload_not_object"
            )
            return MeetingAnalysis.for_review("InvalidAnalysisShape")
        source_text = {
            index: segment.display_text for index, segment in enumerate(segments, 1)
        }

        def reject(
            section: str,
            failure: str,
            *,
            segment_id: int | None = None,
            quote_chars: int = 0,
            rule: str | None = None,
        ) -> None:
            logger.warning(
                "Meeting analysis evidence validation failed; rule=%s "
                "section=%s segment_id=%s quote_chars=%s match_failure=%s",
                rule or f"{section.title()}EvidenceInvalid",
                section,
                segment_id if segment_id is not None else "none",
                quote_chars,
                failure,
            )

        def parse_item(value: Any, section: str) -> AnalysisItem | None:
            if not isinstance(value, dict):
                reject(section, "item_not_object")
                return None
            text = value.get("text")
            refs = value.get("evidence")
            if not isinstance(text, str) or not text.strip():
                reject(section, "item_text_missing")
                return None
            if not isinstance(refs, list):
                reject(section, "evidence_not_array")
                return None
            if not refs:
                reject(section, "missing_evidence")
                return None
            owner = value.get("owner")
            deadline = value.get("deadline")
            if owner is not None and (not isinstance(owner, str) or not owner.strip()):
                reject(section, "owner_invalid")
                return None
            if deadline is not None and (
                not isinstance(deadline, str) or not deadline.strip()
            ):
                reject(section, "deadline_invalid")
                return None
            evidence: list[EvidenceReference] = []
            evidence_texts: list[str] = []
            for ref in refs:
                if not isinstance(ref, dict):
                    reject(section, "evidence_reference_not_object")
                    return None
                segment_id = ref.get("segment_id")
                supplied_quote = ref.get("quote")
                supplied_quote_chars = (
                    len(supplied_quote) if isinstance(supplied_quote, str) else 0
                )
                if not isinstance(segment_id, int) or isinstance(segment_id, bool):
                    reject(
                        section,
                        "segment_id_invalid",
                        quote_chars=supplied_quote_chars,
                    )
                    return None
                if segment_id not in source_text:
                    reject(
                        section,
                        "segment_id_not_found",
                        segment_id=segment_id,
                        quote_chars=supplied_quote_chars,
                    )
                    return None
                segment = source_text[segment_id]
                start_char = ref.get("start_char")
                end_char = ref.get("end_char")
                if start_char is not None or end_char is not None:
                    if "quote" in ref:
                        reject(
                            section,
                            "quote_overrides_source_span",
                            segment_id=segment_id,
                        )
                        return None
                    try:
                        quote_text = resolve_evidence_span(
                            segment, start_char, end_char
                        )
                    except ValueError as error:
                        reject(
                            section,
                            str(error),
                            segment_id=segment_id,
                            quote_chars=(
                                end_char - start_char
                                if isinstance(start_char, int)
                                and not isinstance(start_char, bool)
                                and isinstance(end_char, int)
                                and not isinstance(end_char, bool)
                                else 0
                            ),
                        )
                        return None
                    evidence.append(
                        EvidenceReference(
                            segment_id,
                            quote_text,
                            start_char,
                            end_char,
                        )
                    )
                else:
                    quote_text = ref.get("quote")
                    quote_chars = len(quote_text) if isinstance(quote_text, str) else 0
                    if not isinstance(quote_text, str) or not quote_text.strip():
                        reject(
                            section,
                            "quote_missing",
                            segment_id=segment_id,
                            quote_chars=quote_chars,
                        )
                        return None
                    if not _evidence_quote_matches(segment, quote_text):
                        reject(
                            section,
                            "quote_not_verbatim",
                            segment_id=segment_id,
                            quote_chars=quote_chars,
                        )
                        return None
                    evidence.append(EvidenceReference(segment_id, quote_text))
                evidence_texts.append(quote_text)

            if not _has_related_evidence(text, evidence_texts):
                reject(
                    section,
                    "evidence_unrelated",
                    quote_chars=sum(map(len, evidence_texts)),
                )
                return None
            failure = _claim_evidence_mismatch(
                (text, owner or "", deadline or ""),
                evidence_texts,
            )
            if failure:
                reject(
                    section,
                    failure,
                    quote_chars=sum(map(len, evidence_texts)),
                )
                return None
            if owner and not _field_supported(owner, evidence_texts):
                reject(
                    section,
                    "owner_not_in_evidence",
                    quote_chars=sum(map(len, evidence_texts)),
                )
                return None
            if deadline and not _field_supported(deadline, evidence_texts):
                reject(
                    section,
                    "deadline_not_in_evidence",
                    quote_chars=sum(map(len, evidence_texts)),
                )
                return None
            return AnalysisItem(text.strip(), tuple(evidence), owner, deadline)

        overview = parse_item(payload.get("overview"), "overview")
        if overview is None:
            return MeetingAnalysis.for_review("OverviewEvidenceInvalid")
        parsed_fields: dict[str, tuple[AnalysisItem, ...]] = {}
        for field in (
            "classifications", "topics", "key_issues", "actions", "dependencies",
            "schedules", "risks", "closing_activities",
        ):
            raw_items = payload.get(field, [])
            if not isinstance(raw_items, list):
                reject(
                    field,
                    "section_not_array",
                    rule=f"Invalid{field.title()}Shape",
                )
                return MeetingAnalysis.for_review(f"Invalid{field.title()}Shape")
            parsed_items = [parse_item(item, field) for item in raw_items]
            if any(item is None for item in parsed_items):
                return MeetingAnalysis.for_review(f"{field.title()}EvidenceInvalid")
            parsed_fields[field] = tuple(item for item in parsed_items if item is not None)
        has_offset_evidence = any(
            reference.start_char is not None
            for item in (overview, *parsed_fields["classifications"],
                         *parsed_fields["topics"], *parsed_fields["key_issues"],
                         *parsed_fields["actions"], *parsed_fields["dependencies"],
                         *parsed_fields["schedules"], *parsed_fields["risks"],
                         *parsed_fields["closing_activities"])
            for reference in item.evidence
        )
        return MeetingAnalysis(
            overview=overview,
            classifications=parsed_fields["classifications"],
            key_issues=parsed_fields["key_issues"],
            actions=parsed_fields["actions"],
            dependencies=parsed_fields["dependencies"],
            schedules=parsed_fields["schedules"],
            risks=parsed_fields["risks"],
            status="review_required" if has_offset_evidence else "validated",
            review_reason="SemanticReviewRequired" if has_offset_evidence else None,
            topics=parsed_fields["topics"],
            closing_activities=parsed_fields["closing_activities"],
        )


def _normalize_evidence_text(text: str) -> str:
    canonical = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", canonical).strip()


def _evidence_quote_matches(source: str, quote: str) -> bool:
    return _normalize_evidence_text(quote) in _normalize_evidence_text(source)


def resolve_evidence_span(
    source_text: str,
    start_char: Any,
    end_char: Any,
) -> str:
    if (
        isinstance(start_char, bool)
        or not isinstance(start_char, int)
        or isinstance(end_char, bool)
        or not isinstance(end_char, int)
    ):
        raise ValueError("offset_not_integer")
    if start_char < 0 or end_char < 0:
        raise ValueError("offset_negative")
    if start_char > len(source_text) or end_char > len(source_text):
        raise ValueError("offset_out_of_range")
    if start_char >= end_char:
        raise ValueError("offset_empty")
    span = source_text[start_char:end_char]
    if not span.strip():
        raise ValueError("span_whitespace_only")
    return span


def _canonical_digits(value: str) -> str:
    return "".join(
        str(unicodedata.decimal(character))
        if unicodedata.category(character) == "Nd"
        else character
        for character in value
    )


def _number_unit_counts(text: str) -> Counter[tuple[str, str]]:
    return Counter(
        (
            _canonical_digits(match.group(1)),
            _normalize_evidence_text(match.group(2) or "").casefold(),
        )
        for match in _NUMBER_WITH_UNIT.finditer(text)
    )


def _code_counts(text: str) -> Counter[str]:
    return Counter(
        _canonical_digits(match).casefold() for match in _CODE.findall(text)
    )


def _temporal_markers(text: str) -> Counter[str]:
    normalized = unicodedata.normalize("NFC", text).casefold()
    found: list[str] = []
    for marker in _TEMPORAL_WORDS:
        if marker.isascii():
            pattern = re.compile(
                rf"(?<![a-z]){re.escape(marker)}(?![a-z])",
                re.IGNORECASE,
            )
        else:
            pattern = re.compile(re.escape(marker))
        found.extend(match.group(0).casefold() for match in pattern.finditer(normalized))
    return Counter(found)


def _has_related_evidence(text: str, evidence_texts: list[str]) -> bool:
    normalized_claim = unicodedata.normalize("NFC", text).casefold()
    claim_words = set(re.findall(r"[a-z0-9]{3,}", normalized_claim))
    claim_thai = {
        normalized_claim[index:index + 3]
        for index in range(max(0, len(normalized_claim) - 2))
        if any("\u0e00" <= char <= "\u0e7f" for char in normalized_claim[index:index + 3])
    }
    for evidence_text in evidence_texts:
        normalized_evidence = unicodedata.normalize("NFC", evidence_text).casefold()
        evidence_words = set(re.findall(r"[a-z0-9]{3,}", normalized_evidence))
        evidence_thai = {
            normalized_evidence[index:index + 3]
            for index in range(max(0, len(normalized_evidence) - 2))
            if any(
                "\u0e00" <= char <= "\u0e7f"
                for char in normalized_evidence[index:index + 3]
            )
        }
        if claim_words & evidence_words or claim_thai & evidence_thai:
            return True
    return False


def _field_supported(value: str, evidence_texts: list[str]) -> bool:
    normalized_value = _normalize_evidence_text(value).casefold()
    return any(
        normalized_value in _normalize_evidence_text(text).casefold()
        for text in evidence_texts
    )


def _claim_evidence_mismatch(
    claims: tuple[str, ...],
    evidence_texts: list[str],
) -> str | None:
    evidence_text = "\n".join(evidence_texts)
    evidence_numbers = _number_unit_counts(evidence_text)
    evidence_values = {number for number, _unit in evidence_numbers}
    evidence_codes = _code_counts(evidence_text)
    evidence_dates = _temporal_markers(evidence_text)
    for claim in claims:
        for (number, unit), count in _number_unit_counts(claim).items():
            if unit and evidence_numbers[(number, unit)] < count:
                return "claim_number_or_unit_not_in_evidence"
            if not unit and number not in evidence_values:
                return "claim_number_not_in_evidence"
            if (
                not unit
                and any(
                    value == number and source_unit
                    for value, source_unit in evidence_numbers
                )
                and not evidence_numbers[(number, "")]
            ):
                return "claim_unit_missing"

        claim_codes = _code_counts(claim)
        if any(count > evidence_codes[code] for code, count in claim_codes.items()):
            return "claim_code_not_in_evidence"

        claim_dates = _temporal_markers(claim)
        if any(
            count > evidence_dates[marker]
            for marker, count in claim_dates.items()
        ):
            return "claim_date_or_deadline_not_in_evidence"

    if any(marker in claims[0].casefold() for marker in _OWNER_MARKERS):
        if not claims[1].strip():
            return "owner_not_explicit"
    return None
