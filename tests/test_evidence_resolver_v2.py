import logging

import pytest

from app.meeting_analysis import (
    GPTMeetingAnalyzer,
    resolve_evidence_span,
)
from app.transcript import TranscriptSegment


_SECTIONS = (
    "classifications", "topics", "key_issues", "actions", "dependencies",
    "schedules", "risks", "closing_activities",
)


def _payload(
    source,
    evidence,
    *,
    text=None,
    owner=None,
    deadline=None,
):
    return {
        "overview": {
            "text": text or source,
            "owner": owner,
            "deadline": deadline,
            "evidence": evidence,
        },
        **{section: [] for section in _SECTIONS},
    }


def _parse(payload, source):
    return GPTMeetingAnalyzer._parse_and_validate(
        payload,
        (TranscriptSegment(1, 2, None, source),),
    )


def _reference(source, start=0, end=None, *, segment_id=1):
    return {
        "segment_id": segment_id,
        "start_char": start,
        "end_char": len(source) if end is None else end,
    }


def test_v2_resolves_thai_unicode_by_python_character_offsets():
    source = "เริ่มตรวจเก\u0e49าอี้ Café\u0301 Lot 123"
    phrase = "เก\u0e49าอี้ Café\u0301 Lot 123"
    start = source.index(phrase)
    end = start + len(phrase)
    payload = _payload(source, [_reference(source, start, end)], text=phrase)

    analysis = _parse(payload, source)

    assert analysis.status == "review_required"
    assert analysis.review_reason == "SemanticReviewRequired"
    reference = analysis.overview.evidence[0]
    assert reference.quote == source[start:end] == phrase
    assert (reference.start_char, reference.end_char) == (start, end)
    assert end - start == len(phrase)


def test_resolver_preserves_decomposed_combining_characters_without_normalizing():
    source = "Aเก\u0e49า\u0e35\u0e48B"
    expected = "เก\u0e49า\u0e35\u0e48"
    start, end = 1, len(source) - 1

    resolved = resolve_evidence_span(source, start, end)

    assert resolved == expected
    assert len(resolved) == end - start


@pytest.mark.parametrize(
    ("source", "start", "end", "reason"),
    [
        ("abc", -1, 2, "offset_negative"),
        ("abc", 4, 5, "offset_out_of_range"),
        ("abc", 0, 4, "offset_out_of_range"),
        ("abc", 1, 1, "offset_empty"),
        ("a b", 1, 2, "span_whitespace_only"),
        ("abc", True, 2, "offset_not_integer"),
        ("abc", 0, 1.0, "offset_not_integer"),
    ],
)
def test_v2_rejects_invalid_spans(source, start, end, reason, caplog):
    payload = _payload(source, [_reference(source, start, end)], text=source)

    with caplog.at_level(logging.WARNING, logger="app.meeting_analysis"):
        analysis = _parse(payload, source)

    assert analysis.status == "review_required"
    assert analysis.review_reason == "OverviewEvidenceInvalid"
    assert f"match_failure={reason}" in caplog.text


def test_v2_rejects_missing_segment_id():
    source = "ตรวจ Encoder Lot 123"
    payload = _payload(
        source,
        [_reference(source, 0, len(source), segment_id=2)],
    )

    analysis = _parse(payload, source)

    assert analysis.status == "review_required"
    assert analysis.review_reason == "OverviewEvidenceInvalid"


def test_v2_rejects_boolean_segment_id():
    source = "ตรวจ Encoder Lot 123"
    reference = _reference(source, 0, len(source), segment_id=True)

    analysis = _parse(_payload(source, [reference]), source)

    assert analysis.review_reason == "OverviewEvidenceInvalid"


@pytest.mark.parametrize(
    ("source", "claim", "failure"),
    [
        ("ตรวจ Encoder Lot 123", "ตรวจ Encoder Lot 124", "claim_number_not_in_evidence"),
        ("ตรวจ Model AB-21", "ตรวจ Model CD-21", "claim_code_not_in_evidence"),
        ("ตรวจ 10 kg", "ตรวจ 10 g", "claim_number_or_unit_not_in_evidence"),
        ("ตรวจ 10 kg", "ตรวจ 10", "claim_unit_missing"),
        ("ตรวจ Encoder วันนี้", "ตรวจ Encoder พรุ่งนี้",
         "claim_date_or_deadline_not_in_evidence"),
    ],
)
def test_v2_rejects_unsupported_numbers_codes_units_and_dates(
    source, claim, failure, caplog
):
    payload = _payload(source, [_reference(source)], text=claim)

    with caplog.at_level(logging.WARNING, logger="app.meeting_analysis"):
        analysis = _parse(payload, source)

    assert analysis.status == "review_required"
    assert f"match_failure={failure}" in caplog.text


def test_v2_rejects_unsupported_owner_and_deadline_fields(caplog):
    source = "ตรวจ Encoder Lot 123"
    payload = _payload(
        source,
        [_reference(source)],
        owner="สมชาย",
        deadline="พรุ่งนี้",
    )

    with caplog.at_level(logging.WARNING, logger="app.meeting_analysis"):
        analysis = _parse(payload, source)

    assert analysis.status == "review_required"
    assert "match_failure=claim_date_or_deadline_not_in_evidence" in caplog.text

    payload["overview"]["deadline"] = None
    with caplog.at_level(logging.WARNING, logger="app.meeting_analysis"):
        analysis = _parse(payload, source)
    assert "match_failure=owner_not_in_evidence" in caplog.text


def test_v2_accepts_explicit_owner_and_deadline_when_source_backed():
    source = "สมชายตรวจ Encoder Lot 123 ภายในพรุ่งนี้"
    payload = _payload(
        source,
        [_reference(source)],
        text="สมชายตรวจ Encoder Lot 123 ภายในพรุ่งนี้",
        owner="สมชาย",
        deadline="พรุ่งนี้",
    )

    analysis = _parse(payload, source)

    assert analysis.review_reason == "SemanticReviewRequired"
    assert analysis.overview.owner == "สมชาย"
    assert analysis.overview.deadline == "พรุ่งนี้"


def test_v2_accepts_multiple_spans_but_retains_semantic_review_status():
    source = "ตรวจ Encoder Lot 123 และทดสอบพรุ่งนี้"
    first = (0, source.index(" และ"))
    second = (source.index("ทดสอบ"), len(source))
    payload = _payload(
        source,
        [
            _reference(source, *first),
            _reference(source, *second),
        ],
    )

    analysis = _parse(payload, source)

    assert analysis.review_reason == "SemanticReviewRequired"
    assert len(analysis.overview.evidence) == 2
    assert analysis.overview.evidence[0].quote == source[slice(*first)]
    assert analysis.overview.evidence[1].quote == source[slice(*second)]


def test_v2_rejects_a_supplied_quote_that_could_override_offsets():
    source = "ตรวจ Encoder Lot 123"
    reference = _reference(source)
    reference["quote"] = "different source wording"

    analysis = _parse(_payload(source, [reference]), source)

    assert analysis.review_reason == "OverviewEvidenceInvalid"


def test_legacy_validated_analysis_remains_compatible():
    source = "ตรวจ Encoder Lot 123"
    legacy = {
        "segment_id": 1,
        "quote": source,
    }

    analysis = _parse(
        _payload(source, [legacy]),
        source,
    )

    assert analysis.status == "validated"
    assert analysis.review_reason is None
    assert analysis.overview.evidence[0].quote == source


def test_invalid_legacy_evidence_is_not_silently_accepted():
    source = "ตรวจ Encoder Lot 123"
    payload = _payload(
        source,
        [{"segment_id": 1, "quote": "อนุมัติ Encoder Lot 123"}],
    )

    analysis = _parse(payload, source)

    assert analysis.status == "review_required"
    assert analysis.review_reason == "OverviewEvidenceInvalid"
