import json
from pathlib import Path
from unittest.mock import Mock

from app.factory_context import (
    load_factory_vocabulary,
    normalize_technical_terms,
    normalize_transcript_segments,
)
from app.transcript import TranscriptSegment
from app.transcript_cleanup import TranscriptCleanup


def azure_config():
    from app.config import AzureConfig

    return AzureConfig(
        text_endpoint="https://text.example.test",
        text_api_key="test-key",
        text_deployment="test-deployment",
        transcription_endpoint="https://transcription.example.test",
        transcription_api_key="test-key",
        transcription_deployment="gpt-4o-transcribe",
        transcription_api_version="2025-03-01-preview",
    )


def test_vocabulary_master_records_confirmation_and_unverified_spelling():
    master = load_factory_vocabulary()
    twinloader = next(
        entry for entry in master.entries if entry.master_id == "equipment-twinloader"
    )

    assert twinloader.canonical_english == "Twinloader"
    assert twinloader.thai_phonetic_variants == ("ทวินโอเดอร์", "ทวินโดเตอร์")
    assert twinloader.verified_alternative_spellings == ()
    assert twinloader.candidate_alternative_spellings == ("Twin Loader",)
    assert twinloader.confirmation_status == "USER_CONFIRMED"
    assert twinloader.recording_references == ("meeting-20261103-0813",)
    assert twinloader.chunk_references == ()
    assert twinloader.occurrence_count == 1
    assert master.confirmed_terms == ("Twinloader",)


def test_confirmed_alias_normalizes_only_with_equipment_context():
    raw = "ตรวจเครื่อง ทวินโอเดอร์ แล้วเดินเครื่องต่อ"

    result = normalize_technical_terms(raw, recording="meeting-test")

    assert raw == "ตรวจเครื่อง ทวินโอเดอร์ แล้วเดินเครื่องต่อ"
    assert result.normalized_text == "ตรวจเครื่อง Twinloader แล้วเดินเครื่องต่อ"
    assert len(result.replacements) == 1
    assert result.replacements[0]["vocabulary_master_id"] == "equipment-twinloader"
    assert result.replacements[0]["confirmation_status"] == "USER_CONFIRMED"
    assert result.review_findings == ()


def test_multiple_aliases_keep_evidence_anchored_to_each_source_occurrence():
    unrelated = " สภาพทั่วไป" * 40
    raw = f"ตรวจเครื่อง ทวินโอเดอร์{unrelated} ข้อความ ทวินโดเตอร์"

    result = normalize_technical_terms(raw, recording="meeting-test")

    assert result.normalized_text == (
        f"ตรวจเครื่อง Twinloader{unrelated} ข้อความ ทวินโดเตอร์"
    )
    assert len(result.replacements) == 1
    assert "ทวินโอเดอร์" in result.replacements[0]["evidence_context"]
    assert "ทวินโดเตอร์" not in result.replacements[0]["evidence_context"]
    assert len(result.review_findings) == 1
    assert "ทวินโดเตอร์" in result.review_findings[0]["evidence_context"]
    assert "ทวินโอเดอร์" not in result.review_findings[0]["evidence_context"]


def test_alias_without_equipment_context_is_preserved_and_flagged():
    raw = "ทวินโอเดอร์ วันนี้"

    result = normalize_technical_terms(raw, recording="meeting-test")

    assert result.normalized_text == raw
    assert result.replacements == ()
    assert result.review_findings[0]["reason"] == (
        "ConfirmedAliasWithoutEquipmentContext"
    )


def test_unknown_phrases_and_personal_names_are_never_rewritten():
    raw = "คุณพอครับ สีแปลนแผ่นข้าว"

    result = normalize_technical_terms(raw, recording="meeting-test")

    assert result.normalized_text == raw
    assert result.replacements == ()
    assert {item["original_phrase"] for item in result.review_findings} == {
        "พอครับ",
        "สีแปลนแผ่นข้าว",
    }


def test_segment_normalization_records_audio_time_and_preserves_raw():
    segments = (
        TranscriptSegment(1.0, 3.0, "Speaker A", "ตรวจเครื่องชุดนี้"),
        TranscriptSegment(3.0, 5.0, "Speaker B", "ทวินโดเตอร์มีเสียงดัง"),
    )

    normalized = normalize_transcript_segments(
        segments,
        recording="meeting-test",
    )

    assert normalized.raw_transcript == "ตรวจเครื่องชุดนี้\nทวินโดเตอร์มีเสียงดัง"
    assert normalized.normalized_transcript == (
        "ตรวจเครื่องชุดนี้\nTwinloaderมีเสียงดัง"
    )
    assert normalized.replacements[0].chunk_reference == "3.000-5.000s"
    assert normalized.replacements[0].recording == "meeting-test"
    assert normalized.review_findings == ()


def test_cleanup_cannot_revert_canonical_term_to_confirmed_bad_alias():
    source = (TranscriptSegment(1.0, 3.0, "Speaker A", "ตรวจ Twinloader"),)
    response = Mock(status_code=200)
    response.json.return_value = {
        "choices": [{
            "message": {
                "content": json.dumps(
                    {
                        "segments": [
                            {"index": 0, "text": "ตรวจ ทวินโอเดอร์"}
                        ],
                        "review_findings": [],
                    },
                    ensure_ascii=False,
                )
            }
        }]
    }
    session = Mock()
    session.post.return_value = response

    document = TranscriptCleanup(
        azure_config(),
        api_version="2025-04-01-preview",
        session=session,
    ).clean(source)

    assert document.cleanup_status == "review_required"
    assert document.review_reason == "ValidationFailed:ConfirmedTerminologyChanged"
    assert document.cleaned_transcript == "ตรวจ Twinloader"
