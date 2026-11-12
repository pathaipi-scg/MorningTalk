from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
MIGRATION_002 = ROOT / "migrations" / "002_phase9_transcripts.sql"
MIGRATION_003 = ROOT / "migrations" / "003_meeting_summary_outputs.sql"


def test_phase9_migrations_are_guarded_transactional_and_additive():
    for migration in (MIGRATION_002, MIGRATION_003):
        sql = migration.read_text(encoding="utf-8")
        normalized = re.sub(r"\s+", " ", sql).casefold()

        assert "if db_name() <> n'morningtalk'" in normalized
        assert "try_convert(int, serverproperty('productmajorversion')) <> 17" in normalized
        assert "begin try begin transaction;" in normalized
        assert "commit transaction;" in normalized
        assert "if xact_state() <> 0 rollback transaction;" in normalized
        assert "throw;" in normalized
        assert "alter table dbo.audiofile add " in normalized
        assert not re.search(r"\b(drop|delete|truncate|update|insert)\b", normalized)


def test_phase9_migration_columns_match_repository_contract():
    expected_columns = {
        "002_phase9_transcripts.sql": (
            "RawTranscript",
            "CleanedTranscript",
            "TranscriptSegmentsJson",
            "DiarizationStatus",
            "CleanupStatus",
            "ReviewReason",
            "InputTokens",
            "OutputTokens",
        ),
        "003_meeting_summary_outputs.sql": (
            "MeetingAnalysisJson",
            "MeetingAnalysisStatus",
            "SummaryPdfPath",
            "SummaryPdfStatus",
            "FullTranscriptPdfPath",
            "FullTranscriptPdfStatus",
        ),
    }

    for filename, columns in expected_columns.items():
        sql = (ROOT / "migrations" / filename).read_text(encoding="utf-8")
        for column in columns:
            assert f"COL_LENGTH(N'dbo.AudioFile', N'{column}')" in sql
            assert re.search(
                rf"ALTER TABLE dbo\.AudioFile ADD {column}\s+"
                rf"(?:nvarchar(?:\(max\)|\(\d+\))?|varchar\(\d+\)|int)\s+NULL",
                sql,
                re.IGNORECASE,
            )
