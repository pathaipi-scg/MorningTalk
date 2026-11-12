from pathlib import Path


COMPARISON_REPORT = (
    Path(__file__).parents[1] / "references" / "phase9_sara_comparison.md"
)


def test_sara_comparison_report_does_not_claim_unverified_similarity():
    report = COMPARISON_REPORT.read_text(encoding="utf-8")

    assert "no validated MorningTalk summary" in report
    assert "No SARA similarity" in report
    assert "OverviewEvidenceInvalid" in report
    assert "phase9-summary-preview" in report
    assert "does not exist and is intentionally not generated" in report
    assert "The faithful cleaned transcript, not either reference PDF, remains the factual authority." in report
    for dimension in (
        "Meeting overview",
        "Topic classification",
        "Important discussion points",
        "Action items",
        "Responsibilities",
        "Dependencies",
        "Schedules and deadlines",
        "Risks and unresolved issues",
        "Thai readability",
        "Factual accuracy / unsupported claims",
        "PDF layout and page count",
    ):
        assert f"| {dimension} |" in report
