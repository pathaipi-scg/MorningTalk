from pathlib import Path

import pytest

from app.file_manager import resolve_meeting_output_paths


def test_meeting_outputs_resolve_plant_and_month_from_source(tmp_path: Path):
    source = (
        tmp_path / "Record File" / "CB" / "Nov26"
        / "meeting-20261107-0817.mp3"
    )
    paths = resolve_meeting_output_paths(
        source,
        transcript_root=tmp_path / "Transcript",
    )

    assert paths.directory == tmp_path / "Transcript" / "CB" / "Nov26"
    assert paths.transcript_text == paths.directory / "meeting-20261107-0817.txt"
    assert (
        paths.full_transcript_pdf
        == paths.directory / "meeting-20261107-0817_Full.pdf"
    )
    assert (
        paths.summary_draft_pdf
        == paths.directory / "meeting-20261107-0817_Summary_Draft.pdf"
    )
    assert (
        paths.approved_summary_pdf
        == paths.directory / "meeting-20261107-0817_Summary.pdf"
    )


@pytest.mark.parametrize(
    "source",
    [
        Path("D:/Record File/CB/November26/meeting.mp3"),
        Path("D:/Record File/CB/Nov26/meeting.wav"),
        Path("D:/Record File/CB/meeting.mp3"),
    ],
)
def test_meeting_output_resolver_rejects_noncanonical_source_paths(source: Path):
    with pytest.raises(ValueError):
        resolve_meeting_output_paths(source)
