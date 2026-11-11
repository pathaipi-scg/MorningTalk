"""UTF-8 transcript and Unicode PDF export."""

from pathlib import Path
import os
import tempfile

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from xml.sax.saxutils import escape

from app.file_manager import meeting_datetime


def _thai_font() -> str:
    candidates = [
        Path(__file__).resolve().parent.parent / "fonts" / "NotoSansThai-Regular.ttf",
        Path(r"C:\Windows\Fonts\tahoma.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            if "MorningTalkThai" not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont("MorningTalkThai", str(candidate)))
            return "MorningTalkThai"
    raise FileNotFoundError(
        "Thai Unicode font not found. Add NotoSansThai-Regular.ttf to the project fonts directory."
    )


def write_pdf(path: Path, source: Path, plant: str, transcript: str, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"PDF already exists: {path.name}")
    font = _thai_font()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp.pdf", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    meeting_time = meeting_datetime(source.name)
    details = [
        f"Meeting: {escape(source.name)}",
        f"Date and start time: {meeting_time:%Y-%m-%d %H:%M}" if meeting_time else "Date and start time: unavailable",
        f"Plant/site: {escape(plant)}",
    ]
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "ThaiBody", parent=styles["BodyText"], fontName=font, leading=17, spaceAfter=8, splitLongWords=1
    )
    heading = ParagraphStyle("ThaiHeading", parent=styles["Heading2"], fontName=font)
    story = [Paragraph(item, body) for item in details]
    story.extend([Spacer(1, 12), Paragraph("Full transcript", heading), Spacer(1, 8)])
    for paragraph in transcript.splitlines():
        story.append(Paragraph(escape(paragraph) or "&#160;", body))

    def add_page_number(canvas, document) -> None:
        canvas.saveState()
        canvas.setFont(font, 9)
        canvas.drawCentredString(A4[0] / 2, 20, str(document.page))
        canvas.restoreState()

    try:
        SimpleDocTemplate(str(temporary), pagesize=A4, title=f"Transcript: {source.name}").build(
            story, onFirstPage=add_page_number, onLaterPages=add_page_number
        )
        if overwrite:
            os.replace(temporary, path)
        else:
            temporary.rename(path)
    finally:
        temporary.unlink(missing_ok=True)
