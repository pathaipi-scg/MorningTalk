"""UTF-8 transcript and Unicode PDF export."""

from pathlib import Path
import os
import re
import tempfile

from pypdf import PdfReader, apply_configuration
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer
from xml.sax.saxutils import escape

from app.file_manager import meeting_datetime
from app.meeting_analysis import AnalysisItem, EvidenceReference, MeetingAnalysis
from app.transcript import TranscriptDocument


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", "", value)


def _extract_pdf_text(path: Path) -> tuple[int, str]:
    with apply_configuration(disable_legacy_handling=True):
        reader = PdfReader(str(path))
        page_texts = []
        for page_number, page in enumerate(reader.pages, 1):
            lines = (page.extract_text() or "").splitlines()
            if lines and lines[0].strip() == str(page_number):
                lines = lines[1:]
            page_texts.append("\n".join(lines))
        return len(reader.pages), "\n".join(page_texts)


def _validate_pdf(path: Path, expected_texts: tuple[str, ...] = ()) -> None:
    page_count, extracted_text = _extract_pdf_text(path)
    extracted = _normalized_text(extracted_text)
    position = 0
    for expected in expected_texts:
        normalized = _normalized_text(expected)
        if not normalized:
            continue
        position = extracted.find(normalized, position)
        if position < 0:
            raise ValueError("Generated PDF does not contain the complete transcript.")
        position += len(normalized)
    if not page_count:
        raise ValueError("Generated PDF contains no pages.")


def _publish_pdf(temporary: Path, path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"PDF already exists: {path.name}")
    if overwrite:
        os.replace(temporary, path)
    else:
        temporary.rename(path)


def _thai_font() -> str:
    font_dir = Path(__file__).resolve().parent.parent / "fonts"
    candidates = [
        (font_dir / "Sarabun-Regular.ttf", font_dir / "Sarabun-Bold.ttf"),
        (Path(r"C:\Windows\Fonts\Sarabun-Regular.ttf"), Path(r"C:\Windows\Fonts\Sarabun-Bold.ttf")),
        (font_dir / "NotoSansThai-Regular.ttf", font_dir / "NotoSansThai-Bold.ttf"),
        (Path(r"C:\Windows\Fonts\tahoma.ttf"), Path(r"C:\Windows\Fonts\tahomabd.ttf")),
    ]
    for regular_path, bold_path in candidates:
        if regular_path.is_file():
            regular_name = "MorningTalkThai"
            bold_name = "MorningTalkThai-Bold"
            registered = pdfmetrics.getRegisteredFontNames()
            if regular_name not in registered:
                pdfmetrics.registerFont(TTFont(regular_name, str(regular_path)))
            if bold_path.is_file() and bold_name not in registered:
                pdfmetrics.registerFont(TTFont(bold_name, str(bold_path)))
            pdfmetrics.registerFontFamily(
                regular_name,
                normal=regular_name,
                bold=bold_name if bold_path.is_file() else regular_name,
                italic=regular_name,
                boldItalic=bold_name if bold_path.is_file() else regular_name,
            )
            return regular_name
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
        _validate_pdf(temporary, tuple(transcript.splitlines()))
        _publish_pdf(temporary, path, overwrite=overwrite)
    finally:
        temporary.unlink(missing_ok=True)


def write_transcript_pdf(
    path: Path,
    source: Path,
    plant: str,
    document: TranscriptDocument,
    *,
    overwrite: bool = False,
    fixture_preview: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"PDF already exists: {path.name}")
    font = _thai_font()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp.pdf", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    meeting_time = meeting_datetime(source.name)
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "ThaiTranscriptBody",
        parent=styles["BodyText"],
        fontName=font,
        leading=19,
        spaceAfter=10,
        splitLongWords=1,
    )
    title = ParagraphStyle(
        "ThaiTranscriptTitle",
        parent=styles["Title"],
        fontName=font,
        leading=24,
        spaceAfter=12,
    )
    speaker_style = ParagraphStyle(
        "ThaiTranscriptSpeaker",
        parent=styles["BodyText"],
        fontName=font,
        leading=15,
        spaceBefore=5,
        spaceAfter=3,
    )
    story = [
        Paragraph("Morning Meeting - Full Transcript", title),
        Paragraph(f"Recording: {escape(source.name)}", body),
        Paragraph(
            f"Meeting date/time: {meeting_time:%Y-%m-%d %H:%M}"
            if meeting_time else "Meeting date/time: unavailable",
            body,
        ),
        Paragraph(f"Plant/site: {escape(plant) if plant else 'unavailable'}", body),
    ]
    if fixture_preview:
        story.append(Paragraph(
            "DEMONSTRATION PREVIEW - FIXTURE TEXT, NOT AN ACTUAL TRANSCRIPTION",
            speaker_style,
        ))
    if document.cleanup_status != "completed":
        story.append(Paragraph(
            "Transcript cleanup requires review; raw speech is shown where cleanup was not validated.",
            speaker_style,
        ))
    for segment_index, segment in enumerate(document.segments, 1):
        story.append(Spacer(1, 5))
        if segment.start_seconds is None:
            time_label = "[time unavailable]"
        else:
            seconds = int(segment.start_seconds)
            time_label = f"[{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}]"
        speaker = f" {escape(segment.speaker_id)}" if segment.speaker_id else ""
        story.append(Paragraph(
            f"<b>[segment {segment_index}] {time_label}{speaker}</b>",
            speaker_style,
        ))
        story.append(Paragraph(escape(segment.display_text), body))
        if segment.review_required:
            story.append(Paragraph("REVIEW REQUIRED", speaker_style))

    def add_page_number(canvas, document) -> None:
        canvas.saveState()
        canvas.setFont(font, 9)
        canvas.drawCentredString(A4[0] / 2, 20, str(document.page))
        canvas.restoreState()

    try:
        SimpleDocTemplate(
            str(temporary),
            pagesize=A4,
            title=f"Morning Meeting - Full Transcript: {source.name}",
            leftMargin=54,
            rightMargin=54,
            topMargin=54,
            bottomMargin=42,
        ).build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
        _validate_pdf(
            temporary,
            tuple(segment.display_text for segment in document.segments),
        )
        _publish_pdf(temporary, path, overwrite=overwrite)
    finally:
        temporary.unlink(missing_ok=True)


def write_meeting_summary_pdf(
    path: Path,
    source: Path,
    plant: str,
    analysis: MeetingAnalysis,
    *,
    overwrite: bool = False,
    fixture_preview: bool = False,
    draft: bool = False,
    demo_passages: tuple[EvidenceReference, ...] = (),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"PDF already exists: {path.name}")
    reviewable = analysis.overview is not None
    if overwrite and draft and (
        analysis.status != "validated"
        and not (analysis.status == "review_required" and reviewable)
    ):
        raise ValueError("An existing summary draft requires evidence-validated analysis.")
    font = _thai_font()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp.pdf", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    meeting_time = meeting_datetime(source.name)
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "ThaiSummaryTitle",
        parent=styles["Title"],
        fontName=font,
        fontSize=18,
        leading=25,
        textColor="#263746",
        spaceAfter=8,
    )
    heading = ParagraphStyle(
        "ThaiSummaryHeading",
        parent=styles["Heading2"],
        fontName=font,
        fontSize=13,
        leading=19,
        textColor="#263746",
        spaceBefore=13,
        spaceAfter=4,
        keepWithNext=1,
    )
    body = ParagraphStyle(
        "ThaiSummaryBody",
        parent=styles["BodyText"],
        fontName=font,
        fontSize=11,
        leading=16.5,
        leftIndent=12,
        firstLineIndent=-10,
        bulletFontName=font,
        bulletFontSize=11,
        spaceAfter=6,
        splitLongWords=1,
    )
    metadata = ParagraphStyle(
        "ThaiSummaryMetadata",
        parent=body,
        fontSize=10,
        leading=14,
        leftIndent=0,
        firstLineIndent=0,
        spaceAfter=3,
    )
    story = [
        Paragraph("สรุปการประชุมภาคเช้า", title),
        Paragraph(f"ไฟล์บันทึก: {escape(source.name)}", metadata),
        Paragraph(
            f"วันและเวลาเริ่มประชุม: {meeting_time:%Y-%m-%d %H:%M}"
            if meeting_time else "วันและเวลาเริ่มประชุม: ไม่ระบุ",
            metadata,
        ),
        Paragraph(
            f"โรงงาน/สถานที่: {escape(plant) if plant else 'ไม่ระบุ'}",
            metadata,
        ),
        Spacer(1, 5),
        HRFlowable(width="100%", thickness=1, color="#263746", spaceAfter=6),
    ]
    if fixture_preview:
        story.append(Paragraph(
            "ตัวอย่างจัดหน้าเท่านั้น — ไม่ใช่สรุป AI ที่ผ่านการตรวจสอบ",
            body,
        ))
    if draft:
        story.append(Paragraph(
            "DRAFT - NOT APPROVED / ร่าง - ยังไม่ได้รับอนุมัติ",
            body,
        ))
    if analysis.status != "validated" and not demo_passages:
        story.append(Paragraph(
            "ต้องตรวจทาน: การวิเคราะห์ข้อเท็จจริงยังไม่ผ่านการตรวจสอบ",
            body,
        ))
        if analysis.review_reason:
            story.append(Paragraph(
                f"สาเหตุ: {escape(analysis.review_reason)}",
                body,
            ))

    def add_item(item: AnalysisItem, *, bullet: bool) -> None:
        topic_parts = item.text.split(" — ", maxsplit=1)
        if len(topic_parts) == 2:
            item_text = f"<b>{escape(topic_parts[0])}</b> — {escape(topic_parts[1])}"
        else:
            item_text = escape(item.text)
        if analysis.status != "validated":
            item_text = f"<b>รอตรวจสอบ — </b>{item_text}"
        story.append(
            Paragraph(
                item_text,
                body,
                bulletText="•" if bullet else None,
            )
        )
        if item.owner:
            story.append(
                Paragraph(
                    f"{'รอตรวจสอบ: ' if analysis.status != 'validated' else ''}"
                    f"ผู้รับผิดชอบ: {escape(item.owner)}",
                    metadata,
                )
            )
        if item.deadline:
            story.append(
                Paragraph(
                    f"{'รอตรวจสอบ: ' if analysis.status != 'validated' else ''}"
                    f"กำหนดเวลา: {escape(item.deadline)}",
                    metadata,
                )
            )
        if item.evidence:
            references = sorted({reference.segment_id for reference in item.evidence})
            story.append(Paragraph(
                (
                    "หลักฐาน: ช่วงที่ "
                    if analysis.status == "validated"
                    else "หลักฐานอ้างอิงเพื่อการตรวจทาน: ช่วงที่ "
                )
                + ", ".join(map(str, references)),
                metadata,
            ))

    if demo_passages:
        story.append(Paragraph("ข้อความคัดเลือกเพื่อทดสอบรูปแบบ", heading))
        story.append(HRFlowable(
            width="100%", thickness=0.6, color="#52616B", spaceAfter=6
        ))
        for reference in demo_passages:
            story.append(Paragraph(
                escape(reference.quote),
                body,
                bulletText="•",
            ))
            story.append(Paragraph(
                f"ข้อความต้นทางช่วงที่ {reference.segment_id}",
                metadata,
            ))
    else:
        sections = (
            ("ภาพรวมการประชุม", (analysis.overview,) if analysis.overview else ()),
            ("สรุปแยกตามหัวข้อ", analysis.topics),
            (
                "หมวดหมู่การประชุม",
                analysis.classifications if not analysis.topics else (),
            ),
            ("ประเด็นสำคัญ", analysis.key_issues),
            ("รายการดำเนินการและผู้รับผิดชอบ", analysis.actions),
            ("ความเชื่อมโยงและสิ่งที่ต้องพึ่งพา", analysis.dependencies),
            ("กำหนดการ", analysis.schedules),
            ("ความเสี่ยงและประเด็นที่ยังไม่คลี่คลาย", analysis.risks),
            ("กิจกรรมปิดการประชุม", analysis.closing_activities),
        )
        for label, items in sections:
            if not items:
                continue
            story.append(Paragraph(escape(label), heading))
            story.append(HRFlowable(
                width="100%", thickness=0.6, color="#52616B", spaceAfter=6
            ))
            if label == "ภาพรวมการประชุม":
                add_item(items[0], bullet=False)
            else:
                for item in items:
                    add_item(item, bullet=True)

    def add_page_number(canvas, document) -> None:
        canvas.saveState()
        canvas.setFont(font, 8)
        canvas.setFillColor("#52616B")
        canvas.drawString(54, 24, "MorningTalk | สรุปการประชุม")
        canvas.drawRightString(A4[0] - 54, 24, str(document.page))
        canvas.restoreState()

    try:
        SimpleDocTemplate(
            str(temporary),
            pagesize=A4,
            title=f"Morning Meeting Summary: {source.name}",
            leftMargin=56,
            rightMargin=56,
            topMargin=58,
            bottomMargin=42,
        ).build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
        _validate_pdf(temporary)
        if draft:
            _page_count, draft_text = _extract_pdf_text(temporary)
            if "DRAFT - NOT APPROVED" not in draft_text.upper():
                raise ValueError("Generated summary draft is missing its DRAFT label.")
        _publish_pdf(temporary, path, overwrite=overwrite)
    finally:
        temporary.unlink(missing_ok=True)
