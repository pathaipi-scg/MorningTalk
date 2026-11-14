"""Offline inventory and comparison of MorningTalk artifacts with SARA PDFs."""

import argparse
import csv
from dataclasses import dataclass
import json
import re
import subprocess
from pathlib import Path
from typing import Callable, Iterable

from pypdf import PdfReader, apply_configuration

from app.factory_context import load_factory_vocabulary
from app.workspace_paths import (
    DEFAULT_BENCHMARK_REPORT_ROOT,
    WORKSPACE_ROOT,
)

DEFAULT_RECORDING_ROOT = Path(r"D:\KM\Vault\MorningTalk\Record File")
DEFAULT_TRANSCRIPT_ROOT = Path(r"D:\KM\Vault\MorningTalk\Transcript")
DEFAULT_SARA_ROOT = Path(__file__).resolve().parent.parent / "references" / "SaraAI"
DEFAULT_REPORT_ROOT = DEFAULT_BENCHMARK_REPORT_ROOT
DEFAULT_PRIVATE_ROOT = WORKSPACE_ROOT

_MONTH = re.compile(r"^[A-Za-z]{3}\d{2}$")
_DATE_TIME = re.compile(r"(?<!\d)(\d{8}-\d{4})(?!\d)")
_FACT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("equipment", (
        "twin loader", "stacking", "stack 1", "stack 2", "encoder", "press",
        "curing", "depallet", "packing", "wet line", "dry line", "rack",
        "pallet", "crane", "conveyor", "silo", "pump", "mechanical seal",
        "packing seal", "เครื่องจักร", "แร็ค", "เครน", "สายพาน", "ปั๊ม",
    )),
    ("production_problem", (
        "ปัญหา", "ขัดข้อง", "เสีย", "ติดขัด", "หยุด", "แตก", "หล่น",
        "breakdown", "reject",
    )),
    ("quality_issue", (
        "คุณภาพ", "strength", "thickness", "moisture", "ความแข็งแรง",
        "ความหนา", "ความชื้น", "ไม่ผ่าน", "เคลม",
    )),
    ("maintenance", (
        "ซ่อม", "บำรุง", "maintenance", "preventive", "อะไหล่",
        "เปลี่ยน", "ตรวจสอบ", "จาระบี",
    )),
    ("safety", (
        "ความปลอดภัย", "safety", "body harness", "work permit",
        "working at height", "อุบัติเหตุ", "ทำงานที่สูง", "permit",
    )),
    ("decision_action", (
        "ต้อง", "ให้", "จะ", "มอบหมาย", "ตรวจ", "ปรับ", "สั่ง", "ติดตาม",
        "ดำเนินการ", "แผน", "รับผิดชอบ", "action",
    )),
    ("responsible_person_or_deadline", (
        "ผู้รับผิดชอบ", "รับผิดชอบ", "มอบหมาย", "กำหนด", "ภายใน",
        "deadline", "due date",
    )),
    ("responsible_person", (
        "ผู้รับผิดชอบ", "มอบหมายให้", "assigned to", "owner:",
    )),
    ("explicit_deadline", (
        "ภายในวันที่", "กำหนดส่ง", "กำหนดเสร็จ", "ภายใน", "deadline",
        "due date", "ก่อนวันที่",
    )),
    ("reference_quality_notice", (
        "accuracy score", "ต่ำกว่า 50%", "50%", "โปรดตรวจสอบความถูกต้อง",
    )),
)


@dataclass(frozen=True)
class MeetingInventory:
    plant: str
    month: str
    recording_filename: str
    mp3_path: Path
    duration_seconds: float | None
    sara_pdf_path: Path | None
    sara_match_status: str
    transcript_txt_path: Path | None
    full_transcript_pdf_path: Path | None
    summary_draft_pdf_path: Path | None
    approved_summary_pdf_path: Path | None
    legacy_pdf_paths: tuple[Path, ...]
    source_adjacent_pdf_paths: tuple[Path, ...]
    phase9_json_paths: tuple[Path, ...]
    phase10_json_paths: tuple[Path, ...]
    missing_files: tuple[str, ...]
    duplicate_or_ambiguous_matches: tuple[str, ...]


@dataclass(frozen=True)
class PdfPage:
    page_number: int
    text: str


@dataclass(frozen=True)
class ReferenceFact:
    category: str
    page_number: int
    wording: str


@dataclass(frozen=True)
class InventoryResult:
    meetings: tuple[MeetingInventory, ...]
    unmatched_sara_pdfs: tuple[Path, ...]
    ambiguous_sara_pdfs: tuple[Path, ...]


def discover_files(root: Path, extensions: Iterable[str]) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Inventory root does not exist: {root}")
    accepted = {extension.casefold() for extension in extensions}
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in accepted
    )


def _plant_month(path: Path, root: Path) -> tuple[str, str]:
    relative_parts = path.relative_to(root).parts
    for index, part in enumerate(relative_parts[:-1]):
        if _MONTH.fullmatch(part):
            plant = relative_parts[index - 1] if index > 0 else ""
            return plant, part
        if re.fullmatch(r"\d{4}", part) and index + 1 < len(relative_parts) - 1:
            month_name = relative_parts[index + 1]
            if month_name[:3].casefold() in {
                "jan", "feb", "mar", "apr", "may", "jun",
                "jul", "aug", "sep", "oct", "nov", "dec",
            }:
                plant = relative_parts[index - 1] if index > 0 else ""
                return plant, f"{month_name[:3]}{part[-2:]}"
    return "", ""


def _matching_edges(
    audios: list[Path],
    sara_pdfs: list[Path],
    recording_root: Path,
    sara_root: Path,
) -> tuple[dict[Path, list[Path]], dict[Path, list[Path]]]:
    audio_edges: dict[Path, list[Path]] = {}
    sara_edges: dict[Path, list[Path]] = {}
    for audio in audios:
        audio_plant, audio_month = _plant_month(audio, recording_root)
        candidates = []
        for pdf in sara_pdfs:
            if pdf.stem != audio.stem:
                continue
            sara_plant, sara_month = _plant_month(pdf, sara_root)
            if sara_plant and sara_plant != audio_plant:
                continue
            if sara_month and sara_month.casefold() != audio_month.casefold():
                continue
            candidates.append(pdf)
        audio_edges[audio] = candidates
        for pdf in candidates:
            sara_edges.setdefault(pdf, []).append(audio)
    return audio_edges, sara_edges


def _probe_duration(path: Path) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        duration = float(result.stdout.strip())
        return duration if duration > 0 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def discover_inventory(
    recording_root: Path,
    transcript_root: Path,
    sara_root: Path,
    *,
    private_root: Path | None = None,
    duration_probe: Callable[[Path], float | None] = _probe_duration,
) -> InventoryResult:
    audios = discover_files(recording_root, {".mp3"})
    sara_pdfs = discover_files(sara_root, {".pdf"})
    transcript_files = discover_files(transcript_root, {".txt", ".pdf"})
    audio_edges, sara_edges = _matching_edges(
        audios, sara_pdfs, recording_root, sara_root
    )

    assigned: dict[Path, Path] = {}
    ambiguous_audios: set[Path] = set()
    ambiguous_sara: set[Path] = set()
    for audio in audios:
        candidates = audio_edges.get(audio, [])
        if len(candidates) == 1 and len(sara_edges[candidates[0]]) == 1:
            assigned[audio] = candidates[0]
        elif candidates:
            ambiguous_audios.add(audio)
            ambiguous_sara.update(candidates)

    transcript_index: dict[tuple[str, str, str, str], list[Path]] = {}
    for path in transcript_files:
        plant, month = _plant_month(path, transcript_root)
        kind, stem = _output_kind(path)
        transcript_index.setdefault(
            (plant, month, stem, kind), []
        ).append(path)

    private_json: dict[tuple[str, str], list[Path]] = {}
    if private_root is not None and private_root.is_dir():
        for path in discover_files(private_root, {".json"}):
            if path.name.endswith(".phase9.json"):
                kind = "phase9"
                stem = path.name.removesuffix(".phase9.json")
            elif path.name.endswith(".phase10-recovered.json"):
                kind = "phase10"
                stem = path.name.removesuffix(".phase10-recovered.json")
            elif path.name.endswith(".phase10.json"):
                kind = "phase10"
                stem = path.name.removesuffix(".phase10.json")
            else:
                continue
            private_json.setdefault((stem, kind), []).append(path)

    meetings = []
    for audio in audios:
        plant, month = _plant_month(audio, recording_root)
        exact = audio.stem
        output = lambda kind: _select_output(
            transcript_index, plant, month, exact, kind
        )
        txt = output("txt")
        full = output("full")
        draft = output("draft")
        approved = output("approved")
        legacy = tuple(
            _select_output(transcript_index, plant, month, exact, "legacy")
        )
        missing = []
        if audio not in assigned:
            missing.append("SARA reference PDF")
        if not isinstance(txt, list) or len(txt) != 1:
            missing.append("MorningTalk TXT")
        if not isinstance(full, list) or len(full) != 1:
            missing.append("MorningTalk Full PDF")
        if not isinstance(draft, list) or len(draft) != 1:
            missing.append("MorningTalk Summary Draft PDF")
        duplicates = []
        if audio in ambiguous_audios:
            duplicates.append("SARA PDF match ambiguous for exact filename stem")
        for kind, values in (
            ("TXT", txt),
            ("Full PDF", full),
            ("Summary Draft", draft),
            ("Approved Summary", approved),
        ):
            if isinstance(values, list) and len(values) > 1:
                duplicates.append(f"Multiple MorningTalk {kind} matches")
        adjacent_pdfs = tuple(sorted(
            path
            for path in audio.parent.glob("*.pdf")
            if path.is_file()
            and (path.stem == exact or path.stem.startswith(f"{exact}_"))
        ))
        meetings.append(MeetingInventory(
            plant=plant,
            month=month,
            recording_filename=audio.name,
            mp3_path=audio,
            duration_seconds=duration_probe(audio),
            sara_pdf_path=assigned.get(audio),
            sara_match_status=(
                "matched" if audio in assigned
                else "ambiguous" if audio in ambiguous_audios
                else "missing"
            ),
            transcript_txt_path=txt[0] if isinstance(txt, list) and len(txt) == 1 else None,
            full_transcript_pdf_path=full[0] if isinstance(full, list) and len(full) == 1 else None,
            summary_draft_pdf_path=draft[0] if isinstance(draft, list) and len(draft) == 1 else None,
            approved_summary_pdf_path=(
                approved[0] if isinstance(approved, list) and len(approved) == 1
                else None
            ),
            legacy_pdf_paths=tuple(legacy) if isinstance(legacy, list) else (),
            source_adjacent_pdf_paths=adjacent_pdfs,
            phase9_json_paths=tuple(sorted(private_json.get((exact, "phase9"), []))),
            phase10_json_paths=tuple(sorted(private_json.get((exact, "phase10"), []))),
            missing_files=tuple(missing),
            duplicate_or_ambiguous_matches=tuple(duplicates),
        ))
    unmatched = tuple(sorted(set(sara_pdfs) - set(assigned.values()) - ambiguous_sara))
    ambiguous = tuple(sorted(ambiguous_sara))
    return InventoryResult(tuple(meetings), unmatched, ambiguous)


def _output_kind(path: Path) -> tuple[str, str]:
    if path.suffix.casefold() == ".txt":
        return "txt", path.stem
    stem = path.stem
    for suffix, kind in (
        ("_Summary_Draft", "draft"),
        ("_Full", "full"),
        ("_Summary", "approved"),
    ):
        if stem.endswith(suffix):
            return kind, stem[:-len(suffix)]
    return "legacy", stem


def _select_output(
    index: dict[tuple[str, str, str, str], list[Path]],
    plant: str,
    month: str,
    stem: str,
    kind: str,
) -> list[Path]:
    return sorted(index.get((plant, month, stem, kind), []))


def extract_pdf_pages(path: Path) -> tuple[PdfPage, ...]:
    with apply_configuration(disable_legacy_handling=True):
        reader = PdfReader(str(path))
        return tuple(
            PdfPage(number, page.extract_text() or "")
            for number, page in enumerate(reader.pages, 1)
        )


def extract_reference_facts(pages: tuple[PdfPage, ...]) -> tuple[ReferenceFact, ...]:
    facts: list[ReferenceFact] = []
    seen: set[tuple[str, int, str]] = set()
    for page in pages:
        lines = [line.strip() for line in page.text.splitlines() if line.strip()]
        for line_index, line in enumerate(lines):
            categories = [
                category
                for category, terms in _FACT_PATTERNS
                if any(term.casefold() in line.casefold() for term in terms)
            ]
            if re.search(r"\d", line):
                categories.append("quantities_and_measurements")
            if page.page_number == 1 and line_index < 6:
                categories.append("meeting_overview")
            for category in dict.fromkeys(categories):
                key = (category, page.page_number, line)
                if key not in seen:
                    seen.add(key)
                    facts.append(ReferenceFact(category, page.page_number, line))
    return tuple(facts)


def classify_discrepancy(
    *,
    evidence_kind: str,
    source_wording: str,
    raw_transcript: str | None,
    cleaned_transcript: str | None,
    summary_text: str | None,
    suspicious_phrase: bool = False,
) -> str:
    if suspicious_phrase:
        return "Suspected transcription error"
    if raw_transcript is None:
        return "Insufficient evidence"
    source_normalized = _normalize(source_wording)
    raw_normalized = _normalize(raw_transcript)
    cleaned_normalized = _normalize(cleaned_transcript or "")
    summary_normalized = _normalize(summary_text or "")
    if evidence_kind == "morningtalk_summary":
        if source_normalized and source_normalized in raw_normalized:
            return ""
        return "Insufficient evidence"
    if source_normalized and source_normalized in raw_normalized:
        if summary_text is not None and source_normalized not in summary_normalized:
            return "Summary omission"
        return ""
    if cleaned_transcript is not None and source_normalized in cleaned_normalized:
        return "Cleanup error"
    return "Insufficient evidence"


def _normalize(text: str) -> str:
    return "".join(character.casefold() for character in text if character.isalnum())


def _read_text(path: Path | None) -> str | None:
    return path.read_text(encoding="utf-8-sig") if path and path.is_file() else None


def _json_payloads(paths: tuple[Path, ...]) -> list[tuple[Path, dict]]:
    result = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            result.append((path, payload))
    return result


def _phase9_transcripts(
    paths: tuple[Path, ...],
) -> list[tuple[Path, str | None, str | None, dict]]:
    results = []
    for path, payload in _json_payloads(paths):
        raw = payload.get("raw_transcript")
        cleaned = payload.get("cleaned_transcript")
        if isinstance(raw, str) or isinstance(cleaned, str):
            results.append((path, raw if isinstance(raw, str) else None,
                            cleaned if isinstance(cleaned, str) else None, payload))
    return results


def _analysis_claims(
    payload: dict,
    *,
    source_segments: object = None,
) -> list[tuple[str, str, bool]]:
    analysis = payload.get("meeting_analysis")
    if not isinstance(analysis, dict):
        return []
    segments = source_segments if isinstance(source_segments, list) else []
    claims: list[tuple[str, str, bool]] = []
    for section, values in analysis.items():
        if section in {"status", "review_reason", "input_tokens", "output_tokens"}:
            continue
        items = [values] if isinstance(values, dict) else values
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                continue
            evidence = item.get("evidence")
            refs = evidence if isinstance(evidence, list) else []
            supported = bool(refs) and all(
                _evidence_resolves(reference, segments) for reference in refs
            )
            claims.append((section, item["text"], supported))
    return claims


def _evidence_resolves(reference: object, segments: list[object]) -> bool:
    if not isinstance(reference, dict):
        return False
    segment_id = reference.get("segment_id")
    if (
        isinstance(segment_id, bool)
        or not isinstance(segment_id, int)
        or not 1 <= segment_id <= len(segments)
    ):
        return False
    segment = segments[segment_id - 1]
    if not isinstance(segment, dict):
        return False
    source_text = segment.get("cleaned_text") or segment.get("raw_text")
    if not isinstance(source_text, str):
        return False
    start = reference.get("start_char")
    end = reference.get("end_char")
    if (
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(end, int)
        and not isinstance(end, bool)
        and 0 <= start < end <= len(source_text)
    ):
        return bool(source_text[start:end].strip())
    quote = reference.get("quote")
    return isinstance(quote, str) and bool(quote.strip()) and quote in source_text


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def generate_benchmark_reports(
    result: InventoryResult,
    report_root: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    report_root.mkdir(parents=True, exist_ok=True)
    inventory_rows: list[dict[str, object]] = []
    discrepancy_rows: list[dict[str, object]] = []
    reference_extract_rows: list[dict[str, object]] = []
    summary_lines = [
        "# Offline SARA AI benchmark",
        "",
        "SARA PDFs are comparison references, not verified ground truth. "
        "This report performs exact-stem matching and offline text comparison only; "
        "it does not calculate WER or verify speech against audio.",
        "",
        "## Inventory totals",
        "",
        f"- MP3 recordings: {len(result.meetings)}",
        f"- SARA AI PDFs: {len(result.unmatched_sara_pdfs) + len(result.ambiguous_sara_pdfs) + sum(m.sara_pdf_path is not None for m in result.meetings)}",
        f"- Exact matched pairs: {sum(m.sara_pdf_path is not None for m in result.meetings)}",
        f"- Missing SARA references: {sum(m.sara_match_status == 'missing' for m in result.meetings)}",
        f"- Ambiguous SARA matches: {len(result.ambiguous_sara_pdfs)} PDF(s); "
        f"{sum(m.sara_match_status == 'ambiguous' for m in result.meetings)} recording(s)",
        "",
        "## Unmatched SARA AI PDFs",
        "",
    ]
    for path in result.unmatched_sara_pdfs:
        summary_lines.append(f"- `{path}`")
    if not result.unmatched_sara_pdfs:
        summary_lines.append("- None")
    summary_lines.extend(["", "## Ambiguous SARA AI PDFs", ""])
    for path in result.ambiguous_sara_pdfs:
        summary_lines.append(f"- `{path}` (exact-stem match could not be disambiguated)")
    if not result.ambiguous_sara_pdfs:
        summary_lines.append("- None")
    summary_lines.extend(["", "## Per-meeting inventory and comparison", ""])

    vocabulary = load_factory_vocabulary()
    pdf_pages_total = 0
    pdf_pages_with_text = 0
    source_adjacent_pdf_paths: set[Path] = set()
    for meeting in result.meetings:
        source_adjacent_pdf_paths.update(meeting.source_adjacent_pdf_paths)
        transcript_txt = _read_text(meeting.transcript_txt_path)
        raw_text = transcript_txt
        cleaned_text = raw_text
        phase9 = _phase9_transcripts(meeting.phase9_json_paths)
        if phase9:
            phase9.sort(
                key=lambda item: (
                    bool(transcript_txt is not None and item[2] == transcript_txt),
                    item[3].get("cleanup_status") == "completed",
                    item[0].stat().st_mtime,
                ),
                reverse=True,
            )
            _, phase9_raw, phase9_cleaned, phase9_payload = phase9[0]
            raw_text = phase9_raw or raw_text
            cleaned_text = phase9_cleaned or cleaned_text
            phase9_txt_exact_match = bool(
                transcript_txt is not None and phase9_cleaned == transcript_txt
            )
        else:
            phase9_payload = {}
            phase9_txt_exact_match = False
        summary_path = meeting.summary_draft_pdf_path
        legacy_summary = None
        if summary_path is None and meeting.legacy_pdf_paths:
            legacy_summary = meeting.legacy_pdf_paths[0]
            summary_path = legacy_summary
        summary_pages = extract_pdf_pages(summary_path) if summary_path else ()
        summary_text = "\n".join(page.text for page in summary_pages) if summary_pages else None
        phase10_payloads = _json_payloads(meeting.phase10_json_paths)
        phase10_counts = []
        source_segments = phase9_payload.get("segments", [])
        for json_path, payload in phase10_payloads:
            analysis = payload.get("meeting_analysis", {})
            if isinstance(analysis, dict):
                claims = _analysis_claims(
                    payload, source_segments=source_segments
                )
                phase10_counts.append(
                    (json_path, analysis.get("status", "unknown"),
                     analysis.get("review_reason", ""),
                     len(claims),
                     sum(supported for _section, _claim, supported in claims))
                )

        pages = extract_pdf_pages(meeting.sara_pdf_path) if meeting.sara_pdf_path else ()
        pdf_pages_total += len(pages)
        pdf_pages_with_text += sum(bool(page.text.strip()) for page in pages)
        facts = extract_reference_facts(pages)
        fact_categories: dict[str, int] = {}
        for fact in facts:
            fact_categories[fact.category] = fact_categories.get(fact.category, 0) + 1
            reference_extract_rows.append({
                "plant": meeting.plant,
                "month": meeting.month,
                "meeting": meeting.recording_filename,
                "reference_pdf": str(meeting.sara_pdf_path or ""),
                "page": fact.page_number,
                "category": fact.category,
                "original_wording": fact.wording,
            })
        output_missing = "; ".join(meeting.missing_files)
        inventory_rows.append({
            "plant": meeting.plant,
            "month": meeting.month,
            "recording_filename": meeting.recording_filename,
            "mp3_path": str(meeting.mp3_path),
            "duration_seconds": meeting.duration_seconds,
            "sara_pdf_path": str(meeting.sara_pdf_path) if meeting.sara_pdf_path else "",
            "sara_match_status": meeting.sara_match_status,
            "transcript_txt_path": str(meeting.transcript_txt_path) if meeting.transcript_txt_path else "",
            "full_transcript_pdf_path": str(meeting.full_transcript_pdf_path) if meeting.full_transcript_pdf_path else "",
            "summary_draft_pdf_path": str(meeting.summary_draft_pdf_path) if meeting.summary_draft_pdf_path else "",
            "approved_summary_pdf_path": str(meeting.approved_summary_pdf_path) if meeting.approved_summary_pdf_path else "",
            "legacy_pdf_paths": "; ".join(map(str, meeting.legacy_pdf_paths)),
            "source_adjacent_pdf_paths": "; ".join(map(str, meeting.source_adjacent_pdf_paths)),
            "missing_files": output_missing,
            "duplicate_or_ambiguous_matches": "; ".join(meeting.duplicate_or_ambiguous_matches),
            "phase9_json_paths": "; ".join(map(str, meeting.phase9_json_paths)),
            "phase10_json_paths": "; ".join(map(str, meeting.phase10_json_paths)),
            "phase9_selected_status": phase9_payload.get("cleanup_status", ""),
            "phase9_raw_characters": len(phase9_payload.get("raw_transcript", "")),
            "phase9_cleaned_characters": len(phase9_payload.get("cleaned_transcript", "")),
            "phase9_cleaned_text_matches_transcript_txt": phase9_txt_exact_match,
        })
        summary_lines.extend([
            f"### {meeting.plant}/{meeting.month}/{meeting.recording_filename}",
            "",
            f"- MP3: `{meeting.mp3_path}`"
            + (f" ({meeting.duration_seconds:.2f} seconds)" if meeting.duration_seconds else " (duration unavailable)"),
            f"- SARA reference: `{meeting.sara_pdf_path}` ({len(pages)} pages; "
            f"{sum(bool(page.text.strip()) for page in pages)}/{len(pages)} pages with text)"
            if meeting.sara_pdf_path else f"- SARA reference: {meeting.sara_match_status}",
            f"- MorningTalk TXT: `{meeting.transcript_txt_path}`"
            if meeting.transcript_txt_path else "- MorningTalk TXT: missing",
            f"- MorningTalk Full PDF: `{meeting.full_transcript_pdf_path}`"
            if meeting.full_transcript_pdf_path else "- MorningTalk Full PDF: missing",
            f"- Summary Draft: `{meeting.summary_draft_pdf_path}`"
            if meeting.summary_draft_pdf_path else "- Summary Draft: missing",
            f"- Approved Summary: `{meeting.approved_summary_pdf_path}`"
            if meeting.approved_summary_pdf_path else "- Approved Summary: missing",
            f"- Legacy PDF(s): {'; '.join(map(str, meeting.legacy_pdf_paths)) or 'none'}",
            f"- PDFs beside source MP3: "
            f"{'; '.join(map(str, meeting.source_adjacent_pdf_paths)) or 'none'}",
            f"- Missing required files: {output_missing or 'none'}",
            f"- Duplicate/ambiguous matches: "
            f"{'; '.join(meeting.duplicate_or_ambiguous_matches) or 'none'}",
        ])
        if legacy_summary:
            summary_lines.append(
                "- Comparison summary source is a legacy base-name PDF, not a Summary Draft."
            )
        if phase9_payload:
            summary_lines.append(
                f"- Selected Phase 9 artifact: `{phase9_payload.get('cleanup_status', 'unknown')}`; "
                f"raw/cleaned characters: {len(phase9_payload.get('raw_transcript', ''))}/"
                f"{len(phase9_payload.get('cleaned_transcript', ''))}; "
                f"cleaned text exactly matches Transcript TXT: {phase9_txt_exact_match}."
            )
            for history_path, history_raw, history_cleaned, history_payload in phase9:
                summary_lines.append(
                    f"- Cached Phase 9 version `{history_path}`: "
                    f"`{history_payload.get('cleanup_status', 'unknown')}`, "
                    f"{len(history_raw or '')}/{len(history_cleaned or '')} raw/clean chars."
                )
        for phase10_path, status, reason, count, supported_count in phase10_counts:
            summary_lines.append(
                f"- Cached Phase 10 artifact `{phase10_path}`: status `{status}`, "
                f"reason `{reason}`, {count} populated claims; "
                f"{supported_count}/{count} claims cite nonempty Phase 9 text offsets; "
                "semantic support remains unverified."
            )
        if facts:
            summary_lines.append(
                "- SARA extracted fact lines by category: "
                + "; ".join(
                    f"{category}={count}"
                    for category, count in sorted(fact_categories.items())
                )
                + ". Original wording and PDF page numbers are in "
                "`sara_reference_extracts.csv`."
            )
        summary_lines.append("")

        if meeting.sara_pdf_path:
            for fact in facts:
                classification = classify_discrepancy(
                    evidence_kind="sara_reference",
                    source_wording=fact.wording,
                    raw_transcript=raw_text,
                    cleaned_transcript=cleaned_text,
                    summary_text=summary_text,
                )
                normalized = _normalize(fact.wording)
                exact_summary_only = bool(
                    normalized
                    and summary_text is not None
                    and normalized in _normalize(summary_text)
                    and (raw_text is None or normalized not in _normalize(raw_text))
                )
                if classification in {"Summary omission", "Cleanup error"} or (
                    exact_summary_only and len(normalized) >= 30
                ):
                    discrepancy_rows.append({
                        "plant": meeting.plant,
                        "meeting": meeting.recording_filename,
                        "source_type": (
                            "MorningTalk summary vs transcript"
                            if exact_summary_only
                            else "SARA AI PDF vs MorningTalk"
                        ),
                        "source_path": str(meeting.sara_pdf_path),
                        "page": fact.page_number,
                        "category": fact.category,
                        "source_wording": fact.wording,
                        "transcript_path": str(meeting.transcript_txt_path or ""),
                        "summary_path": str(summary_path or ""),
                        "classification": (
                            "Insufficient evidence" if exact_summary_only else classification
                        ),
                        "evidence_status": (
                            "wording occurs in summary but not cached transcript; "
                            "paraphrase and semantics need human review"
                            if exact_summary_only
                            else "exact phrase comparison only; semantic review required"
                        ),
                        "root_cause": (
                            "Undetermined; matching wording absent from available raw transcript"
                            if exact_summary_only
                            else "Potential Summary omission or cleanup-introduced wording; "
                            "exact text only, not a semantic conclusion"
                        ),
                    })
        if raw_text is not None:
            for suspicious_phrase in vocabulary.suspicious_phrases:
                if suspicious_phrase.casefold() in raw_text.casefold():
                    raw_phase9_available = bool(phase9_payload)
                    remains_in_cleaned = (
                        suspicious_phrase.casefold() in (cleaned_text or "").casefold()
                    )
                    discrepancy_rows.append({
                        "plant": meeting.plant,
                        "meeting": meeting.recording_filename,
                        "source_type": "MorningTalk transcript",
                        "source_path": str(
                            phase9[0][0]
                            if phase9
                            else meeting.transcript_txt_path or ""
                        ),
                        "page": "",
                        "category": "suspicious_phrase",
                        "source_wording": suspicious_phrase,
                        "transcript_path": str(meeting.transcript_txt_path or ""),
                        "summary_path": str(summary_path or ""),
                        "classification": (
                            "Suspected transcription error"
                            if raw_phase9_available
                            else "Insufficient evidence"
                        ),
                        "evidence_status": (
                            f"raw Phase 9 phrase; remains in cleaned text={remains_in_cleaned}; "
                            "audio not independently verified"
                            if raw_phase9_available
                            else "phrase found only in Transcript TXT; raw transcript unavailable"
                        ),
                        "root_cause": (
                            "Raw transcription candidate; cannot establish audio correctness offline"
                            if raw_phase9_available
                            else "Undetermined without cached raw transcription"
                        ),
                    })
        for phase10_path, payload in phase10_payloads:
            for section, claim, has_valid_evidence in _analysis_claims(
                payload, source_segments=source_segments
            ):
                analysis = payload.get("meeting_analysis", {})
                if analysis.get("status") != "validated" or not has_valid_evidence:
                    discrepancy_rows.append({
                        "plant": meeting.plant,
                        "meeting": meeting.recording_filename,
                        "source_type": "MorningTalk Phase 10 analysis",
                        "source_path": str(phase10_path),
                        "page": "",
                        "category": section,
                        "source_wording": claim,
                        "transcript_path": str(meeting.transcript_txt_path or ""),
                        "summary_path": str(summary_path or ""),
                        "classification": "Insufficient evidence",
                        "evidence_status": (
                            "cited offsets point to nonempty Phase 9 text; semantic support remains unverified"
                            if analysis.get("status") != "validated" and has_valid_evidence
                            else "analysis is not validated and spans are not source-resolved"
                            if analysis.get("status") != "validated"
                            else "evidence spans do not resolve exactly to cached Phase 9 segments"
                        ),
                        "root_cause": (
                            f"Phase 10 review state: {analysis.get('review_reason', 'unknown')}; "
                            "semantic support remains unverified"
                        ),
                    })
        if not raw_text:
            summary_lines.append(
                "- Comparison status: no existing MorningTalk TXT/Phase 9 transcript; "
                "reference statements cannot be treated as transcription errors or omissions."
            )
        summary_lines.append("")

    inventory_path = report_root / "meeting_inventory.csv"
    _write_csv(
        inventory_path,
        [
            "plant", "month", "recording_filename", "mp3_path", "duration_seconds",
            "sara_pdf_path", "sara_match_status", "transcript_txt_path",
            "full_transcript_pdf_path", "summary_draft_pdf_path",
            "approved_summary_pdf_path", "legacy_pdf_paths",
            "source_adjacent_pdf_paths", "missing_files",
            "duplicate_or_ambiguous_matches", "phase9_json_paths", "phase10_json_paths",
            "phase9_selected_status", "phase9_raw_characters",
            "phase9_cleaned_characters", "phase9_cleaned_text_matches_transcript_txt",
        ],
        inventory_rows,
    )
    discrepancy_path = report_root / "meeting_discrepancies.csv"
    _write_csv(
        discrepancy_path,
        [
            "plant", "meeting", "source_type", "source_path", "page", "category",
            "source_wording", "transcript_path", "summary_path", "classification",
            "evidence_status", "root_cause",
        ],
        discrepancy_rows,
    )
    extracts_path = report_root / "sara_reference_extracts.csv"
    _write_csv(
        extracts_path,
        [
            "plant", "month", "meeting", "reference_pdf", "page", "category",
            "original_wording",
        ],
        reference_extract_rows,
    )
    summary_path = report_root / "benchmark_summary.md"
    recommendations_path = report_root / "recommended_improvements.md"
    recommendations_path.write_text(_recommendations(result), encoding="utf-8")
    summary_lines.extend([
        "## Extraction and root-cause limits",
        "",
        f"- Matched SARA PDFs: {sum(m.sara_pdf_path is not None for m in result.meetings)}; "
        f"extractable pages: {pdf_pages_with_text}/{pdf_pages_total}. "
        "Exact source wording, category, and page are preserved in `sara_reference_extracts.csv`.",
        f"- Unique PDFs found beside source MP3s: {len(source_adjacent_pdf_paths)}; "
        "they were recorded as adjacent artifacts, not treated as SARA references or modified.",
        "- Raw-to-clean cause is traceable only where cached Phase 9 JSON exists. "
        "A phrase match is a suspected issue, not proof against the recording.",
        "- Exact phrase absence is not proof of a topic omission or hallucination; semantic "
        "paraphrases remain `Insufficient evidence` unless a specific source span resolves.",
        "- No WER was computed. SARA PDF claims were not used as transcript ground truth.",
        "",
    ])
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    return (
        inventory_path,
        summary_path,
        discrepancy_path,
        extracts_path,
        recommendations_path,
    )


def _recommendations(result: InventoryResult) -> str:
    matched = sum(meeting.sara_pdf_path is not None for meeting in result.meetings)
    recordings_with_text = sum(meeting.transcript_txt_path is not None for meeting in result.meetings)
    return "\n".join([
        "# Recommended improvements",
        "",
        "Ranked by impact on trustworthy offline comparison; no billable request is implied.",
        "",
        "1. **High — Establish validated transcript coverage.** "
        f"{recordings_with_text}/{len(result.meetings)} recordings currently have a Transcript-folder TXT. "
        "Prioritize validating existing transcripts before considering any new transcription.",
        "2. **High — Human-review suspicious terminology.** Keep raw audio-derived text, cleaned text, "
        "and source-grounded correction candidates separate; do not promote SARA wording to ground truth.",
        "3. **High — Evidence-check cached Phase 10 claims.** Require exact transcript quotes and retain "
        "review-required status whenever a claim is unvalidated or its support is unclear.",
        "4. **Medium — Resolve inventory gaps.** Exact-stem match the available references and investigate "
        "unmatched/ambiguous pairs without using date-nearby fallback matching.",
        "5. **Medium — Improve summary coverage only after transcript validation.** Compare equipment, "
        "production/quality problems, maintenance, safety, measures, actions, owners, and deadlines against "
        "transcript evidence; keep paraphrase-only findings as insufficient evidence.",
        "6. **Low — Preserve PDF extraction provenance.** Keep page-numbered original SARA excerpts and "
        "review non-extractable or layout-split text manually.",
        "",
        f"Current SARA exact matches: {matched}. SARA references remain non-authoritative.",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-root", type=Path, default=DEFAULT_RECORDING_ROOT)
    parser.add_argument("--transcript-root", type=Path, default=DEFAULT_TRANSCRIPT_ROOT)
    parser.add_argument("--sara-root", type=Path, default=DEFAULT_SARA_ROOT)
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    args = parser.parse_args()
    result = discover_inventory(
        args.recording_root,
        args.transcript_root,
        args.sara_root,
        private_root=args.private_root,
    )
    paths = generate_benchmark_reports(result, args.report_root)
    print(f"Recordings: {len(result.meetings)}")
    print(f"Matched SARA references: {sum(m.sara_pdf_path is not None for m in result.meetings)}")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
