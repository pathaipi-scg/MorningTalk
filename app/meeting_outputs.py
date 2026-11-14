"""Publish meeting outputs with validation and atomic replacement."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

from app.file_manager import resolve_meeting_output_paths
from app.meeting_analysis import MeetingAnalysis
from app.pdf_export import write_meeting_summary_pdf, write_transcript_pdf
from app.transcript import TranscriptDocument
from app.workspace_paths import DEFAULT_PREVIEW_ROOT, DEFAULT_TEMP_ROOT


class OutputPublicationError(RuntimeError):
    """A staged output set could not be published or rolled back safely."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _remove_temporary_files(paths: list[Path]) -> list[str]:
    errors = []
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as error:
            errors.append(f"{path}:{type(error).__name__}")
    return errors


def publish_staged_output_set(
    staged_outputs: dict[Path, Path],
    *,
    approved_summary: Path,
    backup_root: Path | None = None,
) -> Path:
    """Publish validated staged files, restoring the prior set if replacement fails."""
    if not staged_outputs:
        raise ValueError("At least one staged output is required.")

    approved_path = approved_summary.resolve()
    targets = list(staged_outputs)
    resolved_targets = [path.resolve() for path in targets]
    if len(set(resolved_targets)) != len(resolved_targets):
        raise ValueError("Output destinations must be unique.")
    if any(
        target == approved_path or target.name.casefold().endswith("_summary.pdf")
        for target in resolved_targets
    ):
        raise ValueError("Approved summary PDFs cannot be published by this workflow.")
    if len({target.parent for target in resolved_targets}) != 1:
        raise ValueError("A meeting output set must share one destination directory.")
    for candidate in staged_outputs.values():
        if not candidate.is_file():
            raise OutputPublicationError(f"StagedOutputMissing:{candidate}")

    backup_root = backup_root or DEFAULT_TEMP_ROOT / "output_publication_backups"
    backup_root.mkdir(parents=True, exist_ok=True)
    backup_dir = backup_root / uuid.uuid4().hex
    backup_dir.mkdir()
    local_staged: dict[Path, Path] = {}
    backups: dict[Path, Path | None] = {}
    temporary_paths: list[Path] = []

    try:
        for target, candidate in staged_outputs.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.",
                suffix=".staged",
                dir=target.parent,
            )
            temporary = Path(temporary_name)
            temporary_paths.append(temporary)
            with os.fdopen(descriptor, "wb") as output, candidate.open("rb") as source:
                shutil.copyfileobj(source, output)
                output.flush()
                os.fsync(output.fileno())
            if _file_sha256(temporary) != _file_sha256(candidate):
                raise OSError(f"Staged output copy failed verification: {target.name}")
            local_staged[target] = temporary

            if target.exists():
                backup = backup_dir / target.name
                shutil.copy2(target, backup)
                if _file_sha256(backup) != _file_sha256(target):
                    raise OSError(f"Output backup failed verification: {target.name}")
                backups[target] = backup
            else:
                backups[target] = None

        transaction = {
            "targets": [
                {
                    "path": str(target),
                    "candidate": str(staged_outputs[target]),
                    "candidate_sha256": _file_sha256(staged_outputs[target]),
                    "previous_backup": (
                        str(backups[target]) if backups[target] is not None else None
                    ),
                }
                for target in targets
            ]
        }
        (backup_dir / "transaction.json").write_text(
            json.dumps(transaction, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as error:
        cleanup_errors = _remove_temporary_files(temporary_paths)
        reason = "OutputPublicationStagingFailed"
        if cleanup_errors:
            reason += f":TemporaryCleanupFailed:{'|'.join(cleanup_errors)}"
        raise OutputPublicationError(f"{reason}; backups={backup_dir}") from error

    attempted: list[Path] = []
    try:
        for target in targets:
            attempted.append(target)
            os.replace(local_staged[target], target)
    except OSError as error:
        rollback_errors = []
        for target in reversed(attempted):
            backup = backups[target]
            try:
                if backup is not None:
                    descriptor, rollback_name = tempfile.mkstemp(
                        prefix=f".{target.name}.",
                        suffix=".rollback",
                        dir=target.parent,
                    )
                    rollback = Path(rollback_name)
                    temporary_paths.append(rollback)
                    with os.fdopen(descriptor, "wb") as output, backup.open("rb") as source:
                        shutil.copyfileobj(source, output)
                        output.flush()
                        os.fsync(output.fileno())
                    if _file_sha256(rollback) != _file_sha256(backup):
                        raise OSError(f"Rollback copy failed verification: {target.name}")
                    os.replace(rollback, target)
                elif target.exists():
                    target.unlink()
            except OSError as rollback_error:
                rollback_errors.append(
                    f"{target}:{type(rollback_error).__name__}"
                )
        cleanup_errors = _remove_temporary_files(temporary_paths)
        rollback_errors.extend(cleanup_errors)
        if rollback_errors:
            raise OutputPublicationError(
                "OutputPublicationRollbackIncomplete:"
                f"{'|'.join(rollback_errors)}; backups={backup_dir}"
            ) from error
        raise OutputPublicationError(
            f"OutputPublicationFailedAndRolledBack; backups={backup_dir}"
        ) from error

    return backup_dir


@dataclass(frozen=True)
class PdfOutputStatus:
    path: Path
    status: str
    error_class: str | None = None


@dataclass(frozen=True)
class MeetingPdfOutputs:
    summary: PdfOutputStatus
    full_transcript: PdfOutputStatus
    transcript_text: PdfOutputStatus | None = None


def meeting_pdf_paths(
    source: Path, output_dir: Path | None = None
) -> tuple[Path, Path]:
    if output_dir is None:
        paths = resolve_meeting_output_paths(source)
        return paths.summary_draft_pdf, paths.full_transcript_pdf
    directory = output_dir
    return (
        directory / f"{source.stem}_Summary_Draft.pdf",
        directory / f"{source.stem}_Full.pdf",
    )


def write_meeting_pdfs(
    source: Path,
    plant: str,
    transcript: TranscriptDocument,
    analysis: MeetingAnalysis,
    *,
    output_dir: Path | None = None,
    fixture_preview: bool = False,
) -> MeetingPdfOutputs:
    summary_path, full_path = meeting_pdf_paths(source, output_dir)
    text_path = (
        output_dir / f"{source.stem}.txt"
        if output_dir is not None
        else resolve_meeting_output_paths(source).transcript_text
    )
    if transcript.cleanup_status != "completed" and not fixture_preview:
        error = "TranscriptCleanupRequiresReview"
        return MeetingPdfOutputs(
            PdfOutputStatus(summary_path, "not_published", error),
            PdfOutputStatus(full_path, "not_published", error),
            PdfOutputStatus(text_path, "not_published", error),
        )

    output_existed = {
        text_path: text_path.exists(),
        full_path: full_path.exists(),
    }
    summary_existed = summary_path.exists()
    stage_root = (
        DEFAULT_PREVIEW_ROOT
        / "meeting_output_staging"
        / source.stem
        / uuid.uuid4().hex
    )
    stage_root.mkdir(parents=True)
    staged_text = stage_root / text_path.name
    staged_full = stage_root / full_path.name
    staged_summary = stage_root / summary_path.name

    write_transcript_text(
        staged_text,
        transcript.cleaned_transcript,
        overwrite=False,
    )
    write_transcript_pdf(
        staged_full,
        source,
        plant,
        transcript,
        overwrite=False,
        fixture_preview=fixture_preview,
    )
    if staged_text.read_text(encoding="utf-8") != transcript.cleaned_transcript:
        raise ValueError("Transcript text staging validation failed.")
    _validate_staged_pdf(staged_full, transcript.cleaned_transcript)

    publish_summary = (
        analysis.overview is not None
        and bool(analysis.overview.text.strip())
        and bool(analysis.overview.evidence)
        and (
            analysis.status == "validated"
            or not summary_existed
        )
    )
    if publish_summary:
        write_meeting_summary_pdf(
            staged_summary,
            source,
            plant,
            analysis,
            overwrite=False,
            fixture_preview=fixture_preview,
            draft=True,
        )
        _validate_staged_pdf(
            staged_summary,
            analysis.overview.text,
            draft=True,
        )

    staged_outputs = {
        text_path: staged_text,
        full_path: staged_full,
    }
    if publish_summary:
        staged_outputs[summary_path] = staged_summary

    approved_summary = (
        resolve_meeting_output_paths(source).approved_summary_pdf
        if output_dir is None
        else output_dir / f"{source.stem}_Summary.pdf"
    )
    publish_staged_output_set(
        staged_outputs,
        approved_summary=approved_summary,
        backup_root=DEFAULT_TEMP_ROOT / "output_publication_backups",
    )

    text_status = PdfOutputStatus(
        text_path,
        "overwritten" if output_existed[text_path] else "created",
    )
    full_transcript = PdfOutputStatus(
        full_path,
        "overwritten" if output_existed[full_path] else "created",
    )
    if publish_summary:
        summary = PdfOutputStatus(
            summary_path,
            "overwritten" if summary_existed else "created",
        )
    elif summary_existed:
        summary = PdfOutputStatus(summary_path, "existing")
    else:
        summary = PdfOutputStatus(
            summary_path,
            "not_published",
            "MeetingAnalysisUnavailable",
        )
    return MeetingPdfOutputs(summary, full_transcript, text_status)


def _validate_staged_pdf(
    path: Path,
    expected_text: str,
    *,
    draft: bool = False,
) -> None:
    from app.pdf_export import _extract_pdf_text, _normalized_text

    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Staged PDF is missing or empty: {path.name}")
    pages, extracted = _extract_pdf_text(path)
    normalized_extracted = _normalized_text(extracted)
    position = 0
    for line in expected_text.splitlines():
        normalized_line = _normalized_text(line)
        if not normalized_line:
            continue
        position = normalized_extracted.find(normalized_line, position)
        if position < 0:
            raise ValueError(f"Staged PDF content validation failed: {path.name}")
        position += len(normalized_line)
    if pages < 1:
        raise ValueError(f"Staged PDF content validation failed: {path.name}")
    if draft and "DRAFT - NOT APPROVED" not in extracted.upper():
        raise ValueError(f"Staged summary draft label is missing: {path.name}")


def write_transcript_text(
    path: Path,
    text: str,
    *,
    overwrite: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Transcript already exists: {path.name}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".tmp.txt",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    encoded = text.encode("utf-8")
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        if temporary.read_bytes() != encoded:
            raise OSError("Temporary transcript validation failed.")
        if path.exists() and not overwrite:
            raise FileExistsError(f"Transcript already exists: {path.name}")
        if overwrite:
            os.replace(temporary, path)
        else:
            temporary.rename(path)
    finally:
        temporary.unlink(missing_ok=True)
