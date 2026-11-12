"""MorningTalk command-line transcription workflow."""

import argparse
import logging
from pathlib import Path
import sys

from app.auto_scan import process_scanned_file, synchronize_scan
from app.config import DEFAULT_INPUT_DIR, get_azure_config, get_sql_config
from app.database import AudioRecord, SqlRepository, connect_sql
from app.file_manager import discover_mp3_files, pdf_path_for_source, source_fingerprint, source_processing_lock
from app.monitor import run_monitor
from app.pdf_export import write_pdf
from app.phase10 import (
    AnalysisResumePreflightError,
    preflight_analysis_resume,
    run_analysis_resume,
)
from app.phase10_6 import (
    DEFAULT_ANALYSIS_DIR,
    DEFAULT_PREVIEW_ROOT as PHASE10_PREVIEW_ROOT,
    DEFAULT_RECORDING_ROOT,
    DEFAULT_REVIEW_ROOT,
    DEFAULT_SARA_ROOT,
    ReviewWorkflowError,
    evaluate_batch,
    review_and_maybe_publish,
)
from app.phase10_7 import (
    DEFAULT_MANIFEST_PATH as PILOT_MANIFEST_PATH,
    DEFAULT_PAIRS_CSV as PILOT_PAIRS_CSV,
    DEFAULT_REPORT_PATH as PILOT_REPORT_PATH,
    DEFAULT_RECORDING_ROOT as PILOT_RECORDING_ROOT,
    DEFAULT_SARA_ROOT as PILOT_SARA_ROOT,
    DEFAULT_STAGE_ROOT as PILOT_STAGE_ROOT,
    PilotBatchError,
    authorize_stage_retry,
    execute_pilot_batch,
    prepare_pilot_dry_run,
)
from app.phase9 import (
    PREVIEW_DIRECTORY,
    RESUME_PREVIEW_DIRECTORY,
    run_controlled_live_test,
    run_controlled_transcript_resume,
    run_fixture_preview,
)
from app.scanner import ScannedAudio, scan_directory
from app.summary_preview import (
    SummaryPreviewNotReady,
    write_sara_layout_demo,
    write_sara_summary_preview,
)
from app.transcriber import AzureOpenAITranscriber, estimate_chunk_count


def process_file(source: Path, input_dir: Path, transcriber, force: bool = False) -> bool:
    pdf_path = pdf_path_for_source(source)
    with source_processing_lock(source):
        if pdf_path.exists() and not force:
            logging.info("Skipping existing PDF: %s", pdf_path)
            return False
        if not source.is_file() or source.stat().st_size == 0:
            raise ValueError(f"Input is missing or empty: {source}")
        original_fingerprint = source_fingerprint(source)
        transcript = transcriber.transcribe(source)
        if not transcript.strip():
            raise ValueError(f"Transcription returned no text for {source.name}")
        plant = source.relative_to(input_dir).parts[0] if source.relative_to(input_dir).parts else ""
        if source_fingerprint(source) != original_fingerprint:
            raise RuntimeError(f"Source MP3 changed during processing: {source.name}")
        write_pdf(pdf_path, source, plant, transcript, overwrite=force)
    logging.info("Completed: %s", source)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create complete Thai meeting transcripts and PDFs.")
    parser.add_argument(
        "command",
        nargs="?",
        choices=(
            "scan-once",
            "status",
            "monitor",
            "phase9-preview",
            "phase9-live-test",
            "phase9-resume",
            "phase9-summary-preview",
            "phase10-summary-demo",
            "phase10-analysis-resume",
            "phase10-batch-evaluate",
            "phase10-review",
            "phase10-pilot-batch",
        ),
    )
    parser.add_argument("--file", type=Path, help="Process one MP3 file.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="List expected adjacent PDFs without calling Azure.")
    parser.add_argument("--force", action="store_true", help="Reprocess even if outputs are current.")
    parser.add_argument("--sync-db", action="store_true", help="Persist scan state to MSSQL.")
    parser.add_argument("--process", action="store_true", help="Explicitly opt in to Azure processing.")
    parser.add_argument("--interval", type=int, default=300, help="Monitor interval in seconds.")
    parser.add_argument("--once", action="store_true", help="Run one monitor cycle and exit.")
    parser.add_argument("--local-only", action="store_true", help="Monitor without MSSQL synchronization.")
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=PREVIEW_DIRECTORY,
        help="Separate output directory for Phase 9 preview artifacts.",
    )
    parser.add_argument(
        "--resume-preview-dir",
        type=Path,
        default=RESUME_PREVIEW_DIRECTORY,
        help="New versioned preview directory for resuming AudioFileId 4 without transcription.",
    )
    parser.add_argument(
        "--phase9-json",
        type=Path,
        help="Existing Phase 9 JSON input for offline validated-summary preview.",
    )
    parser.add_argument(
        "--summary-source",
        type=Path,
        help="Source MP3 path used only for summary metadata and fingerprint verification.",
    )
    parser.add_argument(
        "--summary-output-dir",
        type=Path,
        help="Directory for a new <basename>_SARA_Preview.pdf output.",
    )
    parser.add_argument(
        "--confirm-azure-billing",
        action="store_true",
        help="Explicitly authorize billable Azure requests for Phase 9 tests/resume or Phase 10 analysis resume.",
    )
    parser.add_argument(
        "--confirm-mssql-write",
        action="store_true",
        help="Explicitly authorize Phase 9 status/output writes to the existing MSSQL audio record.",
    )
    parser.add_argument("--recording-root", type=Path, default=DEFAULT_RECORDING_ROOT)
    parser.add_argument("--sara-root", type=Path, default=DEFAULT_SARA_ROOT)
    parser.add_argument(
        "--pairs-csv",
        type=Path,
        default=DEFAULT_ANALYSIS_DIR / "recording_pdf_pairs.csv",
    )
    parser.add_argument("--evaluation-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--preview-root", type=Path, default=PHASE10_PREVIEW_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--analysis-json", type=Path)
    parser.add_argument("--transcript-json", type=Path)
    parser.add_argument("--review-source", type=Path)
    parser.add_argument("--reviewer")
    parser.add_argument("--item-id", help="Review item key, e.g. topics:1.")
    parser.add_argument("--decision", choices=("approve", "reject", "edit"))
    parser.add_argument("--review-reason")
    parser.add_argument("--edit-text")
    parser.add_argument("--edit-owner")
    parser.add_argument("--edit-deadline")
    parser.add_argument("--clear-owner", action="store_true")
    parser.add_argument("--clear-deadline", action="store_true")
    parser.add_argument(
        "--confirm-review",
        action="store_true",
        help="Enable an interactive confirmation prompt for review or PDF approval.",
    )
    parser.add_argument("--generate-approved-pdf", action="store_true")
    parser.add_argument("--approved-pdf-output-dir", type=Path)
    parser.add_argument("--pilot-manifest", type=Path, default=PILOT_MANIFEST_PATH)
    parser.add_argument("--pilot-report", type=Path, default=PILOT_REPORT_PATH)
    parser.add_argument("--pilot-stage-root", type=Path, default=PILOT_STAGE_ROOT)
    parser.add_argument("--pilot-recording-root", type=Path, default=PILOT_RECORDING_ROOT)
    parser.add_argument("--pilot-sara-root", type=Path, default=PILOT_SARA_ROOT)
    parser.add_argument("--pilot-pairs-csv", type=Path, default=PILOT_PAIRS_CSV)
    parser.add_argument("--pilot-preview-root", type=Path, default=PHASE10_PREVIEW_ROOT)
    parser.add_argument("--pilot-review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--pilot-max-requests", type=int)
    parser.add_argument(
        "--pilot-execute",
        action="store_true",
        help="Execute the pilot only with --confirm-azure-billing; never used by dry-run.",
    )
    parser.add_argument("--pilot-reconcile-attempt")
    parser.add_argument(
        "--pilot-confirm-retry",
        action="store_true",
        help="Enable typed confirmation to authorize a possible duplicate stage retry.",
    )
    return parser


def _records_by_path(repository: SqlRepository) -> dict[str, AudioRecord]:
    return {record.source_path.casefold(): record for record in repository.list_audio_files()}


def _print_scan(items: list[ScannedAudio], states: list[tuple[ScannedAudio, int | None, str]] | None = None) -> None:
    if states is None:
        states = [(item, None, item.status) for item in items]
    counts: dict[str, int] = {}
    for item, _audio_file_id, status in states:
        counts[status] = counts.get(status, 0) + 1
        print(f"{status}: {item.path} -> {item.pdf_path}")
    print("MP3 files:", len(items))
    print("Status counts:", ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))


def _scan_once(
    input_dir: Path,
    *,
    sync_db: bool,
    process_enabled: bool,
    files: list[Path] | None = None,
) -> int:
    repository = None
    connection = None
    try:
        if sync_db or process_enabled:
            connection = connect_sql(get_sql_config())
            repository = SqlRepository(connection)
            repository.recover_expired_leases()
            records = _records_by_path(repository)
        else:
            records = None
        items = scan_directory(input_dir, files=files, records=records)
        if repository is None:
            _print_scan(items)
            print("Mode: local scan-only; database and Azure were not contacted.")
            return 0

        states = synchronize_scan(items, repository)
        if process_enabled:
            azure_config = get_azure_config()
            failed = 0
            updated = []
            for item, audio_file_id, status in states:
                full_pdf = item.path.with_name(f"{item.path.stem}_Full.pdf")
                needs_missing_output = (
                    status == "review_required" and not full_pdf.exists()
                )
                if audio_file_id is None or (
                    status not in {"discovered", "modified", "failed"}
                    and not needs_missing_output
                ):
                    updated.append((item, audio_file_id, status))
                    continue
                try:
                    status = process_scanned_file(
                        item,
                        input_dir,
                        audio_file_id,
                        repository,
                        azure_config,
                        allow_review_required=needs_missing_output,
                    )
                except Exception as error:
                    failed += 1
                    status = "failed"
                    logging.error("Processing failed for %s; error_class=%s", item.path.name, type(error).__name__)
                updated.append((item, audio_file_id, status))
            states = updated
            _print_scan(items, states)
            return 1 if failed else 0
        _print_scan(items, states)
        print("Mode: database scan-only; no Azure requests were made.")
        return 0
    except Exception as error:
        logging.error("Scan failed; error_class=%s", type(error).__name__)
        return 2
    finally:
        if connection is not None:
            connection.close()


def _show_status() -> int:
    connection = None
    try:
        connection = connect_sql(get_sql_config())
        counts = SqlRepository(connection).status_counts()
        print("Database status counts:")
        for status, count in sorted(counts.items()):
            print(f"{status}: {count}")
        return 0
    except Exception as error:
        logging.error("Status query failed; error_class=%s", type(error).__name__)
        return 2
    finally:
        if connection is not None:
            connection.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "monitor":
        try:
            return run_monitor(
                lambda directory, **options: _scan_once(directory, **options),
                args.input_dir.resolve(),
                interval_seconds=args.interval,
                local_only=args.local_only,
                process_enabled=args.process,
                once=args.once,
            )
        except ValueError as error:
            logging.error("%s", error)
            return 2
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "phase10-pilot-batch":
        if args.pilot_reconcile_attempt:
            if not args.pilot_confirm_retry:
                logging.error(
                    "Request reconciliation requires --pilot-confirm-retry."
                )
                return 2
            expected = f"RETRY {args.pilot_reconcile_attempt}"
            try:
                response = input(
                    "This may repeat a billable request. Type "
                    f"{expected!r} to authorize: "
                )
            except EOFError:
                logging.error("Interactive retry confirmation was unavailable.")
                return 2
            if response.strip() != expected:
                logging.error("Retry confirmation did not match.")
                return 2
            try:
                authorize_stage_retry(
                    args.pilot_manifest,
                    args.pilot_reconcile_attempt,
                    confirmed=True,
                )
                print("Retry authorization recorded; no request was sent.")
                return 0
            except PilotBatchError as error:
                logging.error("Reconciliation failed; reason=%s", error.reason)
                return 2
        if args.pilot_execute:
            if args.dry_run:
                logging.error("Choose either --dry-run or --pilot-execute, not both.")
                return 2
            if not args.confirm_azure_billing:
                logging.error(
                    "Execution requires --confirm-azure-billing. No request was sent."
                )
                return 2
            try:
                manifest = execute_pilot_batch(
                    manifest_path=args.pilot_manifest,
                    stage_root=args.pilot_stage_root,
                    max_requests=args.pilot_max_requests,
                )
                print(f"Pilot manifest: {args.pilot_manifest}")
                print(f"Execution status: {manifest['execution_status']}")
                return 0
            except PilotBatchError as error:
                logging.error(
                    "Pilot execution stopped; reason=%s",
                    error.reason,
                )
                return 2
        if not args.dry_run:
            logging.error(
                "Use --dry-run for offline preflight or --pilot-execute with "
                "--confirm-azure-billing for separately authorized processing."
            )
            return 2
        try:
            manifest, blockers = prepare_pilot_dry_run(
                manifest_path=args.pilot_manifest,
                report_path=args.pilot_report,
                stage_root=args.pilot_stage_root,
                recording_root=args.pilot_recording_root,
                sara_root=args.pilot_sara_root,
                pairs_csv=args.pilot_pairs_csv,
                preview_root=args.pilot_preview_root,
                review_root=args.pilot_review_root,
                max_requests=args.pilot_max_requests,
            )
            print(f"Selected recordings: {len(manifest['recordings'])}")
            for recording in manifest["recordings"]:
                requests_by_stage = recording.get(
                    "expected_requests",
                    {
                        "transcription": recording["expected_audio_chunks"],
                        "cleanup": 1,
                        "analysis": 1,
                        "total": recording["expected_audio_chunks"] + 2,
                    },
                )
                print(
                    f"{recording['month']}: {recording['source']} "
                    f"duration={recording['duration_seconds']:.3f}s "
                    f"chunks={recording['expected_audio_chunks']} "
                    f"requests={requests_by_stage['total']}"
                )
            print(f"Manifest: {args.pilot_manifest}")
            print(f"Dry-run report: {args.pilot_report}")
            if blockers:
                print("Blockers: " + ", ".join(blockers))
                return 2
            print("Mode: offline preflight; zero Azure requests and zero SQL writes.")
            return 0
        except PilotBatchError as error:
            logging.error("Pilot preflight failed; reason=%s", error.reason)
            return 2
    if args.command == "phase10-batch-evaluate":
        try:
            result = evaluate_batch(
                recording_root=args.recording_root,
                sara_root=args.sara_root,
                pairs_csv=args.pairs_csv,
                preview_root=args.preview_root,
                review_root=args.review_root,
                output_dir=args.evaluation_dir,
            )
            counts: dict[str, int] = {}
            for row in result.rows:
                status = row["next_step"]
                counts[status] = counts.get(status, 0) + 1
            print(f"Matched pairs evaluated: {len(result.rows)}")
            for status, count in sorted(counts.items()):
                print(f"{status}: {count}")
            print(f"CSV report: {result.csv_path}")
            print(f"Markdown report: {result.markdown_path}")
            print("Mode: offline; no Azure or SQL requests were made.")
            return 0
        except ReviewWorkflowError as error:
            logging.error("Batch evaluation failed; reason=%s", error.reason)
            return 2
        except Exception as error:
            logging.error(
                "Batch evaluation failed; error_class=%s",
                type(error).__name__,
            )
            return 2
    if args.command == "phase10-review":
        required = (
            args.analysis_json,
            args.transcript_json,
            args.review_source,
            args.reviewer,
        )
        if not all(required):
            logging.error(
                "Review requires --analysis-json, --transcript-json, "
                "--review-source, and --reviewer."
            )
            return 2
        if args.decision and not args.item_id:
            logging.error("--decision requires --item-id.")
            return 2
        if args.generate_approved_pdf and not args.approved_pdf_output_dir:
            logging.error(
                "--generate-approved-pdf requires --approved-pdf-output-dir."
            )
            return 2
        edits: dict[str, str | None] = {}
        if args.edit_text is not None:
            edits["text"] = args.edit_text
        if args.edit_owner is not None:
            edits["owner"] = args.edit_owner
        if args.edit_deadline is not None:
            edits["deadline"] = args.edit_deadline
        if args.clear_owner:
            edits["owner"] = None
        if args.clear_deadline:
            edits["deadline"] = None
        if args.decision == "edit" and not edits:
            logging.error("An edit decision requires at least one --edit-* option.")
            return 2
        if args.decision != "edit" and edits:
            logging.error("--edit-* options require --decision edit.")
            return 2
        if (args.decision or args.generate_approved_pdf) and not args.confirm_review:
            logging.error("Review or PDF generation requires --confirm-review.")
            return 2
        confirmed = False
        if args.decision:
            action = args.decision.upper()
            try:
                response = input(f"Type {action} to confirm this item decision: ")
            except EOFError:
                logging.error("Interactive confirmation was unavailable.")
                return 2
            if response.strip() != action:
                logging.error("Review confirmation did not match.")
                return 2
            confirmed = True
        if args.generate_approved_pdf:
            try:
                response = input(
                    "Type GENERATE APPROVED PDF to confirm PDF publication: "
                )
            except EOFError:
                logging.error("Interactive confirmation was unavailable.")
                return 2
            if response.strip() != "GENERATE APPROVED PDF":
                logging.error("PDF confirmation did not match.")
                return 2
            confirmed = True
        try:
            report, pdf_path = review_and_maybe_publish(
                analysis_path=args.analysis_json,
                transcript_path=args.transcript_json,
                source=args.review_source,
                reviewer=args.reviewer,
                review_dir=args.review_root,
                item_id=args.item_id,
                decision=args.decision,
                reason=args.review_reason,
                edits=edits or None,
                confirmed=confirmed,
                generate_pdf=args.generate_approved_pdf,
                output_dir=args.approved_pdf_output_dir,
            )
            print(report)
            if pdf_path is not None:
                print(f"Approved PDF: {pdf_path}")
            return 0
        except ReviewWorkflowError as error:
            logging.error("Human review failed; reason=%s", error.reason)
            return 2
        except Exception as error:
            logging.error(
                "Human review failed; error_class=%s",
                type(error).__name__,
            )
            return 2
    if args.command == "phase9-summary-preview":
        if not args.phase9_json or not args.summary_source or not args.summary_output_dir:
            logging.error(
                "Offline summary preview requires --phase9-json, --summary-source, "
                "and --summary-output-dir."
            )
            return 2
        try:
            output_path = write_sara_summary_preview(
                args.phase9_json,
                args.summary_source,
                args.summary_output_dir,
            )
            print(f"Validated SARA-style summary preview: {output_path}")
            return 0
        except SummaryPreviewNotReady as error:
            logging.warning(
                "Summary preview not created; reason=%s",
                error.reason,
            )
            return 2
        except FileExistsError as error:
            logging.error("%s", error)
            return 2
        except Exception as error:
            logging.error(
                "Offline summary preview failed; error_class=%s",
                type(error).__name__,
            )
            return 2
    if args.command == "phase10-summary-demo":
        if not args.phase9_json or not args.summary_source or not args.summary_output_dir:
            logging.error(
                "Offline layout demo requires --phase9-json, --summary-source, "
                "and --summary-output-dir."
            )
            return 2
    if args.command == "phase10-analysis-resume":
        if not args.confirm_azure_billing:
            logging.error("Analysis resume requires --confirm-azure-billing.")
            return 2
        if not args.phase9_json or not args.summary_source or not args.summary_output_dir:
            logging.error(
                "Analysis resume requires --phase9-json, --summary-source, "
                "and --summary-output-dir."
            )
            return 2
        try:
            preflight = preflight_analysis_resume(
                args.phase9_json,
                args.summary_source,
                args.summary_output_dir,
            )
            print(
                "Preflight passed: "
                f"endpoint={preflight.endpoint_url}; "
                f"deployment={preflight.deployment}; "
                "analysis_requests_planned=1; retries=disabled; "
                "transcription_requests=0; cleanup_requests=0; sql_writes=0"
            )
            result = run_analysis_resume(
                args.phase9_json,
                args.summary_source,
                args.summary_output_dir,
            )
            print(f"Meeting Analysis status: {result.analysis.status}")
            print(f"Analysis JSON: {result.json_path}")
            if result.summary_pdf is None:
                print(
                    "Summary PDF not created; validation rule: "
                    f"{result.analysis.review_reason or 'UnknownValidationFailure'}"
                )
                return 2
            from pypdf import PdfReader

            page_count = len(PdfReader(str(result.summary_pdf)).pages)
            print(f"Summary PDF: {result.summary_pdf}; pages={page_count}")
            return 0
        except AnalysisResumePreflightError as error:
            logging.error("Analysis resume preflight failed; rule=%s", error.reason)
            return 2
        except Exception as error:
            logging.error(
                "Analysis resume failed; error_class=%s",
                type(error).__name__,
            )
            return 2
        try:
            output_path = write_sara_layout_demo(
                args.phase9_json,
                args.summary_source,
                args.summary_output_dir,
            )
            print(f"SARA-style layout demonstration (not a validated summary): {output_path}")
            return 0
        except SummaryPreviewNotReady as error:
            logging.warning(
                "Layout demonstration not created; reason=%s",
                error.reason,
            )
            return 2
        except FileExistsError as error:
            logging.error("%s", error)
            return 2
        except Exception as error:
            logging.error(
                "Offline layout demonstration failed; error_class=%s",
                type(error).__name__,
            )
            return 2
    if args.command == "phase9-preview":
        try:
            summary_path = run_fixture_preview(args.preview_dir)
            print(f"Summary fixture PDF: {summary_path}")
            print(f"Full-transcript fixture PDF: {summary_path.with_name(summary_path.stem + '_Full.pdf')}")
            return 0
        except Exception as error:
            logging.error("Fixture preview failed; error_class=%s", type(error).__name__)
            return 2
    if args.command == "phase9-live-test":
        try:
            summary_path, full_path, json_path, document, analysis, outputs = run_controlled_live_test(
                input_dir=args.input_dir.resolve(),
                output_dir=args.preview_dir.resolve(),
                confirm_azure_billing=args.confirm_azure_billing,
                confirm_mssql_write=args.confirm_mssql_write,
            )
            print(f"Summary PDF: {summary_path} ({outputs.summary.status})")
            print(f"Full transcript PDF: {full_path} ({outputs.full_transcript.status})")
            print(f"Transcript and analysis JSON: {json_path}")
            print(f"Diarization status: {document.diarization_status}")
            print(f"Cleanup status: {document.cleanup_status}")
            print(f"Meeting analysis status: {analysis.status}")
            print(
                "Detected speakers: "
                f"{len({s.speaker_id for s in document.segments if s.speaker_id and s.speaker_id != 'Speaker unknown'})}"
            )
            print(f"Segments requiring review: {sum(s.review_required for s in document.segments)}")
            print(
                "Cleanup tokens: "
                f"{document.input_tokens if document.input_tokens is not None else 'unavailable'} input / "
                f"{document.output_tokens if document.output_tokens is not None else 'unavailable'} output"
            )
            print(
                "Analysis tokens: "
                f"{analysis.input_tokens if analysis.input_tokens is not None else 'unavailable'} input / "
                f"{analysis.output_tokens if analysis.output_tokens is not None else 'unavailable'} output"
            )
            print("Estimated cost: unavailable; verify current Azure pricing and usage billing.")
            return 0
        except Exception as error:
            logging.error("Controlled live test failed; error_class=%s", type(error).__name__)
            return 2
    if args.command == "phase9-resume":
        try:
            summary_path, full_path, json_path, document, analysis, outputs = (
                run_controlled_transcript_resume(
                    input_dir=args.input_dir.resolve(),
                    output_dir=args.resume_preview_dir.resolve(),
                    confirm_azure_billing=args.confirm_azure_billing,
                    confirm_mssql_write=args.confirm_mssql_write,
                )
            )
            print(f"Summary PDF: {summary_path} ({outputs.summary.status})")
            print(f"Full transcript PDF: {full_path} ({outputs.full_transcript.status})")
            print(f"Transcript and analysis JSON: {json_path}")
            print(f"Diarization status: {document.diarization_status}")
            print(f"Cleanup status: {document.cleanup_status}")
            print(f"Meeting analysis status: {analysis.status}")
            print(
                "Cleanup tokens: "
                f"{document.input_tokens if document.input_tokens is not None else 'unavailable'} input / "
                f"{document.output_tokens if document.output_tokens is not None else 'unavailable'} output"
            )
            print(
                "Analysis tokens: "
                f"{analysis.input_tokens if analysis.input_tokens is not None else 'unavailable'} input / "
                f"{analysis.output_tokens if analysis.output_tokens is not None else 'unavailable'} output"
            )
            return 0
        except Exception as error:
            logging.error("Transcript resume failed; error_class=%s", type(error).__name__)
            return 2
    if args.command is None:
        if not args.process:
            args.dry_run = True
        else:
            input_dir = args.input_dir.resolve()
            files = [args.file.resolve()] if args.file else None
            return _scan_once(input_dir, sync_db=True, process_enabled=True, files=files)
    if args.command == "status":
        return _show_status()
    if args.command == "scan-once":
        return _scan_once(
            args.input_dir.resolve(),
            sync_db=args.sync_db or args.process,
            process_enabled=args.process,
        )
    try:
        if args.file:
            files = [args.file.resolve()]
            input_dir = args.input_dir.resolve()
        else:
            input_dir = args.input_dir.resolve()
            files = discover_mp3_files(input_dir)
        if not files:
            logging.info("No MP3 files found.")
            return 0
        if args.dry_run:
            pending = 0
            config = get_azure_config()
            for source in files:
                pdf = pdf_path_for_source(source)
                duration = AzureOpenAITranscriber._probe_duration(source)
                chunks = estimate_chunk_count(
                    duration, config.chunk_duration_seconds, config.chunk_overlap_seconds
                )
                exists = pdf.is_file()
                pending += not exists
                print(f"{source} -> {pdf} | duration={duration:.1f}s | chunks={chunks} | "
                      f"{'existing PDF' if exists else 'requires processing'}")
            print(f"MP3 files: {len(files)}; existing PDFs: {len(files) - pending}; requiring processing: {pending}")
            return 0
        config = get_azure_config()
        transcriber = AzureOpenAITranscriber(config)
        failures = 0
        for source in files:
            try:
                logging.info("Processing %s", source)
                process_file(source, input_dir, transcriber, args.force)
            except Exception:
                failures += 1
                logging.exception("Failed to process %s", source)
        return 1 if failures else 0
    except (OSError, ValueError) as error:
        logging.error("%s", error)
        return 2


if __name__ == "__main__":
    sys.exit(main())
