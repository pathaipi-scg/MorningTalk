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
from app.scanner import ScannedAudio, scan_directory
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
    parser.add_argument("command", nargs="?", choices=("scan-once", "status", "monitor"))
    parser.add_argument("--file", type=Path, help="Process one MP3 file.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="List expected adjacent PDFs without calling Azure.")
    parser.add_argument("--force", action="store_true", help="Reprocess even if outputs are current.")
    parser.add_argument("--sync-db", action="store_true", help="Persist scan state to MSSQL.")
    parser.add_argument("--process", action="store_true", help="Explicitly opt in to Azure processing.")
    parser.add_argument("--interval", type=int, default=300, help="Monitor interval in seconds.")
    parser.add_argument("--once", action="store_true", help="Run one monitor cycle and exit.")
    parser.add_argument("--local-only", action="store_true", help="Monitor without MSSQL synchronization.")
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
                if audio_file_id is None or status not in {"discovered", "modified", "failed"}:
                    updated.append((item, audio_file_id, status))
                    continue
                try:
                    status = process_scanned_file(
                        item, input_dir, audio_file_id, repository, azure_config
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
