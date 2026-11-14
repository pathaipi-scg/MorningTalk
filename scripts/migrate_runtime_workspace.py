"""Copy MorningTalk runtime artifacts into the private project workspace."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.workspace_paths import (  # noqa: E402
    WORKSPACE_ROOT,
    WORKSPACE_ENV_VAR,
)

DEFAULT_WORKSPACE_ROOT = WORKSPACE_ROOT

DEFAULT_LEGACY_RUNTIME_ROOT = (
    Path(os.getenv("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    / "MorningTalk"
)
DEFAULT_TEMP_RUNTIME_ROOT = Path(tempfile.gettempdir())


class MigrationError(RuntimeError):
    """Raised when a source changes or a destination conflicts."""


@dataclass(frozen=True)
class MigrationEntry:
    source: Path
    destination: Path
    size: int
    sha256: str
    modified_ns: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iter_source_files(source: Path) -> Iterable[Path]:
    if source.is_file():
        yield source
        return
    if not source.is_dir():
        return
    for current, directory_names, file_names in os.walk(source, followlinks=False):
        current_path = Path(current)
        directory_names[:] = sorted(
            name for name in directory_names
            if not (current_path / name).is_symlink()
        )
        for name in sorted(file_names):
            path = current_path / name
            if path.is_symlink():
                raise MigrationError(f"SourceSymlinkUnsupported:{path}")
            yield path


def default_sources(
    project_root: Path,
    legacy_runtime_root: Path,
    temp_runtime_root: Path = DEFAULT_TEMP_RUNTIME_ROOT,
) -> list[tuple[Path, Path]]:
    workspace = DEFAULT_WORKSPACE_ROOT
    mappings = [(legacy_runtime_root, workspace)]
    benchmark_reports = project_root / "reports" / "benchmark"
    if benchmark_reports.is_dir():
        mappings.append((benchmark_reports, workspace / "reports" / "benchmark"))
    analysis_root = project_root / "references" / "SaraAI" / "analysis"
    for filename in ("phase10_6_batch_evaluation.csv", "phase10_6_batch_evaluation.md"):
        source = analysis_root / filename
        if not source.is_file():
            continue
        mappings.append(
            (
                source,
                workspace / "reports" / "phase10_6" / filename,
            )
        )
    for filename in ("dataset_inventory.csv", "dataset_quality_report.md"):
        source = analysis_root / filename
        if not source.is_file():
            continue
        mappings.append(
            (
                source,
                workspace / "reports" / "legacy_analysis" / filename,
            )
        )
    for filename in ("phase10_7_pilot_manifest.json", "phase10_7_pilot_dry_run.md"):
        source = analysis_root / filename
        if not source.is_file():
            continue
        mappings.append(
            (
                source,
                workspace / "reports" / "phase10_7" / filename,
            )
        )
    if temp_runtime_root.is_dir():
        for path in sorted(temp_runtime_root.iterdir(), key=lambda item: item.name.casefold()):
            if path.is_dir() and path.name.startswith("morningtalk-"):
                mappings.append((path, workspace / "temp" / path.name))
            elif (
                path.is_file()
                and path.name.startswith("morningtalk-pilot-")
                and path.suffix.casefold() == ".lock"
            ):
                mappings.append((path, workspace / "temp" / path.name))
    return mappings


def build_plan(
    mappings: Iterable[tuple[Path, Path]],
) -> list[MigrationEntry]:
    entries: list[MigrationEntry] = []
    seen_destinations: set[Path] = set()
    for source_root, destination_root in mappings:
        source_root = source_root.resolve()
        destination_root = destination_root.resolve()
        source_is_file = source_root.is_file()
        for source in _iter_source_files(source_root):
            relative = (
                Path()
                if source_is_file
                else source.relative_to(source_root)
            )
            destination = (
                destination_root
                if source_is_file
                else (destination_root / relative).resolve()
            )
            if destination in seen_destinations:
                raise MigrationError(f"DuplicateMigrationDestination:{destination}")
            seen_destinations.add(destination)
            stat = source.stat()
            entries.append(
                MigrationEntry(
                    source=source,
                    destination=destination,
                    size=stat.st_size,
                    sha256=sha256_file(source),
                    modified_ns=stat.st_mtime_ns,
                )
            )
    return sorted(entries, key=lambda entry: str(entry.destination).casefold())


def _validate_destinations(entries: list[MigrationEntry]) -> dict[Path, str]:
    states: dict[Path, str] = {}
    conflicts = []
    for entry in entries:
        if not entry.destination.exists():
            states[entry.destination] = "copy"
            continue
        if not entry.destination.is_file():
            conflicts.append(entry.destination)
            continue
        if sha256_file(entry.destination) != entry.sha256:
            conflicts.append(entry.destination)
        else:
            states[entry.destination] = "already_verified"
    if conflicts:
        joined = ", ".join(str(path) for path in conflicts[:5])
        raise MigrationError(f"DestinationContentConflict:{joined}")
    return states


def copy_verified(entries: list[MigrationEntry]) -> dict[str, int]:
    states = _validate_destinations(entries)
    copied = 0
    already_verified = 0
    total_bytes = 0
    for entry in entries:
        state = states[entry.destination]
        if state == "already_verified":
            already_verified += 1
            total_bytes += entry.size
            continue
        if sha256_file(entry.source) != entry.sha256:
            raise MigrationError(f"SourceChangedSincePreflight:{entry.source}")
        entry.destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = entry.destination.with_name(
            f".{entry.destination.name}.{os.getpid()}.migration.tmp"
        )
        if temporary.exists():
            raise MigrationError(f"TemporaryDestinationAlreadyExists:{temporary}")
        try:
            shutil.copy2(entry.source, temporary)
            if sha256_file(temporary) != entry.sha256:
                raise MigrationError(f"CopiedFileHashMismatch:{entry.destination}")
            if entry.destination.exists():
                if sha256_file(entry.destination) != entry.sha256:
                    raise MigrationError(f"DestinationContentConflict:{entry.destination}")
                temporary.unlink()
                already_verified += 1
                total_bytes += entry.size
                continue
            try:
                os.rename(temporary, entry.destination)
            except FileExistsError:
                if sha256_file(entry.destination) != entry.sha256:
                    raise MigrationError(
                        f"DestinationContentConflict:{entry.destination}"
                    )
                already_verified += 1
                total_bytes += entry.size
                continue
            shutil.copystat(entry.source, entry.destination)
            if sha256_file(entry.destination) != entry.sha256:
                raise MigrationError(f"DestinationHashMismatch:{entry.destination}")
            copied += 1
            total_bytes += entry.size
        finally:
            temporary.unlink(missing_ok=True)
    return {
        "copied_files": copied,
        "already_verified_files": already_verified,
        "total_files": len(entries),
        "total_bytes": total_bytes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--source-runtime-root",
        type=Path,
        default=DEFAULT_LEGACY_RUNTIME_ROOT,
    )
    parser.add_argument(
        "--source-temp-root",
        type=Path,
        default=DEFAULT_TEMP_RUNTIME_ROOT,
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=DEFAULT_WORKSPACE_ROOT,
    )
    args = parser.parse_args(argv)
    if not args.source_runtime_root.is_dir():
        raise MigrationError(f"LegacyRuntimeRootUnavailable:{args.source_runtime_root}")
    if args.workspace_root.resolve() != DEFAULT_WORKSPACE_ROOT.resolve():
        # A supplied root is honored by the environment-backed application config.
        if os.getenv(WORKSPACE_ENV_VAR, "").strip():
            raise MigrationError("WorkspaceRootArgumentConflictsWithEnvironment")
    mappings = default_sources(
        PROJECT_ROOT,
        args.source_runtime_root,
        args.source_temp_root,
    )
    mappings[0] = (args.source_runtime_root, args.workspace_root)
    for index in range(1, len(mappings)):
        source, destination = mappings[index]
        mappings[index] = (
            source,
            args.workspace_root / destination.relative_to(DEFAULT_WORKSPACE_ROOT),
        )
    entries = build_plan(mappings)
    states = _validate_destinations(entries)
    totals = {
        "files": len(entries),
        "bytes": sum(entry.size for entry in entries),
        "already_verified": sum(value == "already_verified" for value in states.values()),
        "to_copy": sum(value == "copy" for value in states.values()),
    }
    print(
        f"MODE={'dry-run' if args.dry_run else 'copy'} "
        f"FILES={totals['files']} BYTES={totals['bytes']} "
        f"ALREADY_VERIFIED={totals['already_verified']} TO_COPY={totals['to_copy']}"
    )
    if args.dry_run:
        if args.verbose:
            for entry in entries:
                print(
                    f"{entry.sha256} {entry.size} "
                    f"{entry.source} -> {entry.destination}"
                )
        return 0
    result = copy_verified(entries)
    for directory in (
        "previews",
        "production_batches",
        "reports",
        "temp",
        "logs",
    ):
        (args.workspace_root / directory).mkdir(parents=True, exist_ok=True)
    print(
        f"COPIED={result['copied_files']} "
        f"ALREADY_VERIFIED={result['already_verified_files']} "
        f"FILES={result['total_files']} BYTES={result['total_bytes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
