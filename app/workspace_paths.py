"""Centralized paths for MorningTalk's private runtime workspace."""

import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ENV_VAR = "MORNINGTALK_WORKSPACE_ROOT"
DEFAULT_WORKSPACE_ROOT = PROJECT_ROOT / "workspace"


def get_workspace_root() -> Path:
    override = os.getenv(WORKSPACE_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_WORKSPACE_ROOT.resolve()


WORKSPACE_ROOT = get_workspace_root()
DEFAULT_PREVIEW_ROOT = WORKSPACE_ROOT / "previews"
DEFAULT_PRODUCTION_BATCH_ROOT = WORKSPACE_ROOT / "production_batches"
DEFAULT_REPORT_ROOT = WORKSPACE_ROOT / "reports"
DEFAULT_TEMP_ROOT = WORKSPACE_ROOT / "temp"
DEFAULT_LOG_ROOT = WORKSPACE_ROOT / "logs"
DEFAULT_REVIEW_ROOT = DEFAULT_PREVIEW_ROOT / "reviews"
DEFAULT_BENCHMARK_REPORT_ROOT = DEFAULT_REPORT_ROOT / "benchmark"
DEFAULT_PHASE10_6_REPORT_ROOT = DEFAULT_REPORT_ROOT / "phase10_6"
DEFAULT_PHASE10_7_REPORT_ROOT = DEFAULT_REPORT_ROOT / "phase10_7"


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def rebase_runtime_paths(
    value: Any,
    *,
    legacy_root: Path,
    workspace_root: Path | None = None,
) -> Any:
    """Rebase stored runtime path values while preserving non-path history."""
    legacy_root = legacy_root.resolve()
    workspace_root = (workspace_root or WORKSPACE_ROOT).resolve()

    def rebase_path(path_value: str) -> str:
        path = Path(path_value)
        if not path.is_absolute():
            return path_value
        try:
            relative = path.resolve().relative_to(legacy_root)
        except ValueError:
            return path_value
        return str((workspace_root / relative).resolve())

    def visit(item: Any, key: str | None = None) -> Any:
        if isinstance(item, dict):
            return {child_key: visit(child, child_key) for child_key, child in item.items()}
        if isinstance(item, list):
            return [visit(child, key) for child in item]
        is_path_field = (
            key == "path"
            or key == "root"
            or (key is not None and key.endswith(("_path", "_root", "_dir", "_directory")))
        )
        if is_path_field and isinstance(item, str):
            return rebase_path(item)
        return item

    return visit(value)


def rebase_manifest_runtime_paths(
    payload: dict[str, Any],
    manifest_path: Path,
    *,
    workspace_root: Path | None = None,
) -> dict[str, Any]:
    """Rebase copied manifests only when they are loaded from this workspace."""
    workspace_root = (workspace_root or WORKSPACE_ROOT).resolve()
    if not is_within(manifest_path, workspace_root):
        return payload
    paths = payload.get("paths")
    if not isinstance(paths, dict):
        return payload
    preview_root = paths.get("preview_root")
    if not isinstance(preview_root, str):
        return payload
    legacy_preview_root = Path(preview_root)
    if not legacy_preview_root.is_absolute():
        return payload
    legacy_root = legacy_preview_root.parent
    if legacy_root.resolve() == workspace_root:
        return payload
    return rebase_runtime_paths(
        payload,
        legacy_root=legacy_root,
        workspace_root=workspace_root,
    )
