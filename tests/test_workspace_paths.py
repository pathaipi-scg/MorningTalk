from pathlib import Path

import app.workspace_paths as workspace_paths


def test_workspace_defaults_to_project_workspace_not_local_app_data(monkeypatch):
    monkeypatch.delenv(workspace_paths.WORKSPACE_ENV_VAR, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(Path("C:/unrelated/local-data")))

    assert workspace_paths.get_workspace_root() == (
        workspace_paths.PROJECT_ROOT / "workspace"
    ).resolve()


def test_workspace_environment_override_is_used(monkeypatch, tmp_path):
    override = tmp_path / "custom-workspace"
    monkeypatch.setenv(workspace_paths.WORKSPACE_ENV_VAR, str(override))

    assert workspace_paths.get_workspace_root() == override.resolve()


def test_internal_runtime_defaults_share_the_central_workspace_root(monkeypatch):
    from app import benchmark, monitor, phase10_6, phase10_7, phase9, production_batch

    monkeypatch.setenv(
        workspace_paths.WORKSPACE_ENV_VAR,
        str(workspace_paths.WORKSPACE_ROOT),
    )
    roots = (
        phase9.PREVIEW_DIRECTORY,
        phase10_6.DEFAULT_PREVIEW_ROOT,
        phase10_6.DEFAULT_REVIEW_ROOT,
        phase10_6.DEFAULT_EVALUATION_DIR,
        phase10_7.DEFAULT_MANIFEST_PATH,
        phase10_7.DEFAULT_STAGE_ROOT,
        production_batch.PRIVATE_ROOT,
        benchmark.DEFAULT_PRIVATE_ROOT,
        benchmark.DEFAULT_REPORT_ROOT,
        monitor.runtime_directory(),
    )

    assert all(
        workspace_paths.is_within(root, workspace_paths.WORKSPACE_ROOT)
        for root in roots
    )


def test_migrated_manifest_paths_rebase_without_changing_request_history(
    monkeypatch, tmp_path
):
    old_root = tmp_path / "old" / "MorningTalk"
    new_root = tmp_path / "project" / "workspace"
    monkeypatch.setattr(workspace_paths, "WORKSPACE_ROOT", new_root)
    manifest_path = new_root / "production_batches" / "batch-one" / "manifest.json"
    payload = {
        "paths": {
            "preview_root": str(old_root / "previews"),
            "batch_root": str(old_root / "production_batches" / "batch-one"),
        },
        "recordings": [
            {
                "source": r"D:\KM\Vault\Record File\meeting.mp3",
                "phase9_artifact": {"path": str(old_root / "previews" / "transcript.json")},
            }
        ],
        "request_attempts": [
            {
                "attempt_id": "request-unknown",
                "status": "unknown",
                "raw_response_path": None,
            }
        ],
    }

    rebased = workspace_paths.rebase_manifest_runtime_paths(payload, manifest_path)

    assert rebased["paths"]["batch_root"] == str(
        new_root / "production_batches" / "batch-one"
    )
    assert rebased["recordings"][0]["phase9_artifact"]["path"] == str(
        new_root / "previews" / "transcript.json"
    )
    assert rebased["recordings"][0]["source"] == payload["recordings"][0]["source"]
    assert rebased["request_attempts"] == payload["request_attempts"]
