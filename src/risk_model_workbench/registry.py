"""Artifact registry helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from risk_model_workbench.agent.workspace_store import WorkspaceStore
from risk_model_workbench.manifest import describe_file


def manifest_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "audit" / "artifact_manifest.json"


def load_artifact_manifest(run_dir: str | Path) -> dict[str, Any]:
    path = manifest_path(run_dir)
    if not path.exists():
        return {"version": 1, "artifacts": []}
    return WorkspaceStore(run_dir).read_json("audit/artifact_manifest.json").payload


def save_artifact_manifest(
    run_dir: str | Path,
    manifest: dict[str, Any],
    *,
    transaction_id: str = "",
) -> Path:
    path = manifest_path(run_dir)
    manifest.setdefault("version", 1)
    manifest.setdefault("artifacts", [])
    manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["transaction_id"] = transaction_id or _new_unpaired_transaction_id()
    WorkspaceStore(run_dir).update_json("audit/artifact_manifest.json", lambda _current: dict(manifest))
    return path


def register_artifact(
    run_dir: str | Path,
    artifact: str | Path,
    *,
    stage: str,
    kind: str = "file",
    source: str = "generated",
    description: str = "",
    transaction_id: str = "",
) -> dict[str, Any]:
    """Register an artifact relative to the run directory when possible."""
    run_path = Path(run_dir).resolve()
    artifact_path = Path(artifact)
    if not artifact_path.is_absolute():
        artifact_path = run_path / artifact_path
    artifact_path = artifact_path.resolve()

    if artifact_path.exists() and artifact_path.is_file():
        entry = describe_file(artifact_path, run_path)
    else:
        try:
            display_path = artifact_path.relative_to(run_path)
        except ValueError:
            display_path = artifact_path
        entry = {"path": str(display_path), "exists": artifact_path.exists()}

    entry.update(
        {
            "stage": stage,
            "kind": kind,
            "source": source,
            "description": description,
            "registered_at": datetime.now().isoformat(timespec="seconds"),
        }
    )

    resolved_transaction_id = transaction_id or _new_unpaired_transaction_id()

    def update_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
        manifest.setdefault("version", 1)
        artifacts = [
            item
            for item in manifest.get("artifacts", [])
            if not (item.get("path") == entry["path"] and item.get("stage") == stage)
        ]
        artifacts.append(entry)
        manifest["artifacts"] = artifacts
        manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
        manifest["transaction_id"] = resolved_transaction_id
        return manifest

    WorkspaceStore(run_path).update_json("audit/artifact_manifest.json", update_manifest)
    return entry


def _new_unpaired_transaction_id() -> str:
    """Identify a manifest mutation that has no matching version-state write."""
    return f"txn_unpaired_{uuid4().hex}"
