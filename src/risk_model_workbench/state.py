"""Workspace state management for workflow executions.

New workspaces are model versions under ``versions/<version_id>/``. The legacy
``runs/<run_id>/`` layout remains readable for compatibility.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from risk_model_workbench.agent.workspace_store import WorkspaceStore, tracked_payload
from risk_model_workbench.registry import register_artifact as registry_register_artifact


DEFAULT_STAGES = [
    "validate_config",
    "sample_check",
    "feature_metadata",
    "feature_prescreen",
    "build_wide_sql",
    "feature_refine",
    "train_baseline",
    "evaluate",
    "compare",
    "report",
]


def run_dir(project_dir: str | Path, run_id: str) -> Path:
    return Path(project_dir).resolve() / "runs" / run_id


def version_dir(project_dir: str | Path, version_id: str) -> Path:
    return Path(project_dir).resolve() / "versions" / version_id


def state_path(run_path: str | Path) -> Path:
    workspace = Path(run_path)
    version_state = workspace / "version_state.yml"
    if version_state.exists():
        return version_state
    return workspace / "run_state.yml"


def version_state_path(version_path: str | Path) -> Path:
    return Path(version_path) / "version_state.yml"


def create_run_state(
    project_dir: str | Path,
    *,
    run_id: str,
    workflow: str,
    stages: list[str] | None = None,
    status: str = "running",
) -> dict[str, Any]:
    selected_stages = stages or DEFAULT_STAGES
    return {
        "run_id": run_id,
        "project": str(Path(project_dir)),
        "workflow": workflow,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "current_stage": selected_stages[0] if selected_stages else None,
        "stages": {stage: {"status": "pending", "artifacts": []} for stage in selected_stages},
        "decisions": [],
    }


def create_version_state(
    project_dir: str | Path,
    *,
    version_id: str,
    workflow: str,
    stages: list[str] | None = None,
    status: str = "running",
    source_type: str = "workbench",
    managed_by: str = "workbench",
    lineage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected_stages = stages or DEFAULT_STAGES
    state: dict[str, Any] = {
        "version_id": version_id,
        "project": str(Path(project_dir)),
        "workflow": workflow,
        "source_type": source_type,
        "managed_by": managed_by,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "current_stage": selected_stages[0] if selected_stages else None,
        "stages": {stage: {"status": "pending", "artifacts": []} for stage in selected_stages},
        "decisions": [],
    }
    if lineage:
        state["lineage"] = lineage
    return state


def load_run_state(run_path: str | Path) -> dict[str, Any]:
    path = state_path(run_path)
    return tracked_payload(WorkspaceStore(run_path).read_yaml(path.name))


def save_run_state(run_path: str | Path, state: dict[str, Any]) -> Path:
    path = state_path(run_path)
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    store = WorkspaceStore(run_path)
    expected_revision = getattr(state, "store_revision", store.read_yaml(path.name).revision)
    revision = store.write_yaml(path.name, dict(state), expected_revision)
    if hasattr(state, "store_revision"):
        state.store_revision = revision
    return path


def save_version_state(version_path: str | Path, state: dict[str, Any]) -> Path:
    path = version_state_path(version_path)
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    store = WorkspaceStore(version_path)
    expected_revision = getattr(state, "store_revision", store.read_yaml("version_state.yml").revision)
    revision = store.write_yaml("version_state.yml", dict(state), expected_revision)
    if hasattr(state, "store_revision"):
        state.store_revision = revision
    return path


def workspace_id(state: dict[str, Any], fallback: str = "") -> str:
    return str(state.get("version_id") or state.get("run_id") or fallback)


def _emit_progress_safely(run_path: str | Path, stage: str, event: str, *, reason: str = "", scaffold: bool = False) -> None:
    try:
        from risk_model_workbench.progress import emit_stage_done, emit_stage_failed, emit_stage_started

        if event == "started":
            emit_stage_started(run_path, stage)
        elif event == "done":
            emit_stage_done(run_path, stage, scaffold=scaffold)
        elif event == "failed":
            emit_stage_failed(run_path, stage, reason)
    except Exception:
        return


def _ensure_stage(state: dict[str, Any], stage: str) -> dict[str, Any]:
    stages = state.setdefault("stages", {})
    return stages.setdefault(stage, {"status": "pending", "artifacts": []})


def mark_stage_started(run_path: str | Path, stage: str) -> dict[str, Any]:
    state = load_run_state(run_path)
    stage_state = _ensure_stage(state, stage)
    stage_state["status"] = "running"
    stage_state["started_at"] = datetime.now().isoformat(timespec="seconds")
    state["status"] = "running"
    state["current_stage"] = stage
    save_run_state(run_path, state)
    _emit_progress_safely(run_path, stage, "started")
    return load_run_state(run_path)


def mark_stage_done(run_path: str | Path, stage: str, *, scaffold: bool = False) -> dict[str, Any]:
    state = load_run_state(run_path)
    stage_state = _ensure_stage(state, stage)
    stage_state["status"] = "scaffold" if scaffold else "done"
    stage_state["finished_at"] = datetime.now().isoformat(timespec="seconds")
    stage_statuses = [item.get("status") for item in state.get("stages", {}).values()]
    if str(state.get("workflow", "")).startswith("imported"):
        state["status"] = "imported"
    elif stage_statuses and all(status in {"done", "scaffold"} for status in stage_statuses):
        state["status"] = "done"
    else:
        state["status"] = "running"
    state["current_stage"] = stage
    save_run_state(run_path, state)
    _emit_progress_safely(run_path, stage, "done", scaffold=scaffold)
    return load_run_state(run_path)


def mark_stage_failed(run_path: str | Path, stage: str, reason: str) -> dict[str, Any]:
    state = load_run_state(run_path)
    stage_state = _ensure_stage(state, stage)
    stage_state["status"] = "failed"
    stage_state["reason"] = reason
    stage_state["finished_at"] = datetime.now().isoformat(timespec="seconds")
    state["status"] = "failed"
    state["current_stage"] = stage
    save_run_state(run_path, state)
    _emit_progress_safely(run_path, stage, "failed", reason=reason)
    return load_run_state(run_path)


def register_artifact(
    run_path: str | Path,
    stage: str,
    artifact: str | Path,
    *,
    kind: str = "file",
    source: str = "generated",
    description: str = "",
) -> dict[str, Any]:
    state = load_run_state(run_path)
    transaction_id = f"txn_{uuid4().hex}"
    entry = registry_register_artifact(
        run_path,
        artifact,
        stage=stage,
        kind=kind,
        source=source,
        description=description,
        transaction_id=transaction_id,
    )
    stage_state = _ensure_stage(state, stage)
    artifacts = stage_state.setdefault("artifacts", [])
    if entry["path"] not in artifacts:
        artifacts.append(entry["path"])
    state["transaction_id"] = transaction_id
    save_run_state(run_path, state)
    return entry


def pair_audit_artifact_transaction(
    run_path: str | Path,
    artifact: str | Path,
    *,
    stage: str = "agent_runtime",
    description: str = "",
) -> dict[str, Any]:
    """Pair an audit-only manifest entry without adding a workflow stage.

    Agent runtime receipts are closure evidence, not domain workflow stages.
    The shared transaction id makes a crash between the two writes detectable
    and replayable while preserving the workflow's declared stage set.
    """
    state = load_run_state(run_path)
    transaction_id = f"txn_{uuid4().hex}"
    entry = registry_register_artifact(
        run_path,
        artifact,
        stage=stage,
        kind="audit",
        source="generated",
        description=description,
        transaction_id=transaction_id,
    )
    state["transaction_id"] = transaction_id
    save_run_state(run_path, state)
    return entry


def append_decision(run_path: str | Path, *, stage: str, decision: str, reason: str) -> Path:
    state = load_run_state(run_path)
    item = {
        "stage": stage,
        "decision": decision,
        "reason": reason,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    state.setdefault("decisions", []).append(item)
    save_run_state(run_path, state)

    log_path = Path(run_path) / "audit" / "decision_log.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"- {item['created_at']} [{stage}] {decision}: {reason}\n")
    return log_path
