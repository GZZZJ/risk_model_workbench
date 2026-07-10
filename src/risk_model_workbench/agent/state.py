"""Agent state files for version-scoped deterministic execution."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from risk_model_workbench.agent.transitions import apply_task_transition, apply_transition
from risk_model_workbench.agent.workspace_store import WorkspaceStore, tracked_payload


AGENT_STATE_VERSION = 2
LEGACY_AGENT_STATE_VERSION = 1
TERMINAL_TASK_STATUSES = {"done", "scaffold", "skipped"}


def agent_state_path(workspace: str | Path) -> Path:
    return Path(workspace) / "audit" / "agent_state.yml"


def load_agent_state(workspace: str | Path) -> dict[str, Any]:
    path = agent_state_path(workspace)
    if not path.exists():
        raise FileNotFoundError(f"agent_state.yml not found: {path}")
    payload = tracked_payload(WorkspaceStore(workspace).read_yaml("audit/agent_state.yml"))
    payload.setdefault("version", LEGACY_AGENT_STATE_VERSION)
    payload.setdefault("tasks", [])
    payload.setdefault("blocker", {})
    return payload


def save_agent_state(workspace: str | Path, state: dict[str, Any]) -> Path:
    path = agent_state_path(workspace)
    state.setdefault("version", LEGACY_AGENT_STATE_VERSION)
    state["updated_at"] = _now()
    store = WorkspaceStore(workspace)
    expected_revision = getattr(state, "store_revision", store.read_yaml("audit/agent_state.yml").revision)
    revision = store.write_yaml("audit/agent_state.yml", dict(state), expected_revision)
    if hasattr(state, "store_revision"):
        state.store_revision = revision
    return path


def init_agent_state(
    workspace: str | Path,
    *,
    project: str,
    version_id: str,
    agent_plan: dict[str, Any],
    mode: str = "semi_autonomous",
) -> dict[str, Any]:
    now = _now()
    plan_version = int(agent_plan.get("version") or LEGACY_AGENT_STATE_VERSION)
    state_version = AGENT_STATE_VERSION if plan_version >= 2 else LEGACY_AGENT_STATE_VERSION
    tasks = []
    for task in agent_plan.get("tasks", []) or []:
        derived = task.get("derived_metadata") if isinstance(task.get("derived_metadata"), dict) else {}
        tasks.append(
            {
                "task_id": task.get("task_id", ""),
                "status": "pending",
                "action_id": derived.get("action_id") or task.get("action_id", ""),
                "tool_name": (task.get("invocation") or {}).get("tool_name") or task.get("tool_name", ""),
                "permission": derived.get("permission") or task.get("permission", ""),
                "depends_on": list(task.get("depends_on") or []),
                "started_at": "",
                "finished_at": "",
                "message": "",
            }
        )
    state = {
        "version": state_version,
        "agent_run_id": f"agent_{now.replace(':', '').replace('-', '')}",
        "mode": mode,
        "status": "draft",
        "project": project,
        "version_id": version_id,
        "plan_id": agent_plan.get("plan_id", ""),
        "plan_hash": agent_plan.get("plan_hash", "") if state_version >= 2 else "",
        "registry_digest": agent_plan.get("registry_digest", "") if state_version >= 2 else "",
        "current_task": "",
        "policy": {
            "safe_permissions": ["read_only", "writes_run"],
            "approval_required_permissions": ["dp_sql_pull"],
            "blocked_permissions": ["external_data"],
            "blocked_flags": ["--force", "--skip-split-check"],
        },
        "blocker": {},
        "tasks": tasks,
        "created_at": now,
        "updated_at": now,
    }
    save_agent_state(workspace, state)
    return state


def task_status_map(state: dict[str, Any]) -> dict[str, str]:
    return {str(task.get("task_id")): str(task.get("status")) for task in state.get("tasks", []) or []}


def mark_task_running(workspace: str | Path, task_id: str) -> dict[str, Any]:
    state = load_agent_state(workspace)
    task = _task(state, task_id)
    if int(state.get("version") or 1) >= 2:
        task.update(apply_task_transition(task, "start" if task.get("status") == "pending" else "resume", "running"))
        event = "start" if state.get("status") == "draft" else "run_task"
        state = apply_transition(state, event, {"target_state": "running"})
        task = _task(state, task_id)
    else:
        task["status"] = "running"
    task["started_at"] = _now()
    state["status"] = "running"
    state["current_task"] = task_id
    state["blocker"] = {}
    save_agent_state(workspace, state)
    return state


def mark_task_done(workspace: str | Path, task_id: str, *, scaffold: bool = False, message: str = "") -> dict[str, Any]:
    state = load_agent_state(workspace)
    task = _task(state, task_id)
    target = "scaffold" if scaffold else "done"
    if int(state.get("version") or 1) >= 2:
        task.update(apply_task_transition(task, "record_scaffold" if scaffold else "complete", target))
    else:
        task["status"] = target
    task["finished_at"] = _now()
    task["message"] = message
    state["current_task"] = task_id
    _refresh_overall_status(state)
    save_agent_state(workspace, state)
    return state


def mark_task_failed(workspace: str | Path, task_id: str, *, reason: str, status: str = "failed") -> dict[str, Any]:
    state = load_agent_state(workspace)
    task = _task(state, task_id)
    if int(state.get("version") or 1) >= 2:
        task.update(apply_task_transition(task, "fail", status))
        state = apply_transition(state, "fail", {"target_state": "failed"})
        task = _task(state, task_id)
    else:
        task["status"] = status
    task["finished_at"] = _now()
    task["message"] = reason
    state["status"] = "failed"
    state["current_task"] = task_id
    state["blocker"] = {"reason": reason, "task_id": task_id}
    save_agent_state(workspace, state)
    return state


def pause_agent(
    workspace: str | Path,
    *,
    status: str,
    reason: str,
    task_id: str,
    approval_id: str = "",
    command_hash: str = "",
    advisor_request: str = "",
    advisor_request_id: str = "",
) -> dict[str, Any]:
    state = load_agent_state(workspace)
    if task_id:
        try:
            task = _task(state, task_id)
            if int(state.get("version") or 1) >= 2:
                task.update(apply_task_transition(task, "pause", "paused"))
            else:
                task["status"] = "paused"
        except KeyError:
            pass
    blocker = {
        "reason": reason,
        "task_id": task_id,
        "created_at": _now(),
        "next_safe_action": _next_safe_action(status),
    }
    if approval_id:
        blocker["approval_id"] = approval_id
    if command_hash:
        blocker["command_hash"] = command_hash
    if advisor_request:
        blocker["advisor_request"] = advisor_request
    if advisor_request_id:
        blocker["advisor_request_id"] = advisor_request_id
    blocker["blocker_type"] = {
        "waiting_for_approval": "approval",
        "waiting_for_advisor": "advisor",
        "waiting_for_user": "user",
        "reconciliation_required": "reconciliation",
        "blocked": "dependency",
    }.get(status, "dependency")
    blocker["blocker_id"] = (
        approval_id
        or advisor_request_id
        or f"{blocker['blocker_type']}:{task_id or 'agent'}:{command_hash[:12] or reason}"
    )
    if int(state.get("version") or 1) >= 2:
        event = {
            "waiting_for_approval": "request_approval",
            "waiting_for_advisor": "request_advisor",
            "blocked": "block",
        }.get(status)
        if event is None:
            raise ValueError(f"unsupported pause status for v2: {status}")
        state = apply_transition(state, event, {"target_state": status, "blocker": blocker})
    else:
        state["status"] = status
    state["current_task"] = task_id
    state["blocker"] = blocker
    save_agent_state(workspace, state)
    return state


def reset_paused_task(workspace: str | Path, task_id: str) -> dict[str, Any]:
    state = load_agent_state(workspace)
    task = _task(state, task_id)
    if int(state.get("version") or 1) >= 2:
        if state.get("status") in {
            "waiting_for_approval",
            "waiting_for_advisor",
            "waiting_for_user",
            "blocked",
            "reconciliation_required",
        }:
            raise ValueError("v2 blockers require a dedicated evidence-consuming transition")
        if task.get("status") == "paused":
            task.update(apply_task_transition(task, "resume", "pending"))
        event = {
            "waiting_for_approval": "consume_approval",
            "waiting_for_advisor": "consume_response",
            "blocked": "resolve_blocker",
        }.get(str(state.get("status") or ""))
        if event:
            state = apply_transition(state, event, {"target_state": "running"})
    else:
        if task.get("status") == "paused":
            task["status"] = "pending"
        state["blocker"] = {}
        if state.get("status") in {"waiting_for_approval", "waiting_for_advisor", "blocked"}:
            state["status"] = "running"
    save_agent_state(workspace, state)
    return state


def requeue_interrupted_task(workspace: str | Path, task_id: str, *, attempt_id: str) -> dict[str, Any]:
    """Requeue a provably safe interrupted attempt without erasing its identity."""
    state = load_agent_state(workspace)
    task = _task(state, task_id)
    if task.get("status") != "running":
        raise ValueError(f"interrupted task is not running: {task_id}")
    if str(task.get("attempt_id") or "") != attempt_id:
        raise ValueError(f"interrupted attempt does not match task: {attempt_id}")
    task["status"] = "pending"
    task["recovered_attempt_id"] = attempt_id
    task["attempt_finished_at"] = _now()
    task["message"] = "interrupted attempt safely requeued"
    state["status"] = "running"
    state["current_task"] = ""
    state["blocker"] = {}
    save_agent_state(workspace, state)
    return state


def _refresh_overall_status(state: dict[str, Any]) -> None:
    if int(state.get("version") or 1) >= 2:
        state["status"] = "running"
        return
    statuses = [str(task.get("status")) for task in state.get("tasks", []) or []]
    if statuses and all(status in TERMINAL_TASK_STATUSES for status in statuses):
        state["status"] = "done_with_gaps" if "scaffold" in statuses else "done"
        return
    state["status"] = "running"


def _task(state: dict[str, Any], task_id: str) -> dict[str, Any]:
    for task in state.get("tasks", []) or []:
        if task.get("task_id") == task_id:
            return task
    raise KeyError(f"unknown task_id: {task_id}")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _next_safe_action(status: str) -> dict[str, str]:
    action, evidence = {
        "waiting_for_approval": ("consume_bound_approval", "approved one-time approval record"),
        "waiting_for_advisor": ("consume_advisor_response", "advisor consumption receipt"),
        "blocked": ("resolve_dependency", "registered dependency evidence"),
    }.get(status, ("inspect_failure", "operator diagnosis"))
    return {"action": action, "required_evidence": evidence}
