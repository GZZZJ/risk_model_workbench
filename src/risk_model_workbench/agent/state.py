"""Agent state files for version-scoped deterministic execution."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


AGENT_STATE_VERSION = 1
TERMINAL_TASK_STATUSES = {"done", "scaffold", "skipped"}


def agent_state_path(workspace: str | Path) -> Path:
    return Path(workspace) / "audit" / "agent_state.yml"


def load_agent_state(workspace: str | Path) -> dict[str, Any]:
    path = agent_state_path(workspace)
    if not path.exists():
        raise FileNotFoundError(f"agent_state.yml not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    payload.setdefault("version", AGENT_STATE_VERSION)
    payload.setdefault("tasks", [])
    payload.setdefault("blocker", {})
    return payload


def save_agent_state(workspace: str | Path, state: dict[str, Any]) -> Path:
    path = agent_state_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    state["version"] = AGENT_STATE_VERSION
    state["updated_at"] = _now()
    path.write_text(yaml.safe_dump(state, allow_unicode=True, sort_keys=False), encoding="utf-8")
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
    tasks = []
    for task in agent_plan.get("tasks", []) or []:
        tasks.append(
            {
                "task_id": task.get("task_id", ""),
                "status": "pending",
                "action_id": task.get("action_id", ""),
                "tool_name": task.get("tool_name", ""),
                "permission": task.get("permission", ""),
                "depends_on": list(task.get("depends_on") or []),
                "started_at": "",
                "finished_at": "",
                "message": "",
            }
        )
    state = {
        "version": AGENT_STATE_VERSION,
        "agent_run_id": f"agent_{now.replace(':', '').replace('-', '')}",
        "mode": mode,
        "status": "draft",
        "project": project,
        "version_id": version_id,
        "plan_id": agent_plan.get("plan_id", ""),
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
    task["status"] = "scaffold" if scaffold else "done"
    task["finished_at"] = _now()
    task["message"] = message
    state["current_task"] = task_id
    _refresh_overall_status(state)
    save_agent_state(workspace, state)
    return state


def mark_task_failed(workspace: str | Path, task_id: str, *, reason: str, status: str = "failed") -> dict[str, Any]:
    state = load_agent_state(workspace)
    task = _task(state, task_id)
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
            _task(state, task_id)["status"] = "paused"
        except KeyError:
            pass
    blocker = {
        "reason": reason,
        "task_id": task_id,
        "created_at": _now(),
    }
    if approval_id:
        blocker["approval_id"] = approval_id
    if command_hash:
        blocker["command_hash"] = command_hash
    if advisor_request:
        blocker["advisor_request"] = advisor_request
    if advisor_request_id:
        blocker["advisor_request_id"] = advisor_request_id
    state["status"] = status
    state["current_task"] = task_id
    state["blocker"] = blocker
    save_agent_state(workspace, state)
    return state


def reset_paused_task(workspace: str | Path, task_id: str) -> dict[str, Any]:
    state = load_agent_state(workspace)
    task = _task(state, task_id)
    if task.get("status") == "paused":
        task["status"] = "pending"
    state["blocker"] = {}
    if state.get("status") in {"waiting_for_approval", "waiting_for_advisor", "blocked"}:
        state["status"] = "running"
    save_agent_state(workspace, state)
    return state


def _refresh_overall_status(state: dict[str, Any]) -> None:
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
