"""Deterministic execution loop for RMW Agent plans."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from risk_model_workbench.agent.advisor import advisor_request_is_answered, create_advisor_request
from risk_model_workbench.agent.plan import load_agent_plan
from risk_model_workbench.agent.policy import evaluate_task_policy
from risk_model_workbench.agent.state import (
    init_agent_state,
    load_agent_state,
    mark_task_done,
    mark_task_failed,
    mark_task_running,
    pause_agent,
    reset_paused_task,
    task_status_map,
)
from risk_model_workbench.agent.trace import append_trace
from risk_model_workbench.harness.actions import get_action_spec
from risk_model_workbench.project_state import audit_run
from risk_model_workbench.state import load_run_state
from risk_model_workbench.versioning import resolve_workspace_dir


Runner = Callable[[list[str]], int]


def run_agent(project_dir: str | Path, version_id: str, *, runner: Runner | None = None) -> dict[str, Any]:
    project_path = Path(project_dir)
    workspace = resolve_workspace_dir(project_path, version_id=version_id)
    plan = load_agent_plan(workspace)
    try:
        state = load_agent_state(workspace)
    except FileNotFoundError:
        state = init_agent_state(workspace, project=str(project_path), version_id=version_id, agent_plan=plan)
    runner = runner or _default_runner
    append_trace(workspace, "observation", {"summary": "Agent execution started.", "version_id": version_id})
    if state.get("status") == "waiting_for_advisor":
        blocker = state.get("blocker") if isinstance(state.get("blocker"), dict) else {}
        request_id = str(blocker.get("advisor_request_id") or "")
        if request_id and not advisor_request_is_answered(workspace, request_id):
            append_trace(
                workspace,
                "decision",
                {
                    "summary": f"Advisor response pending for {request_id}.",
                    "advisor_request_id": request_id,
                },
            )
            return state

    while True:
        state = load_agent_state(workspace)
        status_by_task = task_status_map(state)
        runnable = _next_runnable_task(plan, status_by_task)
        if runnable is None:
            final = _finalize_if_complete(project_path, workspace, version_id)
            append_trace(workspace, "result", {"summary": "Agent execution stopped.", "status": final.get("status")})
            return final

        if status_by_task.get(str(runnable.get("task_id"))) == "paused":
            reset_paused_task(workspace, str(runnable.get("task_id")))

        decision = evaluate_task_policy(runnable, workspace)
        if not decision.allowed:
            state = pause_agent(
                workspace,
                status=decision.status,
                reason=decision.reason,
                task_id=str(runnable.get("task_id") or ""),
                approval_id=decision.approval_id,
                command_hash=decision.command_hash,
            )
            append_trace(
                workspace,
                "decision",
                {
                    "summary": f"Paused before task {runnable.get('task_id')}: {decision.reason}",
                    "task_id": runnable.get("task_id"),
                    "policy": decision.to_dict(),
                    "command": (runnable.get("command") or {}).get("args") or [],
                },
            )
            return state

        task_id = str(runnable.get("task_id") or "")
        args = list((runnable.get("command") or {}).get("args") or [])
        mark_task_running(workspace, task_id)
        append_trace(workspace, "action", {"summary": f"Executing task {task_id}.", "task_id": task_id, "command": args})
        code = runner(args)
        result = _stage_result(workspace, runnable)
        append_trace(
            workspace,
            "result",
            {
                "summary": f"Task {task_id} exited with code {code}.",
                "task_id": task_id,
                "exit_code": code,
                "stage_result": result,
            },
        )
        failure_code = str(result.get("failure_code") or "")
        stage_status = str(result.get("status") or "")
        if code == 0:
            mark_task_done(
                workspace,
                task_id,
                scaffold=stage_status == "scaffold" or failure_code == "scaffold_only",
                message=str(result.get("message") or ""),
            )
            continue
        if failure_code == "advisor_required":
            request = create_advisor_request(
                workspace,
                project_dir=project_path,
                version_id=version_id,
                task=runnable,
                reason="advisor_required",
                message=str(result.get("message") or ""),
            )
            state = pause_agent(
                workspace,
                status="waiting_for_advisor",
                reason="advisor_required",
                task_id=task_id,
                advisor_request=str(request.get("path") or ""),
                advisor_request_id=str(request.get("request_id") or ""),
            )
            append_trace(
                workspace,
                "decision",
                {
                    "summary": f"Paused for host-agent advisor on task {task_id}.",
                    "task_id": task_id,
                    "advisor_request": str(request.get("path") or ""),
                    "advisor_request_id": str(request.get("request_id") or ""),
                },
            )
            return state
        if failure_code == "sql_approval_required":
            decision = evaluate_task_policy({**runnable, "permission": "dp_sql_pull", "requires_approval": True}, workspace)
            state = pause_agent(
                workspace,
                status="waiting_for_approval",
                reason="sql_approval_required",
                task_id=task_id,
                approval_id=decision.approval_id,
                command_hash=decision.command_hash,
            )
            return state
        if failure_code == "scaffold_only" or stage_status == "scaffold":
            mark_task_done(workspace, task_id, scaffold=True, message=str(result.get("message") or "scaffold output"))
            continue
        return mark_task_failed(workspace, task_id, reason=failure_code or f"exit_code_{code}")


def _next_runnable_task(plan: dict[str, Any], status_by_task: dict[str, str]) -> dict[str, Any] | None:
    done_statuses = {"done", "scaffold", "skipped"}
    for task in plan.get("tasks", []) or []:
        task_id = str(task.get("task_id") or "")
        status = status_by_task.get(task_id, "pending")
        if status not in {"pending", "paused"}:
            continue
        deps = [str(dep) for dep in task.get("depends_on") or []]
        if all(status_by_task.get(dep) in done_statuses for dep in deps):
            return task
    return None


def _stage_result(workspace: Path, task: dict[str, Any]) -> dict[str, Any]:
    try:
        action = get_action_spec(str(task.get("action_id") or ""))
        stage = str(action.stage or "")
        state = load_run_state(workspace)
        stage_state = (state.get("stages") or {}).get(stage) or {}
        last = stage_state.get("last_result") if isinstance(stage_state.get("last_result"), dict) else {}
        return {
            "stage": stage,
            "status": stage_state.get("status", ""),
            "failure_code": stage_state.get("failure_code") or last.get("failure_code", ""),
            "message": last.get("message") or stage_state.get("reason", ""),
        }
    except Exception as exc:
        return {"stage": "", "status": "", "failure_code": "unknown", "message": str(exc)}


def _finalize_if_complete(project_dir: Path, workspace: Path, version_id: str) -> dict[str, Any]:
    state = load_agent_state(workspace)
    statuses = [str(task.get("status")) for task in state.get("tasks", []) or []]
    if statuses and all(status in {"done", "scaffold", "skipped"} for status in statuses):
        state["status"] = "done_with_gaps" if "scaffold" in statuses else "done"
        try:
            audit = audit_run(project_dir, version_id)
            if audit.get("verdict") != "complete" and state["status"] == "done":
                state["status"] = "done_with_gaps"
            state["latest_audit_verdict"] = audit.get("verdict", "")
        except Exception as exc:
            state["status"] = "done_with_gaps"
            state["latest_audit_error"] = str(exc)
        from risk_model_workbench.agent.state import save_agent_state

        save_agent_state(workspace, state)
    elif any(status in {"pending", "paused", "running"} for status in statuses):
        state["status"] = "blocked"
        state["blocker"] = {"reason": "no_runnable_task", "created_at": datetime.now().isoformat(timespec="seconds")}
        from risk_model_workbench.agent.state import save_agent_state

        save_agent_state(workspace, state)
    return state


def _default_runner(argv: list[str]) -> int:
    from risk_model_workbench.cli import main

    return main(argv)
