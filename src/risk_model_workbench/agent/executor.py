"""Deterministic execution loop for RMW Agent plans."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from risk_model_workbench.agent.advisor import advisor_request_is_answered, create_advisor_request
from risk_model_workbench.agent.plan import (
    invocation_for_task,
    load_agent_plan,
    rendered_argv_for_task,
    validate_agent_plan,
)
from risk_model_workbench.agent.policy import evaluate_task_policy
from risk_model_workbench.agent.state import (
    init_agent_state,
    load_agent_state,
    mark_task_done,
    mark_task_failed,
    mark_task_running,
    pause_agent,
    reset_paused_task,
    save_agent_state,
    task_status_map,
)
from risk_model_workbench.agent.trace import append_trace
from risk_model_workbench.agent.transitions import BLOCKER_STATES, apply_task_transition, apply_transition
from risk_model_workbench.harness.actions import get_action_spec
from risk_model_workbench.harness.errors import InvalidActionResultError, MissingActionResultError
from risk_model_workbench.harness.runtime import ActionAttempt, ActionResult, action_attempt, load_action_result
from risk_model_workbench.harness.tools import TOOL_REGISTRY, registry_digest
from risk_model_workbench.agent.workspace_store import WorkspaceStore
from risk_model_workbench.project_state import audit_run
from risk_model_workbench.state import load_run_state
from risk_model_workbench.versioning import resolve_workspace_dir


Runner = Callable[[list[str]], int]


def run_agent(project_dir: str | Path, version_id: str, *, runner: Runner | None = None) -> dict[str, Any]:
    project_path = Path(project_dir)
    workspace = resolve_workspace_dir(project_path, version_id=version_id)
    with WorkspaceStore(workspace).runner_lock():
        return _run_agent_locked(project_path, workspace, version_id, runner=runner)


def _run_agent_locked(
    project_path: Path,
    workspace: Path,
    version_id: str,
    *,
    runner: Runner | None,
) -> dict[str, Any]:
    plan = load_agent_plan(workspace)
    if int(plan.get("version") or 1) >= 2:
        errors = validate_agent_plan(
            plan,
            TOOL_REGISTRY,
            expected_project=project_path,
            expected_version_id=version_id,
        )
        if errors:
            raise ValueError("invalid agent plan: " + "; ".join(errors))
    try:
        state = load_agent_state(workspace)
    except FileNotFoundError:
        state = init_agent_state(workspace, project=str(project_path), version_id=version_id, agent_plan=plan)
    if int(plan.get("version") or 1) >= 2:
        if Path(str(state.get("project") or "")).resolve() != project_path.resolve():
            raise ValueError("agent state project does not match runtime project")
        if state.get("version_id") != version_id:
            raise ValueError("agent state version_id does not match runtime version")
        if state.get("plan_id") != plan.get("plan_id"):
            raise ValueError("agent state plan_id does not match bound plan")
        if state.get("plan_hash") != plan.get("plan_hash"):
            raise ValueError("agent state plan_hash does not match bound plan")
        if state.get("registry_digest") != registry_digest(TOOL_REGISTRY):
            raise ValueError("agent state registry_digest does not match current registry")
        if state.get("status") == "draft":
            state = apply_transition(state, "start", {"target_state": "running"})
            from risk_model_workbench.agent.state import save_agent_state

            save_agent_state(workspace, state)
    runner = runner or _default_runner
    append_trace(workspace, "observation", {"summary": "Agent execution started.", "version_id": version_id})
    if int(plan.get("version") or 1) >= 2 and state.get("status") in BLOCKER_STATES:
        append_trace(
            workspace,
            "decision",
            {
                "summary": "A dedicated consumption receipt is required before resume.",
                "status": state.get("status"),
            },
        )
        return state
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

        invocation = invocation_for_task(runnable, plan)
        args = rendered_argv_for_task(runnable, plan, TOOL_REGISTRY)
        decision = evaluate_task_policy(invocation, workspace, task_id=str(runnable.get("task_id") or ""))
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
                    "command": args,
                },
            )
            return state

        task_id = str(runnable.get("task_id") or "")
        spec = TOOL_REGISTRY[invocation.tool_name]
        attempt_id = f"attempt_{uuid4().hex}"
        mark_task_running(workspace, task_id)
        _record_attempt(workspace, task_id, attempt_id, invocation.digest())
        append_trace(
            workspace,
            "action",
            {
                "summary": f"Executing task {task_id}.",
                "task_id": task_id,
                "attempt_id": attempt_id,
                "invocation_hash": invocation.digest(),
                "command": args,
            },
        )
        if int(plan.get("version") or 1) >= 2:
            attempt = ActionAttempt(
                workspace=workspace,
                attempt_id=attempt_id,
                task_id=task_id,
                action_id=spec.action_id,
                invocation_hash=invocation.digest(),
                project=str(project_path),
                version_id=version_id,
            )
            runner_error = ""
            try:
                with action_attempt(attempt):
                    code = runner(args)
            except Exception as exc:
                code = 1
                runner_error = f"{type(exc).__name__}: {exc}"
            try:
                semantic = load_action_result(
                    workspace,
                    attempt_id,
                    task_id=task_id,
                    action_id=spec.action_id,
                    invocation_hash=invocation.digest(),
                    project=str(project_path),
                    version_id=version_id,
                )
            except MissingActionResultError as exc:
                append_trace(
                    workspace,
                    "result",
                    {
                        "summary": str(exc),
                        "task_id": task_id,
                        "attempt_id": attempt_id,
                        "exit_code": code,
                        "runner_error": runner_error,
                    },
                )
                return mark_task_failed(workspace, task_id, reason="missing_action_result")
            except InvalidActionResultError as exc:
                append_trace(
                    workspace,
                    "result",
                    {
                        "summary": str(exc),
                        "task_id": task_id,
                        "attempt_id": attempt_id,
                        "exit_code": code,
                        "runner_error": runner_error,
                    },
                )
                return mark_task_failed(workspace, task_id, reason="invalid_action_result")
            append_trace(
                workspace,
                "result",
                {
                    "summary": f"Task {task_id} emitted semantic status {semantic.status}.",
                    "task_id": task_id,
                    "attempt_id": attempt_id,
                    "exit_code": code,
                    "runner_error": runner_error,
                    "action_result": semantic.to_dict(),
                },
            )
            reduced = _reduce_semantic_result(
                project_path,
                workspace,
                version_id,
                runnable,
                semantic,
                invocation_hash=invocation.digest(),
            )
            if reduced is not None:
                return reduced
            continue

        code = runner(args)
        result = _stage_result(workspace, runnable, plan)
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
            if int(plan.get("version") or 1) >= 2:
                return mark_task_failed(workspace, task_id, reason="sql_two_phase_contract_required")
            decision = evaluate_task_policy(runnable, workspace, task_id=task_id)
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


def _record_attempt(workspace: Path, task_id: str, attempt_id: str, invocation_hash: str) -> None:
    state = load_agent_state(workspace)
    for task in state.get("tasks", []) or []:
        if task.get("task_id") == task_id:
            task["attempt_id"] = attempt_id
            task["invocation_hash"] = invocation_hash
            task["attempt_started_at"] = datetime.now().isoformat(timespec="seconds")
            break
    save_agent_state(workspace, state)


def _reduce_semantic_result(
    project_path: Path,
    workspace: Path,
    version_id: str,
    task: dict[str, Any],
    result: ActionResult,
    *,
    invocation_hash: str,
) -> dict[str, Any] | None:
    task_id = str(task.get("task_id") or "")
    if result.next_required_action == "approval":
        return pause_agent(
            workspace,
            status="waiting_for_approval",
            reason=result.failure_code or "approval_required",
            task_id=task_id,
            command_hash=invocation_hash,
        )
    if result.next_required_action == "advisor":
        request = create_advisor_request(
            workspace,
            project_dir=project_path,
            version_id=version_id,
            task=task,
            reason=result.failure_code or "advisor_required",
            message=result.message,
        )
        return pause_agent(
            workspace,
            status="waiting_for_advisor",
            reason=result.failure_code or "advisor_required",
            task_id=task_id,
            advisor_request=str(request.get("path") or ""),
            advisor_request_id=str(request.get("request_id") or ""),
        )
    if result.next_required_action == "reconciliation":
        return _pause_for_reconciliation(workspace, task_id, result)
    if result.next_required_action == "user":
        return mark_task_failed(workspace, task_id, reason="user_transition_requires_advisor_consumption")
    if result.status == "done":
        mark_task_done(workspace, task_id, message=result.message)
        return None
    if result.status == "scaffold":
        mark_task_done(workspace, task_id, scaffold=True, message=result.message or "scaffold output")
        return None
    if result.status == "review_ready":
        return pause_agent(
            workspace,
            status="blocked",
            reason="review_ready_without_next_action",
            task_id=task_id,
            command_hash=invocation_hash,
        )
    return mark_task_failed(workspace, task_id, reason=result.failure_code or "semantic_action_failed")


def _pause_for_reconciliation(workspace: Path, task_id: str, result: ActionResult) -> dict[str, Any]:
    state = load_agent_state(workspace)
    for task in state.get("tasks", []) or []:
        if task.get("task_id") == task_id:
            task.update(apply_task_transition(task, "unknown_external_outcome", "reconciliation_required"))
            break
    blocker_id = f"reconciliation:{result.attempt_id}"
    blocker = {
        "blocker_type": "reconciliation",
        "blocker_id": blocker_id,
        "attempt_id": result.attempt_id,
        "reason": result.failure_code or "external_outcome_unknown",
        "next_safe_action": {
            "action": "reconcile_external_operation",
            "required_evidence": "operator-supplied external operation receipt",
        },
    }
    state = apply_transition(
        state,
        "unknown_external_outcome",
        {"target_state": "reconciliation_required", "blocker": blocker},
    )
    state["current_task"] = task_id
    save_agent_state(workspace, state)
    return state


def _next_runnable_task(plan: dict[str, Any], status_by_task: dict[str, str]) -> dict[str, Any] | None:
    done_statuses = {"done"} if int(plan.get("version") or 1) >= 2 else {"done", "scaffold", "skipped"}
    for task in plan.get("tasks", []) or []:
        task_id = str(task.get("task_id") or "")
        status = status_by_task.get(task_id, "pending")
        if status not in {"pending", "paused"}:
            continue
        deps = [str(dep) for dep in task.get("depends_on") or []]
        if all(status_by_task.get(dep) in done_statuses for dep in deps):
            return task
    return None


def _stage_result(workspace: Path, task: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    try:
        invocation = invocation_for_task(task, plan)
        action = get_action_spec(TOOL_REGISTRY[invocation.tool_name].action_id)
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
    state_version = int(state.get("version") or 1)
    if statuses and all(status in {"done", "scaffold", "skipped"} for status in statuses):
        target = "done_with_gaps" if any(status != "done" for status in statuses) else "done"
        try:
            audit = audit_run(project_dir, version_id)
            if audit.get("verdict") != "complete":
                target = "done_with_gaps"
            state["latest_audit_verdict"] = audit.get("verdict", "")
        except Exception as exc:
            target = "done_with_gaps"
            state["latest_audit_error"] = str(exc)
        if state_version >= 2:
            state = apply_transition(
                state,
                "strict_close" if target == "done" else "incomplete_close",
                {"target_state": target},
            )
        else:
            state["status"] = target
        from risk_model_workbench.agent.state import save_agent_state

        save_agent_state(workspace, state)
    elif any(status in {"pending", "paused", "running"} for status in statuses):
        blocker = {
            "blocker_type": "dependency",
            "blocker_id": f"dependency:{version_id}:{state.get('current_task') or 'agent'}",
            "reason": "no_runnable_task",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "next_safe_action": {
                "action": "resolve_dependency",
                "required_evidence": "real upstream completion evidence",
            },
        }
        if state_version >= 2:
            state = apply_transition(state, "block", {"target_state": "blocked", "blocker": blocker})
        else:
            state["status"] = "blocked"
            state["blocker"] = blocker
        from risk_model_workbench.agent.state import save_agent_state

        save_agent_state(workspace, state)
    return state


def _default_runner(argv: list[str]) -> int:
    from risk_model_workbench.cli import main

    return main(argv)
