"""Deterministic execution loop for RMW Agent plans."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from risk_model_workbench.agent.advisor import advisor_request_is_answered, create_advisor_request
from risk_model_workbench.agent.approvals import (
    approval_by_id,
    build_approval_subject,
    consume_approval,
    ensure_subject_approval,
    revoke_approval,
)
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
        if state.get("status") == "waiting_for_approval":
            state = _consume_ready_sql_approval(workspace, plan, state)
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
        decision = evaluate_task_policy(
            invocation,
            workspace,
            task_id=str(runnable.get("task_id") or ""),
            subject_bound=int(plan.get("version") or 1) >= 2,
        )
        if not decision.allowed:
            state = pause_agent(
                workspace,
                status=decision.status,
                reason=decision.reason,
                task_id=str(runnable.get("task_id") or ""),
                approval_id=decision.approval_id,
                command_hash=decision.command_hash,
            )
            if decision.status == "waiting_for_approval":
                review_ready_dependencies = [
                    dep
                    for dep in runnable.get("depends_on") or []
                    if task_status_map(state).get(str(dep)) == "review_ready"
                ]
                if review_ready_dependencies:
                    state["blocker"]["prepare_task_id"] = str(review_ready_dependencies[0])
                    save_agent_state(workspace, state)
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
        approval_binding = _task_approval_binding(state, task_id)
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
                approval_id=str(approval_binding.get("approval_id") or ""),
                approval_subject_hash=str(approval_binding.get("subject_hash") or ""),
                approval_consumption_receipt=str(approval_binding.get("consumption_receipt") or ""),
                parent_operation_id=str(approval_binding.get("parent_operation_id") or ""),
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


def _task_approval_binding(state: dict[str, Any], task_id: str) -> dict[str, Any]:
    for task in state.get("tasks", []) or []:
        if task.get("task_id") == task_id and isinstance(task.get("approval_binding"), dict):
            return dict(task["approval_binding"])
    return {}


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
        tool_name = str(((task.get("invocation") or {}).get("tool_name") or task.get("tool_name") or ""))
        if tool_name.endswith("_prepare"):
            _mark_task_review_ready(workspace, task_id, result.message or "SQL evidence ready for review")
            return None
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
    approval_binding: dict[str, Any] = {}
    for task in state.get("tasks", []) or []:
        if task.get("task_id") == task_id:
            task.update(apply_task_transition(task, "unknown_external_outcome", "reconciliation_required"))
            if isinstance(task.get("approval_binding"), dict):
                approval_binding = dict(task["approval_binding"])
            break
    blocker_id = f"reconciliation:{result.attempt_id}"
    consumption_receipt = str(approval_binding.get("consumption_receipt") or "")
    consumption_receipt_status = "missing"
    if consumption_receipt:
        receipt_path = (workspace / consumption_receipt).resolve()
        if workspace.resolve() in receipt_path.parents and receipt_path.is_file():
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                consumption_receipt_status = "consumed" if receipt.get("consumed") is True else "invalid"
            except (OSError, json.JSONDecodeError):
                consumption_receipt_status = "invalid"
    external_operations = []
    for intent_path in sorted((workspace / "audit" / "external_operations").glob("*.intent.json")):
        try:
            intent = json.loads(intent_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if intent.get("attempt_id") == result.attempt_id:
            receipt_path = intent_path.with_name(intent_path.name.replace(".intent.json", ".receipt.json"))
            receipt: dict[str, Any] = {}
            if receipt_path.is_file():
                try:
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    receipt = {}
            external_operations.append(
                {
                    "operation_id": intent.get("operation_id", ""),
                    "parent_operation_id": intent.get("parent_operation_id", ""),
                    "approval_id": intent.get("approval_id", ""),
                    "subject_hash": intent.get("subject_hash", ""),
                    "attempt_id": intent.get("attempt_id", ""),
                    "intent_path": str(intent_path.relative_to(workspace)),
                    "receipt_path": str(receipt_path.relative_to(workspace)) if receipt_path.is_file() else "",
                    "receipt_status": receipt.get("status", "unknown"),
                }
            )
    blocker = {
        "blocker_type": "reconciliation",
        "blocker_id": blocker_id,
        "attempt_id": result.attempt_id,
        "task_id": task_id,
        "reason": result.failure_code or "external_outcome_unknown",
        "approval_id": str(approval_binding.get("approval_id") or ""),
        "subject_hash": str(approval_binding.get("subject_hash") or ""),
        "consumption_receipt": consumption_receipt,
        "consumption_receipt_status": consumption_receipt_status,
        "parent_operation_id": str(approval_binding.get("parent_operation_id") or ""),
        "external_operations": external_operations,
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
        tool_name = str(((task.get("invocation") or {}).get("tool_name") or task.get("tool_name") or ""))
        dependency_statuses = [status_by_task.get(dep) for dep in deps]
        execute_after_review = tool_name.endswith("_execute")
        if all(
            status in done_statuses or (execute_after_review and status == "review_ready")
            for status in dependency_statuses
        ):
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


def _mark_task_review_ready(workspace: Path, task_id: str, message: str) -> None:
    state = load_agent_state(workspace)
    for task in state.get("tasks", []) or []:
        if task.get("task_id") == task_id:
            task.update(apply_task_transition(task, "prepare_complete", "review_ready"))
            task["message"] = message
            task["finished_at"] = datetime.now().isoformat(timespec="seconds")
            break
    state["status"] = "running"
    save_agent_state(workspace, state)


def _consume_ready_sql_approval(workspace: Path, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    blocker = state.get("blocker") if isinstance(state.get("blocker"), dict) else {}
    approval_id = str(blocker.get("approval_id") or "")
    approval = approval_by_id(workspace, approval_id) if approval_id else None
    if not approval or approval.get("status") in {"pending", "rejected", "consumed"}:
        return state
    execute_task_id = str(blocker.get("task_id") or "")
    execute_task = next(
        (task for task in plan.get("tasks", []) or [] if str(task.get("task_id") or "") == execute_task_id),
        None,
    )
    if execute_task is None:
        return state
    invocation = invocation_for_task(execute_task, plan)
    try:
        subject = build_approval_subject(
            workspace,
            project=invocation.project,
            version_id=invocation.version_id,
            task_id=execute_task_id,
            invocation_hash=invocation.digest(),
            operation_id=invocation.tool_name,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        if approval.get("status") == "approved":
            revoke_approval(workspace, approval_id, reason="approval_subject_invalid")
        state["blocker"]["reason"] = f"sql_evidence_invalid:{exc}"
        save_agent_state(workspace, state)
        return state
    if approval.get("status") == "revoked":
        replacement = ensure_subject_approval(workspace, subject, reason="approval_subject_revalidation")
        state["blocker"]["approval_id"] = str(replacement.get("approval_id") or "")
        state["blocker"]["reason"] = "approval_required"
        save_agent_state(workspace, state)
        return state
    try:
        receipt = consume_approval(workspace, approval_id, subject, consumed_by="agent_runtime")
    except ValueError as exc:
        replacement = ensure_subject_approval(workspace, subject, reason="approval_subject_drift")
        state["blocker"]["reason"] = str(exc)
        state["blocker"]["approval_id"] = str(replacement.get("approval_id") or "")
        save_agent_state(workspace, state)
        return state
    for task in state.get("tasks", []) or []:
        if task.get("task_id") == execute_task_id:
            if task.get("status") == "paused":
                task.update(apply_task_transition(task, "resume", "pending"))
            task["approval_binding"] = {
                "approval_id": approval_id,
                "subject_hash": receipt["subject_hash"],
                "consumption_receipt": str(Path("audit") / "approval_consumptions" / f"{approval_id}.json"),
                "parent_operation_id": invocation.tool_name,
            }
        if task.get("task_id") == blocker.get("prepare_task_id") and task.get("status") == "review_ready":
            task.update(apply_task_transition(task, "approval_confirmed", "done"))
    state = apply_transition(
        state,
        "consume_approval",
        {"target_state": "running", "consumption_evidence": receipt},
    )
    state["current_task"] = ""
    save_agent_state(workspace, state)
    return state


def _default_runner(argv: list[str]) -> int:
    from risk_model_workbench.cli import main

    return main(argv)
