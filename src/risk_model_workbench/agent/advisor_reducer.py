"""Deterministic reducer for accepted Advisor responses."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from risk_model_workbench.agent.advisor import (
    create_replacement_advisor_request,
    current_context_hash_for_request,
    load_accepted_advisor_response,
    load_advisor_request,
    request_identity,
)
from risk_model_workbench.agent.state import save_agent_state
from risk_model_workbench.agent.transitions import apply_task_transition, apply_transition
from risk_model_workbench.agent.workspace_store import WorkspaceStore
from risk_model_workbench.modeling.llm_tuning import resolve_tuning_config, validate_tuning_plan


@dataclass(frozen=True)
class AdvisorConsumptionResult:
    consumed: bool
    status: str
    state: dict[str, Any]
    receipt_path: str = ""
    replacement_request_id: str = ""
    errors: list[str] = field(default_factory=list)


def consume_advisor_response(
    workspace: Path,
    request_id: str,
    current_state: dict[str, object],
) -> AdvisorConsumptionResult:
    workspace_path = Path(workspace)
    state = deepcopy(current_state)
    try:
        request = load_advisor_request(workspace_path, request_id)
    except (KeyError, ValueError) as exc:
        return _error(state, str(exc))

    receipt_relative = Path("audit") / "advisor_consumptions" / f"{request_id}.json"
    receipt_path = workspace_path / receipt_relative
    if receipt_path.exists() or request.get("status") in {"consumed", "waiting_for_user"}:
        return _error(state, f"advisor response already consumed: {request_id}")
    if request.get("status") not in {"answered", "rejected"}:
        return _error(state, f"advisor response pending: {request_id}")

    blocker = state.get("blocker") if isinstance(state.get("blocker"), dict) else {}
    if state.get("status") != "waiting_for_advisor" or blocker.get("advisor_request_id") != request_id:
        return _error(state, "advisor request does not match current blocker")
    task = _task(state, str(request.get("task_id") or ""))
    if task is None:
        return _error(state, f"unknown advisor task: {request.get('task_id')}")
    identity_errors = _current_identity_errors(request, task)
    if identity_errors:
        return _error(state, "; ".join(identity_errors))
    try:
        current_context_hash = current_context_hash_for_request(workspace_path, request)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return _error(state, f"context pack invalid: {exc}")
    if current_context_hash != request.get("context_hash"):
        return _error(state, "context hash mismatch")

    try:
        response = load_accepted_advisor_response(workspace_path, request)
    except (OSError, json.JSONDecodeError) as exc:
        return _error(state, f"advisor response unreadable: {exc}")
    response_identity = response.get("request_identity") if isinstance(response.get("request_identity"), dict) else {}
    if response_identity != request_identity(request):
        return _error(state, "response identity mismatch")

    tuning_errors = _validate_tuning_outputs(workspace_path, request, response)
    if tuning_errors:
        return _error(state, "; ".join(tuning_errors))

    if request.get("status") == "rejected" or response.get("status") == "rejected":
        replacement = create_replacement_advisor_request(workspace_path, request, reason="advisor_response_rejected")
        request["status"] = "consumed"
        receipt = _receipt(
            request=request,
            response=response,
            blocker_id=str(blocker.get("blocker_id") or ""),
            target_status="waiting_for_advisor",
            replacement_request_id=str(replacement.get("request_id") or ""),
        )
        if not WorkspaceStore(workspace_path).create_once(receipt_relative, receipt):
            return _error(state, f"advisor response already consumed: {request_id}")
        request["consumed_at"] = _now()
        request["consumption_receipt"] = str(receipt_relative)
        _save_request(workspace_path, request)
        state["blocker"] = {
            **blocker,
            "advisor_request": replacement.get("path", ""),
            "advisor_request_id": replacement.get("request_id", ""),
            "reason": "advisor_response_rejected",
            "blocker_id": f"advisor:{replacement.get('request_id')}",
            "next_safe_action": {
                "action": "consume_advisor_response",
                "required_evidence": "accepted Advisor response",
            },
        }
        save_agent_state(workspace_path, state)
        return AdvisorConsumptionResult(
            consumed=True,
            status="waiting_for_advisor",
            state=state,
            receipt_path=str(receipt_relative),
            replacement_request_id=str(replacement.get("request_id") or ""),
        )

    decision = str(response.get("decision") or "")
    retry_errors = _apply_task_decision(task, decision, request)
    if retry_errors:
        return _error(state, "; ".join(retry_errors))
    target_status = {
        "continue": "running",
        "retry": "running",
        "stop": "stopped",
        "needs_user_confirmation": "waiting_for_user",
    }.get(decision)
    if not target_status:
        return _error(state, f"unknown advisor response decision: {decision}")

    receipt = _receipt(request=request, response=response, blocker_id=str(blocker.get("blocker_id") or ""), target_status=target_status)
    if not WorkspaceStore(workspace_path).create_once(receipt_relative, receipt):
        return _error(state, f"advisor response already consumed: {request_id}")

    request["status"] = "waiting_for_user" if target_status == "waiting_for_user" else "consumed"
    request["consumed_at"] = _now()
    request["consumption_receipt"] = str(receipt_relative)
    _save_request(workspace_path, request)

    if target_status == "waiting_for_user":
        user_blocker = {
            "blocker_type": "user",
            "blocker_id": f"user:{request_id}",
            "advisor_request_id": request_id,
            "task_id": request.get("task_id", ""),
            "reason": "advisor_requires_user_confirmation",
            "created_at": _now(),
            "next_safe_action": {
                "action": "confirm_advisor_response",
                "required_evidence": "explicit user confirmation",
            },
        }
        state = apply_transition(
            state,
            "consume_response",
            {"target_state": target_status, "blocker": user_blocker, "consumption_evidence": receipt},
        )
    else:
        state = apply_transition(
            state,
            "consume_response",
            {"target_state": target_status, "consumption_evidence": receipt},
        )
    state["current_task"] = "" if target_status == "running" else str(request.get("task_id") or "")
    save_agent_state(workspace_path, state)
    return AdvisorConsumptionResult(consumed=True, status=target_status, state=state, receipt_path=str(receipt_relative))


def confirm_advisor_response(workspace: str | Path, request_id: str, current_state: dict[str, object], *, confirmed_by: str) -> AdvisorConsumptionResult:
    workspace_path = Path(workspace)
    state = deepcopy(current_state)
    blocker = state.get("blocker") if isinstance(state.get("blocker"), dict) else {}
    if state.get("status") != "waiting_for_user" or blocker.get("advisor_request_id") != request_id:
        return _error(state, "advisor request does not match current user confirmation blocker")
    try:
        request = load_advisor_request(workspace_path, request_id)
    except (KeyError, ValueError) as exc:
        return _error(state, str(exc))
    if not _claim_user_decision(workspace_path, request_id, decision="confirmed", actor=confirmed_by):
        return _error(state, f"advisor user decision already recorded: {request_id}")
    task = _task(state, str(request.get("task_id") or ""))
    if task is not None and task.get("status") == "paused":
        task.update(apply_task_transition(task, "resume", "pending"))
    receipt_relative = Path("audit") / "advisor_consumptions" / f"{request_id}.confirmation.json"
    receipt = {
        "receipt_id": f"advisor_confirmation:{request_id}",
        "blocker_id": blocker.get("blocker_id", ""),
        "request_id": request_id,
        "confirmed_by": confirmed_by,
        "consumed": True,
        "created_at": _now(),
    }
    if not WorkspaceStore(workspace_path).create_once(receipt_relative, receipt):
        return _error(state, f"advisor confirmation already recorded: {request_id}")
    request["status"] = "consumed"
    request["confirmed_by"] = confirmed_by
    request["confirmed_at"] = _now()
    request["consumption_receipt"] = str(receipt_relative)
    _save_request(workspace_path, request)
    state = apply_transition(state, "confirm", {"target_state": "running", "consumption_evidence": receipt})
    state["current_task"] = ""
    save_agent_state(workspace_path, state)
    return AdvisorConsumptionResult(consumed=True, status="running", state=state, receipt_path=str(receipt_relative))


def reject_advisor_response(workspace: str | Path, request_id: str, current_state: dict[str, object], *, reason: str) -> AdvisorConsumptionResult:
    workspace_path = Path(workspace)
    state = deepcopy(current_state)
    blocker = state.get("blocker") if isinstance(state.get("blocker"), dict) else {}
    if state.get("status") != "waiting_for_user" or blocker.get("advisor_request_id") != request_id:
        return _error(state, "advisor request does not match current user confirmation blocker")
    try:
        request = load_advisor_request(workspace_path, request_id)
    except (KeyError, ValueError) as exc:
        return _error(state, str(exc))
    if not _claim_user_decision(workspace_path, request_id, decision="rejected", actor=reason):
        return _error(state, f"advisor user decision already recorded: {request_id}")
    task = _task(state, str(request.get("task_id") or ""))
    if task is not None and task.get("status") == "paused":
        task.update(apply_task_transition(task, "stop", "stopped"))
    receipt_relative = Path("audit") / "advisor_consumptions" / f"{request_id}.rejection.json"
    receipt = {
        "receipt_id": f"advisor_rejection:{request_id}",
        "blocker_id": blocker.get("blocker_id", ""),
        "request_id": request_id,
        "reason": reason,
        "consumed": True,
        "created_at": _now(),
    }
    if not WorkspaceStore(workspace_path).create_once(receipt_relative, receipt):
        return _error(state, f"advisor rejection already recorded: {request_id}")
    request["status"] = "consumed"
    request["rejected_reason"] = reason
    request["rejected_at"] = _now()
    request["consumption_receipt"] = str(receipt_relative)
    _save_request(workspace_path, request)
    state = apply_transition(state, "reject", {"target_state": "stopped", "consumption_evidence": receipt})
    state["current_task"] = str(request.get("task_id") or "")
    save_agent_state(workspace_path, state)
    return AdvisorConsumptionResult(consumed=True, status="stopped", state=state, receipt_path=str(receipt_relative))


def _apply_task_decision(task: dict[str, Any], decision: str, request: dict[str, Any]) -> list[str]:
    if decision == "continue":
        if task.get("status") == "paused":
            task.update(apply_task_transition(task, "resume", "pending"))
        return []
    if decision == "needs_user_confirmation":
        return []
    if decision == "stop":
        if task.get("status") == "paused":
            task.update(apply_task_transition(task, "stop", "stopped"))
        return []
    if decision == "retry":
        retry_count = int(task.get("advisor_retry_count", 0) or 0)
        max_retries = int(((request.get("retry_budget") or {}) if isinstance(request.get("retry_budget"), dict) else {}).get("max_retries", 0) or 0)
        if retry_count >= max_retries:
            return ["retry budget exceeded"]
        task["advisor_retry_count"] = retry_count + 1
        if task.get("status") == "paused":
            task.update(apply_task_transition(task, "resume", "pending"))
        return []
    return [f"unknown advisor response decision: {decision}"]


def _current_identity_errors(request: dict[str, Any], task: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if str(task.get("attempt_id") or "") != str(request.get("attempt_id") or ""):
        errors.append("attempt_id mismatch")
    if str(task.get("invocation_hash") or "") != str(request.get("invocation_hash") or ""):
        errors.append("invocation_hash mismatch")
    return errors


def _validate_tuning_outputs(workspace: Path, request: dict[str, Any], response: dict[str, Any]) -> list[str]:
    if response.get("type") != "tuning_plan" or response.get("status") != "answered":
        return []
    tuning_cfg = _load_tuning_config(workspace)
    errors: list[str] = []
    expected_experiment = _experiment_from_request(request)
    expected_round = int(request.get("round", 0) or 0)
    plan_outputs = [
        str(relative)
        for relative in response.get("output_files") or []
        if Path(str(relative)).name.startswith("llm_tuning_plan_round_") and Path(str(relative)).suffix == ".json"
    ]
    if response.get("decision") in {"continue", "retry"} and not plan_outputs:
        errors.append("tuning_plan response requires llm_tuning_plan output")
    for relative in response.get("output_files") or []:
        path = workspace / str(relative)
        if path.name.startswith("llm_tuning_plan_round_") and path.suffix == ".json":
            try:
                plan = json.loads(path.read_text(encoding="utf-8"))
                validate_tuning_plan(
                    plan,
                    tuning_cfg,
                    advisor_type="embedded_agent_response",
                    expected_experiment=expected_experiment,
                    expected_round=expected_round,
                )
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                errors.append(str(exc))
    return errors


def _load_tuning_config(workspace: Path) -> dict[str, Any]:
    config_path = workspace / "configs_runtime" / "train.yaml"
    if config_path.exists():
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if isinstance(payload, dict):
            return resolve_tuning_config(payload)
    return resolve_tuning_config({"training": {"mode": "llm_guided_tune"}})


def _experiment_from_request(request: dict[str, Any]) -> str | None:
    command = list(request.get("command") or [])
    for index, value in enumerate(command):
        if value == "--experiment" and index + 1 < len(command):
            return str(command[index + 1])
    return None


def _receipt(
    *,
    request: dict[str, Any],
    response: dict[str, Any],
    blocker_id: str,
    target_status: str,
    replacement_request_id: str = "",
) -> dict[str, Any]:
    request_id = str(request.get("request_id") or "")
    return {
        "version": 1,
        "receipt_id": f"advisor_consumption:{request_id}",
        "blocker_id": blocker_id,
        "request_id": request_id,
        "request_hash": request.get("request_hash", ""),
        "context_hash": request.get("context_hash", ""),
        "task_id": request.get("task_id", ""),
        "attempt_id": request.get("attempt_id", ""),
        "invocation_hash": request.get("invocation_hash", ""),
        "round": request.get("round", 0),
        "response_hash": _hash(response),
        "decision": response.get("decision", "rejected") if response.get("status") != "rejected" else "rejected",
        "target_status": target_status,
        "replacement_request_id": replacement_request_id,
        "output_files": list(response.get("output_files") or []),
        "consumed": True,
        "created_at": _now(),
    }


def _task(state: dict[str, Any], task_id: str) -> dict[str, Any] | None:
    for task in state.get("tasks", []) or []:
        if task.get("task_id") == task_id:
            return task
    return None


def _save_request(workspace: Path, request: dict[str, Any]) -> None:
    relative = Path("audit") / "advisor_requests" / f"{request['request_id']}.json"
    store = WorkspaceStore(workspace)
    expected_revision = getattr(request, "store_revision", store.read_json(relative).revision)
    revision = store.write_json(relative, dict(request), expected_revision)
    if hasattr(request, "store_revision"):
        request.store_revision = revision


def _claim_user_decision(workspace: Path, request_id: str, *, decision: str, actor: str) -> bool:
    relative = Path("audit") / "advisor_consumptions" / f"{request_id}.user_decision.json"
    return WorkspaceStore(workspace).create_once(
        relative,
        {
            "receipt_id": f"advisor_user_decision:{request_id}",
            "request_id": request_id,
            "decision": decision,
            "actor": actor,
            "consumed": True,
            "created_at": _now(),
        },
    )


def _error(state: dict[str, Any], message: str) -> AdvisorConsumptionResult:
    return AdvisorConsumptionResult(consumed=False, status=str(state.get("status") or ""), state=state, errors=[message])


def _hash(payload: object) -> str:
    import hashlib

    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
