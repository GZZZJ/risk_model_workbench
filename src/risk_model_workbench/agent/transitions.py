"""Fail-closed Agent and task state transition reducer."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


BLOCKER_STATES = {
    "waiting_for_approval",
    "waiting_for_advisor",
    "waiting_for_user",
    "reconciliation_required",
    "blocked",
}

ALLOWED_AGENT_TRANSITIONS = {
    ("draft", "start", "running"),
    ("running", "run_task", "running"),
    ("running", "request_approval", "waiting_for_approval"),
    ("waiting_for_approval", "consume_approval", "running"),
    ("waiting_for_approval", "reject_approval", "blocked"),
    ("running", "request_advisor", "waiting_for_advisor"),
    ("waiting_for_advisor", "consume_response", "running"),
    ("waiting_for_advisor", "consume_response", "waiting_for_user"),
    ("waiting_for_advisor", "consume_response", "stopped"),
    ("waiting_for_user", "confirm", "running"),
    ("waiting_for_user", "reject", "stopped"),
    ("running", "unknown_external_outcome", "reconciliation_required"),
    ("reconciliation_required", "reconcile_success", "running"),
    ("reconciliation_required", "reconcile_failure", "failed"),
    ("reconciliation_required", "reconcile_abandoned", "stopped"),
    ("running", "block", "blocked"),
    ("blocked", "resolve_blocker", "running"),
    ("blocked", "stop", "stopped"),
    ("blocked", "fail", "failed"),
    ("running", "fail", "failed"),
    ("running", "stop", "stopped"),
    ("running", "strict_close", "done"),
    ("running", "incomplete_close", "done_with_gaps"),
    ("done_with_gaps", "resume", "running"),
}

ALLOWED_TASK_TRANSITIONS = {
    ("pending", "start", "running"),
    ("pending", "pause", "paused"),
    ("pending", "skip", "skipped"),
    ("pending", "stop", "stopped"),
    ("running", "prepare_complete", "review_ready"),
    ("running", "pause", "paused"),
    ("running", "complete", "done"),
    ("running", "record_scaffold", "scaffold"),
    ("running", "fail", "failed"),
    ("running", "skip", "skipped"),
    ("running", "stop", "stopped"),
    ("running", "unknown_external_outcome", "reconciliation_required"),
    ("review_ready", "request_approval", "paused"),
    ("review_ready", "approval_confirmed", "done"),
    ("review_ready", "revise", "pending"),
    ("review_ready", "stop", "stopped"),
    ("paused", "resume", "pending"),
    ("paused", "fail", "failed"),
    ("paused", "stop", "stopped"),
    ("reconciliation_required", "confirm_success", "done"),
    ("reconciliation_required", "confirm_failure", "failed"),
    ("reconciliation_required", "abandon", "stopped"),
}


def validate_transition(current_state: str, event: str, target_state: str) -> list[str]:
    if (current_state, event, target_state) not in ALLOWED_AGENT_TRANSITIONS:
        return ["illegal_transition"]
    return []


def validate_task_transition(current_state: str, event: str, target_state: str) -> list[str]:
    if (current_state, event, target_state) not in ALLOWED_TASK_TRANSITIONS:
        return ["illegal_task_transition"]
    return []


def apply_transition(state: dict[str, Any], event: str, payload: dict[str, Any]) -> dict[str, Any]:
    current = str(state.get("status") or "")
    target = str(payload.get("target_state") or "")
    errors = validate_transition(current, event, target)
    blocker = payload.get("blocker")
    if current in BLOCKER_STATES and target != current:
        errors.extend(_validate_consumption_evidence(state, event, payload.get("consumption_evidence")))
    if target in BLOCKER_STATES:
        errors.extend(_validate_blocker(blocker))
    if errors:
        raise ValueError(",".join(errors))
    updated = deepcopy(state)
    updated["status"] = target
    if target in BLOCKER_STATES:
        updated["blocker"] = deepcopy(blocker)
    elif current in BLOCKER_STATES:
        updated["blocker"] = {}
    return updated


def apply_task_transition(task: dict[str, Any], event: str, target_state: str) -> dict[str, Any]:
    errors = validate_task_transition(str(task.get("status") or ""), event, target_state)
    if errors:
        raise ValueError(",".join(errors))
    updated = deepcopy(task)
    updated["status"] = target_state
    return updated


def _validate_blocker(blocker: object) -> list[str]:
    if not isinstance(blocker, dict):
        return ["missing_next_safe_action"]
    if not str(blocker.get("blocker_type") or "").strip() or not str(blocker.get("blocker_id") or "").strip():
        return ["invalid_blocker_identity"]
    action = blocker.get("next_safe_action")
    if not isinstance(action, dict):
        return ["missing_next_safe_action"]
    allowed = {"action", "required_evidence", "command_hint"}
    if set(action) - allowed:
        return ["invalid_next_safe_action"]
    if not str(action.get("action") or "").strip() or not str(action.get("required_evidence") or "").strip():
        return ["invalid_next_safe_action"]
    return []


def _validate_consumption_evidence(state: dict[str, Any], event: str, evidence: object) -> list[str]:
    if not isinstance(evidence, dict):
        return ["consumption_evidence_required"]
    required = {"receipt_id", "blocker_id", "consumed"}
    if not required.issubset(evidence) or evidence.get("consumed") is not True:
        return ["invalid_consumption_evidence"]
    if not isinstance(evidence.get("receipt_id"), str) or not str(evidence.get("receipt_id") or "").strip():
        return ["invalid_consumption_evidence"]
    blocker = state.get("blocker") if isinstance(state.get("blocker"), dict) else {}
    expected = blocker.get("blocker_id")
    if not expected or evidence.get("blocker_id") != expected:
        return ["consumption_subject_mismatch"]
    return []
