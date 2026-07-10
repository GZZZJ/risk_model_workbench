from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ADR = ROOT / "docs" / "adr" / "0003-host-agent-harness-contract.md"

AGENT_STATES = {
    "draft",
    "running",
    "waiting_for_approval",
    "waiting_for_advisor",
    "waiting_for_user",
    "reconciliation_required",
    "blocked",
    "failed",
    "stopped",
    "done",
    "done_with_gaps",
}

TASK_STATES = {
    "pending",
    "running",
    "review_ready",
    "paused",
    "done",
    "scaffold",
    "failed",
    "skipped",
    "stopped",
    "reconciliation_required",
}

NON_TERMINAL_BLOCKERS = {
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
    ("review_ready", "revise", "pending"),
    ("review_ready", "stop", "stopped"),
    ("paused", "resume", "pending"),
    ("paused", "fail", "failed"),
    ("paused", "stop", "stopped"),
    ("reconciliation_required", "confirm_success", "done"),
    ("reconciliation_required", "confirm_failure", "failed"),
    ("reconciliation_required", "abandon", "stopped"),
}


def contract_errors(
    current_state: str,
    event: str,
    target_state: str,
    *,
    task_statuses: tuple[str, ...] = (),
    permission: str = "writes_run",
    approval_consumed: bool = False,
    advisor_response_accepted: bool = False,
    advisor_response_consumed: bool = False,
    next_safe_action: dict[str, str] | None = None,
) -> list[str]:
    """Small executable specification used until the P0.1 reducer exists."""
    errors: list[str] = []
    if current_state not in AGENT_STATES or target_state not in AGENT_STATES:
        errors.append("unknown_agent_state")
        return errors
    if any(status not in TASK_STATES for status in task_statuses):
        errors.append("unknown_task_state")
    if (current_state, event, target_state) not in ALLOWED_AGENT_TRANSITIONS:
        errors.append("illegal_transition")
    if current_state in NON_TERMINAL_BLOCKERS or target_state in NON_TERMINAL_BLOCKERS:
        if next_safe_action is None:
            errors.append("missing_next_safe_action")
        elif set(next_safe_action) - {"action", "required_evidence", "command_hint"}:
            errors.append("invalid_next_safe_action")
        elif not str(next_safe_action.get("action", "")).strip() or not str(
            next_safe_action.get("required_evidence", "")
        ).strip():
            errors.append("invalid_next_safe_action")
    if event == "run_task" and permission == "dp_sql_pull" and not approval_consumed:
        errors.append("approval_not_consumed")
    if current_state == "waiting_for_advisor" and event == "run_task":
        if not advisor_response_accepted:
            errors.append("advisor_response_missing")
        elif not advisor_response_consumed:
            errors.append("advisor_response_not_consumed")
    if target_state == "done" and any(status != "done" for status in task_statuses):
        errors.append("terminal_evidence_incomplete")
    return errors


def task_transition_errors(current_state: str, event: str, target_state: str) -> list[str]:
    if current_state not in TASK_STATES or target_state not in TASK_STATES:
        return ["unknown_task_state"]
    if (current_state, event, target_state) not in ALLOWED_TASK_TRANSITIONS:
        return ["illegal_task_transition"]
    return []


def dependencies_unlocked(statuses: tuple[str, ...]) -> bool:
    """Only real completion may unlock a real downstream action."""
    return bool(statuses) and all(status == "done" for status in statuses)


def test_contract_adr_exists_and_lists_the_complete_state_vocabulary():
    text = ADR.read_text(encoding="utf-8")
    for state in sorted(AGENT_STATES | TASK_STATES):
        assert f"`{state}`" in text


def test_unapproved_dp_task_is_never_runnable():
    assert "approval_not_consumed" in contract_errors(
        "running",
        "run_task",
        "running",
        task_statuses=("pending",),
        permission="dp_sql_pull",
    )


def test_unconsumed_advisor_response_does_not_resume_task():
    errors = contract_errors(
        "waiting_for_advisor",
        "run_task",
        "running",
        task_statuses=("paused",),
        advisor_response_accepted=True,
        advisor_response_consumed=False,
        next_safe_action={
            "action": "consume_advisor_response",
            "required_evidence": "advisor consumption receipt",
        },
    )
    assert "advisor_response_not_consumed" in errors


def test_review_ready_is_not_a_terminal_success():
    assert "terminal_evidence_incomplete" in contract_errors(
        "running",
        "strict_close",
        "done",
        task_statuses=("review_ready",),
    )


def test_scaffold_does_not_unlock_real_downstream_work():
    assert dependencies_unlocked(("scaffold",)) is False
    assert dependencies_unlocked(("done",)) is True


def test_unlisted_and_terminal_agent_transitions_are_rejected():
    assert contract_errors("draft", "strict_close", "done") == ["illegal_transition"]
    assert contract_errors("done", "start", "running") == ["illegal_transition"]


def test_unlisted_and_terminal_task_transitions_are_rejected():
    assert task_transition_errors("pending", "start", "running") == []
    assert task_transition_errors("review_ready", "complete", "done") == [
        "illegal_task_transition"
    ]
    assert task_transition_errors("done", "start", "running") == [
        "illegal_task_transition"
    ]


def test_blocker_requires_exactly_one_persisted_next_safe_action():
    assert "missing_next_safe_action" in contract_errors(
        "waiting_for_approval",
        "consume_approval",
        "running",
    )
    assert "missing_next_safe_action" not in contract_errors(
        "waiting_for_approval",
        "consume_approval",
        "running",
        next_safe_action={
            "action": "consume_bound_approval",
            "required_evidence": "approval consumption receipt",
        },
    )
    assert "missing_next_safe_action" in contract_errors(
        "running",
        "block",
        "blocked",
    )
    assert "invalid_next_safe_action" in contract_errors(
        "running",
        "request_approval",
        "waiting_for_approval",
        next_safe_action={"action": "approve"},
    )
    assert "invalid_next_safe_action" not in contract_errors(
        "running",
        "request_approval",
        "waiting_for_approval",
        next_safe_action={
            "action": "review_bound_sql",
            "required_evidence": "approval record",
        },
    )


def test_production_transition_reducer_conforms_to_contract():
    from risk_model_workbench.agent.transitions import validate_transition

    assert validate_transition("done", "start", "running") == ["illegal_transition"]


def test_v2_blockers_require_bound_consumption_evidence_and_consume_once():
    from risk_model_workbench.agent.transitions import apply_transition

    approval_state = {
        "status": "waiting_for_approval",
        "blocker": {
            "blocker_type": "approval",
            "blocker_id": "approval_1",
            "approval_id": "approval_1",
        },
    }
    with pytest.raises(ValueError, match="consumption_evidence_required"):
        apply_transition(approval_state, "consume_approval", {"target_state": "running"})
    consumed = apply_transition(
        approval_state,
        "consume_approval",
        {
            "target_state": "running",
            "consumption_evidence": {
                "receipt_id": "receipt_1",
                "blocker_id": "approval_1",
                "consumed": True,
            },
        },
    )
    assert consumed["status"] == "running"
    with pytest.raises(ValueError, match="illegal_transition"):
        apply_transition(
            consumed,
            "consume_approval",
            {
                "target_state": "running",
                "consumption_evidence": {
                    "receipt_id": "receipt_1",
                    "blocker_id": "approval_1",
                    "consumed": True,
                },
            },
        )


def test_accepted_advisor_response_is_not_consumed_and_can_require_user_confirmation():
    from risk_model_workbench.agent.transitions import apply_transition

    advisor_state = {
        "status": "waiting_for_advisor",
        "blocker": {
            "blocker_type": "advisor",
            "blocker_id": "advisor_1",
            "advisor_request_id": "advisor_1",
            "response_status": "accepted",
        },
    }
    with pytest.raises(ValueError, match="consumption_evidence_required"):
        apply_transition(advisor_state, "consume_response", {"target_state": "running"})
    waiting_for_user = apply_transition(
        advisor_state,
        "consume_response",
        {
            "target_state": "waiting_for_user",
            "consumption_evidence": {
                "receipt_id": "receipt_2",
                "blocker_id": "advisor_1",
                "consumed": True,
            },
            "blocker": {
                "blocker_type": "user",
                "blocker_id": "user:advisor_1",
                "advisor_request_id": "advisor_1",
                "next_safe_action": {
                    "action": "confirm_advisor_decision",
                    "required_evidence": "user confirmation receipt",
                },
            },
        },
    )
    assert waiting_for_user["status"] == "waiting_for_user"


@pytest.mark.parametrize(
    ("state", "event", "target"),
    [
        (
            {
                "status": "blocked",
                "blocker": {"blocker_type": "dependency", "blocker_id": "dependency_1"},
            },
            "resolve_blocker",
            "running",
        ),
        (
            {
                "status": "reconciliation_required",
                "blocker": {"blocker_type": "reconciliation", "blocker_id": "operation_1"},
            },
            "reconcile_success",
            "running",
        ),
    ],
)
def test_dependency_and_reconciliation_blockers_require_bound_receipts(state, event, target):
    from risk_model_workbench.agent.transitions import apply_transition

    with pytest.raises(ValueError, match="consumption_evidence_required"):
        apply_transition(state, event, {"target_state": target})


@pytest.mark.parametrize(
    ("current", "event", "target"),
    [
        ("waiting_for_approval", "consume_approval", "running"),
        ("waiting_for_approval", "reject_approval", "blocked"),
        ("waiting_for_advisor", "consume_response", "running"),
        ("waiting_for_advisor", "consume_response", "waiting_for_user"),
        ("waiting_for_advisor", "consume_response", "stopped"),
        ("waiting_for_user", "confirm", "running"),
        ("waiting_for_user", "reject", "stopped"),
        ("reconciliation_required", "reconcile_success", "running"),
        ("reconciliation_required", "reconcile_failure", "failed"),
        ("reconciliation_required", "reconcile_abandoned", "stopped"),
        ("blocked", "resolve_blocker", "running"),
        ("blocked", "stop", "stopped"),
        ("blocked", "fail", "failed"),
    ],
)
def test_every_allowed_blocker_exit_requires_a_nonempty_bound_receipt(current, event, target):
    from risk_model_workbench.agent.transitions import apply_transition

    state = {
        "status": current,
        "blocker": {
            "blocker_type": "test",
            "blocker_id": "blocker_1",
            "next_safe_action": {"action": "resolve", "required_evidence": "receipt"},
        },
    }
    payload = {"target_state": target}
    if target in NON_TERMINAL_BLOCKERS:
        payload["blocker"] = {
            "blocker_type": "next",
            "blocker_id": "blocker_2",
            "next_safe_action": {"action": "continue", "required_evidence": "receipt"},
        }
    with pytest.raises(ValueError, match="consumption_evidence_required"):
        apply_transition(state, event, payload)


@pytest.mark.parametrize("receipt_id", ["", None, "   "])
def test_blocker_exit_rejects_empty_receipt_identity(receipt_id):
    from risk_model_workbench.agent.transitions import apply_transition

    state = {
        "status": "blocked",
        "blocker": {
            "blocker_type": "dependency",
            "blocker_id": "blocker_1",
            "next_safe_action": {"action": "resolve", "required_evidence": "receipt"},
        },
    }
    with pytest.raises(ValueError, match="invalid_consumption_evidence"):
        apply_transition(
            state,
            "resolve_blocker",
            {
                "target_state": "running",
                "consumption_evidence": {
                    "receipt_id": receipt_id,
                    "blocker_id": "blocker_1",
                    "consumed": True,
                },
            },
        )
