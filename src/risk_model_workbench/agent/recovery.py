"""Durable attempt journal, startup diagnosis, and explicit reconciliation.

The journal is append-only.  Every line is a complete snapshot so a process
crash can at worst lose the transition currently being written; earlier
snapshots remain valid through :class:`WorkspaceStore`'s durable JSONL append.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from risk_model_workbench.agent.state import load_agent_state, save_agent_state
from risk_model_workbench.agent.transitions import apply_task_transition, apply_transition
from risk_model_workbench.agent.workspace_store import WorkspaceStore
from risk_model_workbench.harness.actions import automatic_recovery_allowed


ATTEMPT_JOURNAL = Path("audit") / "attempt_journal.jsonl"
SAFE_RETRY_SEMANTICS = {"read_only", "idempotent_write"}
EXECUTION_SEMANTICS = SAFE_RETRY_SEMANTICS | {"non_idempotent_write", "external_unknown"}
RECONCILIATION_OUTCOMES = {"confirmed_succeeded", "confirmed_failed", "abandoned"}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def begin_attempt(
    workspace: str | Path,
    *,
    attempt_id: str,
    transition_id: str,
    task_id: str,
    action_id: str,
    invocation_hash: str,
    execution_semantics: str,
    transaction_id: str = "",
) -> dict[str, Any]:
    """Persist execution intent before a command can be dispatched."""
    if execution_semantics not in EXECUTION_SEMANTICS:
        raise ValueError(f"invalid execution_semantics: {execution_semantics}")
    if not all(str(item).strip() for item in (attempt_id, transition_id, task_id, action_id, invocation_hash)):
        raise ValueError("attempt journal identity fields are required")
    if attempt_id in load_attempt_journal(workspace):
        raise ValueError(f"attempt already journaled: {attempt_id}")
    snapshot = {
        "schema_version": 1,
        "event_id": f"journal_{uuid4().hex}",
        "event": "intent_recorded",
        "attempt_id": attempt_id,
        "transition_id": transition_id,
        "task_id": task_id,
        "action_id": action_id,
        "invocation_hash": invocation_hash,
        "execution_semantics": execution_semantics,
        "intent_status": "recorded",
        "result_status": "pending",
        "transaction_id": transaction_id,
        "started_at": _now(),
        "finished_at": "",
    }
    _append_snapshot(workspace, snapshot)
    return snapshot


def mark_attempt_dispatched(workspace: str | Path, attempt_id: str) -> dict[str, Any]:
    """Durably mark the ambiguity boundary immediately before command dispatch."""
    return _update_attempt(workspace, attempt_id, event="command_dispatched", intent_status="dispatched")


def mark_attempt_command_finished(workspace: str | Path, attempt_id: str) -> dict[str, Any]:
    return _update_attempt(workspace, attempt_id, event="command_finished", intent_status="command_finished")


def mark_attempt_result_committed(
    workspace: str | Path,
    attempt_id: str,
    *,
    transaction_id: str = "",
) -> dict[str, Any]:
    return _update_attempt(
        workspace,
        attempt_id,
        event="result_committed",
        result_status="committed",
        transaction_id=transaction_id,
        finished_at=_now(),
    )


def mark_attempt_transitioned(workspace: str | Path, attempt_id: str) -> dict[str, Any]:
    return _update_attempt(
        workspace,
        attempt_id,
        event="state_transitioned",
        result_status="applied",
        finished_at=_now(),
    )


def mark_attempt_requeued(workspace: str | Path, attempt_id: str) -> dict[str, Any]:
    return _update_attempt(
        workspace,
        attempt_id,
        event="requeued",
        result_status="requeued",
        finished_at=_now(),
    )


def mark_attempt_reconciliation(workspace: str | Path, attempt_id: str) -> dict[str, Any]:
    return _update_attempt(
        workspace,
        attempt_id,
        event="reconciliation_required",
        result_status="reconciliation_required",
        finished_at=_now(),
    )


def load_attempt_journal(workspace: str | Path) -> dict[str, dict[str, Any]]:
    """Fold the append-only journal to the latest valid snapshot per attempt."""
    path = Path(workspace) / ATTEMPT_JOURNAL
    if not path.exists():
        return {}
    latest: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed attempt journal line {line_number}") from exc
        if not isinstance(item, dict) or not str(item.get("attempt_id") or ""):
            raise ValueError(f"invalid attempt journal line {line_number}")
        latest[str(item["attempt_id"])] = item
    return latest


def recovery_policy(execution_semantics: str, *, intent_status: str, result_status: str) -> str:
    """Return a fail-closed recovery decision for one persisted attempt."""
    if result_status == "committed":
        return "apply_receipt"
    if result_status in {"applied", "requeued", "reconciliation_required", "reconciled"}:
        return "none"
    if intent_status == "recorded":
        # Even an intent-only unsafe action requires a human decision.  The
        # absence of a dispatch marker is not authority to retry a tool whose
        # contract explicitly forbids automatic replay.
        return "requeue" if automatic_recovery_allowed(execution_semantics) else "reconciliation_required"
    if automatic_recovery_allowed(execution_semantics):
        return "retry"
    return "reconciliation_required"


def diagnose_recovery(workspace: str | Path) -> dict[str, Any]:
    attempts = []
    for snapshot in load_attempt_journal(workspace).values():
        decision = recovery_policy(
            str(snapshot.get("execution_semantics") or "external_unknown"),
            intent_status=str(snapshot.get("intent_status") or ""),
            result_status=str(snapshot.get("result_status") or ""),
        )
        item = dict(snapshot)
        item["recovery_action"] = "requeue" if decision == "retry" else decision
        attempts.append(item)
    attempts.sort(key=lambda item: (str(item.get("started_at") or ""), str(item.get("attempt_id") or "")))

    store = WorkspaceStore(workspace)
    state_transaction = str(store.read_yaml("version_state.yml").payload.get("transaction_id") or "")
    if not state_transaction:
        state_transaction = str(store.read_yaml("run_state.yml").payload.get("transaction_id") or "")
    manifest_transaction = str(
        store.read_json("audit/artifact_manifest.json").payload.get("transaction_id") or ""
    )
    divergence = bool(state_transaction or manifest_transaction) and state_transaction != manifest_transaction
    return {
        "version": 1,
        "workspace": str(Path(workspace).resolve()),
        "attempts": attempts,
        "unresolved_count": sum(item["recovery_action"] != "none" for item in attempts),
        "transaction_divergence": {
            "detected": divergence,
            "version_state_transaction_id": state_transaction,
            "manifest_transaction_id": manifest_transaction,
        },
    }


def reconcile_operation(
    workspace: str | Path,
    *,
    operation_id: str,
    outcome: str,
    evidence_path: str,
    evidence_sha256: str,
    operator_identity: str,
    note: str,
) -> dict[str, Any]:
    """Consume explicit operator evidence for one unknown external operation.

    This changes only Agent execution state and writes an audit receipt.  It
    intentionally never creates model, evaluation, report, or data artifacts.
    """
    if outcome not in RECONCILIATION_OUTCOMES:
        raise ValueError(f"invalid reconciliation outcome: {outcome}")
    if not str(operator_identity).strip():
        raise ValueError("operator_identity is required")
    if not str(note).strip():
        raise ValueError("reconciliation note is required")
    if not re.fullmatch(r"[A-Fa-f0-9]{64}", str(evidence_sha256)):
        raise ValueError("evidence_sha256 must be a 64-character SHA256")
    evidence = _validated_evidence_path(workspace, evidence_path)
    actual_sha256 = hashlib.sha256(evidence.read_bytes()).hexdigest()
    if actual_sha256.lower() != evidence_sha256.lower():
        raise ValueError("evidence_sha256 does not match evidence bytes")

    state = load_agent_state(workspace)
    if state.get("status") != "reconciliation_required":
        raise ValueError("agent is not awaiting reconciliation")
    blocker = state.get("blocker") if isinstance(state.get("blocker"), dict) else {}
    operations = blocker.get("external_operations") if isinstance(blocker.get("external_operations"), list) else []
    known_ids = {str(item.get("operation_id") or "") for item in operations if isinstance(item, dict)}
    known_ids.update(str(blocker.get(name) or "") for name in ("operation_id", "parent_operation_id"))
    if operation_id not in known_ids:
        raise ValueError(f"operation is not bound to current reconciliation blocker: {operation_id}")

    blocker_id = str(blocker.get("blocker_id") or "")
    task_id = str(blocker.get("task_id") or state.get("current_task") or "")
    task = next((item for item in state.get("tasks", []) or [] if item.get("task_id") == task_id), None)
    if not isinstance(task, dict):
        raise ValueError(f"reconciliation task not found: {task_id}")
    if task.get("status") != "reconciliation_required":
        raise ValueError(f"task is not awaiting reconciliation: {task_id}")

    relative_evidence = str(evidence.relative_to(Path(workspace).resolve()))
    safe_operation_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", operation_id).strip("._-")
    if not safe_operation_id:
        raise ValueError("invalid operation_id")
    receipt_relative = Path("audit") / "reconciliations" / f"{safe_operation_id}.json"
    receipt_path = Path(workspace) / receipt_relative
    expected_subject = {
        "blocker_id": blocker_id,
        "operation_id": operation_id,
        "attempt_id": str(blocker.get("attempt_id") or ""),
        "task_id": task_id,
        "outcome": outcome,
        "evidence_path": relative_evidence,
        "evidence_sha256": actual_sha256,
        "operator_identity": operator_identity.strip(),
        "note": note.strip(),
    }
    if receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid existing reconciliation receipt: {operation_id}") from exc
        if not isinstance(receipt, dict) or any(receipt.get(key) != value for key, value in expected_subject.items()):
            raise ValueError(f"conflicting reconciliation receipt: {operation_id}")
        if receipt.get("consumed") is not True or not str(receipt.get("receipt_id") or ""):
            raise ValueError(f"invalid existing reconciliation receipt: {operation_id}")
    else:
        receipt = {
            "version": 1,
            "receipt_id": f"reconciliation_{uuid4().hex}",
            **expected_subject,
            "consumed": True,
            "created_at": _now(),
        }
        if not WorkspaceStore(workspace).create_once(receipt_relative, receipt):
            # A peer may have won between the existence check and exclusive
            # publish. Re-enter once and validate the immutable winner.
            return reconcile_operation(
                workspace,
                operation_id=operation_id,
                outcome=outcome,
                evidence_path=evidence_path,
                evidence_sha256=evidence_sha256,
                operator_identity=operator_identity,
                note=note,
            )

    receipt_id = str(receipt["receipt_id"])

    if outcome == "confirmed_succeeded":
        task.update(apply_task_transition(task, "confirm_success", "done"))
        task["finished_at"] = _now()
        target_state, event = "running", "reconcile_success"
    elif outcome == "confirmed_failed":
        task.update(apply_task_transition(task, "confirm_failure", "failed"))
        task["finished_at"] = _now()
        task["message"] = "external operation confirmed failed"
        target_state, event = "failed", "reconcile_failure"
    else:
        task.update(apply_task_transition(task, "abandon", "stopped"))
        task["finished_at"] = _now()
        target_state, event = "stopped", "reconcile_abandoned"
    state = apply_transition(
        state,
        event,
        {
            "target_state": target_state,
            "consumption_evidence": {
                "receipt_id": receipt_id,
                "blocker_id": blocker_id,
                "consumed": True,
            },
        },
    )
    state["current_task"] = task_id
    state["last_reconciliation_receipt"] = str(receipt_relative)
    save_agent_state(workspace, state)
    attempt_id = str(receipt.get("attempt_id") or "")
    if attempt_id and attempt_id in load_attempt_journal(workspace):
        _update_attempt(
            workspace,
            attempt_id,
            event="reconciled",
            result_status="reconciled",
            finished_at=_now(),
        )
    receipt["receipt_path"] = str(receipt_relative)
    return receipt


def _validated_evidence_path(workspace: str | Path, evidence_path: str) -> Path:
    relative = Path(evidence_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("evidence_path must be workspace-relative")
    root = Path(workspace).resolve()
    candidate = root / relative
    # Reject symlinks at every existing path component before resolving.
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("evidence_path may not contain symlinks")
    resolved = candidate.resolve()
    if root not in resolved.parents or not resolved.is_file():
        raise ValueError("evidence_path must reference an existing workspace file")
    return resolved


def _append_snapshot(workspace: str | Path, snapshot: dict[str, Any]) -> None:
    WorkspaceStore(workspace).append_jsonl(ATTEMPT_JOURNAL, snapshot)


def _update_attempt(workspace: str | Path, attempt_id: str, *, event: str, **changes: Any) -> dict[str, Any]:
    current = load_attempt_journal(workspace).get(attempt_id)
    if current is None:
        raise KeyError(f"attempt not journaled: {attempt_id}")
    updated = dict(current)
    updated.update(changes)
    updated["event_id"] = f"journal_{uuid4().hex}"
    updated["event"] = event
    _append_snapshot(workspace, updated)
    return updated
