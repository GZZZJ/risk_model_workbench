import hashlib
import json
from pathlib import Path

import pytest
import risk_model_workbench.agent.recovery as recovery_module

from risk_model_workbench.agent.recovery import (
    begin_attempt,
    diagnose_recovery,
    load_attempt_journal,
    mark_attempt_dispatched,
    mark_attempt_result_committed,
    reconcile_operation,
    recovery_policy,
)
from risk_model_workbench.agent.eval import emit_scenario_evidence
from risk_model_workbench.agent.state import init_agent_state, load_agent_state, requeue_interrupted_task, save_agent_state
from risk_model_workbench.harness.errors import WorkspaceLockedError
from risk_model_workbench.agent.workspace_store import WorkspaceStore


def test_attempt_journal_records_required_transition_fields(tmp_path):
    begin_attempt(
        tmp_path,
        attempt_id="attempt_1",
        transition_id="transition_1",
        task_id="task_1",
        action_id="sample_check",
        invocation_hash="abc123",
        execution_semantics="idempotent_write",
    )
    mark_attempt_dispatched(tmp_path, "attempt_1")
    mark_attempt_result_committed(tmp_path, "attempt_1", transaction_id="txn_1")

    latest = load_attempt_journal(tmp_path)["attempt_1"]
    assert latest["attempt_id"] == "attempt_1"
    assert latest["transition_id"] == "transition_1"
    assert latest["invocation_hash"] == "abc123"
    assert latest["intent_status"] == "dispatched"
    assert latest["result_status"] == "committed"
    assert latest["transaction_id"] == "txn_1"
    assert latest["started_at"]
    assert latest["finished_at"]


@pytest.mark.parametrize(
    ("semantics", "expected"),
    [
        ("read_only", "retry"),
        ("idempotent_write", "retry"),
        ("non_idempotent_write", "reconciliation_required"),
        ("external_unknown", "reconciliation_required"),
    ],
)
def test_recovery_policy_never_retries_unsafe_execution(semantics, expected):
    assert recovery_policy(semantics, intent_status="dispatched", result_status="pending") == expected


@pytest.mark.parametrize("semantics", ["external_unknown", "non_idempotent_write"])
def test_unsafe_intent_only_attempt_is_never_automatically_requeued(semantics):
    assert recovery_policy(semantics, intent_status="recorded", result_status="pending") == "reconciliation_required"


def test_diagnose_detects_each_crash_shape_and_cross_file_divergence(tmp_path):
    begin_attempt(tmp_path, attempt_id="before_command", transition_id="t1", task_id="a", action_id="sample_check", invocation_hash="h1", execution_semantics="idempotent_write")
    begin_attempt(tmp_path, attempt_id="after_command", transition_id="t2", task_id="b", action_id="feature_prescreen", invocation_hash="h2", execution_semantics="external_unknown")
    mark_attempt_dispatched(tmp_path, "after_command")
    begin_attempt(tmp_path, attempt_id="after_result", transition_id="t3", task_id="c", action_id="report", invocation_hash="h3", execution_semantics="idempotent_write")
    mark_attempt_dispatched(tmp_path, "after_result")
    mark_attempt_result_committed(tmp_path, "after_result", transaction_id="txn_manifest")
    WorkspaceStore(tmp_path).write_yaml("version_state.yml", {"transaction_id": "txn_state"}, 0)
    WorkspaceStore(tmp_path).write_json("audit/artifact_manifest.json", {"transaction_id": "txn_manifest", "artifacts": []}, 0)

    result = diagnose_recovery(tmp_path)
    by_id = {item["attempt_id"]: item for item in result["attempts"]}
    assert by_id["before_command"]["recovery_action"] == "requeue"
    assert by_id["after_command"]["recovery_action"] == "reconciliation_required"
    assert by_id["after_result"]["recovery_action"] == "apply_receipt"
    assert result["transaction_divergence"]["detected"] is True


def test_runner_lock_rejects_concurrent_resume(tmp_path):
    with WorkspaceStore(tmp_path).runner_lock():
        with pytest.raises(WorkspaceLockedError):
            with WorkspaceStore(tmp_path).runner_lock():
                pass
    emit_scenario_evidence(
        workspace=tmp_path,
    )


def test_safe_missing_result_can_requeue_after_executor_marked_task_failed(tmp_path):
    plan = {
        "version": 2,
        "plan_id": "p",
        "plan_hash": "h",
        "registry_digest": "r",
        "tasks": [{"task_id": "task_1", "derived_metadata": {}, "invocation": {}}],
    }
    init_agent_state(tmp_path, project="/project", version_id="v1", agent_plan=plan)
    state = load_agent_state(tmp_path)
    state["status"] = "failed"
    state["tasks"][0].update({"status": "failed", "attempt_id": "attempt_1"})
    save_agent_state(tmp_path, state)

    recovered = requeue_interrupted_task(tmp_path, "task_1", attempt_id="attempt_1")

    assert recovered["status"] == "running"
    assert recovered["tasks"][0]["status"] == "pending"
    assert recovered["tasks"][0]["recovered_attempt_id"] == "attempt_1"


def test_stale_runner_lock_metadata_does_not_block_after_process_exit(tmp_path):
    lock = tmp_path / "audit" / "agent_runner.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("pid=999999999\n", encoding="utf-8")

    with WorkspaceStore(tmp_path).runner_lock():
        assert lock.read_text(encoding="utf-8").startswith("pid=")


def test_reconcile_requires_exact_evidence_hash_and_operator_and_does_not_create_domain_artifacts(tmp_path, monkeypatch):
    plan = {"version": 2, "plan_id": "p", "plan_hash": "h", "registry_digest": "r", "tasks": [{"task_id": "task_1", "derived_metadata": {"action_id": "feature_prescreen"}, "invocation": {"tool_name": "feature_prescreen_execute"}}]}
    init_agent_state(tmp_path, project="/project", version_id="v1", agent_plan=plan)
    state = load_agent_state(tmp_path)
    state["status"] = "reconciliation_required"
    state["tasks"][0]["status"] = "reconciliation_required"
    state["blocker"] = {
        "blocker_type": "reconciliation",
        "blocker_id": "reconciliation:attempt_1",
        "attempt_id": "attempt_1",
        "task_id": "task_1",
        "external_operations": [{"operation_id": "op_1"}],
        "next_safe_action": {"action": "reconcile_external_operation", "required_evidence": "operator evidence"},
    }
    save_agent_state(tmp_path, state)
    evidence = tmp_path / "audit" / "operator_evidence.json"
    evidence.write_text('{"remote_status":"succeeded"}\n', encoding="utf-8")
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="sha256"):
        reconcile_operation(tmp_path, operation_id="op_1", outcome="confirmed_succeeded", evidence_path="audit/operator_evidence.json", evidence_sha256="0" * 64, operator_identity="operator@example", note="checked remote job")
    with pytest.raises(ValueError, match="operator_identity"):
        reconcile_operation(tmp_path, operation_id="op_1", outcome="confirmed_succeeded", evidence_path="audit/operator_evidence.json", evidence_sha256=digest, operator_identity="", note="checked remote job")

    real_save = recovery_module.save_agent_state
    failures = 1

    def crash_after_receipt(workspace, payload):
        nonlocal failures
        if failures:
            failures -= 1
            raise RuntimeError("crash after reconciliation receipt")
        return real_save(workspace, payload)

    monkeypatch.setattr(recovery_module, "save_agent_state", crash_after_receipt)
    with pytest.raises(RuntimeError, match="crash after reconciliation receipt"):
        reconcile_operation(tmp_path, operation_id="op_1", outcome="confirmed_succeeded", evidence_path="audit/operator_evidence.json", evidence_sha256=digest, operator_identity="operator@example", note="checked remote job")
    assert load_agent_state(tmp_path)["status"] == "reconciliation_required"
    with pytest.raises(ValueError, match="conflicting reconciliation receipt"):
        reconcile_operation(tmp_path, operation_id="op_1", outcome="confirmed_succeeded", evidence_path="audit/operator_evidence.json", evidence_sha256=digest, operator_identity="operator@example", note="different note")

    receipt = reconcile_operation(tmp_path, operation_id="op_1", outcome="confirmed_succeeded", evidence_path="audit/operator_evidence.json", evidence_sha256=digest, operator_identity="operator@example", note="checked remote job")
    assert receipt["evidence_sha256"] == digest
    assert load_agent_state(tmp_path)["tasks"][0]["status"] == "done"
    assert not (tmp_path / "modeling").exists()
    assert not (tmp_path / "evaluation").exists()
    assert not (tmp_path / "reports").exists()
