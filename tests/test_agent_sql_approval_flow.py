from __future__ import annotations

import json
import hashlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from risk_model_workbench.agent.eval import emit_scenario_evidence

from risk_model_workbench.agent.approvals import (
    ApprovalBindingError,
    approval_by_id,
    approve_request,
    build_approval_subject,
    consume_approval,
    ensure_subject_approval,
    load_approvals,
    reject_request,
)
from risk_model_workbench.agent.executor import run_agent
from risk_model_workbench.agent.plan import canonical_plan_hash, save_agent_plan
from risk_model_workbench.agent.plan import bind_agent_plan
from risk_model_workbench.agent.state import init_agent_state, load_agent_state
from risk_model_workbench.agent.workspace_store import WorkspaceStore
from risk_model_workbench.cli import main
from risk_model_workbench.dp_feather import (
    ExternalOutcomeUnknown,
    execute_dp_sql,
    fetch_dp_query_to_feather,
    write_external_operation_intent,
    write_external_operation_receipt,
)
from risk_model_workbench.request import parse_model_request as _initialize_request_package
from risk_model_workbench.planning import create_execution_plan
from risk_model_workbench.data.sql_evidence import write_sql_evidence
from risk_model_workbench.harness.errors import SQL_APPROVAL_REQUIRED
from risk_model_workbench.harness.invocation import ActionInvocation
from risk_model_workbench.harness.runtime import ActionAttempt, action_attempt, stage_action_done
from risk_model_workbench.harness.runtime import stage_action_failed
from risk_model_workbench.harness.tools import TOOL_REGISTRY, registry_digest
from risk_model_workbench.application.context import VersionContext
from risk_model_workbench.application.handlers import feature_selection as feature_actions
from risk_model_workbench.state import create_version_state, save_version_state


def test_remote_feature_plan_uses_fixed_prepare_execute_pairs():
    plan = create_execution_plan(_request(), Path("projects/2026-05-fujie-gcard-v1"))
    tasks = {task["task_id"]: task for task in plan["tasks"]}

    assert tasks["feature_prescreen_prepare"]["command"]["args"][-1] == "--dry-run-sql"
    assert tasks["feature_prescreen_execute"]["command"]["args"][-1] == "--sql-approved"
    assert tasks["feature_prescreen_execute"]["depends_on"] == ["feature_prescreen_prepare"]
    assert "--execute" in tasks["build_wide_sql_execute"]["command"]["args"]
    assert tasks["feature_refine_execute"]["depends_on"] == ["feature_refine_prepare"]


def test_local_feather_plan_contains_no_remote_execute_task():
    request = _request(data_source_mode="local_feather", sample_location="data/raw/sample.feather")
    plan = create_execution_plan(request, Path("projects/2026-05-fujie-gcard-v1"))
    task_ids = {task["task_id"] for task in plan["tasks"]}

    assert not {task_id for task_id in task_ids if task_id.endswith("_execute")}

    bound = bind_agent_plan(plan, project_dir="projects/2026-05-fujie-gcard-v1", version_id="local_v1")
    remote_flags = {"--dry-run-sql", "--sql-approved", "--execute"}
    local_tasks = [task for task in bound["tasks"] if task["task_id"] in {"feature_prescreen", "build_wide_sql", "feature_refine"}]
    assert {task["tool_name"] for task in local_tasks} == {
        "feature_prescreen_local",
        "build_wide_sql_local",
        "feature_refine_local",
    }
    assert all(not remote_flags.intersection(task["command"]["args"]) for task in local_tasks)


def test_approval_subject_is_exact_bound_and_consumed_once(tmp_path):
    workspace = _sql_workspace(tmp_path)
    subject = build_approval_subject(
        workspace,
        project="projects/demo",
        version_id="demo_v1",
        task_id="feature_refine_execute",
        invocation_hash="a" * 64,
        operation_id="feature_refine_execute",
    )
    approval = ensure_subject_approval(workspace, subject, reason="sql_review_required")
    approve_request(workspace, approval["approval_id"], approved_by="pm", note="reviewed")

    receipt = consume_approval(workspace, approval["approval_id"], subject, consumed_by="agent")

    assert receipt["subject_hash"] == approval["subject_hash"]
    assert load_approvals(workspace)["approvals"][0]["status"] == "consumed"
    with pytest.raises(ValueError, match="already consumed"):
        consume_approval(workspace, approval["approval_id"], subject, consumed_by="agent")


def test_subject_drift_revokes_old_approval(tmp_path):
    workspace = _sql_workspace(tmp_path, stage="build_wide_sql")
    subject = build_approval_subject(
        workspace,
        project="projects/demo",
        version_id="demo_v1",
        task_id="build_wide_sql_execute",
        invocation_hash="b" * 64,
        operation_id="build_wide_sql_execute",
    )
    old = ensure_subject_approval(workspace, subject, reason="sql_review_required")
    approve_request(workspace, old["approval_id"], approved_by="pm")
    (workspace / "queries" / "generated" / "query.sql").write_text("select 2", encoding="utf-8")
    manifest_path = workspace / "queries" / "sql_evidence_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entries"][0]["sql_sha256"] = hashlib.sha256(b"select 2").hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    changed = build_approval_subject(
        workspace,
        project="projects/demo",
        version_id="demo_v1",
        task_id="build_wide_sql_execute",
        invocation_hash="b" * 64,
        operation_id="build_wide_sql_execute",
    )
    new = ensure_subject_approval(workspace, changed, reason="sql_review_required")

    approvals = load_approvals(workspace)["approvals"]
    assert old["approval_id"] != new["approval_id"]
    assert next(item for item in approvals if item["approval_id"] == old["approval_id"])["status"] == "revoked"
    assert new["status"] == "pending"


def test_rejected_approval_cannot_be_consumed(tmp_path):
    workspace = _sql_workspace(tmp_path, stage="feature_prescreen")
    subject = build_approval_subject(
        workspace,
        project="projects/demo",
        version_id="demo_v1",
        task_id="feature_prescreen_execute",
        invocation_hash="c" * 64,
        operation_id="feature_prescreen_execute",
    )
    approval = ensure_subject_approval(workspace, subject, reason="sql_review_required")
    reject_request(workspace, approval["approval_id"], rejected_by="pm", note="unsafe")

    with pytest.raises(ValueError, match="not approved"):
        consume_approval(workspace, approval["approval_id"], subject, consumed_by="agent")


def test_dp_intent_and_receipt_are_immutable_and_unknown_outcome_is_explicit(tmp_path):
    workspace = tmp_path / "version"
    intent = write_external_operation_intent(
        workspace,
        operation_id="op_001",
        subject_hash="d" * 64,
        sql_sha256="e" * 64,
    )
    receipt = write_external_operation_receipt(workspace, operation_id="op_001", status="succeeded")

    assert json.loads(intent.read_text(encoding="utf-8"))["status"] == "intent_recorded"
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "succeeded"
    with pytest.raises(ExternalOutcomeUnknown):
        write_external_operation_receipt(workspace, operation_id="op_missing", status="unknown")
    assert WorkspaceStore(workspace).create_once(
        "audit/external_operations/op_001.receipt.json", {"status": "overwritten"}
    ) is False


def test_invalid_sql_manifest_entry_fails_closed(tmp_path):
    workspace = _sql_workspace(tmp_path)
    manifest_path = workspace / "queries" / "sql_evidence_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entries"][0]["path"] = "../outside.sql"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid SQL evidence path"):
        build_approval_subject(
            workspace,
            project="projects/demo",
            version_id="demo_v1",
            task_id="feature_refine_execute",
            invocation_hash="f" * 64,
            operation_id="feature_refine_execute",
        )


def test_agent_prepare_approval_consume_execute_is_exactly_bound(tmp_path):
    project, workspace, version_id, plan = _agent_workspace(tmp_path)
    calls: list[list[str]] = []

    def runner(argv):
        calls.append(list(argv))
        if "--dry-run-sql" in argv:
            write_sql_evidence(
                workspace,
                "select 1 as feature_value",
                source="test",
                purpose="feature_refine",
                stage="feature_refine",
                sql_kind="generated",
                name="feature_refine.sql",
            )
            stage_action_done(
                workspace,
                "feature_refine",
                scaffold=True,
                message="SQL review ready",
                failure_code=SQL_APPROVAL_REQUIRED,
            )
            return 0
        stage_action_done(workspace, "feature_refine", message="approved execution completed")
        return 0

    waiting = run_agent(project, version_id, runner=runner)
    approval = load_approvals(workspace)["approvals"][0]
    assert waiting["status"] == "waiting_for_approval"
    assert len(calls) == 1
    assert load_agent_state(workspace)["tasks"][0]["status"] == "review_ready"

    approve_request(workspace, approval["approval_id"], approved_by="pm", note="reviewed exact SQL")
    finished = run_agent(project, version_id, runner=runner)
    state = load_agent_state(workspace)

    assert len(calls) == 2
    assert "--sql-approved" in calls[1]
    assert [task["status"] for task in state["tasks"]] == ["done", "done"]
    assert load_approvals(workspace)["approvals"][0]["status"] == "consumed"
    assert finished["status"] in {"done", "done_with_gaps"}
    emit_scenario_evidence(
        state_pairs=[(waiting, finished)],
        workspace=workspace,
        runner_calls=calls,
    )


@pytest.mark.parametrize("pair", ["feature_prescreen", "build_wide_sql", "feature_refine"])
def test_each_remote_prepare_pair_reaches_review_ready(pair, tmp_path):
    project, workspace, version_id, _ = _agent_workspace(tmp_path, pair=pair)

    def runner(_argv):
        write_sql_evidence(
            workspace,
            f"select 1 /* {pair} */",
            source="test",
            purpose=pair,
            stage=pair,
            sql_kind="generated",
            name=f"{pair}.sql",
        )
        stage_action_done(workspace, pair, scaffold=True, failure_code=SQL_APPROVAL_REQUIRED)
        return 0

    state = run_agent(project, version_id, runner=runner)
    statuses = {task["task_id"]: task["status"] for task in state["tasks"]}

    assert state["status"] == "waiting_for_approval"
    assert statuses[f"{pair}_prepare"] == "review_ready"
    assert statuses[f"{pair}_execute"] == "paused"


def test_train_cannot_run_while_sql_execute_waits_for_approval(tmp_path):
    project, workspace, version_id, _ = _agent_workspace(tmp_path, include_train=True)
    calls: list[list[str]] = []

    def runner(argv):
        calls.append(list(argv))
        assert argv[0] != "train"
        write_sql_evidence(
            workspace,
            "select 1",
            source="test",
            purpose="feature_refine",
            stage="feature_refine",
            sql_kind="generated",
            name="feature_refine.sql",
        )
        stage_action_done(workspace, "feature_refine", scaffold=True, failure_code=SQL_APPROVAL_REQUIRED)
        return 0

    state = run_agent(project, version_id, runner=runner)

    assert state["status"] == "waiting_for_approval"
    assert len(calls) == 1
    assert next(task for task in state["tasks"] if task["task_id"] == "train_baseline")["status"] == "pending"


def test_rejected_agent_approval_remains_blocked_and_does_not_execute(tmp_path):
    project, workspace, version_id, _ = _agent_workspace(tmp_path)
    calls: list[list[str]] = []

    def runner(argv):
        calls.append(list(argv))
        write_sql_evidence(
            workspace,
            "select 1",
            source="test",
            purpose="feature_refine",
            stage="feature_refine",
            sql_kind="generated",
            name="feature_refine.sql",
        )
        stage_action_done(workspace, "feature_refine", scaffold=True, failure_code=SQL_APPROVAL_REQUIRED)
        return 0

    waiting = run_agent(project, version_id, runner=runner)
    approval_id = waiting["blocker"]["approval_id"]
    reject_request(workspace, approval_id, rejected_by="pm", note="unsafe")

    still_waiting = run_agent(project, version_id, runner=runner)
    assert still_waiting["status"] == "waiting_for_approval"
    assert len(calls) == 1
    emit_scenario_evidence(
        state_pairs=[(waiting, still_waiting)],
        workspace=workspace,
        runner_calls=calls,
    )


def test_agent_approval_drift_creates_replacement_and_does_not_execute(tmp_path):
    project, workspace, version_id, _ = _agent_workspace(tmp_path)
    calls: list[list[str]] = []

    def runner(argv):
        calls.append(list(argv))
        write_sql_evidence(
            workspace,
            "select 1",
            source="test",
            purpose="feature_refine",
            stage="feature_refine",
            sql_kind="generated",
            name="feature_refine.sql",
        )
        stage_action_done(workspace, "feature_refine", scaffold=True, failure_code=SQL_APPROVAL_REQUIRED)
        return 0

    waiting = run_agent(project, version_id, runner=runner)
    old_id = waiting["blocker"]["approval_id"]
    approve_request(workspace, old_id, approved_by="pm")
    (workspace / "configs_runtime" / "project.yml").write_text("project: changed\n", encoding="utf-8")

    drifted = run_agent(project, version_id, runner=runner)
    assert drifted["status"] == "waiting_for_approval"
    assert drifted["blocker"]["approval_id"] != old_id
    assert len(calls) == 1
    statuses = {item["approval_id"]: item["status"] for item in load_approvals(workspace)["approvals"]}
    assert statuses[old_id] == "revoked"
    assert statuses[drifted["blocker"]["approval_id"]] == "pending"
    emit_scenario_evidence(
        state_pairs=[(waiting, drifted)],
        workspace=workspace,
        runner_calls=calls,
    )


def test_unregistered_sql_tamper_revokes_approval_and_fails_closed(tmp_path):
    project, workspace, version_id, _ = _agent_workspace(tmp_path)
    calls: list[list[str]] = []

    def runner(argv):
        calls.append(list(argv))
        write_sql_evidence(
            workspace,
            "select 1",
            source="test",
            purpose="feature_refine",
            stage="feature_refine",
            sql_kind="generated",
            name="feature_refine.sql",
        )
        stage_action_done(workspace, "feature_refine", scaffold=True, failure_code=SQL_APPROVAL_REQUIRED)
        return 0

    waiting = run_agent(project, version_id, runner=runner)
    approval_id = waiting["blocker"]["approval_id"]
    approve_request(workspace, approval_id, approved_by="pm")
    (workspace / "queries" / "generated" / "feature_refine.sql").write_text("select 999", encoding="utf-8")

    blocked = run_agent(project, version_id, runner=runner)
    assert blocked["status"] == "waiting_for_approval"
    assert "sql_evidence_invalid" in blocked["blocker"]["reason"]
    assert approval_by_id(workspace, approval_id)["status"] == "revoked"
    assert len(calls) == 1


def test_unknown_remote_outcome_enters_reconciliation(tmp_path):
    project, workspace, version_id, _ = _agent_workspace(tmp_path)
    approval_calls: list[list[str]] = []

    def prepare_runner(argv):
        approval_calls.append(list(argv))
        write_sql_evidence(
            workspace,
            "select 1",
            source="test",
            purpose="feature_refine",
            stage="feature_refine",
            sql_kind="generated",
            name="feature_refine.sql",
        )
        stage_action_done(workspace, "feature_refine", scaffold=True, failure_code=SQL_APPROVAL_REQUIRED)
        return 0

    waiting = run_agent(project, version_id, runner=prepare_runner)
    approve_request(workspace, waiting["blocker"]["approval_id"], approved_by="pm")

    def unknown_runner(argv):
        approval_calls.append(list(argv))
        stage_action_failed(
            workspace,
            "feature_refine",
            "external operation outcome is unknown: feature_refine_execute",
            failure_code="external_outcome_unknown",
        )
        return 1

    reconcilee = run_agent(project, version_id, runner=unknown_runner)
    assert reconcilee["status"] == "reconciliation_required"
    blocker = reconcilee["blocker"]
    assert blocker["task_id"] == "feature_refine_execute"
    assert blocker["approval_id"]
    assert blocker["subject_hash"]
    assert blocker["consumption_receipt"]
    assert blocker["consumption_receipt_status"] == "consumed"
    assert blocker["parent_operation_id"] == "feature_refine_execute"
    assert len(approval_calls) == 2
    assert run_agent(project, version_id, runner=unknown_runner)["status"] == "reconciliation_required"
    assert len(approval_calls) == 2


def test_external_boundary_submits_immutable_approved_sql(monkeypatch, tmp_path):
    workspace, approval, receipt, subject = _consumed_sql_approval(tmp_path, "build_wide_sql_execute")
    submitted: list[str] = []

    class Result:
        def execute(self):
            return None

    class Client:
        def sql(self, sql):
            submitted.append(sql)
            return Result()

        def stop(self):
            return None

    _install_fake_tmlpatch(monkeypatch, Client)
    attempt = _approved_attempt(workspace, approval, receipt, subject)
    with action_attempt(attempt):
        result = execute_dp_sql(
            project_dir=tmp_path,
            sql="  select   1  ",
            operation_id="build_wide_sql_execute",
            description="test",
            metadata_path=workspace / "execution.json",
            sql_approved=True,
            audit_workspace=workspace,
        )

    assert submitted == ["select 1"]
    assert result["sql_sha256"] == hashlib.sha256(b"select 1").hexdigest()


def test_build_wide_handler_dispatches_exact_consumed_sql(monkeypatch, tmp_path):
    context, invocation, attempt = _wide_handler_context(tmp_path, generated_sql="  select   1  ")
    submitted: list[str] = []

    class Result:
        def execute(self):
            return None

    class Client:
        def sql(self, sql):
            submitted.append(sql)
            return Result()

        def stop(self):
            return None

    _install_fake_tmlpatch(monkeypatch, Client)
    with action_attempt(attempt):
        result = feature_actions.run_build_wide_sql(invocation, context, attempt.attempt_id)

    assert result.status == "done"
    assert submitted == ["select 1"]
    assert (context.workspace / "audit/external_operations/build_wide_sql_execute.intent.json").exists()
    assert (context.workspace / "audit/external_operations/build_wide_sql_execute.receipt.json").exists()


def test_build_wide_handler_blocks_generated_sql_drift_before_dispatch(monkeypatch, tmp_path):
    context, invocation, attempt = _wide_handler_context(tmp_path, generated_sql="select 2")
    submitted: list[str] = []

    class Client:
        def sql(self, sql):
            submitted.append(sql)

        def stop(self):
            return None

    _install_fake_tmlpatch(monkeypatch, Client)
    with action_attempt(attempt):
        result = feature_actions.run_build_wide_sql(invocation, context, attempt.attempt_id)

    assert result.status == "failed"
    assert submitted == []
    assert not (context.workspace / "audit/external_operations/build_wide_sql_execute.intent.json").exists()


def test_build_wide_handler_marks_unknown_external_outcome_for_reconciliation(monkeypatch, tmp_path):
    context, invocation, attempt = _wide_handler_context(tmp_path, generated_sql="select 1")

    class Client:
        def sql(self, _sql):
            raise RuntimeError("connection dropped after submit")

        def stop(self):
            return None

    _install_fake_tmlpatch(monkeypatch, Client)
    with action_attempt(attempt):
        result = feature_actions.run_build_wide_sql(invocation, context, attempt.attempt_id)

    assert result.status == "failed"
    assert result.failure_code == "external_outcome_unknown"
    assert result.next_required_action == "reconciliation"
    assert (context.workspace / "audit/external_operations/build_wide_sql_execute.intent.json").exists()
    assert not (context.workspace / "audit/external_operations/build_wide_sql_execute.receipt.json").exists()


@pytest.mark.parametrize("domain", ["feature_prescreen", "feature_refine"])
def test_feature_execute_handler_dispatches_exact_consumed_sql(monkeypatch, tmp_path, domain):
    context, invocation, attempt = _feature_execute_handler_context(
        monkeypatch, tmp_path, domain=domain, candidate_sql="  select   1  "
    )
    submitted: list[str] = []

    class Result:
        def execute(self):
            return None

    class Client:
        def sql(self, sql):
            submitted.append(sql)
            return Result()

        def stop(self):
            return None

    _install_fake_tmlpatch(monkeypatch, Client)
    with action_attempt(attempt):
        result = _run_feature_execute_handler(domain, invocation, context, attempt)

    assert result.status == "done"
    assert submitted == ["select 1"]
    operation = f"{domain}_execute"
    assert (context.workspace / f"audit/external_operations/{operation}.intent.json").exists()
    assert (context.workspace / f"audit/external_operations/{operation}.receipt.json").exists()


@pytest.mark.parametrize("domain", ["feature_prescreen", "feature_refine"])
def test_feature_execute_handler_blocks_sql_drift_before_dispatch(monkeypatch, tmp_path, domain):
    context, invocation, attempt = _feature_execute_handler_context(
        monkeypatch, tmp_path, domain=domain, candidate_sql="select 2"
    )
    submitted: list[str] = []

    class Client:
        def sql(self, sql):
            submitted.append(sql)

        def stop(self):
            return None

    _install_fake_tmlpatch(monkeypatch, Client)
    with action_attempt(attempt):
        result = _run_feature_execute_handler(domain, invocation, context, attempt)

    assert result.status == "failed"
    assert submitted == []
    assert not (context.workspace / f"audit/external_operations/{domain}_execute.intent.json").exists()


@pytest.mark.parametrize("domain", ["feature_prescreen", "feature_refine"])
def test_feature_execute_handler_marks_unknown_outcome_for_reconciliation(monkeypatch, tmp_path, domain):
    context, invocation, attempt = _feature_execute_handler_context(
        monkeypatch, tmp_path, domain=domain, candidate_sql="select 1"
    )

    class Client:
        def sql(self, _sql):
            raise RuntimeError("connection dropped after submit")

        def stop(self):
            return None

    _install_fake_tmlpatch(monkeypatch, Client)
    with action_attempt(attempt):
        result = _run_feature_execute_handler(domain, invocation, context, attempt)

    assert result.status == "failed"
    assert result.failure_code == "external_outcome_unknown"
    assert result.next_required_action == "reconciliation"
    assert (context.workspace / f"audit/external_operations/{domain}_execute.intent.json").exists()
    assert not (context.workspace / f"audit/external_operations/{domain}_execute.receipt.json").exists()


def test_external_boundary_blocks_regenerated_sql_drift(monkeypatch, tmp_path):
    workspace, approval, receipt, subject = _consumed_sql_approval(tmp_path, "build_wide_sql_execute")
    submitted: list[str] = []

    class Client:
        def sql(self, sql):
            submitted.append(sql)

        def stop(self):
            return None

    _install_fake_tmlpatch(monkeypatch, Client)
    with action_attempt(_approved_attempt(workspace, approval, receipt, subject)):
        with pytest.raises(ValueError, match="candidate SQL is not present"):
            execute_dp_sql(
                project_dir=tmp_path,
                sql="select 2",
                operation_id="build_wide_sql_execute",
                description="test",
                metadata_path=workspace / "execution.json",
                sql_approved=True,
                audit_workspace=workspace,
            )

    assert submitted == []
    assert not (workspace / "audit" / "external_operations" / "build_wide_sql_execute.intent.json").exists()


def test_external_boundary_rejects_tampered_consumed_subject(monkeypatch, tmp_path):
    workspace, approval, receipt, subject = _consumed_sql_approval(tmp_path, "build_wide_sql_execute")
    evil_path = workspace / "queries" / "generated" / "evil.sql"
    evil_path.write_text("select 999", encoding="utf-8")
    ledger_path = workspace / "audit" / "approvals.yml"
    ledger = yaml.safe_load(ledger_path.read_text(encoding="utf-8"))
    record = next(item for item in ledger["approvals"] if item["approval_id"] == approval["approval_id"])
    record["subject"]["sql_files"] = [
        {"path": "queries/generated/evil.sql", "sha256": hashlib.sha256(b"select 999").hexdigest()}
    ]
    ledger_path.write_text(yaml.safe_dump(ledger, allow_unicode=True, sort_keys=False), encoding="utf-8")
    submitted: list[str] = []

    class Client:
        def sql(self, sql):
            submitted.append(sql)

        def stop(self):
            return None

    _install_fake_tmlpatch(monkeypatch, Client)
    with action_attempt(_approved_attempt(workspace, approval, receipt, subject)):
        with pytest.raises(ApprovalBindingError, match="subject drift"):
            execute_dp_sql(
                project_dir=tmp_path,
                sql="select 999",
                operation_id="build_wide_sql_execute",
                description="test",
                metadata_path=workspace / "execution.json",
                sql_approved=True,
                audit_workspace=workspace,
            )

    assert submitted == []
    assert not (workspace / "audit" / "external_operations" / "build_wide_sql_execute.intent.json").exists()


def test_explicit_worker_context_records_unknown_external_outcome(monkeypatch, tmp_path):
    workspace, approval, receipt, subject = _consumed_sql_approval(tmp_path, "feature_prescreen_execute")

    class Client:
        def sql(self, _sql):
            raise RuntimeError("connection dropped after submit")

        def stop(self):
            return None

    _install_fake_tmlpatch(monkeypatch, Client)
    context = {
        "workspace": str(workspace),
        "attempt_id": "attempt_worker",
        "task_id": "feature_prescreen_execute",
        "approval_id": approval["approval_id"],
        "subject_hash": approval["subject_hash"],
        "consumption_receipt": str(Path("audit") / "approval_consumptions" / f"{approval['approval_id']}.json"),
        "parent_operation_id": "feature_prescreen_execute",
    }

    with pytest.raises(ExternalOutcomeUnknown):
        fetch_dp_query_to_feather(
            project_dir=tmp_path,
            sql="select 1",
            dataset_id="table_a",
            description="test",
            feather_path=tmp_path / "table_a.feather",
            metadata_path=tmp_path / "table_a.json",
            sql_approved=True,
            external_operation_context=context,
        )

    intents = list((workspace / "audit" / "external_operations").glob("*.intent.json"))
    assert len(intents) == 1
    payload = json.loads(intents[0].read_text(encoding="utf-8"))
    assert payload["attempt_id"] == "attempt_worker"
    assert payload["parent_operation_id"] == "feature_prescreen_execute"
    assert not list((workspace / "audit" / "external_operations").glob("*.receipt.json"))


@pytest.mark.parametrize(
    ("worker_error", "expected_error"),
    [
        (ExternalOutcomeUnknown("external operation outcome is unknown: subop"), ExternalOutcomeUnknown),
        (ApprovalBindingError("candidate SQL is not present in the consumed approval subject"), ApprovalBindingError),
    ],
)
def test_prescreen_worker_receives_context_and_fails_closed(monkeypatch, tmp_path, worker_error, expected_error):
    import risk_model_workbench.batch_feature_select as batch
    import risk_model_workbench.dp_feather as dp

    context = {
        "workspace": str(tmp_path / "version"),
        "attempt_id": "attempt_pool",
        "task_id": "feature_prescreen_execute",
        "approval_id": "approval_1",
        "subject_hash": "a" * 64,
        "consumption_receipt": "audit/approval_consumptions/approval_1.json",
        "parent_operation_id": "feature_prescreen_execute",
    }
    settings = SimpleNamespace(
        feature_columns="features.csv",
        output_dir="out",
        dp_data_dir="dp_data",
        dp_metadata_dir="dp_meta",
        d01_thresholds={},
        d02_psi_threshold=0.2,
        partition_col="ds",
        sample_where="1=1",
        split_col="split",
        train_value="DEV",
        valid_value="OOT",
        train_partitions=["202601"],
        valid_partitions=["202602"],
        target_col="target",
        workers=2,
        round_num=1,
        random_seed=7,
    )
    submitted: list[dict] = []

    class Future:
        def result(self):
            raise worker_error

    class Executor:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, _fn, **kwargs):
            submitted.append(kwargs)
            return Future()

    monkeypatch.setattr(batch, "load_batch_settings", lambda _project, _args: settings)
    monkeypatch.setattr(batch, "find_feature_select_code_dir", lambda *_args: tmp_path)
    monkeypatch.setattr(batch, "load_feature_map", lambda _path: {"mart.table_a": ["f1"]})
    monkeypatch.setattr(batch, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(batch, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(dp, "external_operation_context_for_current_attempt", lambda: context)

    with pytest.raises(expected_error, match="mart.table_a"):
        batch.run_prescreen_service(
            project_dir=tmp_path,
            stage="feature_prescreen",
            refresh_dp_cache=True,
            sql_approved=True,
        )

    assert submitted[0]["external_operation_context"] == context


def test_agent_approval_cli_rejects_and_duplicate_approve_is_stable(tmp_path, capsys):
    project, workspace, version_id, _ = _agent_workspace(tmp_path)

    def runner(_argv):
        write_sql_evidence(
            workspace,
            "select 1",
            source="test",
            purpose="feature_refine",
            stage="feature_refine",
            sql_kind="generated",
            name="feature_refine.sql",
        )
        stage_action_done(workspace, "feature_refine", scaffold=True, failure_code=SQL_APPROVAL_REQUIRED)
        return 0

    waiting = run_agent(project, version_id, runner=runner)
    approval_id = waiting["blocker"]["approval_id"]
    approve_argv = [
        "agent", "approve", "--project", str(project), "--version-id", version_id,
        "--approval-id", approval_id, "--approved-by", "pm",
    ]
    assert main(approve_argv) == 0
    assert main(approve_argv) == 1
    assert "approval is not pending" in capsys.readouterr().out

    project2, workspace2, version_id2, _ = _agent_workspace(tmp_path / "reject")

    def runner2(_argv):
        write_sql_evidence(
            workspace2,
            "select 1",
            source="test",
            purpose="feature_refine",
            stage="feature_refine",
            sql_kind="generated",
            name="feature_refine.sql",
        )
        stage_action_done(workspace2, "feature_refine", scaffold=True, failure_code=SQL_APPROVAL_REQUIRED)
        return 0

    waiting2 = run_agent(project2, version_id2, runner=runner2)
    reject_id = waiting2["blocker"]["approval_id"]
    assert main([
        "agent", "reject", "--project", str(project2), "--version-id", version_id2,
        "--approval-id", reject_id, "--rejected-by", "pm", "--note", "unsafe",
    ]) == 0
    assert approval_by_id(workspace2, reject_id)["status"] == "rejected"


def _agent_workspace(tmp_path, *, include_train=False, pair="feature_refine"):
    project = tmp_path / "project"
    for directory in ["configs", "queries", "reports", "versions"]:
        (project / directory).mkdir(parents=True, exist_ok=True)
    (project / "project.yml").write_text(
        yaml.safe_dump(
            {
                "project": {"name": "demo", "display_name": "Demo", "project_key": "demo"},
                "data": {"source_table": "demo.source", "id_columns": ["uid"], "target_column": "target", "period_column": "ds"},
                "segments": [{"name": "all", "display_name": "All", "filter": None}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    version_id = "demo_sql_v1"
    assert main(["version", "init", "--project", str(project), "--workflow", "feature_selection", "--version-id", version_id]) == 0
    workspace = project / "versions" / version_id
    (workspace / "configs_runtime").mkdir(parents=True, exist_ok=True)
    (workspace / "configs_runtime" / "project.yml").write_text("project: demo\n", encoding="utf-8")
    prepare = _bound_task(f"{pair}_prepare", f"{pair}_prepare", [], str(project), version_id)
    execute = _bound_task(f"{pair}_execute", f"{pair}_execute", [f"{pair}_prepare"], str(project), version_id)
    tasks = [prepare, execute]
    if include_train:
        tasks.append(
            _bound_task(
                "train_baseline",
                "train_baseline",
                [f"{pair}_execute"],
                str(project),
                version_id,
                params={"experiment": "baseline"},
            )
        )
    plan = {
        "version": 2,
        "plan_id": "sql_test_plan",
        "project": str(project),
        "version_id": version_id,
        "workflow": "feature_selection",
        "tasks": tasks,
        "registry_digest": registry_digest(),
    }
    plan["plan_hash"] = canonical_plan_hash(plan)
    save_agent_plan(workspace, plan)
    init_agent_state(workspace, project=str(project), version_id=version_id, agent_plan=plan)
    return project, workspace, version_id, plan


def _bound_task(tool_name, task_id, depends_on, project, version_id, *, params=None):
    invocation = ActionInvocation(tool_name, params or {}, project, version_id)
    spec = TOOL_REGISTRY[tool_name]
    return {
        "task_id": task_id,
        "depends_on": depends_on,
        "invocation": invocation.canonical_payload(),
        "invocation_hash": invocation.digest(),
        "command": {"executable": "rmw", "args": spec.render_argv(invocation)},
        "action_id": spec.action_id,
        "tool_name": spec.name,
        "derived_metadata": {
            "action_id": spec.action_id,
            "permission": spec.permission,
            "requires_approval": spec.requires_approval,
            "allowed_for_auditor": spec.allowed_for_auditor,
            "execution_semantics": spec.execution_semantics,
            "approval_type": spec.approval_type,
        },
    }


def _request(**overrides):
    metadata = {
        "request_id": "sql-two-phase-test",
        "project": "demo",
        "workflow": "feature_selection",
        "target_column": "target",
        "id_columns": ["uid"],
        "split_column": "ds",
        "data_source_mode": "remote_table",
        "feature_selection": {"rounds": ["prescreen", "refine"]},
        "experiments": [{"name": "baseline"}],
        "evaluation": {"metrics": ["auc"], "champions": []},
        "reports": {"outputs": ["model_report.md"]},
    }
    metadata.update(overrides)
    return {"path": "/tmp/sql-two-phase-test.md", "metadata": metadata, "body": ""}


def _sql_workspace(tmp_path, *, stage="feature_refine"):
    workspace = tmp_path / "version"
    (workspace / "configs_runtime").mkdir(parents=True)
    (workspace / "queries" / "generated").mkdir(parents=True)
    (workspace / "configs_runtime" / "project.yml").write_text("project: demo\n", encoding="utf-8")
    sql_path = workspace / "queries" / "generated" / "query.sql"
    sql_path.write_text("select 1", encoding="utf-8")
    (workspace / "queries" / "sql_evidence_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "entries": [
                    {
                        "path": "queries/generated/query.sql",
                        "sql_sha256": hashlib.sha256(b"select 1").hexdigest(),
                        "stage": stage,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return workspace


def _consumed_sql_approval(tmp_path, operation_id):
    workspace = _sql_workspace(tmp_path, stage=operation_id.removesuffix("_execute"))
    subject = build_approval_subject(
        workspace,
        project="projects/demo",
        version_id="demo_v1",
        task_id=operation_id,
        invocation_hash="a" * 64,
        operation_id=operation_id,
    )
    approval = ensure_subject_approval(workspace, subject, reason="test")
    approve_request(workspace, approval["approval_id"], approved_by="pm")
    receipt = consume_approval(workspace, approval["approval_id"], subject, consumed_by="agent")
    return workspace, approval_by_id(workspace, approval["approval_id"]), receipt, subject


def _approved_attempt(workspace, approval, receipt, subject):
    return ActionAttempt(
        workspace=workspace,
        attempt_id="attempt_external",
        task_id=subject["task_id"],
        action_id=subject["task_id"].removesuffix("_execute"),
        invocation_hash=subject["invocation_hash"],
        project=subject["project"],
        version_id=subject["version_id"],
        approval_id=approval["approval_id"],
        approval_subject_hash=approval["subject_hash"],
        approval_consumption_receipt=str(Path("audit") / "approval_consumptions" / f"{approval['approval_id']}.json"),
        parent_operation_id=subject["operation_id"],
    )


def _wide_handler_context(tmp_path, *, generated_sql: str):
    workspace, approval, receipt, subject = _consumed_sql_approval(
        tmp_path, "build_wide_sql_execute"
    )
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    (project / "project.yml").write_text(
        yaml.safe_dump(
            {"project": {"name": "demo"}, "data": {"source_table": "mart.base"}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (project / "configs" / "feature_select.yaml").write_text(
        yaml.safe_dump(
            {
                "feature_select": {
                    "wide_table": {
                        "base_table": "mart.base",
                        "output_table": "mart.wide",
                        "join_keys": ["uid"],
                        "base_columns": ["uid"],
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    remain = project / "remain.json"
    remain.write_text('{"mart.features": ["f1"]}\n', encoding="utf-8")
    state = create_version_state(
        project,
        version_id="demo_v1",
        workflow="feature_selection",
        stages=["build_wide_sql"],
    )
    save_version_state(workspace, state)

    def generator(**kwargs):
        sql_path = kwargs["sql_output_path"]
        feature_map = kwargs["feature_map_path"]
        summary = kwargs["summary_path"]
        sql_path.parent.mkdir(parents=True, exist_ok=True)
        feature_map.parent.mkdir(parents=True, exist_ok=True)
        summary.parent.mkdir(parents=True, exist_ok=True)
        sql_path.write_text(generated_sql, encoding="utf-8")
        feature_map.write_text("output_feature\nf1\n", encoding="utf-8")
        summary.write_text('{"output_table":"mart.wide","features":1}\n', encoding="utf-8")
        return sql_path, feature_map, summary

    feature_actions.WIDE_SQL_GENERATOR = generator
    context = VersionContext(
        project_dir=project,
        version_id="demo_v1",
        workspace=workspace,
        runtime_config_dir=workspace / "configs_runtime",
        manifest_path=workspace / "audit/artifact_manifest.json",
        version_state_path=workspace / "version_state.yml",
    )
    invocation = ActionInvocation(
        "build_wide_sql_execute",
        {"remain_features": str(remain), "sql_approved": True},
        subject["project"],
        subject["version_id"],
    )
    return context, invocation, _approved_attempt(workspace, approval, receipt, subject)


def _feature_execute_handler_context(monkeypatch, tmp_path, *, domain: str, candidate_sql: str):
    operation = f"{domain}_execute"
    workspace, approval, receipt, subject = _consumed_sql_approval(tmp_path, operation)
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    (project / "project.yml").write_text(
        yaml.safe_dump({"project": {"name": "demo"}, "data": {}}, sort_keys=False),
        encoding="utf-8",
    )
    (project / "configs" / "feature_select.yaml").write_text(
        "feature_select: {}\n", encoding="utf-8"
    )
    (project / "configs" / "refine_features.yaml").write_text(
        yaml.safe_dump({"feature_refine": {"output_dir": str(workspace / "feature_selection")}}, sort_keys=False),
        encoding="utf-8",
    )
    state = create_version_state(
        project, version_id="demo_v1", workflow="feature_selection", stages=[domain]
    )
    save_version_state(workspace, state)

    def service(**_kwargs):
        execute_dp_sql(
            project_dir=project,
            sql=candidate_sql,
            operation_id=operation,
            description="handler boundary test",
            metadata_path=workspace / "feature_selection/execution.json",
            sql_approved=True,
            audit_workspace=workspace,
        )
        return 0

    if domain == "feature_prescreen":
        import risk_model_workbench.batch_feature_select as batch

        monkeypatch.setattr(batch, "run_prescreen_service", service)
    else:
        import risk_model_workbench.feature_refine as refine

        monkeypatch.setattr(refine, "run_refine_service", service)
    context = VersionContext(
        project_dir=project,
        version_id="demo_v1",
        workspace=workspace,
        runtime_config_dir=workspace / "configs_runtime",
        manifest_path=workspace / "audit/artifact_manifest.json",
        version_state_path=workspace / "version_state.yml",
    )
    invocation = ActionInvocation(operation, {}, subject["project"], subject["version_id"])
    return context, invocation, _approved_attempt(workspace, approval, receipt, subject)


def _run_feature_execute_handler(domain, invocation, context, attempt):
    if domain == "feature_prescreen":
        return feature_actions.run_feature_prescreen(invocation, context, attempt.attempt_id)
    return feature_actions.run_feature_refine(invocation, context, attempt.attempt_id)


def _install_fake_tmlpatch(monkeypatch, client_type):
    package = types.ModuleType("tmlpatch")
    database = types.ModuleType("tmlpatch.database")
    database.TMLSQLClient = client_type
    package.database = database
    monkeypatch.setitem(sys.modules, "tmlpatch", package)
    monkeypatch.setitem(sys.modules, "tmlpatch.database", database)
