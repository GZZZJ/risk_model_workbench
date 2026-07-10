import yaml

from risk_model_workbench.agent import executor
from risk_model_workbench.agent.executor import run_agent
from risk_model_workbench.agent.recovery import begin_attempt, diagnose_recovery, mark_attempt_dispatched, mark_attempt_result_committed
from risk_model_workbench.agent.state import load_agent_state
from risk_model_workbench.cli import main
from risk_model_workbench.harness.runtime import stage_action_done
from risk_model_workbench.registry import register_artifact


def test_four_crash_points_have_fail_closed_recovery_decisions(tmp_path):
    begin_attempt(tmp_path, attempt_id="intent_only", transition_id="t1", task_id="one", action_id="sample_check", invocation_hash="h1", execution_semantics="read_only")
    begin_attempt(tmp_path, attempt_id="command_unknown", transition_id="t2", task_id="two", action_id="feature_refine", invocation_hash="h2", execution_semantics="external_unknown")
    mark_attempt_dispatched(tmp_path, "command_unknown")
    begin_attempt(tmp_path, attempt_id="manifest_only", transition_id="t3", task_id="three", action_id="report", invocation_hash="h3", execution_semantics="idempotent_write")
    mark_attempt_dispatched(tmp_path, "manifest_only")
    begin_attempt(tmp_path, attempt_id="result_only", transition_id="t4", task_id="four", action_id="evaluate", invocation_hash="h4", execution_semantics="idempotent_write")
    mark_attempt_dispatched(tmp_path, "result_only")
    mark_attempt_result_committed(tmp_path, "result_only", transaction_id="txn_4")

    actions = {item["attempt_id"]: item["recovery_action"] for item in diagnose_recovery(tmp_path)["attempts"]}
    assert actions == {
        "command_unknown": "reconciliation_required",
        "intent_only": "requeue",
        "manifest_only": "requeue",
        "result_only": "apply_receipt",
    }


class SimulatedProcessCrash(BaseException):
    pass


def test_healthy_action_result_is_paired_without_transaction_divergence(tmp_path):
    project, workspace, version_id = _agent_project(tmp_path)

    run_agent(project, version_id, runner=lambda _argv: stage_action_done(workspace, "sample_check") or 0)

    assert load_agent_state(workspace)["tasks"][0]["status"] == "done"
    assert diagnose_recovery(workspace)["transaction_divergence"]["detected"] is False


def test_crash_after_intent_before_command_requeues_without_phantom_execution(tmp_path, monkeypatch):
    project, workspace, version_id = _agent_project(tmp_path)
    calls = []
    original = executor.mark_attempt_dispatched

    def crash_before_dispatch(_workspace, _attempt_id):
        raise SimulatedProcessCrash()

    monkeypatch.setattr(executor, "mark_attempt_dispatched", crash_before_dispatch)
    try:
        run_agent(project, version_id, runner=lambda argv: calls.append(argv) or 0)
    except SimulatedProcessCrash:
        pass
    assert calls == []
    assert next(iter(diagnose_recovery(workspace)["attempts"]))["recovery_action"] == "requeue"

    monkeypatch.setattr(executor, "mark_attempt_dispatched", original)
    run_agent(project, version_id, runner=lambda argv: calls.append(argv) or stage_action_done(workspace, "sample_check") or 0)
    assert len(calls) == 1
    assert load_agent_state(workspace)["tasks"][0]["status"] == "done"


def test_crash_after_safe_command_before_result_is_retried_once(tmp_path):
    project, workspace, version_id = _agent_project(tmp_path)
    calls = 0

    def crashed_runner(_argv):
        nonlocal calls
        calls += 1
        raise SimulatedProcessCrash()

    try:
        run_agent(project, version_id, runner=crashed_runner)
    except SimulatedProcessCrash:
        pass
    assert diagnose_recovery(workspace)["attempts"][0]["recovery_action"] == "requeue"

    def successful_runner(_argv):
        nonlocal calls
        calls += 1
        stage_action_done(workspace, "sample_check")
        return 0

    run_agent(project, version_id, runner=successful_runner)
    assert calls == 2


def test_crash_after_manifest_before_version_state_is_detected_and_safe_action_requeues(tmp_path):
    project, workspace, version_id = _agent_project(tmp_path)

    def crashed_runner(_argv):
        artifact = workspace / "sample_check" / "partial.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("{}\n", encoding="utf-8")
        register_artifact(workspace, artifact, stage="sample_check", transaction_id="txn_manifest_only")
        raise SimulatedProcessCrash()

    try:
        run_agent(project, version_id, runner=crashed_runner)
    except SimulatedProcessCrash:
        pass
    diagnosis = diagnose_recovery(workspace)
    assert diagnosis["transaction_divergence"]["detected"] is True
    assert diagnosis["attempts"][0]["recovery_action"] == "requeue"

    run_agent(project, version_id, runner=lambda _argv: stage_action_done(workspace, "sample_check") or 0)
    assert load_agent_state(workspace)["tasks"][0]["status"] == "done"
    assert diagnose_recovery(workspace)["transaction_divergence"]["detected"] is False


def test_crash_after_result_commit_before_agent_transition_consumes_receipt_without_rerun(tmp_path):
    project, workspace, version_id = _agent_project(tmp_path)
    calls = 0

    def crashed_runner(_argv):
        nonlocal calls
        calls += 1
        stage_action_done(workspace, "sample_check")
        raise SimulatedProcessCrash()

    try:
        run_agent(project, version_id, runner=crashed_runner)
    except SimulatedProcessCrash:
        pass
    assert list((workspace / "audit" / "action_results").glob("*.json"))

    run_agent(project, version_id, runner=lambda _argv: (_ for _ in ()).throw(AssertionError("must not rerun")))
    assert calls == 1
    assert load_agent_state(workspace)["tasks"][0]["status"] == "done"
    assert diagnose_recovery(workspace)["transaction_divergence"]["detected"] is False


def _agent_project(tmp_path):
    project = tmp_path / "project"
    for directory in ["configs", "queries", "reports", "versions"]:
        (project / directory).mkdir(parents=True, exist_ok=True)
    (project / "project.yml").write_text(
        yaml.safe_dump(
            {
                "project": {"name": "demo", "display_name": "Demo", "project_key": "demo_model"},
                "data": {
                    "source_table": "demo.sample",
                    "id_columns": ["uid"],
                    "target_column": "label",
                    "time_column": "event_time",
                    "period_column": "ds",
                },
                "segments": [{"name": "all", "display_name": "All", "filter": None}],
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    request = project / "request.md"
    request.write_text(
        "---\n"
        + yaml.safe_dump(
            {
                "request_id": "crash-recovery-test",
                "project": "demo",
                "workflow": "sample_audit",
                "target_column": "label",
                "id_columns": ["uid"],
                "split_column": "ds",
                "sample_checks": ["sample_check_001"],
                "experiments": [{"name": "baseline"}],
                "evaluation": {"metrics": ["auc"], "champions": []},
                "reports": {"outputs": ["model_report.md"]},
            },
            allow_unicode=True,
            sort_keys=False,
        )
        + "---\n",
        encoding="utf-8",
    )
    version_id = "demo_model_v1_20260711"
    assert main(["agent", "start", "--project", str(project), "--request", str(request), "--version-id", version_id, "--workflow", "sample_audit"]) == 0
    return project, project / "versions" / version_id, version_id
