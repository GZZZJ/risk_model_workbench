"""Attempt-scoped semantic ActionResult contract and Executor precedence."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from risk_model_workbench.agent.executor import run_agent
from risk_model_workbench.agent.state import load_agent_state
from risk_model_workbench.agent.workspace_store import WorkspaceStore
from risk_model_workbench.cli import main
from risk_model_workbench.harness.errors import (
    DuplicateActionResultError,
    InvalidActionResultError,
    MissingActionResultError,
    SQL_APPROVAL_REQUIRED,
    WorkspaceLockedError,
)
from risk_model_workbench.harness.runtime import (
    ActionResult,
    action_result_path,
    load_action_result,
    stage_action_done,
    stage_action_failed,
    write_action_result,
)


def test_action_result_receipt_is_exclusive_and_strongly_bound(tmp_path):
    workspace = tmp_path / "version"
    result = _result()

    path = write_action_result(workspace, result)

    assert path == action_result_path(workspace, result.attempt_id)
    assert load_action_result(
        workspace,
        result.attempt_id,
        task_id=result.task_id,
        action_id=result.action_id,
        invocation_hash=result.invocation_hash,
        project=result.project,
        version_id=result.version_id,
    ) == result
    with pytest.raises(DuplicateActionResultError):
        write_action_result(workspace, result)
    with pytest.raises(InvalidActionResultError, match="subject mismatch"):
        load_action_result(workspace, result.attempt_id, task_id="another_task")


def test_old_receipt_cannot_satisfy_a_new_attempt(tmp_path):
    workspace = tmp_path / "version"
    write_action_result(workspace, _result(attempt_id="attempt_old"))

    with pytest.raises(MissingActionResultError):
        load_action_result(workspace, "attempt_new")


def test_action_result_schema_and_runtime_fields_remain_conformant():
    schema = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "schemas" / "action_result.schema.yml").read_text(encoding="utf-8")
    )
    runtime_fields = set(ActionResult.__dataclass_fields__)
    assert set(schema["required_fields"]) == runtime_fields
    assert set(schema["field_types"]) == runtime_fields


@pytest.mark.parametrize(
    "overrides",
    [
        {"attempt_id": "../escape"},
        {"created_at": ""},
        {"retryable": "yes"},
        {"artifacts": ["not-an-object"]},
        {"status": "failed", "scaffold": True},
        {"status": "done", "scaffold": True},
        {"status": "review_ready", "scaffold": True},
    ],
)
def test_malformed_result_is_rejected_before_receipt_write(tmp_path, overrides):
    workspace = tmp_path / "version"
    with pytest.raises(InvalidActionResultError):
        write_action_result(workspace, _result(**overrides))
    assert not (workspace / "audit" / "action_results").exists()


def test_exit_zero_plus_semantic_approval_pauses(tmp_path):
    project, workspace, version_id = _agent_project(tmp_path)

    def runner(_argv):
        stage_action_failed(
            workspace,
            "sample_check",
            "SQL review required",
            failure_code=SQL_APPROVAL_REQUIRED,
        )
        return 0

    result = run_agent(project, version_id, runner=runner)
    receipt = _only_receipt(workspace)

    assert result["status"] == "waiting_for_approval"
    assert receipt["next_required_action"] == "approval"
    assert receipt["status"] == "failed"


def test_nonzero_exit_plus_semantic_scaffold_is_not_unknown(tmp_path):
    project, workspace, version_id = _agent_project(tmp_path)

    def runner(_argv):
        stage_action_done(workspace, "sample_check", scaffold=True, message="local input absent")
        return 17

    result = run_agent(project, version_id, runner=runner)
    state = load_agent_state(workspace)

    assert result["status"] == "done_with_gaps"
    assert state["tasks"][0]["status"] == "scaffold"
    assert _only_receipt(workspace)["status"] == "scaffold"


def test_old_stage_last_result_cannot_replace_missing_attempt_receipt(tmp_path):
    project, workspace, version_id = _agent_project(tmp_path)
    stage_action_done(workspace, "sample_check", scaffold=True, message="old result")

    result = run_agent(project, version_id, runner=lambda _argv: 0)

    assert result["status"] == "failed"
    assert load_agent_state(workspace)["tasks"][0]["message"] == "missing_action_result"
    assert list((workspace / "audit" / "action_results").glob("*.json")) == []


def test_wrong_subject_receipt_fails_even_when_process_succeeds(tmp_path):
    project, workspace, version_id = _agent_project(tmp_path)

    def runner(_argv):
        state = load_agent_state(workspace)
        attempt_id = state["tasks"][0]["attempt_id"]
        write_action_result(
            workspace,
            _result(
                attempt_id=attempt_id,
                task_id="wrong_task",
                project=str(project),
                version_id=version_id,
                invocation_hash=state["tasks"][0]["invocation_hash"],
            ),
        )
        return 0

    result = run_agent(project, version_id, runner=runner)

    assert result["status"] == "failed"
    assert load_agent_state(workspace)["tasks"][0]["message"] == "invalid_action_result"


def test_valid_receipt_is_consumed_even_when_runner_raises_after_write(tmp_path):
    project, workspace, version_id = _agent_project(tmp_path)

    def runner(_argv):
        stage_action_done(workspace, "sample_check", scaffold=True, message="semantic result exists")
        raise RuntimeError("post-result diagnostic failure")

    result = run_agent(project, version_id, runner=runner)

    assert result["status"] == "done_with_gaps"
    assert load_agent_state(workspace)["tasks"][0]["status"] == "scaffold"


def test_runner_exception_without_receipt_becomes_missing_result(tmp_path):
    project, workspace, version_id = _agent_project(tmp_path)

    def runner(_argv):
        raise RuntimeError("no semantic result")

    result = run_agent(project, version_id, runner=runner)

    assert result["status"] == "failed"
    assert load_agent_state(workspace)["tasks"][0]["message"] == "missing_action_result"


def test_malformed_receipt_becomes_invalid_result_instead_of_leaving_task_running(tmp_path):
    project, workspace, version_id = _agent_project(tmp_path)

    def runner(_argv):
        state = load_agent_state(workspace)
        attempt_id = state["tasks"][0]["attempt_id"]
        path = action_result_path(workspace, attempt_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"truncated":', encoding="utf-8")
        return 0

    result = run_agent(project, version_id, runner=runner)

    assert result["status"] == "failed"
    assert load_agent_state(workspace)["tasks"][0]["message"] == "invalid_action_result"


def test_load_action_result_rejects_path_escape(tmp_path):
    with pytest.raises(InvalidActionResultError, match="attempt_id"):
        load_action_result(tmp_path, "../outside")


def test_single_runner_lock_prevents_duplicate_execution(tmp_path):
    project, workspace, version_id = _agent_project(tmp_path)
    calls: list[list[str]] = []

    with WorkspaceStore(workspace).runner_lock():
        with pytest.raises(WorkspaceLockedError):
            run_agent(project, version_id, runner=lambda argv: calls.append(argv) or 0)

    assert calls == []


def _result(**overrides) -> ActionResult:
    payload = {
        "schema_version": 1,
        "attempt_id": "attempt_001",
        "task_id": "sample_check_001",
        "action_id": "sample_check",
        "invocation_hash": "a" * 64,
        "project": "projects/example",
        "version_id": "example_v1",
        "status": "done",
        "failure_code": "",
        "next_required_action": "none",
        "retryable": False,
        "scaffold": False,
        "artifacts": [],
        "message": "",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    payload.update(overrides)
    return ActionResult(**payload)


def _only_receipt(workspace: Path) -> dict:
    paths = list((workspace / "audit" / "action_results").glob("*.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text(encoding="utf-8"))


def _agent_project(tmp_path: Path) -> tuple[Path, Path, str]:
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
                "request_id": "attempt-result-test",
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
    version_id = "demo_model_v1_20260710"
    assert main(["agent", "start", "--project", str(project), "--request", str(request), "--version-id", version_id, "--workflow", "sample_audit"]) == 0
    return project, project / "versions" / version_id, version_id
