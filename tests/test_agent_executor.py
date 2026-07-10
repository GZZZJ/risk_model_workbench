import json
from pathlib import Path

import yaml

from risk_model_workbench.agent.executor import run_agent
from risk_model_workbench.agent.plan import save_agent_plan
from risk_model_workbench.agent.state import init_agent_state, load_agent_state
from risk_model_workbench.cli import main
from risk_model_workbench.harness.runtime import stage_action_failed


def test_agent_executor_runs_safe_task_and_marks_scaffold_gap(tmp_path):
    project = _make_project(tmp_path)
    version_id = "demo_model_v1_20260706"
    assert main(["version", "init", "--project", str(project), "--workflow", "sample_audit", "--version-id", version_id]) == 0
    workspace = project / "versions" / version_id
    plan = _agent_plan(project, version_id, [_task("sample_check_001", ["sample", "check", "--project", str(project), "--version-id", version_id])])
    save_agent_plan(workspace, plan)
    init_agent_state(workspace, project=str(project), version_id=version_id, agent_plan=plan)

    result = run_agent(project, version_id, runner=main)
    state = load_agent_state(workspace)

    assert result["status"] == "done_with_gaps"
    assert state["tasks"][0]["status"] == "scaffold"
    assert (workspace / "audit" / "agent_trace.jsonl").exists()


def test_agent_executor_pauses_for_approval_and_can_resume_after_approval(tmp_path):
    project = _make_project(tmp_path)
    version_id = "demo_model_v1_20260706"
    assert main(["version", "init", "--project", str(project), "--workflow", "sample_audit", "--version-id", version_id]) == 0
    workspace = project / "versions" / version_id
    task = _task(
        "dp_pull",
        ["feature", "prescreen", "--project", str(project), "--version-id", version_id, "--sql-approved"],
        permission="dp_sql_pull",
        requires_approval=True,
    )
    plan = _agent_plan(project, version_id, [task])
    save_agent_plan(workspace, plan)
    init_agent_state(workspace, project=str(project), version_id=version_id, agent_plan=plan)

    result = run_agent(project, version_id, runner=main)
    state = load_agent_state(workspace)
    approval_id = state["blocker"]["approval_id"]

    assert result["status"] == "waiting_for_approval"
    assert (workspace / "audit" / "approvals.yml").exists()

    assert main(["agent", "approve", "--project", str(project), "--version-id", version_id, "--approval-id", approval_id, "--approved-by", "pm", "--note", "sql reviewed"]) == 0
    approved = yaml.safe_load((workspace / "audit" / "approvals.yml").read_text(encoding="utf-8"))
    assert approved["approvals"][0]["status"] == "approved"


def test_agent_executor_pauses_for_advisor_required_failure(tmp_path):
    project = _make_project(tmp_path)
    version_id = "demo_model_v1_20260706"
    assert main(["version", "init", "--project", str(project), "--workflow", "train_baseline", "--version-id", version_id]) == 0
    workspace = project / "versions" / version_id
    task = _task("train_main", ["train", "--project", str(project), "--version-id", version_id, "--experiment", "main_lgbm"])
    plan = _agent_plan(project, version_id, [task])
    save_agent_plan(workspace, plan)
    init_agent_state(workspace, project=str(project), version_id=version_id, agent_plan=plan)

    def advisor_runner(_argv):
        stage_action_failed(workspace, "train_baseline", "host agent plan required", failure_code="advisor_required")
        return 1

    result = run_agent(project, version_id, runner=advisor_runner)
    state = load_agent_state(workspace)
    requests = sorted((workspace / "audit" / "advisor_requests").glob("*.json"))

    assert result["status"] == "waiting_for_advisor"
    assert state["blocker"]["reason"] == "advisor_required"
    assert requests
    payload = json.loads(requests[0].read_text(encoding="utf-8"))
    assert payload["task_id"] == "train_main"


def _agent_plan(project: Path, version_id: str, tasks: list[dict]) -> dict:
    return {
        "version": 1,
        "plan_id": "agent_plan",
        "version_id": version_id,
        "project": str(project),
        "workflow": "sample_audit",
        "tasks": tasks,
    }


def _task(
    task_id: str,
    args: list[str],
    *,
    permission: str = "writes_run",
    requires_approval: bool = False,
) -> dict:
    return {
        "task_id": task_id,
        "type": "sample_check",
        "status": "pending",
        "workspace": f"tasks/{task_id}",
        "depends_on": [],
        "command": {"executable": "rmw", "args": args},
        "outputs": ["sample_check/sample_summary.json"],
        "scenario_profile": "generic",
        "step_ids": [],
        "step_params": {},
        "action_id": "sample_check" if task_id.startswith("sample") else "feature_prescreen" if task_id == "dp_pull" else "train_baseline",
        "tool_name": "sample_check" if task_id.startswith("sample") else "feature_prescreen_pull" if task_id == "dp_pull" else "train_baseline",
        "permission": permission,
        "requires_approval": requires_approval,
        "expected_outputs": ["sample_check/sample_summary.json"],
    }


def _make_project(tmp_path: Path) -> Path:
    project = tmp_path / "demo_project"
    for directory in ["configs", "queries", "reports", "versions"]:
        (project / directory).mkdir(parents=True, exist_ok=True)
    (project / "project.yml").write_text(
        yaml.safe_dump(
            {
                "project": {"name": "demo_project", "display_name": "Demo Project", "project_key": "demo_model"},
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
    return project
