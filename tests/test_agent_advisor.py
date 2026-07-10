import json
from pathlib import Path

import yaml

from risk_model_workbench.agent.advisor import (
    accept_advisor_response,
    create_advisor_request,
    list_advisor_requests,
    load_advisor_request,
    validate_advisor_response,
)
from risk_model_workbench.agent.executor import run_agent
from risk_model_workbench.agent.plan import save_agent_plan
from risk_model_workbench.agent.state import init_agent_state, load_agent_state
from risk_model_workbench.cli import main
from risk_model_workbench.harness.runtime import stage_action_failed


def test_create_and_accept_standard_advisor_response(tmp_path):
    project = _make_project(tmp_path)
    workspace = _init_version(project)
    task = _task("train_main")

    request = create_advisor_request(
        workspace,
        project_dir=project,
        version_id="demo_model_v1_20260709",
        task=task,
        reason="advisor_required",
        message="host agent plan required",
    )

    assert request["type"] == "tuning_plan_required"
    assert request["status"] == "pending"
    assert request["expected_response"]["type"] == "tuning_plan"
    assert request["command_hash"]
    assert Path(workspace / request["path"]).exists()

    response_path = workspace / "advisor_001.response.json"
    response_path.write_text(
        json.dumps(
            {
                "version": 1,
                "request_id": request["request_id"],
                "type": "tuning_plan",
                "status": "answered",
                "decision": "continue",
                "summary": "Use bounded LightGBM candidates.",
                "output_files": [],
                "risk_notes": ["Do not optimize directly on OOT."],
                "requires_user_confirmation": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    accepted = accept_advisor_response(workspace, response_path)
    loaded = load_advisor_request(workspace, request["request_id"])

    assert accepted["accepted"] is True
    assert loaded["status"] == "answered"
    assert loaded["accepted_response"].endswith(f"{request['request_id']}.response.json")
    assert (workspace / loaded["accepted_response"]).exists()


def test_advisor_response_validation_rejects_mismatch(tmp_path):
    project = _make_project(tmp_path)
    workspace = _init_version(project)
    request = create_advisor_request(
        workspace,
        project_dir=project,
        version_id="demo_model_v1_20260709",
        task=_task("train_main"),
        reason="advisor_required",
        message="host agent plan required",
    )

    errors = validate_advisor_response(
        workspace,
        {
            "version": 1,
            "request_id": request["request_id"],
            "type": "failure_diagnosis",
            "status": "answered",
            "decision": "continue",
            "summary": "wrong type",
        },
    )

    assert any("response type mismatch" in error for error in errors)


def test_agent_advisor_cli_list_show_accept(tmp_path, capsys):
    project = _make_project(tmp_path)
    workspace = _init_version(project)
    request = create_advisor_request(
        workspace,
        project_dir=project,
        version_id="demo_model_v1_20260709",
        task=_task("train_main"),
        reason="advisor_required",
        message="host agent plan required",
    )
    response_path = workspace / "response.json"
    response_path.write_text(
        json.dumps(
            {
                "version": 1,
                "request_id": request["request_id"],
                "type": "tuning_plan",
                "status": "answered",
                "decision": "continue",
                "summary": "Accepted.",
                "output_files": [],
                "risk_notes": [],
                "requires_user_confirmation": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    capsys.readouterr()
    assert main(["agent", "advisor", "list", "--project", str(project), "--version-id", "demo_model_v1_20260709", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["request_id"] == request["request_id"]

    assert (
        main(
            [
                "agent",
                "advisor",
                "show",
                "--project",
                str(project),
                "--version-id",
                "demo_model_v1_20260709",
                "--request-id",
                request["request_id"],
                "--json",
            ]
        )
        == 0
    )
    shown = json.loads(capsys.readouterr().out)
    assert shown["expected_response"]["type"] == "tuning_plan"

    assert (
        main(
            [
                "agent",
                "advisor",
                "accept",
                "--project",
                str(project),
                "--version-id",
                "demo_model_v1_20260709",
                "--response",
                str(response_path),
            ]
        )
        == 0
    )
    assert "advisor_response: accepted" in capsys.readouterr().out


def test_resume_requires_accepted_advisor_response(tmp_path, capsys):
    project = _make_project(tmp_path)
    version_id = "demo_model_v1_20260709"
    workspace = _init_version(project, version_id=version_id)
    task = _task("train_main")
    task["command"]["args"] = ["train", "--project", str(project), "--version-id", version_id, "--experiment", "main_lgbm"]
    plan = {
        "version": 1,
        "plan_id": "agent_plan",
        "version_id": version_id,
        "project": str(project),
        "workflow": "train_baseline",
        "tasks": [task],
    }
    save_agent_plan(workspace, plan)
    init_agent_state(workspace, project=str(project), version_id=version_id, agent_plan=plan)

    def advisor_runner(_argv):
        stage_action_failed(workspace, "train_baseline", "host agent plan required", failure_code="advisor_required")
        return 1

    state = run_agent(project, version_id, runner=advisor_runner)
    request_id = state["blocker"]["advisor_request_id"]

    assert main(["agent", "resume", "--project", str(project), "--version-id", version_id]) == 1
    assert "advisor response pending" in capsys.readouterr().out

    response_path = workspace / "response.json"
    response_path.write_text(
        json.dumps(
            {
                "version": 1,
                "request_id": request_id,
                "type": "tuning_plan",
                "status": "answered",
                "decision": "continue",
                "summary": "Ready.",
                "output_files": [],
                "risk_notes": [],
                "requires_user_confirmation": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    assert main(["agent", "advisor", "accept", "--project", str(project), "--version-id", version_id, "--response", str(response_path)]) == 0
    assert main(["agent", "resume", "--project", str(project), "--version-id", version_id]) in {0, 1}
    assert "advisor response pending" not in capsys.readouterr().out


def test_list_advisor_requests_is_stable_when_empty(tmp_path):
    workspace = tmp_path / "version"
    (workspace / "audit").mkdir(parents=True)

    assert list_advisor_requests(workspace) == []


def _task(task_id: str) -> dict:
    return {
        "task_id": task_id,
        "type": "train",
        "status": "pending",
        "workspace": f"tasks/{task_id}",
        "depends_on": [],
        "command": {"executable": "rmw", "args": ["train", "--project", "p", "--version-id", "v", "--experiment", "main_lgbm"]},
        "outputs": ["modeling/main_lgbm/tuning_summary.json"],
        "action_id": "train_baseline",
        "tool_name": "train_baseline",
        "permission": "writes_run",
        "requires_approval": False,
    }


def _init_version(project: Path, *, version_id: str = "demo_model_v1_20260709") -> Path:
    assert main(["version", "init", "--project", str(project), "--workflow", "train_baseline", "--version-id", version_id]) == 0
    return project / "versions" / version_id


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
