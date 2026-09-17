"""LangGraph standalone Agent runtime tests."""

from __future__ import annotations

from pathlib import Path

import yaml

from risk_model_workbench.agent.advisor import create_advisor_request, load_advisor_request
from risk_model_workbench.agent.advisor_reducer import consume_advisor_response
from risk_model_workbench.agent.embedded_runtime import run_embedded_agent
from risk_model_workbench.agent.model_gateway import FakeModelGateway
from risk_model_workbench.agent.state import (
    init_agent_state,
    load_agent_state,
    mark_task_running,
    pause_agent,
    save_agent_state,
)
from risk_model_workbench.cli import main


def test_langgraph_runtime_answers_advisor_and_reaches_terminal_state(tmp_path):
    project, workspace, version_id, task = _init_agent_workspace(tmp_path)
    mark_task_running(workspace, task["task_id"])
    state = load_agent_state(workspace)
    state["tasks"][0]["attempt_id"] = "attempt_1"
    state["tasks"][0]["invocation_hash"] = "invocation_1"
    save_agent_state(workspace, state)
    request = create_advisor_request(
        workspace,
        project_dir=project,
        version_id=version_id,
        task={**task, "attempt_id": "attempt_1", "invocation_hash": "invocation_1"},
        reason="data_missing",
        message="required sample evidence is missing",
    )
    pause_agent(
        workspace,
        status="waiting_for_advisor",
        reason="data_missing",
        task_id=task["task_id"],
        advisor_request=request["path"],
        advisor_request_id=request["request_id"],
    )
    gateway = FakeModelGateway(
        [
            {
                "type": "data_gap_decision",
                "status": "answered",
                "decision": "stop",
                "summary": "Stop safely until real sample evidence is provided.",
            }
        ]
    )
    harness_calls = []

    def harness_runner(_project: Path, _version_id: str):
        harness_calls.append("run")
        current = load_agent_state(workspace)
        if current["status"] == "waiting_for_advisor":
            current_request = load_advisor_request(workspace, request["request_id"])
            if current_request["status"] in {"answered", "rejected"}:
                return consume_advisor_response(workspace, request["request_id"], current).state
        return current

    result = run_embedded_agent(
        project,
        version_id,
        gateway=gateway,
        harness_runner=harness_runner,
    )

    assert result["status"] == "stopped"
    assert result["interrupted"] is False
    assert result["model_calls"] == 1
    assert len(harness_calls) == 2
    assert load_advisor_request(workspace, request["request_id"])["status"] == "consumed"
    assert (workspace / "audit" / "agent_graph.sqlite").exists()


def test_langgraph_runtime_interrupts_at_human_gate_without_approving_it(tmp_path):
    project, workspace, version_id, _task = _init_agent_workspace(tmp_path)
    state = load_agent_state(workspace)
    state["status"] = "waiting_for_approval"
    state["blocker"] = {
        "reason": "approval_required",
        "blocker_type": "approval",
        "blocker_id": "approval_1",
        "approval_id": "approval_1",
        "next_safe_action": {
            "action": "consume_bound_approval",
            "required_evidence": "approved one-time approval record",
        },
    }
    save_agent_state(workspace, state)

    result = run_embedded_agent(
        project,
        version_id,
        gateway=FakeModelGateway([]),
        harness_runner=lambda _project, _version_id: load_agent_state(workspace),
    )

    assert result["status"] == "waiting_for_approval"
    assert result["interrupted"] is True
    assert load_agent_state(workspace)["status"] == "waiting_for_approval"


def _init_agent_workspace(tmp_path: Path):
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
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    version_id = "demo_model_v1_20260823"
    assert main(["version", "init", "--project", str(project), "--workflow", "train_baseline", "--version-id", version_id]) == 0
    workspace = project / "versions" / version_id
    task = {
        "task_id": "sample_check_1",
        "depends_on": [],
        "command": {"executable": "rmw", "args": ["sample", "check", "--project", str(project), "--version-id", version_id]},
        "action_id": "sample_check",
        "tool_name": "sample_check",
        "permission": "writes_run",
        "requires_approval": False,
    }
    plan = {
        "version": 1,
        "plan_id": "embedded_test_plan",
        "project": str(project),
        "version_id": version_id,
        "tasks": [task],
    }
    init_agent_state(workspace, project=str(project), version_id=version_id, agent_plan=plan)
    return project, workspace, version_id, task
