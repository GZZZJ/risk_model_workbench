import json
from pathlib import Path

import yaml

from risk_model_workbench.agent.state import init_agent_state, load_agent_state, mark_task_done, mark_task_running, pause_agent
from risk_model_workbench.agent.trace import append_trace, load_recent_trace


def test_agent_state_lifecycle_and_trace_are_auditable(tmp_path):
    workspace = tmp_path / "version"
    (workspace / "audit").mkdir(parents=True)
    plan = {
        "plan_id": "agent-request_agent_plan",
        "tasks": [
            {
                "task_id": "sample_check_001",
                "action_id": "sample_check",
                "tool_name": "sample_check",
                "permission": "writes_run",
            }
        ],
    }

    state = init_agent_state(workspace, project=str(tmp_path), version_id="demo_v1", agent_plan=plan)
    assert state["status"] == "draft"
    assert state["tasks"][0]["status"] == "pending"

    mark_task_running(workspace, "sample_check_001")
    assert load_agent_state(workspace)["current_task"] == "sample_check_001"

    mark_task_done(workspace, "sample_check_001", scaffold=True, message="local data missing")
    done = load_agent_state(workspace)
    assert done["status"] == "done_with_gaps"
    assert done["tasks"][0]["status"] == "scaffold"

    paused = pause_agent(workspace, status="waiting_for_advisor", reason="advisor_required", task_id="train")
    assert paused["blocker"]["reason"] == "advisor_required"

    append_trace(
        workspace,
        "decision",
        {
            "summary": "Paused for advisor.",
            "command": ["rmw", "train", "--version-id", "demo_v1"],
            "hidden_reasoning": "must not be persisted",
        },
    )
    trace = load_recent_trace(workspace)
    raw_trace = (workspace / "audit" / "agent_trace.jsonl").read_text(encoding="utf-8")

    assert trace[-1]["event"] == "decision"
    assert "hidden_reasoning" not in trace[-1]
    assert "hidden_reasoning" not in raw_trace
    assert json.loads(raw_trace.splitlines()[-1])["summary"] == "Paused for advisor."

    serialized = yaml.safe_load((workspace / "audit" / "agent_state.yml").read_text(encoding="utf-8"))
    assert serialized["status"] == "waiting_for_advisor"
