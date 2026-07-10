from pathlib import Path

import yaml

from risk_model_workbench.agent.plan import bind_agent_plan
from risk_model_workbench.cli import main
from risk_model_workbench.planning import create_execution_plan


def test_execution_plan_binds_to_version_agent_plan(tmp_path):
    project = _make_project(tmp_path)
    request_doc = _request_doc(project)
    execution_plan = create_execution_plan(request_doc, project)

    agent_plan = bind_agent_plan(
        execution_plan,
        project_dir=project,
        version_id="demo_model_v1_20260706",
    )

    assert agent_plan["version"] == 1
    assert agent_plan["version_id"] == "demo_model_v1_20260706"
    assert agent_plan["source_plan_id"] == "agent-request_plan"
    assert agent_plan["run_id_placeholder"] == ""
    assert agent_plan["tasks"]

    for task in agent_plan["tasks"]:
        args = task["command"]["args"]
        assert "--run-id" not in args
        assert "<run_id>" not in args
        assert "--version-id" in args
        assert args[args.index("--version-id") + 1] == "demo_model_v1_20260706"
        assert task["action_id"]
        assert task["tool_name"]
        assert task["permission"] in {"read_only", "writes_run", "dp_sql_pull", "external_data"}
        assert isinstance(task["requires_approval"], bool)
        assert task["expected_outputs"]


def test_agent_start_cli_writes_bound_plan_state_and_trace(tmp_path, capsys):
    project = _make_project(tmp_path)
    request_path = _write_request(project)

    assert (
        main(
            [
                "agent",
                "start",
                "--project",
                str(project),
                "--request",
                str(request_path),
                "--version-id",
                "demo_model_v1_20260706",
                "--workflow",
                "sample_audit",
            ]
        )
        == 0
    )

    version_dir = project / "versions" / "demo_model_v1_20260706"
    agent_plan = yaml.safe_load((version_dir / "agent_plan.yml").read_text(encoding="utf-8"))
    agent_state = yaml.safe_load((version_dir / "audit" / "agent_state.yml").read_text(encoding="utf-8"))

    assert agent_plan["version_id"] == "demo_model_v1_20260706"
    assert agent_plan["tasks"][0]["command"]["args"][-2:] == ["--version-id", "demo_model_v1_20260706"]
    assert agent_state["status"] == "draft"
    assert agent_state["version_id"] == "demo_model_v1_20260706"
    assert agent_state["tasks"][0]["status"] == "pending"
    assert (version_dir / "audit" / "agent_trace.jsonl").exists()

    capsys.readouterr()
    assert main(["agent", "tools", "--json"]) == 0
    tools = yaml.safe_load(capsys.readouterr().out)
    assert any(tool["name"] == "sample_check" for tool in tools)


def _request_doc(project: Path) -> dict:
    return {
        "path": str(project / "request.md"),
        "metadata": {
            "request_id": "agent-request",
            "project": "demo_project",
            "workflow": "sample_audit",
            "target_column": "label",
            "id_columns": ["uid"],
            "split_column": "ds",
            "sample_checks": ["sample_check_001"],
            "experiments": [{"name": "baseline_all"}],
            "evaluation": {"metrics": ["auc"], "champions": []},
            "reports": {"outputs": ["model_report.md"]},
        },
        "body": "",
    }


def _write_request(project: Path) -> Path:
    request_path = project / "request.md"
    request_path.write_text(
        "---\n"
        + yaml.safe_dump(_request_doc(project)["metadata"], allow_unicode=True, sort_keys=False)
        + "---\n",
        encoding="utf-8",
    )
    return request_path


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
