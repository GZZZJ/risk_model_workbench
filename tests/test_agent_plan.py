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

    assert agent_plan["version"] == 2
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
        assert task["invocation"]["project"] == str(project)
        assert task["invocation"]["version_id"] == "demo_model_v1_20260706"
        assert len(task["invocation_hash"]) == 64
        assert task["derived_metadata"]["permission"] in {
            "read_only",
            "writes_run",
            "dp_sql_pull",
            "external_data",
        }
        assert isinstance(task["derived_metadata"]["requires_approval"], bool)
        assert task["expected_outputs"]
    assert len(agent_plan["registry_digest"]) == 64
    assert len(agent_plan["plan_hash"]) == 64


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
    assert agent_plan["version"] == 2
    assert agent_plan["tasks"][0]["command"]["args"][-2:] == ["--version-id", "demo_model_v1_20260706"]
    assert agent_state["status"] == "draft"
    assert agent_state["version"] == 2
    assert agent_state["plan_hash"] == agent_plan["plan_hash"]
    assert agent_state["registry_digest"] == agent_plan["registry_digest"]
    assert agent_state["version_id"] == "demo_model_v1_20260706"
    assert agent_state["tasks"][0]["status"] == "pending"
    assert (version_dir / "audit" / "agent_trace.jsonl").exists()

    capsys.readouterr()
    assert main(["agent", "tools", "--json"]) == 0
    tools = yaml.safe_load(capsys.readouterr().out)
    assert any(tool["name"] == "sample_check" for tool in tools)


def test_agent_plan_rebind_cli_previews_then_applies_only_in_safe_state(tmp_path, capsys):
    project = _make_project(tmp_path)
    request_path = _write_request(project)
    version_id = "demo_model_v1_20260706"
    assert main(["agent", "start", "--project", str(project), "--request", str(request_path), "--version-id", version_id, "--workflow", "sample_audit"]) == 0
    workspace = project / "versions" / version_id
    plan_path = workspace / "agent_plan.yml"
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))

    capsys.readouterr()
    assert main(["agent", "plan", "rebind", "--project", str(project), "--version-id", version_id, "--dry-run", "--json"]) == 0
    preview = yaml.safe_load(capsys.readouterr().out)
    assert preview["mode"] == "dry_run"
    assert preview["changed"] is False
    assert yaml.safe_load(plan_path.read_text(encoding="utf-8"))["plan_hash"] == plan["plan_hash"]

    assert main(["agent", "plan", "rebind", "--project", str(project), "--version-id", version_id, "--apply", "--json"]) == 0
    applied = yaml.safe_load(capsys.readouterr().out)
    assert applied["mode"] == "apply"
    assert yaml.safe_load(plan_path.read_text(encoding="utf-8"))["registry_digest"] == plan["registry_digest"]

    tampered = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    tampered["tasks"][0]["invocation"]["project"] = "projects/other"
    plan_path.write_text(yaml.safe_dump(tampered, allow_unicode=True, sort_keys=False), encoding="utf-8")
    before = plan_path.read_text(encoding="utf-8")
    assert main(["agent", "plan", "rebind", "--project", str(project), "--version-id", version_id, "--apply"]) == 1
    assert plan_path.read_text(encoding="utf-8") == before

    plan_path.write_text(yaml.safe_dump(plan, allow_unicode=True, sort_keys=False), encoding="utf-8")

    state_path = workspace / "audit" / "agent_state.yml"
    state = yaml.safe_load(state_path.read_text(encoding="utf-8"))
    state["status"] = "running"
    state_path.write_text(yaml.safe_dump(state, allow_unicode=True, sort_keys=False), encoding="utf-8")
    assert main(["agent", "plan", "rebind", "--project", str(project), "--version-id", version_id, "--dry-run"]) == 1


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
