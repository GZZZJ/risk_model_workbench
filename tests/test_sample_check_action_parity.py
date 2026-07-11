import json
from pathlib import Path

import yaml

from risk_model_workbench.application.action_runner import ActionRunner
from risk_model_workbench.application.context import VersionContext
from risk_model_workbench.application.handlers import production_handler_registry
from risk_model_workbench.cli import main
from risk_model_workbench.harness.invocation import ActionInvocation
from risk_model_workbench.registry import load_artifact_manifest
from risk_model_workbench.state import create_run_state, load_run_state, save_run_state


def _workspace(project: Path, run_id: str) -> VersionContext:
    workspace = project / "runs" / run_id
    (project / "configs").mkdir(parents=True, exist_ok=True)
    (workspace / "audit").mkdir(parents=True, exist_ok=True)
    (workspace / "configs_runtime").mkdir(parents=True, exist_ok=True)
    save_run_state(workspace, create_run_state(project, run_id=run_id, workflow="sample_audit"))
    (workspace / "audit" / "artifact_manifest.json").write_text(
        json.dumps({"version": 2, "artifacts": []}), encoding="utf-8"
    )
    project_config = {
        "project": {"name": "sample-parity"},
        "data": {"raw_path": "data/missing.feather", "target_column": "label"},
    }
    (project / "project.yml").write_text(yaml.safe_dump(project_config), encoding="utf-8")
    (workspace / "configs_runtime" / "project.yml").write_text(
        yaml.safe_dump(project_config), encoding="utf-8"
    )
    return VersionContext(
        project_dir=project.resolve(),
        version_id=run_id,
        workspace=workspace,
        runtime_config_dir=workspace / "configs_runtime",
        manifest_path=workspace / "audit" / "artifact_manifest.json",
        version_state_path=workspace / "run_state.yml",
    )


def _snapshot(context: VersionContext) -> dict:
    state = load_run_state(context.workspace)
    manifest = load_artifact_manifest(context.workspace)
    return {
        "status": state["stages"]["sample_check"]["status"],
        "scaffold": bool(state["stages"]["sample_check"].get("scaffold")),
        "artifacts": sorted(
            row["path"] for row in manifest["artifacts"] if row.get("stage") == "sample_check"
        ),
        "decisions": [
            (row["decision"], row["reason"])
            for row in state.get("decisions", [])
            if row.get("stage") == "sample_check"
        ],
    }


def test_sample_check_cli_and_action_runner_have_scaffold_parity(tmp_path, capsys):
    project = tmp_path / "project"
    cli_context = _workspace(project, "cli_sample")
    direct_context = _workspace(project, "direct_sample")

    assert main(["sample", "check", "--project", str(project), "--run-id", "cli_sample"]) == 0
    cli_output = capsys.readouterr().out
    invocation = ActionInvocation(
        tool_name="sample_check",
        params={},
        project=str(project.resolve()),
        version_id="direct_sample",
    )
    result = ActionRunner(
        handlers=production_handler_registry(), policy_check=lambda *_: True
    ).run(invocation=invocation, context=direct_context, attempt_id="attempt_sample")

    assert result.status == "scaffold"
    assert "local data not available" in cli_output
    assert _snapshot(cli_context) == _snapshot(direct_context)
