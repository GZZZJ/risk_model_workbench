import json
from pathlib import Path

import pandas as pd
import yaml

from risk_model_workbench.application.action_runner import ActionRunner
from risk_model_workbench.application.context import VersionContext
from risk_model_workbench.application.handlers import production_handler_registry
from risk_model_workbench.cli import main
from risk_model_workbench.harness.invocation import ActionInvocation
from risk_model_workbench.registry import load_artifact_manifest
from risk_model_workbench.state import create_run_state, load_run_state, save_run_state


def _context(project: Path, run_id: str) -> VersionContext:
    workspace = project / "runs" / run_id
    (workspace / "audit").mkdir(parents=True)
    (workspace / "configs_runtime").mkdir()
    save_run_state(workspace, create_run_state(project, run_id=run_id, workflow="full_modeling"))
    (workspace / "audit" / "artifact_manifest.json").write_text(json.dumps({"version": 2, "artifacts": []}))
    for name in ["project.yml", "feature_select.yaml", "refine_features.yaml"]:
        source = project / "configs" / name
        if source.exists():
            (workspace / "configs_runtime" / name).write_text(source.read_text())
    return VersionContext(project.resolve(), run_id, workspace, workspace / "configs_runtime", workspace / "audit" / "artifact_manifest.json", workspace / "run_state.yml")


def _snapshot(context: VersionContext, stage: str) -> dict:
    return {
        "status": load_run_state(context.workspace)["stages"][stage]["status"],
        "artifacts": sorted(x["path"] for x in load_artifact_manifest(context.workspace)["artifacts"] if x.get("stage") == stage),
    }


def test_feature_metadata_cli_and_runner_have_local_feather_parity(tmp_path):
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    feather = project / "sample.feather"
    pd.DataFrame({"uid": [1, 2], "label": [0, 1], "x": [0.1, 0.2]}).to_feather(feather)
    (project / "configs" / "project.yml").write_text(yaml.safe_dump({"data": {"raw_path": str(feather)}}))
    (project / "configs" / "feature_select.yaml").write_text(yaml.safe_dump({"feature_select": {"runtime_request": {"data_source_mode": "local_feather", "sample_location": str(feather)}}}))
    cli = _context(project, "cli")
    direct = _context(project, "direct")

    assert main(["feature", "metadata", "--project", str(project), "--run-id", "cli"]) == 0
    result = ActionRunner(handlers=production_handler_registry(), policy_check=lambda *_: True).run(
        invocation=ActionInvocation(tool_name="feature_metadata", params={}, project=str(project.resolve()), version_id="direct"),
        context=direct,
        attempt_id="attempt_feature_metadata",
    )

    assert result.status == "done"
    assert _snapshot(cli, "feature_metadata") == _snapshot(direct, "feature_metadata")


def test_feature_prescreen_and_wide_local_cli_and_runner_parity(tmp_path):
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    feather = project / "sample.feather"
    pd.DataFrame({"uid": [1, 2, 3], "label": [0, 1, 0], "split": ["DEV", "DEV", "OOT"], "x": [0.1, 0.2, 0.3]}).to_feather(feather)
    (project / "configs" / "project.yml").write_text(yaml.safe_dump({"data": {"raw_path": str(feather), "id_columns": ["uid"], "target_column": "label", "split_column": "split"}}))
    (project / "configs" / "feature_select.yaml").write_text(yaml.safe_dump({"feature_select": {"runtime_request": {"data_source_mode": "local_feather", "sample_location": str(feather)}}}))
    cli = _context(project, "cli")
    direct = _context(project, "direct")
    runner = ActionRunner(handlers=production_handler_registry(), policy_check=lambda *_: True)

    assert main(["feature", "prescreen", "--project", str(project), "--run-id", "cli"]) == 0
    result = runner.run(invocation=ActionInvocation(tool_name="feature_prescreen_local", params={}, project=str(project.resolve()), version_id="direct"), context=direct, attempt_id="prescreen")
    assert result.status == "done"
    assert _snapshot(cli, "feature_prescreen") == _snapshot(direct, "feature_prescreen")

    assert main(["build-wide-sql", "--project", str(project), "--run-id", "cli"]) == 0
    result = runner.run(invocation=ActionInvocation(tool_name="build_wide_sql_local", params={}, project=str(project.resolve()), version_id="direct"), context=direct, attempt_id="wide")
    assert result.status == "done"
    assert _snapshot(cli, "build_wide_sql") == _snapshot(direct, "build_wide_sql")


def test_feature_refine_cli_and_runner_have_semantic_parity(tmp_path, monkeypatch):
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    output = project / "refine-output"
    (project / "configs" / "project.yml").write_text(yaml.safe_dump({"data": {"target_column": "label"}}))
    (project / "configs" / "feature_select.yaml").write_text(yaml.safe_dump({"feature_select": {"runtime_request": {"data_source_mode": "local_feather"}}}))
    (project / "configs" / "refine_features.yaml").write_text(yaml.safe_dump({"feature_refine": {"output_dir": str(output), "input": {"local_feather_path": "data/local.feather"}}}))

    def fake_refine(**kwargs):
        output.mkdir(parents=True, exist_ok=True)
        (output / "stage_summary.json").write_text('{"status":"done"}')
        (output / "final_features.txt").write_text("x1\nx2\n")
        return 0

    monkeypatch.setattr("risk_model_workbench.feature_selection.refine.execute_refine_action", fake_refine)
    cli = _context(project, "cli")
    direct = _context(project, "direct")
    runner = ActionRunner(handlers=production_handler_registry(), policy_check=lambda *_: True)
    assert main(["feature", "refine", "--project", str(project), "--run-id", "cli"]) == 0
    result = runner.run(invocation=ActionInvocation(tool_name="feature_refine_local", params={}, project=str(project.resolve()), version_id="direct"), context=direct, attempt_id="refine")
    assert result.status == "done"
    assert _snapshot(cli, "feature_refine") == _snapshot(direct, "feature_refine")


def test_feature_prescreen_sql_prepare_and_execute_cli_runner_parity(tmp_path, monkeypatch):
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    (project / "configs" / "project.yml").write_text(yaml.safe_dump({"data": {"target_column": "label"}}))
    (project / "configs" / "feature_select.yaml").write_text(yaml.safe_dump({"feature_select": {"runtime_request": {}}}))
    cli = _context(project, "cli")
    direct = _context(project, "direct")

    def fake_prescreen(**kwargs):
        workspace = Path(kwargs["workspace"])
        output = workspace / "feature_selection" / "prescreen" / "results"
        output.mkdir(parents=True, exist_ok=True)
        for name in ["prescreen_run_summary.json", "prescreen_final_remain_features.json", "prescreen_table_summary.csv"]:
            (output / name).write_text("{}" if name.endswith("json") else "table\n", encoding="utf-8")
        return 0

    monkeypatch.setattr("risk_model_workbench.feature_selection.refine.execute_prescreen_action", fake_prescreen)
    runner = ActionRunner(handlers=production_handler_registry(), policy_check=lambda *_: True)
    assert main(["feature", "prescreen", "--project", str(project), "--run-id", "cli"]) == 0
    result = runner.run(invocation=ActionInvocation(tool_name="feature_prescreen_prepare", params={}, project=str(project.resolve()), version_id="direct"), context=direct, attempt_id="prepare")
    assert result.status == "scaffold"
    assert _snapshot(cli, "feature_prescreen") == _snapshot(direct, "feature_prescreen")

    assert main(["feature", "prescreen", "--project", str(project), "--run-id", "cli", "--sql-approved"]) == 0
    result = runner.run(invocation=ActionInvocation(tool_name="feature_prescreen_execute", params={}, project=str(project.resolve()), version_id="direct"), context=direct, attempt_id="execute")
    assert result.status == "done"
    assert _snapshot(cli, "feature_prescreen") == _snapshot(direct, "feature_prescreen")
