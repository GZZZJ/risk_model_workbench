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


def test_sample_check_cli_and_action_runner_have_valid_input_parity(tmp_path, capsys):
    """A real local sample must preserve its semantic profile across both entrypoints."""
    project = tmp_path / "project"
    cli_context = _workspace(project, "cli_sample")
    direct_context = _workspace(project, "direct_sample")
    raw = project / "data" / "sample.csv"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(
        "uid,label,split,event_time,channel,amount\n"
        "1,0,DEV,2026-01-02,APP,10\n"
        "2,1,DEV,2026-01-15,WEB,20\n"
        "2,1,OOT,2026-02-03,APP,30\n",
        encoding="utf-8",
    )
    config = yaml.safe_load((project / "project.yml").read_text(encoding="utf-8"))
    config["data"].update(
        {
            "raw_path": "data/sample.csv",
            "id_columns": ["uid"],
            "split_column": "split",
            "time_column": "event_time",
            "segment_columns": ["channel"],
        }
    )
    payload = yaml.safe_dump(config)
    (project / "project.yml").write_text(payload, encoding="utf-8")
    for context in [cli_context, direct_context]:
        (context.runtime_config_dir / "project.yml").write_text(payload, encoding="utf-8")

    assert main(["sample", "check", "--project", str(project), "--run-id", "cli_sample"]) == 0
    assert "local data not available" not in capsys.readouterr().out
    result = ActionRunner(
        handlers=production_handler_registry(), policy_check=lambda *_: True
    ).run(
        invocation=ActionInvocation(
            tool_name="sample_check", params={}, project=str(project.resolve()), version_id="direct_sample"
        ),
        context=direct_context,
        attempt_id="attempt_sample_valid",
    )

    assert result.status == "done"
    assert _snapshot(cli_context) == _snapshot(direct_context)
    for context in [cli_context, direct_context]:
        summary = json.loads(
            (context.workspace / "sample_check" / "sample_summary.json").read_text(encoding="utf-8")
        )
        assert summary["status"] == "done"
        assert summary["rows"] == 3
        assert summary["duplicate_key_rows"] == 1
        assert (context.workspace / "sample_check" / "label_distribution.csv").exists()
        assert (context.workspace / "sample_check" / "sample_split_summary.csv").exists()
        monthly = pd.read_csv(
            context.workspace / "sample_check" / "monthly_label_distribution.csv"
        )
        assert monthly.to_dict("records") == [
            {"_month": "2026-01", "samples": 2, "positive": 1, "target_rate": 0.5},
            {"_month": "2026-02", "samples": 1, "positive": 1, "target_rate": 1.0},
        ]
        segments = pd.read_csv(context.workspace / "sample_check" / "segment_distribution.csv")
        assert segments[["segment_column", "segment_value", "count"]].to_dict("records") == [
            {"segment_column": "channel", "segment_value": "APP", "count": 2},
            {"segment_column": "channel", "segment_value": "WEB", "count": 1},
        ]


def test_sample_check_cli_and_action_runner_have_failure_parity(tmp_path, capsys, monkeypatch):
    project = tmp_path / "project"
    cli_context = _workspace(project, "cli_sample")
    direct_context = _workspace(project, "direct_sample")
    raw = project / "data" / "sample.csv"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text("uid,label\n1,0\n", encoding="utf-8")
    config = yaml.safe_load((project / "project.yml").read_text(encoding="utf-8"))
    config["data"]["raw_path"] = "data/sample.csv"
    payload = yaml.safe_dump(config)
    (project / "project.yml").write_text(payload, encoding="utf-8")
    for context in [cli_context, direct_context]:
        (context.runtime_config_dir / "project.yml").write_text(payload, encoding="utf-8")

    def fail_read_csv(*_args, **_kwargs):
        raise OSError("sample read failed")

    monkeypatch.setattr(pd, "read_csv", fail_read_csv)
    assert main(["sample", "check", "--project", str(project), "--run-id", "cli_sample"]) == 1
    assert "sample read failed" in capsys.readouterr().err
    result = ActionRunner(
        handlers=production_handler_registry(), policy_check=lambda *_: True
    ).run(
        invocation=ActionInvocation(
            tool_name="sample_check",
            params={},
            project=str(project.resolve()),
            version_id="direct_sample",
        ),
        context=direct_context,
        attempt_id="attempt_sample_failure",
    )

    assert result.status == "failed"
    assert result.message == "sample read failed"
    assert _snapshot(cli_context) == _snapshot(direct_context)
