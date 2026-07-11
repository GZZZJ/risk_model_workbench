import json
from pathlib import Path

import yaml

from risk_model_workbench.application.action_runner import ActionRunner
from risk_model_workbench.application.context import VersionContext
from risk_model_workbench.application.handlers import production_handler_registry
from risk_model_workbench.cli import main
from risk_model_workbench.harness.invocation import ActionInvocation
from risk_model_workbench.harness.runtime import ActionAttempt, action_attempt
from risk_model_workbench.registry import load_artifact_manifest
from risk_model_workbench.state import create_run_state, load_run_state, save_run_state


def _workspace(project: Path, run_id: str) -> VersionContext:
    workspace = project / "runs" / run_id
    (workspace / "audit").mkdir(parents=True)
    (workspace / "configs_runtime").mkdir(parents=True)
    for directory in ["evaluation", "modeling", "reports"]:
        (workspace / directory).mkdir()
    state = create_run_state(project, run_id=run_id, workflow="full_modeling")
    save_run_state(workspace, state)
    (workspace / "audit" / "artifact_manifest.json").write_text(
        json.dumps({"version": 2, "artifacts": []}), encoding="utf-8"
    )
    (workspace / "configs_runtime" / "evaluate.yaml").write_text(
        yaml.safe_dump({"evaluation": {"metrics": ["auc"], "score_columns": ["model_score"]}}),
        encoding="utf-8",
    )
    (workspace / "configs_runtime" / "report.yaml").write_text(
        yaml.safe_dump({"report": {"outputs": ["model_report.md"]}}), encoding="utf-8"
    )
    return VersionContext(
        project_dir=project.resolve(),
        version_id=run_id,
        workspace=workspace,
        runtime_config_dir=workspace / "configs_runtime",
        manifest_path=workspace / "audit" / "artifact_manifest.json",
        version_state_path=workspace / "run_state.yml",
    )


def _stage_snapshot(context: VersionContext, stage: str) -> dict:
    state = load_run_state(context.workspace)
    manifest = load_artifact_manifest(context.workspace)
    return {
        "status": state["stages"][stage]["status"],
        "scaffold": bool(state["stages"][stage].get("scaffold")),
        "artifacts": sorted(
            item["path"] for item in manifest["artifacts"] if item.get("stage") == stage
        ),
        "decisions": [
            {
                "decision": item["decision"],
                "reason": str(item["reason"]).replace(str(context.workspace), "<workspace>"),
            }
            for item in state.get("decisions", [])
            if item.get("stage") == stage
        ],
    }


def _direct(context: VersionContext, tool_name: str, params: dict, attempt_id: str):
    invocation = ActionInvocation(
        tool_name=tool_name,
        params=params,
        project=str(context.project_dir),
        version_id=context.version_id,
    )
    return ActionRunner(
        handlers=production_handler_registry(), policy_check=lambda *_: True
    ).run(invocation=invocation, context=context, attempt_id=attempt_id)


def test_evaluate_cli_and_direct_runner_have_semantic_parity(tmp_path, capsys):
    cli_context = _workspace(tmp_path / "project", "cli_eval")
    direct_context = _workspace(tmp_path / "project", "direct_eval")

    exit_code = main(
        ["evaluate", "--project", str(cli_context.project_dir), "--run-id", cli_context.version_id]
    )
    cli_output = capsys.readouterr().out
    direct_result = _direct(direct_context, "evaluate", {}, "attempt_direct_eval")

    assert exit_code == 0
    assert "evaluation scaffold:" in cli_output
    assert direct_result.status == "scaffold"
    assert _stage_snapshot(cli_context, "evaluate") == _stage_snapshot(direct_context, "evaluate")


def test_compare_cli_and_direct_runner_have_semantic_parity(tmp_path, capsys):
    cli_context = _workspace(tmp_path / "project", "cli_compare")
    direct_context = _workspace(tmp_path / "project", "direct_compare")
    for context in [cli_context, direct_context]:
        (context.workspace / "evaluation" / "benchmark_uplift.csv").write_text(
            "score,auc_uplift\nmodel_score,0.01\n", encoding="utf-8"
        )

    exit_code = main(
        [
            "compare",
            "--project",
            str(cli_context.project_dir),
            "--run-id",
            cli_context.version_id,
            "--champion",
            "legacy_score",
        ]
    )
    cli_output = capsys.readouterr().out
    direct_result = _direct(
        direct_context, "compare", {"champions": ["legacy_score"]}, "attempt_direct_compare"
    )

    assert exit_code == 0
    assert "compare complete:" in cli_output
    assert direct_result.status == "done"
    assert _stage_snapshot(cli_context, "compare") == _stage_snapshot(direct_context, "compare")


def test_report_cli_and_direct_runner_have_semantic_parity(tmp_path, capsys):
    cli_context = _workspace(tmp_path / "project", "cli_report")
    direct_context = _workspace(tmp_path / "project", "direct_report")

    exit_code = main(
        ["report", "--project", str(cli_context.project_dir), "--run-id", cli_context.version_id]
    )
    cli_output = capsys.readouterr().out
    direct_result = _direct(direct_context, "report", {}, "attempt_direct_report")

    assert exit_code == 0
    assert "report scaffold:" in cli_output
    assert direct_result.status == "scaffold"
    assert _stage_snapshot(cli_context, "report") == _stage_snapshot(direct_context, "report")
    assert (cli_context.workspace / "reports" / "model_report.html").exists()
    assert (direct_context.workspace / "reports" / "model_report.html").exists()


def test_cli_adapter_writes_exactly_one_receipt_for_outer_agent_attempt(tmp_path):
    context = _workspace(tmp_path / "project", "agent_eval")
    invocation = ActionInvocation(
        tool_name="evaluate", params={}, project=str(context.project_dir), version_id=context.version_id
    )
    attempt = ActionAttempt(
        workspace=context.workspace,
        attempt_id="attempt_agent_eval",
        task_id="evaluate_task",
        action_id="evaluate",
        invocation_hash=invocation.digest(),
        project=str(context.project_dir),
        version_id=context.version_id,
    )

    with action_attempt(attempt):
        assert main(["evaluate", "--project", str(context.project_dir), "--run-id", context.version_id]) == 0

    receipts = sorted((context.workspace / "audit" / "action_results").glob("*.json"))
    assert [item.name for item in receipts] == ["attempt_agent_eval.json"]
    payload = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert payload["task_id"] == "evaluate_task"
    assert payload["invocation_hash"] == invocation.digest()
