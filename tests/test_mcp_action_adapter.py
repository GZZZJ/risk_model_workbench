import ast
import inspect
import json
from pathlib import Path

import pytest
import yaml

from risk_model_workbench.adapters.mcp import MCPActionAdapter
import risk_model_workbench.adapters.mcp as mcp_module
from risk_model_workbench.application.action_runner import ActionRunner, HandlerRegistry
from risk_model_workbench.application.handlers import production_handler_registry
from risk_model_workbench.cli import main
from risk_model_workbench.harness.runtime import ActionResult
from risk_model_workbench.state import create_run_state, load_run_state, save_run_state


def _workspace(project: Path, run_id: str) -> Path:
    workspace = project / "runs" / run_id
    (workspace / "audit").mkdir(parents=True)
    (workspace / "configs_runtime").mkdir(parents=True)
    for name in ["evaluation", "modeling", "reports"]:
        (workspace / name).mkdir()
    save_run_state(workspace, create_run_state(project, run_id=run_id, workflow="full_modeling"))
    (workspace / "audit/artifact_manifest.json").write_text(
        json.dumps({"version": 2, "artifacts": []}), encoding="utf-8"
    )
    (workspace / "configs_runtime/evaluate.yaml").write_text(
        yaml.safe_dump({"evaluation": {"metrics": ["auc"]}}), encoding="utf-8"
    )
    return workspace


def _adapter(*, allowed=True) -> MCPActionAdapter:
    return MCPActionAdapter(
        runner=ActionRunner(
            handlers=production_handler_registry(),
            policy_check=lambda *_: allowed,
        )
    )


class _CountingRunner:
    def __init__(self, delegate):
        self.delegate = delegate
        self.calls = 0

    def run(self, **kwargs):
        self.calls += 1
        return self.delegate.run(**kwargs)


def _counting_adapter(*, allowed=True):
    handler_calls = []
    handlers = HandlerRegistry()
    handlers.register(
        "evaluate",
        lambda *_: handler_calls.append("evaluate") or ActionResult(status="done"),
    )
    runner = _CountingRunner(
        ActionRunner(handlers=handlers, policy_check=lambda *_: allowed)
    )
    return MCPActionAdapter(runner=runner), runner, handler_calls


def test_mcp_adapter_exposes_typed_capabilities_without_command_templates():
    capabilities = _adapter().capabilities()
    evaluate = next(item for item in capabilities if item["name"] == "evaluate")

    assert evaluate["action_id"] == "evaluate"
    assert evaluate["input_schema"]["additionalProperties"] is False
    assert evaluate["permission"] == "writes_run"
    assert "command" not in evaluate
    assert "argv" not in json.dumps(capabilities).lower()
    evaluate["input_schema"]["properties"].clear()
    refreshed = next(item for item in _adapter().capabilities() if item["name"] == "evaluate")
    assert "scores_feather" in refreshed["input_schema"]["properties"]


def test_mcp_adapter_rejects_unknown_or_invalid_params_before_runner(tmp_path):
    project = tmp_path / "project"
    _workspace(project, "bad_params")
    adapter, runner, handler_calls = _counting_adapter()

    with pytest.raises(ValueError, match="unknown_param:bogus"):
        adapter.call_action(
            tool_name="evaluate",
            params={"bogus": True},
            project=project,
            version_id="bad_params",
            attempt_id="attempt_bad",
        )
    assert runner.calls == 0
    assert handler_calls == []


def test_mcp_adapter_calls_runner_and_handler_exactly_once_on_success(tmp_path):
    project = tmp_path / "project"
    _workspace(project, "success")
    adapter, runner, handler_calls = _counting_adapter()

    result = adapter.call_action(
        tool_name="evaluate",
        params={},
        project=project,
        version_id="success",
        attempt_id="attempt_success",
    )

    assert result["status"] == "done"
    assert runner.calls == 1
    assert handler_calls == ["evaluate"]


def test_mcp_adapter_preserves_action_runner_policy_denial(tmp_path):
    project = tmp_path / "project"
    _workspace(project, "denied")
    adapter, runner, handler_calls = _counting_adapter(allowed=False)

    with pytest.raises(PermissionError, match="denied by policy"):
        adapter.call_action(
            tool_name="evaluate",
            params={},
            project=project,
            version_id="denied",
            attempt_id="attempt_denied",
        )
    assert runner.calls == 1
    assert handler_calls == []


def test_mcp_and_cli_evaluate_have_semantic_parity(tmp_path, capsys):
    project = tmp_path / "project"
    cli_workspace = _workspace(project, "cli_eval")
    mcp_workspace = _workspace(project, "mcp_eval")

    assert main(["evaluate", "--project", str(project), "--run-id", "cli_eval"]) == 0
    capsys.readouterr()
    payload = _adapter().call_action(
        tool_name="evaluate",
        params={},
        project=project,
        version_id="mcp_eval",
        attempt_id="attempt_mcp_eval",
        task_id="evaluate",
    )

    cli_result = load_run_state(cli_workspace)["stages"]["evaluate"]["last_result"]
    mcp_result = load_run_state(mcp_workspace)["stages"]["evaluate"]["last_result"]
    for field in ["status", "failure_code", "next_required_action", "retryable", "scaffold"]:
        assert payload[field] == cli_result[field] == mcp_result[field]
    for result in [payload, cli_result, mcp_result]:
        decision = result["decision"] or {}
        assert {key: value for key, value in decision.items() if key != "created_at"} == {
            "stage": "evaluate",
            "decision": "scaffold",
            "reason": "prediction data not available",
        }
    assert payload["attempt_id"] == "attempt_mcp_eval"
    assert (mcp_workspace / "audit/action_results/attempt_mcp_eval.json").exists()


def test_mcp_adapter_has_no_state_retry_cli_or_shell_execution_logic():
    tree = ast.parse(inspect.getsource(mcp_module))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "risk_model_workbench.cli" not in imported_modules
    assert "risk_model_workbench.state" not in imported_modules
    assert "subprocess" not in imported_modules
    assert not ({"system", "popen", "run_with_retry"} & called_names)
