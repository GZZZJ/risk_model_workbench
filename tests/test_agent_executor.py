import json
import ast
import inspect
from pathlib import Path

import yaml

from risk_model_workbench.agent.executor import run_agent
from risk_model_workbench.agent.eval import emit_scenario_evidence
import risk_model_workbench.agent.executor as executor_module
from risk_model_workbench.application.action_runner import HandlerRegistry
from risk_model_workbench.application.handlers import production_handler_registry
import risk_model_workbench.application.handlers.feature_selection as feature_handler_module
import risk_model_workbench.application.handlers.sample_check as sample_handler_module
import risk_model_workbench.application.handlers.train as train_handler_module
import risk_model_workbench.feature_selection.metadata as metadata_service_module
import risk_model_workbench.feature_selection.refine as refine_service_module
import risk_model_workbench.feature_metadata as metadata_domain_module
import risk_model_workbench.batch_feature_select as prescreen_domain_module
import risk_model_workbench.feature_refine as refine_domain_module
from risk_model_workbench.harness.invocation import ActionInvocation
from risk_model_workbench.harness.tools import TOOL_REGISTRY, registry_digest
from risk_model_workbench.agent.plan import canonical_plan_hash, save_agent_plan
from risk_model_workbench.agent.state import init_agent_state, load_agent_state
from risk_model_workbench.cli import main
from risk_model_workbench.harness.runtime import ActionResult, stage_action_failed


def test_agent_executor_runs_safe_task_and_marks_scaffold_gap(tmp_path):
    project = _make_project(tmp_path)
    version_id = "demo_model_v1_20260706"
    assert main(["version", "init", "--project", str(project), "--workflow", "sample_audit", "--version-id", version_id]) == 0
    workspace = project / "versions" / version_id
    plan = _agent_plan(project, version_id, [_task("sample_check_001", ["sample", "check", "--project", str(project), "--version-id", version_id])])
    save_agent_plan(workspace, plan)
    initial = init_agent_state(workspace, project=str(project), version_id=version_id, agent_plan=plan)

    result = run_agent(project, version_id, runner=main)
    state = load_agent_state(workspace)

    assert result["status"] == "done_with_gaps"
    assert state["tasks"][0]["status"] == "scaffold"
    assert (workspace / "audit" / "agent_trace.jsonl").exists()
    trace_rows = [json.loads(line) for line in (workspace / "audit" / "agent_trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(
        row.get("event") == "result"
        and (row.get("status") == "scaffold" or (row.get("stage_result") or {}).get("status") == "scaffold")
        for row in trace_rows
    )
    emit_scenario_evidence(
        state_pairs=[(initial, state)],
        workspace=workspace,
    )


def test_agent_executor_default_runs_through_action_runner(tmp_path):
    project = _make_project(tmp_path)
    version_id = "demo_model_v1_20260707"
    assert main(["version", "init", "--project", str(project), "--workflow", "sample_audit", "--version-id", version_id]) == 0
    workspace = project / "versions" / version_id
    plan = _agent_plan(project, version_id, [_task("sample_check_001", ["sample", "check", "--project", str(project), "--version-id", version_id])])
    save_agent_plan(workspace, plan)
    init_agent_state(workspace, project=str(project), version_id=version_id, agent_plan=plan)

    result = run_agent(project, version_id)

    assert result["status"] == "done_with_gaps"
    receipts = list((workspace / "audit" / "action_results").glob("*.json"))
    assert len(receipts) == 1


def test_agent_run_cli_preserves_bound_invocation_hash_for_typed_defaults(tmp_path):
    project = _make_project(tmp_path)
    version_id = "demo_model_v1_20260708"
    assert main(["version", "init", "--project", str(project), "--workflow", "feature_selection", "--version-id", version_id]) == 0
    workspace = project / "versions" / version_id
    (workspace / "configs_runtime" / "feature_select.yaml").write_text(
        yaml.safe_dump({"feature_select": {"runtime_request": {"data_source_mode": "local_feather"}}}, sort_keys=False),
        encoding="utf-8",
    )
    invocation = ActionInvocation("build_wide_sql_local", {}, str(project), version_id)
    spec = TOOL_REGISTRY[invocation.tool_name]
    task = {
        "task_id": "build_wide_sql",
        "depends_on": [],
        "invocation": invocation.canonical_payload(),
        "invocation_hash": invocation.digest(),
        "command": {"executable": "rmw", "args": spec.render_argv(invocation)},
        "action_id": spec.action_id,
        "tool_name": spec.name,
        "derived_metadata": {
            "action_id": spec.action_id,
            "permission": spec.permission,
            "requires_approval": spec.requires_approval,
            "allowed_for_auditor": spec.allowed_for_auditor,
            "execution_semantics": spec.execution_semantics,
            "approval_type": spec.approval_type,
        },
    }
    plan = {
        "version": 2,
        "plan_id": "typed_defaults_plan",
        "project": str(project),
        "version_id": version_id,
        "workflow": "feature_selection",
        "tasks": [task],
        "registry_digest": registry_digest(),
    }
    plan["plan_hash"] = canonical_plan_hash(plan)
    save_agent_plan(workspace, plan)
    init_agent_state(workspace, project=str(project), version_id=version_id, agent_plan=plan)

    assert main(["agent", "run", "--project", str(project), "--version-id", version_id]) == 0

    receipt = next((workspace / "audit/action_results").glob("*.json"))
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["invocation_hash"] == invocation.digest()
    state = load_agent_state(workspace)
    assert state["status"] == "done_with_gaps"
    assert state["tasks"][0]["status"] == "done"


def test_action_runner_advisor_failure_pauses_once_and_creates_immutable_request(
    tmp_path, monkeypatch
):
    project = _make_project(tmp_path)
    version_id = "demo_model_v1_20260709"
    assert main(["version", "init", "--project", str(project), "--workflow", "train_baseline", "--version-id", version_id]) == 0
    workspace = project / "versions" / version_id
    invocation = ActionInvocation(
        "train_baseline", {"experiment": "main"}, str(project), version_id
    )
    spec = TOOL_REGISTRY[invocation.tool_name]
    task = {
        "task_id": "train_main",
        "depends_on": [],
        "invocation": invocation.canonical_payload(),
        "invocation_hash": invocation.digest(),
        "command": {"executable": "rmw", "args": spec.render_argv(invocation)},
        "action_id": spec.action_id,
        "tool_name": spec.name,
        "derived_metadata": {
            "action_id": spec.action_id,
            "permission": spec.permission,
            "requires_approval": spec.requires_approval,
            "allowed_for_auditor": spec.allowed_for_auditor,
            "execution_semantics": spec.execution_semantics,
            "approval_type": spec.approval_type,
        },
    }
    plan = {
        "version": 2,
        "plan_id": "advisor_pause_plan",
        "project": str(project),
        "version_id": version_id,
        "workflow": "train_baseline",
        "tasks": [task],
        "registry_digest": registry_digest(),
    }
    plan["plan_hash"] = canonical_plan_hash(plan)
    save_agent_plan(workspace, plan)
    init_agent_state(workspace, project=str(project), version_id=version_id, agent_plan=plan)
    handler_calls = []
    handlers = HandlerRegistry()

    def require_advisor(_invocation, context, _attempt_id):
        handler_calls.append("train")
        tuning_context = context.workspace / "modeling/main/tuning_context_round_1.json"
        tuning_context.parent.mkdir(parents=True, exist_ok=True)
        tuning_context.write_text('{"round":1,"experiment":"main"}\n', encoding="utf-8")
        return ActionResult(
            status="failed",
            failure_code="advisor_required",
            message=f"host agent tuning plan required from {tuning_context}",
        )

    handlers.register("train_baseline", require_advisor)
    monkeypatch.setattr(executor_module, "production_handler_registry", lambda: handlers)

    first = run_agent(project, version_id)
    request_paths = sorted((workspace / "audit/advisor_requests").glob("*.json"))
    context_paths = sorted((workspace / "audit/context_packs").glob("*.json"))
    request_bytes = request_paths[0].read_bytes()

    assert first["status"] == "waiting_for_advisor"
    assert handler_calls == ["train"]
    assert len(request_paths) == 1
    assert len(context_paths) == 1
    receipt = next((workspace / "audit/action_results").glob("*.json"))
    assert json.loads(receipt.read_text(encoding="utf-8"))["next_required_action"] == "advisor"

    second = run_agent(project, version_id)

    assert second["status"] == "waiting_for_advisor"
    assert handler_calls == ["train"]
    assert sorted((workspace / "audit/advisor_requests").glob("*.json")) == request_paths
    assert request_paths[0].read_bytes() == request_bytes


def test_production_handlers_cover_every_workflow_write_action():
    registry = production_handler_registry()
    write_actions = {
        spec.action_id for spec in TOOL_REGISTRY.values() if spec.permission in {"writes_run", "dp_sql_pull"}
    }
    assert write_actions <= registry.action_ids()


def test_agent_executor_has_no_cli_main_import_or_call():
    tree = ast.parse(inspect.getsource(executor_module))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "risk_model_workbench.cli":
            assert all(alias.name != "main" for alias in node.names)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert not (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "cli"
                and node.func.attr == "main"
            )


def test_application_handlers_do_not_import_cli():
    for module in [feature_handler_module, sample_handler_module, train_handler_module]:
        tree = ast.parse(inspect.getsource(module))
        assert not any(
            isinstance(node, ast.ImportFrom) and node.module == "risk_model_workbench.cli"
            for node in ast.walk(tree)
        )


def test_handler_dependency_modules_do_not_route_through_main_or_argv():
    modules = [metadata_service_module, refine_service_module]
    for module in modules:
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert all(alias.name != "main" for alias in node.names)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id != "main"
            if isinstance(node, ast.Name):
                assert node.id != "argv"


def test_transitive_handler_service_functions_are_typed_not_cli_routed():
    callables = [
        metadata_service_module.execute_metadata_action,
        refine_service_module.execute_prescreen_action,
        refine_service_module.execute_refine_action,
        metadata_domain_module.run_metadata_service,
        prescreen_domain_module.run_prescreen_service,
        prescreen_domain_module._run_prescreen,
        refine_domain_module.run_refine_service,
        refine_domain_module._run_refine,
    ]
    for target in callables:
        tree = ast.parse(inspect.getsource(target))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert all(alias.name != "main" for alias in node.names)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id != "main"
            if isinstance(node, ast.Name):
                assert node.id != "argv"


def test_agent_executor_pauses_for_approval_and_can_resume_after_approval(tmp_path):
    project = _make_project(tmp_path)
    version_id = "demo_model_v1_20260706"
    assert main(["version", "init", "--project", str(project), "--workflow", "sample_audit", "--version-id", version_id]) == 0
    workspace = project / "versions" / version_id
    task = _task(
        "dp_pull",
        ["feature", "prescreen", "--project", str(project), "--version-id", version_id, "--sql-approved"],
        permission="dp_sql_pull",
        requires_approval=True,
    )
    plan = _agent_plan(project, version_id, [task])
    save_agent_plan(workspace, plan)
    init_agent_state(workspace, project=str(project), version_id=version_id, agent_plan=plan)

    result = run_agent(project, version_id, runner=main)
    state = load_agent_state(workspace)
    approval_id = state["blocker"]["approval_id"]

    assert result["status"] == "waiting_for_approval"
    assert (workspace / "audit" / "approvals.yml").exists()

    assert main(["agent", "approve", "--project", str(project), "--version-id", version_id, "--approval-id", approval_id, "--approved-by", "pm", "--note", "sql reviewed"]) == 0
    approved = yaml.safe_load((workspace / "audit" / "approvals.yml").read_text(encoding="utf-8"))
    assert approved["approvals"][0]["status"] == "approved"


def test_agent_executor_pauses_for_advisor_required_failure(tmp_path):
    project = _make_project(tmp_path)
    version_id = "demo_model_v1_20260706"
    assert main(["version", "init", "--project", str(project), "--workflow", "train_baseline", "--version-id", version_id]) == 0
    workspace = project / "versions" / version_id
    task = _task("train_main", ["train", "--project", str(project), "--version-id", version_id, "--experiment", "main_lgbm"])
    plan = _agent_plan(project, version_id, [task])
    save_agent_plan(workspace, plan)
    init_agent_state(workspace, project=str(project), version_id=version_id, agent_plan=plan)

    def advisor_runner(_argv):
        stage_action_failed(workspace, "train_baseline", "host agent plan required", failure_code="advisor_required")
        return 1

    result = run_agent(project, version_id, runner=advisor_runner)
    state = load_agent_state(workspace)
    requests = sorted((workspace / "audit" / "advisor_requests").glob("*.json"))

    assert result["status"] == "waiting_for_advisor"
    assert state["blocker"]["reason"] == "advisor_required"
    assert requests
    payload = json.loads(requests[0].read_text(encoding="utf-8"))
    assert payload["task_id"] == "train_main"


def _agent_plan(project: Path, version_id: str, tasks: list[dict]) -> dict:
    return {
        "version": 1,
        "plan_id": "agent_plan",
        "version_id": version_id,
        "project": str(project),
        "workflow": "sample_audit",
        "tasks": tasks,
    }


def _task(
    task_id: str,
    args: list[str],
    *,
    permission: str = "writes_run",
    requires_approval: bool = False,
) -> dict:
    return {
        "task_id": task_id,
        "type": "sample_check",
        "status": "pending",
        "workspace": f"tasks/{task_id}",
        "depends_on": [],
        "command": {"executable": "rmw", "args": args},
        "outputs": ["sample_check/sample_summary.json"],
        "scenario_profile": "generic",
        "step_ids": [],
        "step_params": {},
        "action_id": "sample_check" if task_id.startswith("sample") else "feature_prescreen" if task_id == "dp_pull" else "train_baseline",
        "tool_name": "sample_check" if task_id.startswith("sample") else "feature_prescreen_pull" if task_id == "dp_pull" else "train_baseline",
        "permission": permission,
        "requires_approval": requires_approval,
        "expected_outputs": ["sample_check/sample_summary.json"],
    }


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
