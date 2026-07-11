from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace

import pytest

from risk_model_workbench.agent.plan import (
    canonical_plan_hash,
    rebind_agent_plan,
    save_agent_plan,
    validate_agent_plan,
)
from risk_model_workbench.agent.executor import run_agent
from risk_model_workbench.agent.eval import emit_scenario_evidence
from risk_model_workbench.agent.state import init_agent_state, save_agent_state
from risk_model_workbench.agent.policy import evaluate_task_policy
from risk_model_workbench.harness.invocation import ActionInvocation
from risk_model_workbench.harness.tools import TOOL_REGISTRY, ToolSpec, registry_digest


def test_action_invocation_uses_one_recursive_immutable_snapshot():
    original = {"experiment": "baseline", "nested": {"values": [1, 2]}}
    invocation = ActionInvocation(
        tool_name="train_baseline",
        params=original,
        project="projects/demo",
        version_id="demo_v1",
    )
    digest = invocation.digest()

    original["experiment"] = "tampered"
    original["nested"]["values"].append(3)

    assert invocation.digest() == digest
    assert invocation.canonical_payload()["params"] == {
        "experiment": "baseline",
        "nested": {"values": [1, 2]},
    }
    with pytest.raises(TypeError):
        invocation.params["experiment"] = "tampered"
    with pytest.raises((AttributeError, TypeError)):
        invocation.params["nested"]["values"].append(3)


@pytest.mark.parametrize("invalid", [{1: "bad"}, {"x": float("nan")}, {"x": object()}])
def test_action_invocation_rejects_noncanonical_values(invalid):
    with pytest.raises((TypeError, ValueError)):
        ActionInvocation("sample_check", invalid, "projects/demo", "demo_v1")


def test_registry_digest_ignores_callable_identity_but_tracks_stable_metadata():
    base = TOOL_REGISTRY["sample_check"]
    same_metadata = replace(base, render_argv=lambda invocation: ["not", "serialized"])
    changed_metadata = replace(base, approval_type="human_confirmation")

    assert registry_digest({base.name: base}) == registry_digest({base.name: same_metadata})
    assert registry_digest({base.name: base}) != registry_digest({base.name: changed_metadata})


def test_registry_digest_is_stable_across_fresh_processes():
    command = [
        sys.executable,
        "-c",
        "from risk_model_workbench.harness.tools import registry_digest; print(registry_digest())",
    ]
    env = {**os.environ, "PYTHONPATH": "src"}
    first = subprocess.check_output(command, text=True, env=env).strip()
    second = subprocess.check_output(command, text=True, env=env).strip()
    assert first == second == registry_digest()


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (lambda plan: plan["tasks"].append(dict(plan["tasks"][0])), "duplicate_task_id"),
        (lambda plan: plan["tasks"][0].update(depends_on=["missing"]), "missing_dependency"),
        (lambda plan: plan["tasks"].append(_task("second", depends_on=["first"])), "cyclic_dependency"),
        (lambda plan: plan["tasks"][0]["invocation"].update(tool_name="missing"), "unknown_tool"),
        (lambda plan: plan["tasks"][0]["invocation"].update(project="projects/other"), "project_mismatch"),
        (lambda plan: plan["tasks"][0]["invocation"].update(version_id="other_v1"), "version_mismatch"),
        (lambda plan: plan["tasks"][0]["derived_metadata"].update(permission="dp_sql_pull"), "derived_metadata_drift"),
        (lambda plan: plan["tasks"][0].update(permission="dp_sql_pull"), "copied_authority_present"),
        (lambda plan: plan["tasks"][0]["command"].update(args=["--force"]), "rendered_argv_drift"),
        (lambda plan: plan["tasks"][0].update(action_id="report"), "copied_metadata_drift"),
    ],
)
def test_plan_validation_fails_closed_for_tampering(mutation, expected_code):
    plan = _plan()
    mutation(plan)
    if expected_code == "cyclic_dependency":
        plan["tasks"][0]["depends_on"] = ["second"]
    errors = validate_agent_plan(plan, TOOL_REGISTRY)
    assert any(error.startswith(expected_code) for error in errors)


def test_plan_validation_rejects_registry_drift_and_blocked_flags():
    plan = _plan()
    plan["registry_digest"] = "stale"
    drift_errors = validate_agent_plan(plan, TOOL_REGISTRY)
    assert any(error.startswith("registry_digest_drift") for error in drift_errors)

    forced = _plan(params={"extra_args": ["--force"]})
    forced_errors = validate_agent_plan(forced, TOOL_REGISTRY)
    assert any(error.startswith("blocked_flag") for error in forced_errors)
    emit_scenario_evidence(
        validation_errors=[*drift_errors, *forced_errors],
    )


def test_explicit_empty_registry_never_falls_back_to_global_registry(tmp_path):
    assert registry_digest({}) != registry_digest()
    assert any(error.startswith("unknown_tool") for error in validate_agent_plan(_plan(), {}))
    workspace = tmp_path / "version"
    (workspace / "audit").mkdir(parents=True)
    decision = evaluate_task_policy(
        ActionInvocation("sample_check", {}, "projects/demo", "demo_v1"),
        workspace,
        registry={},
    )
    assert decision.allowed is False
    assert decision.reason == "unregistered_tool"


def test_policy_uses_registry_not_copied_permission(tmp_path):
    workspace = tmp_path / "version"
    (workspace / "audit").mkdir(parents=True)
    invocation = ActionInvocation("sample_check", {}, "projects/demo", "demo_v1")

    decision = evaluate_task_policy(invocation, workspace, task_id="sample")
    assert decision.allowed is True

    plan = _plan()
    plan["tasks"][0]["derived_metadata"]["permission"] = "dp_sql_pull"
    assert any(error.startswith("derived_metadata_drift") for error in validate_agent_plan(plan, TOOL_REGISTRY))


def test_plan_rebind_is_previewable_and_restores_only_derived_copies():
    plan = _plan()
    changed_spec = replace(TOOL_REGISTRY["sample_check"], approval_type="human_confirmation")
    changed_registry = {**TOOL_REGISTRY, "sample_check": changed_spec}

    rebound, preview = rebind_agent_plan(plan, changed_registry)

    assert preview["changed"] is True
    assert preview["changes"]
    assert rebound["registry_digest"] == registry_digest(changed_registry)
    assert rebound["plan_hash"] == canonical_plan_hash(rebound)
    assert rebound["tasks"][0]["invocation"] == plan["tasks"][0]["invocation"]
    assert rebound["tasks"][0]["invocation_hash"] == plan["tasks"][0]["invocation_hash"]
    assert rebound["tasks"][0]["command"]["args"] == [
        "sample",
        "check",
        "--project",
        "projects/demo",
        "--version-id",
        "demo_v1",
    ]


def test_plan_rebind_refuses_to_resign_authoritative_invocation_tampering():
    plan = _plan()
    plan["tasks"][0]["invocation"]["project"] = "projects/other"
    with pytest.raises(ValueError, match="unsafe rebind source"):
        rebind_agent_plan(plan, TOOL_REGISTRY)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "missing_dependency",
        "cycle",
        "unknown_tool",
        "project_mismatch",
        "version_mismatch",
        "derived_permission",
        "copied_permission",
        "argv",
        "registry",
        "blocked_flag",
        "action_id",
        "nonmapping_params",
    ],
)
def test_executor_rejects_every_invalid_plan_before_runner_call(tmp_path, mutation):
    project = tmp_path / "project"
    workspace = project / "versions" / "demo_v1"
    workspace.mkdir(parents=True)
    plan = _plan(project=str(project))
    _mutate(plan, mutation)
    save_agent_plan(workspace, plan)
    calls: list[list[str]] = []

    with pytest.raises(ValueError, match="invalid agent plan") as exc_info:
        run_agent(project, "demo_v1", runner=lambda argv: calls.append(argv) or 0)

    assert calls == []
    if mutation == "cycle":
        emit_scenario_evidence(
            workspace=workspace,
            validation_errors=[str(exc_info.value)],
            runner_calls=calls,
        )


def test_executor_binds_plan_and_state_to_runtime_scope_before_trace_or_runner(tmp_path):
    target = tmp_path / "target"
    workspace = target / "versions" / "demo_v1"
    workspace.mkdir(parents=True)
    copied = _plan(project=str(tmp_path / "source"))
    save_agent_plan(workspace, copied)
    calls: list[list[str]] = []
    with pytest.raises(ValueError, match="runtime_project_mismatch"):
        run_agent(target, "demo_v1", runner=lambda argv: calls.append(argv) or 0)
    assert calls == []
    assert not (workspace / "audit" / "agent_trace.jsonl").exists()

    valid = _plan(project=str(target))
    save_agent_plan(workspace, valid)
    for field, value, message in [
        ("project", str(tmp_path / "other"), "state project"),
        ("version_id", "other_v1", "state version_id"),
        ("plan_id", "other_plan", "state plan_id"),
    ]:
        state = init_agent_state(workspace, project=str(target), version_id="demo_v1", agent_plan=valid)
        state[field] = value
        save_agent_state(workspace, state)
        with pytest.raises(ValueError, match=message):
            run_agent(target, "demo_v1", runner=lambda argv: calls.append(argv) or 0)
    assert calls == []


def test_executor_rejects_runtime_version_mismatch_before_runner(tmp_path):
    project = tmp_path / "project"
    workspace = project / "versions" / "runtime_v2"
    workspace.mkdir(parents=True)
    plan = _plan(project=str(project), version_id="bound_v1")
    save_agent_plan(workspace, plan)
    calls: list[list[str]] = []
    with pytest.raises(ValueError, match="runtime_version_mismatch"):
        run_agent(project, "runtime_v2", runner=lambda argv: calls.append(argv) or 0)
    assert calls == []
    assert not (workspace / "audit" / "agent_trace.jsonl").exists()


@pytest.mark.parametrize(
    ("status", "blocker"),
    [
        (
            "waiting_for_approval",
            {
                "blocker_type": "approval",
                "blocker_id": "approval_1",
                "approval_id": "approval_1",
                "next_safe_action": {"action": "consume_bound_approval", "required_evidence": "receipt"},
            },
        ),
        (
            "waiting_for_advisor",
            {
                "blocker_type": "advisor",
                "blocker_id": "advisor_1",
                "advisor_request_id": "advisor_1",
                "response_status": "accepted",
                "next_safe_action": {"action": "consume_advisor_response", "required_evidence": "receipt"},
            },
        ),
        (
            "waiting_for_user",
            {
                "blocker_type": "user",
                "blocker_id": "user_1",
                "advisor_request_id": "advisor_1",
                "next_safe_action": {"action": "confirm_advisor_decision", "required_evidence": "receipt"},
            },
        ),
        (
            "blocked",
            {
                "blocker_type": "dependency",
                "blocker_id": "dependency_1",
                "next_safe_action": {"action": "resolve_dependency", "required_evidence": "receipt"},
            },
        ),
        (
            "reconciliation_required",
            {
                "blocker_type": "reconciliation",
                "blocker_id": "operation_1",
                "next_safe_action": {"action": "reconcile_operation", "required_evidence": "receipt"},
            },
        ),
    ],
)
def test_executor_never_generic_resets_v2_blockers(tmp_path, status, blocker):
    project = tmp_path / "project"
    workspace = project / "versions" / "demo_v1"
    workspace.mkdir(parents=True)
    plan = _plan(project=str(project))
    save_agent_plan(workspace, plan)
    state = init_agent_state(workspace, project=str(project), version_id="demo_v1", agent_plan=plan)
    state["status"] = status
    state["tasks"][0]["status"] = "reconciliation_required" if status == "reconciliation_required" else "paused"
    state["blocker"] = blocker
    save_agent_state(workspace, state)
    calls: list[list[str]] = []

    result = run_agent(project, "demo_v1", runner=lambda argv: calls.append(argv) or 0)

    assert result["status"] == status
    assert calls == []


def test_command_template_is_display_only_and_cannot_change_typed_argv():
    base = TOOL_REGISTRY["sample_check"]
    changed_display = ToolSpec(
        name=base.name,
        action_id=base.action_id,
        command="rmw report --project totally-different",
        permission=base.permission,
        description=base.description,
    )
    invocation = ActionInvocation("sample_check", {}, "projects/demo", "demo_v1")
    assert changed_display.command_template != base.command_template
    assert changed_display.render_argv(invocation) == base.render_argv(invocation)


def _plan(*, params=None, project="projects/demo", version_id="demo_v1"):
    task = _task("first", params=params, project=project, version_id=version_id)
    plan = {
        "version": 2,
        "plan_id": "demo_agent_plan",
        "project": project,
        "version_id": version_id,
        "workflow": "sample_audit",
        "tasks": [task],
        "registry_digest": registry_digest(TOOL_REGISTRY),
    }
    plan["plan_hash"] = canonical_plan_hash(plan)
    return plan


def _task(task_id: str, *, depends_on=None, params=None, project="projects/demo", version_id="demo_v1"):
    invocation = ActionInvocation(
        "sample_check",
        params or {},
        project,
        version_id,
    )
    spec = TOOL_REGISTRY["sample_check"]
    return {
        "task_id": task_id,
        "depends_on": list(depends_on or []),
        "invocation": invocation.canonical_payload(),
        "invocation_hash": invocation.digest(),
        "command": {"executable": "rmw", "args": spec.render_argv(invocation)},
        "derived_metadata": {
            "action_id": spec.action_id,
            "permission": spec.permission,
            "requires_approval": spec.requires_approval,
            "allowed_for_auditor": spec.allowed_for_auditor,
            "execution_semantics": spec.execution_semantics,
            "approval_type": spec.approval_type,
        },
    }


def _mutate(plan, mutation):
    task = plan["tasks"][0]
    if mutation == "duplicate":
        plan["tasks"].append(dict(task))
    elif mutation == "missing_dependency":
        task["depends_on"] = ["missing"]
    elif mutation == "cycle":
        second = _task(
            "second",
            depends_on=["first"],
            project=plan["project"],
            version_id=plan["version_id"],
        )
        task["depends_on"] = ["second"]
        plan["tasks"].append(second)
    elif mutation == "unknown_tool":
        task["invocation"]["tool_name"] = "missing"
    elif mutation == "project_mismatch":
        task["invocation"]["project"] = "projects/other"
    elif mutation == "version_mismatch":
        task["invocation"]["version_id"] = "other_v1"
    elif mutation == "derived_permission":
        task["derived_metadata"]["permission"] = "dp_sql_pull"
    elif mutation == "copied_permission":
        task["permission"] = "dp_sql_pull"
    elif mutation == "argv":
        task["command"]["args"] = ["--force"]
    elif mutation == "registry":
        plan["registry_digest"] = "stale"
    elif mutation == "blocked_flag":
        task["invocation"]["params"] = {"extra_args": ["--force"]}
    elif mutation == "action_id":
        task["action_id"] = "report"
    elif mutation == "nonmapping_params":
        task["invocation"]["params"] = []
    else:
        raise AssertionError(mutation)
