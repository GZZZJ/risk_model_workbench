import json
from pathlib import Path

import pytest
import yaml

from risk_model_workbench.application.action_runner import ActionRunner, HandlerRegistry
from risk_model_workbench.application.context import VersionContext
from risk_model_workbench.harness.errors import DuplicateActionResultError
from risk_model_workbench.harness.invocation import ActionInvocation
from risk_model_workbench.harness.runtime import ActionResult


def _context(tmp_path: Path) -> VersionContext:
    project = tmp_path / "project"
    context = VersionContext.from_project(project, "v1")
    (context.workspace / "audit").mkdir(parents=True)
    context.version_state_path.write_text(
        yaml.safe_dump({"status": "running", "stages": {"sample_check": {"status": "pending", "artifacts": []}}}),
        encoding="utf-8",
    )
    context.manifest_path.write_text(json.dumps({"artifacts": []}), encoding="utf-8")
    return context


def _invocation(context: VersionContext) -> ActionInvocation:
    return ActionInvocation(
        tool_name="sample_check",
        params={},
        project=str(context.project_dir),
        version_id=context.version_id,
    )


def test_runner_checks_policy_then_calls_registered_handler_and_writes_correlated_receipt(tmp_path):
    context = _context(tmp_path)
    events = []
    handlers = HandlerRegistry()

    def handler(invocation, received_context, attempt_id):
        events.append(("handler", attempt_id))
        assert received_context == context
        return ActionResult(status="done", message="ok")

    handlers.register("sample_check", handler)
    runner = ActionRunner(
        handlers=handlers,
        policy_check=lambda invocation, received_context: events.append(("policy", invocation.tool_name)) or True,
    )

    result = runner.run(invocation=_invocation(context), context=context, attempt_id="attempt_001", task_id="sample")

    assert events == [("policy", "sample_check"), ("handler", "attempt_001")]
    assert result.attempt_id == "attempt_001"
    assert result.task_id == "sample"
    assert result.action_id == "sample_check"
    receipt = json.loads((context.workspace / "audit/action_results/attempt_001.json").read_text(encoding="utf-8"))
    assert receipt["invocation_hash"] == _invocation(context).digest()
    trace = (context.workspace / "audit/agent_trace.jsonl").read_text(encoding="utf-8")
    assert trace.count('"attempt_id":"attempt_001"') == 2


def test_runner_denies_before_handler(tmp_path):
    context = _context(tmp_path)
    called = []
    handlers = HandlerRegistry()
    handlers.register("sample_check", lambda *_: called.append(True) or ActionResult(status="done"))
    runner = ActionRunner(handlers=handlers, policy_check=lambda *_: False)

    with pytest.raises(PermissionError, match="denied by policy"):
        runner.run(invocation=_invocation(context), context=context, attempt_id="attempt_001")

    assert called == []
    assert not (context.workspace / "audit/action_results/attempt_001.json").exists()


@pytest.mark.parametrize(
    ("project", "version_id"),
    [("/another/project", "v1"), (None, "another_version")],
)
def test_runner_rejects_invocation_subject_mismatch_before_handler(
    tmp_path, project, version_id
):
    context = _context(tmp_path)
    called = []
    handlers = HandlerRegistry()
    handlers.register("sample_check", lambda *_: called.append(True) or ActionResult(status="done"))
    runner = ActionRunner(handlers=handlers, policy_check=lambda *_: True)
    invocation = ActionInvocation(
        tool_name="sample_check",
        params={},
        project=project or str(context.project_dir),
        version_id=version_id,
    )

    with pytest.raises(ValueError, match="invocation subject does not match VersionContext"):
        runner.run(invocation=invocation, context=context, attempt_id="attempt_001")

    assert called == []
    assert not (context.workspace / "audit/action_results/attempt_001.json").exists()


@pytest.mark.parametrize("malformed_decision", [None, object()])
def test_runner_fails_closed_for_malformed_policy_decision(tmp_path, malformed_decision):
    context = _context(tmp_path)
    called = []
    handlers = HandlerRegistry()
    handlers.register("sample_check", lambda *_: called.append(True) or ActionResult(status="done"))
    runner = ActionRunner(handlers=handlers, policy_check=lambda *_: malformed_decision)

    with pytest.raises(PermissionError, match="denied by policy"):
        runner.run(invocation=_invocation(context), context=context, attempt_id="attempt_001")

    assert called == []
    assert not (context.workspace / "audit/action_results/attempt_001.json").exists()


def test_missing_handler_fails_closed_without_shell_fallback(tmp_path):
    context = _context(tmp_path)
    runner = ActionRunner(handlers=HandlerRegistry(), policy_check=lambda *_: True)

    with pytest.raises(ValueError, match="no registered action handler"):
        runner.run(invocation=_invocation(context), context=context, attempt_id="attempt_001")


def test_attempt_receipt_is_create_once(tmp_path):
    context = _context(tmp_path)
    handlers = HandlerRegistry()
    calls = []
    handlers.register("sample_check", lambda *_: calls.append(True) or ActionResult(status="done"))
    runner = ActionRunner(handlers=handlers, policy_check=lambda *_: True)
    runner.run(invocation=_invocation(context), context=context, attempt_id="attempt_001")

    with pytest.raises(DuplicateActionResultError):
        runner.run(invocation=_invocation(context), context=context, attempt_id="attempt_001")

    assert calls == [True]


@pytest.mark.parametrize(
    ("failure_code", "expected_action"),
    [
        ("advisor_required", "advisor"),
        ("sql_approval_required", "approval"),
        ("external_outcome_unknown", "reconciliation"),
    ],
)
def test_runner_canonicalizes_failure_code_required_action(
    tmp_path, failure_code, expected_action
):
    context = _context(tmp_path)
    handlers = HandlerRegistry()
    handlers.register(
        "sample_check",
        lambda *_: ActionResult(
            status="failed", failure_code=failure_code, next_required_action="none"
        ),
    )
    runner = ActionRunner(handlers=handlers, policy_check=lambda *_: True)

    result = runner.run(
        invocation=_invocation(context), context=context, attempt_id="attempt_required"
    )

    assert result.next_required_action == expected_action
    receipt = json.loads(
        (context.workspace / "audit/action_results/attempt_required.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["next_required_action"] == expected_action
