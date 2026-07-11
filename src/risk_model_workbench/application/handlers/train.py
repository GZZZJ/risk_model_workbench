"""Model-training application handler shared by CLI and Agent Runtime."""

from __future__ import annotations

from risk_model_workbench.application.context import VersionContext
from risk_model_workbench.harness.invocation import ActionInvocation
from risk_model_workbench.harness.runtime import ActionResult, detached_action_attempt
from risk_model_workbench.modeling.experiment import run_experiment
from risk_model_workbench.state import load_run_state


def run_train(invocation: ActionInvocation, context: VersionContext, attempt_id: str) -> ActionResult:
    del attempt_id
    params = invocation.canonical_payload()["params"]
    assert isinstance(params, dict)
    with detached_action_attempt():
        run_experiment(context, params)
    payload = dict(load_run_state(context.workspace)["stages"]["train_baseline"]["last_result"])
    allowed = set(ActionResult.__dataclass_fields__)
    return ActionResult(**{key: value for key, value in payload.items() if key in allowed})


__all__ = ["run_train"]
