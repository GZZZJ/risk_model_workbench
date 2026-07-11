"""In-process, policy-gated execution for typed workbench actions."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Callable

from risk_model_workbench.agent.trace import append_trace
from risk_model_workbench.application.context import VersionContext
from risk_model_workbench.harness.invocation import ActionInvocation
from risk_model_workbench.harness.errors import DuplicateActionResultError
from risk_model_workbench.harness.runtime import (
    ActionResult,
    action_result_path,
    canonicalize_required_action,
    write_action_result,
)
from risk_model_workbench.harness.tools import action_id_for_tool


Handler = Callable[[ActionInvocation, VersionContext, str], ActionResult]
PolicyCheck = Callable[[ActionInvocation, VersionContext], object]


class HandlerRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, Handler] = {}

    def register(self, action_id: str, handler: Handler) -> None:
        if not action_id or action_id in self._handlers:
            raise ValueError(f"action handler already registered or invalid: {action_id}")
        self._handlers[action_id] = handler

    def resolve(self, action_id: str) -> Handler:
        try:
            return self._handlers[action_id]
        except KeyError as exc:
            raise ValueError(f"no registered action handler: {action_id}") from exc

    def action_ids(self) -> frozenset[str]:
        """Return an immutable registry snapshot for workflow coverage checks."""
        return frozenset(self._handlers)


class ActionRunner:
    """Run a registered Python handler; never fall back to shell execution."""

    def __init__(self, *, handlers: HandlerRegistry, policy_check: PolicyCheck) -> None:
        self._handlers = handlers
        self._policy_check = policy_check

    def run(
        self,
        *,
        invocation: ActionInvocation,
        context: VersionContext,
        attempt_id: str,
        task_id: str = "",
    ) -> ActionResult:
        if invocation.project != str(context.project_dir) or invocation.version_id != context.version_id:
            raise ValueError("invocation subject does not match VersionContext")
        action_id = action_id_for_tool(invocation.tool_name)
        handler = self._handlers.resolve(action_id)
        if action_result_path(context.workspace, attempt_id).exists():
            raise DuplicateActionResultError(f"ActionResult already exists for attempt {attempt_id}")
        decision = self._policy_check(invocation, context)
        explicitly_allowed = decision is True or getattr(decision, "allowed", None) is True
        if not explicitly_allowed:
            raise PermissionError(f"action denied by policy: {action_id}")

        invocation_hash = invocation.digest()
        correlated_task_id = task_id or action_id
        append_trace(
            context.workspace,
            "action",
            {
                "summary": f"ActionRunner executing {action_id}.",
                "attempt_id": attempt_id,
                "task_id": correlated_task_id,
                "action_id": action_id,
                "invocation_hash": invocation_hash,
            },
        )
        result = handler(invocation, context, attempt_id)
        if not isinstance(result, ActionResult):
            raise TypeError("action handler must return ActionResult")
        correlated = replace(
            canonicalize_required_action(result),
            attempt_id=attempt_id,
            task_id=correlated_task_id,
            action_id=action_id,
            invocation_hash=invocation_hash,
            project=str(context.project_dir),
            version_id=context.version_id,
            created_at=result.created_at or datetime.now().isoformat(timespec="seconds"),
        )
        write_action_result(context.workspace, correlated)
        append_trace(
            context.workspace,
            "result",
            {
                "summary": f"ActionRunner completed {action_id} with {correlated.status}.",
                "attempt_id": attempt_id,
                "task_id": correlated_task_id,
                "action_id": action_id,
                "invocation_hash": invocation_hash,
                "status": correlated.status,
            },
        )
        return correlated
