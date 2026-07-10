"""Permission policy for the semi-autonomous RMW Agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from risk_model_workbench.agent.approvals import (
    approval_id_for,
    build_approval_subject,
    command_hash,
    ensure_approval_request,
    ensure_subject_approval,
    is_subject_approval_consumed,
    is_command_approved,
)
from risk_model_workbench.harness.invocation import ActionInvocation
from risk_model_workbench.harness.tools import TOOL_REGISTRY, ToolSpec


SAFE_PERMISSIONS = {"read_only", "writes_run"}
RISKY_PERMISSIONS = {"dp_sql_pull", "external_data"}
BLOCKED_FLAGS = {"--force": "force_flag_blocked", "--skip-split-check": "skip_split_check_blocked"}


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    status: str
    reason: str
    approval_id: str = ""
    command_hash: str = ""

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "allowed": self.allowed,
            "status": self.status,
            "reason": self.reason,
            "approval_id": self.approval_id,
            "command_hash": self.command_hash,
        }


def evaluate_task_policy(
    invocation_or_task: ActionInvocation | dict[str, Any],
    workspace: str | Path,
    *,
    task_id: str = "",
    registry: dict[str, ToolSpec] | None = None,
    subject_bound: bool = False,
) -> PolicyDecision:
    source = TOOL_REGISTRY if registry is None else registry
    raw_args = (
        [str(item) for item in (invocation_or_task.get("command") or {}).get("args") or []]
        if isinstance(invocation_or_task, dict)
        else []
    )
    for flag, reason in BLOCKED_FLAGS.items():
        if flag in raw_args:
            return PolicyDecision(False, "blocked", reason, command_hash=command_hash(raw_args))
    invocation = (
        invocation_or_task
        if isinstance(invocation_or_task, ActionInvocation)
        else _legacy_invocation(invocation_or_task, source)
    )
    if invocation is None or invocation.tool_name not in source:
        return PolicyDecision(False, "blocked", "unregistered_tool")
    spec = source[invocation.tool_name]
    params = invocation.canonical_payload()["params"]
    assert isinstance(params, dict)
    properties = spec.params_schema.get("properties") if isinstance(spec.params_schema.get("properties"), dict) else {}
    required = spec.params_schema.get("required") if isinstance(spec.params_schema.get("required"), list) else []
    if any(not params.get(name) for name in required) or (
        spec.params_schema.get("additionalProperties") is False
        and any(name not in properties for name in params)
    ):
        return PolicyDecision(False, "blocked", "invalid_invocation_params")
    for flag, reason in BLOCKED_FLAGS.items():
        if _contains_scalar(params, flag):
            return PolicyDecision(False, "blocked", reason)
    args = spec.render_argv(invocation)
    digest = command_hash(args)
    for flag, reason in BLOCKED_FLAGS.items():
        if flag in args:
            return PolicyDecision(False, "blocked", reason, command_hash=digest)

    permission = spec.permission
    requires_approval = spec.approval_type != "none"
    if permission in SAFE_PERMISSIONS and not requires_approval:
        return PolicyDecision(True, "allowed", "safe_permission", command_hash=digest)

    if permission == "dp_sql_pull" or requires_approval:
        if subject_bound:
            try:
                subject = build_approval_subject(
                    workspace,
                    project=invocation.project,
                    version_id=invocation.version_id,
                    task_id=task_id,
                    invocation_hash=invocation.digest(),
                    operation_id=spec.name,
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                return PolicyDecision(False, "blocked", f"sql_evidence_invalid:{exc}", command_hash=digest)
            if is_subject_approval_consumed(workspace, subject):
                return PolicyDecision(True, "allowed", "subject_approval_consumed", command_hash=digest)
            request = ensure_subject_approval(workspace, subject, reason="approval_required")
            return PolicyDecision(
                False,
                "waiting_for_approval",
                "approval_required",
                approval_id=str(request.get("approval_id") or ""),
                command_hash=digest,
            )
        approval_id = approval_id_for(args)
        if is_command_approved(workspace, args):
            return PolicyDecision(True, "allowed", "approved", approval_id=approval_id, command_hash=digest)
        request = ensure_approval_request(
            workspace,
            {
                "task_id": task_id or (invocation_or_task.get("task_id", "") if isinstance(invocation_or_task, dict) else ""),
                "tool_name": spec.name,
                "permission": permission,
                "command": {"executable": "rmw", "args": args},
            },
            reason="approval_required",
        )
        return PolicyDecision(
            False,
            "waiting_for_approval",
            "approval_required",
            approval_id=str(request.get("approval_id") or approval_id),
            command_hash=digest,
        )

    if permission in RISKY_PERMISSIONS:
        return PolicyDecision(False, "blocked", f"{permission}_blocked", command_hash=digest)

    return PolicyDecision(False, "blocked", "unknown_permission", command_hash=digest)


def _legacy_invocation(task: dict[str, Any], registry: dict[str, ToolSpec]) -> ActionInvocation | None:
    args = [str(item) for item in (task.get("command") or {}).get("args") or []]
    tool_name = str(task.get("tool_name") or "")
    if tool_name not in registry:
        tool_name = _infer_tool_name(args)
    if tool_name not in registry:
        return None
    params: dict[str, object] = {}
    if tool_name == "train_baseline":
        params["experiment"] = _arg_value(args, "--experiment") or ""
    elif tool_name == "compare":
        params["champions"] = [args[index + 1] for index, token in enumerate(args[:-1]) if token == "--champion"]
    elif tool_name == "workflow_validate":
        params["workflow"] = _arg_value(args, "--workflow") or ""
    return ActionInvocation(
        tool_name=tool_name,
        params=params,
        project=_arg_value(args, "--project") or "legacy_project",
        version_id=_arg_value(args, "--version-id") or _arg_value(args, "--run-id") or "legacy_version",
    )


def _infer_tool_name(args: list[str]) -> str:
    if args[:2] == ["version", "audit"] or args[:2] == ["run", "audit"]:
        return "run_audit"
    if args[:2] == ["sample", "check"]:
        return "sample_check"
    if args[:2] == ["feature", "metadata"]:
        return "feature_metadata"
    if args[:2] == ["feature", "prescreen"]:
        if "--sql-approved" in args:
            return "feature_prescreen_execute"
        return "feature_prescreen_prepare" if "--dry-run-sql" in args else "feature_prescreen_local"
    if args[:1] == ["build-wide-sql"]:
        return "build_wide_sql_execute" if "--execute" in args or "--sql-approved" in args else "build_wide_sql_local"
    if args[:2] == ["feature", "refine"]:
        if "--sql-approved" in args:
            return "feature_refine_execute"
        return "feature_refine_prepare" if "--dry-run-sql" in args else "feature_refine_local"
    if args[:1] == ["train"]:
        return "train_baseline"
    if args[:1] == ["evaluate"]:
        return "evaluate"
    if args[:1] == ["compare"]:
        return "compare"
    if args[:1] == ["report"]:
        return "report"
    return ""


def _arg_value(args: list[str], flag: str) -> str | None:
    try:
        return args[args.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def _contains_scalar(value: object, expected: str) -> bool:
    if isinstance(value, dict):
        return any(_contains_scalar(item, expected) for item in value.values())
    if isinstance(value, list):
        return any(_contains_scalar(item, expected) for item in value)
    return value == expected
