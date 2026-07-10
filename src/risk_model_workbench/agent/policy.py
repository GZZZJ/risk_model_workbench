"""Permission policy for the semi-autonomous RMW Agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from risk_model_workbench.agent.approvals import (
    approval_id_for,
    command_hash,
    ensure_approval_request,
    is_command_approved,
)


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


def evaluate_task_policy(task: dict[str, Any], workspace: str | Path) -> PolicyDecision:
    args = list((task.get("command") or {}).get("args") or [])
    digest = command_hash(args)
    for flag, reason in BLOCKED_FLAGS.items():
        if flag in args:
            return PolicyDecision(False, "blocked", reason, command_hash=digest)

    permission = str(task.get("permission") or "")
    requires_approval = bool(task.get("requires_approval"))
    if permission in SAFE_PERMISSIONS and not requires_approval:
        return PolicyDecision(True, "allowed", "safe_permission", command_hash=digest)

    if permission == "dp_sql_pull" or requires_approval:
        approval_id = approval_id_for(args)
        if is_command_approved(workspace, args):
            return PolicyDecision(True, "allowed", "approved", approval_id=approval_id, command_hash=digest)
        request = ensure_approval_request(workspace, task, reason="approval_required")
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
