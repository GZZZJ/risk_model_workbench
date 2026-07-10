from pathlib import Path

from risk_model_workbench.agent.approvals import approve_request, load_approvals
from risk_model_workbench.agent.policy import evaluate_task_policy


def test_policy_allows_safe_permissions_and_blocks_risky_permissions(tmp_path):
    workspace = _workspace(tmp_path)

    read_only = _task(permission="read_only", args=["version", "audit", "--project", "p", "--version-id", "v"])
    writes_run = _task(permission="writes_run", args=["sample", "check", "--project", "p", "--version-id", "v"])
    dp_pull = _task(permission="dp_sql_pull", args=["feature", "prescreen", "--project", "p", "--version-id", "v", "--sql-approved"])
    external = _task(permission="external_data", args=["external", "pull"])

    assert evaluate_task_policy(read_only, workspace).allowed is True
    assert evaluate_task_policy(writes_run, workspace).allowed is True
    assert evaluate_task_policy(dp_pull, workspace).allowed is False
    assert evaluate_task_policy(dp_pull, workspace).status == "waiting_for_approval"
    assert evaluate_task_policy(external, workspace).allowed is False
    assert evaluate_task_policy(external, workspace).status == "blocked"


def test_policy_blocks_escape_hatches_and_approval_hash_mismatch(tmp_path):
    workspace = _workspace(tmp_path)
    forced = _task(permission="writes_run", args=["version", "init", "--project", "p", "--version-id", "v", "--force"])
    skipped = _task(permission="writes_run", args=["train", "--project", "p", "--version-id", "v", "--skip-split-check"])
    risky = _task(permission="dp_sql_pull", args=["feature", "refine", "--project", "p", "--version-id", "v", "--sql-approved"])

    assert evaluate_task_policy(forced, workspace).reason == "force_flag_blocked"
    assert evaluate_task_policy(skipped, workspace).reason == "skip_split_check_blocked"

    decision = evaluate_task_policy(risky, workspace)
    assert decision.approval_id
    approve_request(workspace, decision.approval_id, approved_by="pm", note="reviewed")
    assert evaluate_task_policy(risky, workspace).allowed is True

    changed = _task(permission="dp_sql_pull", args=["feature", "refine", "--project", "p", "--version-id", "v2", "--sql-approved"])
    changed_decision = evaluate_task_policy(changed, workspace)
    approvals = load_approvals(workspace)

    assert changed_decision.allowed is False
    assert changed_decision.approval_id != decision.approval_id
    assert approvals["approvals"][0]["command_hash"] != changed_decision.command_hash


def _task(*, permission: str, args: list[str]) -> dict:
    return {
        "task_id": "task1",
        "tool_name": "tool1",
        "permission": permission,
        "requires_approval": permission == "dp_sql_pull",
        "command": {"executable": "rmw", "args": args},
    }


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "version"
    (workspace / "audit").mkdir(parents=True)
    return workspace
