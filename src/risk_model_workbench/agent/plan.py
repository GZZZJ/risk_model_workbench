"""Bind request execution plans into version-scoped Agent plans."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from risk_model_workbench.harness.actions import get_action_spec
from risk_model_workbench.harness.tools import ToolSpec, get_tool_spec, list_tool_specs, tool_to_dict


AGENT_PLAN_VERSION = 1


def agent_plan_path(workspace: str | Path) -> Path:
    return Path(workspace) / "agent_plan.yml"


def bind_agent_plan(execution_plan: dict[str, Any], *, project_dir: str | Path, version_id: str) -> dict[str, Any]:
    tasks = []
    for raw_task in execution_plan.get("tasks", []) or []:
        task = deepcopy(raw_task)
        args = _bind_args(list((task.get("command") or {}).get("args") or []), version_id=version_id)
        task.setdefault("command", {})["executable"] = "rmw"
        task["command"]["args"] = args
        tool = _infer_tool(task)
        action = get_action_spec(tool.action_id)
        task["action_id"] = tool.action_id
        task["tool_name"] = tool.name
        task["permission"] = tool.permission
        task["requires_approval"] = bool(tool.requires_approval)
        task["allowed_for_auditor"] = bool(tool.allowed_for_auditor)
        task["expected_inputs"] = list(action.inputs)
        task["expected_outputs"] = list(action.outputs or task.get("outputs") or [])
        task["failure_codes"] = list(action.failure_codes)
        tasks.append(task)

    return {
        "version": AGENT_PLAN_VERSION,
        "plan_id": f"{execution_plan.get('request_id', 'request')}_agent_plan",
        "source_plan_id": execution_plan.get("plan_id", ""),
        "request_id": execution_plan.get("request_id", ""),
        "request_path": execution_plan.get("request_path", ""),
        "project": str(project_dir),
        "workflow": execution_plan.get("workflow", ""),
        "version_id": version_id,
        "run_id_placeholder": "",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "tasks": tasks,
        "source_plan": execution_plan,
    }


def save_agent_plan(workspace: str | Path, agent_plan: dict[str, Any]) -> Path:
    path = agent_plan_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(agent_plan, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def load_agent_plan(workspace: str | Path) -> dict[str, Any]:
    path = agent_plan_path(workspace)
    if not path.exists():
        raise FileNotFoundError(f"agent_plan.yml not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def agent_tool_schema() -> list[dict[str, Any]]:
    rows = []
    for tool in list_tool_specs():
        action = get_action_spec(tool.action_id)
        item = tool_to_dict(tool)
        item["command_template"] = item.pop("command")
        item["inputs"] = list(action.inputs)
        item["outputs"] = list(action.outputs)
        item["failure_codes"] = list(action.failure_codes)
        rows.append(item)
    return rows


def _bind_args(args: list[str], *, version_id: str) -> list[str]:
    bound: list[str] = []
    index = 0
    saw_workspace_arg = False
    while index < len(args):
        token = args[index]
        if token == "--run-id":
            index += 2
            continue
        if token == "--version-id":
            saw_workspace_arg = True
            bound.extend([token, version_id])
            index += 2
            continue
        if token == "<run_id>":
            index += 1
            continue
        bound.append(token)
        index += 1
    if not saw_workspace_arg:
        bound.extend(["--version-id", version_id])
    return bound


def _infer_tool(task: dict[str, Any]) -> ToolSpec:
    args = list((task.get("command") or {}).get("args") or [])
    command = args[:2]
    task_type = str(task.get("type") or "")
    task_id = str(task.get("task_id") or "")
    if command[:2] == ["sample", "check"] or task_type == "sample_check":
        return get_tool_spec("sample_check")
    if command[:2] == ["feature", "metadata"] or task_id == "feature_metadata":
        return get_tool_spec("feature_metadata")
    if command[:2] == ["feature", "prescreen"] or task_id == "feature_prescreen":
        return get_tool_spec("feature_prescreen_pull" if "--sql-approved" in args else "feature_prescreen_dry_run")
    if args and args[0] == "build-wide-sql" or task_id == "build_wide_sql":
        return get_tool_spec("build_wide_sql_execute" if "--execute" in args or "--sql-approved" in args else "build_wide_sql")
    if command[:2] == ["feature", "refine"] or task_id == "feature_refine":
        return get_tool_spec("feature_refine_pull" if "--sql-approved" in args else "feature_refine_dry_run")
    if args and args[0] == "train" or task_type == "train":
        return get_tool_spec("train_baseline")
    if args and args[0] == "evaluate" or task_type == "evaluate":
        return get_tool_spec("evaluate")
    if args and args[0] == "compare" or task_type == "compare":
        return get_tool_spec("compare")
    if args and args[0] == "report" or task_type == "report":
        return get_tool_spec("report")
    if args[:2] == ["version", "audit"]:
        return get_tool_spec("run_audit")
    raise KeyError(f"cannot infer agent tool for task: {task_id or task_type}")
