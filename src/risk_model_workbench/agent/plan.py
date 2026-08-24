"""Bind request execution plans into version-scoped Agent plans."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from risk_model_workbench.agent.context_pack import context_pack_capabilities
from risk_model_workbench.harness.actions import get_action_spec
from risk_model_workbench.harness.invocation import ActionInvocation, canonical_json
from risk_model_workbench.harness.tools import (
    TOOL_REGISTRY,
    ToolSpec,
    get_tool_spec,
    list_tool_specs,
    registry_digest,
    tool_to_dict,
)


AGENT_PLAN_VERSION = 2


def agent_plan_path(workspace: str | Path) -> Path:
    return Path(workspace) / "agent_plan.yml"


def bind_agent_plan(execution_plan: dict[str, Any], *, project_dir: str | Path, version_id: str) -> dict[str, Any]:
    tasks = []
    for raw_task in execution_plan.get("tasks", []) or []:
        task = deepcopy(raw_task)
        args = _bind_args(list((task.get("command") or {}).get("args") or []), version_id=version_id)
        tool = _infer_tool(task)
        action = get_action_spec(tool.action_id)
        invocation = ActionInvocation(
            tool_name=tool.name,
            params=_params_from_args(tool, args),
            project=str(project_dir),
            version_id=version_id,
        )
        rendered = tool.render_argv(invocation)
        task["command"] = {"executable": "rmw", "args": rendered}
        task["invocation"] = invocation.canonical_payload()
        task["invocation_hash"] = invocation.digest()
        task["derived_metadata"] = _derived_metadata(tool)
        for copied_authority in ("permission", "requires_approval", "allowed_for_auditor"):
            task.pop(copied_authority, None)
        task["action_id"] = tool.action_id
        task["tool_name"] = tool.name
        task["expected_inputs"] = list(action.inputs)
        task["expected_outputs"] = list(action.outputs or task.get("outputs") or [])
        task["failure_codes"] = list(action.failure_codes)
        tasks.append(task)

    plan = {
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
        "registry_digest": registry_digest(),
    }
    plan["plan_hash"] = canonical_plan_hash(plan)
    return plan


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
        item.pop("command", None)
        item["inputs"] = list(action.inputs)
        item["outputs"] = list(action.outputs)
        item["failure_codes"] = list(action.failure_codes)
        rows.append(item)
    return rows


def agent_capabilities() -> dict[str, Any]:
    """Machine-readable embedded Agent contract, independent of CLI help text."""
    return {
        "version": 1,
        "context_pack": context_pack_capabilities(),
        "tools": agent_tool_schema(),
    }


def canonical_plan_hash(plan: dict[str, Any]) -> str:
    payload = deepcopy(plan)
    payload.pop("plan_hash", None)
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def validate_agent_plan(
    plan: dict[str, Any],
    registry: dict[str, ToolSpec] | None = None,
    *,
    expected_project: str | Path | None = None,
    expected_version_id: str | None = None,
) -> list[str]:
    source = TOOL_REGISTRY if registry is None else registry
    if int(plan.get("version") or 1) < 2:
        return []
    errors: list[str] = []
    project = str(plan.get("project") or "")
    version_id = str(plan.get("version_id") or "")
    if expected_project is not None and Path(project).resolve() != Path(expected_project).resolve():
        errors.append(f"runtime_project_mismatch:{project}:{Path(expected_project).resolve()}")
    if expected_version_id is not None and version_id != expected_version_id:
        errors.append(f"runtime_version_mismatch:{version_id}:{expected_version_id}")
    tasks = list(plan.get("tasks") or [])
    task_ids = [str(task.get("task_id") or "") for task in tasks]
    duplicates = sorted({task_id for task_id in task_ids if task_id and task_ids.count(task_id) > 1})
    errors.extend(f"duplicate_task_id:{task_id}" for task_id in duplicates)
    known = set(task_ids)
    for task in tasks:
        task_id = str(task.get("task_id") or "")
        for dependency in task.get("depends_on") or []:
            if str(dependency) not in known:
                errors.append(f"missing_dependency:{task_id}:{dependency}")
    if _has_cycle(tasks):
        errors.append("cyclic_dependency")

    expected_registry_digest = registry_digest(source)
    if plan.get("registry_digest") != expected_registry_digest:
        errors.append(f"registry_digest_drift:{plan.get('registry_digest', '')}:{expected_registry_digest}")

    for task in tasks:
        task_id = str(task.get("task_id") or "")
        for copied_authority in ("permission", "requires_approval", "allowed_for_auditor"):
            if copied_authority in task:
                errors.append(f"copied_authority_present:{task_id}:{copied_authority}")
        raw_invocation = task.get("invocation")
        if not isinstance(raw_invocation, dict):
            errors.append(f"invalid_invocation:{task_id}")
            continue
        if set(raw_invocation) != {"tool_name", "params", "project", "version_id"}:
            errors.append(f"invalid_invocation_shape:{task_id}")
        try:
            invocation = ActionInvocation.from_dict(raw_invocation)
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid_invocation:{task_id}:{exc}")
            continue
        if invocation.project != project:
            errors.append(f"project_mismatch:{task_id}")
        if invocation.version_id != version_id:
            errors.append(f"version_mismatch:{task_id}")
        spec = source.get(invocation.tool_name)
        if spec is None:
            errors.append(f"unknown_tool:{task_id}:{invocation.tool_name}")
            continue
        if task.get("action_id") not in (None, "", spec.action_id):
            errors.append(f"copied_metadata_drift:{task_id}:action_id")
        if task.get("tool_name") not in (None, "", spec.name):
            errors.append(f"copied_metadata_drift:{task_id}:tool_name")
        errors.extend(_validate_params(task_id, invocation, spec))
        if task.get("invocation_hash") != invocation.digest():
            errors.append(f"invocation_hash_drift:{task_id}")
        rendered = spec.render_argv(invocation)
        copied_args = list((task.get("command") or {}).get("args") or [])
        if copied_args != rendered:
            errors.append(f"rendered_argv_drift:{task_id}")
        if task.get("derived_metadata") != _derived_metadata(spec):
            errors.append(f"derived_metadata_drift:{task_id}")
        for flag in spec.blocked_flags:
            if flag in rendered or _contains_scalar(invocation.canonical_payload()["params"], flag):
                errors.append(f"blocked_flag:{task_id}:{flag}")

    expected_plan_hash = canonical_plan_hash(plan)
    if plan.get("plan_hash") != expected_plan_hash:
        errors.append(f"plan_hash_drift:{plan.get('plan_hash', '')}:{expected_plan_hash}")
    return errors


def rebind_agent_plan(
    plan: dict[str, Any],
    registry: dict[str, ToolSpec] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = TOOL_REGISTRY if registry is None else registry
    rebound = deepcopy(plan)
    if int(rebound.get("version") or 1) < 2:
        raise ValueError("v1 plans require explicit migration before rebind")
    source_errors = _validate_rebind_source(rebound, source)
    if source_errors:
        raise ValueError("unsafe rebind source: " + "; ".join(source_errors))
    before_registry = str(rebound.get("registry_digest") or "")
    before_plan = str(rebound.get("plan_hash") or "")
    for task in rebound.get("tasks") or []:
        invocation = ActionInvocation.from_dict(task.get("invocation") or {})
        try:
            spec = source[invocation.tool_name]
        except KeyError as exc:
            raise ValueError(f"cannot rebind unknown tool: {invocation.tool_name}") from exc
        task["command"] = {"executable": "rmw", "args": spec.render_argv(invocation)}
        task["derived_metadata"] = _derived_metadata(spec)
        task["action_id"] = spec.action_id
        task["tool_name"] = spec.name
        for copied_authority in ("permission", "requires_approval", "allowed_for_auditor"):
            task.pop(copied_authority, None)
    rebound["registry_digest"] = registry_digest(source)
    rebound["plan_hash"] = canonical_plan_hash(rebound)
    rebound_errors = validate_agent_plan(rebound, source)
    if rebound_errors:
        raise ValueError("rebound plan is invalid: " + "; ".join(rebound_errors))
    changes = _field_changes(plan, rebound)
    preview = {
        "changed": bool(changes),
        "changes": changes,
        "registry_digest_before": before_registry,
        "registry_digest_after": rebound["registry_digest"],
        "plan_hash_before": before_plan,
        "plan_hash_after": rebound["plan_hash"],
    }
    return rebound, preview


def invocation_for_task(task: dict[str, Any], plan: dict[str, Any]) -> ActionInvocation:
    if int(plan.get("version") or 1) >= 2:
        return ActionInvocation.from_dict(task.get("invocation") or {})
    args = list((task.get("command") or {}).get("args") or [])
    tool = get_tool_spec(str(task.get("tool_name") or _infer_tool(task).name))
    return ActionInvocation(
        tool_name=tool.name,
        params=_params_from_args(tool, args),
        project=str(plan.get("project") or _arg_value(args, "--project") or ""),
        version_id=str(plan.get("version_id") or _arg_value(args, "--version-id") or _arg_value(args, "--run-id") or ""),
    )


def rendered_argv_for_task(
    task: dict[str, Any],
    plan: dict[str, Any],
    registry: dict[str, ToolSpec] | None = None,
) -> list[str]:
    source = TOOL_REGISTRY if registry is None else registry
    invocation = invocation_for_task(task, plan)
    try:
        return source[invocation.tool_name].render_argv(invocation)
    except KeyError as exc:
        raise ValueError(f"unknown tool: {invocation.tool_name}") from exc


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
        if "--sql-approved" in args:
            return get_tool_spec("feature_prescreen_execute")
        return get_tool_spec("feature_prescreen_prepare" if "--dry-run-sql" in args else "feature_prescreen_local")
    if args and args[0] == "build-wide-sql" or task_id == "build_wide_sql":
        if "--execute" in args or "--sql-approved" in args:
            return get_tool_spec("build_wide_sql_execute")
        return get_tool_spec("build_wide_sql_local" if task_id == "build_wide_sql" else "build_wide_sql_prepare")
    if command[:2] == ["feature", "refine"] or task_id == "feature_refine":
        if "--sql-approved" in args:
            return get_tool_spec("feature_refine_execute")
        return get_tool_spec("feature_refine_prepare" if "--dry-run-sql" in args else "feature_refine_local")
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


def _params_from_args(tool: ToolSpec, args: list[str]) -> dict[str, object]:
    if tool.name == "train_baseline":
        return {"experiment": _arg_value(args, "--experiment") or ""}
    if tool.name == "compare":
        return {"champions": _arg_values(args, "--champion")}
    if tool.name == "workflow_validate":
        return {"workflow": _arg_value(args, "--workflow") or ""}
    return {}


def _arg_value(args: list[str], flag: str) -> str | None:
    try:
        return str(args[args.index(flag) + 1])
    except (ValueError, IndexError):
        return None


def _arg_values(args: list[str], flag: str) -> list[str]:
    return [str(args[index + 1]) for index, token in enumerate(args[:-1]) if token == flag]


def _derived_metadata(spec: ToolSpec) -> dict[str, object]:
    return {
        "action_id": spec.action_id,
        "permission": spec.permission,
        "requires_approval": spec.requires_approval,
        "allowed_for_auditor": spec.allowed_for_auditor,
        "execution_semantics": spec.execution_semantics,
        "approval_type": spec.approval_type,
    }


def _validate_params(task_id: str, invocation: ActionInvocation, spec: ToolSpec) -> list[str]:
    params = invocation.canonical_payload()["params"]
    assert isinstance(params, dict)
    schema = spec.params_schema
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = schema.get("required") if isinstance(schema.get("required"), list) else []
    errors = [f"missing_param:{task_id}:{name}" for name in required if not params.get(name)]
    if schema.get("additionalProperties") is False:
        errors.extend(f"unknown_param:{task_id}:{name}" for name in params if name not in properties)
    for name, value in params.items():
        definition = properties.get(name) if isinstance(properties, dict) else None
        expected = definition.get("type") if isinstance(definition, dict) else None
        if expected == "string" and not isinstance(value, str):
            errors.append(f"invalid_param_type:{task_id}:{name}")
        if expected == "array" and not isinstance(value, list):
            errors.append(f"invalid_param_type:{task_id}:{name}")
    return errors


def _contains_scalar(value: object, expected: str) -> bool:
    if isinstance(value, dict):
        return any(_contains_scalar(item, expected) for item in value.values())
    if isinstance(value, list):
        return any(_contains_scalar(item, expected) for item in value)
    return value == expected


def _has_cycle(tasks: list[dict[str, Any]]) -> bool:
    graph = {str(task.get("task_id") or ""): [str(dep) for dep in task.get("depends_on") or []] for task in tasks}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for dependency in graph.get(node, []):
            if dependency in graph and visit(dependency):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in graph)


def _validate_rebind_source(plan: dict[str, Any], registry: dict[str, ToolSpec]) -> list[str]:
    errors: list[str] = []
    if plan.get("plan_hash") != canonical_plan_hash(plan):
        errors.append("plan_hash_drift")
    project = str(plan.get("project") or "")
    version_id = str(plan.get("version_id") or "")
    tasks = list(plan.get("tasks") or [])
    task_ids = [str(task.get("task_id") or "") for task in tasks]
    if any(not task_id for task_id in task_ids):
        errors.append("missing_task_id")
    if len(task_ids) != len(set(task_ids)):
        errors.append("duplicate_task_id")
    known = set(task_ids)
    if any(str(dep) not in known for task in tasks for dep in task.get("depends_on") or []):
        errors.append("missing_dependency")
    if _has_cycle(tasks):
        errors.append("cyclic_dependency")
    for task in tasks:
        task_id = str(task.get("task_id") or "")
        raw = task.get("invocation")
        if not isinstance(raw, dict) or set(raw) != {"tool_name", "params", "project", "version_id"}:
            errors.append(f"invalid_invocation:{task_id}")
            continue
        try:
            invocation = ActionInvocation.from_dict(raw)
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid_invocation:{task_id}:{exc}")
            continue
        if task.get("invocation_hash") != invocation.digest():
            errors.append(f"invocation_hash_drift:{task_id}")
        if invocation.project != project:
            errors.append(f"project_mismatch:{task_id}")
        if invocation.version_id != version_id:
            errors.append(f"version_mismatch:{task_id}")
        spec = registry.get(invocation.tool_name)
        if spec is None:
            errors.append(f"unknown_tool:{task_id}:{invocation.tool_name}")
            continue
        errors.extend(_validate_params(task_id, invocation, spec))
        for flag in spec.blocked_flags:
            if _contains_scalar(invocation.canonical_payload()["params"], flag):
                errors.append(f"blocked_flag:{task_id}:{flag}")
    return errors


def _field_changes(before: object, after: object, path: str = "") -> list[dict[str, object]]:
    changes: list[dict[str, object]] = []
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before) | set(after)):
            child = f"{path}.{key}" if path else key
            if key not in before:
                changes.append({"path": child, "before": None, "after": after[key]})
            elif key not in after:
                changes.append({"path": child, "before": before[key], "after": None})
            else:
                changes.extend(_field_changes(before[key], after[key], child))
        return changes
    if before != after:
        changes.append({"path": path, "before": before, "after": after})
    return changes
