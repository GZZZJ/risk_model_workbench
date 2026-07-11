"""Thin MCP-facing adapter for typed RMW actions.

This module is deliberately transport-neutral. An MCP SDK registration layer
may expose these methods, but all policy and execution remain owned by the
injected :class:`ActionRunner`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from risk_model_workbench.application.action_runner import ActionRunner
from risk_model_workbench.application.context import VersionContext
from risk_model_workbench.harness.invocation import ActionInvocation
from risk_model_workbench.harness.tools import ToolSpec, get_tool_spec, list_tool_specs


class MCPActionAdapter:
    """Parse typed MCP inputs and delegate exactly once to ActionRunner."""

    def __init__(
        self,
        *,
        runner: ActionRunner,
        tool_specs: Sequence[ToolSpec] | None = None,
    ) -> None:
        self._runner = runner
        selected = tool_specs or tuple(
            spec
            for spec in list_tool_specs()
            if spec.permission in {"writes_run", "dp_sql_pull"}
        )
        self._tools = {spec.name: spec for spec in selected}

    def capabilities(self) -> list[dict[str, Any]]:
        """Return MCP-safe tool metadata without CLI command templates."""
        return [
            {
                "name": spec.name,
                "action_id": spec.action_id,
                "description": spec.description,
                "permission": spec.permission,
                "requires_approval": spec.requires_approval,
                "execution_semantics": spec.execution_semantics,
                "input_schema": deepcopy(spec.params_schema),
            }
            for spec in sorted(self._tools.values(), key=lambda item: item.name)
        ]

    def call_action(
        self,
        *,
        tool_name: str,
        params: Mapping[str, object],
        project: str | Path,
        version_id: str,
        attempt_id: str,
        task_id: str = "",
    ) -> dict[str, object]:
        """Validate one typed call and return the semantic ActionResult payload."""
        try:
            spec = self._tools[tool_name]
        except KeyError as exc:
            # Preserve the registry's stable unknown-tool distinction.
            try:
                get_tool_spec(tool_name)
            except KeyError:
                raise ValueError(f"unknown MCP action tool: {tool_name}") from exc
            raise ValueError(f"tool is not exposed by this MCP adapter: {tool_name}") from exc
        normalized = dict(params)
        errors = _validate_params(normalized, spec.params_schema)
        if errors:
            raise ValueError("invalid MCP action params: " + ",".join(errors))

        project_dir = Path(project).resolve()
        context = _context(project_dir, version_id)
        invocation = ActionInvocation(
            tool_name=tool_name,
            params=normalized,
            project=str(project_dir),
            version_id=version_id,
        )
        result = self._runner.run(
            invocation=invocation,
            context=context,
            attempt_id=attempt_id,
            task_id=task_id,
        )
        return result.to_dict()


def _context(project_dir: Path, version_id: str) -> VersionContext:
    version_workspace = project_dir / "versions" / version_id
    legacy_workspace = project_dir / "runs" / version_id
    workspace = version_workspace if version_workspace.exists() or not legacy_workspace.exists() else legacy_workspace
    state_path = workspace / "version_state.yml"
    if not state_path.exists() and (workspace / "run_state.yml").exists():
        state_path = workspace / "run_state.yml"
    return VersionContext(
        project_dir=project_dir,
        version_id=version_id,
        workspace=workspace,
        runtime_config_dir=workspace / "configs_runtime",
        manifest_path=workspace / "audit" / "artifact_manifest.json",
        version_state_path=state_path,
    )


def _validate_params(params: dict[str, object], schema: dict[str, object]) -> list[str]:
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = schema.get("required") if isinstance(schema.get("required"), list) else []
    errors = [f"missing_param:{name}" for name in required if name not in params or params[name] in (None, "")]
    if schema.get("additionalProperties") is False:
        errors.extend(f"unknown_param:{name}" for name in params if name not in properties)
    for name, value in params.items():
        definition = properties.get(name) if isinstance(properties, dict) else None
        if not isinstance(definition, dict):
            continue
        expected = definition.get("type")
        if expected == "string" and not isinstance(value, str):
            errors.append(f"invalid_param_type:{name}:string")
        elif expected == "boolean" and not isinstance(value, bool):
            errors.append(f"invalid_param_type:{name}:boolean")
        elif expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
            errors.append(f"invalid_param_type:{name}:integer")
        elif expected == "array":
            if not isinstance(value, (list, tuple)):
                errors.append(f"invalid_param_type:{name}:array")
            else:
                item_type = (definition.get("items") or {}).get("type") if isinstance(definition.get("items"), dict) else None
                if item_type == "string" and any(not isinstance(item, str) for item in value):
                    errors.append(f"invalid_param_items:{name}:string")
        if isinstance(value, str) and definition.get("minLength") and len(value) < int(definition["minLength"]):
            errors.append(f"invalid_param_length:{name}")
    return errors


__all__ = ["MCPActionAdapter"]
