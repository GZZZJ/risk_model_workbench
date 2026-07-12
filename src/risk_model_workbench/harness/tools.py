"""Permission-scoped, typed tool registry for rmw commands."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable

from risk_model_workbench.harness.actions import get_action_spec
from risk_model_workbench.harness.command_metadata import argv_template, display_template
from risk_model_workbench.harness.invocation import ActionInvocation


BLOCKED_TOOL_FLAGS = ("--force", "--skip-split-check")
EXECUTION_SEMANTICS = {"read_only", "idempotent_write", "non_idempotent_write", "external_unknown"}


def action_id_for_tool(tool_name: str) -> str:
    """Resolve a typed tool to its declared action without rendering a command."""
    try:
        return TOOL_REGISTRY[tool_name].action_id
    except KeyError as exc:
        raise ValueError(f"unknown typed tool: {tool_name}") from exc

def _default_params_schema(name: str) -> dict[str, object]:
    properties: dict[str, object] = {}
    required: list[str] = []
    if name == "train_baseline":
        properties["experiment"] = {"type": "string", "minLength": 1}
        for key in ["input_feather", "feature_list", "score_output", "input_dir", "config"]:
            properties[key] = {"type": "string", "minLength": 1}
        properties["plan_only"] = {"type": "boolean"}
        required.append("experiment")
    elif name == "compare":
        properties["champions"] = {"type": "array", "items": {"type": "string"}}
    elif name == "evaluate":
        properties["scores_feather"] = {"type": "string", "minLength": 1}
        properties["output_dir"] = {"type": "string", "minLength": 1}
    elif name == "report":
        properties["report_target"] = {"type": "string", "minLength": 1}
    elif name == "workflow_validate":
        properties["workflow"] = {"type": "string", "minLength": 1}
        required.append("workflow")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _typed_renderer(tool_name: str) -> Callable[[ActionInvocation], list[str]]:
    try:
        tokens = argv_template(tool_name)
    except KeyError as exc:
        raise ValueError(f"tool {tool_name} requires an explicit typed renderer") from exc

    def render(invocation: ActionInvocation) -> list[str]:
        if invocation.tool_name != tool_name:
            raise ValueError(f"invocation tool mismatch: expected {tool_name}, got {invocation.tool_name}")
        params = invocation.canonical_payload()["params"]
        assert isinstance(params, dict)
        values = {
            "{project}": invocation.project,
            "{version_id}": invocation.version_id,
            "{experiment}": str(params.get("experiment") or ""),
            "{workflow}": str(params.get("workflow") or ""),
        }
        rendered = [values.get(token, token) for token in tokens]
        if tool_name == "compare":
            for champion in params.get("champions") or []:
                rendered.extend(["--champion", str(champion)])
        elif tool_name == "evaluate":
            for name in ["scores_feather", "output_dir"]:
                if params.get(name):
                    rendered.extend(["--" + name.replace("_", "-"), str(params[name])])
        elif tool_name == "report" and params.get("report_target"):
            rendered.extend(["--report-target", str(params["report_target"])])
        elif tool_name == "train_baseline":
            for name in ["input_feather", "feature_list", "score_output", "input_dir", "config"]:
                if params.get(name):
                    rendered.extend(["--" + name.replace("_", "-"), str(params[name])])
            if params.get("plan_only") is True:
                rendered.append("--plan-only")
        return rendered

    return render


@dataclass(frozen=True)
class ToolSpec:
    name: str
    action_id: str
    command: str
    permission: str
    description: str
    requires_approval: bool = False
    allowed_for_auditor: bool = False
    params_schema: dict[str, object] = field(default_factory=dict)
    execution_semantics: str = ""
    approval_type: str = ""
    blocked_flags: tuple[str, ...] = BLOCKED_TOOL_FLAGS
    render_argv: Callable[[ActionInvocation], list[str]] | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.params_schema:
            object.__setattr__(self, "params_schema", _default_params_schema(self.name))
        if not self.execution_semantics:
            semantics = {
                "read_only": "read_only",
                "writes_run": "idempotent_write",
                "dp_sql_pull": "external_unknown",
                "external_data": "external_unknown",
            }.get(self.permission, "external_unknown")
            object.__setattr__(self, "execution_semantics", semantics)
        if not self.approval_type:
            object.__setattr__(self, "approval_type", "sql_review" if self.requires_approval else "none")
        if self.render_argv is None:
            object.__setattr__(self, "render_argv", _typed_renderer(self.name))

    @property
    def command_template(self) -> str:
        return self.command

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "action_id": self.action_id,
            "command": self.command,
            "command_template": self.command_template,
            "permission": self.permission,
            "description": self.description,
            "requires_approval": self.requires_approval,
            "allowed_for_auditor": self.allowed_for_auditor,
            "params_schema": self.params_schema,
            "execution_semantics": self.execution_semantics,
            "approval_type": self.approval_type,
            "blocked_flags": list(self.blocked_flags),
        }


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="project_status",
        action_id="project_status",
        command=display_template("project_status"),
        permission="read_only",
        description="Read project continuity status.",
        allowed_for_auditor=True,
    ),
    ToolSpec(
        name="run_status",
        action_id="run_status",
        command=display_template("run_status"),
        permission="read_only",
        description="Read run_state.yml.",
        allowed_for_auditor=True,
    ),
    ToolSpec(
        name="run_audit",
        action_id="run_audit",
        command=display_template("run_audit"),
        permission="read_only",
        description="Audit run or stage evidence without mutation.",
        allowed_for_auditor=True,
    ),
    ToolSpec(
        name="workflow_validate",
        action_id="workflow_validate",
        command=display_template("workflow_validate"),
        permission="read_only",
        description="Validate workflow contracts.",
        allowed_for_auditor=True,
    ),
    ToolSpec(
        name="rules_list",
        action_id="rules_list",
        command=display_template("rules_list"),
        permission="read_only",
        description="Read promoted workbench rules.",
        allowed_for_auditor=True,
    ),
    ToolSpec(
        name="sample_check",
        action_id="sample_check",
        command=display_template("sample_check"),
        permission="writes_run",
        description="Write sample check artifacts and stage state.",
    ),
    ToolSpec(
        name="feature_metadata",
        action_id="feature_metadata",
        command=display_template("feature_metadata"),
        permission="writes_run",
        description="Write feature metadata artifacts and stage state.",
    ),
    ToolSpec(
        name="feature_prescreen_local",
        action_id="feature_prescreen",
        command=display_template("feature_prescreen_local"),
        permission="writes_run",
        description="Complete feature prescreen by design from an approved local Feather source.",
    ),
    ToolSpec(
        name="feature_prescreen_prepare",
        action_id="feature_prescreen",
        command=display_template("feature_prescreen_prepare"),
        permission="writes_run",
        description="Generate feature prescreen SQL review artifacts without DP pull.",
    ),
    ToolSpec(
        name="feature_prescreen_execute",
        action_id="feature_prescreen",
        command=display_template("feature_prescreen_execute"),
        permission="dp_sql_pull",
        description="Run feature prescreening with approved DP/SQL access.",
        requires_approval=True,
    ),
    ToolSpec(
        name="build_wide_sql_local",
        action_id="build_wide_sql",
        command=display_template("build_wide_sql_local"),
        permission="writes_run",
        description="Record the local-Feather wide SQL stage as skipped by design.",
    ),
    ToolSpec(
        name="build_wide_sql_prepare",
        action_id="build_wide_sql",
        command=display_template("build_wide_sql_prepare"),
        permission="writes_run",
        description="Generate wide-table SQL artifacts.",
    ),
    ToolSpec(
        name="build_wide_sql_execute",
        action_id="build_wide_sql",
        command=display_template("build_wide_sql_execute"),
        permission="dp_sql_pull",
        description="Execute reviewed wide-table create SQL through TMLSQLClient.",
        requires_approval=True,
    ),
    ToolSpec(
        name="feature_refine_local",
        action_id="feature_refine",
        command=display_template("feature_refine_local"),
        permission="writes_run",
        description="Run feature refinement against the approved local Feather source.",
    ),
    ToolSpec(
        name="feature_refine_prepare",
        action_id="feature_refine",
        command=display_template("feature_refine_prepare"),
        permission="writes_run",
        description="Generate feature refine SQL review artifacts without DP pull.",
    ),
    ToolSpec(
        name="feature_refine_execute",
        action_id="feature_refine",
        command=display_template("feature_refine_execute"),
        permission="dp_sql_pull",
        description="Run feature refinement with approved DP/SQL access.",
        requires_approval=True,
    ),
    ToolSpec(
        name="train_baseline",
        action_id="train_baseline",
        command=display_template("train_baseline"),
        permission="writes_run",
        description="Train or scaffold baseline model artifacts.",
    ),
    ToolSpec(
        name="evaluate",
        action_id="evaluate",
        command=display_template("evaluate"),
        permission="writes_run",
        description="Write model evaluation artifacts.",
    ),
    ToolSpec(
        name="compare",
        action_id="compare",
        command=display_template("compare"),
        permission="writes_run",
        description="Write champion/challenger comparison artifacts.",
    ),
    ToolSpec(
        name="report",
        action_id="report",
        command=display_template("report"),
        permission="writes_run",
        description="Write report artifacts.",
    ),
)

TOOL_REGISTRY: dict[str, ToolSpec] = {spec.name: spec for spec in TOOL_SPECS}
TOOL_ALIASES = {
    "feature_d01_d02_dry_run": "feature_prescreen_prepare",
    "feature_d01_d02_pull": "feature_prescreen_execute",
    "feature_prescreen_dry_run": "feature_prescreen_prepare",
    "feature_prescreen_pull": "feature_prescreen_execute",
    "build_wide_sql": "build_wide_sql_prepare",
    "feature_refine_dry_run": "feature_refine_prepare",
    "feature_refine_pull": "feature_refine_execute",
}


def list_tool_specs(*, permission: str | None = None) -> tuple[ToolSpec, ...]:
    specs = TOOL_SPECS
    if permission:
        specs = tuple(spec for spec in specs if spec.permission == permission)
    return tuple(sorted(specs, key=lambda spec: spec.name))


def get_tool_spec(name: str) -> ToolSpec:
    name = TOOL_ALIASES.get(name, name)
    try:
        return TOOL_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown tool: {name}") from exc


def validate_tool_registry() -> list[str]:
    errors: list[str] = []
    for spec in TOOL_SPECS:
        try:
            action = get_action_spec(spec.action_id)
        except KeyError:
            errors.append(f"tool {spec.name} references unknown action: {spec.action_id}")
            continue
        if spec.requires_approval and not action.approval_required and "sql_approval_required" not in action.failure_codes:
            errors.append(f"tool {spec.name} requires approval but action {spec.action_id} does not")
        if spec.allowed_for_auditor and spec.permission != "read_only":
            errors.append(f"tool {spec.name} is auditor-allowed but permission is {spec.permission}")
        if spec.execution_semantics not in EXECUTION_SEMANTICS:
            errors.append(f"tool {spec.name} has invalid execution semantics: {spec.execution_semantics}")
        if not callable(spec.render_argv):
            errors.append(f"tool {spec.name} has no argv renderer")
    return errors


def registry_digest(registry: dict[str, ToolSpec] | None = None) -> str:
    source = TOOL_REGISTRY if registry is None else registry
    rows = []
    for name in sorted(source):
        spec = source[name]
        rows.append(
            {
                "name": spec.name,
                "action_id": spec.action_id,
                "params_schema": spec.params_schema,
                "execution_semantics": spec.execution_semantics,
                "approval_type": spec.approval_type,
                "blocked_flags": list(spec.blocked_flags),
                "command_template": spec.command_template,
                "permission": spec.permission,
                "requires_approval": spec.requires_approval,
                "allowed_for_auditor": spec.allowed_for_auditor,
            }
        )
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def tool_to_dict(spec: ToolSpec) -> dict[str, object]:
    return spec.to_dict()


def format_tool_list(specs: tuple[ToolSpec, ...]) -> str:
    lines = ["Tool Name                  Permission    Approval  Auditor  Action ID             Command"]
    lines.append("-" * 110)
    for spec in specs:
        approval = "yes" if spec.requires_approval else "no"
        auditor = "yes" if spec.allowed_for_auditor else "no"
        lines.append(
            f"{spec.name:<26} {spec.permission:<13} {approval:<9} {auditor:<8} {spec.action_id:<21} {spec.command}"
        )
    return "\n".join(lines) + "\n"

def format_tool_detail(spec: ToolSpec) -> str:
    lines = [
        f"tool: {spec.name}",
        f"action_id: {spec.action_id}",
        f"permission: {spec.permission}",
        f"requires_approval: {spec.requires_approval}",
        f"allowed_for_auditor: {spec.allowed_for_auditor}",
        f"command: {spec.command}",
        f"execution_semantics: {spec.execution_semantics}",
        f"approval_type: {spec.approval_type}",
        f"description: {spec.description}",
    ]
    return "\n".join(lines) + "\n"
