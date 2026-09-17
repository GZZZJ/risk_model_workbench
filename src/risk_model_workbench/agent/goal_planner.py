"""Natural-language goal interpretation into a validated RMW request draft."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from risk_model_workbench.agent.context_pack import context_pack_hash
from risk_model_workbench.agent.model_gateway import ModelGateway
from risk_model_workbench.agent.reasoning_contracts import GoalInterpretation, GoalReasoningRequest
from risk_model_workbench.config import load_yaml
from risk_model_workbench.paths import REPO_ROOT, project_config_path, workflow_path
from risk_model_workbench.request import parse_model_request, validate_model_request


class GoalPlanningError(RuntimeError):
    """Raised when a natural-language goal cannot become a valid request."""


def create_request_draft_from_objective(
    project_dir: str | Path,
    *,
    objective: str,
    request_id: str,
    gateway: ModelGateway,
    output: str | Path | None = None,
) -> dict[str, Any]:
    project = Path(project_dir).resolve()
    config = load_yaml(project_config_path(project))
    contract = _project_contract(config)
    pack = _goal_context_pack(project, contract)
    request = GoalReasoningRequest(
        request_id=request_id,
        question=objective.strip(),
        context_hash=pack["context_hash"],
        context_pack=pack,
        constraints=[
            "Project data fields are authoritative and may not be changed.",
            "SQL/DP approval may not be bypassed.",
            "The generated request is a draft requiring human confirmation.",
        ],
    )
    interpretation, model_metadata = gateway.generate_structured(request, GoalInterpretation)
    if not workflow_path(interpretation.workflow).exists():
        raise GoalPlanningError(f"model selected an unknown workflow: {interpretation.workflow}")
    if interpretation.workflow not in contract["available_workflows"]:
        raise GoalPlanningError(f"model selected a workflow outside the project contract: {interpretation.workflow}")

    metadata: dict[str, Any] = {
        "request_id": request_id,
        "project": contract["project"],
        "workflow": interpretation.workflow,
        "target_column": contract["target_column"],
        "id_columns": contract["id_columns"],
        "split_column": contract["split_column"],
        "sample_checks": ["sample_check_001"],
        "experiments": [
            {
                "name": interpretation.experiment_name,
                "method": interpretation.method,
                "description": interpretation.objective_summary,
            }
        ],
        "training": {"mode": interpretation.training_mode},
        "evaluation": {"metrics": interpretation.metrics, "champions": []},
        "reports": {"outputs": interpretation.report_outputs},
        "agent_goal": {
            "objective": objective.strip(),
            "summary": interpretation.objective_summary,
            "assumptions": interpretation.assumptions,
            "requires_user_confirmation": True,
            "context_hash": pack["context_hash"],
            "model_provider": model_metadata.provider,
            "model": model_metadata.model,
            "prompt_hash": model_metadata.prompt_hash,
            "response_hash": model_metadata.response_hash,
        },
    }
    if interpretation.scenario_profile:
        metadata["scenario_profile"] = interpretation.scenario_profile
    if contract.get("data_source_mode"):
        metadata["data_source_mode"] = contract["data_source_mode"]
    if contract.get("sample_location"):
        metadata["sample_location"] = contract["sample_location"]

    body = (
        f"# Embedded Agent Request Draft\n\n"
        f"## Objective\n\n{objective.strip()}\n\n"
        f"## Interpretation\n\n{interpretation.objective_summary}\n\n"
        "Review the YAML contract before starting the Agent. The model output is not execution authority.\n"
    )
    target = Path(output) if output else project / "requests" / f"{request_id}.md"
    if not target.is_absolute():
        target = project / target
    target = target.resolve()
    requests_root = (project / "requests").resolve()
    if target != requests_root and requests_root not in target.parents:
        raise GoalPlanningError("generated request output must stay under project/requests")
    target.parent.mkdir(parents=True, exist_ok=True)
    content = "---\n" + yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False) + "---\n" + body
    target_existed = target.exists()
    if target_existed and target.read_text(encoding="utf-8") != content:
        raise GoalPlanningError(f"request draft already exists with different content: {target}")
    if not target_existed:
        target.write_text(content, encoding="utf-8")
    request_doc = parse_model_request(target)
    validation = validate_model_request(request_doc, project)
    blocking = list(validation.get("errors") or []) + list(validation.get("split_errors") or [])
    if blocking:
        if not target_existed:
            target.unlink(missing_ok=True)
        raise GoalPlanningError("generated request failed deterministic validation: " + "; ".join(blocking))
    audit_path = project / "requests" / "audit" / f"{request_id}.model_invocation.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_content = json.dumps(
        {
            "version": 1,
            "request_id": request_id,
            "context_hash": pack["context_hash"],
            "model": model_metadata.model_dump(mode="json"),
            "requires_user_confirmation": True,
            "request_path": str(target.relative_to(project)),
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    if audit_path.exists() and audit_path.read_text(encoding="utf-8") != audit_content:
        raise GoalPlanningError(f"goal model invocation evidence already exists with different content: {audit_path}")
    if not audit_path.exists():
        audit_path.write_text(audit_content, encoding="utf-8")
    return {
        "request_path": str(target),
        "audit_path": str(audit_path),
        "validation": validation,
        "requires_user_confirmation": True,
        "interpretation": interpretation.model_dump(mode="json"),
    }


def _project_contract(config: dict[str, Any]) -> dict[str, Any]:
    project = config.get("project") if isinstance(config.get("project"), dict) else {}
    data = config.get("data") if isinstance(config.get("data"), dict) else {}
    project_name = str(project.get("name") or "").strip()
    target = str(data.get("target_column") or "").strip()
    ids = [str(item) for item in data.get("id_columns") or [] if str(item)]
    split = str(data.get("period_column") or data.get("time_column") or "").strip()
    missing = [name for name, value in [("project.name", project_name), ("data.target_column", target), ("data.id_columns", ids), ("data.period_column/time_column", split)] if not value]
    if missing:
        raise GoalPlanningError("project config is missing authoritative goal fields: " + ", ".join(missing))
    available_workflows = sorted(path.stem for path in (REPO_ROOT / "workflows").glob("*.yml"))
    return {
        "project": project_name,
        "target_column": target,
        "id_columns": ids,
        "split_column": split,
        "data_source_mode": str(data.get("data_source_mode") or config.get("data_source_mode") or "").strip(),
        "sample_location": str(data.get("sample_location") or config.get("sample_location") or "").strip(),
        "available_workflows": available_workflows,
        "supported_methods": [
            "lightgbm",
            "xgboost",
            "logistic_regression",
            "custom",
            "hier_ranknet",
            "teacher_student_distillation",
        ],
        "supported_metrics": ["auc", "ks", "decile_lift", "ranking_inversion", "psi", "business_risk"],
    }


def _goal_context_pack(project: Path, contract: dict[str, Any]) -> dict[str, Any]:
    model_contract = {key: value for key, value in contract.items() if key != "sample_location"}
    pack = {
        "version": 1,
        "project": project.name,
        "version_id": "request_draft",
        "task_id": "goal_interpretation",
        "attempt_id": "goal_interpretation",
        "request_type": "goal_interpretation",
        "files": [
            {
                "path": project_config_path(project).name,
                "sha256": "",
                "size": 0,
                "summary": {"kind": "project_contract", "content": model_contract},
                "missing": False,
                "status": "included",
            }
        ],
        "constraints": [],
        "allowed_tools": [],
        "output_contract": {"type": "goal_plan"},
        "provenance": [project_config_path(project).name],
        "limits": {},
        "truncation": {"requested_entries": 1, "recorded_entries": 1, "omitted_entries": 0},
    }
    pack["context_hash"] = context_pack_hash(pack)
    return pack
