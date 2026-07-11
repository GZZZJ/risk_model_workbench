"""Workflow stage contract validation and artifact matching."""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import yaml

from risk_model_workbench.paths import REPO_ROOT, workflow_path


CONTRACT_FIELDS = {
    "required_artifacts",
    "accepted_artifact_sets",
    "allow_scaffold",
    "allow_imported",
    "closure_required",
}


def load_stage_contracts(workflow: str) -> tuple[dict[str, dict[str, Any]], str]:
    """Load stage contracts for a workflow name when its YAML is available."""
    path = workflow_path(workflow)
    if not path.exists():
        return {}, ""
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    contracts = payload.get("stage_contracts")
    if not isinstance(contracts, dict):
        return {}, _display_path(path)
    return {str(stage): contract for stage, contract in contracts.items() if isinstance(contract, dict)}, _display_path(path)


def validate_workflow_definition(workflow: dict[str, Any]) -> list[str]:
    """Return validation errors for workflow structure and stage contracts."""
    errors: list[str] = []
    stages = workflow.get("stages")
    if not workflow.get("name"):
        errors.append("missing workflow name")
    if not isinstance(stages, list) or not stages:
        errors.append("stages must be a non-empty list")
        stages = []
    elif any(not isinstance(stage, str) or not stage.strip() for stage in stages):
        errors.append("stages must contain non-empty strings")

    stage_names = set(stages)
    contracts = workflow.get("stage_contracts", {})
    if contracts in (None, {}):
        return errors
    if not isinstance(contracts, dict):
        errors.append("stage_contracts must be a mapping")
        return errors

    for stage, contract in contracts.items():
        stage_name = str(stage)
        if stage_name not in stage_names:
            errors.append(f"stage_contracts references unknown stage: {stage_name}")
        if not isinstance(contract, dict):
            errors.append(f"stage_contracts.{stage_name} must be a mapping")
            continue
        unknown_fields = sorted(set(contract) - CONTRACT_FIELDS)
        if unknown_fields:
            errors.append(f"stage_contracts.{stage_name} has unknown fields: {', '.join(unknown_fields)}")
        for key in ["allow_scaffold", "allow_imported", "closure_required"]:
            if key in contract and not isinstance(contract[key], bool):
                errors.append(f"stage_contracts.{stage_name}.{key} must be a boolean")
        errors.extend(_validate_pattern_list(stage_name, "required_artifacts", contract.get("required_artifacts")))
        errors.extend(_validate_artifact_sets(stage_name, contract.get("accepted_artifact_sets")))
    return errors


def audit_contract_artifacts(
    contract: dict[str, Any],
    manifest_items: list[dict[str, Any]],
    run_path: str | Path,
) -> list[str]:
    """Return contract issues for a closed stage."""
    if not contract or not contract.get("closure_required", True):
        return []

    issues: list[str] = []
    for pattern in contract.get("required_artifacts") or []:
        ok, reason = _pattern_satisfied(str(pattern), manifest_items, run_path)
        if not ok:
            issues.append(f"contract required artifact {reason}: {pattern}")

    accepted_sets = contract.get("accepted_artifact_sets") or []
    if accepted_sets:
        set_results = []
        for artifact_set in accepted_sets:
            failures = []
            for pattern in artifact_set:
                ok, reason = _pattern_satisfied(str(pattern), manifest_items, run_path)
                if not ok:
                    failures.append(f"{pattern} ({reason})")
            if not failures:
                return issues
            set_results.append("; ".join(failures))
        issues.append(f"no accepted artifact set satisfied: {' | '.join(set_results)}")
    return issues


def artifact_exists(run_path: str | Path, artifact: dict[str, Any]) -> bool:
    """Check the artifact's current filesystem existence."""
    if artifact.get("exists") is False:
        return False
    if artifact.get("storage_class") in {"local_only", "external"}:
        return False
    raw_path = artifact.get("path")
    if not raw_path:
        return False
    root = Path(run_path).resolve()
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return False
    if artifact.get("kind") == "directory":
        return path.is_dir()
    return path.is_file()


def _pattern_satisfied(pattern: str, manifest_items: list[dict[str, Any]], run_path: str | Path) -> tuple[bool, str]:
    matches = [item for item in manifest_items if fnmatch(str(item.get("path", "")), pattern)]
    if not matches:
        return False, "not registered"
    if not any(artifact_exists(run_path, item) for item in matches):
        return False, "registered but missing"
    return True, ""


def _validate_pattern_list(stage: str, field: str, value: Any) -> list[str]:
    if value in (None, []):
        return []
    if not isinstance(value, list):
        return [f"stage_contracts.{stage}.{field} must be a list"]
    errors = []
    for idx, pattern in enumerate(value):
        errors.extend(_validate_pattern(stage, f"{field}[{idx}]", pattern))
    return errors


def _validate_artifact_sets(stage: str, value: Any) -> list[str]:
    if value in (None, []):
        return []
    if not isinstance(value, list):
        return [f"stage_contracts.{stage}.accepted_artifact_sets must be a list"]
    errors = []
    for set_idx, artifact_set in enumerate(value):
        if not isinstance(artifact_set, list) or not artifact_set:
            errors.append(f"stage_contracts.{stage}.accepted_artifact_sets[{set_idx}] must be a non-empty list")
            continue
        for pattern_idx, pattern in enumerate(artifact_set):
            errors.extend(_validate_pattern(stage, f"accepted_artifact_sets[{set_idx}][{pattern_idx}]", pattern))
    return errors


def _validate_pattern(stage: str, field: str, pattern: Any) -> list[str]:
    if not isinstance(pattern, str) or not pattern.strip():
        return [f"stage_contracts.{stage}.{field} must be a non-empty string"]
    path = Path(pattern)
    if path.is_absolute() or ".." in path.parts:
        return [f"stage_contracts.{stage}.{field} must be a run-relative artifact pattern"]
    return []


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())
