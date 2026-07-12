"""Workflow stage contract validation and artifact matching."""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from risk_model_workbench.config import ConfigFormatError, load_yaml
from risk_model_workbench.paths import REPO_ROOT, workflow_path


CONTRACT_FIELDS = {
    "required_artifacts",
    "accepted_artifact_sets",
    "allow_scaffold",
    "allow_imported",
    "closure_required",
}

STAGE_CONTRACT_REGISTRY = REPO_ROOT / "workflows" / "stage_contracts.yml"
REGISTRY_FIELDS = {"name", "description", "stages", "stage_contracts"}


class WorkflowContractError(ValueError):
    """Raised when workflow contracts cannot be resolved safely."""

    def __init__(self, path: str | Path, detail: str):
        self.path = Path(path)
        self.detail = detail
        super().__init__(f"invalid workflow contract {self.path}: {detail}")


def load_stage_contracts(workflow: str) -> tuple[dict[str, dict[str, Any]], str]:
    """Resolve workflow overrides against the shared stage contract registry."""
    path = workflow_path(workflow)
    if not path.exists():
        return {}, ""
    payload = _load_contract_yaml(path)
    errors = validate_workflow_definition(payload)
    if errors:
        raise WorkflowContractError(path, "; ".join(errors))
    stages = payload.get("stages")
    assert isinstance(stages, list)
    raw_overrides = payload.get("stage_contracts")
    overrides = {} if raw_overrides is None else raw_overrides
    assert isinstance(overrides, dict)
    registry = _load_contract_registry()
    contracts = _compose_stage_contracts(stages, registry, overrides)
    return contracts, _display_path(path)


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

    stage_names = {stage for stage in stages if isinstance(stage, str)}
    raw_contracts = workflow.get("stage_contracts")
    contracts = {} if raw_contracts is None else raw_contracts
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
        if stage_name not in stage_names:
            errors.extend(_validate_contract(stage_name, contract))

    registry = _load_contract_registry()
    effective_contracts = _compose_stage_contracts(stages, registry, contracts)
    for stage_name in stages:
        if not isinstance(stage_name, str):
            continue
        override = contracts.get(stage_name)
        has_standard_contract = stage_name in registry
        explicitly_open = isinstance(override, dict) and override.get("closure_required") is False
        if not has_standard_contract and not explicitly_open:
            errors.append(
                f"stage {stage_name} must resolve a standard contract or explicitly set closure_required: false"
            )
            continue
        if isinstance(override, dict):
            errors.extend(_validate_override(stage_name, registry.get(stage_name), override))
        errors.extend(_validate_contract(stage_name, effective_contracts.get(stage_name, {})))
    return errors


def _load_contract_registry() -> dict[str, dict[str, Any]]:
    path = STAGE_CONTRACT_REGISTRY
    if not path.exists():
        raise WorkflowContractError(path, "registry file is missing")
    payload = _load_contract_yaml(path)
    errors: list[str] = []
    unknown_fields = sorted(set(payload) - REGISTRY_FIELDS)
    if unknown_fields:
        errors.append(f"unknown registry fields: {', '.join(unknown_fields)}")
    if payload.get("name") != "stage_contracts":
        errors.append("name must be stage_contracts")
    stages = payload.get("stages")
    if not isinstance(stages, list) or not stages:
        errors.append("stages must be a non-empty list")
        stages = []
    elif any(not isinstance(stage, str) or not stage.strip() for stage in stages):
        errors.append("stages must contain non-empty strings")
    elif len(stages) != len(set(stages)):
        errors.append("stages must not contain duplicates")
    contracts = payload.get("stage_contracts")
    if not isinstance(contracts, dict):
        errors.append("stage_contracts must be a mapping")
        contracts = {}
    elif not contracts:
        errors.append("stage_contracts must not be empty")

    registry: dict[str, dict[str, Any]] = {}
    for raw_stage, contract in contracts.items():
        stage = str(raw_stage)
        if not isinstance(contract, dict):
            errors.append(f"stage_contracts.{stage} must be a mapping")
            continue
        registry[stage] = dict(contract)
        errors.extend(_validate_contract(stage, contract))

    valid_stages = {stage for stage in stages if isinstance(stage, str)}
    if valid_stages != set(registry):
        errors.append("stages must exactly match stage_contracts keys")
    if errors:
        raise WorkflowContractError(path, "; ".join(errors))
    return registry


def _load_contract_yaml(path: Path) -> dict[str, Any]:
    try:
        return load_yaml(path)
    except ConfigFormatError as exc:
        raise WorkflowContractError(path, exc.detail) from exc
    except OSError as exc:
        raise WorkflowContractError(path, str(exc)) from exc


def _compose_stage_contracts(
    stages: list[Any],
    registry: dict[str, dict[str, Any]],
    overrides: dict[Any, Any],
) -> dict[str, dict[str, Any]]:
    contracts: dict[str, dict[str, Any]] = {}
    for raw_stage in stages:
        stage = str(raw_stage)
        standard = registry.get(stage)
        override = overrides.get(stage)
        if standard is None and not isinstance(override, dict):
            continue
        contract = dict(standard or {})
        if isinstance(override, dict):
            contract.update(override)
        contracts[stage] = contract
    return contracts


def _validate_contract(stage: str, contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    unknown_fields = sorted(set(contract) - CONTRACT_FIELDS)
    if unknown_fields:
        errors.append(f"stage_contracts.{stage} has unknown fields: {', '.join(unknown_fields)}")
    for key in ["allow_scaffold", "allow_imported", "closure_required"]:
        if key in contract and not isinstance(contract[key], bool):
            errors.append(f"stage_contracts.{stage}.{key} must be a boolean")
    errors.extend(_validate_pattern_list(stage, "required_artifacts", contract.get("required_artifacts")))
    errors.extend(_validate_artifact_sets(stage, contract.get("accepted_artifact_sets")))
    closure_required = contract.get("closure_required", True)
    if closure_required is not False and not _has_artifact_rule(contract):
        errors.append(
            f"stage_contracts.{stage} with closure_required true must declare at least one non-empty artifact rule"
        )
    return errors


def _validate_override(
    stage: str,
    standard: dict[str, Any] | None,
    override: dict[str, Any],
) -> list[str]:
    if not standard:
        return []
    errors = []
    for field in ["required_artifacts", "accepted_artifact_sets"]:
        if field in standard and field in override and (override[field] is None or override[field] == []):
            errors.append(f"stage_contracts.{stage}.{field} cannot clear standard artifact rule")
    return errors


def _has_artifact_rule(contract: dict[str, Any]) -> bool:
    required = contract.get("required_artifacts")
    if isinstance(required, list) and bool(required):
        return True
    accepted = contract.get("accepted_artifact_sets")
    return isinstance(accepted, list) and any(
        isinstance(artifact_set, list) and bool(artifact_set) for artifact_set in accepted
    )


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
