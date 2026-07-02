"""Validate model request documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from risk_model_workbench.config import load_yaml
from risk_model_workbench.paths import project_config_path, workflow_path
from risk_model_workbench.request.data_source import (
    LOCAL_FEATHER,
    REMOTE_TABLE,
    VALID_DATA_SOURCE_MODES,
    has_explicit_data_source_mode,
    is_feather_location,
    resolve_data_source_mode,
    sample_location,
)
from risk_model_workbench.planning.steps import resolve_step_configuration
from risk_model_workbench.request.splits import check_split_consistency


REQUIRED_FIELDS = [
    "request_id",
    "project",
    "target_column",
    "split_column",
    "evaluation",
    "reports",
]

SUPPORTED_TRAINING_METHODS = {
    "lightgbm",
    "xgboost",
    "logistic_regression",
    "custom",
    "hier_ranknet",
    "teacher_student_distillation",
}

METHOD_ALIASES = {
    "lgb": "lightgbm",
    "lgbm": "lightgbm",
    "xgb": "xgboost",
    "lr": "logistic_regression",
    "logistic": "logistic_regression",
    "ranknet": "hier_ranknet",
    "teacher_student": "teacher_student_distillation",
}

SUPPORTED_METRICS = {"auc", "ks", "decile_lift", "ranking_inversion", "psi", "business_risk"}
SUPPORTED_REPORT_EXTENSIONS = {"", ".md", ".markdown", ".html", ".xlsx", ".json"}
SUPPORTED_TRAINING_MODES = {"single_train", "fixed", "baseline", "llm_guided_tune", "llm-guided-tune", "llm_guided"}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normal_method(value: Any) -> str:
    raw = str(value or "lightgbm").strip().lower()
    return METHOD_ALIASES.get(raw, raw)


def _custom_entrypoint(project_config: dict[str, Any]) -> Any:
    return (
        project_config.get("training", {}).get("custom_entrypoint")
        or project_config.get("custom_training", {}).get("entrypoint")
        or project_config.get("modeling", {}).get("custom_training_entrypoint")
    )


def validate_model_request(request_doc: dict[str, Any], project_dir: str | Path | None = None) -> dict[str, Any]:
    """Return validation errors and warnings for a parsed model request."""
    metadata = request_doc.get("metadata", {})
    errors: list[str] = []
    warnings: list[str] = []
    project_config: dict[str, Any] = {}
    configured_ids: list[Any] = []

    if project_dir:
        project_path = Path(project_dir).resolve()
        config_path = project_config_path(project_path)
        if not config_path.exists():
            errors.append(f"missing project config: {config_path}")
        else:
            project_config = load_yaml(config_path)
            configured_ids = project_config.get("data", {}).get("id_columns") or []

    for field in REQUIRED_FIELDS:
        if field not in metadata or metadata.get(field) in (None, "", []):
            errors.append(f"missing required field: {field}")

    experiments = metadata.get("experiments")
    experiment_description = str(metadata.get("experiment_description") or "").strip()
    if experiments in (None, "", []) and not experiment_description:
        errors.append("missing required field: experiments")
    if "experiments" in metadata and not isinstance(metadata.get("experiments"), list):
        errors.append("experiments must be a list")
    if "id_columns" in metadata and not isinstance(metadata.get("id_columns"), list):
        errors.append("id_columns must be a list")
    for index, item in enumerate(experiments if isinstance(experiments, list) else [], start=1):
        if not isinstance(item, dict):
            errors.append(f"experiments[{index}] must be a mapping")
            continue
        method = _normal_method(item.get("method") or item.get("algorithm"))
        if method not in SUPPORTED_TRAINING_METHODS:
            errors.append(f"unsupported training method in experiments[{index}]: {method}")
        if method == "custom" and not _custom_entrypoint(project_config):
            errors.append("custom training requires training.custom_entrypoint or custom_training.entrypoint in project config")

    training = metadata.get("training") or {}
    if "training" in metadata and not isinstance(metadata.get("training"), dict):
        errors.append("training must be a mapping")
    elif isinstance(training, dict):
        mode = str(training.get("mode") or "").strip().lower()
        if mode and mode not in SUPPORTED_TRAINING_MODES:
            errors.append(f"unsupported training.mode: {mode}")
        tuning = training.get("tuning") or {}
        if "tuning" in training and not isinstance(tuning, dict):
            errors.append("training.tuning must be a mapping")
        elif isinstance(tuning, dict):
            for key in ["max_rounds", "candidates_per_round", "max_trials"]:
                if key in tuning:
                    try:
                        if int(tuning[key]) < 1:
                            errors.append(f"training.tuning.{key} must be >= 1")
                    except (TypeError, ValueError):
                        errors.append(f"training.tuning.{key} must be an integer")

    workflow = metadata.get("workflow", "full_modeling")
    if workflow and not workflow_path(str(workflow)).exists():
        errors.append(f"unknown workflow: {workflow}")

    raw_data_source_mode = str(metadata.get("data_source_mode") or "").strip()
    if raw_data_source_mode and raw_data_source_mode not in VALID_DATA_SOURCE_MODES:
        errors.append(f"unsupported data_source_mode: {raw_data_source_mode}")
    data_source_mode = resolve_data_source_mode(metadata)
    location = sample_location(metadata)
    if data_source_mode == LOCAL_FEATHER:
        if not location:
            errors.append("local_feather data_source_mode requires sample_location")
        elif not is_feather_location(location):
            errors.append("local_feather data_source_mode requires sample_location ending with .feather")
        else:
            local_path = Path(location)
            if not local_path.is_absolute():
                first_part = local_path.parts[0] if local_path.parts else ""
                if first_part not in {"data", "runs"}:
                    warnings.append("local_feather sample_location should usually live under an ignored data/ or runs/ path")
            if not has_explicit_data_source_mode(metadata):
                warnings.append("data_source_mode inferred as local_feather from .feather sample_location")
    elif raw_data_source_mode == REMOTE_TABLE and is_feather_location(location):
        errors.append("remote_table data_source_mode must not use a .feather sample_location")

    evaluation = metadata.get("evaluation") or {}
    if not isinstance(evaluation, dict):
        errors.append("evaluation must be a mapping")
    elif not _as_list(evaluation.get("metrics")):
        warnings.append("evaluation.metrics is empty")
    else:
        unknown_metrics = sorted({str(metric) for metric in _as_list(evaluation.get("metrics"))} - SUPPORTED_METRICS)
        if unknown_metrics:
            errors.append(f"unsupported evaluation metric: {', '.join(unknown_metrics)}")

    reports = metadata.get("reports") or {}
    if not isinstance(reports, dict):
        errors.append("reports must be a mapping")
    elif not _as_list(reports.get("outputs")):
        warnings.append("reports.outputs is empty")
    else:
        unsupported_outputs = []
        for output in _as_list(reports.get("outputs")):
            suffix = Path(str(output)).suffix.lower()
            if suffix not in SUPPORTED_REPORT_EXTENSIONS:
                unsupported_outputs.append(str(output))
        if unsupported_outputs:
            errors.append(f"unsupported report output type: {', '.join(unsupported_outputs)}")
    if isinstance(reports, dict):
        score_labels = reports.get("score_labels")
        if "score_labels" in reports and not isinstance(score_labels, dict):
            errors.append("reports.score_labels must be a mapping")
        targets = reports.get("targets") or []
        if "targets" in reports and not isinstance(targets, list):
            errors.append("reports.targets must be a list")
        elif isinstance(targets, list):
            seen_targets: set[str] = set()
            for index, target in enumerate(targets, start=1):
                if not isinstance(target, dict):
                    errors.append(f"reports.targets[{index}] must be a mapping")
                    continue
                target_name = str(target.get("name") or "").strip()
                if not target_name:
                    errors.append(f"reports.targets[{index}].name is required")
                elif target_name in seen_targets:
                    errors.append(f"duplicate reports.targets name: {target_name}")
                seen_targets.add(target_name)
                target_labels = target.get("score_labels")
                if "score_labels" in target and not isinstance(target_labels, dict):
                    errors.append(f"reports.targets[{index}].score_labels must be a mapping")
                for field in ["train_dir", "eval_dir", "output_dir", "experiment"]:
                    if field in target and target.get(field) not in (None, "") and not isinstance(target.get(field), str):
                        errors.append(f"reports.targets[{index}].{field} must be a string")
                unsupported_target_outputs = []
                for output in _as_list(target.get("outputs")):
                    suffix = Path(str(output)).suffix.lower()
                    if suffix not in SUPPORTED_REPORT_EXTENSIONS:
                        unsupported_target_outputs.append(str(output))
                if unsupported_target_outputs:
                    errors.append(
                        f"unsupported report output type in reports.targets[{index}]: {', '.join(unsupported_target_outputs)}"
                    )

    try:
        step_config = resolve_step_configuration(metadata, project_dir)
        if not metadata.get("scenario_profile"):
            warnings.append(f"scenario_profile inferred as {step_config['scenario_profile']}")
        unresolved = [step["id"] for step in step_config.get("resolved_steps", []) if step.get("implementation_status") != "implemented"]
        if unresolved:
            errors.append(f"selected steps do not have executors: {', '.join(unresolved)}")
    except ValueError as exc:
        errors.append(str(exc))

    if project_dir and project_config:
        configured_project = project_config.get("project", {}).get("name")
        configured_data = project_config.get("data", {})
        if configured_project and metadata.get("project") and metadata["project"] != configured_project:
            warnings.append(f"request project differs from project.yml: {metadata['project']} != {configured_project}")
        if configured_data.get("target_column") and metadata.get("target_column") != configured_data.get("target_column"):
            warnings.append(
                f"request target_column differs from project.yml: {metadata.get('target_column')} != {configured_data.get('target_column')}"
            )
        if configured_ids and metadata.get("id_columns") and metadata.get("id_columns") != configured_ids:
            warnings.append(f"request id_columns differs from project.yml: {metadata.get('id_columns')} != {configured_ids}")

    request_ids = _as_list(metadata.get("id_columns"))
    if not request_ids and configured_ids:
        warnings.append("request id_columns omitted; using project.yml data.id_columns")
    elif not request_ids and not configured_ids:
        errors.append("missing required field: id_columns (not found in request or project.yml)")

    # Split consistency: time-out (oot) must never feed the in-time validation
    # set (oos) used by early stopping / tuning. Errors are returned under a
    # dedicated "split_errors" key so the CLI can downgrade them to warnings via
    # --skip-split-check without touching structural errors.
    split_check = check_split_consistency(metadata, project_config or {})
    warnings.extend(split_check["warnings"])

    return {
        "status": "ok" if not errors and not split_check["errors"] else "failed",
        "errors": errors,
        "split_errors": split_check["errors"],
        "warnings": warnings,
    }
