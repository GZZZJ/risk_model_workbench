"""Materialize model requests into run-scoped runtime configs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from risk_model_workbench.config import dump_yaml, load_yaml
from risk_model_workbench.paths import project_config_path
from risk_model_workbench.request.data_source import LOCAL_FEATHER, resolve_data_source_mode, sample_location as request_sample_location


RUNTIME_CONFIG_DIR = "configs_runtime"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in _as_list(value) if str(item) != ""]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _load_optional_yaml(path: Path) -> dict[str, Any]:
    return load_yaml(path) if path.exists() else {}


def _project_config(project_dir: Path) -> dict[str, Any]:
    path = project_config_path(project_dir)
    return load_yaml(path) if path.exists() else {}


def _config(project_dir: Path, name: str) -> dict[str, Any]:
    for suffix in [".yaml", ".yml"]:
        path = project_dir / "configs" / f"{name}{suffix}"
        if path.exists():
            return load_yaml(path)
    return {}


def _split_values(metadata: dict[str, Any], project_cfg: dict[str, Any], key: str) -> list[str]:
    splits = metadata.get("splits") if isinstance(metadata.get("splits"), dict) else {}
    if isinstance(splits.get(key), dict):
        values = _string_list(splits[key].get("values"))
        if values:
            return values
    if key == "dev":
        return _string_list(project_cfg.get("split", {}).get("ins_values") or ["DEV"])
    if key == "oos":
        return _string_list(project_cfg.get("split", {}).get("oos_values") or ["DEV-OOS"])
    return _string_list(project_cfg.get("split", {}).get("oot_values") or ["OOT"])


def _looks_like_local_data(value: str) -> bool:
    lowered = value.lower()
    return lowered.endswith((".feather", ".parquet", ".csv")) or "/" in value


def _step_params(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = metadata.get("step_params") or {}
    return raw if isinstance(raw, dict) else {}


def _param(metadata: dict[str, Any], step: str, key: str, default: Any = None) -> Any:
    return _step_params(metadata).get(step, {}).get(key, default)


def _experiments(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    raw = metadata.get("experiments")
    if isinstance(raw, list) and raw:
        return [item if isinstance(item, dict) else {"name": str(item)} for item in raw]
    description = str(metadata.get("experiment_description") or "").strip()
    if description:
        return [{"name": "baseline_from_description", "method": "lightgbm", "segment": "all", "description": description}]
    return []


def _feature_round_names(metadata: dict[str, Any]) -> list[str]:
    feature_cfg = metadata.get("feature_selection") if isinstance(metadata.get("feature_selection"), dict) else {}
    rounds = _as_list(feature_cfg.get("rounds"))
    return [str(item.get("name") if isinstance(item, dict) else item).replace("-", "_") for item in rounds]


def _normal_algorithm(value: Any, default: str = "lightgbm") -> str:
    raw = str(value or default).strip().lower()
    aliases = {
        "lgb": "lightgbm",
        "lgbm": "lightgbm",
        "xgb": "xgboost",
        "lr": "logistic_regression",
        "logistic": "logistic_regression",
        "hier_ranknet": "hier_ranknet",
        "ranknet": "hier_ranknet",
        "teacher_student": "teacher_student_distillation",
    }
    return aliases.get(raw, raw)


def _segment_filter(project_cfg: dict[str, Any], segment: str) -> str | None:
    if segment in {"", "all"}:
        return None
    for item in project_cfg.get("segments") or []:
        if isinstance(item, dict) and item.get("name") == segment:
            return item.get("filter")
    return None


def _score_columns(metadata: dict[str, Any], project_cfg: dict[str, Any]) -> list[str]:
    champions = _string_list((metadata.get("evaluation") or {}).get("champions"))
    if not champions:
        champions_cfg = project_cfg.get("champions") or {}
        champions = _string_list(champions_cfg.get("score_columns") if isinstance(champions_cfg, dict) else champions_cfg)
    return list(dict.fromkeys(["model_score", *champions]))


def _write_runtime_config(run_dir: Path, name: str, payload: dict[str, Any]) -> Path:
    path = run_dir / RUNTIME_CONFIG_DIR / name
    dump_yaml(payload, path)
    return path


def materialize_request_runtime_configs(
    *,
    request_doc: dict[str, Any],
    project_dir: str | Path,
    run_dir: str | Path,
    plan: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write request-derived runtime configs into a run workspace.

    The project workspace remains unchanged. Runtime configs are intentionally
    normal YAML files so every stage can consume them through the same config
    loaders used for project-level defaults.
    """
    project_path = Path(project_dir).resolve()
    run_path = Path(run_dir).resolve()
    metadata = request_doc.get("metadata", {})
    project_cfg = _project_config(project_path)
    feature_select_cfg = _config(project_path, "feature_select")
    refine_cfg = _config(project_path, "refine_features")
    train_cfg = _config(project_path, "train")
    evaluate_cfg = _config(project_path, "evaluate")
    report_cfg = _config(project_path, "report")
    sample_cfg = _config(project_path, "sample")

    dev_values = _split_values(metadata, project_cfg, "dev")
    oos_values = _split_values(metadata, project_cfg, "oos")
    oot_values = _split_values(metadata, project_cfg, "oot")
    id_columns = _string_list(metadata.get("id_columns") or project_cfg.get("data", {}).get("id_columns"))
    target_column = str(metadata.get("target_column") or project_cfg.get("data", {}).get("target_column") or "")
    split_column = str(metadata.get("split_column") or project_cfg.get("data", {}).get("split_column") or project_cfg.get("split", {}).get("source_column") or "")
    time_column = str(metadata.get("time_column") or project_cfg.get("data", {}).get("time_column") or "")
    period_column = str(metadata.get("period_column") or project_cfg.get("data", {}).get("period_column") or "")

    data_override: dict[str, Any] = {
        "id_columns": id_columns,
        "target_column": target_column,
        "split_column": split_column,
    }
    if time_column:
        data_override["time_column"] = time_column
    if period_column:
        data_override["period_column"] = period_column
    sample_location = request_sample_location(metadata)
    data_source_mode = resolve_data_source_mode(metadata)
    run_feature_selection_dir = run_path / "feature_selection"
    prescreen_output_dir = run_feature_selection_dir / "prescreen"
    prescreen_remain_features_path = prescreen_output_dir / "results" / "prescreen_final_remain_features.json"
    wide_sql_output_path = run_path / "queries" / "06_build_prescreen_wide_table.sql"
    wide_feature_map_path = run_feature_selection_dir / "prescreen_wide_feature_map.csv"
    wide_sql_summary_path = run_feature_selection_dir / "wide_sql_summary.json"
    refine_output_dir = run_feature_selection_dir / "refine"
    refine_dp_data_dir = run_path / "data" / "dp_feather" / "feature_refine"
    refine_dp_metadata_dir = run_feature_selection_dir / "dp_feather_datasets" / "feature_refine"
    prescreen_dp_data_dir = run_path / "data" / "dp_feather" / "feature_prescreen"
    prescreen_dp_metadata_dir = run_feature_selection_dir / "dp_feather_datasets" / "feature_prescreen"
    if sample_location:
        if data_source_mode == LOCAL_FEATHER:
            data_override["raw_path"] = sample_location
            data_override["source_table"] = None
        else:
            data_override["source_table"] = sample_location
            data_override["raw_path"] = None
    if metadata.get("sample_definition"):
        data_override["label_definition"] = metadata.get("sample_definition")
    eval_meta = metadata.get("evaluation") or {}
    segment_columns = list(
        dict.fromkeys(
            _string_list(project_cfg.get("data", {}).get("segment_columns"))
            + _string_list(eval_meta.get("comparison_dimensions"))
            + _string_list(eval_meta.get("risk_profile_dimensions"))
        )
    )
    if segment_columns:
        data_override["segment_columns"] = segment_columns

    runtime_project = _deep_merge(
        project_cfg,
        {
            "data": data_override,
            "split": {
                "source_column": split_column,
                "ins_values": dev_values,
                "oos_values": oos_values,
                "oot_values": oot_values,
            },
            "splits": {
                "dev": {"values": dev_values},
                "oos": {"values": oos_values},
                "oot": {"values": oot_values},
            },
            "champions": {"score_columns": _score_columns(metadata, project_cfg)[1:]},
            "request": {
                "request_id": metadata.get("request_id"),
                "title": metadata.get("title"),
                "workflow": metadata.get("workflow"),
                "data_source_mode": data_source_mode,
                "sample_location": sample_location,
                "business_domain": metadata.get("business_domain"),
                "scenario_profile": metadata.get("scenario_profile") or (plan or {}).get("scenario_profile"),
            },
        },
    )

    feature_thresholds = {}
    missing_threshold = _param(metadata, "missing_rate_filter", "threshold")
    if missing_threshold not in (None, ""):
        feature_thresholds["empty"] = float(missing_threshold)
    iv_min = _param(metadata, "iv_filter", "min_iv")
    if iv_min not in (None, ""):
        feature_thresholds["iv"] = float(iv_min)
    psi_max = _param(metadata, "psi_filter", "max_psi")
    if psi_max not in (None, ""):
        feature_thresholds["psi"] = float(psi_max)
    corr_max = _param(metadata, "correlation_dedup", "max_abs_corr")
    if corr_max not in (None, ""):
        feature_thresholds["corr"] = float(corr_max)

    wide_table_override: dict[str, Any] = {
        "join_keys": id_columns,
        "remain_features_path": str(prescreen_remain_features_path),
        "sql_output": str(wide_sql_output_path),
        "feature_map_output": str(wide_feature_map_path),
        "summary_output": str(wide_sql_summary_path),
    }
    if runtime_project.get("data", {}).get("source_table"):
        wide_table_override["base_table"] = runtime_project["data"]["source_table"]

    feature_select_override = {
        "feature_select": {
            "thresholds": feature_thresholds,
            "prescreen": {
                "output_dir": str(prescreen_output_dir),
                "target_col": target_column,
                "split_col": split_column,
                "train_value": dev_values[0] if dev_values else "DEV",
                "valid_value": oot_values[0] if oot_values else "OOT",
                "thresholds": feature_thresholds,
                "sampling": {"partition_col": period_column or None},
                "dp_feather": {
                    "data_dir": str(prescreen_dp_data_dir),
                    "metadata_dir": str(prescreen_dp_metadata_dir),
                },
            },
            "wide_table": wide_table_override,
            "runtime_request": {
                "data_source_mode": data_source_mode,
                "sample_location": sample_location,
                "stage_steps": (plan or {}).get("stage_steps") or metadata.get("stage_steps") or {},
                "step_params": (plan or {}).get("step_params") or metadata.get("step_params") or {},
            },
        }
    }
    runtime_feature_select = _deep_merge(feature_select_cfg, feature_select_override)

    max_unique = _param(metadata, "constant_value_filter", "max_unique_values")
    min_non_null_rate = None
    if missing_threshold not in (None, ""):
        min_non_null_rate = max(0.0, min(1.0, 1.0 - float(missing_threshold)))
    refine_override: dict[str, Any] = {
        "feature_refine": {
            "output_dir": str(refine_output_dir),
            "input": {
                "label_column": target_column,
                "split_column": split_column,
                "train_value": dev_values[0] if dev_values else "DEV",
                "valid_value": oot_values[0] if oot_values else "OOT",
                "id_columns": id_columns,
            },
            "preprocessing": {},
            "global_corr": {
                "enabled": "correlation_dedup" in ((plan or {}).get("step_params") or metadata.get("step_params") or {}),
            },
            "d03_random_importance": {"enabled": "random_noise_importance" in ((plan or {}).get("step_params") or metadata.get("step_params") or {})},
            "d04_null_importance": {"enabled": "null_importance_filter" in ((plan or {}).get("step_params") or metadata.get("step_params") or {})},
            "d05_baseline_importance": {"enabled": "baseline_importance_filter" in ((plan or {}).get("step_params") or metadata.get("step_params") or {})},
            "dp_feather": {
                "data_dir": str(refine_dp_data_dir),
                "metadata_dir": str(refine_dp_metadata_dir),
            },
            "runtime_request": {
                "data_source_mode": data_source_mode,
                "sample_location": sample_location,
                "step_params": (plan or {}).get("step_params") or metadata.get("step_params") or {},
            },
        }
    }
    if min_non_null_rate is not None:
        refine_override["feature_refine"]["preprocessing"]["min_non_null_rate"] = min_non_null_rate
    if max_unique not in (None, ""):
        refine_override["feature_refine"]["preprocessing"]["max_unique_values"] = int(max_unique)
    if corr_max not in (None, ""):
        refine_override["feature_refine"]["global_corr"]["threshold"] = float(corr_max)
    corr_method = _param(metadata, "correlation_dedup", "method")
    if corr_method:
        refine_override["feature_refine"]["global_corr"]["method"] = corr_method
    step_params = (plan or {}).get("step_params") or metadata.get("step_params") or {}
    for step_name, config_key in [
        ("correlation_dedup", "global_corr"),
        ("random_noise_importance", "d03_random_importance"),
        ("null_importance_filter", "d04_null_importance"),
        ("baseline_importance_filter", "d05_baseline_importance"),
        ("gain_importance_filter", "d05_baseline_importance"),
    ]:
        if isinstance(step_params.get(step_name), dict):
            refine_override["feature_refine"].setdefault(config_key, {}).update(step_params[step_name])
    keep_top = _param(metadata, "baseline_importance_filter", "keep_top_n")
    if keep_top not in (None, ""):
        refine_override["feature_refine"]["d05_baseline_importance"]["keep_top_n"] = int(keep_top)
        refine_override["feature_refine"]["target_feature_count"] = int(keep_top)
    feature_round_names = _feature_round_names(metadata)
    direct_refine_remote_table = (
        data_source_mode != LOCAL_FEATHER
        and bool(sample_location)
        and feature_round_names
        and "refine" in feature_round_names
        and not any(round_name in {"prescreen", "feature_prescreen", "coarse_screening", "coarse", "d01_d02", "d01d02", "build_wide_sql", "wide_sql", "build_wide"} for round_name in feature_round_names)
    )
    if direct_refine_remote_table:
        refine_override["feature_refine"].setdefault("input", {})["wide_table"] = sample_location
        refine_override["feature_refine"].setdefault("input", {})["feature_map"] = None
    elif data_source_mode != LOCAL_FEATHER:
        refine_input = refine_override["feature_refine"].setdefault("input", {})
        refine_input["feature_map"] = str(wide_feature_map_path)
        output_table = (runtime_feature_select.get("feature_select", {}).get("wide_table", {}) or {}).get("output_table")
        if output_table:
            refine_input["wide_table"] = output_table
    if sample_location and data_source_mode == LOCAL_FEATHER:
        refine_input = refine_override["feature_refine"].setdefault("input", {})
        refine_input["local_feather_path"] = sample_location
        # In local_feather mode the wide table already exists as the feather file.
        # Clear any deep-merged remote-flow pointers so load_feature_list falls back
        # to reading candidate feature columns from local_feather_path instead of a
        # stale/non-existent prescreen feature_map.
        refine_input["feature_map"] = None
        refine_input["wide_table"] = None
    runtime_refine = _deep_merge(refine_cfg, refine_override)

    experiments = []
    for item in _experiments(metadata):
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        segment = str(item.get("segment") or "all")
        method = _normal_algorithm(item.get("method") or item.get("algorithm"), train_cfg.get("training", {}).get("default_algorithm", "lightgbm"))
        experiments.append(
            {
                "name": name,
                "display_name": item.get("description") or name,
                "segment": segment,
                "segment_filter": _segment_filter(project_cfg, segment),
                "algorithm": method,
                "method": method,
                "description": item.get("description", ""),
                "sample_weight": item.get("sample_weight"),
            }
        )
    training_override = {
        "training": {
            "label_column": target_column,
            "split_column": split_column,
            "train_values": dev_values,
            # Early-stopping validation must NOT use OOT: OOT is the held-out time
            # window for final generalization evaluation, and using it to pick
            # best_iter leaks into (and over-optimistically inflates) the OOT AUC.
            # Prefer the in-sample OOS split (DEV-OOS) for early stopping.
            "valid_values": oos_values[:1] or oot_values[:1],
            "oos_values": oos_values + oot_values,
            "experiments": experiments,
            "candidate_targets": _string_list(metadata.get("candidate_targets")),
            "sample_variants": _string_list(metadata.get("sample_variants")),
            "feature_list_path": str(run_path / "feature_selection" / "final_features.txt"),
            "runtime_step_params": (plan or {}).get("step_params") or metadata.get("step_params") or {},
        },
        "input": {
            "id_columns": id_columns,
            "label_column": target_column,
            "split_column": split_column,
            "historical_score_columns": _score_columns(metadata, project_cfg)[1:],
        },
    }
    request_training = metadata.get("training") if isinstance(metadata.get("training"), dict) else {}
    if request_training:
        training_override["training"] = _deep_merge(request_training, training_override["training"])
    if sample_location and data_source_mode == LOCAL_FEATHER:
        training_override["input"]["feather_path"] = sample_location
    runtime_train = _deep_merge(train_cfg, training_override)

    runtime_evaluate = _deep_merge(
        evaluate_cfg,
        {
            "evaluation": {
                "label_column": target_column,
                "split_column": split_column,
                "time_column": time_column,
                "period_column": period_column,
                "score_columns": _score_columns(metadata, project_cfg),
                "comparison_dimensions": _string_list(eval_meta.get("comparison_dimensions")),
                "risk_profile_dimensions": _string_list(eval_meta.get("risk_profile_dimensions")),
                "segment_columns": list(
                    dict.fromkeys(
                        _string_list((evaluate_cfg.get("evaluation") or {}).get("segment_columns"))
                        + _string_list(eval_meta.get("comparison_dimensions"))
                        + _string_list(eval_meta.get("risk_profile_dimensions"))
                    )
                ),
            },
            "metrics": _string_list(eval_meta.get("metrics")),
            "runtime_step_params": (plan or {}).get("step_params") or metadata.get("step_params") or {},
        },
    )

    report_meta = metadata.get("reports") or {}
    report_stage_steps = ((plan or {}).get("stage_steps") or metadata.get("stage_steps") or {}).get("report", [])
    report_outputs = _string_list(report_meta.get("outputs"))
    if "model_recovery_report" in report_stage_steps and "model_recovery_report.md" not in report_outputs:
        report_outputs.append("model_recovery_report.md")
    if "credit_product_report" in report_stage_steps and "credit_product_report.md" not in report_outputs:
        report_outputs.append("credit_product_report.md")
    runtime_report = _deep_merge(
        report_cfg,
        {
            "report": {
                "sections": _string_list(report_meta.get("sections")),
                "outputs": report_outputs,
                "output_formats": _infer_output_formats(report_outputs),
                "stage_steps": report_stage_steps,
            },
            "sections": _string_list(report_meta.get("sections")),
            "outputs": report_outputs,
        },
    )

    runtime_sample = _deep_merge(
        sample_cfg,
        {
            "sample": {
                "target_column": target_column,
                "id_columns": id_columns,
                "time_column": time_column,
                "period_column": period_column,
                "split_column": split_column,
                "splits": {"dev": dev_values, "oos": oos_values, "oot": oot_values},
                "definition": metadata.get("sample_definition", ""),
                "stage_steps": (plan or {}).get("stage_steps", {}).get("sample_check", []),
                "step_params": {
                    key: value
                    for key, value in ((plan or {}).get("step_params") or metadata.get("step_params") or {}).items()
                    if key in {"field_contract", "key_uniqueness", "monthly_label_distribution", "segment_distribution", "channel_distribution", "account_status_distribution"}
                },
            }
        },
    )

    payloads = {
        "project.yml": runtime_project,
        "sample.yaml": runtime_sample,
        "feature_select.yaml": runtime_feature_select,
        "refine_features.yaml": runtime_refine,
        "train.yaml": runtime_train,
        "evaluate.yaml": runtime_evaluate,
        "report.yaml": runtime_report,
        "request_runtime.yaml": {
            "request_id": metadata.get("request_id"),
            "request_path": request_doc.get("path"),
            "plan_id": (plan or {}).get("plan_id"),
            "workflow": metadata.get("workflow"),
            "data_source_mode": data_source_mode,
            "sample_location": sample_location,
            "materialized_configs": [
                "project.yml",
                "sample.yaml",
                "feature_select.yaml",
                "refine_features.yaml",
                "train.yaml",
                "evaluate.yaml",
                "report.yaml",
            ],
        },
    }
    return {name: _write_runtime_config(run_path, name, payload) for name, payload in payloads.items()}


def _infer_output_formats(outputs: Any) -> list[str]:
    values = _string_list(outputs)
    formats = []
    for item in values:
        suffix = Path(item).suffix.lower().lstrip(".")
        if suffix == "md":
            suffix = "markdown"
        if suffix and suffix not in formats:
            formats.append(suffix)
    return formats or ["markdown"]
