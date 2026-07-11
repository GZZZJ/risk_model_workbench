"""Application-facing model experiment orchestration."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from risk_model_workbench.application.context import VersionContext
from risk_model_workbench.config import load_yaml
from risk_model_workbench.harness.runtime import (
    classify_exception,
    register_action_artifact,
    stage_action_done,
    stage_action_failed,
    stage_action_started,
)
from risk_model_workbench.progress import ProgressReporter
from risk_model_workbench.request.training import merge_training_config, project_training_defaults
from risk_model_workbench.state import append_decision


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _runtime_config(
    context: VersionContext, name: str, explicit: object = None
) -> tuple[dict[str, Any], Path]:
    if explicit:
        path = Path(str(explicit))
        if not path.is_absolute():
            path = context.project_dir / path
        return (load_yaml(path) if path.exists() else {}), path
    for path in [
        context.runtime_config_dir / f"{name}.yaml",
        context.runtime_config_dir / f"{name}.yml",
        context.project_dir / "configs" / f"{name}.yaml",
        context.project_dir / "configs" / f"{name}.yml",
    ]:
        if path.exists():
            return load_yaml(path), path
    return {}, context.project_dir / "configs" / f"{name}.yaml"


def _project_config(context: VersionContext) -> dict[str, Any]:
    for path in [context.runtime_config_dir / "project.yml", context.project_dir / "project.yml"]:
        if path.exists():
            return load_yaml(path)
    return {}


def _normal_algorithm(value: Any) -> str:
    normalized = str(value or "lightgbm").strip().lower().replace("-", "_")
    return {
        "lgb": "lightgbm",
        "lgbm": "lightgbm",
        "logit": "logistic_regression",
        "logistic": "logistic_regression",
        "lr": "logistic_regression",
        "xgb": "xgboost",
    }.get(normalized, normalized)


def _experiment(config: dict[str, Any], name: str) -> dict[str, Any]:
    experiments = config.get("experiments") or (config.get("training") or {}).get("experiments") or []
    if isinstance(experiments, dict):
        value = experiments.get(name)
        if isinstance(value, dict):
            return {"name": name, **value}
    if isinstance(experiments, list):
        for value in experiments:
            if isinstance(value, dict) and str(value.get("name")) == name:
                return value
    training = config.get("training") or {}
    return {"name": name, "algorithm": training.get("default_algorithm", "lightgbm")}


def _status(
    output: Path,
    *,
    status: str,
    experiment: str,
    algorithm: str,
    message: str,
    failure_code: str = "",
    error: Exception | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    previous = _read_json(output / "training_status.json")
    now = datetime.now().isoformat(timespec="seconds")
    payload = {
        "status": status,
        "experiment": experiment,
        "algorithm": algorithm,
        "pid": os.getpid(),
        "started_at": previous.get("started_at") or now,
        "heartbeat_at": now,
        "completed_at": now if status in {"done", "failed", "scaffold", "advisor_required"} else "",
        "percent": 100 if status in {"done", "scaffold"} else None,
        "message": message,
        "failure_code": failure_code,
        "error_type": type(error).__name__ if error else "",
        "error_message": str(error) if error else "",
        "metrics": metrics or {},
    }
    _write_json(output / "training_status.json", payload)
    return payload


def _summary(output: Path, status: dict[str, Any], plan: dict[str, Any], metrics: dict[str, Any] | None = None) -> Path:
    metrics = metrics or {}
    lines = [
        "# Training Summary",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Status | {status['status']} |",
        f"| Experiment | {status['experiment']} |",
        f"| Algorithm | {status['algorithm']} |",
        f"| Valid AUC | {metrics.get('valid_auc', '')} |",
        f"| Valid KS | {metrics.get('valid_ks', '')} |",
        f"| Input feather | {plan['input_feather']['path']} |",
        f"| Feature list | {plan['feature_list']['path']} |",
    ]
    path = output / "training_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _register(context: VersionContext, experiment: str, names: list[str]) -> None:
    for name in names:
        path = context.workspace / "modeling" / experiment / name
        if path.exists():
            register_action_artifact(context.workspace, "train_baseline", path)


def run_experiment(context: VersionContext, params: dict[str, Any]) -> None:
    """Execute one typed training experiment and persist semantic stage evidence."""
    experiment = str(params.get("experiment") or "")
    stage_action_started(context.workspace, "train_baseline")
    reporter = ProgressReporter(context.workspace, "train_baseline")
    train_config, config_path = _runtime_config(context, "train", params.get("config"))
    project_config = _project_config(context)
    effective = deepcopy(train_config)
    defaults = project_training_defaults(project_config)
    if defaults:
        effective["training"] = merge_training_config(defaults, effective.get("training", {}))
    runtime_experiment = _experiment(effective, experiment)
    algorithm = _normal_algorithm(runtime_experiment.get("algorithm") or runtime_experiment.get("method"))
    effective["runtime_experiment"] = runtime_experiment
    effective["runtime_step_params"] = (effective.get("training") or {}).get("runtime_step_params", {})
    input_cfg = effective.setdefault("input", {})
    data_cfg = project_config.get("data") or {}
    for key in ["time_column", "period_column", "segment_columns"]:
        if data_cfg.get(key):
            input_cfg.setdefault(key, data_cfg[key])
    training_cfg = effective.get("training") or {}
    input_feather = Path(str(params.get("input_feather") or input_cfg.get("feather_path") or ""))
    feature_list = Path(str(params.get("feature_list") or training_cfg.get("feature_list_path") or "runs/modeling_feature_set/feature_list.txt"))
    if not input_feather.is_absolute():
        input_feather = context.project_dir / input_feather
    if not feature_list.is_absolute():
        feature_list = context.project_dir / feature_list
    output = context.workspace / "modeling" / experiment
    output.mkdir(parents=True, exist_ok=True)
    score_output = Path(str(params.get("score_output") or output / "scores_all_splits.feather"))
    input_dir = Path(str(params.get("input_dir") or context.workspace / "modeling_input"))
    if not score_output.is_absolute():
        # Relative scored-row outputs are version evidence, not project-global
        # exports. Resolve before trainer side effects so registration cannot
        # fail after writing outside the auditable workspace.
        score_output = context.workspace / score_output
    if not input_dir.is_absolute():
        input_dir = context.project_dir / input_dir
    plan = {
        "status": "ready" if input_feather.is_file() and feature_list.is_file() and bool(train_config) else "not_ready",
        "workspace": str(context.workspace),
        "experiment": experiment,
        "algorithm": algorithm,
        "config": {"path": str(config_path), "exists": config_path.exists()},
        "input_feather": {"path": str(input_feather), "exists": input_feather.exists(), "is_file": input_feather.is_file()},
        "feature_list": {
            "path": str(feature_list),
            "exists": feature_list.exists(),
            "is_file": feature_list.is_file(),
            "feature_count": sum(1 for line in feature_list.read_text(encoding="utf-8").splitlines() if line.strip()) if feature_list.is_file() else None,
        },
        "outputs": {"score_output": str(score_output), "input_snapshot_dir": str(input_dir), "modeling_dir": str(output)},
        "splits": {
            "train_values": training_cfg.get("train_values", ["DEV"]),
            "valid_values": training_cfg.get("valid_values", ["DEV-OOS"]),
            "oos_values": training_cfg.get("oos_values", ["DEV-OOS", "OOT-OOS"]),
            "skip_split_check": bool(params.get("skip_split_check")),
        },
        "training_mode": str(training_cfg.get("mode") or (training_cfg.get("tuning") or {}).get("mode") or "single_train"),
        "runtime_experiment": runtime_experiment,
        "will_train": input_feather.is_file() and feature_list.is_file() and bool(train_config),
    }
    raw_valid = training_cfg.get("valid_values")
    valid_values = raw_valid if isinstance(raw_valid, list) else ([] if raw_valid is None else [raw_valid])
    if valid_values:
        from risk_model_workbench.request.splits import resolve_split_values

        oot_values = set(resolve_split_values({}, project_config)["oot"]["values"])
        polluting = sorted(set(map(str, valid_values)) & oot_values)
        if polluting:
            reason = (
                f"training.valid_values {list(map(str, valid_values))} 含时间外标签 {polluting} "
                f"(oot={sorted(oot_values)})；早停验证集不得用时间外样本，会污染时间外评估。"
            )
            if params.get("skip_split_check") is True:
                print(f"warning (split, suppressed by --skip-split-check): {reason}")
            else:
                _write_json(output / "train_metrics.json", {"status": "failed", "reason": reason, "experiment": experiment, "algorithm": algorithm})
                status = _status(output, status="failed", experiment=experiment, algorithm=algorithm, message=reason, failure_code="data_missing")
                _summary(output, status, plan)
                _register(context, experiment, ["train_metrics.json", "training_status.json", "training_summary.md"])
                stage_action_failed(context.workspace, "train_baseline", reason, failure_code="data_missing")
                return
    if params.get("plan_only") is True:
        _write_json(output / "train_plan.json", plan)
        (output / "train_plan.md").write_text(f"# Training Plan\n\nWill train: {plan['will_train']}\n", encoding="utf-8")
        _register(context, experiment, ["train_plan.json", "train_plan.md"])
        append_decision(context.workspace, stage="train_baseline", decision="plan_only", reason="training plan preview generated; model training was not executed")
        stage_action_done(context.workspace, "train_baseline", scaffold=True, message="training plan preview only")
        return
    if algorithm == "custom" and not (
        (train_config.get("custom_training") or {}).get("entrypoint")
        or (train_config.get("training") or {}).get("custom_entrypoint")
    ):
        reason = "custom training requires training.custom_entrypoint or custom_training.entrypoint in project/runtime config"
        _write_json(output / "train_metrics.json", {"status": "failed", "reason": reason, "experiment": experiment, "algorithm": algorithm})
        status = _status(output, status="failed", experiment=experiment, algorithm=algorithm, message=reason, failure_code="data_missing")
        _summary(output, status, plan)
        _register(context, experiment, ["train_metrics.json", "training_status.json", "training_summary.md"])
        stage_action_failed(context.workspace, "train_baseline", reason, failure_code="data_missing")
        return
    if not plan["will_train"]:
        reason = "training data not available"
        payload = {"status": "scaffold", "reason": reason, "experiment": experiment, "algorithm": algorithm}
        _write_json(output / "train_metrics.json", payload)
        status = _status(output, status="scaffold", experiment=experiment, algorithm=algorithm, message=reason, failure_code="scaffold_only")
        _summary(output, status, plan)
        _register(context, experiment, ["train_metrics.json", "training_status.json", "training_summary.md"])
        append_decision(context.workspace, stage="train_baseline", decision="scaffold", reason=reason)
        stage_action_done(context.workspace, "train_baseline", scaffold=True, message=reason)
        reporter.emit(step="train_scaffold", status="scaffold", message=reason, percent=100)
        return
    try:
        _status(output, status="running", experiment=experiment, algorithm=algorithm, message="training started")
        if algorithm == "lightgbm":
            from risk_model_workbench.modeling.train_lgb import train_lightgbm_from_feather

            metrics = train_lightgbm_from_feather(input_feather=input_feather, feature_list_path=feature_list, output_dir=output, score_output=score_output, input_snapshot_dir=input_dir, config=effective, progress=reporter)
        else:
            from risk_model_workbench.modeling.train_xgb import train_tabular_from_feather

            metrics = train_tabular_from_feather(input_feather=input_feather, feature_list_path=feature_list, output_dir=output, score_output=score_output, input_snapshot_dir=input_dir, config=effective, algorithm=algorithm, progress=reporter)
        _write_json(output / "train_metrics.json", {"status": "done", "metrics": metrics, "experiment": experiment, "algorithm": algorithm})
        status = _status(output, status="done", experiment=experiment, algorithm=algorithm, message="training completed", metrics=metrics)
        _summary(output, status, plan, metrics)
        artifact_names = [
            "train_metrics.json", "training_status.json", "training_summary.md",
            "metrics_train_valid.json", "feature_importance.csv", "feature_drop_detail.csv",
            "actual_feature_list.txt", "preprocessing.json", "run_config.json", "model.pkl",
            "score_column_summary.csv", "distillation_summary.json", "tuning_context.json",
            "tuning_trials.csv", "tuning_summary.json", "best_params.json",
            "selection_reason.md", "llm_tuning_decisions.md",
        ]
        _register(context, experiment, artifact_names)
        for artifact in sorted(output.glob("llm_tuning_plan_round_*.json")) + sorted(output.glob("tuning_context_round_*.json")):
            register_action_artifact(context.workspace, "train_baseline", artifact)
        if score_output.exists():
            register_action_artifact(
                context.workspace,
                "train_baseline",
                score_output,
                storage_class="local_only",
                contract_role="optional",
                retention_reason="Row-level scored dataset is intentionally retained only in the local version workspace.",
                regeneration="Rerun the train action for this version and experiment from the approved local feature source.",
            )
        woe_dir = output / "woe_top_features"
        if woe_dir.exists():
            for artifact in sorted([*woe_dir.glob("*.csv"), *woe_dir.glob("images/*.png")]):
                register_action_artifact(context.workspace, "train_baseline", artifact)
        append_decision(context.workspace, stage="train_baseline", decision="done", reason=f"{algorithm} training completed from local feather data")
        stage_action_done(context.workspace, "train_baseline")
    except Exception as exc:
        from risk_model_workbench.modeling.llm_tuning import HostAgentTuningPlanRequired

        advisor = isinstance(exc, HostAgentTuningPlanRequired)
        status_name = "advisor_required" if advisor else "failed"
        failure_code = "advisor_required" if advisor else classify_exception(exc)
        _write_json(output / "train_metrics.json", {"status": status_name, "reason": str(exc), "experiment": experiment, "algorithm": algorithm})
        status = _status(output, status=status_name, experiment=experiment, algorithm=algorithm, message=str(exc), failure_code=failure_code, error=exc)
        _summary(output, status, plan)
        _register(context, experiment, ["train_metrics.json", "training_status.json", "training_summary.md"])
        if advisor:
            for artifact in sorted(output.glob("tuning_context*.json")):
                register_action_artifact(context.workspace, "train_baseline", artifact)
        append_decision(context.workspace, stage="train_baseline", decision=status_name, reason=str(exc))
        stage_action_failed(context.workspace, "train_baseline", str(exc), failure_code=failure_code)


__all__ = ["run_experiment"]
