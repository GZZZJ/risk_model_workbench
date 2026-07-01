"""LightGBM training utilities for binary business models."""

from __future__ import annotations

import json
import pickle
import re
import time
from pathlib import Path
from typing import Any


def _workbench_git_commit() -> dict[str, Any]:
    """Best-effort code provenance for the workbench build that produced a model.

    Lets a run's run_config.json trace model.pkl back to the exact workbench
    code version. Returns commit=None when not run inside a git worktree.
    """
    import subprocess

    repo = Path(__file__).resolve().parent
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=repo, stderr=subprocess.DEVNULL, text=True
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def _file_sha256(path: Path) -> str | None:
    """Streaming SHA-256 of a (possibly large) input file; None if unreadable."""
    import hashlib

    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return None


def coerce_features(
    df,
    features: list[str],
    sentinels: list[int],
    min_non_null_rate: float,
    drop_constant: bool,
    max_unique_values: int = 1,
):
    """Coerce configured features to numeric and drop unusable columns."""
    import numpy as np
    import pandas as pd

    available = [feature for feature in features if feature in df.columns]
    missing = [feature for feature in features if feature not in df.columns]
    x = df.loc[:, available].copy()
    kept: list[str] = []
    stats: list[dict[str, Any]] = []

    for feature in available:
        series = pd.to_numeric(x[feature], errors="coerce")
        if sentinels:
            series = series.replace(sentinels, np.nan)
        series = series.replace([np.inf, -np.inf], np.nan)
        non_null_rate = float(series.notna().mean())
        unique_count = int(series.nunique(dropna=True))
        drop_reason = ""
        if non_null_rate < min_non_null_rate:
            drop_reason = "low_non_null_rate"
        elif drop_constant and unique_count <= max_unique_values:
            drop_reason = "constant"
        else:
            kept.append(feature)
            x[feature] = series
        stats.append(
            {
                "feature": feature,
                "non_null_rate": non_null_rate,
                "unique_count": unique_count,
                "drop_reason": drop_reason,
            }
        )

    for feature in missing:
        stats.append({"feature": feature, "non_null_rate": 0.0, "unique_count": 0, "drop_reason": "missing_from_data"})

    return x.loc[:, kept], kept, pd.DataFrame(stats)


def fill_na_from_train(train_x, valid_x, *other_frames):
    """Fill missing values with train-set medians and fallback zero."""
    medians = train_x.median(numeric_only=True).replace([float("inf"), float("-inf")], None).fillna(0)
    result = [train_x.fillna(medians).fillna(0), valid_x.fillna(medians).fillna(0)]
    for frame in other_frames:
        result.append(frame.fillna(medians).fillna(0))
    return tuple(result), medians


def train_lightgbm_from_feather(
    *,
    input_feather: str | Path,
    feature_list_path: str | Path,
    output_dir: str | Path,
    score_output: str | Path,
    input_snapshot_dir: str | Path,
    config: dict[str, Any],
    progress: Any | None = None,
) -> dict[str, Any]:
    """Train a LightGBM model and score all splits.

    Heavy dependencies are imported inside this function so basic CLI smoke tests
    do not require the full modeling stack.
    """
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from scipy.stats import ks_2samp
    from sklearn.metrics import roc_auc_score
    from risk_model_workbench.modeling.llm_tuning import (
        TRAIN_CONTROL_PARAMS,
        build_tuning_context,
        llm_guided_tuning_enabled,
        public_lgb_params,
        resolve_tuning_config,
        select_best_trial,
        suggest_lgb_candidates,
        write_selection_reason,
    )

    input_feather = Path(input_feather)
    feature_list_path = Path(feature_list_path)
    output_dir = Path(output_dir)
    score_output = Path(score_output)
    input_snapshot_dir = Path(input_snapshot_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    score_output.parent.mkdir(parents=True, exist_ok=True)
    input_snapshot_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = config["training"]
    input_cfg = config["input"]
    lgb_cfg = config["lightgbm"]
    preproc_cfg = config.get("preprocessing", {})
    runtime_experiment = config.get("runtime_experiment", {})

    candidate_features = [line.strip() for line in feature_list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if progress:
        progress.emit(step="load_feature_list", message=f"训练特征列表读取完成，共 {len(candidate_features)} 个候选变量", percent=8)
    id_cols = input_cfg.get("id_columns", ["uid", "mdl_dte"])
    base_cols = input_cfg.get("base_columns", [])
    configured_historical_scores = input_cfg.get("historical_score_columns", [])
    label_col = input_cfg["label_column"]
    split_col = input_cfg["split_column"]
    segment_filter = str(runtime_experiment.get("segment_filter") or "").strip()
    segment_tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", segment_filter)
    segment_cols = input_cfg.get("segment_columns", [])
    time_cols = [input_cfg.get("time_column"), input_cfg.get("period_column")]
    read_cols = list(
        dict.fromkeys(
            id_cols
            + base_cols
            + segment_cols
            + [token for token in segment_tokens if token not in {"in", "and", "or", "not", "True", "False"}]
            + configured_historical_scores
            + [item for item in time_cols if item]
            + [label_col, split_col]
            + candidate_features
        )
    )
    if progress:
        progress.emit(step="read_input_schema", message=f"正在读取训练数据字段：{input_feather}", percent=12)
    all_cols = pd.read_feather(input_feather, columns=None).columns.tolist()
    if progress:
        progress.emit(step="read_input_data", message=f"正在读取训练数据：{len(read_cols)} 个目标字段", percent=18)
    raw = pd.read_feather(input_feather, columns=[column for column in read_cols if column in all_cols])
    if segment_filter:
        before_rows = len(raw)
        raw = raw.query(segment_filter).copy()
        if progress:
            progress.emit(
                step="segment_filter",
                message=f"实验分群过滤完成：{before_rows} -> {len(raw)} 行",
                percent=22,
                metrics={"before_rows": int(before_rows), "after_rows": int(len(raw)), "segment_filter": segment_filter},
            )
    if progress:
        progress.emit(
            step="read_input_done",
            message=f"训练数据读取完成：{len(raw)} 行 {len(raw.columns)} 列",
            percent=25,
            metrics={"rows": int(len(raw)), "columns": int(len(raw.columns))},
        )

    train_values = train_cfg.get("train_values", ["DEV"])
    valid_values = train_cfg.get("valid_values", ["OOT"])
    oos_values = train_cfg.get("oos_values", ["DEV-OOS", "OOT-OOS"])
    train_mask = raw[split_col].isin(train_values) & raw[label_col].isin([0, 1])
    valid_mask = raw[split_col].isin(valid_values) & raw[label_col].isin([0, 1])

    sentinels = preproc_cfg.get("missing_sentinels", [-999, -998])
    min_non_null_rate = float(preproc_cfg.get("min_non_null_rate", 0.01))
    drop_constant = bool(preproc_cfg.get("drop_constant", True))
    max_unique_values = int(preproc_cfg.get("max_unique_values", 1))
    x_all, kept_features, drop_detail = coerce_features(raw, candidate_features, sentinels, min_non_null_rate, drop_constant, max_unique_values)
    if progress:
        progress.emit(
            step="preprocess_done",
            message=f"训练预处理完成：保留 {len(kept_features)}/{len(candidate_features)} 个变量",
            percent=38,
            metrics={
                "candidate_features": len(candidate_features),
                "kept_features": len(kept_features),
                "dropped_features": int((drop_detail["drop_reason"] != "").sum()),
            },
        )

    tr_x = x_all[train_mask].reset_index(drop=True)
    tr_y = raw.loc[train_mask, label_col].astype(int).reset_index(drop=True)
    va_x = x_all[valid_mask].reset_index(drop=True)
    va_y = raw.loc[valid_mask, label_col].astype(int).reset_index(drop=True)
    (tr_x, va_x), medians = fill_na_from_train(tr_x, va_x)

    preprocessing = {
        "candidate_feature_count": len(drop_detail),
        "kept_feature_count": len(kept_features),
        "dropped_feature_count": int((drop_detail["drop_reason"] != "").sum()),
        "missing_sentinels": sentinels,
        "min_non_null_rate": min_non_null_rate,
        "max_unique_values": max_unique_values,
        "fill_strategy": "train_median_fill_zero",
        "drop_reason_counts": drop_detail["drop_reason"].value_counts().to_dict(),
        "fill_values": {feature: float(value) for feature, value in medians.items()},
    }
    (output_dir / "preprocessing.json").write_text(json.dumps(preprocessing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "candidate_feature_list.txt").write_text("\n".join(candidate_features) + "\n", encoding="utf-8")
    (output_dir / "actual_feature_list.txt").write_text("\n".join(kept_features) + "\n", encoding="utf-8")
    drop_detail.to_csv(output_dir / "feature_drop_detail.csv", index=False, encoding="utf-8-sig")

    params = {
        "objective": lgb_cfg.get("objective", "binary"),
        "metric": lgb_cfg.get("metric", "auc"),
        "learning_rate": lgb_cfg.get("learning_rate", 0.05),
        "num_leaves": lgb_cfg.get("num_leaves", 31),
        "max_depth": lgb_cfg.get("max_depth", 5),
        "min_child_samples": lgb_cfg.get("min_child_samples", 100),
        "subsample": lgb_cfg.get("subsample", 0.7),
        "colsample_bytree": lgb_cfg.get("colsample_bytree", 0.7),
        "reg_alpha": lgb_cfg.get("reg_alpha", 0.1),
        "reg_lambda": lgb_cfg.get("reg_lambda", 1.0),
        "verbose": -1,
        "seed": train_cfg.get("random_seed", 0),
        "feature_fraction_seed": train_cfg.get("random_seed", 0),
        "bagging_seed": train_cfg.get("random_seed", 0),
    }
    optional_param_aliases = {
        "bagging_freq": ("bagging_freq", "subsample_freq"),
        "min_gain_to_split": ("min_gain_to_split", "min_split_gain"),
        "max_bin": ("max_bin",),
    }
    for target, aliases in optional_param_aliases.items():
        for alias in aliases:
            if alias in lgb_cfg:
                params[target] = lgb_cfg[alias]
                break
    scale_cfg = (config.get("runtime_step_params") or train_cfg.get("runtime_step_params") or {}).get("scale_pos_weight", {})
    if scale_cfg:
        positives = float((tr_y == 1).sum())
        negatives = float((tr_y == 0).sum())
        if positives > 0:
            params["scale_pos_weight"] = float(scale_cfg.get("value") or negatives / positives)
    train_ds = lgb.Dataset(tr_x, label=tr_y, feature_name=kept_features, free_raw_data=False)
    valid_ds = lgb.Dataset(va_x, label=va_y, feature_name=kept_features, reference=train_ds, free_raw_data=False)
    base_num_boost_round = int(lgb_cfg.get("num_boost_round", 1000))
    base_early_stopping_rounds = int(lgb_cfg.get("early_stopping_rounds", 50))

    def _split_trial_params(trial_params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
        trial_lgb_params = dict(params)
        control = {
            "num_boost_round": base_num_boost_round,
            "early_stopping_rounds": base_early_stopping_rounds,
        }
        for key, value in trial_params.items():
            if key in TRAIN_CONTROL_PARAMS:
                control[key] = int(value)
            else:
                trial_lgb_params[key] = value
        return trial_lgb_params, control

    def _fit_once(trial_params: dict[str, Any], *, log_period: int = 0) -> tuple[Any, int, dict[str, Any], dict[str, Any], dict[str, int]]:
        trial_lgb_params, control = _split_trial_params(trial_params)
        callbacks = [lgb.early_stopping(control["early_stopping_rounds"], verbose=False)]
        if log_period > 0:
            callbacks.append(lgb.log_evaluation(period=log_period))
        start_time = time.time()
        fitted = lgb.train(
            trial_lgb_params,
            train_ds,
            num_boost_round=control["num_boost_round"],
            valid_sets=[valid_ds],
            callbacks=callbacks,
        )
        iteration = int(fitted.best_iteration or control["num_boost_round"])
        tr_pred = fitted.predict(tr_x, num_iteration=iteration)
        va_pred = fitted.predict(va_x, num_iteration=iteration)
        trial_metrics = {
            "train_auc": float(roc_auc_score(tr_y, tr_pred)),
            "valid_auc": float(roc_auc_score(va_y, va_pred)),
            "train_ks": float(ks_2samp(tr_pred[tr_y == 1], tr_pred[tr_y == 0]).statistic),
            "valid_ks": float(ks_2samp(va_pred[va_y == 1], va_pred[va_y == 0]).statistic),
            "train_samples": int(len(tr_y)),
            "valid_samples": int(len(va_y)),
            "train_bad_rate": float(tr_y.mean()),
            "valid_bad_rate": float(va_y.mean()),
            "best_iteration": iteration,
            "train_time_seconds": round(time.time() - start_time, 1),
        }
        trial_metrics["auc_gap"] = trial_metrics["train_auc"] - trial_metrics["valid_auc"]
        return fitted, iteration, trial_metrics, trial_lgb_params, control

    def _trial_row(trial: dict[str, Any]) -> dict[str, Any]:
        row = {key: value for key, value in trial.items() if not key.startswith("_") and key != "params"}
        for key, value in (trial.get("params") or {}).items():
            row[f"param_{key}"] = value
        return row

    if progress:
        progress.emit(
            step="train_model",
            message=f"LightGBM 训练开始：训练样本 {len(tr_y)}，验证样本 {len(va_y)}，变量 {len(kept_features)} 个",
            percent=50,
            metrics={"train_samples": int(len(tr_y)), "valid_samples": int(len(va_y)), "features": len(kept_features)},
        )
    training_mode = "single_train"
    tuning_summary: dict[str, Any] | None = None
    final_control = {"num_boost_round": base_num_boost_round, "early_stopping_rounds": base_early_stopping_rounds}
    if llm_guided_tuning_enabled(config):
        training_mode = "llm_guided_tune"
        tuning_cfg = resolve_tuning_config(config)
        trial_history: list[dict[str, Any]] = []
        tuning_decisions: list[str] = []
        experiment_name = runtime_experiment.get("name") or "main_lgbm"
        training_summary = {
            "train_samples": int(len(tr_y)),
            "valid_samples": int(len(va_y)),
            "train_bad_rate": float(tr_y.mean()),
            "valid_bad_rate": float(va_y.mean()),
            "candidate_features": len(candidate_features),
            "kept_features": len(kept_features),
            "train_values": train_values,
            "valid_values": valid_values,
        }

        def _run_trial(round_index: int, name: str, trial_params: dict[str, Any], reason: str, advisor_type: str) -> dict[str, Any]:
            fitted, iteration, trial_metrics, trial_lgb_params, control = _fit_once(trial_params)
            display_params = {**public_lgb_params(trial_lgb_params), **control}
            record: dict[str, Any] = {
                "trial_id": len(trial_history),
                "round": round_index,
                "candidate_name": name,
                "advisor_type": advisor_type,
                "reason": reason,
                "params": display_params,
                **trial_metrics,
                "_model": fitted,
                "_best_iteration": iteration,
                "_lgb_params": trial_lgb_params,
                "_control": control,
            }
            trial_history.append(record)
            return record

        if progress:
            progress.emit(step="tuning_baseline", message="LLM-guided tuning：开始 baseline trial", percent=52)
        baseline = _run_trial(0, "baseline", {}, "configured baseline parameters", "baseline")
        best_valid_ks = float(baseline.get("valid_ks", 0.0) or 0.0)
        no_improvement_rounds = 0
        stop_rules = tuning_cfg.get("stop_rules") if isinstance(tuning_cfg.get("stop_rules"), dict) else {}
        min_improvement = float(stop_rules.get("min_ks_improvement", 0.001))
        stop_patience = int(stop_rules.get("stop_if_no_improvement_rounds", 1))
        for round_index in range(1, int(tuning_cfg["max_rounds"]) + 1):
            if len(trial_history) - 1 >= int(tuning_cfg["max_trials"]):
                tuning_decisions.append(f"round {round_index}: skipped because max_trials reached")
                break
            context = build_tuning_context(
                round_index=round_index,
                experiment=experiment_name,
                algorithm="lightgbm",
                base_params={**public_lgb_params(params), **final_control},
                training_summary=training_summary,
                trial_history=trial_history,
                tuning_cfg=tuning_cfg,
            )
            context_path = output_dir / f"tuning_context_round_{round_index}.json"
            plan_path = output_dir / f"llm_tuning_plan_round_{round_index}.json"
            (output_dir / "tuning_context.json").write_text(json.dumps(context, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            context_path.write_text(json.dumps(context, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            plan = suggest_lgb_candidates(context, tuning_cfg, plan_path=plan_path, context_path=context_path)
            plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            tuning_decisions.append(f"round {round_index}: {plan.get('diagnosis', '')}")
            if plan.get("stop"):
                tuning_decisions.append(f"round {round_index}: advisor requested stop")
                break
            before_round_best = max(float(item.get("valid_ks", 0.0) or 0.0) for item in trial_history)
            for candidate in plan.get("candidates", []):
                if len(trial_history) - 1 >= int(tuning_cfg["max_trials"]):
                    break
                if progress:
                    progress.emit(
                        step="tuning_trial",
                        message=f"LLM-guided tuning：round {round_index} trial {len(trial_history)} {candidate.get('name')}",
                        percent=min(70, 52 + round_index * 6),
                    )
                _run_trial(
                    round_index,
                    str(candidate.get("name") or f"round_{round_index}_candidate"),
                    candidate.get("params") if isinstance(candidate.get("params"), dict) else {},
                    str(candidate.get("reason") or ""),
                    str(plan.get("advisor_type") or "unknown"),
                )
            after_round_best = max(float(item.get("valid_ks", 0.0) or 0.0) for item in trial_history)
            if after_round_best - before_round_best < min_improvement:
                no_improvement_rounds += 1
                tuning_decisions.append(
                    f"round {round_index}: improvement {after_round_best - before_round_best:.6f} < {min_improvement:.6f}"
                )
            else:
                best_valid_ks = max(best_valid_ks, after_round_best)
                no_improvement_rounds = 0
            if no_improvement_rounds >= stop_patience:
                tuning_decisions.append(f"round {round_index}: stopped after {no_improvement_rounds} no-improvement round(s)")
                break
        best_trial, selection = select_best_trial(trial_history, tuning_cfg)
        model = best_trial["_model"]
        best_iter = int(best_trial["_best_iteration"])
        params = dict(best_trial["_lgb_params"])
        final_control = dict(best_trial["_control"])
        metrics = {
            key: best_trial[key]
            for key in [
                "train_auc",
                "valid_auc",
                "train_ks",
                "valid_ks",
                "train_samples",
                "valid_samples",
                "train_bad_rate",
                "valid_bad_rate",
                "best_iteration",
                "train_time_seconds",
                "auc_gap",
            ]
            if key in best_trial
        }
        trial_rows = [_trial_row(item) for item in trial_history]
        pd.DataFrame(trial_rows).to_csv(output_dir / "tuning_trials.csv", index=False, encoding="utf-8-sig")
        tuning_summary = {
            "mode": training_mode,
            "advisor_types": sorted({str(item.get("advisor_type")) for item in trial_history}),
            "trial_count": len(trial_history),
            "best_trial_id": selection.get("best_trial_id"),
            "selection": selection,
            "best_params": best_trial.get("params"),
            "decisions": tuning_decisions,
        }
        (output_dir / "tuning_summary.json").write_text(json.dumps(tuning_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output_dir / "best_params.json").write_text(json.dumps(best_trial.get("params"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_selection_reason(output_dir / "selection_reason.md", best_trial, selection)
        (output_dir / "llm_tuning_decisions.md").write_text(
            "# LLM Tuning Decisions\n\n" + "\n".join(f"- {item}" for item in tuning_decisions) + "\n",
            encoding="utf-8",
        )
    else:
        model, best_iter, metrics, params, final_control = _fit_once({}, log_period=50)
    if progress:
        progress.emit(
            step="train_model_done",
            message=(
                f"LightGBM 训练完成：valid_auc={metrics['valid_auc']:.4f}，"
                f"valid_ks={metrics['valid_ks']:.4f}，best_iter={metrics['best_iteration']}"
            ),
            percent=72,
            metrics=metrics,
        )
    (output_dir / "metrics_train_valid.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    importance = pd.DataFrame(
        {
            "feature": kept_features,
            "gain": model.feature_importance(importance_type="gain"),
            "split": model.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    explainability_cfg = config.get("explainability", {}).get("top_feature_woe", {})
    if explainability_cfg.get("enabled", True):
        from risk_model_workbench.explainability.woe import generate_top_feature_woe

        generate_top_feature_woe(
            raw,
            importance,
            output_dir=output_dir / "woe_top_features",
            label_col=label_col,
            split_col=split_col,
            top_n=int(explainability_cfg.get("top_n", 20)),
            n_bins=int(explainability_cfg.get("n_bins", 10)),
            base_split_value=explainability_cfg.get("base_split_value", "DEV"),
            missing_values=explainability_cfg.get("missing_sentinels", sentinels),
        )
    with (output_dir / "model.pkl").open("wb") as handle:
        pickle.dump(model, handle)

    run_config = {
        "experiment": runtime_experiment.get("name") or "main_lgbm",
        "data_source": str(input_feather),
        "train_values": train_values,
        "valid_values": valid_values,
        "oos_values": oos_values,
        "label_column": label_col,
        "split_column": split_col,
        "feature_list_path": str(feature_list_path),
        "candidate_feature_count": len(candidate_features),
        "actual_feature_count": len(kept_features),
        "algorithm": "lightgbm",
        "params": {key: value for key, value in params.items() if not key.endswith("_seed") and key != "seed"},
        "training_mode": training_mode,
        "training_control": final_control,
        "tuning_summary": tuning_summary,
        "random_seed": train_cfg.get("random_seed", 0),
        "workbench_git_commit": _workbench_git_commit(),
        "input_feather": {"path": str(input_feather), "sha256": _file_sha256(input_feather)},
        **metrics,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    x_all_filled = x_all.fillna(medians).fillna(0)
    if progress:
        progress.emit(step="score_all", message="开始对全量样本打分", percent=82)
    all_pred = model.predict(x_all_filled[kept_features].values, num_iteration=best_iter)
    desired_base = list(
        dict.fromkeys(
            id_cols
            + base_cols
            + [item for item in time_cols if item]
            + [split_col, label_col, "ds", "blue_customer_flag", "zc_level", "prc_amt_xz_30d_3m", "ovd_amt_xz_30d_3m"]
        )
    )
    historical_scores = [column for column in configured_historical_scores if column in raw.columns]
    scores = raw[[column for column in desired_base if column in raw.columns]].copy()
    for column in historical_scores:
        # 强转 numeric：部分历史分（champion）在源表中可能存为字符串，转 float 后
        # score_column_summary 的 mean() 以及下游 evaluate/compare 的 AUC/KS/decile 才正确。
        scores[column] = pd.to_numeric(raw[column], errors="coerce")
    scores["model_score"] = all_pred
    scores.reset_index(drop=True).to_feather(str(score_output))
    if progress:
        progress.emit(
            step="score_written",
            message=f"全量打分写入完成：{len(scores)} 行",
            percent=90,
            metrics={"rows": int(len(scores)), "score_output": str(score_output)},
        )

    pd.DataFrame(
        [
            {
                "score_column": column,
                "non_null_count": int(scores[column].notna().sum()),
                "null_count": int(scores[column].isna().sum()),
                "mean": float(scores[column].mean()),
                "available": True,
            }
            for column in ["model_score", *historical_scores]
        ]
    ).to_csv(score_output.parent / "score_column_summary.csv", index=False, encoding="utf-8-sig")

    _write_input_snapshot(
        raw=raw,
        scores=scores,
        input_snapshot_dir=input_snapshot_dir,
        input_feather=input_feather,
        feature_list_path=feature_list_path,
        label_col=label_col,
        split_col=split_col,
        train_values=train_values,
        valid_values=valid_values,
        oos_values=oos_values,
        candidate_feature_count=len(candidate_features),
        kept_feature_count=len(kept_features),
        historical_scores=historical_scores,
        id_columns=id_cols,
    )
    if progress:
        progress.emit(
            step="write_artifacts",
            status="done",
            message=f"训练产物写入完成：模型、指标、重要性和打分文件已生成",
            percent=100,
            metrics={"output_dir": str(output_dir), "score_output": str(score_output)},
        )
    return metrics


def _write_input_snapshot(**kwargs: Any) -> None:
    import pandas as pd

    raw = kwargs["raw"]
    scores = kwargs["scores"]
    output_dir = Path(kwargs["input_snapshot_dir"])
    label_col = kwargs["label_col"]
    split_col = kwargs["split_col"]
    id_columns = kwargs.get("id_columns") or []
    count_col = next((column for column in id_columns if column in raw.columns), label_col)
    input_config = {
        "data_source": str(kwargs["input_feather"]),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "label_column": label_col,
        "split_column": split_col,
        "train_values": kwargs["train_values"],
        "valid_values": kwargs["valid_values"],
        "oos_values": kwargs["oos_values"],
        "feature_list_source": str(kwargs["feature_list_path"]),
        "feature_count_in_list": kwargs["candidate_feature_count"],
        "feature_count_in_data": kwargs["kept_feature_count"],
        "historical_score_columns": kwargs["historical_scores"],
    }
    (output_dir / "input_config.json").write_text(json.dumps(input_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(
        [{"column": col, "dtype": str(scores[col].dtype), "non_null": int(scores[col].notna().sum()), "null": int(scores[col].isna().sum())} for col in scores.columns]
    ).to_csv(output_dir / "input_schema.csv", index=False, encoding="utf-8-sig")
    raw.groupby(split_col).agg(samples=(count_col, "count"), positive=(label_col, "sum"), bad_rate=(label_col, "mean")).reset_index().to_csv(
        output_dir / "sample_split_summary.csv", index=False, encoding="utf-8-sig"
    )
    label_dist = raw[label_col].value_counts().reset_index()
    label_dist.columns = ["label", "count"]
    label_dist["ratio"] = label_dist["count"] / label_dist["count"].sum()
    label_dist.to_csv(output_dir / "label_distribution.csv", index=False, encoding="utf-8-sig")
    if "blue_customer_flag" in raw.columns:
        seg_dist = raw["blue_customer_flag"].value_counts().reset_index()
        seg_dist.columns = ["segment", "count"]
        seg_dist["ratio"] = seg_dist["count"] / seg_dist["count"].sum()
        seg_dist.to_csv(output_dir / "segment_distribution.csv", index=False, encoding="utf-8-sig")
