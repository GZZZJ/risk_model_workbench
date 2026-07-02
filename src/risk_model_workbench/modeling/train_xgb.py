"""Generic tabular training backends beyond the canonical LightGBM path."""

from __future__ import annotations

import json
import pickle
import re
import time
from pathlib import Path
from typing import Any

from risk_model_workbench.modeling.train_lgb import coerce_features, fill_na_from_train, _write_input_snapshot


def _predict_proba(model: Any, frame: Any) -> Any:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(frame)[:, 1]
    if hasattr(model, "decision_function"):
        import numpy as np

        raw = model.decision_function(frame)
        return 1.0 / (1.0 + np.exp(-raw))
    return model.predict(frame)


def _feature_importance(model: Any, features: list[str]):
    import numpy as np
    import pandas as pd

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        values = np.zeros(len(features))
    return pd.DataFrame({"feature": features, "gain": values, "split": values}).sort_values("gain", ascending=False)


def _make_model(algorithm: str, config: dict[str, Any], tr_y):
    positives = float((tr_y == 1).sum())
    negatives = float((tr_y == 0).sum())
    scale_pos_weight = negatives / positives if positives > 0 else 1.0
    step_params = config.get("runtime_step_params") or config.get("training", {}).get("runtime_step_params") or {}
    scale_cfg = step_params.get("scale_pos_weight", {})
    if scale_cfg:
        scale_pos_weight = float(scale_cfg.get("value") or scale_pos_weight)

    if algorithm == "xgboost":
        try:
            from xgboost import XGBClassifier

            xgb_cfg = config.get("xgboost", {})
            return (
                XGBClassifier(
                    n_estimators=int(xgb_cfg.get("n_estimators", 120)),
                    max_depth=int(xgb_cfg.get("max_depth", 3)),
                    learning_rate=float(xgb_cfg.get("learning_rate", 0.05)),
                    subsample=float(xgb_cfg.get("subsample", 0.8)),
                    colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.8)),
                    eval_metric="auc",
                    random_state=int(config.get("training", {}).get("random_seed", 0)),
                    scale_pos_weight=scale_pos_weight,
                    n_jobs=1,
                ),
                "xgboost",
            )
        except Exception:
            from sklearn.ensemble import HistGradientBoostingClassifier

            return (
                HistGradientBoostingClassifier(
                    max_iter=int(config.get("xgboost", {}).get("n_estimators", 120)),
                    learning_rate=float(config.get("xgboost", {}).get("learning_rate", 0.05)),
                    random_state=int(config.get("training", {}).get("random_seed", 0)),
                ),
                "sklearn_hist_gradient_boosting_xgboost_fallback",
            )

    if algorithm == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        class_weight = {0: 1.0, 1: scale_pos_weight} if scale_cfg else None
        return LogisticRegression(max_iter=1000, class_weight=class_weight), "sklearn_logistic_regression"

    if algorithm == "teacher_student_distillation":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return (
            HistGradientBoostingClassifier(
                max_iter=80,
                learning_rate=0.06,
                random_state=int(config.get("training", {}).get("random_seed", 0)),
            ),
            "sklearn_teacher_student_minimal",
        )

    if algorithm == "hier_ranknet":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return (
            HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.05,
                random_state=int(config.get("training", {}).get("random_seed", 0)),
            ),
            "sklearn_hier_ranknet_minimal",
        )

    raise ValueError(f"unsupported training algorithm: {algorithm}")


def train_tabular_from_feather(
    *,
    input_feather: str | Path,
    feature_list_path: str | Path,
    output_dir: str | Path,
    score_output: str | Path,
    input_snapshot_dir: str | Path,
    config: dict[str, Any],
    algorithm: str,
    progress: Any | None = None,
) -> dict[str, Any]:
    """Train a local tabular binary model and emit standard workbench artifacts."""
    import numpy as np
    import pandas as pd
    from scipy.stats import ks_2samp
    from sklearn.metrics import roc_auc_score

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
    preproc_cfg = config.get("preprocessing", {})
    runtime_experiment = config.get("runtime_experiment", {})

    candidate_features = [line.strip() for line in feature_list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    id_cols = input_cfg.get("id_columns", ["uid"])
    base_cols = input_cfg.get("base_columns", [])
    historical_scores = input_cfg.get("historical_score_columns", [])
    label_col = input_cfg["label_column"]
    split_col = input_cfg["split_column"]
    segment_filter = str(runtime_experiment.get("segment_filter") or "").strip()
    segment_tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", segment_filter)
    time_cols = [input_cfg.get("time_column"), input_cfg.get("period_column")]
    read_cols = list(
        dict.fromkeys(
            id_cols
            + base_cols
            + input_cfg.get("segment_columns", [])
            + [token for token in segment_tokens if token not in {"in", "and", "or", "not", "True", "False"}]
            + historical_scores
            + [item for item in time_cols if item]
            + [label_col, split_col]
            + candidate_features
        )
    )
    all_cols = pd.read_feather(input_feather, columns=None).columns.tolist()
    raw = pd.read_feather(input_feather, columns=[column for column in read_cols if column in all_cols])
    if segment_filter:
        raw = raw.query(segment_filter).copy()
    if progress:
        progress.emit(
            step="read_input_done",
            message=f"训练数据读取完成：{len(raw)} 行 {len(raw.columns)} 列",
            percent=20,
            metrics={"rows": int(len(raw)), "columns": int(len(raw.columns)), "algorithm": algorithm},
        )

    train_values = train_cfg.get("train_values", ["DEV"])
    # 默认用 in-time OOS（DEV-OOS）兜底，不能用 OOT：OOT/OOT-OOS 是时间外样本，
    # 一旦默认进入 valid_sets 会参与早停/调参选优，污染时间外评估。
    valid_values = train_cfg.get("valid_values", ["DEV-OOS"])
    oos_values = train_cfg.get("oos_values", ["DEV-OOS", "OOT-OOS"])
    train_mask = raw[split_col].isin(train_values) & raw[label_col].isin([0, 1])
    valid_mask = raw[split_col].isin(valid_values) & raw[label_col].isin([0, 1])
    if int(train_mask.sum()) < 2 or int(valid_mask.sum()) < 2:
        raise ValueError("training and validation splits must each contain at least two labeled rows")

    sentinels = preproc_cfg.get("missing_sentinels", [-999, -998])
    min_non_null_rate = float(preproc_cfg.get("min_non_null_rate", 0.01))
    drop_constant = bool(preproc_cfg.get("drop_constant", True))
    max_unique_values = int(preproc_cfg.get("max_unique_values", 1))
    x_all, kept_features, drop_detail = coerce_features(raw, candidate_features, sentinels, min_non_null_rate, drop_constant, max_unique_values)
    if not kept_features:
        raise ValueError("no usable training features after preprocessing")
    tr_x = x_all[train_mask].reset_index(drop=True)
    tr_y = raw.loc[train_mask, label_col].astype(int).reset_index(drop=True)
    va_x = x_all[valid_mask].reset_index(drop=True)
    va_y = raw.loc[valid_mask, label_col].astype(int).reset_index(drop=True)
    (tr_x, va_x), medians = fill_na_from_train(tr_x, va_x)

    model, backend = _make_model(algorithm, config, tr_y)
    if progress:
        progress.emit(step="train_model", message=f"{algorithm} 训练开始", percent=50)
    start = time.time()
    model.fit(tr_x, tr_y)

    if algorithm == "teacher_student_distillation":
        from sklearn.linear_model import LogisticRegression

        teacher_score = _predict_proba(model, tr_x)
        weights = 0.5 + np.abs(teacher_score - 0.5)
        student = LogisticRegression(max_iter=1000)
        student.fit(tr_x, tr_y, sample_weight=weights)
        (output_dir / "distillation_summary.json").write_text(
            json.dumps(
                {
                    "teacher_backend": backend,
                    "student_backend": "sklearn_logistic_regression",
                    "mean_teacher_confidence_weight": float(weights.mean()),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        model = student
        backend = "teacher_student_distillation_minimal"

    tr_pred = _predict_proba(model, tr_x)
    va_pred = _predict_proba(model, va_x)
    metrics = {
        "train_auc": float(roc_auc_score(tr_y, tr_pred)),
        "valid_auc": float(roc_auc_score(va_y, va_pred)),
        "train_ks": float(ks_2samp(tr_pred[tr_y == 1], tr_pred[tr_y == 0]).statistic),
        "valid_ks": float(ks_2samp(va_pred[va_y == 1], va_pred[va_y == 0]).statistic),
        "train_samples": int(len(tr_y)),
        "valid_samples": int(len(va_y)),
        "train_bad_rate": float(tr_y.mean()),
        "valid_bad_rate": float(va_y.mean()),
        "train_time_seconds": round(time.time() - start, 1),
        "algorithm": algorithm,
        "backend": backend,
    }
    metrics["auc_gap"] = metrics["train_auc"] - metrics["valid_auc"]
    (output_dir / "metrics_train_valid.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

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
    _feature_importance(model, kept_features).to_csv(output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    with (output_dir / "model.pkl").open("wb") as handle:
        pickle.dump(model, handle)

    run_config = {
        "experiment": runtime_experiment.get("name") or "model",
        "data_source": str(input_feather),
        "train_values": train_values,
        "valid_values": valid_values,
        "oos_values": oos_values,
        "label_column": label_col,
        "split_column": split_col,
        "feature_list_path": str(feature_list_path),
        "candidate_feature_count": len(candidate_features),
        "actual_feature_count": len(kept_features),
        "algorithm": algorithm,
        "backend": backend,
        "random_seed": train_cfg.get("random_seed", 0),
        **metrics,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    x_all_filled = x_all.fillna(medians).fillna(0)
    all_pred = _predict_proba(model, x_all_filled[kept_features])
    desired_base = list(
        dict.fromkeys(
            id_cols
            + base_cols
            + [item for item in time_cols if item]
            + [split_col, label_col, "ds", "blue_customer_flag", "zc_level", "prc_amt_xz_30d_3m", "ovd_amt_xz_30d_3m"]
        )
    )
    scores = raw[[column for column in desired_base if column in raw.columns]].copy()
    for column in historical_scores:
        if column in raw.columns:
            scores[column] = raw[column]
    scores["model_score"] = all_pred
    scores.reset_index(drop=True).to_feather(str(score_output))
    pd.DataFrame(
        [
            {
                "score_column": column,
                "non_null_count": int(scores[column].notna().sum()),
                "null_count": int(scores[column].isna().sum()),
                "mean": float(pd.to_numeric(scores[column], errors="coerce").mean()),
                "available": True,
            }
            for column in ["model_score", *[column for column in historical_scores if column in scores.columns]]
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
        historical_scores=[column for column in historical_scores if column in scores.columns],
        id_columns=id_cols,
    )
    if progress:
        progress.emit(
            step="write_artifacts",
            status="done",
            message=f"{algorithm} 训练产物写入完成",
            percent=100,
            metrics={"output_dir": str(output_dir), "score_output": str(score_output)},
        )
    return metrics
