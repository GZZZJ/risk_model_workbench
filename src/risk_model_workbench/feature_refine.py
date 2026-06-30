#!/usr/bin/env python3
"""Refine wide-table features with correlation and importance filters.

This module pulls a sampled dataset from a DP wide table through TMLSQL only
when explicitly approved. It is intended for the post batch-screening
convergence stage:

1. global correlation de-duplication
2. D03 random-importance filtering
3. D04 null-importance filtering
4. D05 baseline model importance top-N selection
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "agent.py").exists():
            return candidate
    raise RuntimeError("Cannot locate repo root from script path")


REPO_ROOT = find_repo_root(SCRIPT_PATH)
sys.path.insert(0, str(REPO_ROOT))

from risk_model_workbench.config import load_yaml
from risk_model_workbench.dp_feather import (
    default_dataset_paths,
    load_or_fetch_dp_feather,
    print_sql_review,
    write_dataset_metadata,
)
from risk_model_workbench.data.local_feather_profile import profile_local_feather, write_local_feather_profile
from risk_model_workbench.manifest import write_manifest
from risk_model_workbench.progress import ProgressReporter
from risk_model_workbench.resource_planning import default_peak_multiplier_for_stage
from risk_model_workbench.resource_usage import ProcessMemoryTracker, dataframe_memory_bytes


DEFAULT_PROJECT_DIR = Path.cwd()


@dataclass(frozen=True)
class DatasetParts:
    train_x: pd.DataFrame
    train_y: pd.Series
    valid_x: pd.DataFrame
    valid_y: pd.Series


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine wide-table features to <=500 candidates.")
    parser.add_argument("--project-dir", default=str(DEFAULT_PROJECT_DIR), help="Project workspace directory.")
    parser.add_argument(
        "--config",
        default="configs/refine_features.yaml",
        help="Refine config path, relative to project-dir unless absolute.",
    )
    parser.add_argument(
        "--dry-run-sql",
        action="store_true",
        help="Only print and save the DP sampling SQL; do not query DP or train models.",
    )
    parser.add_argument(
        "--refresh-dp-cache",
        action="store_true",
        help="Refresh the local feather cache from DP after SQL approval.",
    )
    parser.add_argument(
        "--sql-approved",
        action="store_true",
        help="Confirm that the displayed DP SQL has been reviewed and may be executed.",
    )
    parser.add_argument(
        "--sample-max-rows",
        type=int,
        default=None,
        help="Override feature_refine.sampling.max_rows after external memory probing.",
    )
    parser.add_argument("--run-dir", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def resolve_project_path(project_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def configured_local_feather_path(project_dir: Path, cfg: dict[str, Any]) -> Path | None:
    input_cfg = cfg.get("input", {}) or {}
    dp_cache_cfg = cfg.get("dp_feather", {}) or {}
    runtime_request = cfg.get("runtime_request", {}) or {}
    candidates = [
        input_cfg.get("local_feather_path"),
        input_cfg.get("raw_path"),
        input_cfg.get("feather_path"),
        dp_cache_cfg.get("approved_local_feather_path"),
        dp_cache_cfg.get("local_feather_path"),
    ]
    if runtime_request.get("data_source_mode") == "local_feather":
        candidates.append(runtime_request.get("sample_location"))
    for value in candidates:
        if value:
            return resolve_project_path(project_dir, value)
    return None


def load_feature_list(project_dir: Path, cfg: dict[str, Any]) -> list[str]:
    input_cfg = cfg.get("input", {}) or {}
    feature_map = input_cfg.get("feature_map")
    if not feature_map:
        local_feather = configured_local_feather_path(project_dir, cfg)
        if local_feather is None:
            raise KeyError("feature_refine.input.feature_map")
        columns = [str(column) for column in pd.read_feather(local_feather).columns]
        base_columns = set(input_cfg.get("base_columns", []))
        id_columns = set(input_cfg.get("id_columns", []))
        label_column = input_cfg["label_column"]
        split_column = input_cfg["split_column"]
        exclude = base_columns | id_columns | {label_column, split_column}
        return [column for column in columns if column not in exclude]

    feature_map_path = resolve_project_path(project_dir, feature_map)
    with feature_map_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    features = [row["output_feature"] for row in rows if row.get("output_feature")]
    base_columns = set(cfg["input"].get("base_columns", []))
    id_columns = set(cfg["input"].get("id_columns", []))
    label_column = cfg["input"]["label_column"]
    split_column = cfg["input"]["split_column"]
    exclude = base_columns | id_columns | {label_column, split_column}
    return [feature for feature in features if feature not in exclude]


def sql_identifier(name: str) -> str:
    return name if name.replace("_", "").isalnum() and not name[0].isdigit() else f"`{name}`"


def build_sampling_sql(cfg: dict[str, Any], features: list[str]) -> str:
    input_cfg = cfg["input"]
    sampling = cfg["sampling"]
    base_columns = list(dict.fromkeys(input_cfg["base_columns"]))
    select_columns = base_columns + [feature for feature in features if feature not in base_columns]
    select_expr = ",\n  ".join(sql_identifier(column) for column in select_columns)
    sql = f"select\n  {select_expr}\nfrom {input_cfg['wide_table']}"
    if sampling.get("where"):
        sql += f"\nwhere {sampling['where']}"
    if sampling.get("max_rows"):
        sql += f"\nlimit {int(sampling['max_rows'])}"
    return sql + "\n"


def apply_local_sampling(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    sampling = cfg.get("sampling", {}) or {}
    max_rows = sampling.get("max_rows")
    if not max_rows:
        return df
    max_rows = int(max_rows)
    if max_rows <= 0 or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=int(cfg.get("random_seed", 0))).reset_index(drop=True)


def apply_sample_max_rows_override(cfg: dict[str, Any], sample_max_rows: int | None) -> None:
    if sample_max_rows is None:
        return
    if sample_max_rows <= 0:
        raise ValueError("--sample-max-rows must be a positive integer.")
    cfg.setdefault("sampling", {})["max_rows"] = int(sample_max_rows)


def coerce_feature_frame(df: pd.DataFrame, features: list[str], cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    preprocessing = cfg["preprocessing"]
    sentinels = preprocessing.get("missing_sentinels", [])
    min_non_null_rate = float(preprocessing.get("min_non_null_rate", 0.0))
    drop_constant = bool(preprocessing.get("drop_constant", True))
    max_unique_values = int(preprocessing.get("max_unique_values", 1))

    available = [feature for feature in features if feature in df.columns]
    if len(available) == 0:
        sample_features = features[:5]
        sample_df_cols = list(df.columns[:10])
        print(f"[WARN] available=0: df.columns[:10]={sample_df_cols}, "
              f"first features={sample_features}", file=sys.stderr)
    else:
        base_cols = list(cfg["input"].get("base_columns", []))[:5]
        for c in base_cols:
            if c in df.columns:
                print(f"[DEBUG] base_col={c}, dtype={df[c].dtype}, "
                      f"sample={list(df[c].head(3).values)}, "
                      f"non_null={df[c].notna().mean():.4f}")
        for f in available[:3]:
            print(f"[DEBUG] feature={f}, dtype={df[f].dtype}, "
                  f"sample={list(df[f].head(3).values)}, "
                  f"non_null={df[f].notna().mean():.4f}")
    x = df.loc[:, available].copy()
    stats = []
    kept = []
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
    drop_counts = {}
    for s in stats:
        if s["drop_reason"]:
            drop_counts[s["drop_reason"]] = drop_counts.get(s["drop_reason"], 0) + 1
    if kept:
        print(f"[PREPROCESS] kept={len(kept)}/{len(available)}, drops={drop_counts}")
    else:
        sample_dropped = [s for s in stats if s["drop_reason"]][:3]
        print(f"[PREPROCESS] ALL DROPPED: {drop_counts}, samples={sample_dropped}")
    return x.loc[:, kept], kept, pd.DataFrame(stats)


def make_dataset_parts(df: pd.DataFrame, x: pd.DataFrame, cfg: dict[str, Any]) -> DatasetParts:
    input_cfg = cfg["input"]
    label = input_cfg["label_column"]
    split = input_cfg["split_column"]
    train_mask = (df[split] == input_cfg["train_value"]) & df[label].isin([0, 1])
    valid_mask = (df[split] == input_cfg["valid_value"]) & df[label].isin([0, 1])
    if not train_mask.any() or not valid_mask.any():
        raise RuntimeError("Both train and valid splits are required for importance refinement.")
    return DatasetParts(
        train_x=x.loc[train_mask].reset_index(drop=True),
        train_y=df.loc[train_mask, label].astype(int).reset_index(drop=True),
        valid_x=x.loc[valid_mask].reset_index(drop=True),
        valid_y=df.loc[valid_mask, label].astype(int).reset_index(drop=True),
    )


def fill_for_model(train_x: pd.DataFrame, valid_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    medians = train_x.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0)
    return train_x.fillna(medians).fillna(0), valid_x.fillna(medians).fillna(0)


def univariate_auc_scores(x: pd.DataFrame, y: pd.Series) -> pd.Series:
    from sklearn.metrics import roc_auc_score

    scores = {}
    y_values = y.to_numpy()
    for feature in x.columns:
        values = x[feature].fillna(x[feature].median()).fillna(0).to_numpy()
        try:
            auc = roc_auc_score(y_values, values)
            scores[feature] = abs(float(auc) - 0.5)
        except ValueError:
            scores[feature] = 0.0
    return pd.Series(scores).sort_values(ascending=False)


def _resolve_vendor_feature_select_code_dir(project_dir: Path | None = None) -> Path:
    """Locate vendor/feature-select-v2/scripts/code (mirrors batch_feature_select search)."""
    env_dir = os.environ.get("FEATURE_SELECT_V2_CODE_DIR")
    here = Path(__file__).resolve()
    candidates: list[Path | None] = []
    if env_dir:
        candidates.append(Path(env_dir))
    if project_dir is not None:
        candidates.append(project_dir / "vendor" / "feature-select-v2" / "scripts" / "code")
    candidates.extend(
        [
            here.parents[2] / "vendor" / "feature-select-v2" / "scripts" / "code",
            Path.cwd() / "vendor" / "feature-select-v2" / "scripts" / "code",
        ]
    )
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return here.parents[2] / "vendor" / "feature-select-v2" / "scripts" / "code"


def _load_vendor_feature_select():
    """Import vendored feature_select helpers (same functions remote prescreen uses)."""
    code_dir = _resolve_vendor_feature_select_code_dir()
    for path in (str(code_dir), str(code_dir / "utils")):
        if path not in sys.path:
            sys.path.insert(0, path)
    from utils.feature_select import _corr_filter, _iv_filter, batch_psi  # type: ignore[import-not-found]

    return _iv_filter, _corr_filter, batch_psi


def d01_local_prescreen(
    parts: DatasetParts, available_features: list[str], cfg: dict[str, Any]
) -> tuple[list[str], pd.DataFrame]:
    """Local-feather d01: missing-rate (already applied upstream by coerce_feature_frame)
    + IV + correlation, on the DEV split. Reuses vendor ``_iv_filter`` + ``_corr_filter``
    (the native path ``d01_preselect_by_toad`` falls back to without toad), so local and
    remote d01 share IV/corr semantics. Returns (kept_features, detail_df).
    """
    step_cfg = cfg.get("local_d01", {}) or {}
    if not step_cfg.get("enabled", True):
        detail = pd.DataFrame({"feature": available_features, "iv": 0.0, "drop_reason": "kept"})
        return list(available_features), detail
    iv_threshold = float(step_cfg.get("iv", 0.005))
    corr_threshold = float(step_cfg.get("corr", 0.8))
    n_bins = int(step_cfg.get("n_bins", 10))

    _iv_filter, _corr_filter, _ = _load_vendor_feature_select()
    dev = parts.train_x.loc[:, available_features].copy()
    dev["_target_"] = parts.train_y.values
    iv_drop, iv_dict = _iv_filter(dev, available_features, "_target_", iv_threshold, n_bins=n_bins)
    iv_drop_set = set(iv_drop)
    iv_survivors = [feature for feature in available_features if feature not in iv_drop_set]
    corr_drop = _corr_filter(dev, iv_survivors, iv_dict, corr_threshold)
    corr_drop_set = set(corr_drop)

    kept = [feature for feature in iv_survivors if feature not in corr_drop_set]
    rows = []
    for feature in available_features:
        if feature in iv_drop_set:
            reason = "low_iv"
        elif feature in corr_drop_set:
            reason = "high_corr"
        else:
            reason = "kept"
        rows.append({"feature": feature, "iv": float(iv_dict.get(feature, 0.0)), "drop_reason": reason})
    return kept, pd.DataFrame(rows)


def d02_local_psi(
    parts: DatasetParts, remain_features: list[str], cfg: dict[str, Any]
) -> tuple[list[str], pd.DataFrame]:
    """Local-feather d02 stability filter: per-feature PSI (DEV vs OOT). Reuses vendor
    ``batch_psi`` (same function remote ``run_d02`` uses), so local and remote PSI share
    binning. Returns (kept_features, detail_df).
    """
    step_cfg = cfg.get("local_d02", {}) or {}
    if not step_cfg.get("enabled", True):
        detail = pd.DataFrame({"feature": remain_features, "max_psi": 0.0, "drop_reason": "kept"})
        return list(remain_features), detail
    if not remain_features:
        return [], pd.DataFrame(columns=["feature", "max_psi", "drop_reason"])
    psi_threshold = float(step_cfg.get("psi", 0.2))

    _, _, batch_psi = _load_vendor_feature_select()
    data_iter = iter(
        [
            ("base_DEV", parts.train_x.loc[:, remain_features]),
            ("exp_OOT", parts.valid_x.loc[:, remain_features]),
        ]
    )
    psi_result = batch_psi(data_iter, list(remain_features), method="quantile", num_nbins=10)
    fea_psi = psi_result[2] if isinstance(psi_result, tuple) and len(psi_result) > 2 else psi_result
    feature_max_psi = {feature: float(max(psi.values())) for feature, psi in fea_psi.items()}

    kept: list[str] = []
    rows = []
    for feature in remain_features:
        psi = feature_max_psi.get(feature, 0.0)
        is_kept = psi <= psi_threshold
        if is_kept:
            kept.append(feature)
        rows.append({"feature": feature, "max_psi": psi, "drop_reason": "kept" if is_kept else "high_psi"})
    return kept, pd.DataFrame(rows)


def global_corr_select(train_x: pd.DataFrame, train_y: pd.Series, cfg: dict[str, Any]) -> tuple[list[str], pd.DataFrame]:
    step_cfg = cfg["global_corr"]
    if not step_cfg.get("enabled", True):
        return list(train_x.columns), pd.DataFrame()

    threshold = float(step_cfg["threshold"])
    scores = univariate_auc_scores(train_x, train_y)
    corr = train_x.loc[:, scores.index].corr().abs().fillna(0)
    kept: list[str] = []
    dropped = []
    for feature in scores.index:
        matched = [kept_feature for kept_feature in kept if corr.loc[feature, kept_feature] >= threshold]
        if matched:
            best_match = max(matched, key=lambda item: corr.loc[feature, item])
            dropped.append(
                {
                    "feature": feature,
                    "drop_reason": "global_corr",
                    "kept_feature": best_match,
                    "corr": float(corr.loc[feature, best_match]),
                    "feature_score": float(scores[feature]),
                    "kept_score": float(scores[best_match]),
                }
            )
        else:
            kept.append(feature)
    return kept, pd.DataFrame(dropped)


def lgb_params(cfg: dict[str, Any], seed: int) -> tuple[dict[str, Any], int, int]:
    lgb_cfg = cfg["lightgbm"]
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
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "verbosity": -1,
    }
    return params, int(lgb_cfg.get("num_boost_round", 400)), int(lgb_cfg.get("early_stopping_rounds", 50))


def screening_lgb_params(cfg: dict[str, Any], seed: int) -> tuple[dict[str, Any], int]:
    """feature-select-v2 style screening params: no row/column sampling."""
    params, num_boost_round, _ = lgb_params(cfg, seed)
    params["subsample"] = 1.0
    params["colsample_bytree"] = 1.0
    params["feature_fraction"] = 1.0
    params["bagging_fraction"] = 1.0
    return params, num_boost_round


def train_lgbm(parts: DatasetParts, features: list[str], cfg: dict[str, Any], seed: int):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    train_x, valid_x = fill_for_model(parts.train_x.loc[:, features], parts.valid_x.loc[:, features])
    params, num_boost_round, early_stopping_rounds = lgb_params(cfg, seed)
    train_set = lgb.Dataset(train_x, label=parts.train_y, feature_name=features, free_raw_data=False)
    valid_set = lgb.Dataset(valid_x, label=parts.valid_y, feature_name=features, reference=train_set, free_raw_data=False)
    callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False), lgb.log_evaluation(period=0)]
    model = lgb.train(params, train_set, num_boost_round=num_boost_round, valid_sets=[valid_set], callbacks=callbacks)
    pred = model.predict(valid_x, num_iteration=model.best_iteration)
    auc = float(roc_auc_score(parts.valid_y, pred))
    return model, auc


def train_feature_select_v2_model(train_x: pd.DataFrame, train_y: pd.Series, features: list[str], cfg: dict[str, Any], seed: int):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    params, num_boost_round = screening_lgb_params(cfg, seed)
    num_boost_round = int(cfg.get("d03_random_importance", {}).get("num_boost_round", num_boost_round))
    train_set = lgb.Dataset(train_x.loc[:, features], label=train_y, feature_name=features, free_raw_data=False)
    callbacks = [lgb.log_evaluation(period=0)]
    model = lgb.train(params=params, train_set=train_set, valid_sets=[train_set], valid_names=["INS"], num_boost_round=num_boost_round, callbacks=callbacks)
    pred = model.predict(train_x.loc[:, features])
    auc = float(roc_auc_score(train_y, pred))
    return model, auc


def model_importance(model, features: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": features,
            "split": model.feature_importance(importance_type="split"),
            "gain": model.feature_importance(importance_type="gain"),
        }
    )


def select_feature_select_v2_drops(
    importance: pd.DataFrame,
    random_col: str,
    *,
    thresholds: float | None,
    importance_types: list[str],
    weight: float = 1.0,
) -> tuple[dict[str, set[str]], pd.DataFrame]:
    """Replicate feature-select-v2 D03 random/zero/tail drop rules for one fitted model."""
    if random_col not in set(importance["feature"]):
        raise ValueError(f"random column {random_col!r} missing from importance frame")

    random_row = importance.loc[importance["feature"] == random_col].iloc[0]
    ranked_input = importance.copy()
    real = ranked_input.loc[ranked_input["feature"] != random_col].copy()
    random_thresholds = {kind: float(random_row[kind]) for kind in importance_types}

    random_drop: set[str] = set()
    zero_drop: set[str] = set()
    for kind in importance_types:
        values = pd.to_numeric(ranked_input[kind], errors="coerce").fillna(0)
        random_imp = random_thresholds[kind]
        random_drop.update(ranked_input.loc[(values < random_imp * weight) & (values > 0), "feature"])
        zero_drop.update(ranked_input.loc[values == 0, "feature"])

    threshold_drop: set[str] = set()
    active = ranked_input.loc[~ranked_input["feature"].isin(random_drop | zero_drop)].copy()
    if thresholds is not None and not active.empty:
        threshold_value = float(thresholds)
        for kind in importance_types:
            ranked = active.sort_values(by=kind, ascending=False)
            total = float(pd.to_numeric(ranked[kind], errors="coerce").fillna(0).sum())
            if total <= 0:
                continue
            cumsum_pct = pd.to_numeric(ranked[kind], errors="coerce").fillna(0).cumsum() / total
            threshold_drop.update(ranked.loc[cumsum_pct > threshold_value, "feature"])

    dropped = random_drop | zero_drop | threshold_drop
    rows = []
    for row in real.itertuples(index=False):
        reasons = []
        if row.feature in random_drop:
            reasons.append("random_importance")
        if row.feature in zero_drop:
            reasons.append("zero_importance")
        if row.feature in threshold_drop:
            reasons.append("threshold_tail")
        rows.append(
            {
                "feature": row.feature,
                "split": float(row.split),
                "gain": float(row.gain),
                "random_split_threshold": random_thresholds.get("split", float("nan")),
                "random_gain_threshold": random_thresholds.get("gain", float("nan")),
                "random_drop": row.feature in random_drop,
                "zero_drop": row.feature in zero_drop,
                "threshold_drop": row.feature in threshold_drop,
                "dropped": row.feature in dropped,
                "survives": row.feature not in dropped,
                "drop_reason": ";".join(reasons),
            }
        )

    return {"random": random_drop, "zero": zero_drop, "thresholds": threshold_drop}, pd.DataFrame(rows)


def d03_noise_survival(
    parts: DatasetParts,
    features: list[str],
    cfg: dict[str, Any],
    *,
    progress: ProgressReporter | None = None,
) -> tuple[list[str], pd.DataFrame]:
    step_cfg = cfg["d03_random_importance"]
    rng = np.random.default_rng(int(cfg["random_seed"]))
    random_count = int(step_cfg.get("random_feature_count", 5))
    rounds = int(step_cfg.get("rounds", 3))
    min_survival_rate = float(step_cfg.get("min_survival_rate", 0.67))
    zero_importance_drop = bool(step_cfg.get("zero_importance_drop", True))
    survival = {feature: 0 for feature in features}
    rows = []

    for round_index in range(rounds):
        random_features = [f"__random_noise_{round_index}_{idx}" for idx in range(random_count)]
        train_x = parts.train_x.loc[:, features].copy()
        valid_x = parts.valid_x.loc[:, features].copy()
        for random_feature in random_features:
            train_x[random_feature] = rng.normal(size=len(train_x))
            valid_x[random_feature] = rng.normal(size=len(valid_x))
        round_parts = DatasetParts(train_x, parts.train_y, valid_x, parts.valid_y)
        model_features = features + random_features
        model, auc = train_lgbm(round_parts, model_features, cfg, seed=int(cfg["random_seed"]) + 100 + round_index)
        importance = model_importance(model, model_features)
        random_imp = importance[importance["feature"].isin(random_features)]
        gain_threshold = float(random_imp["gain"].max())
        split_threshold = float(random_imp["split"].max())
        real_imp = importance[~importance["feature"].isin(random_features)]
        round_surv = int(((real_imp["gain"] > gain_threshold) & ((not zero_importance_drop) | (real_imp["split"] > 0))).sum())
        print(f"[D03] round={round_index} n_feat={len(features)} auc={auc:.4f} "
              f"gain_th={gain_threshold:.2f} max_real_gain={real_imp['gain'].max():.2f} "
              f"round_surv={round_surv}/{len(features)}")
        if progress:
            progress.emit(
                step="d03_round",
                message=f"D03 随机重要性第 {round_index + 1}/{rounds} 轮完成，AUC={auc:.4f}，存活 {round_surv} 个变量",
                current=round_index + 1,
                total=rounds,
                percent=45 + ((round_index + 1) / max(rounds, 1)) * 15,
                metrics={"round": round_index + 1, "rounds": rounds, "auc": auc, "survived": round_surv, "features": len(features)},
            )
        for row in importance[~importance["feature"].isin(random_features)].itertuples(index=False):
            survives = row.gain > gain_threshold and (not zero_importance_drop or row.split > 0)
            if survives:
                survival[row.feature] += 1
            rows.append(
                {
                    "round": round_index,
                    "feature": row.feature,
                    "split": float(row.split),
                    "gain": float(row.gain),
                    "random_gain_threshold": gain_threshold,
                    "random_split_threshold": split_threshold,
                    "valid_auc": auc,
                    "survives": survives,
                }
            )

    min_survival = math.ceil(rounds * min_survival_rate)
    kept = [feature for feature in features if survival[feature] >= min_survival]
    return kept, pd.DataFrame(rows)


def d03_feature_select_v2(
    parts: DatasetParts,
    features: list[str],
    cfg: dict[str, Any],
    *,
    progress: ProgressReporter | None = None,
) -> tuple[list[str], pd.DataFrame]:
    step_cfg = cfg["d03_random_importance"]
    rng = np.random.default_rng(int(cfg["random_seed"]))
    random_col = str(step_cfg.get("random_column", "random_col"))
    if random_col in features:
        raise ValueError(f"D03 random_column conflicts with real feature: {random_col}")

    bagging_rounds = int(step_cfg.get("bagging_rounds", step_cfg.get("d03_bagging_round", 5)))
    bagging_fraction = float(step_cfg.get("bagging_fraction", step_cfg.get("d03_bagging_fraction", 0.5)))
    thresholds = step_cfg.get("thresholds", step_cfg.get("d03_thresholds", 0.95))
    importance_types = list(step_cfg.get("importance_types", ["split", "gain"]))
    weight = float(step_cfg.get("weight", 1.0))
    iter_rounds = int(step_cfg.get("iter_rounds", step_cfg.get("iter_round_num", 1)))
    random_low = int(step_cfg.get("random_low", 1))
    random_high = int(step_cfg.get("random_high", 10))

    train_x = parts.train_x.loc[:, features].copy()
    train_x[random_col] = rng.integers(random_low, random_high + 1, size=len(train_x))
    train_frame = train_x.copy()
    target_col = "__d03_target"
    train_frame[target_col] = parts.train_y.reset_index(drop=True).to_numpy()
    dropped_all: set[str] = set()
    rows = []

    for bagging_index in range(bagging_rounds):
        sample_seed = int(rng.integers(10000))
        bag_frame = train_frame.sample(frac=bagging_fraction, random_state=sample_seed).reset_index(drop=True)
        bag_x = bag_frame.drop(columns=[target_col])
        bag_y = bag_frame[target_col].astype(int)
        model_features = features + [random_col]
        final_drop_sets = {"random": set(), "zero": set(), "thresholds": set()}
        final_detail = pd.DataFrame()
        final_iter_index = 0
        train_auc = float("nan")

        for iter_index in range(iter_rounds):
            final_iter_index = iter_index
            model, train_auc = train_feature_select_v2_model(
                bag_x.loc[:, model_features],
                bag_y,
                model_features,
                cfg,
                seed=int(cfg["random_seed"]) + 100 + bagging_index * 100 + iter_index,
            )
            importance = model_importance(model, model_features)
            final_drop_sets, final_detail = select_feature_select_v2_drops(
                importance,
                random_col,
                thresholds=thresholds,
                importance_types=importance_types,
                weight=weight,
            )
            dropped_this_iter = set().union(*final_drop_sets.values())
            model_features = sorted(dropped_this_iter) + [random_col]
            if not dropped_this_iter:
                break

        dropped_round = set().union(*final_drop_sets.values())
        real_dropped_round = dropped_round - {random_col}
        dropped_all.update(real_dropped_round)
        print(f"[D03-v2] bagging_round={bagging_index} n_feat={len(features)} train_auc={train_auc:.4f} "
              f"dropped={len(real_dropped_round)} kept={len(features) - len(real_dropped_round)}")
        if progress:
            progress.emit(
                step="d03_v2_bagging_round",
                message=(
                    f"D03 feature-select-v2 第 {bagging_index + 1}/{bagging_rounds} 轮完成，"
                    f"训练 AUC={train_auc:.4f}，保留 {len(features) - len(real_dropped_round)} 个变量"
                ),
                current=bagging_index + 1,
                total=bagging_rounds,
                percent=45 + ((bagging_index + 1) / max(bagging_rounds, 1)) * 15,
                metrics={
                    "round": bagging_index + 1,
                    "rounds": bagging_rounds,
                    "train_auc": train_auc,
                    "dropped": len(real_dropped_round),
                    "features": len(features),
                },
            )
        if not final_detail.empty:
            final_detail = final_detail.copy()
            final_detail.insert(0, "round", bagging_index)
            final_detail.insert(1, "iteration", final_iter_index)
            final_detail["mode"] = "feature_select_v2"
            final_detail["train_auc"] = train_auc
            final_detail["valid_auc"] = train_auc
            final_detail["bagging_fraction"] = bagging_fraction
            rows.extend(final_detail.to_dict("records"))

    kept = [feature for feature in features if feature not in dropped_all]
    return kept, pd.DataFrame(rows)


def d03_random_importance(
    parts: DatasetParts,
    features: list[str],
    cfg: dict[str, Any],
    *,
    progress: ProgressReporter | None = None,
) -> tuple[list[str], pd.DataFrame]:
    step_cfg = cfg["d03_random_importance"]
    if not step_cfg.get("enabled", True):
        return features, pd.DataFrame()

    mode = str(step_cfg.get("mode", "feature_select_v2"))
    if mode in {"feature_select_v2", "feature_select_v2_compatible", "v2"}:
        return d03_feature_select_v2(parts, features, cfg, progress=progress)
    if mode in {"noise_survival", "workbench_noise_survival"}:
        return d03_noise_survival(parts, features, cfg, progress=progress)
    raise ValueError(f"Unknown d03_random_importance mode: {mode}")


def d04_null_importance(
    parts: DatasetParts,
    features: list[str],
    cfg: dict[str, Any],
    *,
    progress: ProgressReporter | None = None,
) -> tuple[list[str], pd.DataFrame]:
    step_cfg = cfg["d04_null_importance"]
    if not step_cfg.get("enabled", True):
        return features, pd.DataFrame()

    max_features = int(step_cfg.get("max_features_for_null_importance", len(features)))
    working_features = features[:max_features]
    real_rounds = int(step_cfg.get("real_rounds", 3))
    null_rounds = int(step_cfg.get("null_rounds", 20))
    null_percentile = float(step_cfg.get("null_percentile", 75))
    score_threshold = float(step_cfg.get("score_threshold", 1.0))
    seed = int(cfg["random_seed"])
    real_gains = {feature: [] for feature in working_features}
    null_gains = {feature: [] for feature in working_features}

    for round_index in range(real_rounds):
        model, _ = train_lgbm(parts, working_features, cfg, seed=seed + 200 + round_index)
        importance = model_importance(model, working_features)
        for row in importance.itertuples(index=False):
            real_gains[row.feature].append(float(row.gain))
        if progress:
            progress.emit(
                step="d04_real_round",
                message=f"D04 真实重要性第 {round_index + 1}/{real_rounds} 轮完成",
                current=round_index + 1,
                total=real_rounds,
                percent=62 + ((round_index + 1) / max(real_rounds, 1)) * 8,
                metrics={"round": round_index + 1, "rounds": real_rounds, "features": len(working_features)},
            )

    rng = np.random.default_rng(seed + 300)
    for round_index in range(null_rounds):
        shuffled_parts = DatasetParts(
            parts.train_x,
            pd.Series(rng.permutation(parts.train_y.to_numpy())),
            parts.valid_x,
            parts.valid_y,
        )
        model, _ = train_lgbm(shuffled_parts, working_features, cfg, seed=seed + 300 + round_index)
        importance = model_importance(model, working_features)
        for row in importance.itertuples(index=False):
            null_gains[row.feature].append(float(row.gain))
        if progress:
            progress.emit(
                step="d04_null_round",
                message=f"D04 空重要性第 {round_index + 1}/{null_rounds} 轮完成",
                current=round_index + 1,
                total=null_rounds,
                percent=70 + ((round_index + 1) / max(null_rounds, 1)) * 12,
                metrics={"round": round_index + 1, "rounds": null_rounds, "features": len(working_features)},
            )

    rows = []
    kept = []
    eps = 1e-12
    for feature in working_features:
        real_mean = float(np.mean(real_gains[feature])) if real_gains[feature] else 0.0
        null_cut = float(np.percentile(null_gains[feature], null_percentile)) if null_gains[feature] else 0.0
        score = real_mean / (null_cut + eps)
        keep = real_mean > 0 and score >= score_threshold
        if keep:
            kept.append(feature)
        rows.append(
            {
                "feature": feature,
                "real_gain_mean": real_mean,
                "null_gain_percentile": null_cut,
                "null_percentile": null_percentile,
                "null_importance_score": score,
                "survives": keep,
            }
        )
    return kept, pd.DataFrame(rows).sort_values(["survives", "null_importance_score"], ascending=[False, False])


def d05_top_importance(
    parts: DatasetParts,
    features: list[str],
    cfg: dict[str, Any],
    *,
    progress: ProgressReporter | None = None,
) -> tuple[list[str], pd.DataFrame, float]:
    step_cfg = cfg["d05_baseline_importance"]
    keep_top_n = int(step_cfg.get("keep_top_n", cfg.get("target_feature_count", 500)))
    if not step_cfg.get("enabled", True):
        return features[:keep_top_n], pd.DataFrame(), float("nan")

    if progress:
        progress.emit(step="d05_train", message=f"D05 基准模型重要性训练开始，输入 {len(features)} 个变量", percent=84)
    model, auc = train_lgbm(parts, features, cfg, seed=int(cfg["random_seed"]) + 500)
    importance = model_importance(model, features).sort_values("gain", ascending=False).reset_index(drop=True)
    importance["rank"] = np.arange(1, len(importance) + 1)
    kept = importance.head(keep_top_n)["feature"].tolist()
    if progress:
        progress.emit(
            step="d05_done",
            message=f"D05 基准模型重要性完成，AUC={auc:.4f}，最终保留 {len(kept)} 个变量",
            percent=90,
            metrics={"auc": auc, "input_features": len(features), "final_features": len(kept)},
        )
    return kept, importance, auc


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_feature_list(path: Path, features: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(features) + "\n", encoding="utf-8")


def display_path(path: Path, base_dir: Path) -> str:
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


def configured_peak_multiplier(cfg: dict[str, Any]) -> float:
    resource_cfg = cfg.get("resource", {}) or cfg.get("resource_planning", {})
    for key in ("peak_multiplier", "peak_memory_multiplier"):
        if key in resource_cfg:
            return float(resource_cfg[key])
    return default_peak_multiplier_for_stage("feature_refine")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_dir = Path(args.project_dir).resolve()
    reporter = ProgressReporter(args.run_dir, "feature_refine") if args.run_dir else None
    config_path = resolve_project_path(project_dir, args.config)
    cfg = load_yaml(config_path)["feature_refine"]
    apply_sample_max_rows_override(cfg, args.sample_max_rows)
    output_dir = resolve_project_path(project_dir, cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_tracker = ProcessMemoryTracker(stage="feature_refine")
    peak_multiplier = configured_peak_multiplier(cfg)

    initial_features = load_feature_list(project_dir, cfg)
    memory_tracker.record(
        "load_feature_list",
        initial_features=len(initial_features),
        config_path=display_path(config_path, project_dir),
    )
    if reporter:
        reporter.emit(step="load_feature_list", message=f"读取精筛输入变量完成，共 {len(initial_features)} 个", percent=5)
    local_feather_path = configured_local_feather_path(project_dir, cfg)
    sql = "" if local_feather_path is not None else build_sampling_sql(cfg, initial_features)

    dp_cache_cfg = cfg.get("dp_feather", {})
    dataset_id = dp_cache_cfg.get("dataset_id", "feature_refine_wide_sample")
    description = dp_cache_cfg.get(
        "description",
        "Feature prescreen wide-table sample for feature refinement.",
    )
    feather_path, metadata_path = default_dataset_paths(
        project_dir,
        dataset_id=dataset_id,
        data_dir=dp_cache_cfg.get("data_dir", "data/local/dp_feather"),
        metadata_dir=dp_cache_cfg.get("metadata_dir", "data/profile/dp_feather_datasets"),
    )
    if args.dry_run_sql and local_feather_path is not None:
        profile = profile_local_feather(
            local_feather_path,
            required_columns=list(
                dict.fromkeys(
                    list(cfg.get("input", {}).get("id_columns", []))
                    + [cfg["input"]["label_column"], cfg["input"]["split_column"]]
                )
            ),
            split_column=cfg["input"].get("split_column"),
            target_column=cfg["input"].get("label_column"),
            feature_exclude_columns=cfg["input"].get("base_columns", []) + cfg["input"].get("id_columns", []),
            feature_columns=initial_features,
        )
        write_local_feather_profile(profile, output_dir / "local_feather_profile.json")
        write_dataset_metadata(
            project_dir=project_dir,
            metadata_path=metadata_path,
            feather_path=local_feather_path,
            dataset_id=dataset_id,
            description=description,
            sql="",
            status="ready",
            row_count=profile["row_count"],
            column_count=profile["column_count"],
            columns=profile["columns"],
            source="local_feather",
            source_path=local_feather_path,
            note="Local feather mode uses an existing user-provided file; no remote DP SQL was generated or executed.",
        )
        if reporter:
            reporter.emit(
                step="local_feather_ready",
                status="done",
                message="本地 feather 数据源已识别，无需生成 DP SQL",
                percent=100,
                metrics={"feather_path": display_path(local_feather_path, project_dir)},
            )
        return 0

    if args.dry_run_sql:
        write_dataset_metadata(
            project_dir=project_dir,
            metadata_path=metadata_path,
            feather_path=feather_path,
            dataset_id=dataset_id,
            description=description,
            sql=sql,
            status="sql_review_required" if args.refresh_dp_cache or not feather_path.exists() else "ready",
            note="Review this SQL before running DP fetch. The feather file itself is gitignored.",
        )
        print_sql_review(
            dataset_id=dataset_id,
            description=description,
            feather_path=feather_path,
            metadata_path=metadata_path,
            sql=sql,
        )
        if reporter:
            reporter.emit(
                step="dry_run_sql",
                status="waiting_for_approval",
                message="特征精筛抽样 SQL 已生成，等待人工审批后执行 DP 拉数",
                percent=100,
                metrics={"metadata_path": display_path(metadata_path, project_dir)},
            )
        return 0

    if reporter:
        reporter.emit(step="load_sample", message="开始读取特征精筛宽表样本", percent=10)
    raw_df = load_or_fetch_dp_feather(
        project_dir=project_dir,
        sql=sql,
        dataset_id=dataset_id,
        description=description,
        feather_path=feather_path,
        metadata_path=metadata_path,
        refresh=args.refresh_dp_cache,
        sql_approved=args.sql_approved,
        approved_local_feather_path=local_feather_path,
        progress=reporter,
    )
    if local_feather_path is not None:
        raw_df = apply_local_sampling(raw_df, cfg)
    raw_df_memory = dataframe_memory_bytes(raw_df)
    memory_tracker.record(
        "load_sample",
        rows=int(len(raw_df)),
        columns=int(len(raw_df.columns)),
        dataframe_memory_bytes=raw_df_memory,
    )
    x, available_features, preprocess_stats = coerce_feature_frame(raw_df, initial_features, cfg)
    feature_matrix_memory = dataframe_memory_bytes(x)
    memory_tracker.record(
        "preprocess_done",
        available_features=len(available_features),
        feature_matrix_memory_bytes=feature_matrix_memory,
    )
    parts = make_dataset_parts(raw_df, x, cfg)
    memory_tracker.record(
        "split_dataset",
        train_samples=int(len(parts.train_x)),
        valid_samples=int(len(parts.valid_x)),
    )
    print(f"[STAGE] raw_rows={len(raw_df)} initial_feat={len(initial_features)} "
          f"available={len(available_features)} train={len(parts.train_x)} valid={len(parts.valid_x)}")
    if reporter:
        reporter.emit(
            step="preprocess_done",
            message=(
                f"预处理完成：样本 {len(raw_df)} 行，初始变量 {len(initial_features)} 个，"
                f"可用变量 {len(available_features)} 个"
            ),
            percent=28,
            metrics={
                "raw_rows": int(len(raw_df)),
                "initial_features": len(initial_features),
                "available_features": len(available_features),
                "train_samples": int(len(parts.train_x)),
                "valid_samples": int(len(parts.valid_x)),
            },
        )

    d01_kept, d01_detail = d01_local_prescreen(parts, available_features, cfg)
    d02_kept, d02_detail = d02_local_psi(parts, d01_kept, cfg)
    memory_tracker.record("d01_d02_done", d01_kept=len(d01_kept), d02_kept=len(d02_kept))
    print(
        f"[STAGE] after_d01: {len(d01_kept)} (dropped {len(available_features) - len(d01_kept)}) "
        f"| after_d02: {len(d02_kept)} (dropped {len(d01_kept) - len(d02_kept)})"
    )
    corr_features, corr_drops = global_corr_select(parts.train_x.loc[:, d02_kept], parts.train_y, cfg)
    memory_tracker.record(
        "global_corr_done",
        kept=len(corr_features),
        dropped=len(corr_drops),
    )
    print(f"[STAGE] after_global_corr: {len(corr_features)} (dropped {len(corr_drops)})")
    if reporter:
        reporter.emit(
            step="global_corr_done",
            message=f"全局相关性筛选完成，保留 {len(corr_features)} 个，剔除 {len(corr_drops)} 个",
            percent=40,
            metrics={"kept": len(corr_features), "dropped": len(corr_drops)},
        )
    parts_corr = DatasetParts(parts.train_x.loc[:, corr_features], parts.train_y, parts.valid_x.loc[:, corr_features], parts.valid_y)
    d03_features, d03_detail = d03_random_importance(parts_corr, corr_features, cfg, progress=reporter)
    memory_tracker.record(
        "d03_done",
        kept=len(d03_features),
        dropped=len(corr_features) - len(d03_features),
    )
    print(f"[STAGE] after_d03: {len(d03_features)} (dropped {len(corr_features) - len(d03_features)})")
    if reporter:
        reporter.emit(
            step="d03_done",
            message=f"D03 随机重要性完成，保留 {len(d03_features)} 个，剔除 {len(corr_features) - len(d03_features)} 个",
            percent=60,
            metrics={"kept": len(d03_features), "dropped": len(corr_features) - len(d03_features)},
        )
    if len(d03_features) == 0:
        print("[FATAL] D03 eliminated all features, aborting", file=sys.stderr)
        if reporter:
            reporter.emit(step="d03_failed", status="failed", message="D03 剔除了全部变量，流程中止", level="error")
        return 1
    parts_d03 = DatasetParts(parts_corr.train_x.loc[:, d03_features], parts_corr.train_y, parts_corr.valid_x.loc[:, d03_features], parts_corr.valid_y)
    d04_features, d04_detail = d04_null_importance(parts_d03, d03_features, cfg, progress=reporter)
    memory_tracker.record(
        "d04_done",
        kept=len(d04_features),
        dropped=len(d03_features) - len(d04_features),
    )
    if reporter:
        reporter.emit(
            step="d04_done",
            message=f"D04 空重要性完成，保留 {len(d04_features)} 个，剔除 {len(d03_features) - len(d04_features)} 个",
            percent=82,
            metrics={"kept": len(d04_features), "dropped": len(d03_features) - len(d04_features)},
        )
    parts_d04 = DatasetParts(parts_d03.train_x.loc[:, d04_features], parts_d03.train_y, parts_d03.valid_x.loc[:, d04_features], parts_d03.valid_y)
    final_features, d05_importance, d05_auc = d05_top_importance(parts_d04, d04_features, cfg, progress=reporter)
    memory_tracker.record(
        "d05_done",
        final_features=len(final_features),
        d05_valid_auc=d05_auc,
    )
    resource_usage = memory_tracker.summary(
        matrix_bytes=feature_matrix_memory or raw_df_memory,
        row_count=int(len(raw_df)),
        column_count=int(len(raw_df.columns)),
        feature_count=len(available_features),
        configured_peak_multiplier=peak_multiplier,
    )

    preprocess_stats.to_csv(output_dir / "preprocess_feature_stats.csv", index=False, encoding="utf-8-sig")
    corr_drops.to_csv(output_dir / "d00_global_corr_drops.csv", index=False, encoding="utf-8-sig")
    if isinstance(d01_detail, pd.DataFrame) and not d01_detail.empty:
        d01_detail.to_csv(output_dir / "d01_local_prescreen_detail.csv", index=False, encoding="utf-8-sig")
    if isinstance(d02_detail, pd.DataFrame) and not d02_detail.empty:
        d02_detail.to_csv(output_dir / "d02_local_psi_detail.csv", index=False, encoding="utf-8-sig")
    d03_detail.to_csv(output_dir / "d03_random_importance_detail.csv", index=False, encoding="utf-8-sig")
    d04_detail.to_csv(output_dir / "d04_null_importance_detail.csv", index=False, encoding="utf-8-sig")
    d05_importance.to_csv(output_dir / "d05_baseline_importance.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "resource_usage.json", resource_usage)
    write_feature_list(output_dir / "final_500_features.txt", final_features)
    write_feature_list(output_dir / "final_features.txt", final_features)
    with (output_dir / "sample.pkl").open("wb") as handle:
        pickle.dump({"raw_shape": raw_df.shape, "features": final_features}, handle)
    write_json(
        output_dir / "stage_summary.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "wide_table": cfg["input"]["wide_table"],
            "feather_path": display_path(local_feather_path or feather_path, project_dir),
            "data_source_mode": "local_feather" if local_feather_path is not None else "remote_table",
            "raw_rows": int(len(raw_df)),
            "total_rows": int(len(raw_df)),
            "train_samples": int(len(parts.train_x)),
            "valid_samples": int(len(parts.valid_x)),
            "initial_features": len(initial_features),
            "available_features": len(available_features),
            "d01_kept_features": len(d01_kept),
            "d02_kept_features": len(d02_kept),
            "d01_d02_mode": "local_feather",
            "d01_thresholds": cfg.get("local_d01", {}),
            "d02_psi_threshold": float((cfg.get("local_d02", {}) or {}).get("psi", 0.2)),
            "after_global_corr": len(corr_features),
            "d03_mode": str(cfg.get("d03_random_importance", {}).get("mode", "feature_select_v2")),
            "after_d03_random_importance": len(d03_features),
            "after_d04_null_importance": len(d04_features),
            "final_features": len(final_features),
            "d05_valid_auc": d05_auc,
            "sampling_where": cfg["sampling"].get("where"),
            "sampling_max_rows": cfg["sampling"].get("max_rows"),
            "configured_peak_multiplier": peak_multiplier,
            "observed_peak_multiplier": resource_usage.get("observed_peak_multiplier"),
            "resource_usage_path": "resource_usage.json",
        },
    )
    manifest = write_manifest(
        project_dir,
        "refine_wide_features",
        inputs=[
            config_path,
            *(
                [resolve_project_path(project_dir, cfg["input"]["feature_map"])]
                if cfg["input"].get("feature_map")
                else []
            ),
        ],
        outputs=[
            output_dir / "stage_summary.json",
            output_dir / "resource_usage.json",
            output_dir / "final_500_features.txt",
            output_dir / "final_features.txt",
            output_dir / "d05_baseline_importance.csv",
        ],
    )
    print(f"output: {output_dir}")
    print(f"manifest: {manifest}")
    if reporter:
        reporter.emit(
            step="write_outputs",
            status="done",
            message=f"特征精筛产物写入完成，最终保留 {len(final_features)} 个变量",
            percent=100,
            metrics={"final_features": len(final_features), "output_dir": str(output_dir)},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
