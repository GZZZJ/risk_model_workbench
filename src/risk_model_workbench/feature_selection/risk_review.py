"""Deterministic Feature Risk Review evidence for credit-risk models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from risk_model_workbench.feature_selection.leakage_check import (
    load_feature_metadata,
    review_feature_leakage,
)


MISSING_BUCKET = "__MISSING__"
OTHER_BUCKET = "__OTHER__"


@dataclass(frozen=True)
class BinSpec:
    kind: str
    labels: tuple[str, ...]
    edges: tuple[float, ...] = ()
    categories: tuple[str, ...] = ()


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def fit_bin_spec(series: pd.Series, *, n_bins: int = 10, categorical_max_unique: int = 20) -> BinSpec:
    """Fit deterministic bin rules once; callers must reuse the returned spec."""
    numeric = pd.to_numeric(series, errors="coerce")
    non_missing = numeric.dropna()
    original_non_missing = series.dropna()
    numeric_rate = float(numeric.notna().sum() / max(len(original_non_missing), 1))
    if numeric_rate >= 0.95 and not non_missing.empty:
        quantiles = np.linspace(0, 1, max(2, int(n_bins)) + 1)[1:-1]
        internal = np.unique(np.quantile(non_missing.to_numpy(dtype=float), quantiles))
        internal = internal[np.isfinite(internal)]
        edges = tuple([-np.inf, *internal.tolist(), np.inf])
        labels = tuple(f"B{index:02d}" for index in range(1, len(edges)))
        return BinSpec(kind="numeric", labels=labels, edges=edges)

    counts = original_non_missing.astype(str).value_counts()
    categories = tuple(counts.head(max(1, int(categorical_max_unique))).index.tolist())
    labels = tuple(f"C{index:02d}" for index in range(1, len(categories) + 1))
    return BinSpec(kind="categorical", labels=labels, categories=categories)


def apply_bin_spec(series: pd.Series, spec: BinSpec) -> pd.Series:
    """Apply base-fitted rules, preserving missing as its own bucket."""
    result = pd.Series(index=series.index, dtype="object")
    missing = series.isna()
    if spec.kind == "numeric":
        numeric = pd.to_numeric(series, errors="coerce")
        assigned = pd.cut(numeric, bins=list(spec.edges), labels=list(spec.labels), include_lowest=True, right=True)
        result.loc[~numeric.isna()] = assigned.astype("object").loc[~numeric.isna()]
        missing = numeric.isna()
    else:
        mapping = dict(zip(spec.categories, spec.labels))
        values = series.astype("string")
        result.loc[~missing] = values.loc[~missing].map(mapping).fillna(OTHER_BUCKET)
    result.loc[missing] = MISSING_BUCKET
    return result.fillna(OTHER_BUCKET).astype(str)


def _bin_boundary(spec: BinSpec, bucket: str) -> str:
    if bucket == MISSING_BUCKET:
        return "missing"
    if bucket == OTHER_BUCKET:
        return "category_not_seen_in_base"
    if spec.kind == "categorical":
        try:
            return "category=" + spec.categories[spec.labels.index(bucket)]
        except ValueError:
            return bucket
    try:
        index = spec.labels.index(bucket)
    except ValueError:
        return bucket
    return f"({spec.edges[index]}, {spec.edges[index + 1]}]"


def normalize_months(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.to_period("M").astype("string")


def monthly_psi_evidence(
    dev_x: pd.DataFrame,
    dev_months: pd.Series,
    features: Iterable[str],
    *,
    n_bins: int = 10,
    warning_threshold: float = 0.10,
    fail_threshold: float = 0.25,
    min_base_samples: int = 500,
    epsilon: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, BinSpec], list[str]]:
    """Compute monthly DEV PSI against the first natural month using base bins."""
    features = list(features)
    months = pd.Series(dev_months, index=dev_x.index, dtype="string")
    valid_months = sorted(value for value in months.dropna().unique().tolist() if value != "<NA>")
    detail_columns = ["feature", "base_month", "compare_month", "base_samples", "compare_samples", "psi", "psi_status"]
    summary_columns = [
        "feature", "base_month", "base_samples", "monthly_psi_max", "monthly_psi_mean", "monthly_psi_p90",
        "months_above_warning", "months_above_fail", "psi_status", "psi_warning",
    ]
    if not valid_months:
        summary = pd.DataFrame(
            [
                {"feature": feature, "base_month": "", "base_samples": 0, "monthly_psi_max": np.nan, "monthly_psi_mean": np.nan,
                 "monthly_psi_p90": np.nan, "months_above_warning": 0, "months_above_fail": 0, "psi_status": "unknown_month_metadata",
                 "psi_warning": "month_column_missing_or_unparseable"}
                for feature in features
            ],
            columns=summary_columns,
        )
        return pd.DataFrame(columns=detail_columns), summary, {}, ["month_column_missing_or_unparseable"]

    base_month = valid_months[0]
    base_mask = months == base_month
    base_samples = int(base_mask.sum())
    warnings = []
    if base_samples < int(min_base_samples):
        warnings.append(f"base_month_sample_too_small:{base_month}:{base_samples}<{int(min_base_samples)}")

    detail_rows: list[dict[str, Any]] = []
    specs: dict[str, BinSpec] = {}
    all_buckets: dict[str, list[str]] = {}
    for feature in features:
        if feature not in dev_x.columns:
            continue
        spec = fit_bin_spec(dev_x.loc[base_mask, feature], n_bins=n_bins)
        specs[feature] = spec
        all_buckets[feature] = [*spec.labels, OTHER_BUCKET, MISSING_BUCKET]
        base_bins = apply_bin_spec(dev_x.loc[base_mask, feature], spec)
        base_dist = base_bins.value_counts(normalize=True)
        for compare_month in valid_months[1:]:
            compare_mask = months == compare_month
            compare_bins = apply_bin_spec(dev_x.loc[compare_mask, feature], spec)
            compare_dist = compare_bins.value_counts(normalize=True)
            psi = 0.0
            for bucket in all_buckets[feature]:
                expected = max(float(base_dist.get(bucket, 0.0)), epsilon)
                actual = max(float(compare_dist.get(bucket, 0.0)), epsilon)
                psi += (actual - expected) * np.log(actual / expected)
            status = "stable" if psi < warning_threshold else "warning" if psi < fail_threshold else "high_drift"
            detail_rows.append(
                {
                    "feature": feature,
                    "base_month": base_month,
                    "compare_month": compare_month,
                    "base_samples": base_samples,
                    "compare_samples": int(compare_mask.sum()),
                    "psi": float(psi),
                    "psi_status": status,
                }
            )

    detail = pd.DataFrame(detail_rows, columns=detail_columns)
    summary_rows = []
    for feature in features:
        values = detail.loc[detail["feature"] == feature, "psi"] if not detail.empty else pd.Series(dtype=float)
        max_psi = float(values.max()) if not values.empty else 0.0
        status = "stable" if max_psi < warning_threshold else "warning" if max_psi < fail_threshold else "high_drift"
        warning = ";".join(warnings)
        if len(valid_months) < 2:
            warning = ";".join(filter(None, [warning, "only_one_dev_month_available"]))
        summary_rows.append(
            {
                "feature": feature,
                "base_month": base_month,
                "base_samples": base_samples,
                "monthly_psi_max": max_psi,
                "monthly_psi_mean": float(values.mean()) if not values.empty else 0.0,
                "monthly_psi_p90": float(values.quantile(0.9)) if not values.empty else 0.0,
                "months_above_warning": int((values >= warning_threshold).sum()),
                "months_above_fail": int((values >= fail_threshold).sum()),
                "psi_status": status,
                "psi_warning": warning,
            }
        )
    return detail, pd.DataFrame(summary_rows, columns=summary_columns), specs, warnings


def classify_badrate_pattern(
    bad_rates: Iterable[float],
    sample_counts: Iterable[int],
    *,
    min_bin_samples: int = 30,
    min_rate_range: float = 0.02,
    tolerance: float = 1e-9,
) -> str:
    rates = np.asarray(list(bad_rates), dtype=float)
    counts = np.asarray(list(sample_counts), dtype=int)
    valid = np.isfinite(rates)
    rates, counts = rates[valid], counts[valid]
    if len(rates) < 3 or np.any(counts < int(min_bin_samples)):
        return "insufficient_sample"
    diffs = np.diff(rates)
    if np.all(diffs >= -tolerance):
        return "monotonic_increasing"
    if np.all(diffs <= tolerance):
        return "monotonic_decreasing"
    if float(np.max(rates) - np.min(rates)) < min_rate_range:
        return "weak_pattern"
    minimum = int(np.argmin(rates))
    maximum = int(np.argmax(rates))
    if 0 < minimum < len(rates) - 1 and np.all(np.diff(rates[: minimum + 1]) <= tolerance) and np.all(np.diff(rates[minimum:]) >= -tolerance):
        return "u_shape"
    if 0 < maximum < len(rates) - 1 and np.all(np.diff(rates[: maximum + 1]) >= -tolerance) and np.all(np.diff(rates[maximum:]) <= tolerance):
        return "inverted_u_shape"
    return "irregular"


def badrate_evidence(
    dev_x: pd.DataFrame,
    dev_y: pd.Series,
    features: Iterable[str],
    specs: Mapping[str, BinSpec],
    *,
    n_bins: int = 10,
    min_bin_samples: int = 30,
    min_bin_ratio: float = 0.01,
    min_rate_range: float = 0.02,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    patterns: list[dict[str, Any]] = []
    y = pd.Series(dev_y, index=dev_x.index).astype(int)
    for feature in features:
        if feature not in dev_x.columns:
            continue
        spec = specs.get(feature) or fit_bin_spec(dev_x[feature], n_bins=n_bins)
        bins = apply_bin_spec(dev_x[feature], spec)
        frame = pd.DataFrame({"bin": bins, "target": y})
        grouped = frame.groupby("bin", dropna=False, observed=False)["target"].agg(["count", "sum", "mean"]).reset_index()
        ordered = [bucket for bucket in [*spec.labels, OTHER_BUCKET, MISSING_BUCKET] if bucket in set(grouped["bin"])]
        grouped["_order"] = grouped["bin"].map({bucket: index for index, bucket in enumerate(ordered)})
        grouped = grouped.sort_values("_order")
        non_missing = grouped.loc[~grouped["bin"].isin([MISSING_BUCKET, OTHER_BUCKET])]
        pattern = classify_badrate_pattern(
            non_missing["mean"], non_missing["count"], min_bin_samples=min_bin_samples, min_rate_range=min_rate_range
        )
        sparse = bool(((grouped["count"] < min_bin_samples) | (grouped["count"] / max(len(frame), 1) < min_bin_ratio)).any())
        patterns.append({"feature": feature, "badrate_pattern": pattern, "sparse_bin_flag": sparse})
        for item in grouped.itertuples(index=False):
            rows.append(
                {
                    "feature": feature,
                    "bin": item.bin,
                    "sample_count": int(item.count),
                    "sample_ratio": float(item.count / max(len(frame), 1)),
                    "bad_count": int(item.sum),
                    "bad_rate": float(item.mean),
                    "missing_flag": item.bin == MISSING_BUCKET,
                    "bin_boundary": _bin_boundary(spec, item.bin),
                }
            )
    return (
        pd.DataFrame(
            rows,
            columns=["feature", "bin", "sample_count", "sample_ratio", "bad_count", "bad_rate", "missing_flag", "bin_boundary"],
        ),
        pd.DataFrame(patterns, columns=["feature", "badrate_pattern", "sparse_bin_flag"]),
    )


def monthly_badrate_evidence(
    dev_x: pd.DataFrame,
    dev_y: pd.Series,
    dev_months: pd.Series,
    features: Iterable[str],
    specs: Mapping[str, BinSpec],
    *,
    inversion_correlation: float = -0.30,
    min_month_bin_samples: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    months = pd.Series(dev_months, index=dev_x.index, dtype="string")
    y = pd.Series(dev_y, index=dev_x.index).astype(int)
    for feature in features:
        spec = specs.get(feature)
        if spec is None or feature not in dev_x.columns:
            summaries.append({"feature": feature, "relationship_stability": "insufficient_data", "months_with_reversal": 0})
            continue
        bins = apply_bin_spec(dev_x[feature], spec)
        frame = pd.DataFrame({"month": months, "bin": bins, "target": y}).dropna(subset=["month"])
        base_order: pd.Series | None = None
        reversed_months = 0
        comparable_months = 0
        for month in sorted(frame["month"].unique().tolist()):
            monthly = frame.loc[frame["month"] == month]
            grouped = monthly.groupby("bin", observed=False)["target"].agg(["count", "sum", "mean"]).reset_index()
            risk = grouped.loc[~grouped["bin"].isin([MISSING_BUCKET, OTHER_BUCKET])].set_index("bin")["mean"]
            is_base_month = base_order is None
            if is_base_month:
                base_order = risk
            common = base_order.index.intersection(risk.index) if base_order is not None else []
            corr = float(base_order.loc[common].corr(risk.loc[common], method="spearman")) if len(common) >= 3 else np.nan
            reversal = bool(not is_base_month and np.isfinite(corr) and corr <= inversion_correlation)
            comparable_months += int(not is_base_month and np.isfinite(corr))
            reversed_months += int(reversal)
            for item in grouped.itertuples(index=False):
                rows.append(
                    {
                        "feature": feature,
                        "month": str(month),
                        "bin": item.bin,
                        "sample_count": int(item.count),
                        "sample_ratio": float(item.count / max(len(monthly), 1)),
                        "bad_count": int(item.sum),
                        "bad_rate": float(item.mean),
                        "risk_rank_correlation_vs_base": corr,
                        "risk_order_reversal": reversal,
                        "insufficient_bin_sample": bool(item.count < min_month_bin_samples),
                    }
                )
        stability = "unstable" if reversed_months else "stable" if comparable_months else "insufficient_data"
        summaries.append({"feature": feature, "relationship_stability": stability, "months_with_reversal": reversed_months})
    detail = pd.DataFrame(
            rows,
            columns=[
                "feature", "month", "bin", "sample_count", "sample_ratio", "bad_count", "bad_rate",
                "risk_rank_correlation_vs_base", "risk_order_reversal", "insufficient_bin_sample",
            ],
        )
    summary = pd.DataFrame(summaries, columns=["feature", "relationship_stability", "months_with_reversal"])
    if not detail.empty:
        detail = detail.merge(summary[["feature", "relationship_stability"]], on="feature", how="left")
        detail = detail.rename(columns={"relationship_stability": "monthly_relationship_stability_flag"})
    else:
        detail["monthly_relationship_stability_flag"] = pd.Series(dtype="object")
    return detail, summary


def importance_stability_evidence(
    features: Iterable[str],
    d03_detail: pd.DataFrame,
    d05_importance: pd.DataFrame,
    *,
    top_n: int = 20,
    max_rank_std: float = 10.0,
    min_selection_rate: float = 0.60,
) -> pd.DataFrame:
    """Aggregate available multi-seed/bagging evidence; fall back to D05 once."""
    frames = []
    if isinstance(d03_detail, pd.DataFrame) and not d03_detail.empty and {"feature", "gain"}.issubset(d03_detail.columns):
        frame = d03_detail.copy()
        frame["run"] = frame["round"] if "round" in frame.columns else np.arange(len(frame))
        frame["importance"] = pd.to_numeric(frame["gain"], errors="coerce").fillna(0.0)
        frame["rank"] = frame.groupby("run")["importance"].rank(method="min", ascending=False)
        frame["selected_top_n"] = frame["rank"] <= int(top_n)
        frames.append(frame[["feature", "run", "importance", "rank", "selected_top_n"]])
    elif isinstance(d05_importance, pd.DataFrame) and not d05_importance.empty:
        frame = d05_importance.copy()
        frame["run"] = "d05_single_run"
        frame["importance"] = pd.to_numeric(frame.get("gain", 0.0), errors="coerce").fillna(0.0)
        frame["rank"] = pd.to_numeric(frame.get("rank"), errors="coerce")
        frame["selected_top_n"] = frame["rank"] <= int(top_n)
        frames.append(frame[["feature", "run", "importance", "rank", "selected_top_n"]])
    evidence = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    run_ids = evidence["run"].drop_duplicates().tolist() if not evidence.empty else []
    run_fallback_rank = evidence.groupby("run")["rank"].max().add(1).to_dict() if not evidence.empty else {}
    rows = []
    for feature in features:
        values = evidence.loc[evidence["feature"] == feature] if not evidence.empty else pd.DataFrame()
        if run_ids:
            values = values.drop_duplicates("run", keep="last").set_index("run").reindex(run_ids)
            values["feature"] = feature
            values["importance"] = values["importance"].fillna(0.0)
            values["rank"] = values["rank"].fillna(pd.Series(run_fallback_rank))
            values["selected_top_n"] = values["selected_top_n"].fillna(False).astype(bool)
            values = values.reset_index(names="run")
        run_count = len(run_ids)
        importance_median = float(values["importance"].median()) if not values.empty else np.nan
        importance_std = float(values["importance"].std(ddof=0)) if not values.empty else np.nan
        rank_median = float(values["rank"].median()) if not values.empty else np.nan
        rank_std = float(values["rank"].std(ddof=0)) if not values.empty else np.nan
        selection_rate = float(values["selected_top_n"].mean()) if not values.empty else np.nan
        if run_count < 2:
            status = "unknown_importance_stability"
        elif rank_std > max_rank_std or selection_rate < min_selection_rate:
            status = "unstable_importance"
        else:
            status = "stable_importance"
        rows.append(
            {
                "feature": feature,
                "importance_median": importance_median,
                "importance_std": importance_std,
                "rank_median": rank_median,
                "rank_std": rank_std,
                "top_n_selection_rate": selection_rate,
                "importance_runs": run_count,
                "importance_status": status,
            }
        )
    return pd.DataFrame(rows)


def _map_column(frame: pd.DataFrame, key: str, value: str) -> dict[str, Any]:
    if not isinstance(frame, pd.DataFrame) or frame.empty or key not in frame.columns or value not in frame.columns:
        return {}
    return frame.drop_duplicates(key, keep="last").set_index(key)[value].to_dict()


def build_feature_decision_ledger(
    initial_features: Iterable[str],
    *,
    selected_features: Iterable[str],
    force_keep_features: Iterable[str] = (),
    preprocess_stats: pd.DataFrame,
    d01_detail: pd.DataFrame,
    corr_drops: pd.DataFrame,
    leakage: pd.DataFrame,
    psi_summary: pd.DataFrame,
    patterns: pd.DataFrame,
    relationship: pd.DataFrame,
    importance: pd.DataFrame,
    d03_features: Iterable[str] = (),
    d04_features: Iterable[str] = (),
    high_importance_top_n: int = 20,
) -> pd.DataFrame:
    selected, force_keep = set(selected_features), set(force_keep_features)
    d03_set, d04_set = set(d03_features), set(d04_features)
    missing_rate = {key: 1.0 - float(value) for key, value in _map_column(preprocess_stats, "feature", "non_null_rate").items()}
    maps = {
        "preprocess_drop": _map_column(preprocess_stats, "feature", "drop_reason"),
        "iv": _map_column(d01_detail, "feature", "iv"),
        "d01_drop": _map_column(d01_detail, "feature", "drop_reason"),
        "leakage_status": _map_column(leakage, "feature_name", "leakage_status"),
        "leakage_reason": _map_column(leakage, "feature_name", "leakage_reason"),
        "monthly_psi_max": _map_column(psi_summary, "feature", "monthly_psi_max"),
        "monthly_psi_mean": _map_column(psi_summary, "feature", "monthly_psi_mean"),
        "psi_status": _map_column(psi_summary, "feature", "psi_status"),
        "pattern": _map_column(patterns, "feature", "badrate_pattern"),
        "sparse": _map_column(patterns, "feature", "sparse_bin_flag"),
        "business_contradiction": _map_column(patterns, "feature", "business_contradiction"),
        "relationship": _map_column(relationship, "feature", "relationship_stability"),
        "importance_median": _map_column(importance, "feature", "importance_median"),
        "rank_median": _map_column(importance, "feature", "rank_median"),
        "rank_std": _map_column(importance, "feature", "rank_std"),
        "selection_rate": _map_column(importance, "feature", "top_n_selection_rate"),
        "importance_status": _map_column(importance, "feature", "importance_status"),
        "correlation_group": _map_column(corr_drops, "feature", "kept_feature"),
    }
    rows = []
    for feature in initial_features:
        leakage_status = maps["leakage_status"].get(feature, "unknown_metadata")
        pattern = maps["pattern"].get(feature, "not_reviewed")
        relationship_status = maps["relationship"].get(feature, "not_reviewed")
        rank_median = maps["rank_median"].get(feature, np.nan)
        high_importance = bool(pd.notna(rank_median) and float(rank_median) <= int(high_importance_top_n))
        business_review = bool(maps["business_contradiction"].get(feature, False))
        badrate_review = business_review or high_importance and (
            pattern == "irregular" or bool(maps["sparse"].get(feature, False)) or relationship_status == "unstable"
        )
        unstable_review = high_importance and maps["importance_status"].get(feature) == "unstable_importance"
        psi_review = maps["psi_status"].get(feature) == "high_drift"
        review_required = bool(badrate_review or unstable_review or psi_review or leakage_status in {"warning", "unknown_metadata"})

        reasons = []
        if leakage_status == "failed":
            final_status = "dropped"
            reasons.append("deterministic_leakage_failed")
        elif feature in force_keep and feature in selected:
            final_status = "force_kept"
            reasons.append("configured_force_keep")
        elif feature not in selected:
            final_status = "dropped"
            preprocess_reason = maps["preprocess_drop"].get(feature)
            d01_reason = maps["d01_drop"].get(feature)
            if feature in force_keep:
                reasons.append("force_keep_feature_unavailable")
            elif preprocess_reason:
                reasons.append(preprocess_reason)
            elif d01_reason and d01_reason != "kept":
                reasons.append(d01_reason)
            elif feature in maps["correlation_group"]:
                reasons.append("global_corr")
            elif d03_set and feature not in d03_set:
                reasons.append("random_importance")
            elif d04_set and feature not in d04_set:
                reasons.append("null_importance")
            else:
                reasons.append("not_in_final_top_n")
        elif review_required:
            final_status = "review_required"
            reasons.append("retained_with_review_flags")
        else:
            final_status = "selected"
            reasons.append("passed_existing_selection_policy")
        if badrate_review:
            reasons.append("high_importance_badrate_review")
        if unstable_review:
            reasons.append("high_importance_unstable")
        if psi_review:
            reasons.append("monthly_psi_high_drift")
        if leakage_status in {"warning", "unknown_metadata"}:
            reasons.append("leakage_" + leakage_status)

        rows.append(
            {
                "feature": feature,
                "force_keep": feature in force_keep,
                "missing_rate": missing_rate.get(feature, np.nan),
                "iv": maps["iv"].get(feature, np.nan),
                "monthly_psi_max": maps["monthly_psi_max"].get(feature, np.nan),
                "monthly_psi_mean": maps["monthly_psi_mean"].get(feature, np.nan),
                "psi_status": maps["psi_status"].get(feature, "not_reviewed"),
                "leakage_status": leakage_status,
                "leakage_reason": maps["leakage_reason"].get(feature, "missing_metadata"),
                "badrate_pattern": pattern,
                "badrate_review_required": badrate_review,
                "business_contradiction": bool(maps["business_contradiction"].get(feature, False)),
                "relationship_stability": relationship_status,
                "importance_median": maps["importance_median"].get(feature, np.nan),
                "rank_median": rank_median,
                "rank_std": maps["rank_std"].get(feature, np.nan),
                "selection_rate": maps["selection_rate"].get(feature, np.nan),
                "correlation_group": maps["correlation_group"].get(feature, ""),
                "final_status": final_status,
                "review_required": review_required and final_status != "dropped",
                "decision_reason": ";".join(reasons),
            }
        )
    return pd.DataFrame(rows)


def resolve_month_column(raw_df: pd.DataFrame, config: Mapping[str, Any]) -> str | None:
    review_cfg = config.get("feature_risk_review", {}) or {}
    explicit = review_cfg.get("month_column")
    if explicit and str(explicit) in raw_df.columns:
        return str(explicit)
    input_cfg = config.get("input", {}) or {}
    candidates = [input_cfg.get("period_column"), "sample_month", "mdl_month", "month", "ds", "mdl_dte", "sample_date"]
    candidates.extend(
        column
        for column in raw_df.columns
        if any(token in str(column).lower() for token in ("month", "date", "dte", "period"))
        or str(column).lower() == "ds"
        or str(column).lower().endswith("_ds")
    )
    for candidate in candidates:
        if candidate and candidate in raw_df.columns and pd.to_datetime(raw_df[candidate], errors="coerce").notna().any():
            return str(candidate)
    return None


def generate_feature_risk_review(
    *,
    project_dir: Path,
    output_dir: Path,
    raw_df: pd.DataFrame,
    feature_frame: pd.DataFrame,
    config: Mapping[str, Any],
    initial_features: list[str],
    available_features: list[str],
    selected_features: list[str],
    preprocess_stats: pd.DataFrame,
    d01_detail: pd.DataFrame,
    corr_drops: pd.DataFrame,
    d03_features: list[str],
    d03_detail: pd.DataFrame,
    d04_features: list[str],
    d05_importance: pd.DataFrame,
) -> tuple[list[str], list[Path], dict[str, Any]]:
    """Generate all review artifacts inside the existing feature-refine output."""
    review_cfg = config.get("feature_risk_review", {}) or {}
    if not review_cfg.get("enabled", True):
        return selected_features, [], {"enabled": False}
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_map_value = (config.get("input", {}) or {}).get("feature_map")
    feature_map_path = None
    if feature_map_value:
        feature_map_path = Path(str(feature_map_value))
        if not feature_map_path.is_absolute():
            feature_map_path = project_dir / feature_map_path
    metadata = load_feature_metadata(project_dir=project_dir, config=config, feature_map_path=feature_map_path)
    leakage = review_feature_leakage(initial_features, metadata, prediction_time=review_cfg.get("prediction_time"))

    input_cfg = config.get("input", {}) or {}
    split_column = str(input_cfg["split_column"])
    label_column = str(input_cfg["label_column"])
    force_keep = list(review_cfg.get("force_keep_features", []) or [])
    effective_selected = list(dict.fromkeys([*selected_features, *[feature for feature in force_keep if feature in available_features]]))
    dev_mask = (raw_df[split_column] == input_cfg["train_value"]) & raw_df[label_column].isin([0, 1])
    dev_x = feature_frame.loc[dev_mask, [feature for feature in effective_selected if feature in feature_frame.columns]].copy()
    dev_y = raw_df.loc[dev_mask, label_column].astype(int)
    review_features = list(dev_x.columns)
    month_column = resolve_month_column(raw_df, config)
    dev_months = normalize_months(raw_df.loc[dev_mask, month_column]) if month_column else pd.Series(pd.NA, index=dev_x.index, dtype="string")

    psi_cfg = review_cfg.get("psi", {}) or {}
    psi_detail, psi_summary, specs, psi_warnings = monthly_psi_evidence(
        dev_x,
        dev_months,
        review_features,
        n_bins=int(psi_cfg.get("n_bins", 10)),
        warning_threshold=float(psi_cfg.get("warning_threshold", 0.10)),
        fail_threshold=float(psi_cfg.get("fail_threshold", 0.25)),
        min_base_samples=int(psi_cfg.get("min_base_samples", 500)),
    )
    bad_cfg = review_cfg.get("badrate", {}) or {}
    bad_bins, patterns = badrate_evidence(
        dev_x,
        dev_y,
        review_features,
        specs,
        n_bins=int(bad_cfg.get("n_bins", psi_cfg.get("n_bins", 10))),
        min_bin_samples=int(bad_cfg.get("min_bin_samples", 30)),
        min_bin_ratio=float(bad_cfg.get("min_bin_ratio", 0.01)),
        min_rate_range=float(bad_cfg.get("min_rate_range", 0.02)),
    )
    expected_direction = {}
    if "expected_risk_direction" in metadata.columns:
        expected_direction = metadata.set_index("feature_name")["expected_risk_direction"].astype("string").str.lower().to_dict()
    if patterns.empty:
        patterns["business_contradiction"] = pd.Series(dtype=bool)
    else:
        patterns["business_contradiction"] = patterns.apply(
            lambda row: (
                expected_direction.get(row["feature"]) in {"increasing", "positive", "higher_is_riskier"}
                and row["badrate_pattern"] == "monotonic_decreasing"
            )
            or (
                expected_direction.get(row["feature"]) in {"decreasing", "negative", "lower_is_riskier"}
                and row["badrate_pattern"] == "monotonic_increasing"
            ),
            axis=1,
        )
    importance_cfg = review_cfg.get("importance", {}) or {}
    importance = importance_stability_evidence(
        review_features,
        d03_detail,
        d05_importance,
        top_n=int(importance_cfg.get("top_n", 20)),
        max_rank_std=float(importance_cfg.get("max_rank_std", 10.0)),
        min_selection_rate=float(importance_cfg.get("min_selection_rate", 0.60)),
    )
    high_importance = importance.loc[
        pd.to_numeric(importance["rank_median"], errors="coerce") <= int(importance_cfg.get("top_n", 20)), "feature"
    ].tolist()
    monthly_bad, relationship = monthly_badrate_evidence(
        dev_x,
        dev_y,
        dev_months,
        high_importance,
        specs,
        inversion_correlation=float(bad_cfg.get("inversion_correlation", -0.30)),
        min_month_bin_samples=int(bad_cfg.get("min_month_bin_samples", 20)),
    )
    ledger = build_feature_decision_ledger(
        initial_features,
        selected_features=effective_selected,
        force_keep_features=force_keep,
        preprocess_stats=preprocess_stats,
        d01_detail=d01_detail,
        corr_drops=corr_drops,
        leakage=leakage,
        psi_summary=psi_summary,
        patterns=patterns,
        relationship=relationship,
        importance=importance,
        d03_features=d03_features,
        d04_features=d04_features,
        high_importance_top_n=int(importance_cfg.get("top_n", 20)),
    )
    failed = set(leakage.loc[leakage["leakage_status"] == "failed", "feature_name"])
    final_features = [feature for feature in effective_selected if feature not in failed]

    artifacts = {
        "feature_leakage_review.csv": leakage,
        "feature_monthly_psi.csv": psi_detail,
        "feature_monthly_psi_summary.csv": psi_summary,
        "feature_badrate_bins.csv": bad_bins,
        "feature_monthly_badrate.csv": monthly_bad,
        "feature_importance_stability.csv": importance,
        "feature_decision_ledger.csv": ledger,
    }
    paths: list[Path] = []
    for name, frame in artifacts.items():
        path = output_dir / name
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        paths.append(path)

    def feature_list(mask: pd.Series) -> list[str]:
        return ledger.loc[mask, "feature"].astype(str).tolist()

    summary = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "feature_refine_existing_candidate_results",
        "month_column": month_column,
        "dev_only_psi": True,
        "oot_used_for_feature_psi": False,
        "config": _json_value(review_cfg),
        "warnings": psi_warnings + ([] if month_column else ["month_column_missing_or_unparseable"]),
        "counts": {
            "initial_features": len(initial_features),
            "reviewed_model_candidates": len(review_features),
            "leakage_failed": int((ledger["leakage_status"] == "failed").sum()),
            "unknown_metadata": int((ledger["leakage_status"] == "unknown_metadata").sum()),
            "high_drift": int((ledger["psi_status"] == "high_drift").sum()),
            "review_required": int(ledger["review_required"].sum()),
        },
        "leakage_failed_features": feature_list(ledger["leakage_status"] == "failed"),
        "metadata_incomplete_features": feature_list(ledger["leakage_status"] == "unknown_metadata"),
        "psi_high_drift_features": feature_list(ledger["psi_status"] == "high_drift"),
        "high_importance_abnormal_badrate_features": feature_list(ledger["badrate_review_required"]),
        "high_importance_unstable_features": feature_list(ledger["decision_reason"].str.contains("high_importance_unstable", na=False)),
        "top_review_features": ledger.loc[ledger["review_required"]].sort_values("rank_median", na_position="last")["feature"].head(30).tolist(),
        "artifacts": list(artifacts),
    }
    json_path = output_dir / "feature_risk_review.json"
    json_path.write_text(json.dumps(_json_value(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    paths.append(json_path)
    md_path = output_dir / "feature_risk_review.md"
    sections = [
        "# Feature Risk Review",
        "",
        "本报告由确定性 Python 逻辑生成；名称规则仅产生 warning，PSI 或非单调关系不会单独删除特征。",
        "",
        f"- DEV PSI Base 月：{psi_summary['base_month'].iloc[0] if not psi_summary.empty else '不可用'}",
        "- OOT 是否参与特征 PSI：否",
        f"- 确定性 leakage failed：{summary['counts']['leakage_failed']}",
        f"- metadata 不完整：{summary['counts']['unknown_metadata']}",
        f"- PSI high drift：{summary['counts']['high_drift']}",
        f"- 需要人工审查：{summary['counts']['review_required']}",
    ]
    for title, key in [
        ("确定性 Leakage Failed", "leakage_failed_features"),
        ("Metadata 不完整", "metadata_incomplete_features"),
        ("PSI 高漂移", "psi_high_drift_features"),
        ("高重要性且 Badrate 异常", "high_importance_abnormal_badrate_features"),
        ("高重要性且 Importance 不稳定", "high_importance_unstable_features"),
        ("人工审查 Top Features", "top_review_features"),
    ]:
        sections.extend(["", f"## {title}", ""])
        values = summary[key]
        sections.extend([f"- `{feature}`" for feature in values] or ["- 无"])
    md_path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    paths.append(md_path)
    return final_features, paths, summary


__all__ = [
    "BinSpec",
    "MISSING_BUCKET",
    "apply_bin_spec",
    "badrate_evidence",
    "build_feature_decision_ledger",
    "classify_badrate_pattern",
    "fit_bin_spec",
    "generate_feature_risk_review",
    "importance_stability_evidence",
    "monthly_badrate_evidence",
    "monthly_psi_evidence",
    "normalize_months",
    "resolve_month_column",
]
