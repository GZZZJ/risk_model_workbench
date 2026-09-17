"""Deterministic feature availability and leakage evidence.

Name-based rules are deliberately warnings only. A feature is failed only
when supplied time metadata proves that it was unavailable at prediction time
or that its derivation window overlaps the label window.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


LEAKAGE_WARNING_TERMS = (
    "overdue",
    "dpd",
    "future",
    "post",
    "repayment_after",
    "outcome",
    "target",
)

METADATA_COLUMNS = (
    "feature_name",
    "source_table",
    "source_system",
    "feature_definition",
    "observation_time",
    "available_time",
    "label_window_start",
    "derivation_logic",
    "owner",
    "domain",
)


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _timestamp(value: object) -> pd.Timestamp | None:
    text = _text(value)
    if not text:
        return None
    try:
        parsed = pd.to_datetime(text, errors="raise")
    except (TypeError, ValueError, OverflowError):
        return None
    if isinstance(parsed, pd.DatetimeIndex):
        return None
    return pd.Timestamp(parsed)


def _warning_terms(row: Mapping[str, Any]) -> list[str]:
    haystack = " ".join(
        _text(row.get(column)).lower()
        for column in ("feature_name", "source_table", "source_system", "feature_definition", "derivation_logic")
    )
    return [
        term
        for term in LEAKAGE_WARNING_TERMS
        if re.search(rf"(^|[^a-z0-9]){re.escape(term)}([^a-z0-9]|$)", haystack)
    ]


def normalize_feature_metadata(
    metadata: pd.DataFrame | Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]] | None,
) -> pd.DataFrame:
    """Normalize supported metadata shapes to one row per ``feature_name``."""
    if metadata is None:
        return pd.DataFrame(columns=METADATA_COLUMNS)
    if isinstance(metadata, pd.DataFrame):
        frame = metadata.copy()
    elif isinstance(metadata, Mapping):
        rows = []
        for feature, payload in metadata.items():
            row = dict(payload)
            row.setdefault("feature_name", feature)
            rows.append(row)
        frame = pd.DataFrame(rows)
    else:
        frame = pd.DataFrame([dict(row) for row in metadata])

    if "feature_name" not in frame.columns and "feature" in frame.columns:
        frame = frame.rename(columns={"feature": "feature_name"})
    if "feature_name" not in frame.columns and "output_feature" in frame.columns:
        frame = frame.rename(columns={"output_feature": "feature_name"})
    if "feature_name" not in frame.columns:
        return pd.DataFrame(columns=METADATA_COLUMNS)
    for column in METADATA_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame["feature_name"] = frame["feature_name"].map(_text)
    return frame.loc[frame["feature_name"] != ""].drop_duplicates("feature_name", keep="last")


def load_feature_metadata(
    *,
    project_dir: Path,
    config: Mapping[str, Any],
    feature_map_path: Path | None = None,
) -> pd.DataFrame:
    """Load optional review metadata and enrich it with the existing feature map."""
    review_cfg = config.get("feature_risk_review", {}) or {}
    metadata_value = review_cfg.get("metadata_path")
    frames: list[pd.DataFrame] = []
    if feature_map_path is not None and feature_map_path.exists():
        frames.append(pd.read_csv(feature_map_path, encoding="utf-8-sig"))
    if metadata_value:
        path = Path(str(metadata_value))
        if not path.is_absolute():
            path = project_dir / path
        if path.exists():
            if path.suffix.lower() == ".jsonl":
                frames.append(pd.read_json(path, lines=True))
            elif path.suffix.lower() == ".json":
                frames.append(normalize_feature_metadata(json.loads(path.read_text(encoding="utf-8"))))
            else:
                frames.append(pd.read_csv(path, encoding="utf-8-sig"))
    if not frames:
        return normalize_feature_metadata(None)

    normalized = [normalize_feature_metadata(frame) for frame in frames]
    merged = normalized[0].replace("", pd.NA).set_index("feature_name")
    for frame in normalized[1:]:
        incoming = frame.replace("", pd.NA).set_index("feature_name")
        merged = incoming.combine_first(merged)
    return normalize_feature_metadata(merged.reset_index())


def review_feature_leakage(
    features: Iterable[str],
    metadata: pd.DataFrame | Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]] | None,
    *,
    prediction_time: object = None,
) -> pd.DataFrame:
    """Return passed/warning/failed/unknown_metadata evidence for every feature."""
    frame = normalize_feature_metadata(metadata)
    by_feature = {str(row["feature_name"]): row for row in frame.to_dict("records")}
    global_prediction = _timestamp(prediction_time)
    rows: list[dict[str, Any]] = []

    for feature in features:
        source = by_feature.get(str(feature), {"feature_name": str(feature)})
        row = {column: _text(source.get(column)) for column in METADATA_COLUMNS}
        row["feature_name"] = str(feature)
        warning_terms = _warning_terms(row)
        available = _timestamp(row.get("available_time"))
        prediction = global_prediction or _timestamp(row.get("observation_time"))
        label_start = _timestamp(row.get("label_window_start"))
        derivation_end = next(
            (
                value
                for value in (
                    _timestamp(source.get("derivation_window_end")),
                    _timestamp(source.get("feature_window_end")),
                    _timestamp(source.get("observation_time")),
                )
                if value is not None
            ),
            None,
        )

        failures: list[str] = []
        if available is not None and prediction is not None and available > prediction:
            failures.append("available_time_after_prediction_time")
        if derivation_end is not None and label_start is not None and derivation_end > label_start:
            failures.append("feature_window_overlaps_label_window")

        missing = []
        if available is None:
            missing.append("available_time")
        if prediction is None:
            missing.append("prediction_time")
        if label_start is None:
            missing.append("label_window_start")
        if derivation_end is None:
            missing.append("observation_time_or_derivation_window_end")
        if not row.get("derivation_logic"):
            missing.append("derivation_logic")

        warnings = [f"rule_based_signal:{term}" for term in warning_terms]
        if failures:
            status = "failed"
            reasons = failures + warnings
        elif missing:
            status = "unknown_metadata"
            reasons = ["missing_metadata:" + ",".join(missing)] + warnings
        elif warnings:
            status = "warning"
            reasons = warnings
        else:
            status = "passed"
            reasons = ["availability_and_label_window_checks_passed"]

        rows.append(
            {
                **row,
                "leakage_status": status,
                "leakage_reason": ";".join(reasons),
                "availability_check": (
                    "failed"
                    if "available_time_after_prediction_time" in failures
                    else "passed"
                    if available is not None and prediction is not None
                    else "unknown_metadata"
                ),
                "label_window_overlap": (
                    "failed"
                    if "feature_window_overlaps_label_window" in failures
                    else "passed"
                    if derivation_end is not None and label_start is not None
                    else "unknown_metadata"
                ),
                "rule_warning_terms": ",".join(warning_terms),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "LEAKAGE_WARNING_TERMS",
    "METADATA_COLUMNS",
    "load_feature_metadata",
    "normalize_feature_metadata",
    "review_feature_leakage",
]
