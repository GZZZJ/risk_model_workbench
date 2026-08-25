#!/usr/bin/env python3
"""Run per-table coarse feature prescreening for feature tables.

The flow is configuration-driven so a new model project can reuse the same
pre-refinement screening contract.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = next(
    (path for path in [SCRIPT_PATH, *SCRIPT_PATH.parents] if (path / "agent.py").exists()),
    SCRIPT_PATH.parent.parent,
)
sys.path.insert(0, str(REPO_ROOT))

from risk_model_workbench.dp_feather import (
    load_or_fetch_dp_feather,
    print_sql_review,
    write_dataset_metadata,
)
from risk_model_workbench.config import load_yaml
from risk_model_workbench.paths import project_config_path
from risk_model_workbench.progress import ProgressReporter


TARGET_COL = "ftr_30d_ord_flag"
SPLIT_COL = "final_flag"
DEV_VALUE = "DEV"
OOT_VALUE = "OOT"

D01_THRESHOLDS = {
    "empty": 0.95,
    "corr": 0.80,
    "iv": 0.005,
}
D02_PSI_THRESHOLD = 0.10

# All 8 monthly partitions: DEV data in 2025-06 ~ 2025-11, OOT in 2025-12 ~ 2026-01
DEV_PARTITION_DS_LIST = ["20250630", "20250731", "20250831", "20250930", "20251031", "20251130"]
OOT_PARTITION_DS_LIST = ["20251231", "20260131"]

DEFAULT_ROUND_NUM = 500
DEFAULT_RANDOM_SEED = 0
DEFAULT_WORKERS = 4
DEFAULT_STAGE = "feature_prescreen"


DEFAULT_PROJECT_DIR = Path.cwd()


@dataclass(frozen=True)
class BatchSelectSettings:
    feature_columns: str
    output_dir: str
    target_col: str
    split_col: str
    train_value: str
    valid_value: str
    partition_col: str | None
    train_partitions: list[str]
    valid_partitions: list[str]
    sample_where: str
    d01_thresholds: dict[str, float]
    d02_psi_threshold: float
    round_num: int
    random_seed: int
    workers: int
    dp_data_dir: str
    dp_metadata_dir: str


@dataclass(frozen=True)
class PrescreenAction:
    project_dir: str | Path
    config: str = "configs/feature_select.yaml"
    feature_select_code_dir: str | None = None
    feature_columns: str | None = None
    output_dir: str | None = None
    table: tuple[str, ...] = ()
    max_tables: int | None = None
    round_num: int | None = None
    random_seed: int | None = None
    force: bool = False
    use_native: bool = False
    dev_partition_ds: str | None = None
    oot_partition_ds: str | None = None
    partition_col: str | None = None
    sample_where: str | None = None
    workers: int | None = None
    dry_run_sql: bool = False
    refresh_dp_cache: bool = False
    sql_approved: bool = False
    run_dir: str | Path | None = None
    stage: str = DEFAULT_STAGE


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run coarse feature prescreening by feature table.")
    parser.add_argument("--project-dir", default=str(DEFAULT_PROJECT_DIR), help="Project workspace directory.")
    parser.add_argument(
        "--config",
        default="configs/feature_select.yaml",
        help="Feature selection config path, relative to project-dir unless absolute.",
    )
    parser.add_argument(
        "--feature-select-code-dir",
        default=os.environ.get("FEATURE_SELECT_V2_CODE_DIR"),
        help=(
            "Path to feature-select-v2/scripts/code. If omitted, the script searches common paths "
            "under project-dir, cwd, and the script directory."
        ),
    )
    parser.add_argument(
        "--feature-columns",
        default=None,
        help="Feature metadata CSV, relative to project-dir unless absolute.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory, relative to project-dir unless absolute.",
    )
    parser.add_argument("--table", action="append", help="Only run the specified full table name. Repeatable.")
    parser.add_argument("--max-tables", type=int, default=None, help="Optional cap for smoke runs.")
    parser.add_argument("--round-num", type=int, default=None, help="Feature batch size for coarse prescreening.")
    parser.add_argument("--random-seed", type=int, default=None, help="Random seed for feature order.")
    parser.add_argument("--force", action="store_true", help="Overwrite table checkpoints if they already exist.")
    parser.add_argument(
        "--use-native",
        action="store_true",
        help="Force feature-select-v2 native selector instead of toad. Default auto-detects toad.",
    )
    parser.add_argument(
        "--dev-partition-ds",
        default=None,
        help="Comma-separated ds values for DEV partitions.",
    )
    parser.add_argument(
        "--oot-partition-ds",
        default=None,
        help="Comma-separated ds values for OOT partitions.",
    )
    parser.add_argument("--partition-col", default=None, help="Partition column used in sample SQL.")
    parser.add_argument("--sample-where", default=None, help="Where clause appended to every feature table sample query.")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers.")
    parser.add_argument(
        "--dry-run-sql",
        action="store_true",
        help="Write and print DP sample SQL metadata only; do not query DP or run feature prescreening.",
    )
    parser.add_argument(
        "--refresh-dp-cache",
        action="store_true",
        help="Refresh local feather cache from DP after SQL approval.",
    )
    parser.add_argument(
        "--sql-approved",
        action="store_true",
        help="Confirm that all generated DP SQL has been reviewed and may be executed.",
    )
    parser.add_argument("--run-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--stage", default=DEFAULT_STAGE, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def load_batch_settings(project_dir: Path, args: PrescreenAction | argparse.Namespace) -> BatchSelectSettings:
    config_path = resolve_project_path(project_dir, args.config)
    feature_config = load_yaml(config_path).get("feature_select", {})
    prescreen_cfg = feature_config.get("prescreen", {}) or feature_config.get("d01_d02", {}) or {}
    thresholds_cfg = prescreen_cfg.get("thresholds", {}) or feature_config.get("thresholds", {}) or {}
    sampling_cfg = prescreen_cfg.get("sampling", {}) or {}
    dp_feather_cfg = prescreen_cfg.get("dp_feather", {}) or {}

    project_cfg_path = project_config_path(project_dir)
    split_cfg = load_yaml(project_cfg_path).get("split", {})
    data_cfg = load_yaml(project_cfg_path).get("data", {})
    target_col = prescreen_cfg.get("target_col") or data_cfg.get("target_column") or TARGET_COL
    split_col = prescreen_cfg.get("split_col") or split_cfg.get("source_column") or SPLIT_COL
    train_value = prescreen_cfg.get("train_value") or (split_cfg.get("ins_values") or [DEV_VALUE])[0]
    valid_value = prescreen_cfg.get("valid_value") or (split_cfg.get("oot_values") or [OOT_VALUE])[0]
    partition_col = args.partition_col or sampling_cfg.get("partition_col") or data_cfg.get("period_column")

    train_partitions = split_csv(args.dev_partition_ds) or list(sampling_cfg.get("train_partitions", []))
    valid_partitions = split_csv(args.oot_partition_ds) or list(sampling_cfg.get("valid_partitions", []))
    sample_where = args.sample_where or sampling_cfg.get("where")
    if not sample_where:
        sample_where = f"{split_col} in ('{train_value}', '{valid_value}') and {target_col} in (0,1)"

    return BatchSelectSettings(
        feature_columns=args.feature_columns or prescreen_cfg.get("feature_columns", "data/profile/feature_metadata/feature_columns.csv"),
        output_dir=args.output_dir or prescreen_cfg.get("output_dir", "runs/feature_prescreen"),
        target_col=target_col,
        split_col=split_col,
        train_value=train_value,
        valid_value=valid_value,
        partition_col=partition_col,
        train_partitions=train_partitions,
        valid_partitions=valid_partitions,
        sample_where=sample_where,
        d01_thresholds={
            "empty": float(thresholds_cfg.get("empty", D01_THRESHOLDS["empty"])),
            "corr": float(thresholds_cfg.get("corr", D01_THRESHOLDS["corr"])),
            "iv": float(thresholds_cfg.get("iv", D01_THRESHOLDS["iv"])),
            "__psi_fail": float(thresholds_cfg.get("psi_fail", 0.25)),
            "__psi_bins": float(thresholds_cfg.get("psi_bins", 10)),
            "__psi_min_base_samples": float(thresholds_cfg.get("psi_min_base_samples", 500)),
        },
        d02_psi_threshold=float(thresholds_cfg.get("psi", D02_PSI_THRESHOLD)),
        round_num=int(args.round_num if args.round_num is not None else prescreen_cfg.get("round_num", DEFAULT_ROUND_NUM)),
        random_seed=int(args.random_seed if args.random_seed is not None else prescreen_cfg.get("random_seed", DEFAULT_RANDOM_SEED)),
        workers=int(args.workers if args.workers is not None else prescreen_cfg.get("workers", DEFAULT_WORKERS)),
        dp_data_dir=dp_feather_cfg.get("data_dir", "data/local/dp_feather/feature_prescreen"),
        dp_metadata_dir=dp_feather_cfg.get("metadata_dir", "data/profile/dp_feather_datasets/feature_prescreen"),
    )


def find_feature_select_code_dir(project_dir: Path, explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend(
        [
            project_dir / "vendor/feature-select-v2/scripts/code",
            project_dir.parent.parent / "vendor/feature-select-v2/scripts/code",
            Path.cwd() / "vendor/feature-select-v2/scripts/code",
            SCRIPT_PATH.parent / "feature-select-v2/scripts/code",
            SCRIPT_PATH.parent.parent / "feature-select-v2/scripts/code",
        ]
    )

    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "utils" / "feature_select.py").exists():
            return candidate

    searched = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        "Cannot locate feature-select-v2 scripts/code. Pass --feature-select-code-dir or set "
        f"FEATURE_SELECT_V2_CODE_DIR.\nSearched:\n{searched}"
    )


def load_feature_select_functions(code_dir: Path):
    sys.path.insert(0, str(code_dir))
    sys.path.insert(0, str(code_dir / "utils"))
    from utils.feature_select import batch_psi, d01_preselect_by_toad

    return batch_psi, d01_preselect_by_toad


def resolve_project_path(project_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def load_feature_map(feature_columns_path: Path) -> dict[str, list[str]]:
    table_features: dict[str, list[str]] = {}
    with feature_columns_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            table = row["full_table_name"]
            feature = row["feature_name"]
            table_features.setdefault(table, []).append(feature)
    return table_features


def table_slug(table_name: str) -> str:
    return table_name.replace(".", "__dot__")


def build_sample_sql(
    table_name: str,
    train_partitions: list[str],
    valid_partitions: list[str],
    *,
    partition_col: str | None,
    sample_where: str,
) -> str:
    blocks = []
    partitions = train_partitions + valid_partitions
    if partition_col and partitions:
        for partition_value in partitions:
            blocks.append(
                f"select * from {table_name} "
                f"where {partition_col} = '{partition_value}' "
                f"and {sample_where}"
            )
    else:
        blocks.append(f"select * from {table_name} where {sample_where}")
    return "\nunion all\n".join(blocks)


def prescreen_dataset_id(table_name: str) -> str:
    return f"feature_prescreen_{table_slug(table_name)}"


def coerce_features(df: pd.DataFrame, feature_list: list[str]) -> list[str]:
    available = [feature for feature in feature_list if feature in df.columns]
    for feature in available:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
        df[feature] = df[feature].replace([np.inf, -np.inf, -999, -998], np.nan)
    return available


def shuffle_features(feature_list: list[str], random_seed: int) -> list[str]:
    features = feature_list.copy()
    rng = np.random.default_rng(random_seed)
    rng.shuffle(features)
    return features


def get_d01_remain(round_select_result: dict) -> list[str]:
    if not round_select_result:
        return []
    max_round = max(round_select_result)
    return [feature for group_result in round_select_result[max_round] for feature in group_result[2]]


def get_d01_drop_counts(round_select_result: dict, total_features: int, remain_features: list[str]) -> dict[str, int]:
    empty_drop = []
    corr_drop = []
    iv_drop = []
    for group_results in round_select_result.values():
        for select_result in group_results:
            empty_drop.extend(select_result[0].get("empty", []))
            corr_drop.extend(select_result[0].get("corr", []))
            iv_drop.extend(select_result[0].get("iv", []))
    return {
        "d01_empty_drop": len(set(empty_drop)),
        "d01_corr_drop": len(set(corr_drop)),
        "d01_iv_drop": len(set(iv_drop)),
        "d01_total_drop": total_features - len(remain_features),
    }


def run_d01(
    df: pd.DataFrame,
    feature_list: list[str],
    target_col: str,
    split_col: str,
    train_value: str,
    d01_thresholds: dict[str, float],
    round_num: int,
    use_native: bool | None,
    d01_preselect_by_toad_func,
) -> tuple[dict, list[str]]:
    dev_df = df[(df[split_col] == train_value) & (df[target_col].isin([0, 1]))].copy()
    dev_df[target_col] = dev_df[target_col].astype(int)
    select_result = d01_preselect_by_toad_func(
        df=dev_df.loc[:, feature_list + [target_col]],
        target_col=target_col,
        feature_list=feature_list,
        preselect_condition={key: value for key, value in d01_thresholds.items() if not key.startswith("__")},
        round_num=round_num,
        use_native=use_native,
        max_round=None,
    )
    return select_result, get_d01_remain(select_result)


def run_d02(
    df: pd.DataFrame,
    remain_features: list[str],
    split_col: str,
    train_value: str,
    valid_value: str,
    psi_threshold: float,
    batch_psi_func,
) -> tuple[dict, dict[str, float], list[str]]:
    """Generate DEV monthly PSI evidence without OOT-based feature deletion.

    The legacy arguments stay in place for public API compatibility.  OOT and
    the vendored PSI callable are intentionally not used.
    """
    del valid_value, batch_psi_func
    if not remain_features:
        return {}, {}, []
    dev_df = df[df[split_col] == train_value].copy()
    if dev_df.empty:
        raise RuntimeError(f"PSI stability prescreen requires {train_value} rows.")
    from risk_model_workbench.feature_selection.risk_review import monthly_psi_evidence, normalize_months, resolve_month_column

    month_column = resolve_month_column(dev_df, {})
    months = (
        normalize_months(dev_df[month_column])
        if month_column
        else pd.Series(pd.NA, index=dev_df.index, dtype="string")
    )
    detail, summary, _, warnings = monthly_psi_evidence(
        dev_df.loc[:, remain_features],
        months,
        remain_features,
        n_bins=int(df.attrs.get("rmw_psi_bins", 10)),
        warning_threshold=float(psi_threshold),
        fail_threshold=float(df.attrs.get("rmw_psi_fail_threshold", psi_threshold)),
        min_base_samples=int(df.attrs.get("rmw_psi_min_base_samples", 500)),
    )
    feature_max_psi = summary.set_index("feature")["monthly_psi_max"].to_dict() if not summary.empty else {}
    psi_result = {
        "mode": "dev_first_natural_month_base",
        "oot_used": False,
        "month_column": month_column,
        "warnings": warnings,
        "detail": detail.to_dict("records"),
        "summary": summary.to_dict("records"),
    }
    return psi_result, feature_max_psi, []


def write_pickle(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "table",
        "input_features",
        "sample_rows",
        "dev_rows",
        "oot_rows",
        "d01_remain",
        "d01_empty_drop",
        "d01_corr_drop",
        "d01_iv_drop",
        "d01_total_drop",
        "d02_psi_drop",
        "final_remain",
        "elapsed_seconds",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Worker function (module-level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------

def process_single_table(
    table_name: str,
    feature_list: list[str],
    table_index: int,
    total_tables: int,
    train_partitions: list[str],
    valid_partitions: list[str],
    partition_col: str | None,
    sample_where: str,
    target_col: str,
    split_col: str,
    train_value: str,
    valid_value: str,
    d01_thresholds: dict[str, float],
    d02_psi_threshold: float,
    round_num: int,
    random_seed: int,
    use_native: bool | None,
    cache_dir: str,
    sql_dir: str,
    project_dir: str,
    dp_data_dir: str,
    dp_metadata_dir: str,
    refresh_dp_cache: bool,
    sql_approved: bool,
    feature_select_code_dir: str,
    run_dir: str | None,
    stage: str = DEFAULT_STAGE,
    external_operation_context: dict[str, str] | None = None,
) -> dict | None:
    """Process one feature table end-to-end: fetch -> quality screen -> PSI screen -> checkpoint.

    Returns the summary dict, or None if table was skipped via checkpoint.
    Runs in a worker process and reads DP samples through the local feather cache.
    """
    cache_path = Path(cache_dir) / f"{table_slug(table_name)}.pkl"
    reporter = ProgressReporter(run_dir, stage) if run_dir else None
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            checkpoint = pickle.load(handle)
        print(f"[SKIP] ({table_index}/{total_tables}) {table_name}")
        if reporter:
            reporter.emit(
                step="table_skipped",
                status="skipped",
                message=f"表 {table_index}/{total_tables}：已有检查点，跳过 {table_name}",
                current=table_index,
                total=total_tables,
                metrics={"table": table_name},
            )
        return checkpoint["summary"]

    print(f"[INFO] ({table_index}/{total_tables}) {table_name}, features={len(feature_list)}", flush=True)
    if reporter:
        reporter.emit(
            step="table_started",
            message=f"表 {table_index}/{total_tables}：开始筛选 {table_name}，候选变量 {len(feature_list)} 个",
            current=table_index - 1,
            total=total_tables,
            metrics={"table": table_name, "features": len(feature_list)},
        )

    # Each worker sets up its own imports and client
    sys.path.insert(0, feature_select_code_dir)
    sys.path.insert(0, str(Path(feature_select_code_dir) / "utils"))
    from utils.feature_select import batch_psi, d01_preselect_by_toad

    start_time = time.time()
    sample_sql = build_sample_sql(
        table_name,
        train_partitions,
        valid_partitions,
        partition_col=partition_col,
        sample_where=sample_where,
    )

    sql_path = Path(sql_dir) / f"{table_slug(table_name)}.sql"
    sql_path.parent.mkdir(parents=True, exist_ok=True)
    sql_path.write_text(sample_sql.strip() + "\n", encoding="utf-8")

    dataset_id = prescreen_dataset_id(table_name)
    description = f"特征初筛样本：{table_name}，DEV 用于质量初筛和月度 PSI 证据；OOT 不参与特征 PSI。"
    sample = load_or_fetch_dp_feather(
        project_dir=Path(project_dir),
        sql=sample_sql,
        dataset_id=dataset_id,
        description=description,
        feather_path=Path(dp_data_dir) / f"{table_slug(table_name)}.feather",
        metadata_path=Path(dp_metadata_dir) / f"{table_slug(table_name)}.json",
        refresh=refresh_dp_cache,
        sql_approved=sql_approved,
        progress=reporter,
        external_operation_context=external_operation_context,
    )

    available_features = coerce_features(sample, feature_list)
    if not available_features:
        raise RuntimeError(f"No feature columns available in table: {table_name}")
    available_features = shuffle_features(available_features, random_seed + table_index)
    sample.attrs["rmw_psi_fail_threshold"] = d01_thresholds.get("__psi_fail", d02_psi_threshold)
    sample.attrs["rmw_psi_bins"] = d01_thresholds.get("__psi_bins", 10)
    sample.attrs["rmw_psi_min_base_samples"] = d01_thresholds.get("__psi_min_base_samples", 500)

    d01_result, d01_remain = run_d01(
        sample,
        available_features,
        target_col=target_col,
        split_col=split_col,
        train_value=train_value,
        d01_thresholds=d01_thresholds,
        round_num=round_num,
        use_native=use_native,
        d01_preselect_by_toad_func=d01_preselect_by_toad,
    )
    if reporter:
        reporter.emit(
            step="quality_screen_done",
            message=f"表 {table_index}/{total_tables}：质量初筛完成，保留 {len(d01_remain)}/{len(available_features)} 个变量",
            current=table_index - 1,
            total=total_tables,
            metrics={"table": table_name, "input_features": len(available_features), "d01_remain": len(d01_remain)},
        )
    psi_result, feature_max_psi, psi_drop_features = run_d02(
        sample,
        d01_remain,
        split_col=split_col,
        train_value=train_value,
        valid_value=valid_value,
        psi_threshold=d02_psi_threshold,
        batch_psi_func=batch_psi,
    )
    final_remain = list(d01_remain)
    if reporter:
        reporter.emit(
            step="psi_screen_done",
            message=f"表 {table_index}/{total_tables}：DEV 月度 PSI 证据生成完成；PSI 不单独删除变量",
            current=table_index - 1,
            total=total_tables,
            metrics={"table": table_name, "d02_psi_drop": len(psi_drop_features), "final_remain": len(final_remain)},
        )

    elapsed = round(time.time() - start_time, 3)
    drop_counts = get_d01_drop_counts(d01_result, len(available_features), d01_remain)
    summary = {
        "table": table_name,
        "input_features": len(available_features),
        "sample_rows": int(len(sample)),
        "dev_rows": int((sample[split_col] == train_value).sum()),
        "oot_rows": int((sample[split_col] == valid_value).sum()),
        "d01_remain": len(d01_remain),
        **drop_counts,
        "d02_psi_drop": len(psi_drop_features),
        "final_remain": len(final_remain),
        "elapsed_seconds": elapsed,
    }

    checkpoint = {
        "summary": summary,
        "d01_result": d01_result,
        "d01_remain_features": d01_remain,
        "d02_psi_result": psi_result,
        "d02_feature_max_psi": feature_max_psi,
        "d02_psi_drop_features": psi_drop_features,
        "final_remain_features": final_remain,
    }
    write_pickle(cache_path, checkpoint)

    print(
        f"[OK] {table_name}: input={len(available_features)}, "
        f"d01_remain={len(d01_remain)}, final={len(final_remain)}, elapsed={elapsed}s",
        flush=True,
    )
    if reporter:
        reporter.emit(
            step="table_done",
            message=(
                f"表 {table_index}/{total_tables}：筛选完成，输入 {len(available_features)} 个，"
                f"质量初筛保留 {len(d01_remain)} 个，最终保留 {len(final_remain)} 个"
            ),
            current=table_index,
            total=total_tables,
            metrics=summary,
        )
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_prescreen(args: PrescreenAction) -> int:
    project_dir = Path(args.project_dir).resolve()
    stage = args.stage or DEFAULT_STAGE
    reporter = ProgressReporter(args.run_dir, stage) if args.run_dir else None
    settings = load_batch_settings(project_dir, args)
    feature_select_code_dir = find_feature_select_code_dir(project_dir, args.feature_select_code_dir)

    feature_columns_path = resolve_project_path(project_dir, settings.feature_columns)
    output_dir = resolve_project_path(project_dir, settings.output_dir)
    cache_dir = output_dir / "cache"
    sql_dir = output_dir / "sql"
    results_dir = output_dir / "results"
    dp_data_dir = resolve_project_path(project_dir, settings.dp_data_dir)
    dp_metadata_dir = resolve_project_path(project_dir, settings.dp_metadata_dir)

    table_features = load_feature_map(feature_columns_path)
    if args.table:
        requested = set(args.table)
        table_features = {table: features for table, features in table_features.items() if table in requested}
    if args.max_tables is not None:
        table_features = dict(list(table_features.items())[: args.max_tables])
    table_items = list(table_features.items())
    if reporter:
        reporter.emit(
            step="load_feature_map",
            message=f"特征初筛准备完成，共 {len(table_items)} 张表",
            current=0,
            total=len(table_items),
            percent=5,
            metrics={"tables": len(table_items)},
        )

    # Force override old results if not a smoke test
    if args.force:
        import shutil
        for d in [cache_dir, sql_dir]:
            if d.exists():
                shutil.rmtree(d)

    run_meta = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "project_dir": str(project_dir),
        "feature_columns": str(feature_columns_path),
        "feature_select_code_dir": str(feature_select_code_dir),
        "thresholds": {
            "quality": {key: value for key, value in settings.d01_thresholds.items() if not key.startswith("__")},
            "psi": settings.d02_psi_threshold,
            "psi_fail": settings.d01_thresholds.get("__psi_fail"),
            "psi_bins": int(settings.d01_thresholds.get("__psi_bins", 10)),
            "psi_min_base_samples": int(settings.d01_thresholds.get("__psi_min_base_samples", 500)),
        },
        "sampling": {
            "partition_col": settings.partition_col,
            "where": settings.sample_where,
            "quality_sample": f"{settings.split_col} = '{settings.train_value}'",
            "stability_compare": "DEV first natural month vs later DEV months",
            "oot_used_for_feature_psi": False,
        },
        "partitions": {
            "train": settings.train_partitions,
            "valid": settings.valid_partitions,
        },
        "target_col": settings.target_col,
        "split_col": settings.split_col,
        "workers": settings.workers,
    }
    write_json(output_dir / "run_config.json", run_meta)

    # Ensure output dirs
    cache_dir.mkdir(parents=True, exist_ok=True)
    sql_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    dp_metadata_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run_sql:
        for idx, (table_name, _feature_list) in enumerate(table_items, start=1):
            sample_sql = build_sample_sql(
                table_name,
                settings.train_partitions,
                settings.valid_partitions,
                partition_col=settings.partition_col,
                sample_where=settings.sample_where,
            )
            sql_path = sql_dir / f"{table_slug(table_name)}.sql"
            sql_path.parent.mkdir(parents=True, exist_ok=True)
            sql_path.write_text(sample_sql.strip() + "\n", encoding="utf-8")
            dataset_id = prescreen_dataset_id(table_name)
            feather_path = dp_data_dir / f"{table_slug(table_name)}.feather"
            metadata_path = dp_metadata_dir / f"{table_slug(table_name)}.json"
            description = f"特征初筛样本：{table_name}，DEV 用于质量初筛和月度 PSI 证据；OOT 不参与特征 PSI。"
            write_dataset_metadata(
                project_dir=project_dir,
                metadata_path=metadata_path,
                feather_path=feather_path,
                dataset_id=dataset_id,
                description=description,
                sql=sample_sql,
                status="sql_review_required" if args.refresh_dp_cache or not feather_path.exists() else "ready",
                note="Review this SQL before running DP fetch. The feather file itself is gitignored.",
            )
            if idx <= 3:
                print_sql_review(
                    dataset_id=dataset_id,
                    description=description,
                    feather_path=feather_path,
                    metadata_path=metadata_path,
                    sql=sample_sql,
                )
            if reporter:
                reporter.emit(
                    step="dry_run_sql",
                    status="waiting_for_approval",
                    message=f"表 {idx}/{len(table_items)}：SQL 已生成，等待审批 {table_name}",
                    current=idx,
                    total=len(table_items),
                    metrics={"table": table_name, "metadata_path": str(metadata_path)},
                )
        print(f"[DRY-RUN] wrote {len(table_items)} SQL files under {sql_dir}")
        print(f"[DRY-RUN] wrote metadata under {dp_metadata_dir}")
        if len(table_items) > 3:
            print("[DRY-RUN] printed first 3 SQLs only; inspect the SQL files for the rest.")
        if reporter:
            reporter.emit(
                step="dry_run_sql_done",
                status="waiting_for_approval",
                message=f"特征初筛 SQL 生成完成，共 {len(table_items)} 张表，等待人工审批后执行",
                current=len(table_items),
                total=len(table_items),
                percent=100,
            )
        return 0

    use_native = True if args.use_native else None
    total_tables = len(table_features)

    print(f"[INFO] Launching {settings.workers} workers for {total_tables} tables, "
          f"train partitions={len(settings.train_partitions)}, valid partitions={len(settings.valid_partitions)}")
    if reporter:
        reporter.emit(
            step="launch_workers",
            message=f"启动 {settings.workers} 个 worker，开始处理 {total_tables} 张表",
            current=0,
            total=total_tables,
            percent=8,
            metrics={"workers": settings.workers, "tables": total_tables},
        )

    summary_rows: list[dict] = []
    final_remain_by_table: dict[str, list[str]] = {}
    from risk_model_workbench.agent.approvals import ApprovalBindingError
    from risk_model_workbench.dp_feather import ExternalOutcomeUnknown, external_operation_context_for_current_attempt

    external_operation_context = external_operation_context_for_current_attempt()
    unknown_external_errors: list[str] = []
    binding_errors: list[str] = []

    with ProcessPoolExecutor(max_workers=settings.workers) as executor:
        futures = {}
        for idx, (table_name, feature_list) in enumerate(table_items, start=1):
            future = executor.submit(
                process_single_table,
                table_name=table_name,
                feature_list=feature_list,
                table_index=idx,
                total_tables=total_tables,
                train_partitions=settings.train_partitions,
                valid_partitions=settings.valid_partitions,
                partition_col=settings.partition_col,
                sample_where=settings.sample_where,
                target_col=settings.target_col,
                split_col=settings.split_col,
                train_value=settings.train_value,
                valid_value=settings.valid_value,
                d01_thresholds=settings.d01_thresholds,
                d02_psi_threshold=settings.d02_psi_threshold,
                round_num=settings.round_num,
                random_seed=settings.random_seed,
                use_native=use_native,
                cache_dir=str(cache_dir),
                sql_dir=str(sql_dir),
                project_dir=str(project_dir),
                dp_data_dir=str(dp_data_dir),
                dp_metadata_dir=str(dp_metadata_dir),
                refresh_dp_cache=args.refresh_dp_cache,
                sql_approved=args.sql_approved,
                feature_select_code_dir=str(feature_select_code_dir),
                run_dir=args.run_dir,
                stage=stage,
                external_operation_context=external_operation_context,
            )
            futures[future] = table_name

        for future in as_completed(futures):
            table_name = futures[future]
            try:
                summary = future.result()
                if summary is not None:
                    summary_rows.append(summary)
                    # Load checkpoint for final_remain
                    cache_path = cache_dir / f"{table_slug(table_name)}.pkl"
                    if cache_path.exists():
                        with cache_path.open("rb") as handle:
                            ck = pickle.load(handle)
                        final_remain_by_table[table_name] = ck["final_remain_features"]
                    if reporter:
                        reporter.emit(
                            step="aggregate_progress",
                            message=f"整体进度：已完成 {len(summary_rows)}/{total_tables} 张表",
                            current=len(summary_rows),
                            total=total_tables,
                            metrics={"completed_tables": len(summary_rows), "total_tables": total_tables},
                        )
            except Exception as exc:
                print(f"[ERROR] {table_name}: {exc}", file=sys.stderr, flush=True)
                if isinstance(exc, ExternalOutcomeUnknown) or "external operation outcome is unknown" in str(exc).lower():
                    unknown_external_errors.append(f"{table_name}: {exc}")
                elif external_operation_context is not None and args.sql_approved:
                    binding_errors.append(f"{table_name}: {exc}")
                if reporter:
                    reporter.emit(
                        step="table_failed",
                        status="failed",
                        message=f"表处理失败：{table_name}：{exc}",
                        current=len(summary_rows),
                        total=total_tables,
                        metrics={"table": table_name, "completed_tables": len(summary_rows)},
                        level="error",
                    )

    if unknown_external_errors:
        raise ExternalOutcomeUnknown("; ".join(unknown_external_errors))
    if binding_errors:
        raise ApprovalBindingError("; ".join(binding_errors))

    table_order = {name: idx for idx, (name, _) in enumerate(table_items)}
    summary_rows.sort(key=lambda r: table_order.get(r["table"], 999))

    run_summary = {
        "tables": len(summary_rows),
        "input_features": sum(int(row["input_features"]) for row in summary_rows),
        "quality_remain": sum(int(row["d01_remain"]) for row in summary_rows),
        "psi_drop": sum(int(row["d02_psi_drop"]) for row in summary_rows),
        "final_remain": sum(int(row["final_remain"]) for row in summary_rows),
    }
    legacy_run_summary = {
        **run_summary,
        "d01_remain": run_summary["quality_remain"],
        "d02_psi_drop": run_summary["psi_drop"],
    }
    write_summary_csv(results_dir / "prescreen_table_summary.csv", summary_rows)
    write_summary_csv(results_dir / "d01_d02_table_summary.csv", summary_rows)
    write_json(results_dir / "prescreen_final_remain_features.json", final_remain_by_table)
    write_json(results_dir / "d01_d02_final_remain_features.json", final_remain_by_table)
    write_json(results_dir / "prescreen_run_summary.json", run_summary)
    write_json(results_dir / "d01_d02_run_summary.json", legacy_run_summary)
    print(f"[DONE] output: {output_dir}")
    if reporter:
        reporter.emit(
            step="stage_outputs",
            status="done",
            message=f"特征初筛完成：成功 {len(summary_rows)}/{total_tables} 张表，最终保留 {sum(int(row['final_remain']) for row in summary_rows)} 个变量",
            current=len(summary_rows),
            total=total_tables,
            percent=100,
            metrics={
                "tables": len(summary_rows),
                "input_features": sum(int(row["input_features"]) for row in summary_rows),
                "final_remain": sum(int(row["final_remain"]) for row in summary_rows),
            },
        )
    return 0


def run_prescreen_service(
    *,
    project_dir: str | Path,
    config: str = "configs/feature_select.yaml",
    feature_select_code_dir: str | None = None,
    feature_columns: str | None = None,
    output_dir: str | None = None,
    tables: list[str] | tuple[str, ...] = (),
    max_tables: int | None = None,
    round_num: int | None = None,
    random_seed: int | None = None,
    force: bool = False,
    use_native: bool = False,
    dev_partition_ds: str | None = None,
    oot_partition_ds: str | None = None,
    partition_col: str | None = None,
    sample_where: str | None = None,
    workers: int | None = None,
    dry_run_sql: bool = False,
    refresh_dp_cache: bool = False,
    sql_approved: bool = False,
    run_dir: str | Path | None = None,
    stage: str = DEFAULT_STAGE,
) -> int:
    """Typed prescreen service; the CLI parser is only an adapter."""
    return _run_prescreen(
        PrescreenAction(
            project_dir=project_dir,
            config=config,
            feature_select_code_dir=feature_select_code_dir,
            feature_columns=feature_columns,
            output_dir=output_dir,
            table=tuple(tables),
            max_tables=max_tables,
            round_num=round_num,
            random_seed=random_seed,
            force=force,
            use_native=use_native,
            dev_partition_ds=dev_partition_ds,
            oot_partition_ds=oot_partition_ds,
            partition_col=partition_col,
            sample_where=sample_where,
            workers=workers,
            dry_run_sql=dry_run_sql,
            refresh_dp_cache=refresh_dp_cache,
            sql_approved=sql_approved,
            run_dir=run_dir,
            stage=stage,
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_prescreen_service(
        project_dir=args.project_dir,
        config=args.config,
        feature_select_code_dir=args.feature_select_code_dir,
        feature_columns=args.feature_columns,
        output_dir=args.output_dir,
        tables=args.table or [],
        max_tables=args.max_tables,
        round_num=args.round_num,
        random_seed=args.random_seed,
        force=args.force,
        use_native=args.use_native,
        dev_partition_ds=args.dev_partition_ds,
        oot_partition_ds=args.oot_partition_ds,
        partition_col=args.partition_col,
        sample_where=args.sample_where,
        workers=args.workers,
        dry_run_sql=args.dry_run_sql,
        refresh_dp_cache=args.refresh_dp_cache,
        sql_approved=args.sql_approved,
        run_dir=args.run_dir,
        stage=args.stage,
    )


if __name__ == "__main__":
    raise SystemExit(main())
