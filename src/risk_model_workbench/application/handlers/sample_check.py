"""Sample-check application handler shared by CLI and Agent Runtime."""

from __future__ import annotations

import json
from pathlib import Path

from risk_model_workbench.application.context import VersionContext
from risk_model_workbench.config import load_yaml
from risk_model_workbench.harness.invocation import ActionInvocation
from risk_model_workbench.harness.runtime import (
    ActionResult,
    classify_exception,
    detached_action_attempt,
    register_action_artifact,
    stage_action_done,
    stage_action_failed,
    stage_action_started,
)
from risk_model_workbench.paths import project_config_path
from risk_model_workbench.state import append_decision, load_run_state


def _last_result(context: VersionContext) -> ActionResult:
    payload = dict(load_run_state(context.workspace)["stages"]["sample_check"]["last_result"])
    allowed = set(ActionResult.__dataclass_fields__)
    return ActionResult(**{key: value for key, value in payload.items() if key in allowed})


def _config(context: VersionContext) -> dict:
    runtime = context.runtime_config_dir / "project.yml"
    if runtime.exists():
        return load_yaml(runtime)
    path = project_config_path(context.project_dir)
    return load_yaml(path) if path.exists() else {}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def run_sample_check(
    invocation: ActionInvocation, context: VersionContext, attempt_id: str
) -> ActionResult:
    """Profile local input data or emit an explicit scaffold contract."""
    del invocation, attempt_id
    with detached_action_attempt():
        stage_action_started(context.workspace, "sample_check")
        config = _config(context)
        data_cfg = config.get("data") or {}
        raw_path = Path(str(data_cfg.get("raw_path") or "data/raw/sample.feather"))
        if not raw_path.is_absolute():
            raw_path = context.project_dir / raw_path
        try:
            if raw_path.exists():
                _profile_local(context, config, data_cfg, raw_path)
            else:
                _write_scaffold(context, config, data_cfg)
        except Exception as exc:
            stage_action_failed(
                context.workspace, "sample_check", str(exc), failure_code=classify_exception(exc)
            )
    return _last_result(context)


def _profile_local(context: VersionContext, config: dict, data_cfg: dict, raw_path: Path) -> None:
    import pandas as pd

    if raw_path.suffix == ".csv":
        df = pd.read_csv(raw_path)
    elif raw_path.suffix == ".parquet":
        df = pd.read_parquet(raw_path)
    else:
        df = pd.read_feather(raw_path)
    target_col = data_cfg.get("target_column")
    split_col = data_cfg.get("split_column") or (config.get("split") or {}).get("source_column")
    time_col = data_cfg.get("time_column")
    id_columns = [col for col in data_cfg.get("id_columns", []) if col in df.columns]
    summary = {
        "status": "done",
        "reason": "",
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "project": config.get("project", {}),
        "target_column": target_col,
        "target_column_present": bool(target_col in df.columns),
        "id_columns": data_cfg.get("id_columns", []),
        "id_columns_present": id_columns,
        "duplicate_key_rows": int(df.duplicated(subset=id_columns).sum()) if id_columns else None,
        "split_column": split_col,
        "split_column_present": bool(split_col in df.columns),
    }
    output = context.workspace / "sample_check"
    _write_json(output / "sample_summary.json", summary)
    if target_col in df.columns:
        df[target_col].value_counts(dropna=False).rename_axis("label").reset_index(name="count").to_csv(
            output / "label_distribution.csv", index=False, encoding="utf-8-sig"
        )
    if split_col in df.columns:
        rows = df.groupby(split_col, dropna=False).size().reset_index(name="count")
        if target_col in df.columns:
            rates = df.assign(_target_numeric=pd.to_numeric(df[target_col], errors="coerce")).groupby(
                split_col, dropna=False
            )["_target_numeric"].mean().reset_index(name="target_rate")
            rows = rows.merge(rates, on=split_col, how="left")
        rows.to_csv(output / "sample_split_summary.csv", index=False, encoding="utf-8-sig")
    if time_col in df.columns and target_col in df.columns:
        month = pd.to_datetime(df[time_col], errors="coerce").dt.to_period("M").astype(str)
        monthly = (
            df.assign(_month=month)
            .groupby("_month", dropna=False)
            .agg(
                samples=(target_col, "count"),
                positive=(target_col, "sum"),
                target_rate=(target_col, "mean"),
            )
            .reset_index()
        )
        monthly.to_csv(
            output / "monthly_label_distribution.csv", index=False, encoding="utf-8-sig"
        )
    segment_cols = [
        col
        for col in [
            "blue_customer_flag",
            "zc_level",
            "channel",
            "channel_id",
            "account_status",
            "acct_status",
            "roll_rate_status",
            "credit_product",
            "credit_product_code",
            "product_code",
            *data_cfg.get("segment_columns", []),
        ]
        if col in df.columns
    ]
    if segment_cols:
        segment_rows = []
        for column in dict.fromkeys(segment_cols):
            for value, count in df[column].value_counts(dropna=False).items():
                segment_rows.append(
                    {
                        "segment_column": column,
                        "segment_value": str(value),
                        "count": int(count),
                        "ratio": float(count / len(df)) if len(df) else 0,
                    }
                )
        pd.DataFrame(segment_rows).to_csv(
            output / "segment_distribution.csv", index=False, encoding="utf-8-sig"
        )
    (output / "sample_check_report.md").write_text("# Sample Check\n\nstatus: done\n", encoding="utf-8")
    for artifact in sorted(output.iterdir()):
        register_action_artifact(context.workspace, "sample_check", artifact)
    append_decision(
        context.workspace,
        stage="sample_check",
        decision="done",
        reason="Sample profiling completed from local data",
    )
    stage_action_done(context.workspace, "sample_check")


def _write_scaffold(context: VersionContext, config: dict, data_cfg: dict) -> None:
    reason = "local data not available"
    output = context.workspace / "sample_check"
    _write_json(
        output / "sample_summary.json",
        {
            "status": "scaffold",
            "reason": reason,
            "project": config.get("project", {}),
            "target_column": data_cfg.get("target_column"),
            "id_columns": data_cfg.get("id_columns", []),
            "split_column": data_cfg.get("split_column") or (config.get("split") or {}).get("source_column"),
            "expected_outputs": [
                "positive_rate_overall.csv",
                "positive_rate_by_split.csv",
                "positive_rate_by_month.csv",
                "positive_rate_by_segment.csv",
            ],
        },
    )
    (output / "sample_check_report.md").write_text(
        "# Sample Check\n\nstatus: scaffold\n\nreason: local data not available\n", encoding="utf-8"
    )
    register_action_artifact(context.workspace, "sample_check", output / "sample_summary.json")
    register_action_artifact(context.workspace, "sample_check", output / "sample_check_report.md")
    append_decision(context.workspace, stage="sample_check", decision="scaffold", reason=reason)
    stage_action_done(context.workspace, "sample_check", scaffold=True, message=reason)


__all__ = ["run_sample_check"]
