"""Command-line interface for the local business modeling workbench."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from risk_model_workbench.cli_meta import add_metadata_parsers
from risk_model_workbench.agent.advisor import (
    accept_advisor_response,
    advisor_request_is_answered,
    list_advisor_requests,
    load_advisor_request,
)
from risk_model_workbench.agent.approvals import approve_request, load_approvals, reject_request
from risk_model_workbench.agent.executor import run_agent
from risk_model_workbench.agent.plan import (
    agent_tool_schema,
    bind_agent_plan,
    load_agent_plan,
    rebind_agent_plan,
    save_agent_plan,
)
from risk_model_workbench.agent.state import init_agent_state, load_agent_state, save_agent_state
from risk_model_workbench.agent.trace import append_trace, load_recent_trace
from risk_model_workbench.config import load_yaml
from risk_model_workbench.feature_screening import write_feature_screening_summary
from risk_model_workbench.harness.errors import SQL_APPROVAL_REQUIRED
from risk_model_workbench.harness.runtime import (
    classify_exception,
    register_action_artifact as register_artifact,
    run_with_retry,
    stage_action_done,
    stage_action_failed,
    stage_action_started,
)
from risk_model_workbench.manifest import make_run_id
from risk_model_workbench.paths import REPO_ROOT, project_config_path, resolve_project_path, workflow_path
from risk_model_workbench.planning import create_execution_plan, save_execution_plan
from risk_model_workbench.progress import (
    ProgressReporter,
    format_progress_report,
    load_progress_events,
    load_progress_summary,
)
from risk_model_workbench.project import create_project
from risk_model_workbench.project_state import (
    append_lesson,
    audit_run,
    format_run_audit,
    format_project_summary,
    summarize_project,
    update_project_state,
    write_handoff,
    write_project_state_from_summary,
    write_retrospective,
)
from risk_model_workbench.request import parse_model_request, validate_model_request
from risk_model_workbench.request.materialize import RUNTIME_CONFIG_DIR, materialize_request_runtime_configs
from risk_model_workbench.request.training import merge_training_config, project_training_defaults
from risk_model_workbench.rules import format_rules, load_workbench_rules, promote_lesson_to_rule
from risk_model_workbench.registry import load_artifact_manifest
from risk_model_workbench.state import (
    append_decision,
    create_run_state,
    create_version_state,
    load_run_state,
    mark_stage_done,
    run_dir,
    save_run_state,
    save_version_state,
    version_dir,
)
from risk_model_workbench.versioning import (
    list_versions,
    load_version_index,
    migrate_run_to_version,
    resolve_workspace_dir,
    save_version_index,
    standard_run_dirs,
    suggest_version_id,
    upsert_version_index,
    validate_version_id,
)
from risk_model_workbench.workflow_contracts import validate_workflow_definition
from risk_model_workbench.wide_sql import generate_wide_sql


FEATURE_PRESCREEN_STAGE = "feature_prescreen"
LEGACY_FEATURE_PRESCREEN_STAGE = "d01_d02_screening"
DEFAULT_PRESCREEN_REMAIN_FEATURES = "runs/feature_prescreen/results/prescreen_final_remain_features.json"
LEGACY_PRESCREEN_REMAIN_FEATURES = "runs/d01_d02_batch_select/results/d01_d02_final_remain_features.json"
DEFAULT_WIDE_SQL_OUTPUT = "queries/06_build_prescreen_wide_table.sql"
DEFAULT_WIDE_FEATURE_MAP_OUTPUT = "runs/feature_prescreen/results/prescreen_wide_feature_map.csv"
DEFAULT_WIDE_SUMMARY_OUTPUT = "runs/feature_prescreen/results/prescreen_wide_sql_summary.json"


def _run_path(args: argparse.Namespace) -> Path:
    version_id = getattr(args, "version_id", None)
    run_id = getattr(args, "run_id", None)
    if version_id and not run_id:
        setattr(args, "run_id", version_id)
    return resolve_workspace_dir(resolve_project_path(args.project), version_id=version_id, run_id=run_id)


def _workspace_arg(args: argparse.Namespace) -> str:
    value = getattr(args, "version_id", None) or getattr(args, "run_id", None)
    if not value:
        raise ValueError("either --version-id or --run-id is required")
    return str(value)


def _feature_prescreen_stage(run_path: Path) -> str:
    """Prefer the generic stage name while allowing old run_state.yml files."""
    try:
        stages = load_run_state(run_path).get("stages") or {}
    except FileNotFoundError:
        return FEATURE_PRESCREEN_STAGE
    if FEATURE_PRESCREEN_STAGE in stages or LEGACY_FEATURE_PRESCREEN_STAGE not in stages:
        return FEATURE_PRESCREEN_STAGE
    return LEGACY_FEATURE_PRESCREEN_STAGE


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    return path


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _count_text_lines(path: Path) -> int | None:
    try:
        if not path.is_file():
            return None
        return len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])
    except (OSError, UnicodeDecodeError):
        return None


def _build_train_plan(
    *,
    workspace: Path,
    experiment: str,
    algorithm: str,
    config_path: Path,
    train_config: dict[str, Any],
    effective_config: dict[str, Any],
    input_feather: Path,
    feature_list: Path,
    score_output: Path,
    input_snapshot_dir: Path,
    skip_split_check: bool,
) -> dict[str, Any]:
    training = effective_config.get("training") if isinstance(effective_config.get("training"), dict) else {}
    tuning = training.get("tuning") if isinstance(training.get("tuning"), dict) else {}
    runtime_experiment = effective_config.get("runtime_experiment") if isinstance(effective_config.get("runtime_experiment"), dict) else {}
    input_ready = input_feather.is_file()
    feature_ready = feature_list.is_file()
    config_ready = bool(train_config)
    mode = str(training.get("mode") or tuning.get("mode") or "single_train")
    return {
        "status": "ready" if input_ready and feature_ready and config_ready else "not_ready",
        "workspace": str(workspace),
        "experiment": experiment,
        "algorithm": algorithm,
        "training_mode": mode,
        "config": {"path": str(config_path), "exists": config_path.exists(), "keys": sorted(train_config.keys())},
        "input_feather": {"path": str(input_feather), "exists": input_feather.exists(), "is_file": input_ready},
        "feature_list": {
            "path": str(feature_list),
            "exists": feature_list.exists(),
            "is_file": feature_ready,
            "feature_count": _count_text_lines(feature_list),
        },
        "outputs": {
            "score_output": str(score_output),
            "input_snapshot_dir": str(input_snapshot_dir),
            "modeling_dir": str(workspace / "modeling" / experiment),
        },
        "splits": {
            "train_values": training.get("train_values", ["DEV"]),
            "valid_values": training.get("valid_values", ["DEV-OOS"]),
            "oos_values": training.get("oos_values", ["DEV-OOS", "OOT-OOS"]),
            "skip_split_check": skip_split_check,
        },
        "tuning": {
            "enabled": mode in {"llm_guided_tune", "llm-guided-tune", "llm_guided"},
            "max_rounds": tuning.get("max_rounds"),
            "candidates_per_round": tuning.get("candidates_per_round"),
            "max_trials": tuning.get("max_trials"),
        },
        "runtime_experiment": runtime_experiment,
        "will_train": input_ready and feature_ready and config_ready,
    }


def _train_plan_markdown(plan: dict[str, Any]) -> str:
    rows = [
        ("Status", plan.get("status")),
        ("Experiment", plan.get("experiment")),
        ("Algorithm", plan.get("algorithm")),
        ("Training mode", plan.get("training_mode")),
        ("Config", (plan.get("config") or {}).get("path")),
        ("Input feather", (plan.get("input_feather") or {}).get("path")),
        ("Input ready", (plan.get("input_feather") or {}).get("is_file")),
        ("Feature list", (plan.get("feature_list") or {}).get("path")),
        ("Feature count", (plan.get("feature_list") or {}).get("feature_count")),
        ("Will train", plan.get("will_train")),
    ]
    lines = [
        "# Training Plan",
        "",
        "| Item | Value |",
        "| --- | --- |",
    ]
    lines.extend(f"| {key} | {_format_md_value(value)} |" for key, value in rows)
    splits = plan.get("splits") or {}
    lines.extend(
        [
            "",
            "## Split Values",
            "",
            f"- train_values: `{_format_md_value(splits.get('train_values'))}`",
            f"- valid_values: `{_format_md_value(splits.get('valid_values'))}`",
            f"- oos_values: `{_format_md_value(splits.get('oos_values'))}`",
            f"- skip_split_check: `{_format_md_value(splits.get('skip_split_check'))}`",
            "",
        ]
    )
    return "\n".join(lines)


def _write_train_plan(output_dir: Path, plan: dict[str, Any]) -> tuple[Path, Path]:
    json_path = _write_json(output_dir / "train_plan.json", plan)
    md_path = output_dir / "train_plan.md"
    md_path.write_text(_train_plan_markdown(plan), encoding="utf-8")
    return json_path, md_path


def _write_training_status(
    output_dir: Path,
    *,
    status: str,
    experiment: str,
    algorithm: str,
    message: str,
    percent: float | None = None,
    failure_code: str = "",
    error_type: str = "",
    error_message: str = "",
    metrics: dict[str, Any] | None = None,
    completed: bool = False,
) -> dict[str, Any]:
    path = output_dir / "training_status.json"
    previous = _read_json(path)
    now = _now_iso()
    payload = {
        "status": status,
        "experiment": experiment,
        "algorithm": algorithm,
        "pid": os.getpid(),
        "started_at": previous.get("started_at") or now,
        "heartbeat_at": now,
        "completed_at": now if completed else previous.get("completed_at", ""),
        "percent": percent,
        "message": message,
        "failure_code": failure_code,
        "error_type": error_type,
        "error_message": error_message,
        "metrics": metrics or {},
    }
    _write_json(path, payload)
    return payload


class _TrainingStatusProgressReporter:
    """Mirror training progress events into modeling/<experiment>/training_status.json."""

    def __init__(self, reporter: ProgressReporter, output_dir: Path, *, experiment: str, algorithm: str) -> None:
        self._reporter = reporter
        self._output_dir = output_dir
        self._experiment = experiment
        self._algorithm = algorithm

    def emit(self, **kwargs: Any) -> dict[str, Any]:
        status = str(kwargs.get("status") or "running")
        try:
            _write_training_status(
                self._output_dir,
                status=status,
                experiment=self._experiment,
                algorithm=self._algorithm,
                message=str(kwargs.get("message") or ""),
                percent=kwargs.get("percent"),
                metrics=kwargs.get("metrics") if isinstance(kwargs.get("metrics"), dict) else {},
                completed=status in {"done", "failed", "scaffold"},
            )
        except Exception:
            pass
        return self._reporter.emit(**kwargs)


def _write_training_summary(
    output_dir: Path,
    *,
    status_payload: dict[str, Any],
    plan: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> Path:
    metrics = metrics or status_payload.get("metrics") or {}
    run_config = _read_json(output_dir / "run_config.json")
    params = run_config.get("params") if isinstance(run_config.get("params"), dict) else {}
    lines = [
        "# Training Summary",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Status | {_format_md_value(status_payload.get('status'))} |",
        f"| Experiment | {_format_md_value(status_payload.get('experiment') or plan.get('experiment'))} |",
        f"| Algorithm | {_format_md_value(status_payload.get('algorithm') or plan.get('algorithm'))} |",
        f"| Training mode | {_format_md_value(run_config.get('training_mode') or plan.get('training_mode'))} |",
        f"| Message | {_format_md_value(status_payload.get('message'))} |",
        f"| Input feather | {_format_md_value((plan.get('input_feather') or {}).get('path'))} |",
        f"| Feature list | {_format_md_value((plan.get('feature_list') or {}).get('path'))} |",
        f"| Candidate features | {_format_md_value(run_config.get('candidate_feature_count') or (plan.get('feature_list') or {}).get('feature_count'))} |",
        f"| Actual features | {_format_md_value(run_config.get('actual_feature_count'))} |",
        f"| Train values | {_format_md_value((plan.get('splits') or {}).get('train_values'))} |",
        f"| Valid values | {_format_md_value((plan.get('splits') or {}).get('valid_values'))} |",
        f"| Train AUC | {_format_md_value(metrics.get('train_auc'))} |",
        f"| Valid AUC | {_format_md_value(metrics.get('valid_auc'))} |",
        f"| Train KS | {_format_md_value(metrics.get('train_ks'))} |",
        f"| Valid KS | {_format_md_value(metrics.get('valid_ks'))} |",
        f"| AUC gap | {_format_md_value(metrics.get('auc_gap'))} |",
    ]
    if status_payload.get("error_message"):
        lines.extend(
            [
                "",
                "## Failure",
                "",
                f"- failure_code: `{_format_md_value(status_payload.get('failure_code'))}`",
                f"- error_type: `{_format_md_value(status_payload.get('error_type'))}`",
                f"- error_message: `{_format_md_value(status_payload.get('error_message'))}`",
            ]
        )
    if params:
        lines.extend(["", "## Parameters", "", "| Parameter | Value |", "| --- | --- |"])
        for key in sorted(params):
            lines.append(f"| {key} | {_format_md_value(params[key])} |")
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            f"- train_metrics: `{output_dir / 'train_metrics.json'}`",
            f"- training_status: `{output_dir / 'training_status.json'}`",
            f"- feature_importance: `{output_dir / 'feature_importance.csv'}`",
            "",
        ]
    )
    path = output_dir / "training_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _format_md_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value).replace("|", "\\|")


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _copy_if_exists(source: Path, target: Path) -> Path | None:
    if not source.exists():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == target.resolve():
        return target
    shutil.copy2(source, target)
    return target


def _copy_and_register_artifact(
    run_path: Path,
    action_id: str,
    source: Path,
    target_relative: str | Path,
    *,
    description: str = "",
) -> Path | None:
    target_relative = Path(target_relative)
    copied = _copy_if_exists(source, run_path / target_relative)
    if copied is not None:
        register_artifact(run_path, action_id, str(target_relative), description=description)
    return copied


def _query_artifact_relative(project_dir: Path, sql_path: Path) -> Path:
    try:
        relative = sql_path.resolve().relative_to(project_dir.resolve())
    except ValueError:
        return Path("queries") / sql_path.name
    if relative.parts and relative.parts[0] == "queries":
        return relative
    return Path("queries") / sql_path.name


def _resolve_refine_config_path(project_dir: Path, config: str | None) -> Path:
    path = Path(config or "configs/refine_features.yaml")
    return path if path.is_absolute() else project_dir / path


def _feature_refine_output_dir(project_dir: Path, config: str | None) -> Path:
    cfg = load_yaml(_resolve_refine_config_path(project_dir, config))["feature_refine"]
    output_dir = Path(cfg["output_dir"])
    return output_dir if output_dir.is_absolute() else project_dir / output_dir


def _feature_refine_output_dir_for_run(project_dir: Path, run_path: Path, config: str | None) -> Path:
    cfg_path = Path(config) if config else _runtime_config_path(run_path, project_dir, "refine_features")
    cfg_path = cfg_path if cfg_path.is_absolute() else project_dir / cfg_path
    cfg = load_yaml(cfg_path)["feature_refine"]
    output_dir = Path(cfg["output_dir"])
    return output_dir if output_dir.is_absolute() else project_dir / output_dir


def _feature_prescreen_output_dir(project_dir: Path, config: str | None) -> Path:
    candidates = (
        [Path(config)]
        if config
        else [project_dir / "configs" / "feature_select.yaml", project_dir / "configs" / "feature_select.yml"]
    )
    cfg_path = None
    for path in candidates:
        candidate = path if path.is_absolute() else project_dir / path
        if candidate.exists():
            cfg_path = candidate
            break
    cfg = load_yaml(cfg_path) if cfg_path else {}
    feature_cfg = cfg.get("feature_select", {})
    prescreen_cfg = feature_cfg.get("prescreen", {}) or feature_cfg.get("d01_d02", {}) or {}
    output_dir = Path(prescreen_cfg.get("output_dir", "runs/feature_prescreen"))
    return output_dir if output_dir.is_absolute() else project_dir / output_dir


def _register_feature_prescreen_artifacts(run_path: Path, stage: str, project_dir: Path, config: str | None) -> None:
    results_dir = _feature_prescreen_output_dir(project_dir, config) / "results"
    for name in [
        "prescreen_run_summary.json",
        "prescreen_final_remain_features.json",
        "prescreen_table_summary.csv",
    ]:
        _copy_and_register_artifact(
            run_path,
            stage,
            results_dir / name,
            Path("feature_selection") / name,
            description="Feature prescreen tracked evidence",
        )


def _register_feature_metadata_artifacts(run_path: Path, project_dir: Path) -> None:
    """Copy project-level feature metadata exports into the run and register them.

    The feature_metadata stage contract requires ``feature_metadata/*``; without
    this the stage finishes ``done`` but registers nothing, leaving the audit
    verdict ``incomplete``.
    """
    src_dir = project_dir / "data" / "profile" / "feature_metadata"
    for name in (
        "feature_tables_meta.json",
        "feature_table_summary.csv",
        "feature_columns.csv",
    ):
        _copy_and_register_artifact(
            run_path,
            "feature_metadata",
            src_dir / name,
            Path("feature_metadata") / name,
            description="Feature metadata export",
        )


def _register_woe_artifacts(path: Path, stage: str, artifact_dir: Path) -> None:
    if not artifact_dir.exists():
        return
    for artifact in sorted([*artifact_dir.glob("woe_top*_summary.csv"), *artifact_dir.glob("images/*.png")]):
        register_artifact(path, stage, artifact)


def _copy_woe_artifacts(source_dir: Path, target_dir: Path) -> None:
    if not source_dir.exists():
        return
    for source in sorted([*source_dir.glob("woe_top*_summary.csv"), *source_dir.glob("images/*.png")]):
        relative = source.relative_to(source_dir)
        target = target_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _load_project_config(project_dir: Path) -> dict[str, Any]:
    return load_yaml(project_config_path(project_dir))


def _runtime_config_dir(run_path: Path) -> Path:
    return run_path / RUNTIME_CONFIG_DIR


def _runtime_config_path(run_path: Path, project_dir: Path, name: str) -> Path:
    candidates: list[Path] = []
    raw = Path(name)
    if raw.suffix:
        candidates.extend([_runtime_config_dir(run_path) / raw.name, raw if raw.is_absolute() else project_dir / raw])
    else:
        candidates.extend([
            _runtime_config_dir(run_path) / f"{name}.yaml",
            _runtime_config_dir(run_path) / f"{name}.yml",
            project_dir / "configs" / f"{name}.yaml",
            project_dir / "configs" / f"{name}.yml",
        ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_runtime_project_config(project_dir: Path, run_path: Path) -> dict[str, Any]:
    runtime_project = _runtime_config_dir(run_path) / "project.yml"
    return load_yaml(runtime_project) if runtime_project.exists() else _load_project_config(project_dir)


def _load_runtime_config(project_dir: Path, run_path: Path, name: str) -> dict[str, Any]:
    path = _runtime_config_path(run_path, project_dir, name)
    return load_yaml(path) if path.exists() else {}


def _runtime_is_local_feather(run_path: Path | None, project_dir: Path) -> bool:
    """True when the request-materialized runtime config declares local_feather mode.

    Local-feather mode means the wide table pre-exists as the sample ``.feather``
    file and there is no remote DP pull; ``feature_prescreen`` and ``build_wide_sql``
    are by-design evidence-only in that mode.
    """
    if run_path is None:
        return False
    feature_cfg = _load_runtime_config(project_dir, run_path, "feature_select").get("feature_select", {})
    runtime_request = feature_cfg.get("runtime_request") or {}
    return runtime_request.get("data_source_mode") == "local_feather"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [value]


def _first_value(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _resolve_project_relative(project_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def _register_if_exists(run_path: Path, stage: str, relative_path: str | Path, *, description: str = "") -> None:
    if (run_path / relative_path).exists():
        register_artifact(run_path, stage, str(relative_path), description=description)


def _feature_columns_from_csv(path: Path) -> list[str]:
    if not path.exists():
        return []
    import csv

    columns: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = row.get("output_feature") or row.get("feature_name") or row.get("feature")
            if value:
                columns.append(str(value))
    return list(dict.fromkeys(columns))


def _refine_feature_columns(project_dir: Path, cfg: dict[str, Any]) -> list[str]:
    refine_cfg = cfg.get("feature_refine", cfg)
    input_cfg = refine_cfg.get("input", {}) or {}
    feature_map = input_cfg.get("feature_map")
    if feature_map:
        return _feature_columns_from_csv(_resolve_project_relative(project_dir, feature_map))
    local_path = _first_value(
        input_cfg.get("local_feather_path"),
        input_cfg.get("raw_path"),
        input_cfg.get("feather_path"),
        (refine_cfg.get("runtime_request") or {}).get("sample_location")
        if (refine_cfg.get("runtime_request") or {}).get("data_source_mode") == "local_feather"
        else None,
    )
    if not local_path:
        return []
    try:
        import pandas as pd

        frame = pd.read_feather(_resolve_project_relative(project_dir, local_path))
    except Exception:
        return []
    required = set(_as_list(input_cfg.get("id_columns")))
    required.update(_as_list(input_cfg.get("base_columns")))
    required.update(str(column) for column in [input_cfg.get("label_column"), input_cfg.get("split_column")] if column)
    return [str(column) for column in frame.columns if str(column) not in required]


def _write_feature_intake_evidence(
    *,
    run_path: Path,
    project_dir: Path,
    stage: str,
    stage_config: dict[str, Any],
    feature_columns: list[str],
    required_columns: list[str],
    source_table: str | None = None,
    local_feather_path: str | Path | None = None,
    total_rows: int | None = None,
    random_columns: list[str] | None = None,
) -> None:
    from risk_model_workbench.data.local_feather_profile import profile_local_feather, write_local_feather_profile
    from risk_model_workbench.data.pull_engine import select_data_pull_engine, write_execution_environment
    from risk_model_workbench.data.table_profile import build_static_table_profile, write_table_profile
    from risk_model_workbench.feature_selection.intake_plan import build_feature_batch_plan, build_sampling_plan, persist_intake_plan
    from risk_model_workbench.resource_planning import build_resource_plan_payload, default_peak_multiplier_for_stage, probe_memory

    try:
        runtime_project = _load_runtime_project_config(project_dir, run_path)
    except FileNotFoundError:
        runtime_project = {}
    request_cfg = runtime_project.get("request", {}) or {}
    data_cfg = runtime_project.get("data", {}) or {}
    data_source_mode = str(request_cfg.get("data_source_mode") or ("local_feather" if local_feather_path else "remote_table"))
    resolved_local_path = _resolve_project_relative(project_dir, local_feather_path) if local_feather_path else None
    resolved_source_table = source_table or data_cfg.get("source_table")

    contract = {
        "data_source_mode": data_source_mode,
        "source_table": resolved_source_table if data_source_mode != "local_feather" else None,
        "local_feather_path": str(resolved_local_path) if resolved_local_path else None,
        "stage": stage,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "note": (
            "local_feather uses an existing local file and is independent from remote DP pull"
            if data_source_mode == "local_feather"
            else "remote_table keeps DP pull/profiling separate from local feather mode"
        ),
    }
    _write_json(run_path / "feature_selection" / "data_source_contract.json", contract)

    selection = select_data_pull_engine(data_source_mode=data_source_mode)
    write_execution_environment(run_path, selection)

    profile_rows = int(total_rows or 0)
    local_file_size = None
    if data_source_mode == "local_feather" and resolved_local_path is not None:
        profile = profile_local_feather(
            resolved_local_path,
            required_columns=required_columns,
            split_column=data_cfg.get("split_column") or data_cfg.get("source_column") or runtime_project.get("split", {}).get("source_column"),
            target_column=data_cfg.get("target_column"),
            feature_exclude_columns=required_columns,
            feature_columns=feature_columns or None,
        )
        profile_rows = int(profile["row_count"])
        local_file_size = int(profile["size_bytes"])
        write_local_feather_profile(profile, run_path / "feature_selection" / "profiles" / "local_feather_profile.json")
        if not feature_columns:
            feature_columns = list(profile.get("candidate_feature_columns", []))
    else:
        profile = build_static_table_profile(
            str(resolved_source_table or ""),
            row_count=profile_rows or None,
            column_count=(len(feature_columns) + len(required_columns)) if feature_columns or required_columns else None,
            feature_count=len(feature_columns),
            note="Live select-only profiling not executed in this run; static metadata recorded for planning.",
        )
        write_table_profile(profile, run_path / "feature_selection" / "profiles" / "source_table_profile.json")

    resource_cfg = stage_config.get("resource", {}) or stage_config.get("resource_planning", {}) or {}
    memory_snapshot = probe_memory(
        total_bytes=resource_cfg.get("total_memory_bytes"),
        available_bytes=resource_cfg.get("available_memory_bytes"),
    )
    peak_multiplier = float(resource_cfg.get("peak_multiplier") or resource_cfg.get("peak_memory_multiplier") or default_peak_multiplier_for_stage(stage))
    resource_plan = build_resource_plan_payload(
        data_source_mode=data_source_mode,
        stage=stage,
        memory_snapshot=memory_snapshot,
        total_rows=profile_rows,
        feature_column_count=max(len(feature_columns), 0),
        required_non_feature_column_count=max(len(required_columns), 1),
        peak_multiplier=peak_multiplier,
        memory_budget_fraction=float(resource_cfg.get("memory_budget_fraction", 0.8)),
        local_file_size_bytes=local_file_size,
    )
    _write_json(run_path / "feature_selection" / "resource_plan.json", resource_plan)

    max_rows = int(resource_plan["capacity"]["max_rows"])
    sampling_plan = build_sampling_plan(
        data_source_mode=data_source_mode,
        total_rows=profile_rows,
        max_rows=max_rows,
        random_columns=random_columns or [],
        preferred_random_column=(random_columns or [None])[0],
    )
    batch_size = int(resource_cfg.get("max_features_per_batch") or stage_config.get("max_features_per_batch") or 1000)
    batch_plan = build_feature_batch_plan(
        feature_columns=feature_columns,
        required_columns=required_columns,
        max_features_per_batch=batch_size,
    )
    persist_intake_plan(run_path, sampling_plan=sampling_plan, batch_plan=batch_plan)

    for relative in [
        "feature_selection/data_source_contract.json",
        "feature_selection/execution_environment.json",
        "feature_selection/resource_plan.json",
        "feature_selection/sampling_plan.json",
        "feature_selection/batch_plan.json",
    ]:
        _register_if_exists(run_path, stage, relative, description="Resource-aware feature intake evidence")
    for artifact in sorted((run_path / "feature_selection" / "profiles").glob("*.json")):
        register_artifact(run_path, stage, str(artifact.relative_to(run_path)), description="Feature source profile evidence")
    for artifact in sorted((run_path / "feature_selection" / "batches").glob("*.json")):
        register_artifact(run_path, stage, str(artifact.relative_to(run_path)), description="Feature batch plan")


def _write_sql_evidence_from_dir(run_path: Path, stage: str, sql_dir: Path) -> None:
    if not sql_dir.exists():
        return
    from risk_model_workbench.data.sql_evidence import write_sql_evidence

    for sql_file in sorted(sql_dir.glob("*.sql")):
        write_sql_evidence(
            run_path,
            sql_file.read_text(encoding="utf-8"),
            source=str(sql_file),
            purpose=sql_file.stem,
            stage=stage,
            sql_kind="generated",
            name=sql_file.name,
        )
    _register_if_exists(run_path, stage, "queries/sql_evidence_manifest.json", description="SQL evidence manifest")
    for artifact in sorted((run_path / "queries" / "generated").glob("*.sql")):
        register_artifact(run_path, stage, str(artifact.relative_to(run_path)), description="Generated SQL evidence")


def _runtime_config_arg(project_dir: Path, run_path: Path, name: str, explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    path = _runtime_config_path(run_path, project_dir, name)
    return str(path) if path.exists() else None


def _normal_algorithm(value: Any, default: str = "lightgbm") -> str:
    raw = str(value or default).strip().lower()
    aliases = {
        "lgb": "lightgbm",
        "lgbm": "lightgbm",
        "xgb": "xgboost",
        "lr": "logistic_regression",
        "logistic": "logistic_regression",
        "ranknet": "hier_ranknet",
        "hier_ranknet": "hier_ranknet",
        "teacher_student": "teacher_student_distillation",
    }
    return aliases.get(raw, raw)


def _experiment_config(train_config: dict[str, Any], experiment_name: str) -> dict[str, Any]:
    training = train_config.get("training", {})
    for item in training.get("experiments") or []:
        if isinstance(item, dict) and str(item.get("name")) == experiment_name:
            result = deepcopy(item)
            result["algorithm"] = _normal_algorithm(result.get("algorithm") or result.get("method"), training.get("default_algorithm", "lightgbm"))
            return result
    algorithm = _normal_algorithm(training.get("default_algorithm", "lightgbm"))
    return {"name": experiment_name, "algorithm": algorithm, "method": algorithm, "segment": "all"}


def _scores_feather_for_run(run_path: Path, explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    candidates = sorted((run_path / "modeling").glob("*/scores_all_splits.feather"))
    if candidates:
        return candidates[-1]
    return run_path / "modeling" / "scores_all_splits.feather"


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


def _read_only_action(action_id: str, operation):
    result, _retry_count = run_with_retry(action_id, operation)
    return result


def cmd_doctor(_: argparse.Namespace) -> int:
    """Check expected local files and optional dependencies."""
    checks = {
        "planning_doc": REPO_ROOT / "docs" / "legacy" / "AI经营建模Agent规划.md",
        "model_inventory": REPO_ROOT / "docs" / "legacy" / "现有经营模型梳理.md",
        "feature_select_v2_code": REPO_ROOT / "vendor" / "feature-select-v2" / "scripts" / "code" / "main.py",
        "project_template": REPO_ROOT / "templates" / "project" / "project.yml",
        "workflow_full_modeling": REPO_ROOT / "workflows" / "full_modeling.yml",
    }
    optional_checks = {
        "legacy_gcard_workbook": REPO_ROOT / "docs" / "legacy" / "复借G卡模型文档.xlsx",
    }

    ok = True
    for name, path in checks.items():
        exists = path.exists()
        ok = ok and exists
        print(f"{'OK' if exists else 'MISSING':7} {name}: {path}")
    for name, path in optional_checks.items():
        exists = path.exists()
        print(f"{'OK' if exists else 'MISSING':7} optional {name}: {path}")

    yaml_available = importlib.util.find_spec("yaml") is not None
    ok = ok and yaml_available
    print(f"{'OK' if yaml_available else 'MISSING':7} dependency: PyYAML")
    return 0 if ok else 1


def cmd_init_project(args: argparse.Namespace) -> int:
    project_dir = create_project(
        REPO_ROOT,
        name=args.name,
        display_name=args.display_name,
        scenario=args.scenario,
        template=args.template,
        force=args.force,
    )
    print(f"created: {project_dir}")
    return 0


def cmd_project_validate(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    config_path = project_config_path(project_dir)
    errors: list[str] = []
    if not config_path.exists():
        errors.append(f"missing project config: {config_path}")
    else:
        config = load_yaml(config_path)
        for key in ["project", "data", "segments"]:
            if key not in config:
                errors.append(f"missing top-level key: {key}")
        data = config.get("data", {})
        for key in ["source_table", "id_columns", "target_column", "time_column", "period_column"]:
            if not data.get(key):
                errors.append(f"missing data.{key}")
    for directory in ["configs", "queries", "reports"]:
        if not (project_dir / directory).exists():
            errors.append(f"missing directory: {directory}")
    if not (project_dir / "versions").exists() and not (project_dir / "runs").exists():
        errors.append("missing directory: versions or legacy runs")

    if errors:
        print("project validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"project validation ok: {project_dir}")
    return 0


def cmd_project_status(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    def _summarize_project_status() -> dict[str, Any]:
        if args.version_id:
            return summarize_project(project_dir, run_id=args.run_id, version_id=args.version_id)
        return summarize_project(project_dir, run_id=args.run_id)

    if args.write_state:
        summary = _summarize_project_status()
    else:
        summary = _read_only_action("project_status", _summarize_project_status)
    print(format_project_summary(summary), end="")
    if args.write_state:
        command = f"rmw project status --project {args.project}"
        if args.version_id:
            command += f" --version-id {args.version_id}"
        if args.run_id:
            command += f" --run-id {args.run_id}"
        path = write_project_state_from_summary(project_dir, summary, commands=[command])
        print(f"project_state: {path}")
    return 0


def cmd_project_update_state(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    state = update_project_state(
        project_dir,
        active_version_id=args.active_version_id,
        active_run_id=args.active_run_id,
        current_objective=args.objective,
        status=args.status,
        next_actions=args.next_action,
        blockers=args.blocker,
        risks=args.risk,
    )
    print(f"project_state: {project_dir / 'project_state.yml'}")
    if state.get("active_run_id"):
        print(f"active_run_id: {state['active_run_id']}")
    return 0


def cmd_handoff_write(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    path = write_handoff(
        project_dir,
        run_id=args.run_id,
        note=args.note or "",
        output=args.output,
        context_snapshot=args.context_snapshot,
    )
    print(f"handoff: {path}")
    return 0


def cmd_lesson_add(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    body = args.body or ""
    if args.body_file:
        body_path = Path(args.body_file)
        body_path = body_path if body_path.is_absolute() else (REPO_ROOT / body_path)
        body = body_path.read_text(encoding="utf-8")
    try:
        path = append_lesson(
            project_dir,
            title=args.title,
            body=body,
            kind=args.kind,
            scope=args.scope,
            source=args.source or "",
            tags=args.tag or [],
        )
    except ValueError as exc:
        print(f"lesson add failed: {exc}")
        return 1
    print(f"lesson: {path}")
    return 0


def cmd_lesson_promote(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    try:
        path, entry = promote_lesson_to_rule(
            project_dir,
            title=args.title,
            target=args.target,
            rule_id=args.rule_id,
            note=args.note or "",
        )
    except ValueError as exc:
        print(f"lesson promote failed: {exc}")
        return 1
    print(f"rules: {path}")
    print(f"rule_id: {entry.get('id')}")
    print(f"status: {entry.get('status')}")
    return 0


def cmd_rules_list(args: argparse.Namespace) -> int:
    payload = _read_only_action("rules_list", load_workbench_rules)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(format_rules(payload), end="")
    return 0


def cmd_run_audit(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    workspace_id = _workspace_arg(args)
    audit = _read_only_action("run_audit", lambda: audit_run(project_dir, workspace_id, stage=args.stage))
    if args.json:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
    else:
        print(format_run_audit(audit), end="")
    if args.strict and audit.get("verdict") != "complete":
        return 1
    return 0


def cmd_version_list(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    index = _read_only_action("version_list", lambda: load_version_index(project_dir))
    if args.json:
        print(json.dumps(index, ensure_ascii=False, indent=2))
        return 0
    print(f"active_version_id: {index.get('active_version_id', '')}")
    for item in index.get("versions", []) or []:
        print(
            f"- {item.get('version_id')}: "
            f"status={item.get('status', '')}, "
            f"source_type={item.get('source_type', '')}, "
            f"workflow={item.get('workflow', '')}, "
            f"legacy_run_id={item.get('legacy_run_id', '')}"
        )
    return 0


def cmd_version_status(args: argparse.Namespace) -> int:
    args.run_id = None
    return cmd_status(args)


def cmd_version_audit(args: argparse.Namespace) -> int:
    args.run_id = None
    return cmd_run_audit(args)


def cmd_version_show(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    path = resolve_workspace_dir(project_dir, version_id=args.version_id)
    state = _read_only_action("version_show", lambda: load_run_state(path))
    if args.json:
        print(json.dumps(state, ensure_ascii=False, indent=2, default=str))
    else:
        print(yaml.safe_dump(state, allow_unicode=True, sort_keys=False))
    return 0


def cmd_version_migrate_run(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    version_id = args.version_id or suggest_version_id(project_dir, args.run_id)
    result = migrate_run_to_version(
        project_dir,
        run_id=args.run_id,
        version_id=version_id,
        source_type=args.source_type,
        display_name=args.display_name,
        force=args.force,
    )
    print(f"{result['status']}: {result['version_id']}")
    print(f"version_dir: {result['path']}")
    return 0


def cmd_version_migrate_standard_runs(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    mappings = _load_version_migration_map(args.mapping) if args.mapping else {}
    results = []
    for run_path in standard_run_dirs(project_dir):
        run_id = run_path.name
        mapping = mappings.get(run_id, {})
        version_id = mapping.get("version_id") or suggest_version_id(project_dir, run_id)
        result = migrate_run_to_version(
            project_dir,
            run_id=run_id,
            version_id=version_id,
            source_type=mapping.get("source_type"),
            display_name=mapping.get("display_name"),
            force=args.force,
        )
        results.append(result)

    active_version_id = args.active_version_id
    if not active_version_id and any(item["version_id"] == "fujie_gcard_v7_20260630" for item in results):
        active_version_id = "fujie_gcard_v7_20260630"
    if active_version_id:
        index = load_version_index(project_dir)
        index["active_version_id"] = active_version_id
        save_version_index(project_dir, index)
        update_project_state(project_dir, active_version_id=active_version_id, status="active")

    for result in results:
        print(f"{result['status']}: {result['version_id']} <- {result['entry'].get('legacy_run_id', '')}")
    print(f"migrated_count: {len(results)}")
    if active_version_id:
        print(f"active_version_id: {active_version_id}")
    return 0


def cmd_version_register_manual(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    validate_version_id(args.version_id)
    path = version_dir(project_dir, args.version_id)
    if path.exists() and not args.force:
        print(f"version already exists: {path}")
        return 1
    for directory in ["audit", "reports"]:
        (path / directory).mkdir(parents=True, exist_ok=True)
    state = create_version_state(
        project_dir,
        version_id=args.version_id,
        workflow=args.workflow,
        stages=[],
        status=args.status,
        source_type="manual",
    )
    save_version_state(path, state)
    _write_json(path / "audit" / "artifact_manifest.json", {"version": 1, "artifacts": []})
    _write_text(path / "audit" / "decision_log.md", "# Decision Log\n\n- source_type: manual\n")
    upsert_version_index(
        project_dir,
        {
            "version_id": args.version_id,
            "display_name": args.display_name or args.version_id,
            "source_type": "manual",
            "status": args.status,
            "workflow": args.workflow,
            "path": str(path.relative_to(project_dir)),
            "created_at": state.get("created_at", ""),
            "updated_at": state.get("updated_at", ""),
        },
        active=args.active,
    )
    if args.active:
        update_project_state(project_dir, active_version_id=args.version_id, status="active")
    print(f"version_id: {args.version_id}")
    print(f"version_dir: {path}")
    return 0


def _load_version_migration_map(path_value: str) -> dict[str, dict[str, Any]]:
    path = Path(path_value)
    path = path if path.is_absolute() else (REPO_ROOT / path)
    payload = load_yaml(path)
    rows = payload.get("migrations", []) if isinstance(payload, dict) else []
    return {str(item["legacy_run_id"]): dict(item) for item in rows if item.get("legacy_run_id")}


def cmd_retrospective_write(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    try:
        path = write_retrospective(
            project_dir,
            run_id=args.run_id,
            scope=args.scope,
            stage=args.stage,
            outcome=args.outcome or "",
            note=args.note or "",
            lessons=args.lesson or [],
            output=args.output,
        )
    except ValueError as exc:
        print(f"retrospective write failed: {exc}")
        return 1
    print(f"retrospective: {path}")
    return 0


def cmd_workflow_show(args: argparse.Namespace) -> int:
    path = workflow_path(args.workflow)
    print(path.read_text(encoding="utf-8"))
    return 0


def cmd_workflow_validate(args: argparse.Namespace) -> int:
    path = workflow_path(args.workflow)
    if not path.exists():
        print(f"missing workflow: {path}")
        return 1
    errors = _read_only_action("workflow_validate", lambda: validate_workflow_definition(load_yaml(path)))
    if errors:
        print(f"workflow validation failed: {path}")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"workflow validation ok: {path}")
    return 0


def cmd_workflow_list(_: argparse.Namespace) -> int:
    for path in sorted((REPO_ROOT / "workflows").glob("*.yml")):
        print(path.stem)
    return 0


def _init_workflow_workspace(args: argparse.Namespace, *, as_version: bool) -> int:
    project_dir = resolve_project_path(args.project)
    workflow_file = workflow_path(args.workflow)
    workflow = load_yaml(workflow_file)
    workspace_id = args.version_id if as_version else (args.run_id or make_run_id())
    if as_version:
        validate_version_id(workspace_id)
        path = version_dir(project_dir, workspace_id)
    else:
        path = run_dir(project_dir, workspace_id)
    if path.exists() and not args.force:
        label = "version" if as_version else "run"
        print(f"{label} already exists: {path}")
        return 1

    # Parse + validate BEFORE scaffolding so a validation failure leaves no
    # orphan workspace on disk (no half-created directory, no version_index
    # entry, no stale stage state). Parsing reads the request file and does not
    # depend on the workspace path.
    request_doc: dict[str, Any] | None = None
    request_path: Path | None = None
    if getattr(args, "request", None):
        request_path = Path(args.request)
        request_path = request_path if request_path.is_absolute() else (REPO_ROOT / request_path)
        if request_path.exists():
            request_doc = parse_model_request(request_path)
    plan_payload: dict[str, Any] | None = None
    plan_path: Path | None = None
    if getattr(args, "plan", None):
        plan_path = Path(args.plan)
        plan_path = plan_path if plan_path.is_absolute() else (REPO_ROOT / plan_path)
        if plan_path.exists():
            plan_payload = load_yaml(plan_path)
    _request_warnings: list[str] = []
    if request_doc:
        # Gate the producer: validate the request before materializing, so a bad
        # split config (time-out leaking into validation) is caught at the entry
        # instead of silently written into runtime train.yaml.
        try:
            validation = validate_model_request(request_doc, project_dir)
        except Exception as exc:
            print(f"cannot init workspace; request validation failed: {exc}")
            return 1
        blocking = _split_blocking_errors(validation, args)
        if blocking:
            print("cannot init workspace; request validation failed:")
            for error in blocking:
                print(f"- {error}")
            return 1
        _request_warnings = validation["warnings"]

    for directory in ["configs_snapshot", RUNTIME_CONFIG_DIR, "audit", "tasks", "sample_check", "feature_selection", "modeling", "evaluation", "reports"]:
        (path / directory).mkdir(parents=True, exist_ok=True)
    for config_file in [project_config_path(project_dir), *sorted((project_dir / "configs").glob("*.y*ml"))]:
        if config_file.exists():
            shutil.copy2(config_file, path / "configs_snapshot" / config_file.name)

    if as_version:
        state = create_version_state(
            project_dir,
            version_id=workspace_id,
            workflow=workflow.get("name", args.workflow),
            stages=workflow.get("stages"),
            source_type=getattr(args, "source_type", "workbench"),
        )
        save_version_state(path, state)
    else:
        state = create_run_state(project_dir, run_id=workspace_id, workflow=workflow.get("name", args.workflow), stages=workflow.get("stages"))
        save_run_state(path, state)
    _write_json(path / "audit" / "artifact_manifest.json", {"version": 1, "artifacts": []})
    _write_text(path / "audit" / "command_log.jsonl", "")
    _write_text(path / "audit" / "decision_log.md", f"# Decision Log\n\n- imported: false\n")
    stage_action_started(path, "validate_config")
    register_artifact(path, "validate_config", "configs_snapshot", kind="directory", description="Project config snapshot")
    if request_path is not None and request_path.exists():
        shutil.copy2(request_path, path / "model_request.md")
        register_artifact(path, "validate_config", "model_request.md", description="Model request copied into run workspace")
    if plan_path is not None and plan_path.exists():
        shutil.copy2(plan_path, path / "execution_plan.yml")
        register_artifact(path, "validate_config", "execution_plan.yml", description="Execution plan copied into run workspace")
    if request_doc:
        for warning in _request_warnings:
            print(f"warning: {warning}")
        runtime_paths = materialize_request_runtime_configs(
            request_doc=request_doc,
            project_dir=project_dir,
            run_dir=path,
            plan=plan_payload,
            strict=not getattr(args, "skip_split_check", False),
        )
        register_artifact(path, "validate_config", RUNTIME_CONFIG_DIR, kind="directory", description="Request-materialized runtime configs")
        for runtime_path in runtime_paths.values():
            register_artifact(path, "validate_config", runtime_path.relative_to(path), description="Request-materialized runtime config")
    stage_action_done(path, "validate_config")
    if as_version:
        upsert_version_index(
            project_dir,
            {
                "version_id": workspace_id,
                "display_name": getattr(args, "display_name", None) or workspace_id,
                "source_type": getattr(args, "source_type", "workbench"),
                "status": "running",
                "workflow": workflow.get("name", args.workflow),
                "path": str(path.relative_to(project_dir)),
                "created_at": state.get("created_at", ""),
                "updated_at": state.get("updated_at", ""),
            },
            active=True,
        )
        update_project_state(project_dir, active_version_id=workspace_id, status="active")
        print(f"version_id: {workspace_id}")
        print(f"version_dir: {path}")
    else:
        print(f"run_id: {workspace_id}")
        print(f"run_dir: {path}")
    return 0


def cmd_run_init(args: argparse.Namespace) -> int:
    return _init_workflow_workspace(args, as_version=False)


def cmd_version_init(args: argparse.Namespace) -> int:
    return _init_workflow_workspace(args, as_version=True)


def _split_blocking_errors(result: dict[str, Any], args: argparse.Namespace) -> list[str]:
    """Combine structural errors with split errors, honoring --skip-split-check.

    Split errors (time-out pollution / empty in-time split) block by default;
    under --skip-split-check they are downgraded to printed warnings so an
    explicit escape hatch exists for urgent/exploratory runs. Structural errors
    always block regardless of the flag.
    """
    blocking = list(result.get("errors", []))
    split_errors = result.get("split_errors", [])
    if not split_errors:
        return blocking
    if getattr(args, "skip_split_check", False):
        for err in split_errors:
            print(f"warning (split, suppressed by --skip-split-check): {err}")
    else:
        blocking.extend(split_errors)
    return blocking


def cmd_request_validate(args: argparse.Namespace) -> int:
    request_path = Path(args.request)
    request_path = request_path if request_path.is_absolute() else (REPO_ROOT / request_path)
    project_dir = resolve_project_path(args.project) if args.project else None
    try:
        request_doc = parse_model_request(request_path)
        result = validate_model_request(request_doc, project_dir)
    except Exception as exc:
        print(f"request validation failed: {exc}")
        return 1

    blocking = _split_blocking_errors(result, args)
    if blocking:
        print("request validation failed:")
        for error in blocking:
            print(f"- {error}")
    else:
        print(f"request validation ok: {request_path}")
    for warning in result["warnings"]:
        print(f"warning: {warning}")
    return 0 if not blocking else 1


def cmd_plan_create(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    request_path = Path(args.request)
    request_path = request_path if request_path.is_absolute() else (REPO_ROOT / request_path)
    request_doc = parse_model_request(request_path)
    validation = validate_model_request(request_doc, project_dir)
    blocking = _split_blocking_errors(validation, args)
    if blocking:
        print("cannot create plan; request validation failed:")
        for error in blocking:
            print(f"- {error}")
        return 1

    plan = create_execution_plan(request_doc, args.project)
    output = Path(args.output) if args.output else project_dir / "requests" / f"{request_doc['metadata']['request_id']}.execution_plan.yml"
    output = output if output.is_absolute() else (REPO_ROOT / output)
    output_path = save_execution_plan(plan, output)
    print(f"execution_plan: {output_path}")
    print(f"task_count: {len(plan['tasks'])}")
    for warning in validation["warnings"]:
        print(f"warning: {warning}")
    return 0


def cmd_agent_start(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    if cmd_project_validate(argparse.Namespace(project=str(project_dir))) != 0:
        return 1

    request_path = Path(args.request)
    request_path = request_path if request_path.is_absolute() else (REPO_ROOT / request_path)
    try:
        request_doc = parse_model_request(request_path)
    except Exception as exc:
        print(f"agent start failed: request parse failed: {exc}")
        return 1
    if args.workflow:
        request_doc["metadata"]["workflow"] = args.workflow
    validation = validate_model_request(request_doc, project_dir)
    blocking = _split_blocking_errors(validation, args)
    if blocking:
        print("agent start failed: request validation failed:")
        for error in blocking:
            print(f"- {error}")
        return 1

    execution_plan = create_execution_plan(request_doc, project_dir)
    plan_output = project_dir / "requests" / f"{request_doc['metadata']['request_id']}.execution_plan.yml"
    execution_plan_path = save_execution_plan(execution_plan, plan_output)

    init_args = argparse.Namespace(
        project=str(project_dir),
        workflow=args.workflow,
        version_id=args.version_id,
        display_name=args.display_name,
        source_type="workbench",
        request=str(request_path),
        plan=str(execution_plan_path),
        force=False,
        skip_split_check=False,
    )
    if cmd_version_init(init_args) != 0:
        return 1

    workspace = resolve_workspace_dir(project_dir, version_id=args.version_id)
    agent_plan = bind_agent_plan(execution_plan, project_dir=project_dir, version_id=args.version_id)
    save_agent_plan(workspace, agent_plan)
    init_agent_state(workspace, project=str(project_dir), version_id=args.version_id, agent_plan=agent_plan)
    append_trace(
        workspace,
        "observation",
        {
            "summary": "RMW Agent initialized.",
            "version_id": args.version_id,
            "request": str(request_path),
            "execution_plan": str(execution_plan_path),
        },
    )
    register_artifact(workspace, "validate_config", "agent_plan.yml", description="Bound Agent execution plan")
    register_artifact(workspace, "validate_config", "audit/agent_state.yml", description="Agent runtime state")
    register_artifact(workspace, "validate_config", "audit/agent_trace.jsonl", description="Agent trace log")
    print(f"agent_plan: {workspace / 'agent_plan.yml'}")
    print(f"agent_state: {workspace / 'audit' / 'agent_state.yml'}")
    if args.execute:
        state = run_agent(project_dir, args.version_id, runner=main)
        print(f"agent_status: {state.get('status')}")
    return 0


def cmd_agent_run(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    try:
        state = run_agent(project_dir, args.version_id, runner=main)
    except ValueError as exc:
        print(f"agent run failed: {exc}")
        return 1
    print(f"agent_status: {state.get('status')}")
    blocker = state.get("blocker") if isinstance(state.get("blocker"), dict) else {}
    if blocker:
        print(f"blocker: {blocker.get('reason', '')}")
    return 0 if state.get("status") not in {"failed", "blocked"} else 1


def cmd_agent_resume(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    workspace = resolve_workspace_dir(project_dir, version_id=args.version_id)
    try:
        state = load_agent_state(workspace)
    except FileNotFoundError as exc:
        print(f"agent resume failed: {exc}")
        return 1
    blocker = state.get("blocker") if isinstance(state.get("blocker"), dict) else {}
    if state.get("status") == "waiting_for_approval" and blocker.get("approval_id"):
        approvals = load_yaml(workspace / "audit" / "approvals.yml") if (workspace / "audit" / "approvals.yml").exists() else {}
        approved = [
            item
            for item in approvals.get("approvals", []) or []
            if item.get("approval_id") == blocker.get("approval_id") and item.get("status") == "approved"
        ]
        if not approved:
            print(f"agent resume blocked: approval pending: {blocker.get('approval_id')}")
            return 1
    if state.get("status") == "waiting_for_advisor":
        request_id = str(blocker.get("advisor_request_id") or "")
        if request_id and not advisor_request_is_answered(workspace, request_id):
            print(f"agent resume blocked: advisor response pending: {request_id}")
            return 1
    try:
        resumed = run_agent(project_dir, args.version_id, runner=main)
    except ValueError as exc:
        print(f"agent resume failed: {exc}")
        return 1
    print(f"agent_status: {resumed.get('status')}")
    return 0 if resumed.get("status") not in {
        "failed",
        "blocked",
        "waiting_for_approval",
        "waiting_for_advisor",
        "waiting_for_user",
        "reconciliation_required",
    } else 1


def cmd_agent_status(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    workspace = resolve_workspace_dir(project_dir, version_id=args.version_id)
    try:
        agent_state = load_agent_state(workspace)
    except FileNotFoundError:
        agent_state = {}
    version_state = load_run_state(workspace)
    manifest = load_artifact_manifest(workspace)
    try:
        latest_audit = audit_run(project_dir, args.version_id)
    except Exception as exc:
        latest_audit = {"verdict": "unavailable", "error": str(exc)}
    payload = {
        "version": 1,
        "project": str(project_dir),
        "version_id": args.version_id,
        "agent_state": agent_state,
        "version_state": version_state,
        "artifact_manifest": {
            "version": manifest.get("version", 1),
            "artifact_count": len(manifest.get("artifacts", []) or []),
        },
        "latest_audit": latest_audit,
        "recent_trace": load_recent_trace(workspace, limit=args.tail),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return 0
    status = (agent_state or {}).get("status", "unknown")
    print(f"agent_status: {status}")
    print(f"version_status: {version_state.get('status', '')}")
    print(f"audit_verdict: {latest_audit.get('verdict', '')}")
    blocker = (agent_state or {}).get("blocker") if isinstance((agent_state or {}).get("blocker"), dict) else {}
    if blocker:
        print(f"blocker: {blocker.get('reason', '')}")
    return 0


def cmd_agent_approve(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    workspace = resolve_workspace_dir(project_dir, version_id=args.version_id)
    try:
        approval = approve_request(workspace, args.approval_id, approved_by=args.approved_by, note=args.note or "")
    except (KeyError, ValueError) as exc:
        print(str(exc))
        return 1
    append_trace(
        workspace,
        "decision",
        {
            "summary": "Approval recorded.",
            "approval_id": args.approval_id,
            "approved_by": args.approved_by,
        },
    )
    print(f"approval_status: {approval.get('status')}")
    print(f"approval_id: {approval.get('approval_id')}")
    return 0


def cmd_agent_reject(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    workspace = resolve_workspace_dir(project_dir, version_id=args.version_id)
    try:
        approval = reject_request(workspace, args.approval_id, rejected_by=args.rejected_by, note=args.note or "")
    except (KeyError, ValueError) as exc:
        print(str(exc))
        return 1
    append_trace(
        workspace,
        "decision",
        {
            "summary": "Approval rejected.",
            "approval_id": args.approval_id,
            "rejected_by": args.rejected_by,
        },
    )
    print(f"approval_status: {approval.get('status')}")
    print(f"approval_id: {approval.get('approval_id')}")
    return 0


def cmd_agent_tools(args: argparse.Namespace) -> int:
    tools = agent_tool_schema()
    if args.json:
        print(json.dumps(tools, ensure_ascii=False, indent=2))
    else:
        for tool in tools:
            print(f"{tool['name']}: permission={tool['permission']} approval={tool['requires_approval']}")
    return 0


def cmd_agent_plan_rebind(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    workspace = resolve_workspace_dir(project_dir, version_id=args.version_id)
    try:
        plan = load_agent_plan(workspace)
        state = load_agent_state(workspace)
    except (FileNotFoundError, ValueError) as exc:
        print(f"agent plan rebind failed: {exc}")
        return 1
    blocked_statuses = {
        "running",
        "waiting_for_approval",
        "waiting_for_advisor",
        "waiting_for_user",
        "reconciliation_required",
        "failed",
        "stopped",
        "done",
        "done_with_gaps",
    }
    if state.get("status") in blocked_statuses or any(
        task.get("status") == "running" for task in state.get("tasks", []) or []
    ):
        print(f"agent plan rebind failed: state is not rebind-safe: {state.get('status')}")
        return 1
    if state.get("plan_hash") != plan.get("plan_hash"):
        print("agent plan rebind failed: state plan_hash does not match the on-disk plan")
        return 1
    if (
        Path(str(state.get("project") or "")).resolve() != Path(str(plan.get("project") or "")).resolve()
        or state.get("version_id") != plan.get("version_id")
        or state.get("plan_id") != plan.get("plan_id")
    ):
        print("agent plan rebind failed: state scope does not match the on-disk plan")
        return 1
    unresolved_approvals = [
        item
        for item in load_approvals(workspace).get("approvals", []) or []
        if item.get("status") in {"pending", "approved"}
    ]
    unresolved_advisor = [
        item for item in list_advisor_requests(workspace) if item.get("status") not in {"consumed", "rejected"}
    ]
    if unresolved_approvals or unresolved_advisor:
        print("agent plan rebind failed: approval or Advisor evidence is still unresolved")
        return 1
    try:
        rebound, preview = rebind_agent_plan(plan)
    except ValueError as exc:
        print(f"agent plan rebind failed: {exc}")
        return 1
    preview["mode"] = "apply" if args.apply else "dry_run"
    if args.apply:
        save_agent_plan(workspace, rebound)
        state["plan_hash"] = rebound["plan_hash"]
        state["registry_digest"] = rebound["registry_digest"]
        save_agent_state(workspace, state)
    if args.json:
        print(json.dumps(preview, ensure_ascii=False, indent=2))
    else:
        print(f"rebind_mode: {preview['mode']}")
        print(f"changed: {str(preview['changed']).lower()}")
        print(f"registry_digest_after: {preview['registry_digest_after']}")
        print(f"plan_hash_after: {preview['plan_hash_after']}")
    return 0


def cmd_agent_advisor_list(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    workspace = resolve_workspace_dir(project_dir, version_id=args.version_id)
    rows = list_advisor_requests(workspace)
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
    else:
        if not rows:
            print("advisor_requests: []")
        for row in rows:
            print(f"{row.get('request_id')}: type={row.get('type')} status={row.get('status')} task={row.get('task_id')}")
    return 0


def cmd_agent_advisor_show(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    workspace = resolve_workspace_dir(project_dir, version_id=args.version_id)
    try:
        request = load_advisor_request(workspace, args.request_id)
    except KeyError as exc:
        print(str(exc))
        return 1
    if args.json:
        print(json.dumps(request, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"request_id: {request.get('request_id')}")
        print(f"type: {request.get('type')}")
        print(f"status: {request.get('status')}")
        print(f"question: {request.get('question')}")
        print("context_files:")
        for item in request.get("context_files") or []:
            print(f"- {item}")
    return 0


def cmd_agent_advisor_accept(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    workspace = resolve_workspace_dir(project_dir, version_id=args.version_id)
    result = accept_advisor_response(workspace, args.response)
    if not result.get("accepted"):
        print("advisor_response: rejected")
        for error in result.get("errors", []) or []:
            print(f"- {error}")
        return 1
    append_trace(
        workspace,
        "decision",
        {
            "summary": "Advisor response accepted.",
            "request_id": result.get("request_id"),
            "response_path": result.get("response_path"),
        },
    )
    print("advisor_response: accepted")
    print(f"request_id: {result.get('request_id')}")
    print(f"response_path: {result.get('response_path')}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    path = _run_path(args)
    if getattr(args, "progress", False):
        tail = int(getattr(args, "tail", 5) or 5)
        state, summary, events = _read_only_action(
            "run_status",
            lambda: (
                load_run_state(path),
                load_progress_summary(path),
                load_progress_events(path, tail=tail),
            ),
        )
        print(
            format_progress_report(
                run_state=state,
                summary=summary,
                events=events,
            ),
            end="",
        )
        return 0
    state = _read_only_action("run_status", lambda: load_run_state(path))
    print(yaml.safe_dump(state, allow_unicode=True, sort_keys=False))
    return 0


def cmd_run_watch(args: argparse.Namespace) -> int:
    path = _run_path(args)
    while True:
        state = load_run_state(path)
        print(
            format_progress_report(
                run_state=state,
                summary=load_progress_summary(path),
                events=load_progress_events(path, tail=args.tail),
            ),
            end="",
            flush=True,
        )
        if args.once:
            return 0
        time.sleep(args.interval)


def cmd_sample_check(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    path = _run_path(args)
    stage_action_started(path, "sample_check")
    config = _load_runtime_project_config(project_dir, path)
    data_cfg = config.get("data", {})
    raw_path = Path(data_cfg.get("raw_path") or "data/raw/sample.feather")
    if not raw_path.is_absolute():
        raw_path = project_dir / raw_path
    if raw_path.exists():
        try:
            import pandas as pd

            if raw_path.suffix == ".csv":
                df = pd.read_csv(raw_path)
            elif raw_path.suffix == ".parquet":
                df = pd.read_parquet(raw_path)
            else:
                df = pd.read_feather(raw_path)
            target_col = data_cfg.get("target_column")
            split_col = data_cfg.get("split_column") or config.get("split", {}).get("source_column")
            id_columns = [col for col in data_cfg.get("id_columns", []) if col in df.columns]
            time_col = data_cfg.get("time_column")
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
            _write_json(path / "sample_check" / "sample_summary.json", summary)
            if target_col in df.columns:
                df[target_col].value_counts(dropna=False).rename_axis("label").reset_index(name="count").to_csv(
                    path / "sample_check" / "label_distribution.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
            if split_col in df.columns:
                split_rows = df.groupby(split_col, dropna=False).size().reset_index(name="count")
                if target_col in df.columns:
                    label_series = pd.to_numeric(df[target_col], errors="coerce")
                    label_mean = df.assign(_target_numeric=label_series).groupby(split_col, dropna=False)["_target_numeric"].mean().reset_index(name="target_rate")
                    split_rows = split_rows.merge(label_mean, on=split_col, how="left")
                split_rows.to_csv(path / "sample_check" / "sample_split_summary.csv", index=False, encoding="utf-8-sig")
            if time_col in df.columns and target_col in df.columns:
                month = pd.to_datetime(df[time_col], errors="coerce").dt.to_period("M").astype(str)
                monthly = df.assign(_month=month).groupby("_month", dropna=False).agg(samples=(target_col, "count"), positive=(target_col, "sum"), target_rate=(target_col, "mean")).reset_index()
                monthly.to_csv(path / "sample_check" / "monthly_label_distribution.csv", index=False, encoding="utf-8-sig")
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
                    *config.get("data", {}).get("segment_columns", []),
                ]
                if col in df.columns
            ]
            if segment_cols:
                rows = []
                for column in dict.fromkeys(segment_cols):
                    for value, count in df[column].value_counts(dropna=False).items():
                        rows.append({"segment_column": column, "segment_value": str(value), "count": int(count), "ratio": float(count / len(df)) if len(df) else 0})
                pd.DataFrame(rows).to_csv(path / "sample_check" / "segment_distribution.csv", index=False, encoding="utf-8-sig")
            _write_text(path / "sample_check" / "sample_check_report.md", "# Sample Check\n\nstatus: done\n")
            for artifact in [
                "sample_check/sample_summary.json",
                "sample_check/sample_check_report.md",
                "sample_check/label_distribution.csv",
                "sample_check/sample_split_summary.csv",
                "sample_check/monthly_label_distribution.csv",
                "sample_check/segment_distribution.csv",
            ]:
                if (path / artifact).exists():
                    register_artifact(path, "sample_check", artifact)
            append_decision(path, stage="sample_check", decision="done", reason="Sample profiling completed from local data")
            stage_action_done(path, "sample_check")
            print(f"sample_check: {path / 'sample_check' / 'sample_summary.json'}")
            return 0
        except Exception as exc:
            stage_action_failed(path, "sample_check", str(exc), failure_code=classify_exception(exc))
            print(f"sample_check failed: {exc}", file=sys.stderr)
            return 1
    status = "scaffold"
    reason = "local data not available"
    summary = {
        "status": status,
        "reason": reason,
        "project": config.get("project", {}),
        "target_column": data_cfg.get("target_column"),
        "id_columns": data_cfg.get("id_columns", []),
        "split_column": data_cfg.get("split_column") or config.get("split", {}).get("source_column"),
        "expected_outputs": [
            "positive_rate_overall.csv",
            "positive_rate_by_split.csv",
            "positive_rate_by_month.csv",
            "positive_rate_by_segment.csv",
        ],
    }
    _write_json(path / "sample_check" / "sample_summary.json", summary)
    _write_text(
        path / "sample_check" / "sample_check_report.md",
        "# Sample Check\n\nstatus: scaffold\n\nreason: local data not available\n",
    )
    register_artifact(path, "sample_check", "sample_check/sample_summary.json")
    register_artifact(path, "sample_check", "sample_check/sample_check_report.md")
    append_decision(path, stage="sample_check", decision="scaffold", reason=reason)
    stage_action_done(path, "sample_check", scaffold=True, message=reason)
    print(f"sample_check: {path / 'sample_check' / 'sample_summary.json'}")
    return 0


def cmd_feature_metadata(args: argparse.Namespace) -> int:
    path = _run_path(args)
    project_dir = resolve_project_path(args.project)
    stage_action_started(path, "feature_metadata")
    from risk_model_workbench.feature_metadata import main as metadata_main

    argv = ["--project-dir", str(project_dir), "--run-dir", str(path)]
    config_arg = _runtime_config_arg(project_dir, path, "feature_select", args.config)
    project_config_arg = str(_runtime_config_dir(path) / "project.yml") if (_runtime_config_dir(path) / "project.yml").exists() else None
    if config_arg:
        argv.extend(["--config", config_arg])
    if project_config_arg:
        argv.extend(["--project-config", project_config_arg])
    tables_file = args.tables_file
    if not tables_file and config_arg:
        metadata_cfg = load_yaml(config_arg).get("feature_select", {}).get("metadata", {})
        tables_file = metadata_cfg.get("tables_file")
    if tables_file:
        argv.extend(["--tables-file", tables_file])
    code = metadata_main(argv)
    if code == 0:
        _register_feature_metadata_artifacts(path, project_dir)
        stage_action_done(path, "feature_metadata")
    else:
        stage_action_failed(path, "feature_metadata", f"metadata command exited with code {code}")
    return code


def _finish_feature_prescreen_local_feather(path: Path, project_dir: Path, stage: str) -> int:
    """In local_feather mode the wide table is the sample file itself; emit intake
    evidence and finish prescreen as ``done`` (by design) without remote DP pull.

    Produces the ``feature_prescreen`` contract-accepted artifact set #2
    (data_source_contract / resource_plan / sampling_plan / batch_plan), so the
    auditor verdict converges to ``complete`` instead of ``scaffold``.
    """
    try:
        feature_cfg = _load_runtime_config(project_dir, path, "feature_select").get("feature_select", {})
        prescreen_cfg = feature_cfg.get("prescreen", {}) or feature_cfg.get("d01_d02", {}) or {}
        try:
            runtime_project = _load_runtime_project_config(project_dir, path)
        except FileNotFoundError:
            runtime_project = {}
        data_cfg = runtime_project.get("data", {}) or {}
        split_cfg = runtime_project.get("split", {}) or {}
        feature_columns_path = _resolve_project_relative(
            project_dir,
            prescreen_cfg.get("feature_columns", "data/profile/feature_metadata/feature_columns.csv"),
        )
        feature_columns = _feature_columns_from_csv(feature_columns_path)
        required_columns = [
            column
            for column in list(
                dict.fromkeys(
                    [*map(str, _as_list(data_cfg.get("id_columns")))]
                    + [
                        str(_first_value(prescreen_cfg.get("target_col"), data_cfg.get("target_column"), "")),
                        str(
                            _first_value(
                                prescreen_cfg.get("split_col"),
                                split_cfg.get("source_column"),
                                data_cfg.get("split_column"),
                                "",
                            )
                        ),
                    ]
                    + [str(data_cfg.get("period_column") or "")]
                )
            )
            if column
        ]
        resource_cfg = prescreen_cfg.get("resource", {}) or prescreen_cfg.get("resource_planning", {}) or {}
        random_cfg = prescreen_cfg.get("sampling", {}) or {}
        random_columns = [
            str(item)
            for item in _as_list(_first_value(random_cfg.get("random_columns"), random_cfg.get("random_column")))
        ]
        _write_feature_intake_evidence(
            run_path=path,
            project_dir=project_dir,
            stage=stage,
            stage_config=prescreen_cfg,
            feature_columns=feature_columns,
            required_columns=required_columns or ["__required_placeholder"],
            source_table=data_cfg.get("source_table"),
            local_feather_path=data_cfg.get("raw_path") if data_cfg.get("raw_path") else None,
            total_rows=resource_cfg.get("total_rows") or random_cfg.get("total_rows"),
            random_columns=random_columns,
        )
        for artifact_name in (
            "data_source_contract.json",
            "resource_plan.json",
            "sampling_plan.json",
            "batch_plan.json",
            "execution_environment.json",
        ):
            _register_if_exists(path, stage, Path("feature_selection") / artifact_name)
        _register_if_exists(path, stage, "feature_selection/profiles/local_feather_profile.json")
    except Exception as exc:
        stage_action_failed(
            path,
            stage,
            f"local feather prescreen evidence failed: {exc}",
            failure_code=classify_exception(exc),
        )
        print(f"feature prescreen local-feather evidence failed: {exc}", file=sys.stderr)
        return 1
    stage_action_done(
        path,
        stage,
        message="local feather mode: prescreen by design, intake evidence emitted",
    )
    return 0


def cmd_feature_prescreen(args: argparse.Namespace) -> int:
    path = _run_path(args)
    project_dir = resolve_project_path(args.project)
    stage = _feature_prescreen_stage(path)
    stage_action_started(path, stage)
    if _runtime_is_local_feather(path, project_dir):
        return _finish_feature_prescreen_local_feather(path, project_dir, stage)
    from risk_model_workbench.batch_feature_select import main as batch_select_main

    argv = ["--project-dir", str(project_dir), "--run-dir", str(path), "--stage", stage]
    config_arg = _runtime_config_arg(project_dir, path, "feature_select", args.config)
    if config_arg:
        argv.extend(["--config", config_arg])
    if args.max_tables is not None:
        argv.extend(["--max-tables", str(args.max_tables)])
    if args.table:
        for table in args.table:
            argv.extend(["--table", table])
    if args.dry_run_sql:
        argv.append("--dry-run-sql")
    if args.refresh_dp_cache:
        argv.append("--refresh-dp-cache")
    if args.sql_approved:
        argv.append("--sql-approved")
    if args.force:
        argv.append("--force")
    try:
        code = batch_select_main(argv)
    except Exception as exc:
        stage_action_failed(path, stage, str(exc), failure_code=classify_exception(exc))
        print(f"feature prescreen failed: {exc}", file=sys.stderr)
        return 1
    if code == 0:
        try:
            feature_cfg = _load_runtime_config(project_dir, path, "feature_select").get("feature_select", {})
            prescreen_cfg = feature_cfg.get("prescreen", {}) or feature_cfg.get("d01_d02", {}) or {}
            try:
                runtime_project = _load_runtime_project_config(project_dir, path)
            except FileNotFoundError:
                runtime_project = {}
            data_cfg = runtime_project.get("data", {}) or {}
            split_cfg = runtime_project.get("split", {}) or {}
            feature_columns_path = _resolve_project_relative(
                project_dir,
                prescreen_cfg.get("feature_columns", "data/profile/feature_metadata/feature_columns.csv"),
            )
            feature_columns = _feature_columns_from_csv(feature_columns_path)
            required_columns = list(
                dict.fromkeys(
                    [*map(str, _as_list(data_cfg.get("id_columns")))]
                    + [
                        str(_first_value(prescreen_cfg.get("target_col"), data_cfg.get("target_column"), "")),
                        str(_first_value(prescreen_cfg.get("split_col"), split_cfg.get("source_column"), data_cfg.get("split_column"), "")),
                    ]
                    + [str(data_cfg.get("period_column") or "")]
                )
            )
            required_columns = [column for column in required_columns if column]
            resource_cfg = prescreen_cfg.get("resource", {}) or prescreen_cfg.get("resource_planning", {}) or {}
            random_cfg = prescreen_cfg.get("sampling", {}) or {}
            random_columns = [str(item) for item in _as_list(_first_value(random_cfg.get("random_columns"), random_cfg.get("random_column")))]
            _write_feature_intake_evidence(
                run_path=path,
                project_dir=project_dir,
                stage=stage,
                stage_config=prescreen_cfg,
                feature_columns=feature_columns,
                required_columns=required_columns or ["__required_placeholder"],
                source_table=data_cfg.get("source_table"),
                local_feather_path=data_cfg.get("raw_path") if data_cfg.get("raw_path") else None,
                total_rows=resource_cfg.get("total_rows") or random_cfg.get("total_rows"),
                random_columns=random_columns,
            )
            sql_dir = _feature_prescreen_output_dir(project_dir, config_arg) / "sql"
            _write_sql_evidence_from_dir(path, stage, sql_dir)
        except Exception as exc:
            stage_action_failed(path, stage, f"feature intake evidence failed: {exc}", failure_code=classify_exception(exc))
            print(f"feature prescreen evidence registration failed: {exc}", file=sys.stderr)
            return 1
        if not args.dry_run_sql:
            _register_feature_prescreen_artifacts(path, stage, project_dir, config_arg)
        stage_action_done(
            path,
            stage,
            scaffold=args.dry_run_sql,
            message="SQL dry run waiting for approval" if args.dry_run_sql else "",
            failure_code=SQL_APPROVAL_REQUIRED if args.dry_run_sql else "",
        )
    else:
        stage_action_failed(path, stage, f"feature prescreen command exited with code {code}")
    return code


def _finish_build_wide_sql_local_feather(run_path: Path, reporter: Any) -> int:
    """In local_feather mode the wide table pre-exists as the sample file; skip
    SQL generation/DP execution and finish build_wide_sql as ``done`` (by design).

    Emits ``feature_selection/wide_table_skipped.json`` (the contract-accepted
    skip set, ``full_modeling.yml`` build_wide_sql), so the auditor verdict is
    ``complete`` and the stage no longer fails on a missing prescreen remain-features
    artifact.
    """
    skip_payload = {
        "data_source_mode": "local_feather",
        "reason": "local_feather_wide_table_is_feather",
        "note": (
            "In local_feather mode the wide table pre-exists as the sample feather; "
            "wide-table SQL generation and DP execution are skipped by design."
        ),
        "stage": "build_wide_sql",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    skip_path = run_path / "feature_selection" / "wide_table_skipped.json"
    skip_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(skip_path, skip_payload)
    register_artifact(
        run_path,
        "build_wide_sql",
        "feature_selection/wide_table_skipped.json",
        description="Local-feather wide-table skip (by design)",
    )
    if reporter is not None:
        reporter.emit(step="build_sql", message="本地 feather 模式：宽表即样本文件，按设计跳过", percent=100)
    stage_action_done(run_path, "build_wide_sql", message="local feather mode: wide table skipped by design")
    return 0


def cmd_build_wide_sql(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    reporter = None
    run_path = None
    if getattr(args, "version_id", None) or getattr(args, "run_id", None):
        run_path = _run_path(args)
        stage_action_started(run_path, "build_wide_sql")
        reporter = ProgressReporter(run_path, "build_wide_sql")
        reporter.emit(step="build_sql", message="开始生成宽表 SQL", percent=10)
    if run_path is not None and _runtime_is_local_feather(run_path, project_dir):
        return _finish_build_wide_sql_local_feather(run_path, reporter)
    config_path = _runtime_config_path(run_path, project_dir, "feature_select") if run_path else None
    runtime_feature_cfg = load_yaml(config_path).get("feature_select", {}) if config_path and config_path.exists() else {}
    runtime_wide_cfg = runtime_feature_cfg.get("wide_table", {}) or {}
    runtime_config_active = bool(
        run_path
        and config_path
        and config_path.exists()
        and config_path.parent.resolve() == _runtime_config_dir(run_path).resolve()
    )

    def _runtime_default_path(arg_value: str, default_value: str, config_key: str) -> Path:
        value = arg_value
        if runtime_config_active and arg_value == default_value:
            value = str(runtime_wide_cfg.get(config_key) or arg_value)
        path = Path(value)
        return path if path.is_absolute() else project_dir / path

    remain_features_path = _runtime_default_path(
        args.remain_features,
        DEFAULT_PRESCREEN_REMAIN_FEATURES,
        "remain_features_path",
    )
    if not remain_features_path.is_absolute():
        remain_features_path = project_dir / remain_features_path
    if not runtime_config_active and args.remain_features == DEFAULT_PRESCREEN_REMAIN_FEATURES and not remain_features_path.exists():
        legacy_path = project_dir / LEGACY_PRESCREEN_REMAIN_FEATURES
        if legacy_path.exists():
            remain_features_path = legacy_path
    execution_path = Path(args.execution_output)
    if not execution_path.is_absolute():
        execution_path = (run_path if run_path else project_dir) / execution_path
    try:
        project_runtime_path = _runtime_config_dir(run_path) / "project.yml" if run_path else None
        sql_output_path = _runtime_default_path(args.sql_output, DEFAULT_WIDE_SQL_OUTPUT, "sql_output")
        feature_map_output_path = _runtime_default_path(
            args.feature_map_output,
            DEFAULT_WIDE_FEATURE_MAP_OUTPUT,
            "feature_map_output",
        )
        summary_output_path = _runtime_default_path(
            args.summary_output,
            DEFAULT_WIDE_SUMMARY_OUTPUT,
            "summary_output",
        )
        sql_path, feature_map_path, summary_path = generate_wide_sql(
            project_dir=project_dir,
            remain_features_path=remain_features_path,
            sql_output_path=sql_output_path,
            feature_map_path=feature_map_output_path,
            summary_path=summary_output_path,
            base_table=args.base_table,
            output_table=args.output_table,
            base_where=args.base_where,
            feature_where=args.feature_where,
            config_path=config_path if config_path and config_path.exists() else None,
            project_config_path=project_runtime_path if project_runtime_path and project_runtime_path.exists() else None,
        )
        if run_path:
            _copy_and_register_artifact(
                run_path,
                "build_wide_sql",
                sql_path,
                _query_artifact_relative(project_dir, sql_path),
                description="Generated wide-table create SQL",
            )
            _copy_and_register_artifact(
                run_path,
                "build_wide_sql",
                feature_map_path,
                "feature_selection/prescreen_wide_feature_map.csv",
                description="Wide-table feature output mapping",
            )
            _copy_and_register_artifact(
                run_path,
                "build_wide_sql",
                summary_path,
                "feature_selection/wide_sql_summary.json",
                description="Wide-table SQL generation summary",
            )
            from risk_model_workbench.data.sql_evidence import write_sql_evidence

            sql_text = sql_path.read_text(encoding="utf-8")
            write_sql_evidence(
                run_path,
                sql_text,
                source=str(sql_path),
                purpose="build_wide_sql",
                stage="build_wide_sql",
                sql_kind="generated",
                name=sql_path.name,
            )
            _register_if_exists(run_path, "build_wide_sql", "queries/sql_evidence_manifest.json", description="SQL evidence manifest")
            _register_if_exists(run_path, "build_wide_sql", Path("queries") / "generated" / sql_path.name, description="Generated wide SQL evidence")
            from risk_model_workbench.data.sql_review import review_sql_text

            runtime_project = _load_runtime_project_config(project_dir, run_path)
            review_config = load_yaml(config_path).get("feature_select", {}) if config_path and config_path.exists() else {}
            runtime_request = review_config.get("runtime_request", {})
            sql_gate_cfg = (runtime_request.get("step_params") or {}).get("sql_review_gate", {})
            review_payload = review_sql_text(
                sql_text,
                approved_for_execution=bool(args.sql_approved),
                target_columns=[runtime_project.get("data", {}).get("target_column", "")],
                time_columns=[
                    runtime_project.get("data", {}).get("time_column", ""),
                    runtime_project.get("data", {}).get("period_column", ""),
                ],
            )
            review_payload["block_on_high_risk"] = bool(sql_gate_cfg.get("block_on_high_risk", True))
            review_payload["sql_path"] = str(sql_path)
            _write_json(run_path / "feature_selection" / "sql_review.json", review_payload)
            register_artifact(run_path, "build_wide_sql", "feature_selection/sql_review.json", description="Static SQL review result")
            if review_payload["high_risk"]:
                raise RuntimeError("SQL review blocked high-risk generated SQL")
        if args.execute:
            from risk_model_workbench.dp_feather import execute_dp_sql, sha256_text

            sql_text = sql_path.read_text(encoding="utf-8")
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            execution_result = execute_dp_sql(
                project_dir=project_dir,
                sql=sql_text,
                operation_id="build_wide_sql_execute",
                description=f"Create wide feature table {summary_payload.get('output_table', args.output_table or '')}".strip(),
                metadata_path=execution_path,
                sql_approved=args.sql_approved,
                progress=reporter,
                audit_workspace=run_path,
            )
            execution_payload = {
                **execution_result,
                "approved": bool(args.sql_approved),
                "sql_path": str(sql_path),
                "summary_path": str(summary_path),
                "feature_map_path": str(feature_map_path),
                "output_table": summary_payload.get("output_table") or args.output_table,
                "sql_sha256": sha256_text(sql_text),
            }
            _write_json(execution_path, execution_payload)
            if run_path:
                _copy_and_register_artifact(
                    run_path,
                    "build_wide_sql",
                    execution_path,
                    "feature_selection/wide_table_execution.json",
                    description="Wide-table create SQL execution metadata",
                )
                from risk_model_workbench.data.table_profile import build_static_table_profile, write_table_profile

                wide_profile = build_static_table_profile(
                    str(summary_payload.get("output_table") or args.output_table or ""),
                    column_count=summary_payload.get("features"),
                    feature_count=summary_payload.get("features"),
                    source="wide_sql_execution_metadata",
                    status="metadata_only_after_ctas",
                    note="CTAS execution completed; live select-only post-create profiling was not executed in this environment.",
                )
                profile_path = write_table_profile(
                    wide_profile,
                    run_path / "feature_selection" / "profiles" / "wide_table_profile.json",
                )
                register_artifact(
                    run_path,
                    "build_wide_sql",
                    str(profile_path.relative_to(run_path)),
                    description="Wide table post-create profile metadata",
                )
    except Exception as exc:
        if run_path:
            stage_action_failed(run_path, "build_wide_sql", str(exc), failure_code=classify_exception(exc))
        print(f"build-wide-sql failed: {exc}", file=sys.stderr)
        return 1
    if reporter:
        reporter.emit(
            step="write_outputs",
            message="宽表 SQL、特征映射和摘要已生成" + ("，建表 SQL 已执行" if args.execute else ""),
            percent=90,
            metrics={
                "sql_path": str(sql_path),
                "feature_map_path": str(feature_map_path),
                "summary_path": str(summary_path),
                "executed": bool(args.execute),
            },
        )
    if run_path:
        stage_action_done(
            run_path,
            "build_wide_sql",
            scaffold=not args.execute,
            message="SQL generation complete; waiting for approval" if not args.execute else "",
            failure_code=SQL_APPROVAL_REQUIRED if not args.execute else "",
        )
    print(f"sql: {sql_path}")
    print(f"feature_map: {feature_map_path}")
    print(f"summary: {summary_path}")
    if args.execute:
        print(f"execution: {execution_path}")
    return 0


def cmd_feature_refine(args: argparse.Namespace) -> int:
    path = _run_path(args)
    project_dir = resolve_project_path(args.project)
    stage_action_started(path, "feature_refine")
    from risk_model_workbench.feature_refine import main as refine_main

    argv = ["--project-dir", str(project_dir), "--run-dir", str(path)]
    config_arg = _runtime_config_arg(project_dir, path, "refine_features", args.config)
    if config_arg:
        argv.extend(["--config", config_arg])
    if args.dry_run_sql:
        argv.append("--dry-run-sql")
    if args.refresh_dp_cache:
        argv.append("--refresh-dp-cache")
    if args.sql_approved:
        argv.append("--sql-approved")
    if args.sample_max_rows is not None:
        argv.extend(["--sample-max-rows", str(args.sample_max_rows)])
    try:
        code = refine_main(argv)
    except Exception as exc:
        stage_action_failed(path, "feature_refine", str(exc), failure_code=classify_exception(exc))
        print(f"feature refine failed: {exc}", file=sys.stderr)
        return 1
    if code == 0:
        try:
            refine_cfg = _load_runtime_config(project_dir, path, "refine_features").get("feature_refine", {})
            if args.dry_run_sql and not _runtime_is_local_feather(path, project_dir):
                from risk_model_workbench.data.sql_evidence import write_sql_evidence
                from risk_model_workbench.feature_refine import build_sampling_sql

                prepared_sql = build_sampling_sql(
                    refine_cfg,
                    _refine_feature_columns(project_dir, {"feature_refine": refine_cfg}),
                )
                entry = write_sql_evidence(
                    path,
                    prepared_sql,
                    source="feature_refine.build_sampling_sql",
                    purpose="feature_refine_sample",
                    stage="feature_refine",
                    sql_kind="generated",
                    name="feature_refine_sample.sql",
                )
                _register_if_exists(path, "feature_refine", "queries/sql_evidence_manifest.json", description="SQL evidence manifest")
                register_artifact(path, "feature_refine", entry["path"], description="Generated feature refine SQL evidence")
            try:
                runtime_project = _load_runtime_project_config(project_dir, path)
            except FileNotFoundError:
                runtime_project = {}
            data_cfg = runtime_project.get("data", {}) or {}
            input_cfg = refine_cfg.get("input", {}) or {}
            required_columns = list(
                dict.fromkeys(
                    [*map(str, _as_list(input_cfg.get("id_columns") or data_cfg.get("id_columns")))]
                    + [str(item) for item in _as_list(input_cfg.get("base_columns"))]
                    + [
                        str(_first_value(input_cfg.get("label_column"), data_cfg.get("target_column"), "")),
                        str(_first_value(input_cfg.get("split_column"), data_cfg.get("split_column"), "")),
                    ]
                )
            )
            required_columns = [column for column in required_columns if column]
            runtime_request = refine_cfg.get("runtime_request", {}) or {}
            local_path = _first_value(
                input_cfg.get("local_feather_path"),
                input_cfg.get("raw_path"),
                input_cfg.get("feather_path"),
                runtime_request.get("sample_location") if runtime_request.get("data_source_mode") == "local_feather" else None,
                data_cfg.get("raw_path"),
            )
            sampling_cfg = refine_cfg.get("sampling", {}) or {}
            resource_cfg = refine_cfg.get("resource", {}) or refine_cfg.get("resource_planning", {}) or {}
            random_columns = [str(item) for item in _as_list(_first_value(sampling_cfg.get("random_columns"), sampling_cfg.get("random_column")))]
            _write_feature_intake_evidence(
                run_path=path,
                project_dir=project_dir,
                stage="feature_refine",
                stage_config=refine_cfg,
                feature_columns=_refine_feature_columns(project_dir, {"feature_refine": refine_cfg}),
                required_columns=required_columns or ["__required_placeholder"],
                source_table=input_cfg.get("wide_table") or data_cfg.get("source_table"),
                local_feather_path=local_path,
                total_rows=resource_cfg.get("total_rows") or sampling_cfg.get("total_rows"),
                random_columns=random_columns,
            )
        except Exception as exc:
            stage_action_failed(path, "feature_refine", f"feature intake evidence failed: {exc}", failure_code=classify_exception(exc))
            print(f"feature refine evidence registration failed: {exc}", file=sys.stderr)
            return 1
        if not args.dry_run_sql:
            output_dir = _feature_refine_output_dir_for_run(project_dir, path, config_arg)
            try:
                required_outputs = [
                    (output_dir / "stage_summary.json", "feature_selection/stage_summary.json", "Feature refinement stage summary"),
                    (
                        output_dir / "stage_summary.json",
                        "feature_selection/feature_stage_summary.json",
                        "Feature refinement stage summary compatibility copy",
                    ),
                    (output_dir / "final_500_features.txt", "feature_selection/final_500_features.txt", "Final refined feature list"),
                    (output_dir / "final_features.txt", "feature_selection/final_features.txt", "Final refined feature list"),
                ]
                for source, target, description in required_outputs:
                    copied = _copy_and_register_artifact(
                        path,
                        "feature_refine",
                        source,
                        target,
                        description=description,
                    )
                    if copied is None:
                        raise FileNotFoundError(f"required feature refine artifact missing: {source}")
                optional_outputs = [
                    (output_dir / "resource_usage.json", "feature_selection/resource_usage.json", "Feature refinement runtime memory usage"),
                ]
                for source, target, description in optional_outputs:
                    _copy_and_register_artifact(
                        path,
                        "feature_refine",
                        source,
                        target,
                        description=description,
                    )
            except Exception as exc:
                stage_action_failed(path, "feature_refine", str(exc), failure_code=classify_exception(exc))
                print(f"feature refine artifact registration failed: {exc}", file=sys.stderr)
                return 1
        stage_action_done(
            path,
            "feature_refine",
            scaffold=args.dry_run_sql,
            message="SQL dry run waiting for approval" if args.dry_run_sql else "",
            failure_code=SQL_APPROVAL_REQUIRED if args.dry_run_sql else "",
        )
    else:
        stage_action_failed(path, "feature_refine", f"feature refine command exited with code {code}")
    return code


def cmd_train(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    path = _run_path(args)
    stage_action_started(path, "train_baseline")
    reporter = ProgressReporter(path, "train_baseline")
    config_arg = _runtime_config_arg(project_dir, path, "train", args.config)
    config_path = Path(config_arg) if config_arg else project_dir / "configs" / "train.yaml"
    config_path = config_path if config_path.is_absolute() else project_dir / config_path
    train_config = load_yaml(config_path) if config_path.exists() else {}
    project_cfg = _load_runtime_project_config(project_dir, path)
    effective_config = deepcopy(train_config)
    defaults = project_training_defaults(project_cfg)
    if defaults:
        effective_config["training"] = merge_training_config(defaults, effective_config.get("training", {}))
    runtime_experiment = _experiment_config(effective_config, args.experiment) if effective_config else {"name": args.experiment, "algorithm": "lightgbm"}
    algorithm = _normal_algorithm(runtime_experiment.get("algorithm") or runtime_experiment.get("method"))
    if effective_config:
        effective_config["runtime_experiment"] = runtime_experiment
        effective_config["runtime_step_params"] = effective_config.get("training", {}).get("runtime_step_params", {})
        input_cfg = effective_config.setdefault("input", {})
        data_cfg = project_cfg.get("data", {})
        if data_cfg.get("time_column"):
            input_cfg.setdefault("time_column", data_cfg.get("time_column"))
        if data_cfg.get("period_column"):
            input_cfg.setdefault("period_column", data_cfg.get("period_column"))
        if data_cfg.get("segment_columns"):
            input_cfg.setdefault("segment_columns", data_cfg.get("segment_columns"))
    configured_input = train_config.get("input", {}).get("feather_path")
    input_feather = Path(args.input_feather or configured_input or "")
    if input_feather and not input_feather.is_absolute():
        input_feather = project_dir / input_feather
    feature_list = Path(args.feature_list or train_config.get("training", {}).get("feature_list_path", "runs/modeling_feature_set/feature_list.txt"))
    if not feature_list.is_absolute():
        feature_list = project_dir / feature_list
    output_dir = path / "modeling" / args.experiment
    score_output = Path(args.score_output or path / "modeling" / args.experiment / "scores_all_splits.feather")
    if not score_output.is_absolute():
        score_output = project_dir / score_output
    input_snapshot_dir = Path(args.input_dir or path / "modeling_input")
    if not input_snapshot_dir.is_absolute():
        input_snapshot_dir = project_dir / input_snapshot_dir
    train_plan = _build_train_plan(
        workspace=path,
        experiment=args.experiment,
        algorithm=algorithm,
        config_path=config_path,
        train_config=train_config,
        effective_config=effective_config,
        input_feather=input_feather,
        feature_list=feature_list,
        score_output=score_output,
        input_snapshot_dir=input_snapshot_dir,
        skip_split_check=bool(getattr(args, "skip_split_check", False)),
    )

    # Train-time split guard: the request/init gate protects only the
    # request->materialize path. This is the last line of defense for paths that
    # bypass materialization — legacy configs/train.yaml fallback (which may
    # hardcode valid_values:[OOT]), explicit --config, a configs_runtime
    # polluted under a prior --skip-split-check, or a hand-edited runtime config.
    _raw_valid = effective_config.get("training", {}).get("valid_values")
    if not isinstance(_raw_valid, list):
        _raw_valid = [] if _raw_valid is None else [_raw_valid]
    _valid_for_early_stop = [str(v) for v in _raw_valid if str(v)]
    if _valid_for_early_stop:
        from risk_model_workbench.request.splits import resolve_split_values as _resolve_split_values

        _oot_labels = set(_resolve_split_values({}, project_cfg)["oot"]["values"])
        _polluting = sorted(set(_valid_for_early_stop) & _oot_labels)
        if _polluting:
            _reason = (
                f"training.valid_values {_valid_for_early_stop} 含时间外标签 {_polluting} "
                f"(oot={sorted(_oot_labels)})；早停验证集不得用时间外样本，会污染时间外评估。"
                f"修复 config 的 training.valid_values，或用 --skip-split-check 显式承担风险。"
            )
            if getattr(args, "skip_split_check", False):
                print(f"warning (split, suppressed by --skip-split-check): {_reason}")
            else:
                output_dir.mkdir(parents=True, exist_ok=True)
                _write_json(output_dir / "train_metrics.json", {"status": "failed", "reason": _reason, "experiment": args.experiment, "algorithm": algorithm})
                status_payload = _write_training_status(
                    output_dir,
                    status="failed",
                    experiment=args.experiment,
                    algorithm=algorithm,
                    message=_reason,
                    failure_code="data_missing",
                    error_type="SplitValidationError",
                    error_message=_reason,
                    completed=True,
                )
                _write_training_summary(output_dir, status_payload=status_payload, plan=train_plan)
                register_artifact(path, "train_baseline", f"modeling/{args.experiment}/train_metrics.json")
                register_artifact(path, "train_baseline", f"modeling/{args.experiment}/training_status.json")
                register_artifact(path, "train_baseline", f"modeling/{args.experiment}/training_summary.md")
                stage_action_failed(path, "train_baseline", _reason)
                print(f"train blocked: {_reason}", file=sys.stderr)
                return 1

    if getattr(args, "plan_only", False):
        _write_train_plan(output_dir, train_plan)
        for artifact in ["train_plan.json", "train_plan.md"]:
            register_artifact(path, "train_baseline", f"modeling/{args.experiment}/{artifact}")
        append_decision(path, stage="train_baseline", decision="plan_only", reason="training plan preview generated; model training was not executed")
        stage_action_done(path, "train_baseline", scaffold=True, message="training plan preview only")
        print(f"train plan written: {output_dir / 'train_plan.md'}")
        return 0

    if algorithm == "custom" and not (train_config.get("custom_training", {}).get("entrypoint") or train_config.get("training", {}).get("custom_entrypoint")):
        reason = "custom training requires training.custom_entrypoint or custom_training.entrypoint in project/runtime config"
        payload = {"status": "failed", "reason": reason, "experiment": args.experiment, "algorithm": algorithm}
        _write_json(output_dir / "train_metrics.json", payload)
        status_payload = _write_training_status(
            output_dir,
            status="failed",
            experiment=args.experiment,
            algorithm=algorithm,
            message=reason,
            failure_code="data_missing",
            error_type="ConfigurationError",
            error_message=reason,
            completed=True,
        )
        _write_training_summary(output_dir, status_payload=status_payload, plan=train_plan)
        register_artifact(path, "train_baseline", f"modeling/{args.experiment}/train_metrics.json")
        register_artifact(path, "train_baseline", f"modeling/{args.experiment}/training_status.json")
        register_artifact(path, "train_baseline", f"modeling/{args.experiment}/training_summary.md")
        stage_action_failed(path, "train_baseline", reason)
        print(f"train failed: {reason}", file=sys.stderr)
        return 1

    if input_feather and Path(input_feather).is_file() and feature_list.is_file() and train_config:
        try:
            _write_training_status(
                output_dir,
                status="running",
                experiment=args.experiment,
                algorithm=algorithm,
                message="training started",
                percent=0,
            )
            train_progress = _TrainingStatusProgressReporter(
                reporter,
                output_dir,
                experiment=args.experiment,
                algorithm=algorithm,
            )
            if algorithm == "lightgbm":
                from risk_model_workbench.modeling.train_lgb import train_lightgbm_from_feather

                metrics = train_lightgbm_from_feather(
                    input_feather=input_feather,
                    feature_list_path=feature_list,
                    output_dir=output_dir,
                    score_output=score_output,
                    input_snapshot_dir=input_snapshot_dir,
                    config=effective_config,
                    progress=train_progress,
                )
            else:
                from risk_model_workbench.modeling.train_xgb import train_tabular_from_feather

                metrics = train_tabular_from_feather(
                    input_feather=input_feather,
                    feature_list_path=feature_list,
                    output_dir=output_dir,
                    score_output=score_output,
                    input_snapshot_dir=input_snapshot_dir,
                    config=effective_config,
                    algorithm=algorithm,
                    progress=train_progress,
                )
            _write_json(output_dir / "train_metrics.json", {"status": "done", "metrics": metrics, "experiment": args.experiment, "algorithm": algorithm})
            status_payload = _write_training_status(
                output_dir,
                status="done",
                experiment=args.experiment,
                algorithm=algorithm,
                message="training completed",
                percent=100,
                metrics=metrics,
                completed=True,
            )
            _write_training_summary(output_dir, status_payload=status_payload, plan=train_plan, metrics=metrics)
            for artifact in [
                "train_metrics.json",
                "training_status.json",
                "training_summary.md",
                "metrics_train_valid.json",
                "feature_importance.csv",
                "feature_drop_detail.csv",
                "actual_feature_list.txt",
                "preprocessing.json",
                "run_config.json",
                "model.pkl",
                "score_column_summary.csv",
                "distillation_summary.json",
                "tuning_context.json",
                "tuning_trials.csv",
                "tuning_summary.json",
                "best_params.json",
                "selection_reason.md",
                "llm_tuning_decisions.md",
            ]:
                if (output_dir / artifact).exists():
                    register_artifact(path, "train_baseline", f"modeling/{args.experiment}/{artifact}")
            for artifact in sorted(output_dir.glob("llm_tuning_plan_round_*.json")) + sorted(output_dir.glob("tuning_context_round_*.json")):
                register_artifact(path, "train_baseline", artifact)
            if score_output.exists():
                try:
                    register_artifact(path, "train_baseline", score_output)
                except Exception:
                    register_artifact(path, "train_baseline", f"modeling/{args.experiment}/scores_all_splits.feather")
            _register_woe_artifacts(path, "train_baseline", output_dir / "woe_top_features")
            append_decision(path, stage="train_baseline", decision="done", reason=f"{algorithm} training completed from local feather data")
            stage_action_done(path, "train_baseline")
            print(f"train complete: {output_dir}")
            return 0
        except Exception as exc:
            from risk_model_workbench.modeling.llm_tuning import HostAgentTuningPlanRequired

            if isinstance(exc, HostAgentTuningPlanRequired):
                payload = {
                    "status": "advisor_required",
                    "reason": str(exc),
                    "experiment": args.experiment,
                    "algorithm": algorithm,
                    "plan_path": exc.plan_path,
                    "context_path": exc.context_path,
                }
                _write_json(output_dir / "train_metrics.json", payload)
                status_payload = _write_training_status(
                    output_dir,
                    status="advisor_required",
                    experiment=args.experiment,
                    algorithm=algorithm,
                    message=str(exc),
                    failure_code="advisor_required",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    completed=True,
                )
                _write_training_summary(output_dir, status_payload=status_payload, plan=train_plan)
                register_artifact(path, "train_baseline", f"modeling/{args.experiment}/train_metrics.json")
                register_artifact(path, "train_baseline", f"modeling/{args.experiment}/training_status.json")
                register_artifact(path, "train_baseline", f"modeling/{args.experiment}/training_summary.md")
                for artifact in sorted(output_dir.glob("tuning_context*.json")):
                    register_artifact(path, "train_baseline", artifact)
                append_decision(path, stage="train_baseline", decision="advisor_required", reason=str(exc))
                stage_action_failed(path, "train_baseline", str(exc), failure_code="advisor_required")
                print(f"train advisor required: {exc}", file=sys.stderr)
                return 2
            reason = f"training failed: {exc}"
            payload = {"status": "failed", "reason": reason, "experiment": args.experiment, "algorithm": algorithm}
            _write_json(output_dir / "train_metrics.json", payload)
            failure_code = classify_exception(exc)
            status_payload = _write_training_status(
                output_dir,
                status="failed",
                experiment=args.experiment,
                algorithm=algorithm,
                message=reason,
                failure_code=failure_code,
                error_type=type(exc).__name__,
                error_message=str(exc),
                completed=True,
            )
            _write_training_summary(output_dir, status_payload=status_payload, plan=train_plan)
            for artifact in ["train_metrics.json", "training_status.json", "training_summary.md"]:
                register_artifact(path, "train_baseline", f"modeling/{args.experiment}/{artifact}")
            append_decision(path, stage="train_baseline", decision="failed", reason=reason)
            stage_action_failed(path, "train_baseline", reason, failure_code=failure_code)
            print(f"train failed: {reason}", file=sys.stderr)
            return 1
    else:
        payload = {
            "status": "scaffold",
            "reason": "training data not available",
            "experiment": args.experiment,
            "algorithm": algorithm,
            "input_feather": str(input_feather) if input_feather else "",
            "input_feather_exists": bool(input_feather and Path(input_feather).exists()),
            "feature_list": str(feature_list),
            "feature_list_exists": feature_list.exists(),
            "train_config_keys": sorted(train_config.keys()),
        }
    reporter.emit(step="train_scaffold", status="scaffold", message=f"模型训练未执行真实训练：{payload['reason']}", percent=100)
    status_payload = _write_training_status(
        output_dir,
        status="scaffold",
        experiment=args.experiment,
        algorithm=algorithm,
        message=payload["reason"],
        percent=100,
        failure_code="scaffold_only",
        completed=True,
    )
    _write_json(output_dir / "train_metrics.json", payload)
    _write_training_summary(output_dir, status_payload=status_payload, plan=train_plan)
    if feature_list.exists():
        _copy_if_exists(feature_list, output_dir / "feature_list.txt")
    register_artifact(path, "train_baseline", f"modeling/{args.experiment}/train_metrics.json", source="scaffold")
    register_artifact(path, "train_baseline", f"modeling/{args.experiment}/training_status.json", source="scaffold")
    register_artifact(path, "train_baseline", f"modeling/{args.experiment}/training_summary.md", source="scaffold")
    append_decision(path, stage="train_baseline", decision="scaffold", reason=payload["reason"])
    stage_action_done(path, "train_baseline", scaffold=True, message=payload["reason"])
    print(f"train scaffold: {output_dir / 'train_metrics.json'}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    path = _run_path(args)
    stage_action_started(path, "evaluate")
    reporter = ProgressReporter(path, "evaluate")
    evaluate_path = _runtime_config_path(path, project_dir, "evaluate")
    evaluate_config = load_yaml(evaluate_path) if evaluate_path.exists() else {}
    metrics = evaluate_config.get("metrics") or evaluate_config.get("evaluation", {}).get("metrics") or []
    scores_feather = _scores_feather_for_run(path, args.scores_feather)
    if not scores_feather.is_absolute():
        scores_feather = project_dir / scores_feather
    output_dir = Path(args.output_dir or path / "evaluation")
    if not output_dir.is_absolute():
        output_dir = project_dir / output_dir
    if scores_feather.exists() and evaluate_config:
        try:
            from risk_model_workbench.evaluation.run import evaluate_scores_from_feather

            summary = evaluate_scores_from_feather(scores_feather=scores_feather, output_dir=output_dir, config=evaluate_config, progress=reporter)
            for artifact in sorted([*output_dir.glob("*.csv"), *output_dir.glob("*.json")]):
                register_artifact(path, "evaluate", artifact)
            append_decision(path, stage="evaluate", decision="done", reason="Evaluation completed from local score feather")
            stage_action_done(path, "evaluate")
            print(f"evaluation complete: {output_dir / 'evaluation_summary.json'}")
            return 0
        except Exception as exc:
            payload = {"status": "scaffold", "reason": f"evaluation failed or dependency missing: {exc}", "configured_metrics": metrics}
    else:
        payload = {
            "status": "scaffold",
            "reason": "prediction data not available",
            "configured_metrics": metrics,
            "scores_feather": str(scores_feather),
            "scores_feather_exists": scores_feather.exists(),
        }
    reporter.emit(step="evaluate_scaffold", status="scaffold", message=f"模型评估未执行真实评估：{payload['reason']}", percent=100)
    _write_json(path / "evaluation" / "evaluation_summary.json", payload)
    register_artifact(path, "evaluate", "evaluation/evaluation_summary.json")
    append_decision(path, stage="evaluate", decision="scaffold", reason=payload["reason"])
    stage_action_done(path, "evaluate", scaffold=True, message=payload["reason"])
    print(f"evaluation scaffold: {path / 'evaluation' / 'evaluation_summary.json'}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    path = _run_path(args)
    stage_action_started(path, "compare")
    champions = _as_string_list(args.champion)
    if not champions:
        evaluate_config = _load_runtime_config(project_dir, path, "evaluate")
        champions = [
            score
            for score in _as_string_list((evaluate_config.get("evaluation") or {}).get("score_columns"))
            if score != "model_score"
        ]
    champions = list(dict.fromkeys(champions))
    if not champions:
        payload = {"status": "skipped", "reason": "no champion configured", "champion": "", "champions": []}
        _write_json(path / "evaluation" / "champion_challenger.json", payload)
        register_artifact(path, "compare", "evaluation/champion_challenger.json")
        append_decision(path, stage="compare", decision="skipped", reason=payload["reason"])
        stage_action_done(path, "compare", message=payload["reason"])
        print(f"compare skipped: {path / 'evaluation' / 'champion_challenger.json'}")
        return 0
    benchmark_path = path / "evaluation" / "benchmark_uplift.csv"
    if benchmark_path.exists():
        payload = {
            "status": "done",
            "reason": "",
            "champion": champions[-1],
            "champions": champions,
            "benchmark_uplift": str(benchmark_path.relative_to(path)),
        }
        register_artifact(path, "compare", "evaluation/benchmark_uplift.csv")
        decision = "done"
        scaffold = False
    else:
        payload = {
            "status": "scaffold",
            "reason": "candidate and champion predictions not available",
            "champion": champions[-1],
            "champions": champions,
        }
        decision = "scaffold"
        scaffold = True
    _write_json(path / "evaluation" / "champion_challenger.json", payload)
    register_artifact(path, "compare", "evaluation/champion_challenger.json")
    append_decision(path, stage="compare", decision=decision, reason=payload["reason"] or "Champion/challenger comparison materialized")
    stage_action_done(path, "compare", scaffold=scaffold, message=payload["reason"])
    print(f"compare {'scaffold' if scaffold else 'complete'}: {path / 'evaluation' / 'champion_challenger.json'}")
    return 0


def _as_config_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _report_root(report_config: dict[str, Any]) -> dict[str, Any]:
    return _as_config_dict(report_config.get("report"))


def _report_score_labels(report_config: dict[str, Any], target: dict[str, Any] | None = None) -> dict[str, str]:
    root = _report_root(report_config)
    labels: dict[str, str] = {}
    for source in [report_config, root, target or {}]:
        display_name = source.get("model_display_name") or source.get("model_name")
        if display_name:
            labels["model_score"] = str(display_name)
        raw_labels = source.get("score_labels") or {}
        if isinstance(raw_labels, dict):
            labels.update({str(key): str(value) for key, value in raw_labels.items() if str(value) != ""})
    return labels


def _report_score_columns(report_config: dict[str, Any], evaluate_config: dict[str, Any], target: dict[str, Any] | None = None) -> list[str]:
    root = _report_root(report_config)
    eval_root = _as_config_dict(evaluate_config.get("evaluation"))
    raw = (target or {}).get("score_columns") or report_config.get("score_columns") or root.get("score_columns") or eval_root.get("score_columns")
    return _as_string_list(raw) or ["model_score"]


def _report_outputs(report_config: dict[str, Any], target: dict[str, Any] | None = None) -> list[str]:
    root = _report_root(report_config)
    raw = (target or {}).get("outputs") or report_config.get("outputs") or root.get("outputs")
    return _as_string_list(raw) or ["model_report.md", "model_report.html", "model_card.md", "executive_summary.md"]


def _configured_report_targets(report_config: dict[str, Any], selected: str | None = None) -> list[dict[str, Any]]:
    root = _report_root(report_config)
    raw_targets = root.get("targets") or report_config.get("targets") or []
    if not isinstance(raw_targets, list):
        return []
    targets: list[dict[str, Any]] = []
    for index, item in enumerate(raw_targets, start=1):
        if not isinstance(item, dict):
            continue
        target = deepcopy(item)
        target_name = str(target.get("name") or target.get("experiment") or f"target_{index}").strip()
        if not target_name:
            continue
        if selected and target_name != selected:
            continue
        target["name"] = target_name
        if "score_labels" not in target:
            inherited_labels = _report_score_labels(report_config)
            if inherited_labels:
                target["score_labels"] = inherited_labels
        targets.append(target)
    return targets


def _resolve_workspace_path(workspace: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else workspace / path


def _report_generation_config(
    report_config: dict[str, Any],
    *,
    evaluate_config: dict[str, Any],
    target: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = deepcopy(report_config)
    root = deepcopy(_report_root(config))
    labels = _report_score_labels(config, target)
    score_columns = _report_score_columns(config, evaluate_config, target)
    outputs = _report_outputs(config, target)
    if target:
        for key in ["model_display_name", "model_name", "title"]:
            if target.get(key):
                root[key] = target[key]
                config[key] = target[key]
    if labels:
        root["score_labels"] = labels
        config["score_labels"] = labels
    if score_columns:
        root["score_columns"] = score_columns
        config["score_columns"] = score_columns
    if outputs:
        root["outputs"] = outputs
        config["outputs"] = outputs
    if root:
        config["report"] = root
    return config


def _target_train_dir(workspace: Path, target: dict[str, Any], fallback_train_dir: Path | None) -> Path:
    if target.get("train_dir"):
        return _resolve_workspace_path(workspace, target["train_dir"])
    experiment = str(target.get("experiment") or "").strip()
    if experiment:
        return workspace / "modeling" / experiment
    if fallback_train_dir is not None:
        return fallback_train_dir
    return workspace / "modeling" / "main_lgbm"


def _target_eval_dir(workspace: Path, target: dict[str, Any]) -> Path:
    if target.get("eval_dir"):
        return _resolve_workspace_path(workspace, target["eval_dir"])
    experiment = str(target.get("experiment") or "").strip()
    if experiment and (workspace / "evaluation_tuned" / experiment).exists():
        return workspace / "evaluation_tuned" / experiment
    return workspace / "evaluation"


def _target_output_dir(workspace: Path, target: dict[str, Any]) -> Path:
    if target.get("output_dir"):
        return _resolve_workspace_path(workspace, target["output_dir"])
    name = str(target.get("name") or "").strip()
    return workspace / (f"reports_{name}" if name and name != "default" else "reports")


def _target_report_scope(
    *,
    workspace: Path,
    target: dict[str, Any],
    train_dir: Path,
    eval_dir: Path,
    output_dir: Path,
    outputs: list[str],
) -> dict[str, Any]:
    def rel(path: Path) -> str:
        try:
            return path.relative_to(workspace).as_posix()
        except ValueError:
            return path.as_posix()

    return {
        "report_scope": target.get("description") or target.get("name") or "model report",
        "target": target.get("name"),
        "experiment": target.get("experiment"),
        "train_dir": rel(train_dir),
        "eval_dir": rel(eval_dir),
        "output_dir": rel(output_dir),
        "outputs": outputs,
        "score_labels": target.get("score_labels"),
    }


def _generate_excel_report_for_target(
    *,
    workspace: Path,
    project_dir: Path,
    target: dict[str, Any],
    report_config: dict[str, Any],
    evaluate_config: dict[str, Any],
    fallback_train_dir: Path | None,
) -> Path | None:
    from risk_model_workbench.reporting.excel_report import generate_excel_report

    train_dir = _target_train_dir(workspace, target, fallback_train_dir)
    eval_dir = _target_eval_dir(workspace, target)
    output_dir = _target_output_dir(workspace, target)
    input_dir = _resolve_workspace_path(workspace, target.get("input_dir") or "modeling_input")
    feature_dir = _resolve_workspace_path(workspace, target.get("feature_dir") or "feature_selection")
    if not (train_dir / "metrics_train_valid.json").exists():
        append_decision(workspace, stage="report", decision="report_target_skipped", reason=f"{target.get('name')}: missing train metrics at {train_dir}")
        return None
    if not (eval_dir / "evaluation_summary.json").exists():
        append_decision(workspace, stage="report", decision="report_target_skipped", reason=f"{target.get('name')}: missing evaluation summary at {eval_dir}")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_woe_artifacts(train_dir / "woe_top_features", output_dir / "woe_top_features")
    generation_config = _report_generation_config(report_config, evaluate_config=evaluate_config, target=target)
    excel_path = generate_excel_report(
        eval_dir=eval_dir,
        train_dir=train_dir,
        input_dir=input_dir,
        feature_dir=feature_dir,
        output_path=output_dir / "model_report.xlsx",
        project_dir=project_dir,
        report_config=generation_config,
    )
    outputs = [
        "model_report.html",
        "model_report.md",
        "model_report.xlsx",
        "model_report_missing_results.md",
    ]
    scope_path = _write_json(
        output_dir / "report_scope.json",
        _target_report_scope(
            workspace=workspace,
            target=target,
            train_dir=train_dir,
            eval_dir=eval_dir,
            output_dir=output_dir,
            outputs=outputs,
        ),
    )
    register_artifact(workspace, "report", scope_path)
    register_artifact(workspace, "report", excel_path)
    for sidecar in [
        excel_path.with_name("model_report.md"),
        excel_path.with_name("model_report.html"),
        excel_path.with_name("model_report_missing_results.md"),
    ]:
        if sidecar.exists():
            register_artifact(workspace, "report", sidecar)
    _register_woe_artifacts(workspace, "report", output_dir / "woe_top_features")
    append_decision(workspace, stage="report", decision="report_target_done", reason=f"{target.get('name')}: generated {excel_path}")
    return excel_path


def cmd_report(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    path = _run_path(args)
    stage_action_started(path, "report")
    report_path = _runtime_config_path(path, project_dir, "report")
    report_config = load_yaml(report_path) if report_path.exists() else {}
    evaluate_path = _runtime_config_path(path, project_dir, "evaluate")
    evaluate_config = load_yaml(evaluate_path) if evaluate_path.exists() else {}
    selected_target = getattr(args, "report_target", None)
    sections = report_config.get("sections") or report_config.get("report", {}).get("sections") or []
    outputs = report_config.get("outputs") or report_config.get("report", {}).get("outputs") or ["model_report.md", "model_report.html", "model_card.md", "executive_summary.md"]
    report_steps = _as_string_list((report_config.get("report") or {}).get("stage_steps"))
    outputs = _as_string_list(outputs)
    if "model_recovery_report" in report_steps and "model_recovery_report.md" not in outputs:
        outputs.append("model_recovery_report.md")
    if "credit_product_report" in report_steps and "credit_product_report.md" not in outputs:
        outputs.append("credit_product_report.md")
    state = load_run_state(path)
    manifest_path = path / "audit" / "artifact_manifest.json"
    text = (
        "# Model Report\n\n"
        "status: scaffold\n\n"
        "This report is generated only from registered run artifacts. Missing metrics are not fabricated.\n\n"
        f"- run_id: {args.run_id}\n"
        f"- workflow: {state.get('workflow')}\n"
        f"- configured_sections: {', '.join(sections) if sections else 'not configured'}\n"
        f"- artifact_manifest: {manifest_path.relative_to(path)}\n"
    )
    generated_report_paths: list[Path] = []

    def _report_body(name: str) -> str:
        lowered = name.lower()
        if "model_card" in lowered:
            return "# Model Card\n\nstatus: scaffold\n\nThis card is generated from registered run artifacts.\n"
        if "executive" in lowered:
            return "# Executive Summary\n\nstatus: scaffold\n\nModel evaluation evidence is summarized from the run manifest when available.\n"
        if "recovery" in lowered:
            return "# Model Recovery Report\n\nstatus: scaffold\n\nRecovery monitoring inputs were requested; missing artifacts are listed in the run manifest.\n"
        if "credit" in lowered:
            return "# Credit Product Report\n\nstatus: scaffold\n\nCredit product evaluation outputs were requested; missing artifacts are listed in the run manifest.\n"
        return text

    if not selected_target:
        for output_name in outputs:
            target = path / "reports" / Path(output_name).name
            suffix = target.suffix.lower()
            if suffix == ".xlsx":
                continue
            if suffix == ".html":
                from risk_model_workbench.reporting.html_report import render_model_report_html

                body = _report_body(target.name)
                _write_text(target, render_model_report_html(body, title="Model Report", run_id=args.run_id))
            elif suffix == ".json":
                _write_json(target, {"status": "scaffold", "run_id": args.run_id, "sections": sections, "artifact_manifest": str(manifest_path.relative_to(path))})
            else:
                _write_text(target, _report_body(target.name))
            generated_report_paths.append(target)

        for required_name in ["model_report.md", "model_report.html", "model_card.md", "executive_summary.md"]:
            target = path / "reports" / required_name
            if not target.exists():
                body = _report_body(required_name)
                if target.suffix.lower() == ".html":
                    from risk_model_workbench.reporting.html_report import render_model_report_html

                    _write_text(target, render_model_report_html(body, title="Model Report", run_id=args.run_id))
                else:
                    _write_text(target, body)
                generated_report_paths.append(target)
    for artifact_path in generated_report_paths:
        register_artifact(path, "report", artifact_path)
    train_dirs = [item for item in (path / "modeling").glob("*") if item.is_dir() and (item / "metrics_train_valid.json").exists()]
    all_configured_targets = _configured_report_targets(report_config)
    targets = _configured_report_targets(report_config, selected_target)
    if selected_target and all_configured_targets and not targets:
        stage_action_failed(path, "report", f"unknown report target: {selected_target}")
        print(f"unknown report target: {selected_target}", file=sys.stderr)
        return 1
    if not targets:
        targets = [{"name": selected_target or "default", "output_dir": "reports"}]

    excel_paths: list[Path] = []
    fallback_train_dir = sorted(train_dirs)[0] if train_dirs else None
    for target in targets:
        try:
            excel_path = _generate_excel_report_for_target(
                workspace=path,
                project_dir=project_dir,
                target=target,
                report_config=report_config,
                evaluate_config=evaluate_config,
                fallback_train_dir=fallback_train_dir,
            )
            if excel_path is not None:
                excel_paths.append(excel_path)
        except Exception as exc:
            append_decision(path, stage="report", decision="excel_scaffold", reason=f"{target.get('name')}: Excel report not generated: {exc}")
    if excel_paths:
        append_decision(path, stage="report", decision="done", reason=f"Excel report generated for {len(excel_paths)} report target(s)")
        stage_action_done(path, "report")
        print("report complete: " + ", ".join(str(item) for item in excel_paths))
    else:
        append_decision(path, stage="report", decision="scaffold", reason="report generated with missing real evaluation artifacts")
        stage_action_done(path, "report", scaffold=True, message="report generated with missing real evaluation artifacts")
        print(f"report scaffold: {path / 'reports' / 'model_report.md'}")
    return 0


def cmd_feature_screening_summary(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    output_path = write_feature_screening_summary(project_dir, args.output)
    print(f"summary: {output_path}")
    return 0


def cmd_import_gcard(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    run_id = args.run_id
    path = run_dir(project_dir, run_id)
    for directory in ["audit", "feature_selection"]:
        (path / directory).mkdir(parents=True, exist_ok=True)
    state = create_run_state(project_dir, run_id=run_id, workflow="imported_feature_screening", status="imported")
    save_run_state(path, state)
    _write_json(path / "audit" / "artifact_manifest.json", {"version": 1, "artifacts": []})
    _write_text(path / "audit" / "command_log.jsonl", "")
    _write_text(path / "audit" / "decision_log.md", "# Decision Log\n")

    copies = [
        (project_dir / "reports" / "feature_screening_process.json", path / "feature_selection" / "feature_screening_process.json"),
        (project_dir / "runs" / "feature_refine_feather" / "final_500_features.txt", path / "feature_selection" / "final_features.txt"),
    ]
    for source, target in copies:
        copied = _copy_if_exists(source, target)
        if copied:
            register_artifact(path, "feature_refine", copied.relative_to(path), source="imported")

    for source in [
        project_dir / "reports" / "feature_screening_process.xlsx",
        project_dir / "runs" / "feature_refine_feather",
        project_dir / "runs" / "modeling_feature_set",
    ]:
        register_artifact(path, "feature_refine", source, kind="directory" if source.is_dir() else "file", source="imported")
    append_decision(path, stage="feature_refine", decision="imported", reason="本 run 是从历史产物导入，不是重新执行得到。")
    mark_stage_done(path, "feature_refine")
    print(f"imported run: {path}")
    return 0


def cmd_import_gcard_model(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    run_id = args.run_id
    path = run_dir(project_dir, run_id)
    for directory in ["audit", "configs_snapshot", "sample_check", "feature_selection", "modeling/main_lgbm", "modeling_input", "evaluation", "reports"]:
        (path / directory).mkdir(parents=True, exist_ok=True)
    state = create_run_state(project_dir, run_id=run_id, workflow="imported_gcard_main_lgbm", status="imported")
    save_run_state(path, state)
    _write_json(path / "audit" / "artifact_manifest.json", {"version": 1, "artifacts": []})
    _write_text(path / "audit" / "command_log.jsonl", "")
    _write_text(path / "audit" / "decision_log.md", "# Decision Log\n")

    copy_groups = [
        (project_dir / "configs" / "train.yaml", path / "configs_snapshot" / "train.yaml", "validate_config"),
        (project_dir / "runs" / "modeling_input" / "sample_split_summary.csv", path / "sample_check" / "sample_split_summary.csv", "sample_check"),
        (project_dir / "runs" / "modeling_input" / "label_distribution.csv", path / "sample_check" / "label_distribution.csv", "sample_check"),
        (project_dir / "runs" / "modeling_input" / "segment_distribution.csv", path / "sample_check" / "segment_distribution.csv", "sample_check"),
        (project_dir / "runs" / "modeling_feature_set" / "feature_list.txt", path / "feature_selection" / "final_features.txt", "feature_refine"),
        (project_dir / "runs" / "modeling_feature_set" / "feature_availability.csv", path / "feature_selection" / "feature_availability.csv", "feature_refine"),
        (project_dir / "runs" / "modeling_feature_set" / "feature_stage_summary.json", path / "feature_selection" / "feature_stage_summary.json", "feature_refine"),
        (project_dir / "runs" / "modeling_input" / "input_config.json", path / "modeling_input" / "input_config.json", "train_baseline"),
        (project_dir / "runs" / "modeling_input" / "input_schema.csv", path / "modeling_input" / "input_schema.csv", "train_baseline"),
        (project_dir / "runs" / "model_train" / "main_lgbm" / "model.pkl", path / "modeling" / "main_lgbm" / "model.pkl", "train_baseline"),
        (project_dir / "runs" / "model_train" / "main_lgbm" / "metrics_train_valid.json", path / "modeling" / "main_lgbm" / "metrics_train_valid.json", "train_baseline"),
        (project_dir / "runs" / "model_train" / "main_lgbm" / "feature_importance.csv", path / "modeling" / "main_lgbm" / "feature_importance.csv", "train_baseline"),
        (project_dir / "runs" / "model_train" / "main_lgbm" / "feature_drop_detail.csv", path / "modeling" / "main_lgbm" / "feature_drop_detail.csv", "train_baseline"),
        (project_dir / "runs" / "model_train" / "main_lgbm" / "actual_feature_list.txt", path / "modeling" / "main_lgbm" / "actual_feature_list.txt", "train_baseline"),
        (project_dir / "runs" / "model_train" / "main_lgbm" / "preprocessing.json", path / "modeling" / "main_lgbm" / "preprocessing.json", "train_baseline"),
        (project_dir / "runs" / "model_train" / "main_lgbm" / "run_config.json", path / "modeling" / "main_lgbm" / "run_config.json", "train_baseline"),
        (project_dir / "runs" / "model_eval" / "evaluation_summary.json", path / "evaluation" / "evaluation_summary.json", "evaluate"),
        (project_dir / "runs" / "model_eval" / "overall_metrics.csv", path / "evaluation" / "overall_metrics.csv", "evaluate"),
        (project_dir / "runs" / "model_eval" / "monthly_metrics.csv", path / "evaluation" / "monthly_metrics.csv", "evaluate"),
        (project_dir / "runs" / "model_eval" / "segment_metrics.csv", path / "evaluation" / "segment_metrics.csv", "evaluate"),
        (project_dir / "runs" / "model_eval" / "benchmark_uplift.csv", path / "evaluation" / "benchmark_uplift.csv", "compare"),
        (project_dir / "reports" / "model_report.xlsx", path / "reports" / "model_report.xlsx", "report"),
    ]
    for source, target, stage in copy_groups:
        copied = _copy_if_exists(source, target)
        if copied:
            register_artifact(path, stage, copied.relative_to(path), source="imported")

    eval_dir = project_dir / "runs" / "model_eval"
    for source in sorted(eval_dir.glob("decile_lift*.csv")) + sorted(eval_dir.glob("intent_zc*.csv")) + sorted(eval_dir.glob("score_psi_by_month.csv")):
        target = path / "evaluation" / source.name
        copied = _copy_if_exists(source, target)
        if copied:
            register_artifact(path, "evaluate", copied.relative_to(path), source="imported")

    for stage in ["validate_config", "sample_check", "feature_refine", "train_baseline", "evaluate", "compare", "report"]:
        mark_stage_done(path, stage)
    append_decision(path, stage="train_baseline", decision="imported", reason="本 run 从远端真实复借 G 卡 main_lgbm 训练、评估和报告产物导入，不是当前环境重新执行。")
    print(f"imported model run: {path}")
    return 0


def _add_project_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    project = subparsers.add_parser("project", help="project workspace commands")
    project_sub = project.add_subparsers(dest="project_command", required=True)
    validate = project_sub.add_parser("validate", help="validate project config")
    validate.add_argument("--project", required=True)
    validate.set_defaults(func=cmd_project_validate)
    status = project_sub.add_parser("status", help="show project continuity state")
    status.add_argument("--project", required=True)
    status.add_argument("--version-id", default=None)
    status.add_argument("--run-id", default=None)
    status.add_argument("--write-state", action="store_true", help="write or refresh project_state.yml")
    status.set_defaults(func=cmd_project_status)
    update = project_sub.add_parser("update-state", help="update project_state.yml with handoff metadata")
    update.add_argument("--project", required=True)
    update.add_argument("--active-version-id", default=None)
    update.add_argument("--active-run-id", default=None)
    update.add_argument("--objective", default=None)
    update.add_argument("--status", default=None)
    update.add_argument("--next-action", action="append")
    update.add_argument("--blocker", action="append")
    update.add_argument("--risk", action="append")
    update.set_defaults(func=cmd_project_update_state)


def _add_handoff_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    handoff = subparsers.add_parser("handoff", help="session handoff commands")
    handoff_sub = handoff.add_subparsers(dest="handoff_command", required=True)
    write = handoff_sub.add_parser("write", help="write a resumable project handoff")
    write.add_argument("--project", required=True)
    write.add_argument("--run-id", default=None)
    write.add_argument("--note", default="")
    write.add_argument("--output", default=None)
    write.add_argument(
        "--context-snapshot",
        nargs="?",
        const="auto",
        default=None,
        help="include a context snapshot reference; without a value uses the run default path",
    )
    write.set_defaults(func=cmd_handoff_write)


def _add_lesson_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    lesson = subparsers.add_parser("lesson", help="project and workbench lesson commands")
    lesson_sub = lesson.add_subparsers(dest="lesson_command", required=True)
    add = lesson_sub.add_parser("add", help="append a lesson learned")
    add.add_argument("--project", required=True)
    add.add_argument("--title", required=True)
    add.add_argument("--body", default=None)
    add.add_argument("--body-file", default=None)
    add.add_argument("--kind", choices=["pitfall", "method", "guardrail", "decision"], default="method")
    add.add_argument("--scope", choices=["project", "workbench"], default="project")
    add.add_argument("--source", default="")
    add.add_argument("--tag", action="append")
    add.set_defaults(func=cmd_lesson_add)
    promote = lesson_sub.add_parser("promote", help="promote a project lesson into the workbench rule registry")
    promote.add_argument("--project", required=True)
    promote.add_argument("--title", required=True)
    promote.add_argument("--target", choices=["guardrail", "test", "skill", "adr", "glossary"], required=True)
    promote.add_argument("--rule-id", required=True)
    promote.add_argument("--note", default="")
    promote.set_defaults(func=cmd_lesson_promote)


def _add_rules_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    rules = subparsers.add_parser("rules", help="workbench rule registry commands")
    rules_sub = rules.add_subparsers(dest="rules_command", required=True)
    list_cmd = rules_sub.add_parser("list", help="list workbench rules")
    list_cmd.add_argument("--json", action="store_true")
    list_cmd.set_defaults(func=cmd_rules_list)


def _add_retrospective_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    retrospective = subparsers.add_parser("retrospective", help="explicit retrospective checkpoint commands")
    retrospective_sub = retrospective.add_subparsers(dest="retrospective_command", required=True)
    write = retrospective_sub.add_parser("write", help="write a session, stage, or project retrospective")
    write.add_argument("--project", required=True)
    write.add_argument("--run-id", default=None)
    write.add_argument("--scope", choices=["session", "stage", "project"], default="session")
    write.add_argument("--stage", default=None)
    write.add_argument("--outcome", default="")
    write.add_argument("--note", default="")
    write.add_argument("--lesson", action="append")
    write.add_argument("--output", default=None)
    write.set_defaults(func=cmd_retrospective_write)


def _add_workflow_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    workflow = subparsers.add_parser("workflow", help="workflow commands")
    workflow_sub = workflow.add_subparsers(dest="workflow_command", required=True)
    workflow_sub.add_parser("list", help="list workflows").set_defaults(func=cmd_workflow_list)
    show = workflow_sub.add_parser("show", help="show workflow YAML")
    show.add_argument("--workflow", required=True)
    show.set_defaults(func=cmd_workflow_show)
    validate = workflow_sub.add_parser("validate", help="validate workflow YAML")
    validate.add_argument("--workflow", required=True)
    validate.set_defaults(func=cmd_workflow_validate)


def _add_workspace_id_args(parser: argparse.ArgumentParser, *, require_run_legacy: bool = False) -> None:
    parser.add_argument("--version-id", default=None, help="version workspace id")
    parser.add_argument("--run-id", required=require_run_legacy, default=None, help="legacy run id")


def _add_version_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    version = subparsers.add_parser("version", help="version lifecycle commands")
    version_sub = version.add_subparsers(dest="version_command", required=True)

    init = version_sub.add_parser("init", help="initialize a workflow version")
    init.add_argument("--project", required=True)
    init.add_argument("--workflow", required=True)
    init.add_argument("--version-id", required=True)
    init.add_argument("--display-name", default=None)
    init.add_argument("--source-type", default="workbench", choices=["workbench", "imported", "manual"])
    init.add_argument("--request", default=None, help="optional model request Markdown copied into the version")
    init.add_argument("--plan", default=None, help="optional execution plan YAML copied into the version")
    init.add_argument("--force", action="store_true")
    init.add_argument("--skip-split-check", action="store_true", help="跳过 splits 一致性校验(时间外样本不得进验证集)——仅紧急/探索场景")
    init.set_defaults(func=cmd_version_init)

    list_cmd = version_sub.add_parser("list", help="list project versions")
    list_cmd.add_argument("--project", required=True)
    list_cmd.add_argument("--json", action="store_true")
    list_cmd.set_defaults(func=cmd_version_list)

    show = version_sub.add_parser("show", help="show version state")
    show.add_argument("--project", required=True)
    show.add_argument("--version-id", required=True)
    show.add_argument("--json", action="store_true")
    show.set_defaults(func=cmd_version_show)

    status = version_sub.add_parser("status", help="show version state")
    status.add_argument("--project", required=True)
    status.add_argument("--version-id", required=True)
    status.add_argument("--progress", action="store_true", help="show Chinese progress summary and recent events")
    status.add_argument("--tail", type=int, default=5, help="number of recent progress events to show")
    status.set_defaults(func=cmd_version_status)

    watch = version_sub.add_parser("watch", help="watch Chinese version progress")
    watch.add_argument("--project", required=True)
    watch.add_argument("--version-id", required=True)
    watch.add_argument("--interval", type=float, default=10.0, help="poll interval in seconds")
    watch.add_argument("--tail", type=int, default=8, help="number of recent progress events to show")
    watch.add_argument("--once", action="store_true", help="render once and exit")
    watch.set_defaults(func=cmd_run_watch)

    audit = version_sub.add_parser("audit", help="audit version or stage closure readiness")
    audit.add_argument("--project", required=True)
    audit.add_argument("--version-id", required=True)
    audit.add_argument("--stage", default=None)
    audit.add_argument("--strict", action="store_true", help="return non-zero unless the audit verdict is complete")
    audit.add_argument("--json", action="store_true", help="emit machine-readable audit JSON")
    audit.set_defaults(func=cmd_version_audit)

    migrate_run = version_sub.add_parser("migrate-run", help="migrate one legacy standard run into a version")
    migrate_run.add_argument("--project", required=True)
    migrate_run.add_argument("--run-id", required=True)
    migrate_run.add_argument("--version-id", default=None)
    migrate_run.add_argument("--source-type", default=None, choices=["workbench", "imported", "manual"])
    migrate_run.add_argument("--display-name", default=None)
    migrate_run.add_argument("--force", action="store_true")
    migrate_run.set_defaults(func=cmd_version_migrate_run)

    migrate_all = version_sub.add_parser("migrate-standard-runs", help="migrate all legacy standard runs into versions")
    migrate_all.add_argument("--project", required=True)
    migrate_all.add_argument("--mapping", default=None)
    migrate_all.add_argument("--active-version-id", default=None)
    migrate_all.add_argument("--force", action="store_true")
    migrate_all.set_defaults(func=cmd_version_migrate_standard_runs)

    manual = version_sub.add_parser("register-manual", help="register a manual historical model version")
    manual.add_argument("--project", required=True)
    manual.add_argument("--version-id", required=True)
    manual.add_argument("--display-name", default=None)
    manual.add_argument("--workflow", default="manual")
    manual.add_argument("--status", default="imported", choices=["draft", "running", "candidate", "done", "released", "failed", "imported", "archived"])
    manual.add_argument("--active", action="store_true")
    manual.add_argument("--force", action="store_true")
    manual.set_defaults(func=cmd_version_register_manual)


def _add_run_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    run = subparsers.add_parser("run", help="run lifecycle commands")
    run_sub = run.add_subparsers(dest="run_command", required=True)
    init = run_sub.add_parser("init", help="initialize a workflow run")
    init.add_argument("--project", required=True)
    init.add_argument("--workflow", required=True)
    init.add_argument("--run-id", default=None)
    init.add_argument("--request", default=None, help="optional model request Markdown copied into the run")
    init.add_argument("--plan", default=None, help="optional execution plan YAML copied into the run")
    init.add_argument("--force", action="store_true")
    init.add_argument("--skip-split-check", action="store_true", help="跳过 splits 一致性校验(时间外样本不得进验证集)——仅紧急/探索场景")
    init.set_defaults(func=cmd_run_init)
    imported = run_sub.add_parser("import-gcard-artifacts", help="legacy/example: import existing Fujie GCard artifacts")
    imported.add_argument("--project", default="projects/2026-05-fujie-gcard-v1")
    imported.add_argument("--run-id", default="2026-05-imported-feature-screening")
    imported.set_defaults(func=cmd_import_gcard)
    imported_model = run_sub.add_parser(
        "import-gcard-model-artifacts",
        help="legacy/example: import existing Fujie GCard training/evaluation/report artifacts",
    )
    imported_model.add_argument("--project", default="projects/2026-05-fujie-gcard-v1")
    imported_model.add_argument("--run-id", default="2026-06-imported-gcard-main-lgbm")
    imported_model.set_defaults(func=cmd_import_gcard_model)
    status = run_sub.add_parser("status", help="show run state")
    status.add_argument("--project", required=True)
    status.add_argument("--run-id", required=True)
    status.add_argument("--progress", action="store_true", help="show Chinese progress summary and recent events")
    status.add_argument("--tail", type=int, default=5, help="number of recent progress events to show")
    status.set_defaults(func=cmd_status)
    watch = run_sub.add_parser("watch", help="watch Chinese run progress")
    watch.add_argument("--project", required=True)
    watch.add_argument("--run-id", required=True)
    watch.add_argument("--interval", type=float, default=10.0, help="poll interval in seconds")
    watch.add_argument("--tail", type=int, default=8, help="number of recent progress events to show")
    watch.add_argument("--once", action="store_true", help="render once and exit")
    watch.set_defaults(func=cmd_run_watch)
    audit = run_sub.add_parser("audit", help="audit run or stage closure readiness")
    audit.add_argument("--project", required=True)
    audit.add_argument("--run-id", required=True)
    audit.add_argument("--stage", default=None)
    audit.add_argument("--strict", action="store_true", help="return non-zero unless the audit verdict is complete")
    audit.add_argument("--json", action="store_true", help="emit machine-readable audit JSON")
    audit.set_defaults(func=cmd_run_audit)


def _add_request_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    request = subparsers.add_parser("request", help="model request commands")
    request_sub = request.add_subparsers(dest="request_command", required=True)
    validate = request_sub.add_parser("validate", help="validate a model request Markdown file")
    validate.add_argument("--request", required=True)
    validate.add_argument("--project", default=None)
    validate.add_argument("--skip-split-check", action="store_true", help="跳过 splits 一致性校验(时间外样本不得进验证集)——仅紧急/探索场景")
    validate.set_defaults(func=cmd_request_validate)


def _add_plan_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    plan = subparsers.add_parser("plan", help="execution plan commands")
    plan_sub = plan.add_subparsers(dest="plan_command", required=True)
    create = plan_sub.add_parser("create", help="create an execution plan from a model request")
    create.add_argument("--project", required=True)
    create.add_argument("--request", required=True)
    create.add_argument("--output", default=None)
    create.add_argument("--skip-split-check", action="store_true", help="跳过 splits 一致性校验(时间外样本不得进验证集)——仅紧急/探索场景")
    create.set_defaults(func=cmd_plan_create)


def _add_agent_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    agent = subparsers.add_parser("agent", help="RMW Agent commands")
    agent_sub = agent.add_subparsers(dest="agent_command", required=True)

    start = agent_sub.add_parser("start", help="initialize an Agent-managed modeling version")
    start.add_argument("--project", required=True)
    start.add_argument("--request", required=True)
    start.add_argument("--version-id", required=True)
    start.add_argument("--workflow", default="full_modeling")
    start.add_argument("--display-name", default=None)
    start.add_argument("--execute", action="store_true")
    start.set_defaults(func=cmd_agent_start)

    run = agent_sub.add_parser("run", help="run an initialized Agent plan")
    run.add_argument("--project", required=True)
    run.add_argument("--version-id", required=True)
    run.set_defaults(func=cmd_agent_run)

    resume = agent_sub.add_parser("resume", help="resume an Agent plan after approval or advisor input")
    resume.add_argument("--project", required=True)
    resume.add_argument("--version-id", required=True)
    resume.set_defaults(func=cmd_agent_resume)

    status = agent_sub.add_parser("status", help="show Agent status")
    status.add_argument("--project", required=True)
    status.add_argument("--version-id", required=True)
    status.add_argument("--json", action="store_true")
    status.add_argument("--tail", type=int, default=20)
    status.set_defaults(func=cmd_agent_status)

    approve = agent_sub.add_parser("approve", help="record approval for a blocked high-risk Agent action")
    approve.add_argument("--project", required=True)
    approve.add_argument("--version-id", required=True)
    approve.add_argument("--approval-id", required=True)
    approve.add_argument("--approved-by", required=True)
    approve.add_argument("--note", default="")
    approve.set_defaults(func=cmd_agent_approve)

    reject = agent_sub.add_parser("reject", help="reject a blocked high-risk Agent action")
    reject.add_argument("--project", required=True)
    reject.add_argument("--version-id", required=True)
    reject.add_argument("--approval-id", required=True)
    reject.add_argument("--rejected-by", required=True)
    reject.add_argument("--note", default="")
    reject.set_defaults(func=cmd_agent_reject)

    tools = agent_sub.add_parser("tools", help="export Agent tool schema")
    tools.add_argument("--json", action="store_true")
    tools.set_defaults(func=cmd_agent_tools)

    agent_plan = agent_sub.add_parser("plan", help="inspect or safely rebind an Agent plan")
    agent_plan_sub = agent_plan.add_subparsers(dest="agent_plan_command", required=True)
    rebind = agent_plan_sub.add_parser("rebind", help="preview or apply current registry metadata")
    rebind.add_argument("--project", required=True)
    rebind.add_argument("--version-id", required=True)
    mode = rebind.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="preview only (default)")
    mode.add_argument("--apply", action="store_true", help="write the rebound plan")
    rebind.add_argument("--json", action="store_true")
    rebind.set_defaults(func=cmd_agent_plan_rebind)

    advisor = agent_sub.add_parser("advisor", help="Advisor request/response protocol commands")
    advisor_sub = advisor.add_subparsers(dest="advisor_command", required=True)

    advisor_list = advisor_sub.add_parser("list", help="list Advisor requests")
    advisor_list.add_argument("--project", required=True)
    advisor_list.add_argument("--version-id", required=True)
    advisor_list.add_argument("--json", action="store_true")
    advisor_list.set_defaults(func=cmd_agent_advisor_list)

    advisor_show = advisor_sub.add_parser("show", help="show one Advisor request")
    advisor_show.add_argument("--project", required=True)
    advisor_show.add_argument("--version-id", required=True)
    advisor_show.add_argument("--request-id", required=True)
    advisor_show.add_argument("--json", action="store_true")
    advisor_show.set_defaults(func=cmd_agent_advisor_show)

    advisor_accept = advisor_sub.add_parser("accept", help="accept and validate an Advisor response")
    advisor_accept.add_argument("--project", required=True)
    advisor_accept.add_argument("--version-id", required=True)
    advisor_accept.add_argument("--response", required=True)
    advisor_accept.set_defaults(func=cmd_agent_advisor_accept)


def _add_feature_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    feature = subparsers.add_parser("feature", help="feature selection commands")
    feature_sub = feature.add_subparsers(dest="feature_command", required=True)
    metadata = feature_sub.add_parser("metadata")
    metadata.add_argument("--project", required=True)
    _add_workspace_id_args(metadata)
    metadata.add_argument("--tables-file", default=None)
    metadata.add_argument("--config", default=None)
    metadata.set_defaults(func=cmd_feature_metadata)

    prescreen = feature_sub.add_parser("prescreen")
    _add_feature_prescreen_args(prescreen)
    prescreen.set_defaults(func=cmd_feature_prescreen)

    refine = feature_sub.add_parser("refine")
    refine.add_argument("--project", required=True)
    _add_workspace_id_args(refine)
    refine.add_argument("--config", default=None)
    refine.add_argument("--dry-run-sql", action="store_true")
    refine.add_argument("--refresh-dp-cache", action="store_true")
    refine.add_argument("--sql-approved", action="store_true")
    refine.add_argument("--sample-max-rows", type=int, default=None)
    refine.set_defaults(func=cmd_feature_refine)


def _add_feature_prescreen_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project", required=True)
    _add_workspace_id_args(parser)
    parser.add_argument("--config", default=None)
    parser.add_argument("--table", action="append")
    parser.add_argument("--max-tables", type=int, default=None)
    parser.add_argument("--dry-run-sql", action="store_true")
    parser.add_argument("--refresh-dp-cache", action="store_true")
    parser.add_argument("--sql-approved", action="store_true")
    parser.add_argument("--force", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="风险场景 AI 建模工作台 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor", help="check local scaffold and dependencies").set_defaults(func=cmd_doctor)
    _add_project_parser(subparsers)
    _add_handoff_parser(subparsers)
    _add_lesson_parser(subparsers)
    _add_rules_parser(subparsers)
    add_metadata_parsers(subparsers)
    _add_retrospective_parser(subparsers)
    _add_workflow_parser(subparsers)
    _add_version_parser(subparsers)
    _add_run_parser(subparsers)
    _add_request_parser(subparsers)
    _add_plan_parser(subparsers)
    _add_agent_parser(subparsers)
    _add_feature_parser(subparsers)

    status = subparsers.add_parser("status", help="show run state")
    status.add_argument("--project", required=True)
    _add_workspace_id_args(status)
    status.add_argument("--progress", action="store_true", help="show Chinese progress summary and recent events")
    status.add_argument("--tail", type=int, default=5, help="number of recent progress events to show")
    status.set_defaults(func=cmd_status)

    sample = subparsers.add_parser("sample", help="sample commands")
    sample_sub = sample.add_subparsers(dest="sample_command", required=True)
    check = sample_sub.add_parser("check")
    check.add_argument("--project", required=True)
    _add_workspace_id_args(check)
    check.set_defaults(func=cmd_sample_check)

    train = subparsers.add_parser("train")
    train.add_argument("--project", required=True)
    _add_workspace_id_args(train)
    train.add_argument("--experiment", required=True)
    train.add_argument("--input-feather", default=None)
    train.add_argument("--feature-list", default=None)
    train.add_argument("--score-output", default=None)
    train.add_argument("--input-dir", default=None)
    train.add_argument("--config", default=None)
    train.add_argument("--plan-only", action="store_true", help="只生成训练计划预览，不启动模型训练")
    train.add_argument("--skip-split-check", action="store_true", help="跳过 valid_values 时间外污染校验——仅紧急/探索场景")
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--project", required=True)
    _add_workspace_id_args(evaluate)
    evaluate.add_argument("--scores-feather", default=None)
    evaluate.add_argument("--output-dir", default=None)
    evaluate.set_defaults(func=cmd_evaluate)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--project", required=True)
    _add_workspace_id_args(compare)
    compare.add_argument("--champion", action="append", default=[])
    compare.set_defaults(func=cmd_compare)

    report = subparsers.add_parser("report")
    report.add_argument("--project", required=True)
    _add_workspace_id_args(report)
    report.add_argument("--report-target", default=None, help="optional report target name from report.targets")
    report.set_defaults(func=cmd_report)

    init_project = subparsers.add_parser("init-project", help="create a model project workspace")
    init_project.add_argument("--name", required=True)
    init_project.add_argument("--display-name", required=True)
    init_project.add_argument("--scenario", required=True)
    init_project.add_argument("--template", default="generic", choices=["generic", "fujie-gcard"])
    init_project.add_argument("--force", action="store_true")
    init_project.set_defaults(func=cmd_init_project)

    new_run = subparsers.add_parser("new-run", help="legacy alias for run init")
    new_run.add_argument("--project", required=True)
    new_run.add_argument("--step", default="legacy")
    new_run.add_argument("--note", default="")
    new_run.set_defaults(func=lambda args: cmd_run_init(argparse.Namespace(project=args.project, workflow="full_modeling", run_id=None, request=None, plan=None, force=False, skip_split_check=False)))

    screening_summary = subparsers.add_parser("feature-screening-summary")
    screening_summary.add_argument("--project", required=True)
    screening_summary.add_argument("--output", default="reports/feature_screening_process.json")
    screening_summary.set_defaults(func=cmd_feature_screening_summary)

    build_wide_sql = subparsers.add_parser("build-wide-sql")
    build_wide_sql.add_argument("--project", required=True)
    build_wide_sql.add_argument("--version-id", default=None, help="optional version id for progress tracking")
    build_wide_sql.add_argument("--run-id", default=None, help="optional run id for progress tracking")
    build_wide_sql.add_argument("--remain-features", default=DEFAULT_PRESCREEN_REMAIN_FEATURES)
    build_wide_sql.add_argument("--sql-output", default=DEFAULT_WIDE_SQL_OUTPUT)
    build_wide_sql.add_argument("--feature-map-output", default=DEFAULT_WIDE_FEATURE_MAP_OUTPUT)
    build_wide_sql.add_argument("--summary-output", default=DEFAULT_WIDE_SUMMARY_OUTPUT)
    build_wide_sql.add_argument("--execution-output", default="feature_selection/wide_table_execution.json")
    build_wide_sql.add_argument("--base-table", default=None)
    build_wide_sql.add_argument("--output-table", default=None)
    build_wide_sql.add_argument("--base-where", default=None)
    build_wide_sql.add_argument("--feature-where", default=None)
    build_wide_sql.add_argument("--execute", action="store_true")
    build_wide_sql.add_argument("--sql-approved", action="store_true")
    build_wide_sql.set_defaults(func=cmd_build_wide_sql)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if len(raw_argv) >= 2 and raw_argv[0] == "feature" and raw_argv[1] == "d01-d02":
        raw_argv[1] = "prescreen"
    args = parser.parse_args(raw_argv)
    return args.func(args)
