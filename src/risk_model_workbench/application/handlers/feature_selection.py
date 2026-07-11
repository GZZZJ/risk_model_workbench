"""Feature-selection application handlers shared by CLI and Agent Runtime."""

from __future__ import annotations

import json
import shutil
import csv
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

from risk_model_workbench.application.context import VersionContext
from risk_model_workbench.config import load_yaml
from risk_model_workbench.data.sql_review import review_sql_text
from risk_model_workbench.harness.errors import SQL_APPROVAL_REQUIRED
from risk_model_workbench.harness.invocation import ActionInvocation
from risk_model_workbench.harness.runtime import (
    ActionResult,
    action_attempt,
    classify_exception,
    current_action_attempt,
    detached_action_attempt,
    register_action_artifact,
    stage_action_done,
    stage_action_failed,
    stage_action_started,
)
from risk_model_workbench.state import load_run_state
from risk_model_workbench.wide_sql import generate_wide_sql


WIDE_SQL_GENERATOR: Callable[..., tuple[Path, Path, Path]] = generate_wide_sql


def _last_result(context: VersionContext, stage: str) -> ActionResult:
    payload = dict(load_run_state(context.workspace)["stages"][stage]["last_result"])
    allowed = set(ActionResult.__dataclass_fields__)
    return ActionResult(**{key: value for key, value in payload.items() if key in allowed})


def _config_path(context: VersionContext, name: str, explicit: object = None) -> Path:
    if explicit:
        path = Path(str(explicit))
        return path if path.is_absolute() else context.project_dir / path
    for path in [context.runtime_config_dir / f"{name}.yaml", context.project_dir / "configs" / f"{name}.yaml"]:
        if path.exists():
            return path
    return context.project_dir / "configs" / f"{name}.yaml"


def _is_local(context: VersionContext) -> bool:
    for name, root in [("feature_select", "feature_select"), ("refine_features", "feature_refine")]:
        path = _config_path(context, name)
        if not path.exists():
            continue
        cfg = (load_yaml(path).get(root) or {})
        request = cfg.get("runtime_request") or {}
        input_cfg = cfg.get("input") or {}
        if request.get("data_source_mode") == "local_feather" or input_cfg.get("local_feather_path"):
            return True
    return False


def _params(invocation: ActionInvocation) -> dict[str, Any]:
    value = invocation.canonical_payload()["params"]
    assert isinstance(value, dict)
    return value


def _workspace_output(context: VersionContext, value: object, default: str) -> Path:
    """Resolve a requested artifact path and reject escape before side effects."""
    path = Path(str(value)) if value else context.workspace / default
    if not path.is_absolute():
        path = context.workspace / path
    resolved = path.resolve()
    try:
        resolved.relative_to(context.workspace.resolve())
    except ValueError as exc:
        raise ValueError(f"wide SQL output must stay inside version workspace: {resolved}") from exc
    return resolved


def _copy_register(context: VersionContext, stage: str, source: Path, target: str) -> None:
    if not source.exists():
        return
    destination = context.workspace / target
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    register_action_artifact(context.workspace, stage, destination)


def run_feature_metadata(invocation: ActionInvocation, context: VersionContext, attempt_id: str) -> ActionResult:
    del attempt_id
    params = _params(invocation)
    with detached_action_attempt():
        stage_action_started(context.workspace, "feature_metadata")
        try:
            if _is_local(context):
                _run_local_metadata(context)
                return _last_result(context, "feature_metadata")
            from risk_model_workbench.feature_selection.metadata import execute_metadata_action

            config = _config_path(context, "feature_select", params.get("config"))
            code = execute_metadata_action(
                project_dir=context.project_dir,
                workspace=context.workspace,
                config_path=config,
                tables_file=str(params["tables_file"]) if params.get("tables_file") else None,
            )
            if code:
                stage_action_failed(context.workspace, "feature_metadata", f"metadata command exited with code {code}")
            else:
                for artifact in sorted((context.workspace / "feature_metadata").glob("*")):
                    if artifact.is_file():
                        register_action_artifact(context.workspace, "feature_metadata", artifact)
                stage_action_done(context.workspace, "feature_metadata")
        except Exception as exc:
            stage_action_failed(context.workspace, "feature_metadata", str(exc), failure_code=classify_exception(exc))
    return _last_result(context, "feature_metadata")


def _run_local_metadata(context: VersionContext) -> None:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    feature_cfg = load_yaml(_config_path(context, "feature_select")).get("feature_select", {})
    request = feature_cfg.get("runtime_request") or {}
    value = request.get("sample_location")
    if not value:
        refine_cfg = load_yaml(_config_path(context, "refine_features")).get("feature_refine", {})
        value = (refine_cfg.get("input") or {}).get("local_feather_path")
    if not value:
        raise FileNotFoundError("local_feather sample_location is missing")
    feather = Path(str(value))
    if not feather.is_absolute():
        feather = context.project_dir / feather
    schema = ipc.open_file(pa.memory_map(str(feather.resolve()), "r")).schema
    output = context.workspace / "feature_metadata"
    output.mkdir(parents=True, exist_ok=True)
    rows = [
        {"table_index": 1, "full_table_name": "local_feather", "feature_name": field.name, "feature_type": str(field.type), "feature_comment": "", "ordinal": index}
        for index, field in enumerate(schema, start=1)
    ]
    with (output / "feature_columns.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "feature_table_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["table_index", "full_table_name", "feature_count", "source_path"])
        writer.writeheader()
        writer.writerow({"table_index": 1, "full_table_name": "local_feather", "feature_count": len(rows), "source_path": str(feather.resolve())})
    meta = output / "feature_tables_meta.json"
    meta.write_text(json.dumps({"version": 1, "source": "local_feather_schema", "source_path": str(feather.resolve()), "feature_count": len(rows), "fields": [{"name": row["feature_name"], "type": row["feature_type"]} for row in rows]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for artifact in [meta, output / "feature_table_summary.csv", output / "feature_columns.csv"]:
        register_action_artifact(context.workspace, "feature_metadata", artifact)
    stage_action_done(context.workspace, "feature_metadata", message="local feather schema metadata generated")


def run_feature_prescreen(invocation: ActionInvocation, context: VersionContext, attempt_id: str) -> ActionResult:
    del attempt_id
    params = _params(invocation)
    active_attempt = current_action_attempt()
    with detached_action_attempt():
        stage_action_started(context.workspace, "feature_prescreen")
        try:
            if invocation.tool_name == "feature_prescreen_local" or _is_local(context):
                _run_local_prescreen(context)
                return _last_result(context, "feature_prescreen")
            from risk_model_workbench.feature_selection.refine import execute_prescreen_action

            config = _config_path(context, "feature_select", params.get("config"))
            execute = invocation.tool_name == "feature_prescreen_execute"
            attempt_scope = action_attempt(active_attempt) if execute and active_attempt is not None else nullcontext()
            with attempt_scope:
                code = execute_prescreen_action(
                    project_dir=context.project_dir,
                    workspace=context.workspace,
                    config_path=config,
                    prepare=invocation.tool_name == "feature_prescreen_prepare",
                    execute=execute,
                    tables=[str(item) for item in params.get("tables") or []],
                    max_tables=int(params["max_tables"]) if params.get("max_tables") is not None else None,
                )
            if code:
                stage_action_failed(context.workspace, "feature_prescreen", f"prescreen command exited with code {code}")
            else:
                candidates = [
                    context.workspace / "feature_selection" / "prescreen" / "results",
                    context.project_dir / "runs" / "feature_prescreen" / "results",
                ]
                for name in ["prescreen_run_summary.json", "prescreen_final_remain_features.json", "prescreen_table_summary.csv"]:
                    source = next((directory / name for directory in candidates if (directory / name).exists()), Path("__missing__"))
                    _copy_register(context, "feature_prescreen", source, f"feature_selection/{name}")
                prepare = invocation.tool_name == "feature_prescreen_prepare"
                stage_action_done(context.workspace, "feature_prescreen", scaffold=prepare, message="SQL generation complete; waiting for approval" if prepare else "", failure_code=SQL_APPROVAL_REQUIRED if prepare else "")
        except Exception as exc:
            stage_action_failed(context.workspace, "feature_prescreen", str(exc), failure_code=classify_exception(exc))
    return _last_result(context, "feature_prescreen")


def _run_local_prescreen(context: VersionContext) -> None:
    from risk_model_workbench.data.local_feather_profile import profile_local_feather, write_local_feather_profile
    from risk_model_workbench.data.pull_engine import select_data_pull_engine, write_execution_environment

    project_cfg = load_yaml(context.runtime_config_dir / "project.yml") if (context.runtime_config_dir / "project.yml").exists() else load_yaml(context.project_dir / "project.yml")
    feature_cfg = load_yaml(_config_path(context, "feature_select")).get("feature_select", {})
    request = feature_cfg.get("runtime_request") or {}
    data_cfg = project_cfg.get("data") or {}
    value = request.get("sample_location") or data_cfg.get("raw_path")
    if not value:
        raise FileNotFoundError("local feather source is missing")
    feather = Path(str(value))
    if not feather.is_absolute():
        feather = context.project_dir / feather
    required = [*data_cfg.get("id_columns", []), data_cfg.get("target_column"), data_cfg.get("split_column")]
    required = [str(item) for item in required if item]
    profile = profile_local_feather(feather, required_columns=required, split_column=data_cfg.get("split_column"), target_column=data_cfg.get("target_column"), feature_exclude_columns=required)
    output = context.workspace / "feature_selection"
    write_local_feather_profile(profile, output / "profiles/local_feather_profile.json")
    contract = {"data_source_mode": "local_feather", "source_table": None, "local_feather_path": str(feather.resolve()), "stage": "feature_prescreen", "note": "local_feather uses an existing local file and is independent from remote DP pull"}
    (output / "data_source_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_execution_environment(context.workspace, select_data_pull_engine(data_source_mode="local_feather"))
    features = list(profile.get("candidate_feature_columns") or [])
    for name, payload in {
        "resource_plan.json": {"stage": "feature_prescreen", "data_source_mode": "local_feather", "row_count": profile.get("row_count"), "feature_count": len(features)},
        "sampling_plan.json": {"mode": "full", "data_source_mode": "local_feather", "row_count": profile.get("row_count")},
        "batch_plan.json": {"batch_count": 1, "feature_count": len(features), "batches": [{"batch_id": 1, "features": features}]},
    }.items():
        (output / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for artifact in [output / "data_source_contract.json", output / "execution_environment.json", output / "resource_plan.json", output / "sampling_plan.json", output / "batch_plan.json", output / "profiles/local_feather_profile.json"]:
        register_action_artifact(context.workspace, "feature_prescreen", artifact)
    stage_action_done(context.workspace, "feature_prescreen", message="local feather mode: prescreen by design, intake evidence emitted")


def run_build_wide_sql(invocation: ActionInvocation, context: VersionContext, attempt_id: str) -> ActionResult:
    del attempt_id
    params = _params(invocation)
    active_attempt = current_action_attempt()
    with detached_action_attempt():
        stage_action_started(context.workspace, "build_wide_sql")
        try:
            if _is_local(context):
                skipped = context.workspace / "feature_selection" / "wide_table_skipped.json"
                skipped.parent.mkdir(parents=True, exist_ok=True)
                skipped.write_text(json.dumps({"status": "done", "reason": "local_feather_by_design"}) + "\n", encoding="utf-8")
                register_action_artifact(context.workspace, "build_wide_sql", skipped)
                stage_action_done(context.workspace, "build_wide_sql", message="local feather mode: wide table skipped by design")
                return _last_result(context, "build_wide_sql")
            config = _config_path(context, "feature_select", params.get("config"))
            root = (load_yaml(config).get("feature_select") or {}).get("wide_table") or {}
            execute = invocation.tool_name == "build_wide_sql_execute"
            approved = bool(params.get("sql_approved", execute))
            if execute and not approved:
                raise PermissionError("SQL approval is required before wide-table execution")
            runtime_remain = context.workspace / "feature_selection" / "prescreen" / "results" / "prescreen_final_remain_features.json"
            remain = Path(str(root.get("remain_features_path") or (runtime_remain if runtime_remain.exists() else params.get("remain_features")) or context.project_dir / "runs/feature_prescreen/results/prescreen_final_remain_features.json"))
            if not remain.is_absolute():
                remain = context.project_dir / remain
            sql_value = root.get("sql_output") or params.get("sql_output")
            map_value = root.get("feature_map_output") or params.get("feature_map_output")
            summary_value = root.get("summary_output") or params.get("summary_output")
            if sql_value == "queries/06_build_prescreen_wide_table.sql":
                sql_value = None
            if map_value == "runs/feature_prescreen/results/prescreen_wide_feature_map.csv":
                map_value = None
            if summary_value == "runs/feature_prescreen/results/prescreen_wide_sql_summary.json":
                summary_value = None
            sql_path = _workspace_output(context, sql_value, "queries/06_build_prescreen_wide_table.sql")
            map_path = _workspace_output(context, map_value, "feature_selection/prescreen_wide_feature_map.csv")
            summary_path = _workspace_output(context, summary_value, "feature_selection/wide_sql_summary.json")
            execution_value = params.get("execution_output")
            if execution_value == "feature_selection/wide_table_execution.json":
                execution_value = None
            execution_path = _workspace_output(
                context,
                execution_value,
                "feature_selection/wide_table_execution.json",
            )
            runtime_project_path = context.runtime_config_dir / "project.yml"
            WIDE_SQL_GENERATOR(project_dir=context.project_dir, remain_features_path=remain, sql_output_path=sql_path, feature_map_path=map_path, summary_path=summary_path, base_table=params.get("base_table"), output_table=params.get("output_table"), base_where=params.get("base_where"), feature_where=params.get("feature_where"), config_path=config, project_config_path=runtime_project_path if runtime_project_path.exists() else None)
            for artifact in [sql_path, map_path, summary_path]:
                register_action_artifact(context.workspace, "build_wide_sql", artifact)
            from risk_model_workbench.data.sql_evidence import write_sql_evidence

            write_sql_evidence(
                context.workspace,
                sql_path.read_text(encoding="utf-8"),
                source="wide_sql.generate_wide_sql",
                purpose="build_wide_sql",
                stage="build_wide_sql",
                sql_kind="generated",
                name="build_wide_sql.sql",
            )
            register_action_artifact(context.workspace, "build_wide_sql", context.workspace / "queries/sql_evidence_manifest.json")
            runtime_project = load_yaml(runtime_project_path) if runtime_project_path.exists() else (load_yaml(context.project_dir / "project.yml") if (context.project_dir / "project.yml").exists() else {})
            data_cfg = runtime_project.get("data") or {}
            runtime_request = (load_yaml(config).get("feature_select") or {}).get("runtime_request") or {}
            gate = (runtime_request.get("step_params") or {}).get("sql_review_gate") or {}
            review = review_sql_text(
                sql_path.read_text(encoding="utf-8"),
                approved_for_execution=approved,
                target_columns=[str(data_cfg.get("target_column") or "")],
                time_columns=[str(data_cfg.get("time_column") or ""), str(data_cfg.get("period_column") or "")],
            )
            review["block_on_high_risk"] = bool(gate.get("block_on_high_risk", True))
            review["sql_path"] = str(sql_path)
            review_path = context.workspace / "feature_selection" / "sql_review.json"
            review_path.parent.mkdir(parents=True, exist_ok=True)
            review_path.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            register_action_artifact(context.workspace, "build_wide_sql", review_path)
            if execute:
                if review["high_risk"] and review["block_on_high_risk"]:
                    raise ValueError("high-risk SQL review blocked execution")
                from risk_model_workbench.dp_feather import execute_dp_sql

                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                attempt_scope = action_attempt(active_attempt) if active_attempt is not None else nullcontext()
                with attempt_scope:
                    execution_result = execute_dp_sql(
                        project_dir=context.project_dir,
                        sql=sql_path.read_text(encoding="utf-8"),
                        operation_id="build_wide_sql_execute",
                        description=f"Create wide feature table {summary.get('output_table') or params.get('output_table') or ''}".strip(),
                        metadata_path=execution_path,
                        sql_approved=approved,
                        audit_workspace=context.workspace,
                    )
                execution_path.parent.mkdir(parents=True, exist_ok=True)
                execution_path.write_text(json.dumps({**execution_result, "approved": True, "sql_path": str(sql_path), "summary_path": str(summary_path), "feature_map_path": str(map_path), "output_table": summary.get("output_table") or params.get("output_table")}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                register_action_artifact(context.workspace, "build_wide_sql", execution_path)
                profile = context.workspace / "feature_selection" / "profiles" / "wide_table_profile.json"
                profile.parent.mkdir(parents=True, exist_ok=True)
                profile.write_text(json.dumps({"status": "metadata_only_after_ctas", "table": summary.get("output_table"), "feature_count": summary.get("features")}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                register_action_artifact(context.workspace, "build_wide_sql", profile)
            prepare = invocation.tool_name == "build_wide_sql_prepare"
            stage_action_done(context.workspace, "build_wide_sql", scaffold=prepare, message="SQL generation complete; waiting for approval" if prepare else "", failure_code=SQL_APPROVAL_REQUIRED if prepare else "")
        except Exception as exc:
            code = "sql_approval_required" if isinstance(exc, PermissionError) else classify_exception(exc)
            stage_action_failed(context.workspace, "build_wide_sql", str(exc), failure_code=code)
    return _last_result(context, "build_wide_sql")


def run_feature_refine(invocation: ActionInvocation, context: VersionContext, attempt_id: str) -> ActionResult:
    del attempt_id
    params = _params(invocation)
    active_attempt = current_action_attempt()
    with detached_action_attempt():
        stage_action_started(context.workspace, "feature_refine")
        try:
            from risk_model_workbench.feature_selection.refine import execute_refine_action

            config = _config_path(context, "refine_features", params.get("config"))
            execute = invocation.tool_name == "feature_refine_execute"
            attempt_scope = action_attempt(active_attempt) if execute and active_attempt is not None else nullcontext()
            with attempt_scope:
                code = execute_refine_action(
                    project_dir=context.project_dir,
                    workspace=context.workspace,
                    config_path=config,
                    prepare=invocation.tool_name == "feature_refine_prepare",
                    execute=execute,
                    sample_max_rows=int(params["sample_max_rows"]) if params.get("sample_max_rows") is not None else None,
                )
            if code:
                stage_action_failed(context.workspace, "feature_refine", f"feature refine command exited with code {code}", failure_code="data_missing")
            else:
                cfg = load_yaml(config).get("feature_refine", {}) if config.exists() else {}
                output = Path(str(cfg.get("output_dir") or context.workspace / "feature_selection"))
                if not output.is_absolute():
                    output = context.project_dir / output
                for source_name, targets in {
                    "stage_summary.json": ["stage_summary.json", "feature_stage_summary.json"],
                    "resource_usage.json": ["resource_usage.json"],
                    "final_500_features.txt": ["final_500_features.txt"],
                    "final_features.txt": ["final_features.txt"],
                }.items():
                    for target in targets:
                        _copy_register(context, "feature_refine", output / source_name, f"feature_selection/{target}")
                prepare = invocation.tool_name == "feature_refine_prepare"
                stage_action_done(context.workspace, "feature_refine", scaffold=prepare, message="SQL dry run waiting for approval" if prepare else "", failure_code=SQL_APPROVAL_REQUIRED if prepare else "")
        except Exception as exc:
            stage_action_failed(context.workspace, "feature_refine", str(exc), failure_code=classify_exception(exc))
    return _last_result(context, "feature_refine")


__all__ = ["run_build_wide_sql", "run_feature_metadata", "run_feature_prescreen", "run_feature_refine"]
