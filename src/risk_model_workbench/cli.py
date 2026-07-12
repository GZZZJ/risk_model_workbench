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
from uuid import uuid4

import yaml

from risk_model_workbench.cli_meta import add_metadata_parsers
from risk_model_workbench.agent.advisor import (
    accept_advisor_response,
    advisor_request_is_answered,
    list_advisor_requests,
    load_advisor_context_pack,
    load_advisor_request,
)
from risk_model_workbench.agent.advisor_reducer import confirm_advisor_response, reject_advisor_response
from risk_model_workbench.agent.approvals import approve_request, load_approvals, reject_request
from risk_model_workbench.agent.executor import run_agent
from risk_model_workbench.agent.eval import evaluate_harness_suite
from risk_model_workbench.agent.recovery import diagnose_recovery, reconcile_operation
from risk_model_workbench.agent.plan import (
    agent_capabilities,
    agent_tool_schema,
    bind_agent_plan,
    load_agent_plan,
    rebind_agent_plan,
    save_agent_plan,
)
from risk_model_workbench.agent.state import (
    init_agent_state,
    load_agent_state,
    requeue_interrupted_task,
    save_agent_state,
)
from risk_model_workbench.agent.trace import append_trace, load_recent_trace
from risk_model_workbench.agent.transitions import apply_transition
from risk_model_workbench.agent.workspace_store import WorkspaceStore
from risk_model_workbench.application.action_runner import ActionRunner
from risk_model_workbench.application.context import VersionContext
from risk_model_workbench.application.handlers import production_handler_registry
from risk_model_workbench.config import ConfigError, load_yaml
from risk_model_workbench.feature_screening import write_feature_screening_summary
from risk_model_workbench.harness.errors import SQL_APPROVAL_REQUIRED, WorkspaceLockedError
from risk_model_workbench.harness.runtime import (
    classify_exception,
    current_action_attempt,
    register_action_artifact as register_artifact,
    run_with_retry,
    stage_action_done,
    stage_action_failed,
    stage_action_started,
)
from risk_model_workbench.harness.tools import TOOL_REGISTRY
from risk_model_workbench.harness.invocation import ActionInvocation
from risk_model_workbench.manifest import make_run_id
from risk_model_workbench.paths import REPO_ROOT, project_config_path, resolve_project_path, stage_config_path, workflow_path
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



def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    return path





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



def _resolve_refine_config_path(project_dir: Path, config: str | None) -> Path:
    if config:
        path = Path(config)
        return path if path.is_absolute() else project_dir / path
    return stage_config_path(project_dir, "refine_features")




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
        ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if not raw.suffix:
        project_candidate = stage_config_path(project_dir, name)
        if project_candidate.exists():
            return project_candidate
    return candidates[0] if candidates else stage_config_path(project_dir, name)


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




def _resolve_project_relative(project_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def _register_if_exists(run_path: Path, stage: str, relative_path: str | Path, *, description: str = "") -> None:
    if (run_path / relative_path).exists():
        register_artifact(run_path, stage, str(relative_path), description=description)






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
    errors: list[str] = []
    try:
        config_path = project_config_path(project_dir)
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
    except (ConfigError, OSError) as exc:
        errors.append(str(exc))

    configs_dir = project_dir / "configs"
    if configs_dir.exists():
        config_names = {path.stem for path in configs_dir.glob("*.yaml")}
        config_names.update(path.stem for path in configs_dir.glob("*.yml"))
        for name in sorted(config_names):
            try:
                path = stage_config_path(project_dir, name)
                if path.exists():
                    load_yaml(path)
            except (ConfigError, OSError) as exc:
                errors.append(str(exc))
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

    try:
        if args.write_state:
            summary = _summarize_project_status()
        else:
            summary = _read_only_action("project_status", _summarize_project_status)
    except ConfigError as exc:
        print(f"project status failed: {exc}")
        return 1
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
    try:
        audit = _read_only_action("run_audit", lambda: audit_run(project_dir, workspace_id, stage=args.stage))
    except ValueError as exc:
        print(f"run audit failed: {exc}")
        return 1
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
    try:
        project_config_path(project_dir)
    except ConfigError as exc:
        print(f"version migration failed: {exc}")
        return 1
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
    try:
        project_config_path(project_dir)
    except ConfigError as exc:
        print(f"version migration failed: {exc}")
        return 1
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
    try:
        errors = _read_only_action("workflow_validate", lambda: validate_workflow_definition(load_yaml(path)))
    except ValueError as exc:
        print(f"workflow validation failed: {path}")
        print(f"- {exc}")
        return 1
    if errors:
        print(f"workflow validation failed: {path}")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"workflow validation ok: {path}")
    return 0


def cmd_workflow_list(_: argparse.Namespace) -> int:
    for path in sorted((REPO_ROOT / "workflows").glob("*.yml")):
        if path.name == "stage_contracts.yml":
            continue
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
                "managed_by": state.get("managed_by", "workbench"),
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
    version_state = load_run_state(workspace)
    version_state["managed_by"] = "agent"
    save_version_state(workspace, version_state)
    index = load_version_index(project_dir)
    entry = next((item for item in index.get("versions", []) or [] if item.get("version_id") == args.version_id), None)
    if entry:
        entry = dict(entry)
        entry["managed_by"] = "agent"
        upsert_version_index(project_dir, entry, active=index.get("active_version_id") == args.version_id)
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
    register_artifact(
        workspace,
        "validate_config",
        "audit/agent_state.yml",
        kind="audit",
        description="Mutable Agent runtime state",
        integrity_mode="mutable",
    )
    register_artifact(
        workspace,
        "validate_config",
        "audit/agent_trace.jsonl",
        kind="audit",
        description="Append-only mutable Agent trace log",
        integrity_mode="mutable",
    )
    print(f"agent_plan: {workspace / 'agent_plan.yml'}")
    print(f"agent_state: {workspace / 'audit' / 'agent_state.yml'}")
    if args.execute:
        state = run_agent(project_dir, args.version_id)
        print(f"agent_status: {state.get('status')}")
    return 0


def cmd_agent_run(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    try:
        state = run_agent(project_dir, args.version_id)
    except (ValueError, WorkspaceLockedError) as exc:
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
        try:
            advisor_request = load_advisor_request(workspace, request_id) if request_id else {}
        except (KeyError, ValueError) as exc:
            print(f"agent resume blocked: invalid advisor request: {exc}")
            return 1
        if request_id and advisor_request.get("status") not in {"answered", "rejected"}:
            print(f"agent resume blocked: advisor response pending: {request_id}")
            return 1
    if state.get("status") == "waiting_for_user":
        request_id = str(blocker.get("advisor_request_id") or "")
        print(f"agent resume blocked: waiting for user confirmation: {request_id}")
        return 1
    if state.get("status") == "done_with_gaps":
        state = apply_transition(state, "resume", {"target_state": "running"})
        save_agent_state(workspace, state)
    if state.get("status") == "failed":
        current_task_id = str(state.get("current_task") or blocker.get("task_id") or "")
        failed_task = next(
            (
                item
                for item in state.get("tasks", []) or []
                if item.get("task_id") == current_task_id and item.get("status") == "failed"
            ),
            None,
        )
        tool_name = str((failed_task or {}).get("tool_name") or "")
        spec = TOOL_REGISTRY.get(tool_name)
        attempt_id = str((failed_task or {}).get("attempt_id") or "")
        if failed_task is None or spec is None or spec.execution_semantics not in {"read_only", "idempotent_write"} or not attempt_id:
            print("agent resume blocked: failed task is not explicitly safe to retry")
            return 1
        requeue_interrupted_task(workspace, current_task_id, attempt_id=attempt_id)
        append_trace(
            workspace,
            "decision",
            {
                "summary": "Operator explicitly retried a failed idempotent task after correcting its local cause.",
                "task_id": current_task_id,
                "attempt_id": attempt_id,
                "execution_semantics": spec.execution_semantics,
            },
        )
    try:
        resumed = run_agent(project_dir, args.version_id)
    except (ValueError, WorkspaceLockedError) as exc:
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


def cmd_agent_diagnose(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    workspace = resolve_workspace_dir(project_dir, version_id=args.version_id)
    try:
        payload = diagnose_recovery(workspace)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"agent diagnose failed: {exc}")
        return 1
    payload["project"] = str(project_dir)
    payload["version_id"] = args.version_id
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"unresolved_attempts: {payload['unresolved_count']}")
        divergence = payload["transaction_divergence"]
        print(f"transaction_divergence: {str(divergence['detected']).lower()}")
        for item in payload["attempts"]:
            if item["recovery_action"] != "none":
                print(f"- {item['attempt_id']}: {item['recovery_action']}")
    return 0


def cmd_agent_reconcile(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    workspace = resolve_workspace_dir(project_dir, version_id=args.version_id)
    try:
        with WorkspaceStore(workspace).runner_lock():
            receipt = reconcile_operation(
                workspace,
                operation_id=args.operation_id,
                outcome=args.outcome,
                evidence_path=args.evidence_path,
                evidence_sha256=args.evidence_sha256,
                operator_identity=args.operator_identity,
                note=args.note,
            )
    except (OSError, ValueError, WorkspaceLockedError) as exc:
        print(f"agent reconcile failed: {exc}")
        return 1
    append_trace(
        workspace,
        "decision",
        {
            "summary": "External operation explicitly reconciled.",
            "operation_id": args.operation_id,
            "outcome": args.outcome,
            "operator_identity": args.operator_identity,
            "evidence_sha256": args.evidence_sha256,
            "receipt_path": receipt["receipt_path"],
        },
    )
    if args.json:
        print(json.dumps(receipt, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"reconciliation_outcome: {receipt['outcome']}")
        print(f"receipt_path: {receipt['receipt_path']}")
    return 0


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


def cmd_agent_eval(args: argparse.Namespace) -> int:
    report = evaluate_harness_suite()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"suite: {report['suite']}")
        print(f"passed: {str(report['passed']).lower()}")
        print(f"scenario_count: {report['scenario_count']}")
        for name, metric in report["metrics"].items():
            print(f"{name}: {json.dumps(metric, ensure_ascii=False)}")
    return 0 if report["passed"] else 1


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
    if getattr(args, "request_id", ""):
        try:
            state = load_agent_state(workspace)
        except FileNotFoundError as exc:
            print(f"advisor rejection failed: {exc}")
            return 1
        result = reject_advisor_response(
            workspace,
            args.request_id,
            state,
            reason=args.reason or args.note or "rejected_by_user",
        )
        if not result.consumed:
            print("advisor_rejection: rejected")
            for error in result.errors:
                print(f"- {error}")
            return 1
        append_trace(
            workspace,
            "decision",
            {
                "summary": "Advisor user confirmation rejected.",
                "request_id": args.request_id,
                "reason": args.reason or args.note or "",
            },
        )
        print("advisor_rejection: accepted")
        print(f"agent_status: {result.status}")
        print(f"receipt_path: {result.receipt_path}")
        return 0
    if not getattr(args, "approval_id", ""):
        print("approval rejection failed: --approval-id is required")
        return 1
    if not getattr(args, "rejected_by", ""):
        print("approval rejection failed: --rejected-by is required")
        return 1
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


def cmd_agent_confirm(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    workspace = resolve_workspace_dir(project_dir, version_id=args.version_id)
    try:
        state = load_agent_state(workspace)
    except FileNotFoundError as exc:
        print(f"advisor confirmation failed: {exc}")
        return 1
    result = confirm_advisor_response(workspace, args.request_id, state, confirmed_by=args.confirmed_by)
    if not result.consumed:
        print("advisor_confirmation: rejected")
        for error in result.errors:
            print(f"- {error}")
        return 1
    append_trace(
        workspace,
        "decision",
        {
            "summary": "Advisor user confirmation recorded.",
            "request_id": args.request_id,
            "confirmed_by": args.confirmed_by,
        },
    )
    print("advisor_confirmation: accepted")
    print(f"agent_status: {result.status}")
    print(f"receipt_path: {result.receipt_path}")
    return 0


def cmd_agent_tools(args: argparse.Namespace) -> int:
    tools = agent_tool_schema()
    if args.json:
        print(json.dumps(tools, ensure_ascii=False, indent=2))
    else:
        for tool in tools:
            print(f"{tool['name']}: permission={tool['permission']} approval={tool['requires_approval']}")
    return 0


def cmd_agent_capabilities(args: argparse.Namespace) -> int:
    payload = agent_capabilities()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"capabilities_version: {payload['version']}")
        print(f"context_pack_version: {payload['context_pack']['version']}")
        print("tools:")
        for tool in payload["tools"]:
            print(f"- {tool.get('name')}")
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
    except (KeyError, ValueError) as exc:
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


def cmd_agent_advisor_context(args: argparse.Namespace) -> int:
    try:
        project_dir = resolve_project_path(args.project)
        workspace = resolve_workspace_dir(project_dir, version_id=args.version_id)
        request = load_advisor_request(workspace, args.request_id)
        context_pack = load_advisor_context_pack(workspace, request)
    except (KeyError, FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        else:
            print(f"advisor context failed: {exc}")
        return 1
    if args.json:
        print(json.dumps(context_pack, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"context_hash: {context_pack.get('context_hash')}")
        print(f"context_pack: {request.get('context_pack')}")
        for item in context_pack.get("files") or []:
            print(f"- {item.get('path')}: {item.get('status')}")
    return 0


def cmd_agent_advisor_accept(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    workspace = resolve_workspace_dir(project_dir, version_id=args.version_id)
    result = accept_advisor_response(workspace, args.response)
    if not result.get("accepted"):
        if result.get("stored"):
            print("advisor_response: rejected")
            print(f"request_id: {result.get('request_id')}")
            print(f"response_path: {result.get('response_path')}")
            return 0
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


def _run_application_action(args: argparse.Namespace, tool_name: str, params: dict[str, Any]) -> int:
    """Thin compatibility adapter from argparse to the shared in-process runner."""
    project_dir = resolve_project_path(args.project)
    workspace = _run_path(args)
    workspace_id = _workspace_arg(args)
    context = VersionContext(
        project_dir=project_dir,
        version_id=workspace_id,
        workspace=workspace,
        runtime_config_dir=workspace / RUNTIME_CONFIG_DIR,
        manifest_path=workspace / "audit" / "artifact_manifest.json",
        version_state_path=(workspace / "version_state.yml" if (workspace / "version_state.yml").exists() else workspace / "run_state.yml"),
    )
    invocation = ActionInvocation(
        tool_name=tool_name,
        params=params,
        project=str(project_dir),
        version_id=workspace_id,
    )
    active_attempt = current_action_attempt()
    attempt_id = active_attempt.attempt_id if active_attempt is not None else f"cli_{tool_name}_{uuid4().hex}"
    task_id = active_attempt.task_id if active_attempt is not None else tool_name
    spec = TOOL_REGISTRY[tool_name]
    runner = ActionRunner(
        handlers=production_handler_registry(),
        policy_check=lambda candidate, candidate_context: (
            candidate.tool_name == tool_name
            and spec.permission in {"writes_run", "dp_sql_pull"}
            and candidate_context.workspace == workspace
        ),
    )
    result = runner.run(
        invocation=invocation,
        context=context,
        attempt_id=attempt_id,
        task_id=task_id,
    )
    stream = sys.stderr if result.status == "failed" else sys.stdout
    if result.message:
        print(result.message, file=stream)
    if result.failure_code == "advisor_required":
        return 2
    return 1 if result.status == "failed" else 0


def cmd_sample_check(args: argparse.Namespace) -> int:
    return _run_application_action(args, "sample_check", {})


def cmd_feature_metadata(args: argparse.Namespace) -> int:
    return _run_application_action(
        args,
        "feature_metadata",
        {key: value for key, value in {"config": args.config, "tables_file": args.tables_file}.items() if value is not None},
    )


def _feature_tool_for_args(args: argparse.Namespace, *, domain: str) -> str:
    path = _run_path(args)
    project_dir = resolve_project_path(args.project)
    if domain == "prescreen":
        if _runtime_is_local_feather(path, project_dir):
            return "feature_prescreen_local"
        if bool(getattr(args, "sql_approved", False)):
            return "feature_prescreen_execute"
        return "feature_prescreen_prepare"
    if domain == "refine":
        if _runtime_is_local_feather(path, project_dir):
            return "feature_refine_local"
        if bool(getattr(args, "sql_approved", False)):
            return "feature_refine_execute"
        return "feature_refine_prepare"
    if domain == "wide":
        if _runtime_is_local_feather(path, project_dir):
            return "build_wide_sql_local"
        if bool(getattr(args, "execute", False)):
            return "build_wide_sql_execute"
        return "build_wide_sql_prepare"
    raise ValueError(f"unknown feature domain: {domain}")


def cmd_feature_prescreen(args: argparse.Namespace) -> int:
    params = {
        key: value
        for key, value in {
            "config": args.config,
            "tables": args.table,
            "max_tables": args.max_tables,
        }.items()
        if value is not None
    }
    return _run_application_action(args, _feature_tool_for_args(args, domain="prescreen"), params)


def cmd_build_wide_sql(args: argparse.Namespace) -> int:
    from risk_model_workbench.application.handlers import feature_selection as feature_actions

    feature_actions.WIDE_SQL_GENERATOR = generate_wide_sql
    params = {
        key: value
        for key, value in {
            "remain_features": args.remain_features,
            "sql_output": args.sql_output,
            "feature_map_output": args.feature_map_output,
            "summary_output": args.summary_output,
            "execution_output": args.execution_output,
            "base_table": args.base_table,
            "output_table": args.output_table,
            "base_where": args.base_where,
            "feature_where": args.feature_where,
            "sql_approved": bool(args.sql_approved),
        }.items()
        if value is not None
    }
    return _run_application_action(args, _feature_tool_for_args(args, domain="wide"), params)


def cmd_feature_refine(args: argparse.Namespace) -> int:
    params = {
        key: value
        for key, value in {"config": args.config, "sample_max_rows": args.sample_max_rows}.items()
        if value is not None
    }
    return _run_application_action(args, _feature_tool_for_args(args, domain="refine"), params)


def cmd_train(args: argparse.Namespace) -> int:
    params = {"experiment": args.experiment}
    for key in ["input_feather", "feature_list", "score_output", "input_dir", "config"]:
        value = getattr(args, key, None)
        if value is not None:
            params[key] = value
    if bool(getattr(args, "plan_only", False)):
        params["plan_only"] = True
    if bool(getattr(args, "skip_split_check", False)):
        params["skip_split_check"] = True
    return _run_application_action(args, "train_baseline", params)


def cmd_evaluate(args: argparse.Namespace) -> int:
    return _run_application_action(
        args,
        "evaluate",
        {key: value for key, value in {"scores_feather": args.scores_feather, "output_dir": args.output_dir}.items() if value is not None},
    )


def cmd_compare(args: argparse.Namespace) -> int:
    params = {"champions": _as_string_list(args.champion)} if args.champion else {}
    return _run_application_action(args, "compare", params)


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
    params = {"report_target": args.report_target} if args.report_target else {}
    return _run_application_action(args, "report", params)


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

    diagnose = agent_sub.add_parser("diagnose", help="diagnose interrupted Agent attempts and transaction divergence")
    diagnose.add_argument("--project", required=True)
    diagnose.add_argument("--version-id", required=True)
    diagnose.add_argument("--json", action="store_true")
    diagnose.set_defaults(func=cmd_agent_diagnose)

    reconcile = agent_sub.add_parser("reconcile", help="consume explicit evidence for an unknown external operation")
    reconcile.add_argument("--project", required=True)
    reconcile.add_argument("--version-id", required=True)
    reconcile.add_argument("--operation-id", required=True)
    reconcile.add_argument("--outcome", required=True, choices=["confirmed_succeeded", "confirmed_failed", "abandoned"])
    reconcile.add_argument("--evidence-path", "--evidence", dest="evidence_path", required=True)
    reconcile.add_argument("--evidence-sha256", required=True)
    reconcile.add_argument("--operator-identity", required=True)
    reconcile.add_argument("--note", required=True)
    reconcile.add_argument("--json", action="store_true")
    reconcile.set_defaults(func=cmd_agent_reconcile)

    status = agent_sub.add_parser("status", help="show Agent status")
    status.add_argument("--project", required=True)
    status.add_argument("--version-id", required=True)
    status.add_argument("--json", action="store_true")
    status.add_argument("--tail", type=int, default=20)
    status.set_defaults(func=cmd_agent_status)

    agent_eval = agent_sub.add_parser("eval", help="run the Agent Harness product evaluation suite")
    agent_eval.add_argument("--suite", required=True, choices=["harness"])
    agent_eval.add_argument("--json", action="store_true")
    agent_eval.set_defaults(func=cmd_agent_eval)

    approve = agent_sub.add_parser("approve", help="record approval for a blocked high-risk Agent action")
    approve.add_argument("--project", required=True)
    approve.add_argument("--version-id", required=True)
    approve.add_argument("--approval-id", required=True)
    approve.add_argument("--approved-by", required=True)
    approve.add_argument("--note", default="")
    approve.set_defaults(func=cmd_agent_approve)

    confirm = agent_sub.add_parser("confirm", help="confirm an Advisor response that requires explicit user confirmation")
    confirm.add_argument("--project", required=True)
    confirm.add_argument("--version-id", required=True)
    confirm.add_argument("--request-id", required=True)
    confirm.add_argument("--confirmed-by", required=True)
    confirm.set_defaults(func=cmd_agent_confirm)

    reject = agent_sub.add_parser("reject", help="reject a blocked approval or Advisor confirmation")
    reject.add_argument("--project", required=True)
    reject.add_argument("--version-id", required=True)
    reject_target = reject.add_mutually_exclusive_group(required=True)
    reject_target.add_argument("--approval-id")
    reject_target.add_argument("--request-id")
    reject.add_argument("--rejected-by", default="")
    reject.add_argument("--reason", default="")
    reject.add_argument("--note", default="")
    reject.set_defaults(func=cmd_agent_reject)

    tools = agent_sub.add_parser("tools", help="export Agent tool schema")
    tools.add_argument("--json", action="store_true")
    tools.set_defaults(func=cmd_agent_tools)

    capabilities = agent_sub.add_parser("capabilities", help="export the Host-Agent capability contract")
    capabilities.add_argument("--json", action="store_true")
    capabilities.set_defaults(func=cmd_agent_capabilities)

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

    advisor_context = advisor_sub.add_parser("context", help="show the immutable context pack for one request")
    advisor_context.add_argument("--project", required=True)
    advisor_context.add_argument("--version-id", required=True)
    advisor_context.add_argument("--request-id", required=True)
    advisor_context.add_argument("--json", action="store_true")
    advisor_context.set_defaults(func=cmd_agent_advisor_context)

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
