"""Create execution plans from model request documents."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from risk_model_workbench.paths import workflow_path
from risk_model_workbench.planning.steps import implemented_step_ids_for_stage, resolve_step_configuration, step_params_for
from risk_model_workbench.request.data_source import LOCAL_FEATHER, has_remote_feature_source, resolve_data_source_mode


class _NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def _as_list(value: Any, default: list[Any] | None = None) -> list[Any]:
    if value is None:
        return list(default or [])
    if isinstance(value, list):
        return value
    return [value]


def _load_project_yaml(project_path: str | Path) -> dict[str, Any]:
    project_dir = Path(project_path)
    for name in ["project.yml", "project.yaml"]:
        path = project_dir / name
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return data if isinstance(data, dict) else {}
    return {}


def _project_champions(project_path: str | Path) -> list[Any]:
    config = _load_project_yaml(project_path)
    champions = config.get("champions") or {}
    if isinstance(champions, dict):
        return _as_list(champions.get("score_columns"))
    return _as_list(champions)


def _workflow_stages(workflow: str) -> set[str]:
    path = workflow_path(workflow)
    if not path.exists():
        raise ValueError(f"unknown workflow: {workflow}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    stages = data.get("stages") or []
    if not isinstance(stages, list):
        raise ValueError(f"workflow stages must be a list: {workflow}")
    return {str(stage) for stage in stages}


def _experiments_from_metadata(metadata: dict[str, Any]) -> list[Any]:
    experiments = metadata.get("experiments")
    if isinstance(experiments, list) and experiments:
        return experiments

    description = str(metadata.get("experiment_description") or "").strip()
    if not description:
        return []

    return [
        {
            "name": "baseline_from_description",
            "method": "lightgbm",
            "segment": "all",
            "description": description,
        }
    ]


def _task(
    *,
    task_id: str,
    task_type: str,
    depends_on: list[str],
    args: list[str],
    outputs: list[str],
    scenario_profile: str,
    step_ids: list[str],
    step_params: dict[str, dict[str, Any]],
    workspace: str | None = None,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "type": task_type,
        "status": "pending",
        "workspace": workspace or f"tasks/{task_id}",
        "depends_on": depends_on,
        "command": {
            "executable": "rmw",
            "args": args,
        },
        "outputs": outputs,
        "scenario_profile": scenario_profile,
        "step_ids": step_ids,
        "step_params": step_params,
    }


def create_execution_plan(request_doc: dict[str, Any], project_path: str | Path) -> dict[str, Any]:
    """Create a deterministic execution plan from a parsed model request."""
    metadata = request_doc["metadata"]
    project = str(project_path)
    request_id = metadata["request_id"]
    workflow = metadata.get("workflow", "full_modeling")
    workflow_stages = _workflow_stages(str(workflow))
    run_arg = "<run_id>"
    tasks: list[dict[str, Any]] = []
    step_config = resolve_step_configuration(metadata, project_path)
    scenario_profile = step_config["scenario_profile"]
    workflow_stage_steps = {
        stage: steps
        for stage, steps in step_config["stage_steps"].items()
        if stage in workflow_stages
    }
    workflow_step_ids = {step_id for steps in workflow_stage_steps.values() for step_id in steps}
    workflow_step_params = {
        step_id: params
        for step_id, params in step_config["step_params"].items()
        if step_id in workflow_step_ids
    }
    workflow_resolved_steps = [
        step
        for step in step_config["resolved_steps"]
        if step["id"] in workflow_step_ids
    ]
    workflow_planned_steps = [
        step
        for step in workflow_resolved_steps
        if step["implementation_status"] == "planned"
    ]

    sample_checks = _as_list(metadata.get("sample_checks"), default=["sample_check_001"])
    sample_task_ids: list[str] = []
    sample_step_ids = implemented_step_ids_for_stage(step_config, "sample_check")
    if "sample_check" in workflow_stages:
        for index, item in enumerate(sample_checks, start=1):
            name = item.get("name") if isinstance(item, dict) else str(item)
            task_id = name or f"sample_check_{index:03d}"
            sample_task_ids.append(task_id)
            tasks.append(
                _task(
                    task_id=task_id,
                    task_type="sample_check",
                    depends_on=[],
                    args=["sample", "check", "--project", project, "--run-id", run_arg],
                    outputs=["sample_check/sample_summary.json", "sample_check/sample_check_report.md"],
                    scenario_profile=scenario_profile,
                    step_ids=sample_step_ids,
                    step_params=step_params_for(step_config, sample_step_ids),
                )
            )

    feature_cfg = metadata.get("feature_selection") or {}
    local_feather_without_remote_features = (
        resolve_data_source_mode(metadata) == LOCAL_FEATHER
        and not has_remote_feature_source(metadata)
        and "feature_selection" not in metadata
    )
    default_feature_rounds = ["refine"] if local_feather_without_remote_features else ["metadata", "prescreen", "refine"]
    raw_feature_rounds = _as_list(feature_cfg.get("rounds"), default=default_feature_rounds)
    has_wide_sql_round = any(
        (item.get("name") if isinstance(item, dict) else str(item)).replace("-", "_")
        in {"build_wide_sql", "wide_sql", "build_wide"}
        for item in raw_feature_rounds
    )
    prescreen_round_names = {"prescreen", "feature_prescreen", "coarse_screening", "coarse", "d01_d02", "d01d02"}
    feature_rounds: list[Any] = []
    wide_sql_inserted = has_wide_sql_round
    seen_prescreen_round = False
    for item in raw_feature_rounds:
        normalized = (item.get("name") if isinstance(item, dict) else str(item)).replace("-", "_")
        if normalized == "refine" and seen_prescreen_round and not wide_sql_inserted:
            feature_rounds.append("build_wide_sql")
            wide_sql_inserted = True
        feature_rounds.append(item)
        if normalized in prescreen_round_names:
            seen_prescreen_round = True
    feature_task_ids: list[str] = []
    feature_dependency = sample_task_ids[-1:] if sample_task_ids else []
    for item in feature_rounds:
        name = item.get("name") if isinstance(item, dict) else str(item)
        normalized = name.replace("-", "_")
        if normalized == "metadata":
            task_id = "feature_metadata"
            args = ["feature", "metadata", "--project", project, "--run-id", run_arg]
            outputs = ["feature_selection/feature_table_summary.csv", "feature_selection/feature_columns.csv"]
            stage = "feature_metadata"
        elif normalized in prescreen_round_names:
            task_id = "feature_prescreen"
            args = ["feature", "prescreen", "--project", project, "--run-id", run_arg, "--dry-run-sql"]
            outputs = ["feature_selection/prescreen_run_summary.json", "feature_selection/prescreen_final_remain_features.json"]
            stage = "feature_prescreen"
        elif normalized in {"build_wide_sql", "wide_sql", "build_wide"}:
            task_id = "build_wide_sql"
            args = ["build-wide-sql", "--project", project, "--run-id", run_arg]
            outputs = [
                "queries/06_build_prescreen_wide_table.sql",
                "feature_selection/wide_sql_summary.json",
                "feature_selection/prescreen_wide_feature_map.csv",
            ]
            stage = "build_wide_sql"
        elif normalized == "refine":
            task_id = "feature_refine"
            args = ["feature", "refine", "--project", project, "--run-id", run_arg, "--dry-run-sql"]
            outputs = ["feature_selection/stage_summary.json", "feature_selection/final_features.txt"]
            stage = "feature_refine"
        else:
            task_id = f"feature_{normalized}"
            args = ["feature", normalized, "--project", project, "--run-id", run_arg]
            outputs = []
            stage = "feature_refine"
        if stage in workflow_stages:
            step_ids = implemented_step_ids_for_stage(step_config, stage)
            tasks.append(
                _task(
                    task_id=task_id,
                    task_type="feature_selection",
                    depends_on=feature_dependency,
                    args=args,
                    outputs=outputs,
                    scenario_profile=scenario_profile,
                    step_ids=step_ids,
                    step_params=step_params_for(step_config, step_ids),
                )
            )
            feature_dependency = [task_id]
            feature_task_ids.append(task_id)

    train_dep = feature_task_ids[-1:] or sample_task_ids[-1:]
    train_task_ids: list[str] = []
    train_step_ids = implemented_step_ids_for_stage(step_config, "train_baseline")
    tuning_outputs = []
    if "llm_guided_tuning" in train_step_ids:
        tuning_outputs = [
            "tuning_context.json",
            "tuning_trials.csv",
            "tuning_summary.json",
            "best_params.json",
            "llm_tuning_decisions.md",
        ]
    if "train_baseline" in workflow_stages:
        for experiment in _experiments_from_metadata(metadata):
            name = experiment.get("name") if isinstance(experiment, dict) else str(experiment)
            task_id = f"train_{name}"
            train_task_ids.append(task_id)
            tasks.append(
                _task(
                    task_id=task_id,
                    task_type="train",
                    depends_on=train_dep,
                    args=["train", "--project", project, "--run-id", run_arg, "--experiment", name],
                    outputs=[
                        f"modeling/{name}/model.pkl",
                        f"modeling/{name}/prediction.parquet",
                        f"modeling/{name}/train_metrics.json",
                    ]
                    + [f"modeling/{name}/{output}" for output in tuning_outputs],
                    scenario_profile=scenario_profile,
                    step_ids=train_step_ids,
                    step_params=step_params_for(step_config, train_step_ids),
                )
            )

    evaluate_task_added = False
    evaluate_step_ids = implemented_step_ids_for_stage(step_config, "evaluate")
    if "evaluate" in workflow_stages:
        tasks.append(
            _task(
                task_id="evaluate_final",
                task_type="evaluate",
                depends_on=train_task_ids,
                args=["evaluate", "--project", project, "--run-id", run_arg],
                outputs=["evaluation/evaluation_summary.json"],
                scenario_profile=scenario_profile,
                step_ids=evaluate_step_ids,
                step_params=step_params_for(step_config, evaluate_step_ids),
            )
        )
        evaluate_task_added = True

    champions = _as_list((metadata.get("evaluation") or {}).get("champions"))
    if not champions:
        champions = _project_champions(project_path)
    compare_args = ["compare", "--project", project, "--run-id", run_arg]
    for champion in champions:
        compare_args.extend(["--champion", str(champion)])
    compare_step_ids = implemented_step_ids_for_stage(step_config, "compare")
    compare_task_added = False
    if "compare" in workflow_stages:
        compare_dep = ["evaluate_final"] if evaluate_task_added else (train_task_ids or feature_task_ids[-1:] or sample_task_ids[-1:])
        tasks.append(
            _task(
                task_id="compare_final",
                task_type="compare",
                depends_on=compare_dep,
                args=compare_args,
                outputs=["evaluation/champion_challenger.json"],
                scenario_profile=scenario_profile,
                step_ids=compare_step_ids,
                step_params=step_params_for(step_config, compare_step_ids),
            )
        )
        compare_task_added = True

    report_step_ids = implemented_step_ids_for_stage(step_config, "report")
    if "report" in workflow_stages:
        report_dep = (
            ["compare_final"]
            if compare_task_added
            else ["evaluate_final"]
            if evaluate_task_added
            else train_task_ids or feature_task_ids[-1:] or sample_task_ids[-1:]
        )
        tasks.append(
            _task(
                task_id="report_final",
                task_type="report",
                depends_on=report_dep,
                args=["report", "--project", project, "--run-id", run_arg],
                outputs=["reports/model_report.md", "reports/model_report.html", "reports/model_card.md", "reports/executive_summary.md"],
                scenario_profile=scenario_profile,
                step_ids=report_step_ids,
                step_params=step_params_for(step_config, report_step_ids),
            )
        )

    return {
        "version": 1,
        "plan_id": f"{request_id}_plan",
        "request_id": request_id,
        "request_path": request_doc["path"],
        "project": project,
        "workflow": workflow,
        "scenario_profile": scenario_profile,
        "stage_steps": workflow_stage_steps,
        "step_params": workflow_step_params,
        "resolved_steps": workflow_resolved_steps,
        "planned_steps": workflow_planned_steps,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id_placeholder": run_arg,
        "tasks": tasks,
        "improvement_log": "audit/improvement_candidates.md",
    }


def save_execution_plan(plan: dict[str, Any], path: str | Path) -> Path:
    """Write an execution plan YAML file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.dump(plan, handle, Dumper=_NoAliasDumper, allow_unicode=True, sort_keys=False)
    return output_path
