"""Evaluation and champion/challenger application handlers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from risk_model_workbench.application.context import VersionContext
from risk_model_workbench.config import load_yaml
from risk_model_workbench.harness.invocation import ActionInvocation
from risk_model_workbench.harness.runtime import (
    ActionResult,
    detached_action_attempt,
    register_action_artifact,
    stage_action_done,
    stage_action_started,
)
from risk_model_workbench.progress import ProgressReporter
from risk_model_workbench.paths import stage_config_path
from risk_model_workbench.state import append_decision, load_run_state


def _runtime_config(context: VersionContext, name: str) -> dict[str, Any]:
    candidates = [
        context.runtime_config_dir / f"{name}.yaml",
        context.runtime_config_dir / f"{name}.yml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return load_yaml(candidate)
    project_config = stage_config_path(context.project_dir, name)
    return load_yaml(project_config) if project_config.exists() else {}


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def _last_result(context: VersionContext, stage: str) -> ActionResult:
    payload = dict(load_run_state(context.workspace)["stages"][stage]["last_result"])
    allowed = set(ActionResult.__dataclass_fields__)
    return ActionResult(**{key: value for key, value in payload.items() if key in allowed})


def _scores_path(context: VersionContext, explicit: object) -> Path:
    if explicit:
        path = Path(str(explicit))
    else:
        candidates = sorted((context.workspace / "modeling").glob("*/scores_all_splits.feather"))
        path = candidates[-1] if candidates else context.workspace / "modeling" / "scores_all_splits.feather"
    return path if path.is_absolute() else context.project_dir / path


def _output_path(context: VersionContext, explicit: object) -> Path:
    path = Path(str(explicit)) if explicit else context.workspace / "evaluation"
    return path if path.is_absolute() else context.project_dir / path


def run_evaluate(
    invocation: ActionInvocation, context: VersionContext, attempt_id: str
) -> ActionResult:
    """Evaluate local prediction scores or produce an explicit scaffold result."""
    del attempt_id
    params = invocation.canonical_payload()["params"]
    assert isinstance(params, dict)
    with detached_action_attempt():
        stage_action_started(context.workspace, "evaluate")
        reporter = ProgressReporter(context.workspace, "evaluate")
        evaluate_config = _runtime_config(context, "evaluate")
        metrics = evaluate_config.get("metrics") or (evaluate_config.get("evaluation") or {}).get("metrics") or []
        scores_feather = _scores_path(context, params.get("scores_feather"))
        output_dir = _output_path(context, params.get("output_dir"))
        if scores_feather.exists() and evaluate_config:
            try:
                from risk_model_workbench.evaluation.run import evaluate_scores_from_feather

                evaluate_scores_from_feather(
                    scores_feather=scores_feather,
                    output_dir=output_dir,
                    config=evaluate_config,
                    progress=reporter,
                )
                for artifact in sorted([*output_dir.glob("*.csv"), *output_dir.glob("*.json")]):
                    register_action_artifact(context.workspace, "evaluate", artifact)
                append_decision(
                    context.workspace,
                    stage="evaluate",
                    decision="done",
                    reason="Evaluation completed from local score feather",
                )
                stage_action_done(
                    context.workspace,
                    "evaluate",
                    message=f"evaluation complete: {output_dir / 'evaluation_summary.json'}",
                )
                return _last_result(context, "evaluate")
            except Exception as exc:
                payload = {
                    "status": "scaffold",
                    "reason": f"evaluation failed or dependency missing: {exc}",
                    "configured_metrics": metrics,
                }
        else:
            payload = {
                "status": "scaffold",
                "reason": "prediction data not available",
                "configured_metrics": metrics,
                "scores_feather": str(scores_feather),
                "scores_feather_exists": scores_feather.exists(),
            }
        reporter.emit(
            step="evaluate_scaffold",
            status="scaffold",
            message=f"模型评估未执行真实评估：{payload['reason']}",
            percent=100,
        )
        summary_path = _write_json(context.workspace / "evaluation" / "evaluation_summary.json", payload)
        register_action_artifact(context.workspace, "evaluate", summary_path)
        append_decision(context.workspace, stage="evaluate", decision="scaffold", reason=payload["reason"])
        stage_action_done(
            context.workspace,
            "evaluate",
            scaffold=True,
            message=f"evaluation scaffold: {summary_path}",
        )
        return _last_result(context, "evaluate")


def run_compare(
    invocation: ActionInvocation, context: VersionContext, attempt_id: str
) -> ActionResult:
    """Materialize a deterministic champion/challenger comparison summary."""
    del attempt_id
    params = invocation.canonical_payload()["params"]
    assert isinstance(params, dict)
    with detached_action_attempt():
        stage_action_started(context.workspace, "compare")
        champions = _string_list(params.get("champions"))
        if not champions:
            evaluate_config = _runtime_config(context, "evaluate")
            champions = [
                score
                for score in _string_list((evaluate_config.get("evaluation") or {}).get("score_columns"))
                if score != "model_score"
            ]
        champions = list(dict.fromkeys(champions))
        summary_path = context.workspace / "evaluation" / "champion_challenger.json"
        if not champions:
            payload = {"status": "skipped", "reason": "no champion configured", "champion": "", "champions": []}
            _write_json(summary_path, payload)
            register_action_artifact(context.workspace, "compare", summary_path)
            append_decision(context.workspace, stage="compare", decision="skipped", reason=payload["reason"])
            stage_action_done(
                context.workspace,
                "compare",
                message=f"compare skipped: {summary_path}",
            )
            return _last_result(context, "compare")

        benchmark_path = context.workspace / "evaluation" / "benchmark_uplift.csv"
        if benchmark_path.exists():
            payload = {
                "status": "done",
                "reason": "",
                "champion": champions[-1],
                "champions": champions,
                "benchmark_uplift": str(benchmark_path.relative_to(context.workspace)),
            }
            register_action_artifact(context.workspace, "compare", benchmark_path)
            decision, scaffold, label = "done", False, "complete"
        else:
            payload = {
                "status": "scaffold",
                "reason": "candidate and champion predictions not available",
                "champion": champions[-1],
                "champions": champions,
            }
            decision, scaffold, label = "scaffold", True, "scaffold"
        _write_json(summary_path, payload)
        register_action_artifact(context.workspace, "compare", summary_path)
        append_decision(
            context.workspace,
            stage="compare",
            decision=decision,
            reason=payload["reason"] or "Champion/challenger comparison materialized",
        )
        stage_action_done(
            context.workspace,
            "compare",
            scaffold=scaffold,
            message=f"compare {label}: {summary_path}",
        )
        return _last_result(context, "compare")
