"""Report-generation application handler."""

from __future__ import annotations

import json
import shutil
from copy import deepcopy
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
    stage_action_failed,
    stage_action_started,
)
from risk_model_workbench.paths import stage_config_path
from risk_model_workbench.state import append_decision, load_run_state


def _runtime_config(context: VersionContext, name: str) -> dict[str, Any]:
    for candidate in [
        context.runtime_config_dir / f"{name}.yaml",
        context.runtime_config_dir / f"{name}.yml",
    ]:
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


def _write_text(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def _last_result(context: VersionContext) -> ActionResult:
    payload = dict(load_run_state(context.workspace)["stages"]["report"]["last_result"])
    allowed = set(ActionResult.__dataclass_fields__)
    return ActionResult(**{key: value for key, value in payload.items() if key in allowed})


def _root(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("report")
    return value if isinstance(value, dict) else {}


def _score_labels(config: dict[str, Any], target: dict[str, Any] | None = None) -> dict[str, str]:
    labels: dict[str, str] = {}
    for source in [config, _root(config), target or {}]:
        display_name = source.get("model_display_name") or source.get("model_name")
        if display_name:
            labels["model_score"] = str(display_name)
        raw_labels = source.get("score_labels") or {}
        if isinstance(raw_labels, dict):
            labels.update({str(key): str(value) for key, value in raw_labels.items() if str(value)})
    return labels


def _score_columns(
    config: dict[str, Any], evaluate_config: dict[str, Any], target: dict[str, Any] | None = None
) -> list[str]:
    eval_root = evaluate_config.get("evaluation") if isinstance(evaluate_config.get("evaluation"), dict) else {}
    raw = (target or {}).get("score_columns") or config.get("score_columns") or _root(config).get("score_columns") or eval_root.get("score_columns")
    return _string_list(raw) or ["model_score"]


def _outputs(config: dict[str, Any], target: dict[str, Any] | None = None) -> list[str]:
    raw = (target or {}).get("outputs") or config.get("outputs") or _root(config).get("outputs")
    return _string_list(raw) or ["model_report.md", "model_report.html", "model_card.md", "executive_summary.md"]


def _configured_targets(config: dict[str, Any], selected: str | None = None) -> list[dict[str, Any]]:
    raw_targets = _root(config).get("targets") or config.get("targets") or []
    if not isinstance(raw_targets, list):
        return []
    targets: list[dict[str, Any]] = []
    for index, item in enumerate(raw_targets, start=1):
        if not isinstance(item, dict):
            continue
        target = deepcopy(item)
        name = str(target.get("name") or target.get("experiment") or f"target_{index}").strip()
        if not name or (selected and name != selected):
            continue
        target["name"] = name
        if "score_labels" not in target:
            inherited = _score_labels(config)
            if inherited:
                target["score_labels"] = inherited
        targets.append(target)
    return targets


def _resolve(workspace: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else workspace / path


def _generation_config(
    config: dict[str, Any], evaluate_config: dict[str, Any], target: dict[str, Any]
) -> dict[str, Any]:
    generated = deepcopy(config)
    root = deepcopy(_root(generated))
    for key in ["model_display_name", "model_name", "title"]:
        if target.get(key):
            root[key] = target[key]
            generated[key] = target[key]
    labels = _score_labels(generated, target)
    columns = _score_columns(generated, evaluate_config, target)
    outputs = _outputs(generated, target)
    for key, value in [("score_labels", labels), ("score_columns", columns), ("outputs", outputs)]:
        if value:
            root[key] = value
            generated[key] = value
    if root:
        generated["report"] = root
    return generated


def _target_paths(
    workspace: Path, target: dict[str, Any], fallback_train_dir: Path | None
) -> tuple[Path, Path, Path]:
    if target.get("train_dir"):
        train_dir = _resolve(workspace, target["train_dir"])
    elif target.get("experiment"):
        train_dir = workspace / "modeling" / str(target["experiment"])
    else:
        train_dir = fallback_train_dir or workspace / "modeling" / "main_lgbm"
    if target.get("eval_dir"):
        eval_dir = _resolve(workspace, target["eval_dir"])
    elif target.get("experiment") and (workspace / "evaluation_tuned" / str(target["experiment"])).exists():
        eval_dir = workspace / "evaluation_tuned" / str(target["experiment"])
    else:
        eval_dir = workspace / "evaluation"
    if target.get("output_dir"):
        output_dir = _resolve(workspace, target["output_dir"])
    else:
        name = str(target.get("name") or "").strip()
        output_dir = workspace / (f"reports_{name}" if name and name != "default" else "reports")
    return train_dir, eval_dir, output_dir


def _copy_woe(source_dir: Path, target_dir: Path) -> None:
    if not source_dir.exists():
        return
    for source in sorted([*source_dir.glob("woe_top*_summary.csv"), *source_dir.glob("images/*.png")]):
        target = target_dir / source.relative_to(source_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _register_woe(context: VersionContext, artifact_dir: Path) -> None:
    if artifact_dir.exists():
        for artifact in sorted([*artifact_dir.glob("woe_top*_summary.csv"), *artifact_dir.glob("images/*.png")]):
            register_action_artifact(context.workspace, "report", artifact)


def _relative(workspace: Path, path: Path) -> str:
    try:
        return path.relative_to(workspace).as_posix()
    except ValueError:
        return path.as_posix()


def _generate_excel(
    context: VersionContext,
    target: dict[str, Any],
    report_config: dict[str, Any],
    evaluate_config: dict[str, Any],
    fallback_train_dir: Path | None,
) -> Path | None:
    from risk_model_workbench.reporting.excel_report import generate_excel_report

    train_dir, eval_dir, output_dir = _target_paths(context.workspace, target, fallback_train_dir)
    if not (train_dir / "metrics_train_valid.json").exists():
        append_decision(context.workspace, stage="report", decision="report_target_skipped", reason=f"{target.get('name')}: missing train metrics at {train_dir}")
        return None
    if not (eval_dir / "evaluation_summary.json").exists():
        append_decision(context.workspace, stage="report", decision="report_target_skipped", reason=f"{target.get('name')}: missing evaluation summary at {eval_dir}")
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_woe(train_dir / "woe_top_features", output_dir / "woe_top_features")
    excel_path = generate_excel_report(
        eval_dir=eval_dir,
        train_dir=train_dir,
        input_dir=_resolve(context.workspace, target.get("input_dir") or "modeling_input"),
        feature_dir=_resolve(context.workspace, target.get("feature_dir") or "feature_selection"),
        output_path=output_dir / "model_report.xlsx",
        project_dir=context.project_dir,
        report_config=_generation_config(report_config, evaluate_config, target),
    )
    scope = {
        "report_scope": target.get("description") or target.get("name") or "model report",
        "target": target.get("name"),
        "experiment": target.get("experiment"),
        "train_dir": _relative(context.workspace, train_dir),
        "eval_dir": _relative(context.workspace, eval_dir),
        "output_dir": _relative(context.workspace, output_dir),
        "outputs": ["model_report.html", "model_report.md", "model_report.xlsx", "model_report_missing_results.md"],
        "score_labels": target.get("score_labels"),
    }
    register_action_artifact(context.workspace, "report", _write_json(output_dir / "report_scope.json", scope))
    register_action_artifact(context.workspace, "report", excel_path)
    for sidecar in [excel_path.with_name("model_report.md"), excel_path.with_name("model_report.html"), excel_path.with_name("model_report_missing_results.md")]:
        if sidecar.exists():
            register_action_artifact(context.workspace, "report", sidecar)
    _register_woe(context, output_dir / "woe_top_features")
    append_decision(context.workspace, stage="report", decision="report_target_done", reason=f"{target.get('name')}: generated {excel_path}")
    return excel_path


def run_report(invocation: ActionInvocation, context: VersionContext, attempt_id: str) -> ActionResult:
    """Generate configured report artifacts from registered workspace evidence."""
    del attempt_id
    params = invocation.canonical_payload()["params"]
    assert isinstance(params, dict)
    selected_target = str(params.get("report_target") or "") or None
    with detached_action_attempt():
        stage_action_started(context.workspace, "report")
        report_config = _runtime_config(context, "report")
        evaluate_config = _runtime_config(context, "evaluate")
        root = _root(report_config)
        sections = report_config.get("sections") or root.get("sections") or []
        outputs = _outputs(report_config)
        steps = _string_list(root.get("stage_steps"))
        if "model_recovery_report" in steps and "model_recovery_report.md" not in outputs:
            outputs.append("model_recovery_report.md")
        if "credit_product_report" in steps and "credit_product_report.md" not in outputs:
            outputs.append("credit_product_report.md")
        state = load_run_state(context.workspace)
        manifest_path = context.manifest_path
        run_id = context.version_id
        model_report = (
            "# Model Report\n\nstatus: scaffold\n\n"
            "This report is generated only from registered run artifacts. Missing metrics are not fabricated.\n\n"
            f"- run_id: {run_id}\n- workflow: {state.get('workflow')}\n"
            f"- configured_sections: {', '.join(sections) if sections else 'not configured'}\n"
            f"- artifact_manifest: {manifest_path.relative_to(context.workspace)}\n"
        )

        def body(name: str) -> str:
            lowered = name.lower()
            if "model_card" in lowered:
                return "# Model Card\n\nstatus: scaffold\n\nThis card is generated from registered run artifacts.\n"
            if "executive" in lowered:
                return "# Executive Summary\n\nstatus: scaffold\n\nModel evaluation evidence is summarized from the run manifest when available.\n"
            if "recovery" in lowered:
                return "# Model Recovery Report\n\nstatus: scaffold\n\nRecovery monitoring inputs were requested; missing artifacts are listed in the run manifest.\n"
            if "credit" in lowered:
                return "# Credit Product Report\n\nstatus: scaffold\n\nCredit product evaluation outputs were requested; missing artifacts are listed in the run manifest.\n"
            return model_report

        generated: list[Path] = []
        if not selected_target:
            for output_name in outputs:
                target_path = context.workspace / "reports" / Path(output_name).name
                if target_path.suffix.lower() == ".xlsx":
                    continue
                if target_path.suffix.lower() == ".html":
                    from risk_model_workbench.reporting.html_report import render_model_report_html

                    _write_text(target_path, render_model_report_html(body(target_path.name), title="Model Report", run_id=run_id))
                elif target_path.suffix.lower() == ".json":
                    _write_json(target_path, {"status": "scaffold", "run_id": run_id, "sections": sections, "artifact_manifest": str(manifest_path.relative_to(context.workspace))})
                else:
                    _write_text(target_path, body(target_path.name))
                generated.append(target_path)
            for required_name in ["model_report.md", "model_report.html", "model_card.md", "executive_summary.md"]:
                target_path = context.workspace / "reports" / required_name
                if target_path.exists():
                    continue
                if target_path.suffix.lower() == ".html":
                    from risk_model_workbench.reporting.html_report import render_model_report_html

                    _write_text(target_path, render_model_report_html(body(required_name), title="Model Report", run_id=run_id))
                else:
                    _write_text(target_path, body(required_name))
                generated.append(target_path)
        for artifact in generated:
            register_action_artifact(context.workspace, "report", artifact)

        train_dirs = [item for item in (context.workspace / "modeling").glob("*") if item.is_dir() and (item / "metrics_train_valid.json").exists()]
        all_targets = _configured_targets(report_config)
        targets = _configured_targets(report_config, selected_target)
        if selected_target and all_targets and not targets:
            message = f"unknown report target: {selected_target}"
            stage_action_failed(context.workspace, "report", message)
            result = _last_result(context)
            return ActionResult(**{**result.to_dict(), "message": message})
        if not targets:
            targets = [{"name": selected_target or "default", "output_dir": "reports"}]

        excel_paths: list[Path] = []
        fallback = sorted(train_dirs)[0] if train_dirs else None
        for target in targets:
            try:
                excel_path = _generate_excel(context, target, report_config, evaluate_config, fallback)
                if excel_path is not None:
                    excel_paths.append(excel_path)
            except Exception as exc:
                append_decision(context.workspace, stage="report", decision="excel_scaffold", reason=f"{target.get('name')}: Excel report not generated: {exc}")
        if excel_paths:
            message = "report complete: " + ", ".join(str(item) for item in excel_paths)
            append_decision(context.workspace, stage="report", decision="done", reason=f"Excel report generated for {len(excel_paths)} report target(s)")
            stage_action_done(context.workspace, "report", message=message)
        else:
            message = f"report scaffold: {context.workspace / 'reports' / 'model_report.md'}"
            reason = "report generated with missing real evaluation artifacts"
            append_decision(context.workspace, stage="report", decision="scaffold", reason=reason)
            stage_action_done(context.workspace, "report", scaffold=True, message=message)
        return _last_result(context)
