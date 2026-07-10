"""Project continuity helpers for long-running modeling workspaces."""

from __future__ import annotations

import json
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import yaml

from risk_model_workbench.harness.errors import (
    ARTIFACT_CONTRACT_FAILED,
    DATA_MISSING,
    InvalidActionResultError,
    MissingActionResultError,
    SCAFFOLD_ONLY,
    SQL_APPROVAL_REQUIRED,
    UNKNOWN,
)
from risk_model_workbench.harness.runtime import load_action_result
from risk_model_workbench.paths import REPO_ROOT, project_config_path
from risk_model_workbench.request.training import llm_guided_tuning_enabled
from risk_model_workbench.rules import summarize_rules
from risk_model_workbench.run_evidence import load_run_evidence
from risk_model_workbench.state import load_run_state, run_dir
from risk_model_workbench.versioning import load_version_index, resolve_workspace_dir
from risk_model_workbench.workflow_contracts import artifact_exists, audit_contract_artifacts


PROJECT_STATE_VERSION = 1


def project_state_path(project_dir: str | Path) -> Path:
    return Path(project_dir) / "project_state.yml"


def load_project_state(project_dir: str | Path) -> dict[str, Any]:
    path = project_state_path(project_dir)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_project_state(project_dir: str | Path, state: dict[str, Any]) -> Path:
    path = project_state_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    state["version"] = PROJECT_STATE_VERSION
    state["updated_at"] = _now()
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(state, handle, allow_unicode=True, sort_keys=False)
    return path


def update_project_state(
    project_dir: str | Path,
    *,
    active_version_id: str | None = None,
    active_run_id: str | None = None,
    current_objective: str | None = None,
    status: str | None = None,
    next_actions: list[str] | None = None,
    blockers: list[str] | None = None,
    risks: list[str] | None = None,
    last_verified_commands: list[str] | None = None,
) -> dict[str, Any]:
    project_path = Path(project_dir)
    state = load_project_state(project_path)
    # Always reflect the current project location so the field self-heals when the
    # repo is moved/renamed (setdefault left a stale path after jingying_model_agent
    # was renamed to risk_model_workbench).
    state["project"] = str(project_path.resolve())
    if active_version_id is not None:
        state["active_version_id"] = active_version_id
    if active_run_id is not None:
        state["active_run_id"] = active_run_id
    if current_objective is not None:
        state["current_objective"] = current_objective
    if status is not None:
        state["status"] = status
    if next_actions:
        state["next_actions"] = _append_unique(state.get("next_actions", []), next_actions)
    if blockers:
        state["blockers"] = _append_unique(state.get("blockers", []), blockers)
    if risks:
        state["risks"] = _append_unique(state.get("risks", []), risks)
    if last_verified_commands is not None:
        state["last_verified_at"] = _now()
        state["last_verified_commands"] = last_verified_commands
    save_project_state(project_path, state)
    return state


def summarize_project(project_dir: str | Path, run_id: str | None = None, version_id: str | None = None) -> dict[str, Any]:
    project_path = Path(project_dir)
    persisted = load_project_state(project_path)
    selected_version_id = version_id or persisted.get("active_version_id") or _latest_version_id(project_path)
    selected_run_id = run_id or (None if selected_version_id else persisted.get("active_run_id") or _latest_run_id(project_path))
    project_info = _load_project_info(project_path)

    summary: dict[str, Any] = {
        "project": str(project_path.resolve()),
        "project_name": project_info.get("name") or project_path.name,
        "display_name": project_info.get("display_name") or project_path.name,
        "active_version_id": selected_version_id,
        "active_run_id": selected_run_id,
        "current_objective": persisted.get("current_objective", ""),
        "status": persisted.get("status", "not_started"),
        "last_verified_at": persisted.get("last_verified_at", ""),
        "last_verified_commands": persisted.get("last_verified_commands", []),
        "next_actions": list(persisted.get("next_actions", [])),
        "blockers": list(persisted.get("blockers", [])),
        "risks": list(persisted.get("risks", [])),
        "rules": summarize_rules(),
        "run": None,
        "version": None,
    }

    if not selected_version_id and not selected_run_id:
        summary["next_actions"] = summary["next_actions"] or ["Initialize a version or set active_version_id in project_state.yml."]
        return summary

    selected_id = selected_version_id or selected_run_id or ""
    selected_run_dir = resolve_workspace_dir(project_path, version_id=selected_version_id, run_id=selected_run_id)
    try:
        run_state = load_run_state(selected_run_dir)
    except FileNotFoundError:
        summary["status"] = "blocked"
        missing_key = "active_version_id" if selected_version_id else "active_run_id"
        summary["blockers"] = _append_unique(summary["blockers"], [f"{missing_key} does not exist: {selected_id}"])
        return summary

    stage_rows = _stage_rows(run_state)
    try:
        audit = audit_run(project_path, selected_id)
    except Exception:
        audit = {"verdict": "unknown", "stages": []}
    audit_issues = _append_unique([], [issue for item in audit.get("stages", []) for issue in item.get("issues", [])])
    inferred_next, inferred_blockers, inferred_risks = _infer_run_followups(run_state)
    summary["status"] = persisted.get("status") or run_state.get("status", "unknown")
    summary["next_actions"] = summary["next_actions"] or inferred_next
    summary["blockers"] = _append_unique(summary["blockers"], inferred_blockers)
    summary["risks"] = _append_unique(summary["risks"], inferred_risks)
    workspace_summary = {
        "version_id": run_state.get("version_id", selected_version_id or ""),
        "run_id": run_state.get("run_id", selected_run_id or ""),
        "workflow": run_state.get("workflow", ""),
        "status": run_state.get("status", ""),
        "current_stage": run_state.get("current_stage", ""),
        "updated_at": run_state.get("updated_at", ""),
        "stage_counts": _stage_counts(stage_rows),
        "stages": stage_rows,
        "recent_decisions": list(run_state.get("decisions", []))[-5:],
        "audit_verdict": audit.get("verdict", ""),
        "audit_top_issues": audit_issues[:5],
    }
    summary["version"] = workspace_summary if selected_version_id else None
    summary["run"] = workspace_summary
    return summary


def write_project_state_from_summary(project_dir: str | Path, summary: dict[str, Any], commands: list[str]) -> Path:
    project_path = Path(project_dir)
    state = load_project_state(project_path)
    # Always reflect the current project location so the field self-heals when the
    # repo is moved/renamed (setdefault left a stale path after jingying_model_agent
    # was renamed to risk_model_workbench).
    state["project"] = str(project_path.resolve())
    if summary.get("active_version_id"):
        state["active_version_id"] = summary["active_version_id"]
    if summary.get("active_run_id"):
        state["active_run_id"] = summary["active_run_id"]
    state.setdefault("current_objective", summary.get("current_objective", ""))
    state["status"] = summary.get("status", state.get("status", "unknown"))
    state.setdefault("next_actions", summary.get("next_actions", []))
    state["blockers"] = _append_unique(state.get("blockers", []), summary.get("blockers", []))
    state["risks"] = _append_unique(state.get("risks", []), summary.get("risks", []))
    state["last_verified_at"] = _now()
    state["last_verified_commands"] = commands
    return save_project_state(project_path, state)


def format_project_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"project: {summary.get('display_name') or summary.get('project_name')}",
        f"path: {summary.get('project')}",
        f"status: {summary.get('status')}",
        f"active_version_id: {summary.get('active_version_id') or ''}",
        f"active_run_id: {summary.get('active_run_id') or ''}",
    ]
    if summary.get("current_objective"):
        lines.append(f"current_objective: {summary['current_objective']}")
    if summary.get("last_verified_at"):
        lines.append(f"last_verified_at: {summary['last_verified_at']}")

    run = summary.get("run")
    if run:
        counts = ", ".join(f"{key}={value}" for key, value in sorted(run.get("stage_counts", {}).items()))
        label = "version" if summary.get("active_version_id") else "run"
        lines.extend(
            [
                "",
                f"{label}:",
                f"  version_id: {run.get('version_id') or ''}",
                f"  workflow: {run.get('workflow')}",
                f"  status: {run.get('status')}",
                f"  current_stage: {run.get('current_stage')}",
                f"  stage_counts: {counts}",
                f"  audit_verdict: {run.get('audit_verdict') or ''}",
            ]
        )
        _extend_list(lines, "audit_top_issues", run.get("audit_top_issues", []))

    rules = summary.get("rules") or {}
    if rules:
        lines.extend(
            [
                "",
                "rules:",
                f"  proposed: {rules.get('proposed_count', 0)}",
                f"  unenforced_guardrails: {rules.get('unenforced_guardrail_count', 0)}",
            ]
        )
        _extend_list(lines, "proposed_rules", [f"{item.get('id')}: {item.get('title')}" for item in rules.get("proposed_rules", [])[:5]])

    _extend_list(lines, "next_actions", summary.get("next_actions", []))
    _extend_list(lines, "blockers", summary.get("blockers", []))
    _extend_list(lines, "risks", summary.get("risks", []))
    return "\n".join(lines) + "\n"


def write_handoff(
    project_dir: str | Path,
    *,
    run_id: str | None = None,
    note: str = "",
    output: str | Path | None = None,
    context_snapshot: str | Path | None = None,
) -> Path:
    project_path = Path(project_dir)
    summary = summarize_project(project_path, run_id=run_id)
    selected_id = summary.get("active_version_id") or summary.get("active_run_id") or "no-version"
    if output is None:
        output_path = project_path / "handoffs" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}-{selected_id}.md"
    else:
        output_path = Path(output)
        if not output_path.is_absolute():
            output_path = project_path / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        format_handoff(summary, note=note, context_snapshot=_resolve_context_snapshot_ref(project_path, summary, context_snapshot)),
        encoding="utf-8",
    )

    state = update_project_state(
        project_path,
        active_version_id=summary.get("active_version_id"),
        active_run_id=summary.get("active_run_id"),
        status=summary.get("status"),
        next_actions=summary.get("next_actions", []),
        blockers=summary.get("blockers", []),
        risks=summary.get("risks", []),
    )
    state["last_handoff"] = str(output_path)
    save_project_state(project_path, state)
    return output_path


def format_handoff(summary: dict[str, Any], *, note: str = "", context_snapshot: str | None = None) -> str:
    run = summary.get("run") or {}
    lines = [
        f"# Handoff - {summary.get('display_name') or summary.get('project_name')}",
        "",
        f"- generated_at: {_now()}",
        f"- project: {summary.get('project')}",
        f"- active_version_id: {summary.get('active_version_id') or ''}",
        f"- active_run_id: {summary.get('active_run_id') or ''}",
        f"- status: {summary.get('status')}",
    ]
    if summary.get("current_objective"):
        lines.append(f"- current_objective: {summary['current_objective']}")
    if note:
        lines.extend(["", "## Note", "", note])

    lines.extend(["", "## Source Of Truth", ""])
    lines.append("- project_state.yml")
    if summary.get("active_version_id"):
        version_id = summary["active_version_id"]
        lines.append(f"- versions/{version_id}/version_state.yml")
        lines.append(f"- versions/{version_id}/audit/artifact_manifest.json")
    elif summary.get("active_run_id"):
        lines.append(f"- runs/{summary['active_run_id']}/run_state.yml")
        lines.append(f"- runs/{summary['active_run_id']}/audit/artifact_manifest.json")
    if context_snapshot:
        lines.append(f"- {context_snapshot}")

    _extend_markdown_list(lines, "Next Actions", summary.get("next_actions", []))
    _extend_markdown_list(lines, "Blockers", summary.get("blockers", []))
    _extend_markdown_list(lines, "Risks", summary.get("risks", []))

    if run:
        lines.extend(["", "## Run Summary", ""])
        lines.append(f"- workflow: {run.get('workflow')}")
        lines.append(f"- run_status: {run.get('status')}")
        lines.append(f"- current_stage: {run.get('current_stage')}")
        lines.extend(["", "| Stage | Status | Artifacts |", "| --- | --- | ---: |"])
        for stage in run.get("stages", []):
            lines.append(f"| {stage['name']} | {stage['status']} | {stage['artifact_count']} |")
        decisions = run.get("recent_decisions", [])
        if decisions:
            lines.extend(["", "## Recent Decisions", ""])
            for decision in decisions:
                lines.append(
                    f"- {decision.get('created_at', '')} [{decision.get('stage', '')}] "
                    f"{decision.get('decision', '')}: {decision.get('reason', '')}"
                )
    lines.append("")
    return "\n".join(lines)


def append_lesson(
    project_dir: str | Path,
    *,
    title: str,
    body: str,
    kind: str,
    scope: str = "project",
    source: str = "",
    tags: list[str] | None = None,
) -> Path:
    if not body.strip():
        raise ValueError("lesson body cannot be empty")
    path = _lesson_path(project_dir, scope)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("# Lessons\n\n", encoding="utf-8")

    tags_text = ", ".join(tags or [])
    entry = (
        f"\n## {title}\n\n"
        f"- captured_at: {_now()}\n"
        f"- kind: {kind}\n"
        f"- scope: {scope}\n"
        f"- source: {source or 'manual'}\n"
        f"- tags: {tags_text}\n\n"
        f"{body.strip()}\n"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(entry)
    return path


def audit_run(project_dir: str | Path, run_id: str, *, stage: str | None = None) -> dict[str, Any]:
    evidence = load_run_evidence(project_dir, run_id)
    project_path = evidence.project_path
    run_state = evidence.run_state
    contract_source = evidence.contract_source
    workspace_rel = evidence.run_path.relative_to(project_path)
    state_filename = "version_state.yml" if (evidence.run_path / "version_state.yml").exists() else "run_state.yml"

    stage_states = run_state.get("stages") or {}
    selected_names = [stage] if stage else list(stage_states.keys())
    stage_results = []
    for name in selected_names:
        state = stage_states.get(name)
        if state is None:
            stage_results.append(
                {
                    "stage": name,
                    "status": "missing",
                    "verdict": "missing",
                    "artifact_count": 0,
                    "registered_count": 0,
                    "issues": [f"stage is not present in run_state.yml: {name}"],
                    "contract_source": contract_source,
                }
            )
            continue
        stage_results.append(
            _audit_stage(
                name,
                state,
                evidence.manifest_by_stage.get(name, []),
                run_state,
                evidence.run_path,
                evidence.stage_contracts.get(name, {}),
                contract_source,
            )
        )
    agent_result = _audit_agent_runtime(evidence)
    if agent_result is not None:
        stage_results.append(agent_result)

    verdict = _rollup_audit_verdict(stage_results)
    source_of_truth = [
        str(workspace_rel / state_filename),
        str(workspace_rel / "audit" / "artifact_manifest.json"),
    ]
    if contract_source:
        source_of_truth.append(contract_source)
    if agent_result is not None:
        source_of_truth.extend(
            [
                str(workspace_rel / "agent_plan.yml"),
                str(workspace_rel / "audit" / "agent_state.yml"),
                str(workspace_rel / "audit" / "agent_trace.jsonl"),
            ]
        )
    return {
        "project": str(project_path.resolve()),
        "version_id": run_state.get("version_id", ""),
        "run_id": run_state.get("run_id", run_id),
        "workflow": run_state.get("workflow", ""),
        "run_status": run_state.get("status", ""),
        "version_status": run_state.get("status", ""),
        "stage": stage or "",
        "verdict": verdict,
        "source_of_truth": source_of_truth,
        "contract_source": contract_source,
        "stages": stage_results,
    }


def format_run_audit(audit: dict[str, Any]) -> str:
    lines = [
        f"version_id: {audit.get('version_id') or ''}",
        f"run_id: {audit.get('run_id')}",
        f"workflow: {audit.get('workflow')}",
        f"run_status: {audit.get('run_status')}",
        f"verdict: {audit.get('verdict')}",
        f"contract_source: {audit.get('contract_source') or ''}",
        "",
        "stages:",
    ]
    for stage in audit.get("stages", []):
        code = f", failure_code={stage.get('failure_code')}" if stage.get("failure_code") else ""
        lines.append(
            f"  - {stage['stage']}: {stage['verdict']} "
            f"(status={stage['status']}, artifacts={stage['artifact_count']}, registered={stage['registered_count']}{code})"
        )
        for issue in stage.get("issues", []):
            lines.append(f"    issue: {issue}")
    return "\n".join(lines) + "\n"


def write_retrospective(
    project_dir: str | Path,
    *,
    run_id: str | None = None,
    scope: str = "session",
    stage: str | None = None,
    outcome: str = "",
    note: str = "",
    lessons: list[str] | None = None,
    output: str | Path | None = None,
) -> Path:
    if scope == "stage" and not stage:
        raise ValueError("stage retrospective requires --stage")

    project_path = Path(project_dir)
    summary = summarize_project(project_path, run_id=run_id)
    selected_id = run_id or summary.get("active_version_id") or summary.get("active_run_id")
    audit = audit_run(project_path, selected_id, stage=stage) if selected_id else None
    if output is None:
        suffix = stage if scope == "stage" else (selected_id or "no-version")
        output_path = project_path / "retrospectives" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}-{scope}-{suffix}.md"
    else:
        output_path = Path(output)
        if not output_path.is_absolute():
            output_path = project_path / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        format_retrospective(summary, audit=audit, scope=scope, stage=stage, outcome=outcome, note=note, lessons=lessons or []),
        encoding="utf-8",
    )

    state = update_project_state(
        project_path,
        active_version_id=summary.get("active_version_id"),
        active_run_id=summary.get("active_run_id"),
        status=summary.get("status"),
        next_actions=summary.get("next_actions", []),
        blockers=summary.get("blockers", []),
        risks=summary.get("risks", []),
    )
    state["last_retrospective"] = str(output_path)
    save_project_state(project_path, state)
    return output_path


def format_retrospective(
    summary: dict[str, Any],
    *,
    audit: dict[str, Any] | None,
    scope: str,
    stage: str | None,
    outcome: str,
    note: str,
    lessons: list[str],
) -> str:
    lines = [
        f"# Retrospective - {summary.get('display_name') or summary.get('project_name')}",
        "",
        f"- generated_at: {_now()}",
        "- trigger: explicit",
        f"- scope: {scope}",
        f"- project: {summary.get('project')}",
        f"- active_version_id: {summary.get('active_version_id') or ''}",
        f"- active_run_id: {summary.get('active_run_id') or ''}",
    ]
    if stage:
        lines.append(f"- stage: {stage}")
    if outcome:
        lines.append(f"- outcome: {outcome}")
    if note:
        lines.extend(["", "## Note", "", note])

    lines.extend(["", "## Source Of Truth", "", "- project_state.yml"])
    if summary.get("active_version_id"):
        version_id = summary["active_version_id"]
        lines.append(f"- versions/{version_id}/version_state.yml")
        lines.append(f"- versions/{version_id}/audit/artifact_manifest.json")
    elif summary.get("active_run_id"):
        lines.append(f"- runs/{summary['active_run_id']}/run_state.yml")
        lines.append(f"- runs/{summary['active_run_id']}/audit/artifact_manifest.json")

    if audit:
        lines.extend(["", "## Audit", "", f"- verdict: {audit.get('verdict')}"])
        lines.extend(["", "| Stage | Status | Verdict | Artifacts | Registered |", "| --- | --- | --- | ---: | ---: |"])
        for item in audit.get("stages", []):
            lines.append(
                f"| {item['stage']} | {item['status']} | {item['verdict']} | "
                f"{item['artifact_count']} | {item['registered_count']} |"
            )
        issues = _append_unique([], [issue for item in audit.get("stages", []) for issue in item.get("issues", [])])
        _extend_markdown_list(lines, "Audit Issues", issues)

    _extend_markdown_list(lines, "Next Actions", summary.get("next_actions", []))
    _extend_markdown_list(lines, "Risks", summary.get("risks", []))
    _extend_markdown_list(lines, "Lessons", lessons)
    lines.append("")
    return "\n".join(lines)


def _lesson_path(project_dir: str | Path, scope: str) -> Path:
    if scope == "project":
        return Path(project_dir) / "docs" / "lessons.md"
    if scope == "workbench":
        return REPO_ROOT / "docs" / "workbench_lessons.md"
    raise ValueError(f"unknown lesson scope: {scope}")


def _resolve_context_snapshot_ref(
    project_dir: Path,
    summary: dict[str, Any],
    context_snapshot: str | Path | None,
) -> str | None:
    if not context_snapshot:
        return None
    selected_id = summary.get("active_version_id") or summary.get("active_run_id")
    raw = str(context_snapshot)
    if raw == "auto":
        if not selected_id:
            return None
        workspace = resolve_workspace_dir(
            project_dir,
            version_id=summary.get("active_version_id"),
            run_id=summary.get("active_run_id"),
        )
        path = workspace / "audit" / "context_snapshot.json"
    else:
        path = Path(context_snapshot)
        if not path.is_absolute():
            path = project_dir / path
    try:
        return str(path.resolve().relative_to(project_dir.resolve()))
    except ValueError:
        return str(path.resolve())


def _audit_stage(
    name: str,
    state: dict[str, Any],
    manifest_items: list[dict[str, Any]],
    run_state: dict[str, Any],
    run_path: str | Path,
    contract: dict[str, Any],
    contract_source: str,
) -> dict[str, Any]:
    status = state.get("status", "unknown")
    artifacts = state.get("artifacts") or []
    registered_paths = {str(item.get("path")) for item in manifest_items}
    missing_registrations = [artifact for artifact in artifacts if str(artifact) not in registered_paths]
    missing_files = [
        str(item.get("path"))
        for item in manifest_items
        if not artifact_exists(run_path, item)
    ]
    scaffold_sources = [item for item in manifest_items if item.get("source") == "scaffold"]
    imported_sources = [item for item in manifest_items if item.get("source") == "imported"]
    allow_scaffold = bool(contract.get("allow_scaffold", False))
    allow_imported = bool(contract.get("allow_imported", False))
    closure_required = bool(contract.get("closure_required", True))
    issues: list[str] = []
    issues.extend(f"artifact listed in run_state.yml but not registered: {item}" for item in missing_registrations)
    issues.extend(f"registered artifact does not exist: {item}" for item in missing_files)
    if status in {"done", "scaffold"}:
        issues.extend(audit_contract_artifacts(contract, manifest_items, run_path))
        if name == "train_baseline":
            issues.extend(_audit_tuning_artifacts(manifest_items, run_path))

    if status in {"pending", "running", "failed", "missing"}:
        verdict = "open"
    elif (status == "scaffold" or scaffold_sources) and not allow_scaffold:
        verdict = "scaffold"
        issues.append("scaffold evidence is not real modeling evidence")
    elif status == "done" and closure_required and not artifacts and not manifest_items:
        verdict = "incomplete"
        issues.append("done stage has no registered artifacts")
    elif issues:
        verdict = "incomplete"
    elif (
        status == "done"
        and (imported_sources or str(run_state.get("workflow", "")).startswith("imported"))
        and not allow_imported
    ):
        verdict = "imported"
        issues.append("imported evidence should be reviewed before treating the stage as locally reproduced")
    elif status in {"done", "scaffold"}:
        verdict = "complete"
    else:
        verdict = "unknown"

    failure_codes = _classify_stage_failure(status=status, verdict=verdict, issues=issues)
    return {
        "stage": name,
        "status": status,
        "verdict": verdict,
        "artifact_count": len(artifacts),
        "registered_count": len(manifest_items),
        "contract_source": contract_source,
        "failure_code": failure_codes[0] if failure_codes else "",
        "failure_codes": failure_codes,
        "issues": issues,
    }


def _audit_agent_runtime(evidence: RunEvidence) -> dict[str, Any] | None:
    if str(evidence.run_state.get("managed_by") or "workbench") != "agent":
        return None
    run_path = Path(evidence.run_path)
    issues: list[str] = []
    required_files = {
        "agent_plan.yml": run_path / "agent_plan.yml",
        "audit/agent_state.yml": run_path / "audit" / "agent_state.yml",
        "audit/agent_trace.jsonl": run_path / "audit" / "agent_trace.jsonl",
    }
    for label, path in required_files.items():
        if not path.exists():
            issues.append(f"{label} missing for Agent-managed version")

    agent_state: dict[str, Any] = {}
    state_path = required_files["audit/agent_state.yml"]
    if state_path.exists():
        try:
            loaded = yaml.safe_load(state_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                agent_state = loaded
            else:
                issues.append("audit/agent_state.yml is not an object")
        except (OSError, yaml.YAMLError) as exc:
            issues.append(f"audit/agent_state.yml unreadable: {exc}")
    status = str(agent_state.get("status") or "")
    tasks = [task for task in agent_state.get("tasks", []) or [] if isinstance(task, dict)]
    if state_path.exists() and not agent_state:
        issues.append("audit/agent_state.yml is empty or malformed")
    if agent_state and not status:
        issues.append("agent status is missing")
    if agent_state and not tasks:
        issues.append("agent tasks are missing")
    all_tasks_closed = bool(tasks) and all(str(task.get("status") or "") in {"done", "skipped"} for task in tasks)
    if agent_state and status != "done" and not (status == "running" and all_tasks_closed):
        issues.append(f"agent status is not terminal complete: {status}")

    for task in tasks:
        task_id = str(task.get("task_id") or "")
        task_status = str(task.get("status") or "")
        if task_status == "scaffold":
            issues.append(f"task is scaffold evidence, not real closure: {task_id}")
        if task_status not in {"done", "skipped"}:
            issues.append(f"task is not closed: {task_id} status={task_status}")
        attempt_id = str(task.get("attempt_id") or "")
        if task_status == "done":
            if not attempt_id:
                issues.append(f"action result receipt missing for task: {task_id}")
                continue
            try:
                load_action_result(
                    run_path,
                    attempt_id,
                    task_id=task_id,
                    action_id=str(task.get("action_id") or ""),
                    invocation_hash=str(task.get("invocation_hash") or ""),
                    project=str(agent_state.get("project") or ""),
                    version_id=str(agent_state.get("version_id") or ""),
                )
            except (MissingActionResultError, InvalidActionResultError) as exc:
                issues.append(f"action result receipt invalid for task: {task_id}: {exc}")

    approvals_path = run_path / "audit" / "approvals.yml"
    if approvals_path.exists():
        try:
            approvals = yaml.safe_load(approvals_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            approvals = {}
            issues.append(f"audit/approvals.yml unreadable: {exc}")
        for approval in approvals.get("approvals", []) or []:
            if not isinstance(approval, dict):
                continue
            approval_id = str(approval.get("approval_id") or "")
            approval_status = str(approval.get("status") or "")
            if approval_status in {"pending", "approved"}:
                issues.append(f"approval is not consumed: {approval_id} status={approval_status}")
            if approval_status == "consumed" and not (run_path / "audit" / "approval_consumptions" / f"{approval_id}.json").exists():
                issues.append(f"approval consumption receipt missing: {approval_id}")

    advisor_dir = run_path / "audit" / "advisor_requests"
    if advisor_dir.exists():
        for request_path in sorted(advisor_dir.glob("*.json")):
            try:
                request = json.loads(request_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                issues.append(f"Advisor request unreadable: {request_path.name}: {exc}")
                continue
            request_id = str(request.get("request_id") or request_path.stem)
            request_status = str(request.get("status") or "")
            if request_status in {"answered", "waiting_for_user"}:
                issues.append(f"Advisor response is accepted but not consumed: {request_id}")
            elif request_status in {"pending", "rejected"}:
                issues.append(f"Advisor request is unresolved: {request_id} status={request_status}")
            elif request_status == "consumed":
                receipt_ref = str(request.get("consumption_receipt") or "")
                receipt_path = run_path / receipt_ref if receipt_ref else run_path / "audit" / "advisor_consumptions" / f"{request_id}.json"
                if not receipt_path.exists():
                    issues.append(f"Advisor consumption receipt missing: {request_id}")

    verdict = "incomplete" if issues else "complete"
    failure_codes = _classify_stage_failure(status="done" if not issues else "missing", verdict=verdict, issues=issues)
    return {
        "stage": "agent_runtime",
        "status": status or "missing",
        "verdict": verdict,
        "artifact_count": 0,
        "registered_count": 0,
        "contract_source": "agent_runtime",
        "failure_code": failure_codes[0] if failure_codes else "",
        "failure_codes": failure_codes,
        "issues": issues,
    }


def _audit_tuning_artifacts(manifest_items: list[dict[str, Any]], run_path: str | Path) -> list[str]:
    if not _runtime_train_uses_llm_tuning(run_path):
        return []
    required_patterns = [
        "modeling/*/tuning_summary.json",
        "modeling/*/tuning_trials.csv",
        "modeling/*/best_params.json",
        "modeling/*/llm_tuning_decisions.md",
    ]
    issues = []
    for pattern in required_patterns:
        if not _manifest_pattern_satisfied(pattern, manifest_items, run_path):
            issues.append(f"llm tuning artifact missing or not registered: {pattern}")
    return issues


def _runtime_train_uses_llm_tuning(run_path: str | Path) -> bool:
    train_cfg = _load_runtime_train_config(run_path)
    training = train_cfg.get("training") if isinstance(train_cfg.get("training"), dict) else {}
    return llm_guided_tuning_enabled(training)


def _load_runtime_train_config(run_path: str | Path) -> dict[str, Any]:
    for name in ["train.yaml", "train.yml"]:
        path = Path(run_path) / "configs_runtime" / name
        if path.exists():
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError):
                return {}
            return data if isinstance(data, dict) else {}
    return {}


def _manifest_pattern_satisfied(pattern: str, manifest_items: list[dict[str, Any]], run_path: str | Path) -> bool:
    matches = [item for item in manifest_items if fnmatch(str(item.get("path", "")), pattern)]
    return any(artifact_exists(run_path, item) for item in matches)


def _classify_stage_failure(*, status: str, verdict: str, issues: list[str]) -> list[str]:
    codes: list[str] = []
    issue_text = "\n".join(issues).lower()
    if "sql approval" in issue_text or "waiting for approval" in issue_text:
        codes.append(SQL_APPROVAL_REQUIRED)
    if (
        "contract required artifact" in issue_text
        or "no accepted artifact set" in issue_text
        or "stage_contract" in issue_text
    ):
        codes.append(ARTIFACT_CONTRACT_FAILED)
    if status == "scaffold" or verdict == "scaffold" or "scaffold" in issue_text:
        codes.append(SCAFFOLD_ONLY)
    if (
        status in {"pending", "running", "missing"}
        or "not registered" in issue_text
        or "does not exist" in issue_text
        or "no registered artifacts" in issue_text
        or "not present in run_state" in issue_text
    ):
        codes.append(DATA_MISSING)
    if status == "failed" and not codes:
        codes.append(UNKNOWN)
    if verdict == "imported" and not codes:
        codes.append(UNKNOWN)
    if verdict == "unknown" and not codes:
        codes.append(UNKNOWN)
    return _append_unique([], codes)


def _rollup_audit_verdict(stage_results: list[dict[str, Any]]) -> str:
    verdicts = {item.get("verdict") for item in stage_results}
    if not stage_results:
        return "empty"
    if verdicts <= {"complete"}:
        return "complete"
    if "open" in verdicts or "missing" in verdicts:
        return "open"
    if "incomplete" in verdicts:
        return "incomplete"
    if "scaffold" in verdicts:
        return "scaffold"
    if "imported" in verdicts:
        return "imported"
    return "unknown"


def _latest_version_id(project_dir: Path) -> str | None:
    index = load_version_index(project_dir)
    active = index.get("active_version_id")
    if active:
        return str(active)
    candidates: list[tuple[datetime, str]] = []
    for state_file in (project_dir / "versions").glob("*/version_state.yml"):
        try:
            state = yaml.safe_load(state_file.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        version_id = str(state.get("version_id") or state_file.parent.name)
        timestamp = _parse_datetime(state.get("updated_at") or state.get("created_at")) or datetime.fromtimestamp(state_file.stat().st_mtime)
        candidates.append((timestamp, version_id))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item[0], item[1]))[-1][1]


def _latest_run_id(project_dir: Path) -> str | None:
    candidates: list[tuple[datetime, str]] = []
    for state_file in (project_dir / "runs").glob("*/run_state.yml"):
        try:
            state = yaml.safe_load(state_file.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        run_id = str(state.get("run_id") or state_file.parent.name)
        timestamp = _parse_datetime(state.get("updated_at") or state.get("created_at")) or datetime.fromtimestamp(state_file.stat().st_mtime)
        candidates.append((timestamp, run_id))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item[0], item[1]))[-1][1]


def _load_project_info(project_dir: Path) -> dict[str, Any]:
    config_path = project_config_path(project_dir)
    if not config_path.exists():
        return {}
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    project = config.get("project")
    return project if isinstance(project, dict) else {}


def _stage_rows(run_state: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, stage in (run_state.get("stages") or {}).items():
        artifacts = stage.get("artifacts") if isinstance(stage, dict) else []
        rows.append(
            {
                "name": name,
                "status": stage.get("status", "unknown") if isinstance(stage, dict) else "unknown",
                "artifact_count": len(artifacts or []),
            }
        )
    return rows


def _stage_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = row.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _infer_run_followups(run_state: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    stages = run_state.get("stages") or {}
    failed = [name for name, stage in stages.items() if stage.get("status") == "failed"]
    running = [name for name, stage in stages.items() if stage.get("status") == "running"]
    pending = [name for name, stage in stages.items() if stage.get("status") == "pending"]
    scaffold = [name for name, stage in stages.items() if stage.get("status") == "scaffold"]

    next_actions: list[str] = []
    blockers = [f"stage failed: {name}" for name in failed]
    risks: list[str] = []
    if failed:
        next_actions.append(f"Investigate failed stage: {failed[0]}")
    elif running:
        next_actions.append(f"Resume or reconcile running stage: {running[0]}")
    elif pending:
        next_actions.append(f"Continue or explicitly reconcile pending stage: {pending[0]}")
    elif scaffold:
        next_actions.append(f"Replace scaffold artifacts with real evidence where required: {scaffold[0]}")
    else:
        next_actions.append("Review final artifacts and capture lessons before closing the project.")

    if str(run_state.get("workflow", "")).startswith("imported"):
        risks.append("imported run is not proof that the full workflow was rerun locally")
    if scaffold:
        risks.append("scaffold artifacts exist and must not be treated as real modeling evidence")
    return next_actions, blockers, risks


def _append_unique(existing: list[Any], new_items: list[Any]) -> list[Any]:
    result = list(existing or [])
    for item in new_items or []:
        if item and item not in result:
            result.append(item)
    return result


def _extend_list(lines: list[str], title: str, values: list[str]) -> None:
    if not values:
        return
    lines.extend(["", f"{title}:"])
    for value in values:
        lines.append(f"  - {value}")


def _extend_markdown_list(lines: list[str], title: str, values: list[str]) -> None:
    if not values:
        return
    lines.extend(["", f"## {title}", ""])
    for value in values:
        lines.append(f"- {value}")


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
