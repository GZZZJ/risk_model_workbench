"""Explicit context snapshots for handoffs and context compression."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from risk_model_workbench.agent.advisor import list_advisor_requests, load_advisor_request
from risk_model_workbench.agent.context_pack import load_context_pack
from risk_model_workbench.config import load_yaml
from risk_model_workbench.facts import list_facts
from risk_model_workbench.paths import workflow_path
from risk_model_workbench.progress import load_progress_summary
from risk_model_workbench.project_state import audit_run, load_project_state
from risk_model_workbench.run_evidence import load_run_evidence
from risk_model_workbench.state import workspace_id
from risk_model_workbench.versioning import resolve_workspace_dir


def build_context_snapshot(project_dir: str | Path, run_id: str) -> dict[str, Any]:
    evidence = load_run_evidence(project_dir, run_id)
    project_path = evidence.project_path
    selected_run = evidence.run_path
    run_state = evidence.run_state
    contract_source = evidence.contract_source
    selected_id = workspace_id(run_state, run_id)
    workspace_rel = selected_run.relative_to(project_path)
    state_filename = "version_state.yml" if (selected_run / "version_state.yml").exists() else "run_state.yml"

    snapshot = {
        "version": 1,
        "generated_at": _now(),
        "project": str(project_path.resolve()),
        "version_id": run_state.get("version_id", ""),
        "run_id": run_state.get("run_id", run_id),
        "sources": [
            "project_state.yml",
            str(workspace_rel / state_filename),
            str(workspace_rel / "audit" / "artifact_manifest.json"),
            contract_source,
            str(workspace_rel / "audit" / "progress_summary.json"),
            "project_facts.yml",
        ],
        "project_state": load_project_state(project_path),
        "request": _read_run_text(selected_run / "model_request.md"),
        "plan": _read_run_text(selected_run / "execution_plan.yml"),
        "workflow": _workflow_payload(evidence.workflow, evidence.stage_contracts, contract_source),
        "run_state": run_state,
        "artifact_manifest": _compact_manifest(evidence.manifest),
        "latest_audit": audit_run(project_path, selected_id),
        "decision_log": list(run_state.get("decisions", []))[-20:],
        "progress_summary": load_progress_summary(selected_run),
        "facts": list_facts(project_path),
    }
    reference = _latest_advisor_context(selected_run)
    if reference:
        snapshot = attach_context_pack_reference(
            snapshot,
            selected_run,
            reference["context_pack"],
            reference["context_hash"],
        )
    return snapshot


def write_context_snapshot(
    project_dir: str | Path,
    run_id: str,
    *,
    output: str | Path | None = None,
    markdown: bool = False,
) -> tuple[Path, Path | None]:
    project_path = Path(project_dir)
    selected_run = resolve_workspace_dir(project_path, run_id=run_id)
    snapshot = build_context_snapshot(project_path, run_id)
    output_path = Path(output) if output else selected_run / "audit" / "context_snapshot.json"
    if not output_path.is_absolute():
        output_path = project_path / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")

    markdown_path = None
    if markdown:
        markdown_path = output_path.with_suffix(".md")
        markdown_path.write_text(format_context_snapshot(snapshot), encoding="utf-8")
    return output_path, markdown_path


def format_context_snapshot(snapshot: dict[str, Any]) -> str:
    title_id = snapshot.get("version_id") or snapshot.get("run_id")
    lines = [
        f"# Context Snapshot - {title_id}",
        "",
        f"- generated_at: {snapshot.get('generated_at')}",
        f"- project: {snapshot.get('project')}",
        f"- version_id: {snapshot.get('version_id') or ''}",
        f"- run_id: {snapshot.get('run_id') or ''}",
        f"- audit_verdict: {(snapshot.get('latest_audit') or {}).get('verdict')}",
        "",
        "## Sources",
        "",
    ]
    for source in snapshot.get("sources") or []:
        if source:
            lines.append(f"- {source}")

    run_state = snapshot.get("run_state") or {}
    lines.extend(["", "## Run", ""])
    lines.append(f"- workflow: {run_state.get('workflow')}")
    lines.append(f"- status: {run_state.get('status')}")
    lines.append(f"- current_stage: {run_state.get('current_stage')}")

    lines.extend(["", "## Audit Issues", ""])
    issues = [issue for stage in (snapshot.get("latest_audit") or {}).get("stages", []) for issue in stage.get("issues", [])]
    if not issues:
        lines.append("- none")
    else:
        lines.extend(f"- {issue}" for issue in issues[:20])

    facts = snapshot.get("facts") or []
    lines.extend(["", "## Facts", ""])
    if not facts:
        lines.append("- none")
    else:
        for fact in facts[:20]:
            lines.append(f"- [{fact.get('category')}] {fact.get('statement')} (source: {fact.get('source_path')})")
    lines.append("")
    return "\n".join(lines)


def attach_context_pack_reference(
    snapshot: dict[str, Any],
    workspace: str | Path,
    context_pack: str,
    context_hash: str,
) -> dict[str, Any]:
    """Attach a safe immutable reasoning-context reference to a snapshot."""
    normalised = str(context_pack).replace("\\", "/")
    path = PurePosixPath(normalised)
    digest = str(context_hash)
    expected = PurePosixPath("audit") / "context_packs" / f"{digest}.json"
    if path.is_absolute() or ".." in path.parts or path != expected:
        raise ValueError("context pack reference must be workspace-relative")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("context pack reference requires a lowercase SHA256 hash")
    pack = load_context_pack(workspace, normalised)
    if pack.get("context_hash") != digest:
        raise ValueError("context pack reference hash mismatch")
    updated = deepcopy(snapshot)
    reference = {
        "context_pack": normalised,
        "context_hash": digest,
    }
    updated["reasoning_context"] = reference
    # Compatibility alias for snapshots created before ADR 0005.
    updated["host_agent_context"] = dict(reference)
    sources = updated.get("sources")
    if isinstance(sources, list) and normalised not in sources:
        sources.append(normalised)
    return updated


def _latest_advisor_context(workspace: Path) -> dict[str, str] | None:
    candidates = [
        request
        for request in list_advisor_requests(workspace)
        if int(request.get("version") or 1) >= 2
        and request.get("status") not in {"consumed", "rejected"}
        and request.get("context_pack")
    ]
    if not candidates:
        return None
    selected = max(candidates, key=lambda item: (str(item.get("created_at") or ""), str(item.get("request_id") or "")))
    request = load_advisor_request(workspace, str(selected["request_id"]))
    return {
        "context_pack": str(request["context_pack"]),
        "context_hash": str(request["context_hash"]),
    }


def _workflow_payload(workflow: str, stage_contracts: dict[str, dict[str, Any]], contract_source: str) -> dict[str, Any]:
    path = workflow_path(workflow) if workflow else Path()
    payload: dict[str, Any] = {
        "name": workflow,
        "contract_source": contract_source,
        "stage_contracts": stage_contracts,
    }
    if path and path.exists():
        payload["definition"] = load_yaml(path)
    return payload


def _compact_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in manifest.get("artifacts", []) or []:
        stage = str(item.get("stage", ""))
        grouped.setdefault(stage, []).append(
            {
                "path": item.get("path"),
                "kind": item.get("kind"),
                "source": item.get("source"),
                "exists": item.get("exists"),
                "description": item.get("description", ""),
            }
        )
    return {
        "version": manifest.get("version", 1),
        "artifact_count": sum(len(items) for items in grouped.values()),
        "by_stage": grouped,
    }


def _read_run_text(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "content": ""}
    return {
        "path": str(path),
        "exists": True,
        "content": path.read_text(encoding="utf-8"),
    }


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
