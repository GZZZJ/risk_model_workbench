"""Read-only loader for run state, artifact manifest, and workflow contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from risk_model_workbench.registry import load_artifact_manifest
from risk_model_workbench.state import load_run_state, workspace_id
from risk_model_workbench.versioning import resolve_workspace_dir
from risk_model_workbench.workflow_contracts import load_stage_contracts


@dataclass(frozen=True)
class RunEvidence:
    project_path: Path
    run_id: str
    run_path: Path
    run_state: dict[str, Any]
    manifest: dict[str, Any]
    manifest_by_stage: dict[str, list[dict[str, Any]]]
    stage_contracts: dict[str, dict[str, Any]]
    contract_source: str

    @property
    def workflow(self) -> str:
        return str(self.run_state.get("workflow", ""))


def load_run_evidence(project_dir: str | Path, run_id: str) -> RunEvidence:
    project_path = Path(project_dir)
    selected_run_dir = resolve_workspace_dir(project_path, run_id=run_id)
    run_state = load_run_state(selected_run_dir)
    stage_contracts, contract_source = load_stage_contracts(str(run_state.get("workflow", "")))
    manifest = load_artifact_manifest(selected_run_dir)
    return RunEvidence(
        project_path=project_path,
        run_id=workspace_id(run_state, run_id),
        run_path=selected_run_dir,
        run_state=run_state,
        manifest=manifest,
        manifest_by_stage=_manifest_by_stage(manifest),
        stage_contracts=stage_contracts,
        contract_source=contract_source,
    )


def _manifest_by_stage(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for artifact in manifest.get("artifacts", []) or []:
        grouped.setdefault(str(artifact.get("stage", "")), []).append(artifact)
    return grouped
