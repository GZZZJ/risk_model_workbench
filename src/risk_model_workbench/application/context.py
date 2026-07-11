"""Resolved paths shared by application action handlers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VersionContext:
    project_dir: Path
    version_id: str
    workspace: Path
    runtime_config_dir: Path
    manifest_path: Path
    version_state_path: Path

    @classmethod
    def from_project(cls, project_dir: str | Path, version_id: str) -> "VersionContext":
        project = Path(project_dir).resolve()
        workspace = project / "versions" / version_id
        return cls(
            project_dir=project,
            version_id=version_id,
            workspace=workspace,
            runtime_config_dir=workspace / "configs_runtime",
            manifest_path=workspace / "audit" / "artifact_manifest.json",
            version_state_path=workspace / "version_state.yml",
        )
