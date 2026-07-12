"""Path helpers for the local modeling workbench."""

from __future__ import annotations

from pathlib import Path

from risk_model_workbench.config import resolve_yaml_variant


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a project path relative to the repository root."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def project_config_path(project_dir: str | Path) -> Path:
    """Return the preferred project config path, accepting legacy project.yaml."""
    project_path = Path(project_dir)
    return resolve_yaml_variant(project_path / "project.yml", project_path / "project.yaml")


def stage_config_path(project_dir: str | Path, name: str) -> Path:
    """Return the canonical stage config, accepting a legacy ``.yml`` mirror."""
    config_dir = Path(project_dir) / "configs"
    return resolve_yaml_variant(config_dir / f"{name}.yaml", config_dir / f"{name}.yml")


def workflow_path(workflow: str) -> Path:
    """Resolve a workflow name or path."""
    path = Path(workflow)
    if path.suffix:
        return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    return REPO_ROOT / "workflows" / f"{workflow}.yml"
