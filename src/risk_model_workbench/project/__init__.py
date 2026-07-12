"""Project creation API and compatibility namespace."""

from risk_model_workbench.project import create
from risk_model_workbench.project.create import (
    REPO_ROOT,
    TEMPLATE_ROOT,
    ProjectContext,
    create_project,
    default_context,
    render_text,
)

__all__ = [
    "REPO_ROOT",
    "TEMPLATE_ROOT",
    "ProjectContext",
    "default_context",
    "render_text",
    "create_project",
    "create",
]
