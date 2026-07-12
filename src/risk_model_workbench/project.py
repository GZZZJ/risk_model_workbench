"""Compatibility forwarder for the :mod:`risk_model_workbench.project` package.

Python resolves the adjacent package directory before this historical module.
Keeping the file as a forwarding shim protects direct file loaders and archived
import references without maintaining a second project-creation implementation.
"""

from risk_model_workbench.project.create import (  # noqa: F401
    REPO_ROOT,
    TEMPLATE_ROOT,
    ProjectContext,
    create_project,
    default_context,
    render_text,
)
from risk_model_workbench.project import create  # noqa: F401

__all__ = [
    "REPO_ROOT",
    "TEMPLATE_ROOT",
    "ProjectContext",
    "default_context",
    "render_text",
    "create_project",
    "create",
]
