"""Project workspace creation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_ROOT = REPO_ROOT / "templates" / "project"


@dataclass(frozen=True)
class ProjectContext:
    name: str
    display_name: str
    scenario: str
    template: str
    created_date: str
    sample_table: str
    target_column: str
    time_column: str
    period_column: str

    def as_mapping(self) -> dict[str, str]:
        return {
            "project_name": self.name,
            "display_name": self.display_name,
            "scenario": self.scenario,
            "template": self.template,
            "created_date": self.created_date,
            "sample_table": self.sample_table,
            "target_column": self.target_column,
            "time_column": self.time_column,
            "period_column": self.period_column,
        }


def default_context(name: str, display_name: str, scenario: str, template: str) -> ProjectContext:
    """Build template defaults for a project."""
    if template == "fujie-gcard":
        sample_table = "ads_app_off_feature.ds29531_backtrack_fj_gcard_model_v6_1_sample"
        target_column = "ftr_30d_ord_flag"
        time_column = "mdl_dte"
        period_column = "ds"
    else:
        sample_table = ""
        target_column = "target"
        time_column = "sample_date"
        period_column = "sample_month"

    return ProjectContext(
        name=name,
        display_name=display_name,
        scenario=scenario,
        template=template,
        created_date=date.today().isoformat(),
        sample_table=sample_table,
        target_column=target_column,
        time_column=time_column,
        period_column=period_column,
    )


def render_text(text: str, context: ProjectContext) -> str:
    """Render simple {{key}} placeholders."""
    rendered = text
    for key, value in context.as_mapping().items():
        rendered = rendered.replace("{{" + key + "}}", value)
    return rendered


def create_project(
    root_dir: str | Path,
    *,
    name: str,
    display_name: str,
    scenario: str,
    template: str = "generic",
    force: bool = False,
) -> Path:
    """Create a project workspace from the standard template."""
    if not TEMPLATE_ROOT.exists():
        raise FileNotFoundError(f"Template directory not found: {TEMPLATE_ROOT}")

    root_path = Path(root_dir).resolve()
    project_dir = root_path / "projects" / name
    if project_dir.exists() and not force:
        raise FileExistsError(f"Project already exists: {project_dir}")

    context = default_context(name, display_name, scenario, template)
    for source_path in TEMPLATE_ROOT.rglob("*"):
        if source_path.is_dir():
            continue
        relative_path = source_path.relative_to(TEMPLATE_ROOT)
        if relative_path == Path("project.yaml") or (
            relative_path.parent == Path("configs") and relative_path.suffix == ".yml"
        ):
            continue
        target_path = project_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        text = source_path.read_text(encoding="utf-8")
        target_path.write_text(render_text(text, context), encoding="utf-8")

    canonical_project = project_dir / "project.yml"
    (project_dir / "project.yaml").write_text(
        canonical_project.read_text(encoding="utf-8"), encoding="utf-8"
    )
    for canonical in sorted((project_dir / "configs").glob("*.yaml")):
        canonical.with_suffix(".yml").write_text(
            canonical.read_text(encoding="utf-8"), encoding="utf-8"
        )

    for directory in [
        "data/raw",
        "data/sampled",
        "data/processed",
        "data/profile",
        "versions",
        "runs",
        "reports",
    ]:
        keep = project_dir / directory / ".gitkeep"
        keep.parent.mkdir(parents=True, exist_ok=True)
        if not keep.exists():
            keep.write_text("", encoding="utf-8")

    return project_dir
