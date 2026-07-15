#!/usr/bin/env python3
"""Report scaffold for the model development report."""

from __future__ import annotations

import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "agent.py").exists():
            return candidate
    raise RuntimeError("Cannot locate repo root from script path")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
sys.path.insert(0, str(REPO_ROOT))

from risk_model_workbench.config import load_yaml
from risk_model_workbench.manifest import write_manifest


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    report_config = load_yaml(project_dir / "configs" / "report.yaml")
    output_dir = project_dir / report_config["report"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    draft = output_dir / "model_report_draft.md"
    sections = report_config["sections"]
    draft.write_text(
        "# 模型报告草稿\n\n" + "\n\n".join(f"## {section}\n\n待补充。" for section in sections) + "\n",
        encoding="utf-8",
    )

    manifest = write_manifest(
        project_dir,
        "report_scaffold",
        inputs=[project_dir / "configs" / "report.yaml"],
        outputs=[draft],
        extra={"section_count": len(sections)},
    )
    print(f"draft: {draft}")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
