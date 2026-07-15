#!/usr/bin/env python3
"""Evaluation scaffold for score comparison and monitoring tables."""

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
    evaluate_config = load_yaml(project_dir / "configs" / "evaluate.yaml")
    print("metrics:", ", ".join(evaluate_config["metrics"]))
    print("score columns:", ", ".join(evaluate_config["evaluation"]["score_columns"]))

    manifest = write_manifest(
        project_dir,
        "evaluate_scaffold",
        inputs=[project_dir / "configs" / "evaluate.yaml"],
        extra={
            "metrics": evaluate_config["metrics"],
            "score_columns": evaluate_config["evaluation"]["score_columns"],
        },
    )
    print(f"manifest: {manifest}")
    print("status: scaffold only; implement after training outputs are available")


if __name__ == "__main__":
    main()
