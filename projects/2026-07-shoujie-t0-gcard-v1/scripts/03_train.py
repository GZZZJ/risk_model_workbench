#!/usr/bin/env python3
"""Training scaffold for project experiments."""

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
    train_config = load_yaml(project_dir / "configs" / "train.yaml")
    experiments = train_config["training"]["experiments"]
    print("configured experiments:")
    for experiment in experiments:
        print(f"- {experiment['name']}: {experiment['display_name']}")

    manifest = write_manifest(
        project_dir,
        "train_scaffold",
        inputs=[project_dir / "configs" / "train.yaml"],
        extra={"experiment_count": len(experiments)},
    )
    print(f"manifest: {manifest}")
    print("status: scaffold only; implement after sample/features are confirmed")


if __name__ == "__main__":
    main()
