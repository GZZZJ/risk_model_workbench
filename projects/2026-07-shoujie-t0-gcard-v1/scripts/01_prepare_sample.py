#!/usr/bin/env python3
"""Prepare and profile local sample files.

This scaffold intentionally avoids hidden data mutation. Fill in the local
sample/feature paths first, then extend this script with the confirmed sampling
logic.
"""

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
    project = load_yaml(project_dir / "project.yaml")
    sample_config = load_yaml(project_dir / "configs" / "sample.yaml")

    raw_path = project_dir / sample_config["sample"]["local_sample_path"]
    feature_path = project_dir / sample_config["sample"]["local_feature_path"]

    print(f"project: {project['project']['display_name']}")
    print(f"sample path: {raw_path}")
    print(f"feature path: {feature_path}")
    print("status: scaffold only; fill data paths and confirmed sampling rules before execution")

    manifest = write_manifest(
        project_dir,
        "prepare_sample_scaffold",
        inputs=[project_dir / "project.yaml", project_dir / "configs" / "sample.yaml"],
        extra={
            "raw_path_exists": raw_path.exists(),
            "feature_path_exists": feature_path.exists(),
        },
    )
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
