#!/usr/bin/env python3
"""Generate a feature-select-v2 config for this project.

The first version writes the adapter config and records a manifest. Execution is
kept explicit because feature selection may query large tables or consume heavy
local resources.
"""

from __future__ import annotations

import json
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
from risk_model_workbench.manifest import make_run_id, write_manifest


def build_feature_select_config(project_dir: Path, run_id: str) -> dict:
    project = load_yaml(project_dir / "project.yaml")
    fs = load_yaml(project_dir / "configs" / "feature_select.yaml")["feature_select"]

    data = project["data"]
    split = project["split"]
    output_dir = fs["output_dir"].format(run_id=run_id)

    return {
        "project_name": project["project"]["name"],
        "sample": {
            "table": data.get("source_table") or str(project_dir / data["raw_path"]),
            "id_col": data["id_columns"],
            "target_col": data["target_column"],
            "tw_col": split["source_column"],
            "time_col": data["time_column"],
            "period_col": data["period_column"],
            "ins_oos_col": split["source_column"],
        },
        "thresholds": fs["thresholds"],
        "bigtable": fs["bigtable"].get("tables") or fs["bigtable"].get("local_files") or [],
        "feature_info": fs["feature_info"]["table_or_path"],
        "project_path": str(project_dir / output_dir),
        "steps": fs["steps"],
        "train_baseline_model": fs["train_baseline_model"],
        "params": fs.get("params", {}),
    }


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    run_id = make_run_id()
    config = build_feature_select_config(project_dir, run_id)

    run_dir = project_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "feature_select_config.json"
    output_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = write_manifest(
        project_dir,
        "feature_select_config",
        run_id=run_id,
        inputs=[
            project_dir / "project.yaml",
            project_dir / "configs" / "feature_select.yaml",
        ],
        outputs=[output_path],
        extra={"execute_feature_select": False},
    )
    print(f"feature-select config: {output_path}")
    print(f"manifest: {manifest}")
    print("status: config generated only; execute after bigtable/feature_info are confirmed")


if __name__ == "__main__":
    main()
