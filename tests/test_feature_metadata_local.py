from pathlib import Path

import pandas as pd
import yaml

from risk_model_workbench.cli import _finish_feature_metadata_local_feather
from risk_model_workbench.registry import load_artifact_manifest
from risk_model_workbench.state import create_version_state, load_run_state, save_version_state


def test_local_feather_metadata_uses_arrow_schema_without_remote_access(tmp_path):
    project = tmp_path / "project"
    workspace = project / "versions" / "demo_v1"
    runtime = workspace / "configs_runtime"
    runtime.mkdir(parents=True)
    feather = tmp_path / "sample.feather"
    pd.DataFrame({"uid": [1, 2], "label": [0, 1], "x1": [0.1, 0.2]}).to_feather(feather)
    (runtime / "feature_select.yaml").write_text(
        yaml.safe_dump(
            {
                "feature_select": {
                    "runtime_request": {
                        "data_source_mode": "local_feather",
                        "sample_location": str(feather),
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    state = create_version_state(
        project,
        version_id="demo_v1",
        workflow="full_modeling",
        stages=["feature_metadata"],
    )
    save_version_state(workspace, state)

    assert _finish_feature_metadata_local_feather(workspace, project) == 0

    columns = pd.read_csv(workspace / "feature_metadata" / "feature_columns.csv")
    assert columns["feature_name"].tolist() == ["uid", "label", "x1"]
    assert load_run_state(workspace)["stages"]["feature_metadata"]["status"] == "done"
    manifest = load_artifact_manifest(workspace)
    assert {item["path"] for item in manifest["artifacts"]} >= {
        "feature_metadata/feature_tables_meta.json",
        "feature_metadata/feature_table_summary.csv",
        "feature_metadata/feature_columns.csv",
    }
