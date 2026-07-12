from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_repo_root_inference_preserves_source_checkout_behavior():
    import risk_model_workbench.batch_feature_select as batch
    import risk_model_workbench.feature_refine as refine

    assert batch.REPO_ROOT == REPO_ROOT
    assert refine.REPO_ROOT == REPO_ROOT


def test_repo_root_inference_falls_back_to_installed_package_root(tmp_path):
    isolated_source = tmp_path / "installed-layout"
    shutil.copytree(REPO_ROOT / "src", isolated_source / "src")
    unrelated_cwd = tmp_path / "cwd-without-agent-script"
    unrelated_cwd.mkdir()

    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment["PYTHONPATH"] = str(isolated_source / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "import risk_model_workbench.batch_feature_select as batch; "
                "import risk_model_workbench.feature_refine as refine; "
                "print(json.dumps([str(batch.REPO_ROOT), str(refine.REPO_ROOT)]))"
            ),
        ],
        cwd=unrelated_cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    expected_root = str(isolated_source / "src")
    assert result.stdout.strip() == f'["{expected_root}", "{expected_root}"]'
