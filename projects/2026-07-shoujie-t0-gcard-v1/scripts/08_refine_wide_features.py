#!/usr/bin/env python3
"""Project wrapper for the standard wide-feature refinement flow."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = next(path for path in [SCRIPT_PATH, *SCRIPT_PATH.parents] if (path / "agent.py").exists())
PROJECT_DIR = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from risk_model_workbench.feature_refine import main as refine_main


if __name__ == "__main__":
    raise SystemExit(refine_main(["--project-dir", str(PROJECT_DIR), *sys.argv[1:]]))
