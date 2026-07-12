"""Compatibility forwarder for :mod:`risk_model_workbench.feature_metadata`."""

from pathlib import Path as _Path
import sys as _sys

_SOURCE_ROOT = _Path(__file__).resolve().parents[1] / "src"
_SOURCE_ROOT_TEXT = str(_SOURCE_ROOT)
if _SOURCE_ROOT_TEXT in _sys.path:
    _sys.path.remove(_SOURCE_ROOT_TEXT)
_sys.path.insert(0, _SOURCE_ROOT_TEXT)

from risk_model_workbench.feature_metadata import *  # noqa: F401,F403,E402


if __name__ == "__main__":
    raise SystemExit(main())
