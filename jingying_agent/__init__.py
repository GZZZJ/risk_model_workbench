"""Compatibility package for legacy imports.

New code should import from ``risk_model_workbench``.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

_PACKAGE_ROOT = Path(__file__).resolve().parent
_SOURCE_ROOT = _PACKAGE_ROOT.parent / "src"
_SOURCE_ROOT_TEXT = str(_SOURCE_ROOT)
if _SOURCE_ROOT_TEXT in sys.path:
    sys.path.remove(_SOURCE_ROOT_TEXT)
sys.path.insert(0, _SOURCE_ROOT_TEXT)

import risk_model_workbench as _new_package

# Load the explicit root forwarding modules first, then fall through to the
# canonical package for historical submodules that never had a root file.
__path__ = [str(_PACKAGE_ROOT), *list(_new_package.__path__)]
__version__ = getattr(_new_package, "__version__", "0.0.0")
__all__ = list(getattr(_new_package, "__all__", []))


def __getattr__(name: str) -> Any:
    return getattr(_new_package, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_new_package)))
