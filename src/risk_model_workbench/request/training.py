"""Training policy helpers for request-driven workflows."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


LLM_GUIDED_TUNING_MODES = {"llm_guided_tune", "llm-guided-tune", "llm_guided"}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def merge_training_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    return _deep_merge(base, override)


def project_training_defaults(project_config: dict[str, Any]) -> dict[str, Any]:
    """Return project-level training defaults used when a request omits them."""
    raw = project_config.get("training_defaults")
    if not isinstance(raw, dict):
        project_root = project_config.get("project") if isinstance(project_config.get("project"), dict) else {}
        raw = project_root.get("training_defaults")
    return deepcopy(raw) if isinstance(raw, dict) else {}


def request_training_config(metadata: dict[str, Any]) -> dict[str, Any]:
    raw = metadata.get("training")
    return deepcopy(raw) if isinstance(raw, dict) else {}


def training_mode(config: dict[str, Any]) -> str:
    tuning = config.get("tuning") if isinstance(config.get("tuning"), dict) else {}
    mode = config.get("mode") or tuning.get("mode") or ""
    return str(mode).strip().lower()


def is_llm_guided_tuning_mode(mode: str) -> bool:
    return str(mode or "").strip().lower() in LLM_GUIDED_TUNING_MODES


def llm_guided_tuning_enabled(config: dict[str, Any]) -> bool:
    return is_llm_guided_tuning_mode(training_mode(config))


def effective_training_config(metadata: dict[str, Any], project_config: dict[str, Any]) -> dict[str, Any]:
    """Merge project defaults with request-level overrides.

    A request-level non-LLM mode explicitly disables a project LLM tuning
    default. In that case inherited tuning parameters are removed unless the
    request provides its own tuning block.
    """
    defaults = project_training_defaults(project_config)
    request = request_training_config(metadata)
    effective = _deep_merge(defaults, request)
    explicit_mode = training_mode(request)
    if explicit_mode and not is_llm_guided_tuning_mode(explicit_mode) and llm_guided_tuning_enabled(defaults) and "tuning" not in request:
        effective.pop("tuning", None)
    return effective


def request_disables_project_llm_tuning(metadata: dict[str, Any], project_config: dict[str, Any]) -> bool:
    request = request_training_config(metadata)
    explicit_mode = training_mode(request)
    return bool(explicit_mode and not is_llm_guided_tuning_mode(explicit_mode) and llm_guided_tuning_enabled(project_training_defaults(project_config)))


def tuning_disable_reason(metadata: dict[str, Any]) -> str:
    training = request_training_config(metadata)
    return str(training.get("disable_tuning_reason") or training.get("tuning_disable_reason") or "").strip()
