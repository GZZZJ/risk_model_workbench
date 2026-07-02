"""Shared split-value resolution and consistency checks for request-driven modeling.

Single source of truth for how in-time (dev/oos) and out-of-time (oot) split
values are derived from a model_request + project.yml. Consumed by both
``validate_model_request`` (the user-facing entry gate) and
``materialize_request_runtime_configs`` (runtime config production) so the two
never drift apart — the rule the user sees at validation is the exact rule the
producer enforces.

Split semantics (``final_flag`` column):
  dev  in-time train sample            (e.g. DEV)
  oos  in-time out-of-sample holdout   (e.g. DEV-OOS) — MAY feed early stopping / tuning
  oot  out-of-time sample              (e.g. OOT, OOT-OOS) — eval-only, MUST NOT feed model selection

The central correctness invariant: ``oos`` and ``oot`` must be disjoint. If a
time-out label ever reaches ``valid_values``, it silently leaks into
early-stopping / tuning and inflates the OOT evaluation it is meant to measure.
"""

from __future__ import annotations

from typing import Any


DEFAULT_INS_VALUES = ["DEV"]
DEFAULT_OOS_VALUES = ["DEV-OOS"]
DEFAULT_OOT_VALUES = ["OOT"]


class SplitValidationError(ValueError):
    """Raised by the materializer (strict mode) when resolved splits break a hard rule."""


def _clean_string_list(value: Any) -> list[str]:
    """Coerce to a list of non-empty strings, matching materialize._string_list.

    Scalars are wrapped as a single-element list (a bare ``DEV-OOS`` is treated
    like ``[DEV-OOS]``), so a project.yml written as ``oos_values: DEV-OOS`` is
    honored rather than silently dropped on the floor.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    return [str(item) for item in value if str(item) != ""]


def resolve_split_values(metadata: dict[str, Any], project_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Resolve dev/oos/oot value lists with request-first → project → default priority.

    Single source of truth consumed by both ``validate_model_request`` (entry
    gate) and ``materialize_request_runtime_configs`` (producer) so the two
    agree on what each split resolves to. Returns a dict keyed by
    ``dev``/``oos``/``oot`` whose values are ``{"values": [...], "source": ...}``
    where source ∈ ``{"request.splits", "project.split", "default"}`` — the
    source lets consistency checks tell "user explicitly misconfigured" apart
    from "fell through to a safe default".
    """
    rsplits = metadata.get("splits") if isinstance(metadata.get("splits"), dict) else {}
    psplit = project_config.get("split", {}) if isinstance(project_config.get("split"), dict) else {}
    mapping = [
        ("dev", "ins_values", DEFAULT_INS_VALUES),
        ("oos", "oos_values", DEFAULT_OOS_VALUES),
        ("oot", "oot_values", DEFAULT_OOT_VALUES),
    ]
    resolved: dict[str, dict[str, Any]] = {}
    for key, project_key, default in mapping:
        values: list[str] = []
        source = "default"
        entry = rsplits.get(key)
        if isinstance(entry, dict):
            candidate = _clean_string_list(entry.get("values"))
            if candidate:
                values, source = candidate, "request.splits"
        if not values:
            candidate = _clean_string_list(psplit.get(project_key))
            if candidate:
                values, source = candidate, "project.split"
        if not values:
            values, source = list(default), "default"
        resolved[key] = {"values": values, "source": source}
    return resolved


def check_split_consistency(metadata: dict[str, Any], project_config: dict[str, Any]) -> dict[str, Any]:
    """Return ``{"errors", "warnings", "resolved"}`` for a split configuration.

    errors (blocking — time-out pollution / unusable splits):
      - oos ∩ oot ≠ ∅  : time-out label reaches the in-time validation set
      - splits.<key> explicitly set but values empty/invalid (e.g. ['']) and
        silently fell back to project/default — user intent was ignored
    warnings (non-blocking — guidance only):
      - oot empty                      : no time-out eval slice
      - dev ∩ oos ≠ ∅                  : train and validation overlap
      - request.training.valid_values  : dead field, silently overridden by materialize
      - all splits defaulted           : request + project both silent (smoke/template path)
    """
    resolved = resolve_split_values(metadata, project_config)
    dev = resolved["dev"]["values"]
    oos = resolved["oos"]["values"]
    oot = resolved["oot"]["values"]
    errors: list[str] = []
    warnings: list[str] = []

    # Surface two misconfiguration classes that would otherwise silently fall
    # back to project/default: (a) an explicit splits.<key> mapping whose values
    # resolve to nothing (dirty [''] etc.), and (b) a non-mapping shorthand
    # (e.g. `oos: DEV-OOS`) that the resolver skips entirely.
    rsplits = metadata.get("splits") if isinstance(metadata.get("splits"), dict) else {}
    for key in ("dev", "oos", "oot"):
        raw = rsplits.get(key)
        if isinstance(raw, dict):
            if resolved[key]["source"] != "request.splits":
                errors.append(
                    f"splits.{key}.values 显式配置但为空或无效（已回退到 {resolved[key]['source']}: {resolved[key]['values']}）"
                    f"—— 请填写有效的 {key} 标签"
                )
        elif raw is not None:
            warnings.append(f"splits.{key} 不是 mapping（应为 {{values: [...]}}），已忽略")

    oos_oot_intersect = sorted(set(oos) & set(oot))
    if oos_oot_intersect:
        errors.append(
            f"splits.oos 与 splits.oot 相交: {oos_oot_intersect} —— 时间外样本(OOT)不能进 in-time 验证集，"
            f"会让早停/调参污染时间外评估；修复：检查 model_request.md 的 splits.oos.values，应只含 in-time 标签(如 DEV-OOS)"
        )
    # Note: dev/oos/oot emptiness is unreachable in practice — resolve_split_values
    # always backfills non-empty DEFAULT_*_VALUES — so there is no explicit empty
    # check here. An explicit-but-empty mapping is caught by the dirty-value
    # branch above; the silent-default path is covered by the "all defaulted"
    # warning below.

    dev_oos_intersect = sorted(set(dev) & set(oos))
    if dev_oos_intersect:
        warnings.append(f"splits.dev 与 splits.oos 相交: {dev_oos_intersect} —— 训练集与验证集重叠")

    request_training = metadata.get("training") if isinstance(metadata.get("training"), dict) else {}
    if isinstance(request_training.get("valid_values"), list):
        warnings.append(
            "request.training.valid_values 在 request 里无效 —— 会被 materialize 按 splits.oos 自动推导覆盖；"
            "如需调整验证集，请改写 model_request.md 的 splits.oos.values"
        )

    if all(resolved[key]["source"] == "default" for key in ("dev", "oos", "oot")):
        warnings.append("splits 未在 request/project 显式配置，使用默认 dev=DEV / oos=DEV-OOS / oot=OOT")

    return {"errors": errors, "warnings": warnings, "resolved": resolved}
