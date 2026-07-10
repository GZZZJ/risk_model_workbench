"""LLM-guided hyperparameter tuning helpers.

The workbench keeps model training deterministic and auditable: an advisor may
suggest candidate parameters, while metric-based rules select the final trial.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any


TRAIN_CONTROL_PARAMS = {"num_boost_round", "early_stopping_rounds"}

LGB_NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "learning_rate": (0.005, 0.2),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.5, 1.0),
    "reg_alpha": (0.0, 20.0),
    "reg_lambda": (0.0, 50.0),
    "min_gain_to_split": (0.0, 5.0),
}

LGB_INTEGER_BOUNDS: dict[str, tuple[int, int]] = {
    "num_leaves": (7, 255),
    "max_depth": (3, 12),
    "min_child_samples": (20, 1000),
    "bagging_freq": (0, 10),
    "max_bin": (63, 511),
    "num_boost_round": (100, 3000),
    "early_stopping_rounds": (20, 300),
}

DEFAULT_TUNING = {
    "max_rounds": 2,
    "candidates_per_round": 4,
    "max_trials": 8,
    "objective": {"primary_metric": "valid_ks", "secondary_metric": "valid_auc"},
    "guardrails": {"max_train_valid_auc_gap": 0.03},
    "advisor": {"mode": "host_agent", "fallback_to_heuristic": True},
}


class HostAgentTuningPlanRequired(RuntimeError):
    """Raised when Codex/ClaudeCode should provide the next tuning plan."""

    def __init__(self, *, plan_path: str | Path | None, context_path: str | Path | None = None) -> None:
        self.plan_path = str(plan_path or "")
        self.context_path = str(context_path or "")
        detail = f"host agent tuning plan required: write {self.plan_path}"
        if self.context_path:
            detail += f" from {self.context_path}"
        super().__init__(detail)


def llm_guided_tuning_enabled(config: dict[str, Any]) -> bool:
    training = config.get("training") or {}
    tuning = training.get("tuning") if isinstance(training.get("tuning"), dict) else {}
    mode = str(training.get("mode") or tuning.get("mode") or "").strip().lower()
    return mode in {"llm_guided_tune", "llm-guided-tune", "llm_guided"}


def resolve_tuning_config(config: dict[str, Any]) -> dict[str, Any]:
    training = config.get("training") or {}
    raw_tuning = training.get("tuning") if isinstance(training.get("tuning"), dict) else {}
    resolved = _deep_merge(DEFAULT_TUNING, raw_tuning)
    resolved["mode"] = "llm_guided_tune"
    resolved["max_rounds"] = max(0, int(resolved.get("max_rounds", 2)))
    resolved["candidates_per_round"] = max(1, int(resolved.get("candidates_per_round", 4)))
    resolved["max_trials"] = max(1, int(resolved.get("max_trials", resolved["max_rounds"] * resolved["candidates_per_round"])))
    resolved["param_bounds"] = _merge_param_bounds(raw_tuning.get("param_bounds") if isinstance(raw_tuning.get("param_bounds"), dict) else {})
    advisor = resolved.get("advisor") if isinstance(resolved.get("advisor"), dict) else {}
    resolved["advisor"] = advisor
    return resolved


def public_lgb_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in params.items() if key not in {"seed", "feature_fraction_seed", "bagging_seed", "verbose"}}


def sanitize_candidate_params(params: dict[str, Any], tuning_cfg: dict[str, Any]) -> dict[str, Any]:
    bounds = tuning_cfg.get("param_bounds") or _merge_param_bounds({})
    sanitized: dict[str, Any] = {}
    for key, value in (params or {}).items():
        if key in LGB_INTEGER_BOUNDS:
            lower, upper = bounds["integer"].get(key, LGB_INTEGER_BOUNDS[key])
            sanitized[key] = int(max(lower, min(upper, round(float(value)))))
        elif key in LGB_NUMERIC_BOUNDS:
            lower, upper = bounds["numeric"].get(key, LGB_NUMERIC_BOUNDS[key])
            sanitized[key] = float(max(lower, min(upper, float(value))))
    return sanitized


def suggest_lgb_candidates(
    context: dict[str, Any],
    tuning_cfg: dict[str, Any],
    *,
    plan_path: str | Path | None = None,
    context_path: str | Path | None = None,
) -> dict[str, Any]:
    advisor = tuning_cfg.get("advisor") or {}
    command = str(advisor.get("command") or "").strip()
    if command:
        try:
            return _suggest_with_command(command, context, tuning_cfg)
        except Exception as exc:
            if advisor.get("fallback_to_heuristic", True) is False:
                raise
            plan = _suggest_with_heuristic(context, tuning_cfg)
            plan["fallback_reason"] = str(exc)
            return plan
    mode = str(advisor.get("mode") or advisor.get("type") or "host_agent").strip().lower()
    if mode in {"host_agent", "agent_in_loop", "codex", "claudecode", "claude_code", "claude-code"}:
        if plan_path and Path(plan_path).exists():
            plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
            expected_round = int(context.get("round", 0) or 0) or None
            expected_experiment = str(context.get("experiment") or "") or None
            return validate_tuning_plan(
                plan,
                tuning_cfg,
                advisor_type="host_agent_plan_file",
                expected_experiment=expected_experiment,
                expected_round=expected_round,
            )
        if advisor.get("fallback_to_heuristic", True) is False:
            raise HostAgentTuningPlanRequired(plan_path=plan_path, context_path=context_path)
        plan = _suggest_with_heuristic(context, tuning_cfg)
        plan["advisor_type"] = "host_agent_unavailable_local_heuristic_fallback"
        plan["fallback_reason"] = "host agent plan file not found; used local heuristic fallback"
        return plan
    return _suggest_with_heuristic(context, tuning_cfg)


def build_tuning_context(
    *,
    round_index: int,
    experiment: str,
    algorithm: str,
    base_params: dict[str, Any],
    training_summary: dict[str, Any],
    trial_history: list[dict[str, Any]],
    tuning_cfg: dict[str, Any],
) -> dict[str, Any]:
    return {
        "round": round_index,
        "experiment": experiment,
        "task_type": "binary_classification",
        "algorithm": algorithm,
        "base_params": public_lgb_params(base_params),
        "training_summary": training_summary,
        "trial_history": [_public_trial(item) for item in trial_history],
        "constraints": {
            "max_candidates": tuning_cfg["candidates_per_round"],
            "remaining_trials": max(0, tuning_cfg["max_trials"] - max(0, len(trial_history) - 1)),
            "param_bounds": tuning_cfg.get("param_bounds"),
            "objective": tuning_cfg.get("objective"),
            "guardrails": tuning_cfg.get("guardrails"),
        },
    }


def select_best_trial(trials: list[dict[str, Any]], tuning_cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not trials:
        raise ValueError("cannot select best trial from empty trial history")
    objective = tuning_cfg.get("objective") or {}
    guardrails = tuning_cfg.get("guardrails") or {}
    primary = str(objective.get("primary_metric") or "valid_ks")
    secondary = str(objective.get("secondary_metric") or "valid_auc")
    max_gap = float(guardrails.get("max_train_valid_auc_gap", 0.03))

    ranked = []
    for trial in trials:
        gap = float(trial.get("auc_gap", 0.0) or 0.0)
        primary_value = float(trial.get(primary, 0.0) or 0.0)
        secondary_value = float(trial.get(secondary, 0.0) or 0.0)
        guardrail_passed = gap <= max_gap
        penalty = max(0.0, gap - max_gap)
        score = primary_value + secondary_value * 0.05 - penalty * 2.0
        ranked.append(
            {
                "trial": trial,
                "selection_score": score,
                "guardrail_passed": guardrail_passed,
                "primary_metric": primary,
                "primary_value": primary_value,
                "secondary_metric": secondary,
                "secondary_value": secondary_value,
                "auc_gap": gap,
            }
        )
    eligible = [item for item in ranked if item["guardrail_passed"]]
    pool = eligible or ranked
    best = sorted(pool, key=lambda item: (item["selection_score"], item["primary_value"], item["secondary_value"]), reverse=True)[0]
    summary = {
        "best_trial_id": best["trial"].get("trial_id"),
        "selection_score": best["selection_score"],
        "primary_metric": best["primary_metric"],
        "primary_value": best["primary_value"],
        "secondary_metric": best["secondary_metric"],
        "secondary_value": best["secondary_value"],
        "auc_gap": best["auc_gap"],
        "guardrail_passed": best["guardrail_passed"],
        "guardrail_status": "passed" if eligible else "all_trials_failed_guardrail_best_effort",
        "candidate_count": len(trials),
    }
    return best["trial"], summary


def write_selection_reason(path: str | Path, best_trial: dict[str, Any], selection: dict[str, Any]) -> None:
    lines = [
        "# Tuning Selection Reason",
        "",
        f"- Best trial: {selection.get('best_trial_id')}",
        f"- Guardrail status: {selection.get('guardrail_status')}",
        f"- Primary metric: {selection.get('primary_metric')}={selection.get('primary_value'):.6f}",
        f"- Secondary metric: {selection.get('secondary_metric')}={selection.get('secondary_value'):.6f}",
        f"- AUC gap: {selection.get('auc_gap'):.6f}",
        f"- Candidate: {best_trial.get('candidate_name')}",
        f"- Reason: {best_trial.get('reason', '')}",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _suggest_with_command(command: str, context: dict[str, Any], tuning_cfg: dict[str, Any]) -> dict[str, Any]:
    timeout = int((tuning_cfg.get("advisor") or {}).get("timeout_seconds", 120))
    proc = subprocess.run(
        shlex.split(command),
        input=json.dumps(context, ensure_ascii=False),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"advisor command failed with code {proc.returncode}: {proc.stderr.strip()}")
    plan = json.loads(proc.stdout)
    expected_round = int(context.get("round", 0) or 0) or None
    expected_experiment = str(context.get("experiment") or "") or None
    return validate_tuning_plan(
        plan,
        tuning_cfg,
        advisor_type="external_llm_command",
        expected_experiment=expected_experiment,
        expected_round=expected_round,
    )


def _suggest_with_heuristic(context: dict[str, Any], tuning_cfg: dict[str, Any]) -> dict[str, Any]:
    history = context.get("trial_history") or []
    base = context.get("base_params") or {}
    best = _best_public_trial(history) or {"params": base, "auc_gap": 0.0, "valid_ks": 0.0}
    best_params = best.get("params") or base
    gap = float(best.get("auc_gap", 0.0) or 0.0)
    round_index = int(context.get("round", 1))
    max_gap = float((tuning_cfg.get("guardrails") or {}).get("max_train_valid_auc_gap", 0.03))

    if gap > max_gap:
        diagnosis = "best trial shows overfit; propose lower capacity and stronger regularization"
        raw_candidates = [
            _merge_params(best_params, learning_rate=0.025, num_leaves=31, max_depth=5, min_child_samples=200, subsample=0.75, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=5.0, bagging_freq=3, num_boost_round=1200, early_stopping_rounds=80),
            _merge_params(best_params, learning_rate=0.03, num_leaves=15, max_depth=4, min_child_samples=300, subsample=0.8, colsample_bytree=0.75, reg_alpha=1.0, reg_lambda=8.0, bagging_freq=3, num_boost_round=900, early_stopping_rounds=80),
            _merge_params(best_params, learning_rate=0.02, num_leaves=31, max_depth=6, min_child_samples=250, subsample=0.7, colsample_bytree=0.65, reg_alpha=0.2, reg_lambda=10.0, bagging_freq=5, num_boost_round=1500, early_stopping_rounds=100),
        ]
    else:
        diagnosis = "baseline is within overfit guardrail; explore moderate capacity and learning-rate tradeoffs"
        raw_candidates = [
            _merge_params(best_params, learning_rate=0.03, num_leaves=63, max_depth=7, min_child_samples=100, subsample=0.82, colsample_bytree=0.75, reg_alpha=0.1, reg_lambda=2.0, bagging_freq=3, num_boost_round=1200, early_stopping_rounds=80),
            _merge_params(best_params, learning_rate=0.025, num_leaves=63, max_depth=7, min_child_samples=150, subsample=0.86, colsample_bytree=0.72, reg_alpha=0.1, reg_lambda=2.0, bagging_freq=3, num_boost_round=1000, early_stopping_rounds=80),
            _merge_params(best_params, learning_rate=0.04, num_leaves=31, max_depth=6, min_child_samples=80, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.05, reg_lambda=1.0, bagging_freq=2, num_boost_round=800, early_stopping_rounds=60),
            _merge_params(best_params, learning_rate=0.02, num_leaves=95, max_depth=8, min_child_samples=180, subsample=0.78, colsample_bytree=0.68, reg_alpha=0.3, reg_lambda=4.0, bagging_freq=4, num_boost_round=1600, early_stopping_rounds=100),
        ]
    candidates = []
    for index, params in enumerate(raw_candidates[: tuning_cfg["candidates_per_round"]], start=1):
        candidates.append(
            {
                "name": f"heuristic_r{round_index}_{index}",
                "params": sanitize_candidate_params(params, tuning_cfg),
                "reason": diagnosis,
            }
        )
    return {
        "round": round_index,
        "advisor_type": "local_heuristic_fallback",
        "diagnosis": diagnosis,
        "candidates": candidates,
        "stop": False,
    }


def validate_tuning_plan(
    plan: dict[str, Any],
    tuning_cfg: dict[str, Any],
    *,
    advisor_type: str,
    expected_experiment: str | None = None,
    expected_round: int | None = None,
) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise ValueError("advisor output must be a JSON object")
    if expected_experiment:
        actual_experiment = str(plan.get("experiment") or "")
        if actual_experiment and actual_experiment != expected_experiment:
            raise ValueError(f"experiment mismatch: expected {expected_experiment}, got {actual_experiment}")
    round_index = int(plan.get("round", 0) or 0)
    if expected_round is not None and round_index != int(expected_round):
        raise ValueError(f"round mismatch: expected {expected_round}, got {round_index}")
    candidates = plan.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("advisor output requires candidates list")
    max_candidates = int(tuning_cfg.get("candidates_per_round", 1) or 1)
    if not plan.get("stop"):
        if len(candidates) < 1 or len(candidates) > max_candidates:
            raise ValueError(f"candidate count must be between 1 and {max_candidates}")
    normalized = []
    for index, item in enumerate(candidates, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"candidate {index} must be an object")
        raw_params = item.get("params") if isinstance(item.get("params"), dict) else {}
        _validate_candidate_params(raw_params, tuning_cfg, index=index)
        params = sanitize_candidate_params(raw_params, tuning_cfg)
        if not params:
            raise ValueError(f"candidate {index} has no executable params")
        normalized.append(
            {
                "name": str(item.get("name") or f"candidate_{index}"),
                "params": params,
                "reason": str(item.get("reason") or ""),
            }
        )
    if not normalized and not plan.get("stop"):
        raise ValueError("advisor did not return executable candidates")
    return {
        "round": round_index,
        "experiment": str(plan.get("experiment") or expected_experiment or ""),
        "advisor_type": advisor_type,
        "diagnosis": str(plan.get("diagnosis") or ""),
        "candidates": normalized,
        "stop": bool(plan.get("stop", False)),
    }


def _validate_candidate_params(params: dict[str, Any], tuning_cfg: dict[str, Any], *, index: int) -> None:
    bounds = tuning_cfg.get("param_bounds") or _merge_param_bounds({})
    for key, value in (params or {}).items():
        if key in LGB_INTEGER_BOUNDS:
            lower, upper = bounds["integer"].get(key, LGB_INTEGER_BOUNDS[key])
            numeric = float(value)
            if numeric < lower or numeric > upper:
                raise ValueError(f"candidate {index} param {key} outside allowed bounds")
        elif key in LGB_NUMERIC_BOUNDS:
            lower, upper = bounds["numeric"].get(key, LGB_NUMERIC_BOUNDS[key])
            numeric = float(value)
            if numeric < lower or numeric > upper:
                raise ValueError(f"candidate {index} param {key} outside allowed bounds")
        else:
            raise ValueError(f"candidate {index} param {key} is unsupported")


def _merge_param_bounds(override: dict[str, Any]) -> dict[str, dict[str, tuple[float, float] | tuple[int, int]]]:
    numeric = deepcopy(LGB_NUMERIC_BOUNDS)
    integer = deepcopy(LGB_INTEGER_BOUNDS)
    for key, value in override.items():
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            continue
        if key in numeric:
            numeric[key] = (float(value[0]), float(value[1]))
        elif key in integer:
            integer[key] = (int(value[0]), int(value[1]))
    return {"numeric": numeric, "integer": integer}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _merge_params(base: dict[str, Any], **override: Any) -> dict[str, Any]:
    merged = dict(base)
    merged.update(override)
    return merged


def _public_trial(trial: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in trial.items() if not key.startswith("_")}


def _best_public_trial(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    public = [_public_trial(item) for item in history]
    if not public:
        return None
    return sorted(public, key=lambda item: (float(item.get("valid_ks", 0.0) or 0.0), float(item.get("valid_auc", 0.0) or 0.0)), reverse=True)[0]
