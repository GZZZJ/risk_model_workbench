"""Production action handlers shared by the CLI and Agent Runtime."""

from risk_model_workbench.application.action_runner import HandlerRegistry
from risk_model_workbench.application.handlers.evaluate import run_compare, run_evaluate
from risk_model_workbench.application.handlers.feature_selection import (
    run_build_wide_sql,
    run_feature_metadata,
    run_feature_prescreen,
    run_feature_refine,
)
from risk_model_workbench.application.handlers.report import run_report
from risk_model_workbench.application.handlers.sample_check import run_sample_check
from risk_model_workbench.application.handlers.train import run_train


def production_handler_registry() -> HandlerRegistry:
    registry = HandlerRegistry()
    registry.register("evaluate", run_evaluate)
    registry.register("compare", run_compare)
    registry.register("report", run_report)
    registry.register("sample_check", run_sample_check)
    registry.register("feature_metadata", run_feature_metadata)
    registry.register("feature_prescreen", run_feature_prescreen)
    registry.register("build_wide_sql", run_build_wide_sql)
    registry.register("feature_refine", run_feature_refine)
    registry.register("train_baseline", run_train)
    return registry


__all__ = [
    "production_handler_registry",
    "run_build_wide_sql",
    "run_compare",
    "run_evaluate",
    "run_feature_metadata",
    "run_feature_prescreen",
    "run_feature_refine",
    "run_report",
    "run_sample_check",
    "run_train",
]
