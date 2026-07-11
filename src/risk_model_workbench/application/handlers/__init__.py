"""Production action handlers shared by the CLI and Agent Runtime."""

from risk_model_workbench.application.action_runner import HandlerRegistry
from risk_model_workbench.application.handlers.evaluate import run_compare, run_evaluate
from risk_model_workbench.application.handlers.report import run_report


def production_handler_registry() -> HandlerRegistry:
    registry = HandlerRegistry()
    registry.register("evaluate", run_evaluate)
    registry.register("compare", run_compare)
    registry.register("report", run_report)
    return registry


__all__ = ["production_handler_registry", "run_compare", "run_evaluate", "run_report"]
