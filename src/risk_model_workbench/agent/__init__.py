"""Deterministic local Agent runtime for RMW version workflows.

Public conveniences are loaded lazily so low-level workspace storage can be
used by the harness runtime without importing the Executor back into itself.
"""

__all__ = [
    "bind_agent_plan",
    "init_agent_state",
    "load_agent_plan",
    "load_agent_state",
    "run_agent",
    "save_agent_plan",
]


def __getattr__(name: str):
    if name == "run_agent":
        from risk_model_workbench.agent.executor import run_agent

        return run_agent
    if name in {"bind_agent_plan", "load_agent_plan", "save_agent_plan"}:
        from risk_model_workbench.agent import plan

        return getattr(plan, name)
    if name in {"init_agent_state", "load_agent_state"}:
        from risk_model_workbench.agent import state

        return getattr(state, name)
    raise AttributeError(name)
