"""LangGraph orchestration for a standalone RMW Agent."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Callable, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from risk_model_workbench.agent.advisor import load_advisor_request
from risk_model_workbench.agent.embedded_advisor import answer_advisor_request
from risk_model_workbench.agent.executor import run_agent
from risk_model_workbench.agent.model_gateway import (
    AgentModelConfig,
    ModelGateway,
    ModelGatewayError,
    build_model_gateway,
)
from risk_model_workbench.agent.state import load_agent_state
from risk_model_workbench.agent.trace import append_trace
from risk_model_workbench.versioning import resolve_workspace_dir


TERMINAL_STATUSES = {"done", "done_with_gaps", "failed", "stopped", "blocked"}
HUMAN_GATE_STATUSES = {"waiting_for_approval", "waiting_for_user", "reconciliation_required"}


class EmbeddedAgentState(TypedDict, total=False):
    project: str
    version_id: str
    workspace: str
    status: str
    blocker: dict[str, Any]
    graph_steps: int
    model_calls: int
    max_model_calls: int
    advisor_answered: bool
    last_error: str
    human_resume: dict[str, Any]


HarnessRunner = Callable[[Path, str], dict[str, Any]]


def build_embedded_agent_graph(
    *,
    gateway: ModelGateway | None = None,
    harness_runner: HarnessRunner = run_agent,
    checkpointer: Any | None = None,
):
    """Compile the standalone orchestration graph around the RMW Harness."""
    resolved_gateway = gateway

    def harness_node(state: EmbeddedAgentState) -> dict[str, Any]:
        result = harness_runner(Path(state["project"]), state["version_id"])
        return {
            "status": str(result.get("status") or "failed"),
            "blocker": dict(result.get("blocker") or {}),
            "graph_steps": int(state.get("graph_steps", 0)) + 1,
            "advisor_answered": False,
            "last_error": "",
        }

    def route_after_harness(state: EmbeddedAgentState) -> str:
        status = str(state.get("status") or "")
        if status == "waiting_for_advisor":
            if int(state.get("model_calls", 0)) >= int(state.get("max_model_calls", 4)):
                return "finish"
            return "advisor"
        if status in HUMAN_GATE_STATUSES:
            return "human_gate"
        if status in TERMINAL_STATUSES:
            return "finish"
        return "harness"

    def advisor_node(state: EmbeddedAgentState) -> dict[str, Any]:
        nonlocal resolved_gateway
        workspace = Path(state["workspace"])
        blocker = state.get("blocker") if isinstance(state.get("blocker"), dict) else {}
        request_id = str(blocker.get("advisor_request_id") or "")
        model_calls = int(state.get("model_calls", 0)) + 1
        if not request_id:
            return {
                "advisor_answered": False,
                "last_error": "advisor blocker has no request identity",
                "model_calls": model_calls,
            }
        try:
            request = load_advisor_request(workspace, request_id)
            if request.get("status") in {"answered", "rejected", "consumed"}:
                return {"advisor_answered": True, "model_calls": model_calls, "last_error": ""}
            if resolved_gateway is None:
                resolved_gateway = build_model_gateway(AgentModelConfig.from_env())
            result = answer_advisor_request(workspace, request_id, resolved_gateway)
            append_trace(
                workspace,
                "decision",
                {
                    "summary": "Embedded Agent answered an Advisor request.",
                    "advisor_request_id": request_id,
                    "decision": result.get("decision", ""),
                    "model_invocation": result.get("model_invocation", ""),
                },
            )
            return {"advisor_answered": True, "model_calls": model_calls, "last_error": ""}
        except (ModelGatewayError, ValueError, OSError, RuntimeError) as exc:
            error = f"{type(exc).__name__}: {str(exc)[:500]}"
            _record_reasoning_failure(workspace, request_id, error)
            append_trace(
                workspace,
                "decision",
                {
                    "summary": "Embedded Agent failed closed while answering Advisor request.",
                    "advisor_request_id": request_id,
                    "error_type": type(exc).__name__,
                },
            )
            return {"advisor_answered": False, "model_calls": model_calls, "last_error": error}

    def route_after_advisor(state: EmbeddedAgentState) -> str:
        return "harness" if state.get("advisor_answered") else "finish"

    def human_gate_node(state: EmbeddedAgentState) -> dict[str, Any]:
        decision = interrupt(
            {
                "type": "rmw_human_gate",
                "status": state.get("status", ""),
                "version_id": state.get("version_id", ""),
                "blocker": state.get("blocker", {}),
                "instruction": "Complete the required RMW approval/confirmation/reconciliation, then resume the graph.",
            }
        )
        return {
            "human_resume": decision if isinstance(decision, dict) else {"value": decision},
            "graph_steps": int(state.get("graph_steps", 0)) + 1,
        }

    def finish_node(state: EmbeddedAgentState) -> dict[str, Any]:
        return {"graph_steps": int(state.get("graph_steps", 0)) + 1}

    graph = StateGraph(EmbeddedAgentState)
    graph.add_node("harness", harness_node)
    graph.add_node("advisor", advisor_node)
    graph.add_node("human_gate", human_gate_node)
    graph.add_node("finish", finish_node)
    graph.add_edge(START, "harness")
    graph.add_conditional_edges(
        "harness",
        route_after_harness,
        {"harness": "harness", "advisor": "advisor", "human_gate": "human_gate", "finish": "finish"},
    )
    graph.add_conditional_edges("advisor", route_after_advisor, {"harness": "harness", "finish": "finish"})
    graph.add_edge("human_gate", "harness")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer)


def run_embedded_agent(
    project_dir: str | Path,
    version_id: str,
    *,
    gateway: ModelGateway | None = None,
    harness_runner: HarnessRunner = run_agent,
    resume_human_gate: bool = False,
    max_graph_steps: int = 32,
    max_model_calls: int = 4,
) -> dict[str, Any]:
    """Run or resume the embedded graph with a workspace-local checkpoint."""
    project = Path(project_dir).resolve()
    workspace = resolve_workspace_dir(project, version_id=version_id)
    checkpoint_path = workspace / "audit" / "agent_graph.sqlite"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
    try:
        checkpointer = SqliteSaver(connection)
        checkpointer.setup()
        graph = build_embedded_agent_graph(
            gateway=gateway,
            harness_runner=harness_runner,
            checkpointer=checkpointer,
        )
        config = {
            "configurable": {"thread_id": f"{project.name}:{version_id}"},
            "recursion_limit": max_graph_steps,
        }
        snapshot = graph.get_state(config)
        has_interrupt = any(getattr(task, "interrupts", ()) for task in snapshot.tasks)
        if resume_human_gate:
            graph_input: Any = (
                Command(resume={"actor": "rmw_operator", "resume": True})
                if has_interrupt
                else {
                    "project": str(project),
                    "version_id": version_id,
                    "workspace": str(workspace),
                    "status": "",
                    "blocker": {},
                    "graph_steps": 0,
                    "model_calls": 0,
                    "max_model_calls": max_model_calls,
                    "advisor_answered": False,
                    "last_error": "",
                }
            )
        else:
            if has_interrupt:
                current = load_agent_state(workspace)
                return {
                    "status": current.get("status", ""),
                    "blocker": current.get("blocker", {}),
                    "interrupted": True,
                    "checkpoint": str(checkpoint_path),
                }
            graph_input = {
                "project": str(project),
                "version_id": version_id,
                "workspace": str(workspace),
                "status": "",
                "blocker": {},
                "graph_steps": 0,
                "model_calls": 0,
                "max_model_calls": max_model_calls,
                "advisor_answered": False,
                "last_error": "",
            }
        result = graph.invoke(graph_input, config=config)
        current = load_agent_state(workspace)
        return {
            "status": current.get("status", result.get("status", "")),
            "blocker": current.get("blocker", result.get("blocker", {})),
            "interrupted": "__interrupt__" in result,
            "checkpoint": str(checkpoint_path),
            "graph_steps": result.get("graph_steps", 0),
            "model_calls": result.get("model_calls", 0),
            "last_error": result.get("last_error", ""),
        }
    finally:
        connection.close()


def _record_reasoning_failure(workspace: Path, request_id: str, error: str) -> None:
    from risk_model_workbench.agent.workspace_store import WorkspaceStore

    relative = Path("audit") / "model_invocations" / f"{request_id}.failure.json"
    payload = {
        "version": 1,
        "request_id": request_id,
        "status": "failed_closed",
        "error": error,
    }
    WorkspaceStore(workspace).atomic_write(relative, payload)
