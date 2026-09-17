"""Minimal think/act/observe loop. No plan, task graph, or legacy Executor import."""
from __future__ import annotations
import json
from pathlib import Path
from .action_registry import build_registry
from .demo_executor import DemoActionExecutor
from .guardrails import check_guardrails
from .orchestrator import DecisionRequired, ModelingOrchestrator
from .retrieval import EvidenceRetriever
from .schemas import ActionOutcome, ModelingState, TrajectoryStep
from .state_reducer import reduce_state


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class DemoRuntime:
    def __init__(self, adapter, executor=None, retriever=None):
        self.registry = build_registry()
        self.orchestrator = ModelingOrchestrator(adapter)
        self.executor = executor or DemoActionExecutor()
        self.retriever = retriever or EvidenceRetriever()
        self.pending = ""

    def step(self, state: ModelingState) -> TrajectoryStep:
        if state.status != "running" or state.iteration >= state.constraints.max_iterations:
            raise ValueError("session is not runnable")
        retrieved = self.retriever.retrieve(state)
        decision = self.orchestrator.decide(state, self.registry, retrieved)
        denied = check_guardrails(state, decision.selected_action, self.registry)
        if denied:
            outcome = ActionOutcome(outcome_id=f"out_{state.state_id}", action_id=f"act_{state.state_id}",
                action_type=decision.selected_action.action_type, state_id_before=state.state_id,
                execution_status="denied", implementation="REAL", summary="Deterministic guardrail blocked execution",
                guardrail_codes=denied)
        else:
            outcome = self.executor.execute(state, decision.selected_action)
        updated = reduce_state(state, decision, outcome)
        return TrajectoryStep(state_before=state, decision=decision, retrieval=retrieved,
                              outcome=outcome, state_after=updated, adapter=self.orchestrator.adapter.label)

    def run(self, state: ModelingState, output: Path | None = None) -> list[TrajectoryStep]:
        steps = []
        if output:
            write_json(output / "current.json", state)
            write_json(output / "states" / f"{state.state_id}.json", state)
        while state.status == "running" and state.iteration < state.constraints.max_iterations:
            try:
                step = self.step(state)
            except DecisionRequired as exc:
                self.pending = str(exc)
                break
            steps.append(step)
            if output:
                write_json(output / "decisions" / f"{step.decision.decision_id}.json", step.decision)
                write_json(output / "outcomes" / f"{step.outcome.outcome_id}.json", step.outcome)
                write_json(output / "states" / f"{step.state_after.state_id}.json", step.state_after)
                with (output / "trajectory.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(step.model_dump_json() + "\n")
                write_json(output / "current.json", step.state_after)
            state = step.state_after
        return steps


def render_steps(steps: list[TrajectoryStep]) -> str:
    scenario = steps[0].state_before.business_context.get("scenario_id") if steps else "agentic_demo"
    title = {"auxiliary_model_incremental_value": "辅助模型增量价值", "segment_specific_modeling": "客群专项建模", "label_definition_diagnosis": "标签定义诊断"}.get(scenario, "Agentic Modeling")
    limitation = "初始指标中的 Corr 为 Corr(A卡)，不是 Corr(B卡)。" if scenario == "auxiliary_model_incremental_value" else "各指标仅为可追溯 historical fixture，不代表当前重新训练的实测结果。"
    lines = [f"# {title} Agentic Demo Run", "", "**COMPOSITE DEMO SCENARIO**：知识元素均有历史文档依据，但端到端轨迹为架构演示构造，不是单一生产模型迭代的 historical replay。", "", f"Round 1/2 的实验事实来自 MOCK historical fixture，除非被确定性 Guardrail 拦截；未重训历史模型。{limitation}", ""]
    for index, step in enumerate(steps):
        lines.extend([f"## Round {step.state_before.iteration + 1}", "",
            f"- Adapter: {step.adapter}",
            f"- State: `{json.dumps(step.state_before.observations, ensure_ascii=False)}`",
            f"- Retrieved Decision Knowledge: {', '.join(r['id'] for r in step.retrieval['decision_knowledge']) or 'none'}",
            f"- Retrieved Historical Experience: {', '.join(r['id'] for r in step.retrieval['historical_experience']) or 'none'}",
            f"- Retrieved Negative Evidence: {', '.join(r['id'] for r in step.retrieval.get('negative_evidence', [])) or 'none'}",
            f"- Hypothesis: {step.decision.primary_hypothesis.description}",
            f"- Retrieved: {', '.join(step.decision.retrieved_refs)}",
            f"- Selected Action: `{step.decision.selected_action.action_type}`",
            f"- Parameters: `{json.dumps(step.decision.selected_action.parameters, ensure_ascii=False)}`",
            f"- Outcome ({step.outcome.implementation}): {step.outcome.summary}",
            f"- Guardrail: {step.outcome.guardrail_codes or 'passed'}",
            f"- Updated State: `{json.dumps(step.state_after.observations, ensure_ascii=False)}`",
            f"- Status: {step.state_after.status}",
            f"- Next Action (实际下一轮): {steps[index + 1].decision.selected_action.action_type if index + 1 < len(steps) else '无；暂停或等待决策'}", ""])
    return "\n".join(lines)
