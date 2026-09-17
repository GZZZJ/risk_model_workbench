"""One-decision boundary: Host-Agent adapter or explicitly replayed demo fixtures."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Protocol
from .action_registry import validate_action
from .schemas import DecisionContract, ModelingState, SemanticActionSpec

SYSTEM_PROMPT = """你是信贷风险建模 Orchestrator。基于当前 ModelingState、两类检索证据和动作注册表，只输出一个 DecisionContract JSON：诊断当前瓶颈、提出本轮假设、排序候选动作并仅选择一个动作。不得给出未来 workflow、SQL、代码或多个连续动作。历史失败证据应降低重复实验优先级。上一轮 Outcome 是事实；下一轮必须重新决策。OOT 不能用于训练目标调优，标签、新外部数据和生产接入需人工。"""

class DecisionRequired(RuntimeError): pass
class DecisionAdapter(Protocol):
    label: str
    def decide(self, context: dict) -> dict: ...

class HostAgentFileAdapter:
    label = "ADAPTER: Host-Agent file exchange (no embedded LLM SDK)"
    def __init__(self, directory: str | Path): self.directory = Path(directory)
    def decide(self, context: dict) -> dict:
        self.directory.mkdir(parents=True, exist_ok=True)
        state_id = context["state"]["state_id"]
        request, response = self.directory / f"{state_id}.input.json", self.directory / f"{state_id}.decision.json"
        request.write_text(json.dumps(context, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if not response.exists(): raise DecisionRequired(f"Host-Agent: read {request}; write {response}; rerun")
        return json.loads(response.read_text(encoding="utf-8"))

# Name retained for callers that prefer the architecture-oriented adapter name.
HostAgentOrchestratorAdapter = HostAgentFileAdapter

class DemoFixtureOrchestratorAdapter:
    """MOCK / deterministic demo adapter; replays contracts, never LLM reasoning."""
    label = "MOCK / deterministic demo adapter: pre-defined DecisionContract replay"
    def decide(self, context: dict) -> dict:
        state, obs = context["state"], context["state"]["observations"]
        scenario = state["business_context"].get("scenario_id")
        if scenario == "segment_specific_modeling":
            if state["iteration"] == 0:
                selected = ("segment_modeling", {"method": "kmeans"}, "监控发现次新户与低活跃客群局部退化；优先验证全量模型无法覆盖的客群模式。", "检查 KMeans 分群是否改善次新户 KS。", "other", ["segment_modeling", "train_global_model", "feature_enrichment"], ["S2_MONITORING"])
            else:
                selected = ("rule_based_segment_plus_kmeans", {"rule_segment": "new_customer_lifecycle"}, "KMeans 单独使次新户 KS 下降，Updated State 中的新负向证据不支持维持纯聚类；尝试业务生命周期规则与 KMeans 联合。", "检查联合方案是否改善次新户表现。", "other", ["rule_based_segment_plus_kmeans", "keep_kmeans", "return_global_model"], ["S2_MONITORING"])
        elif scenario == "label_definition_diagnosis":
            if state["iteration"] == 0:
                selected = ("revisit_label_definition", {}, "特征与样本诊断未定位原因而标签质量可疑；先检查监督目标是否覆盖真实业务转化。", "确认 T3 标签覆盖与 T15 候选标签的业务代表性。", "other", ["revisit_label_definition", "retune_model", "feature_enrichment"], ["S3_LABEL_WINDOW", "S3_T3_LIMITATION"])
            else:
                selected = ("retrain_with_new_label", {"label_definition": "T15_click_value"}, "标签问题已确认；在人工确认标签定义后，下一步应训练 Challenger，而非继续在旧标签上调参。", "由人工确认标签后训练，并在同口径评估。", "other", ["retrain_with_new_label", "retune_model", "ask_human"], ["S3_LABEL_WINDOW"])
        elif state["iteration"] == 0:
            selected = ("change_training_objective", {"objective": "covloss", "optimization_split": "OOS"}, "候选 KS 尚可但相关性高、CMI低，瓶颈是信息冗余；历史记录显示 covloss 可行且其他四种 loss 已失败。", "验证 covloss 方向是否可行，并记录区分度与相关性权衡。", "information_redundancy", ["change_training_objective", "change_feature_space", "change_sample_composition"], ["D_CMI_EXAMPLE", "H_COVLOSS_OUTCOME"])
        elif obs.get("covloss_outcome") == "corr_improved_ks_severely_degraded":
            selected = ("change_feature_space", {"strategy": "cmi_incremental_features"}, "相关性改善但区分度严重退化，不能直接进入组合价值比较；转向增量特征空间。", "寻找主模型未覆盖的特征信号。", "information_redundancy", ["change_feature_space", "change_sample_composition", "ask_human"], ["D_DIFFERENTIATION"])
        else:
            selected = ("compare_challenger", {}, "历史上 covloss 可行后，下一未知是是否真正增加组合价值。", "比较主模型、原辅助模型、优化辅助模型的 KS、Corr、CMI 与组合增益。", "incremental_value", ["compare_challenger", "change_feature_space", "ask_human"], ["D_CMI_METRIC", "H_COVLOSS_USED"])
        action, parameters, reason, expected, claim, candidates, refs = selected
        return {"decision_id": f"decision_{state['state_id']}", "state_id": state["state_id"], "state_summary": reason, "primary_hypothesis": {"claim": claim, "description": reason, "status": "untested"}, "candidate_actions": [{"action_type": name, "priority": i + 1, "reason": "MOCK fixture candidate"} for i, name in enumerate(candidates)], "selected_action": {"action_type": action, "parameters": parameters}, "reason": "MOCK fixture replay；Host-Agent 模式才是实际 LLM 决策接口。", "expected_evidence": expected, "open_questions": [expected], "retrieved_refs": refs}

class ModelingOrchestrator:
    def __init__(self, adapter: DecisionAdapter): self.adapter = adapter
    def decide(self, state: ModelingState, registry: dict[str, SemanticActionSpec], retrieved: dict) -> DecisionContract:
        context = {"system_prompt": SYSTEM_PROMPT, "state": state.model_dump(mode="json"), "available_actions": [spec.model_dump(mode="json") for spec in registry.values()], "retrieved": retrieved, "decision_schema": DecisionContract.model_json_schema()}
        decision = DecisionContract.model_validate(self.adapter.decide(context))
        if decision.state_id != state.state_id: raise ValueError("decision does not bind the current state")
        validate_action(decision.selected_action, registry)
        if any(c.action_type not in registry for c in decision.candidate_actions): raise ValueError("unregistered candidate")
        available_refs = {row["id"] for rows in retrieved.values() if isinstance(rows, list) for row in rows}
        if not set(decision.retrieved_refs) <= available_refs: raise ValueError("decision cites evidence that was not retrieved")
        return decision
