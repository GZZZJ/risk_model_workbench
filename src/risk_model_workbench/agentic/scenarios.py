"""Initial ModelingState fixtures for the three Level-2 demo scenarios."""
from .schemas import Hypothesis, ModelingState

SCENARIO = "auxiliary_model_incremental_value"
SEGMENT_SCENARIO = "segment_specific_modeling"
LABEL_SCENARIO = "label_definition_diagnosis"
SCENARIOS = (SCENARIO, SEGMENT_SCENARIO, LABEL_SCENARIO)


def initial_state(scenario: str, session_id: str) -> ModelingState:
    if scenario == SCENARIO:
        return ModelingState(session_id=session_id, state_id=f"{session_id}_0", goal="提升辅助模型相对主B卡的增量风险识别能力", business_context={"scenario_id": scenario, "scene": "贷中多模型决策", "primary_model": "主B卡", "knowledge_mode": "current_knowledge_demo", "as_of": None, "scenario_type": "COMPOSITE DEMO SCENARIO", "evidence_transfer_limitation": "初始数值源自知识库中相对主A卡的贷前实例；用于主B卡场景的方法迁移，不是主B卡实测。"}, observations={"candidate_model": {"name": "度小满(dxm_yz_26v1)", "ks": 0.1644, "corr_with_reference_primary": 0.7312, "cmi": 0.001402}, "benchmark_model": {"name": "银联(yl_yz_25v1)", "ks": 0.1615, "corr_with_reference_primary": 0.5085, "cmi": 0.002043}, "source_reference_primary": "主A卡（知识库原始实例）"}, hypotheses=[Hypothesis(claim="information_redundancy", description="候选辅助模型与主模型存在较强信息冗余")], open_questions=["高相关性的主要原因是什么？", "是否能降低相关性同时保留区分能力？", "降相关以后是否真的产生组合增量？"])
    if scenario == SEGMENT_SCENARIO:
        return ModelingState(session_id=session_id, state_id=f"{session_id}_0", goal="improve_segment_risk_discrimination", business_context={"scenario_id": scenario, "scene": "主B卡 V18 分客群优化", "trigger": "segment_monitor_alert", "knowledge_mode": "current_knowledge_demo", "as_of": None, "scenario_type": "COMPOSITE DEMO SCENARIO"}, observations={"segment_metrics": {"new_customer": {"metric": "KS", "before": 0.374, "after": 0.230}, "low_activity_customer": {"status": "degraded"}}}, hypotheses=[Hypothesis(claim="other", description="全量模型可能未充分学习次新户的风险模式")], open_questions=["局部退化是否应以分群子模型处理？", "纯 KMeans 是否足够表达生命周期差异？"])
    if scenario == LABEL_SCENARIO:
        return ModelingState(session_id=session_id, state_id=f"{session_id}_0", goal="recover_model_performance", business_context={"scenario_id": scenario, "scene": "RTA V8→V9 标签窗口优化", "trigger": "model_performance_degradation", "knowledge_mode": "current_knowledge_demo", "as_of": None, "scenario_type": "COMPOSITE DEMO SCENARIO"}, observations={"model_metrics": {"conversion_ks": "decreased", "quality_ks": "decreased"}, "diagnosis_candidates": ["feature_shift_checked", "sample_shift_checked", "label_quality_suspected"]}, hypotheses=[Hypothesis(claim="other", description="性能下降可能来自训练标签不能代表真实业务目标")], open_questions=["T3 标签窗口是否覆盖足够的后续转化？", "应否先确认监督信号再调参？"])
    raise ValueError(f"unsupported scenario: {scenario}")
