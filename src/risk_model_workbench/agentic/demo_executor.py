"""Historical-evidence fixtures only: this module does not train a model."""
from .schemas import ActionOutcome, ModelingState, SelectedAction


class DemoActionExecutor:
    """MOCK executor whose facts are constrained to the cited knowledge fixture."""
    def __init__(self, counterfactual: bool = False):
        self.counterfactual = counterfactual

    def execute(self, state: ModelingState, action: SelectedAction) -> ActionOutcome:
        base = dict(outcome_id=f"out_{state.state_id}", action_id=f"act_{state.state_id}", action_type=action.action_type, state_id_before=state.state_id, implementation="MOCK")
        scenario = state.business_context.get("scenario_id")
        if scenario == "segment_specific_modeling":
            if action.action_type == "segment_modeling":
                return ActionOutcome(**base, execution_status="success", summary="历史 fixture：KMeans 单独分群使次新户 OOT KS 从 0.2247 降至 0.2187，属于局部失败。", observations={"segmentation_result": {"status": "partially_failed", "new_customer_ks_before": 0.2247, "new_customer_ks_after": 0.2187}, "negative_evidence": ["KMeans alone hurts new_customer_segment"]}, evidence=["S2_KMEANS_NEGATIVE"])
            if action.action_type == "rule_based_segment_plus_kmeans":
                return ActionOutcome(**base, execution_status="success", summary="历史 fixture：规则分客群子分与 KMeans 联合后，次新户 OOT KS 为 0.2352，高于 0.2247 基线。", observations={"segmentation_result": {"status": "improved", "new_customer_ks_after": 0.2352}, "negative_evidence": ["KMeans alone hurts new_customer_segment"]}, evidence=["S2_COMBINED_POSITIVE"])
        if scenario == "label_definition_diagnosis":
            if action.action_type == "revisit_label_definition":
                return ActionOutcome(**base, execution_status="success", summary="历史 fixture：V8 的 T3 曝光标签覆盖不足；V9 使用 T15 点击高低单产以覆盖更多后续转化并减少标签噪声。", observations={"label_issue": "confirmed", "label_diagnosis": {"T3_coverage": "about_75_percent", "T15_coverage": "over_90_percent", "recommended_candidate": "T15_click_value"}}, evidence=["S3_LABEL_WINDOW"])
        if action.action_type == "change_training_objective":
            if self.counterfactual:
                return ActionOutcome(**base, execution_status="success", summary="COUNTERFACTUAL MOCK：相关性改善但 KS 严重下降；仅用于检验反馈会改变决策，并非历史事实。", observations={"covloss_outcome": "corr_improved_ks_severely_degraded", "correlation_status": "improved", "discrimination_status": "severely_degraded"}, evidence=["COUNTERFACTUAL MOCK"])
            return ActionOutcome(**base, execution_status="success", summary="历史材料表明 covloss 是唯一可行的降相关损失；需要以 α/β 平衡区分度与相关性。未提供当前候选的精确重训指标。", observations={"covloss_outcome": "historically_feasible", "correlation_status": "improved_or_feasible", "discrimination_status": "requires_validation", "incremental_value_status": "promising"}, evidence=["H_COVLOSS_OUTCOME"])
        if action.action_type == "compare_challenger":
            return ActionOutcome(**base, execution_status="success", summary="历史 CMI 框架支持：辅助模型应从单模型 KS 转向 CMI 与组合增益评估；原始材料未给出本候选三模型组合的实测增益。", observations={"comparison_status": "framework_supported", "incremental_value_status": "requires_local_measurement"}, evidence=["D_CMI_METRIC", "D_CMI_EXAMPLE"])
        if action.action_type == "change_feature_space":
            return ActionOutcome(**base, execution_status="unavailable", summary="该分支没有执行：需要新的、经批准的特征实验。", evidence=[])
        if action.action_type == "ask_human":
            return ActionOutcome(**base, implementation="REAL", execution_status="waiting_for_human", summary=action.parameters["question"], evidence=["local human-review pause; no message sent"])
        return ActionOutcome(**base, execution_status="unavailable", summary="本 Demo 未实现该语义动作的历史 fixture。", evidence=[])
