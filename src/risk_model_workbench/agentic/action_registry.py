"""Minimal semantic actions for the auxiliary-model incremental-value demo."""
from .schemas import ParameterSpec, SelectedAction, SemanticActionSpec

RULES = ["OOT_FOR_HYPERPARAMETER_TUNING_DENIED", "LABEL_CHANGE_REQUIRES_HUMAN", "NEW_EXTERNAL_DATA_SOURCE_REQUIRES_HUMAN", "PRODUCTION_PROMOTION_REQUIRES_HUMAN"]


def build_registry() -> dict[str, SemanticActionSpec]:
    rows = [
        ("inspect_incremental_value", "检查 KS、相关性、CMI 与组合价值", "diagnosis", "rmw evaluate / compare", "ADAPTER"),
        ("change_training_objective", "以一个已登记目标训练 Challenger", "experiment", None, "MOCK"),
        ("change_feature_space", "以增量特征空间训练 Challenger", "experiment", None, "MOCK"),
        ("change_sample_composition", "调整已批准的建模样本构成", "experiment", None, "MOCK"),
        ("change_label_definition", "提出新的标签定义，等待业务确认", "experiment", None, "MOCK"),
        ("compare_challenger", "比较主模型、原辅助模型、优化辅助模型的增量价值", "experiment", "rmw evaluate / compare", "ADAPTER"),
        ("ask_human", "记录需要人工作出的数据、合规或生产决策", "meta", None, "REAL"),
        ("train_global_model", "继续优化全量模型", "experiment", None, "MOCK"),
        ("segment_modeling", "训练数据驱动的客群分群子模型", "experiment", None, "MOCK"),
        ("feature_enrichment", "评估补充行为或资信特征", "experiment", None, "MOCK"),
        ("keep_kmeans", "维持 KMeans 单独分群方案", "experiment", None, "MOCK"),
        ("rule_based_segment_plus_kmeans", "联合业务规则分群与 KMeans 子分", "experiment", None, "MOCK"),
        ("return_global_model", "停止分群方向并回退全量模型", "meta", None, "MOCK"),
        ("retune_model", "在既定标签下重新调参", "experiment", None, "MOCK"),
        ("revisit_label_definition", "诊断监督标签是否代表业务目标", "diagnosis", None, "MOCK"),
        ("retrain_with_new_label", "在人工确认的新标签下训练 Challenger", "experiment", None, "MOCK"),
    ]
    result = {}
    for name, description, category, existing, mode in rows:
        params = {}
        if name == "change_training_objective":
            params["objective"] = ParameterSpec(allowed_values=["covloss"], required=True)
            params["optimization_split"] = ParameterSpec(allowed_values=["DEV", "OOS", "OOT"], required=True)
        if name == "change_feature_space":
            params["strategy"] = ParameterSpec(allowed_values=["cmi_incremental_features"], required=True)
        if name == "change_sample_composition":
            params["strategy"] = ParameterSpec(allowed_values=["approved_differentiated_sample"], required=True)
            params["new_external_source"] = ParameterSpec(kind="boolean")
        if name == "change_label_definition":
            params["label_definition"] = ParameterSpec(required=True)
        if name == "compare_challenger":
            params["production_promotion"] = ParameterSpec(kind="boolean")
        if name == "ask_human":
            params["question"] = ParameterSpec(required=True)
        if name == "segment_modeling":
            params["method"] = ParameterSpec(allowed_values=["kmeans"], required=True)
        if name == "rule_based_segment_plus_kmeans":
            params["rule_segment"] = ParameterSpec(allowed_values=["new_customer_lifecycle"], required=True)
        if name == "retrain_with_new_label":
            params["label_definition"] = ParameterSpec(allowed_values=["T15_click_value"], required=True)
        result[name] = SemanticActionSpec(
            action_type=name, description=description, category=category,
            allowed_parameters=params, underlying_existing_tool=existing,
            implementation=mode,
            implementation_note=("确定性本地暂停，无外部消息" if mode == "REAL" else "映射到既有评估/比较能力；本次结果仍由历史 fixture 提供" if mode == "ADAPTER" else "历史实验 fixture；不重训历史模型"),
            guardrails=RULES,
            training_experiments=int(name in {"change_training_objective", "change_feature_space", "change_sample_composition", "change_label_definition", "train_global_model", "segment_modeling", "feature_enrichment", "keep_kmeans", "rule_based_segment_plus_kmeans", "retune_model", "retrain_with_new_label"}),
        )
    return result


def validate_action(action: SelectedAction, registry: dict[str, SemanticActionSpec]) -> SemanticActionSpec:
    if action.action_type not in registry:
        raise ValueError(f"unregistered semantic action: {action.action_type}")
    spec = registry[action.action_type]
    unknown = set(action.parameters) - set(spec.allowed_parameters)
    if unknown:
        raise ValueError(f"unregistered parameters: {sorted(unknown)}")
    for key, definition in spec.allowed_parameters.items():
        if key not in action.parameters:
            if definition.required:
                raise ValueError(f"required parameter: {key}")
            continue
        value = action.parameters[key]
        expected = bool if definition.kind == "boolean" else str
        if type(value) is not expected or (expected is str and not value.strip()):
            raise ValueError(f"invalid parameter type/value: {key}")
        if definition.allowed_values and value not in definition.allowed_values:
            raise ValueError(f"parameter outside allowlist: {key}")
    return spec
