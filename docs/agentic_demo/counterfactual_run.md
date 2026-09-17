# Agentic Demo Run

所有场景数据和训练结果均为合成演示；adapter 类型逐轮记录。

## Round 1

- Adapter: MOCK/DEMO: retrieval ranking surrogate, not LLM reasoning
- State: `{"oot_ks_delta": -0.08, "score_psi": 0.03, "new_customer_share_delta": 0.25, "ins_ks": 0.42, "oos_ks": 0.38, "oot_ks": 0.3}`
- Hypothesis: 整体衰退可能集中于新客，需要分群证据
- Retrieved: DEMO_V18, K_DRIFT, K_OVERFIT, K_SEGMENT
- Selected Action: `inspect_segment_performance`
- Parameters: `{"segment_dimension": "customer_type"}`
- Outcome (MOCK/DEMO): 合成场景事实：{'old_customer_ks_delta': -0.08, 'new_customer_ks_delta': -0.08, 'old_customer_n': 7000, 'new_customer_n': 3000, 'segment_localized': False, 'segment_uniform': True}
- Guardrail: passed
- Updated State: `{"oot_ks_delta": -0.08, "score_psi": 0.03, "new_customer_share_delta": 0.25, "ins_ks": 0.42, "oos_ks": 0.38, "oot_ks": 0.3, "old_customer_ks_delta": -0.08, "new_customer_ks_delta": -0.08, "old_customer_n": 7000, "new_customer_n": 3000, "segment_localized": false, "segment_uniform": true}`
- Status: running
- Next Action (实际下一轮): inspect_feature_target_relation

## Round 2

- Adapter: MOCK/DEMO: retrieval ranking surrogate, not LLM reasoning
- State: `{"oot_ks_delta": -0.08, "score_psi": 0.03, "new_customer_share_delta": 0.25, "ins_ks": 0.42, "oos_ks": 0.38, "oot_ks": 0.3, "old_customer_ks_delta": -0.08, "new_customer_ks_delta": -0.08, "old_customer_n": 7000, "new_customer_n": 3000, "segment_localized": false, "segment_uniform": true}`
- Hypothesis: 衰退可能涉及标签口径或特征风险关系变化
- Retrieved: K_DRIFT, K_RELATION
- Selected Action: `inspect_feature_target_relation`
- Parameters: `{}`
- Outcome (MOCK/DEMO): 合成场景事实：{'relation_checked': True, 'bin_bad_rate_shift': 0.04, 'label_maturity_verified': False}
- Guardrail: passed
- Updated State: `{"oot_ks_delta": -0.08, "score_psi": 0.03, "new_customer_share_delta": 0.25, "ins_ks": 0.42, "oos_ks": 0.38, "oot_ks": 0.3, "old_customer_ks_delta": -0.08, "new_customer_ks_delta": -0.08, "old_customer_n": 7000, "new_customer_n": 3000, "segment_localized": false, "segment_uniform": true, "relation_checked": true, "bin_bad_rate_shift": 0.04, "label_maturity_verified": false}`
- Status: running
- Next Action (实际下一轮): ask_human

## Round 3

- Adapter: MOCK/DEMO: retrieval ranking surrogate, not LLM reasoning
- State: `{"oot_ks_delta": -0.08, "score_psi": 0.03, "new_customer_share_delta": 0.25, "ins_ks": 0.42, "oos_ks": 0.38, "oot_ks": 0.3, "old_customer_ks_delta": -0.08, "new_customer_ks_delta": -0.08, "old_customer_n": 7000, "new_customer_n": 3000, "segment_localized": false, "segment_uniform": true, "relation_checked": true, "bin_bad_rate_shift": 0.04, "label_maturity_verified": false}`
- Hypothesis: 已有证据可供人工评审，不能替代业务决策
- Retrieved: K_HUMAN_RELATION_CHECKED
- Selected Action: `ask_human`
- Parameters: `{"question": "请评审当前证据、样本定义和实验局限；Demo 不执行上线或外部取数。"}`
- Outcome (REAL): 请评审当前证据、样本定义和实验局限；Demo 不执行上线或外部取数。
- Guardrail: passed
- Updated State: `{"oot_ks_delta": -0.08, "score_psi": 0.03, "new_customer_share_delta": 0.25, "ins_ks": 0.42, "oos_ks": 0.38, "oot_ks": 0.3, "old_customer_ks_delta": -0.08, "new_customer_ks_delta": -0.08, "old_customer_n": 7000, "new_customer_n": 3000, "segment_localized": false, "segment_uniform": true, "relation_checked": true, "bin_bad_rate_shift": 0.04, "label_maturity_verified": false}`
- Status: waiting_for_human
- Next Action (实际下一轮): 无；暂停或等待决策
