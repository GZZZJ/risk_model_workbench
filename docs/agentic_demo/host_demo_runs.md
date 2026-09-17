# 实际 Host-Agent 演示记录

本记录由本次 Codex 会话作为 Host-Agent，每轮读取新 input.json 后单独提交 DecisionContract。未使用 DemoRetrievalAdapter 生成这些决策。数据和训练 Outcome 为合成 fixture；不是实际模型训练效果。


## Scenario A


### Run

所有场景数据和训练结果均为合成演示；adapter 类型逐轮记录。

### Round 1

- Adapter: ADAPTER: Host-Agent file exchange (no embedded LLM SDK)
- State: `{"oot_ks_delta": -0.08, "score_psi": 0.03, "new_customer_share_delta": 0.25, "ins_ks": 0.42, "oos_ks": 0.38, "oot_ks": 0.3}`
- Hypothesis: 效果退化可能集中于新客，先验证是否存在客群局部性。
- Retrieved: K_SEGMENT, DEMO_V18, DEMO_FAILED_TUNE
- Selected Action: `inspect_segment_performance`
- Parameters: `{"segment_dimension": "customer_type"}`
- Outcome (MOCK/DEMO): 合成场景事实：{'old_customer_ks_delta': -0.01, 'new_customer_ks_delta': -0.12, 'old_customer_n': 7000, 'new_customer_n': 3000, 'segment_localized': True, 'segment_uniform': False}
- Guardrail: passed
- Updated State: `{"oot_ks_delta": -0.08, "score_psi": 0.03, "new_customer_share_delta": 0.25, "ins_ks": 0.42, "oos_ks": 0.38, "oot_ks": 0.3, "old_customer_ks_delta": -0.01, "new_customer_ks_delta": -0.12, "old_customer_n": 7000, "new_customer_n": 3000, "segment_localized": true, "segment_uniform": false}`
- Status: running
- Next Action (实际下一轮): inspect_sample_shift

### Round 2

- Adapter: ADAPTER: Host-Agent file exchange (no embedded LLM SDK)
- State: `{"oot_ks_delta": -0.08, "score_psi": 0.03, "new_customer_share_delta": 0.25, "ins_ks": 0.42, "oos_ks": 0.38, "oot_ks": 0.3, "old_customer_ks_delta": -0.01, "new_customer_ks_delta": -0.12, "old_customer_n": 7000, "new_customer_n": 3000, "segment_localized": true, "segment_uniform": false}`
- Hypothesis: 开发样本对当前新客的代表性可能不足，尚未确认原因。
- Retrieved: K_SAMPLE, DEMO_FAILED_TUNE
- Selected Action: `inspect_sample_shift`
- Parameters: `{}`
- Outcome (MOCK/DEMO): 合成场景事实：{'reference_new_customer_share': 0.2, 'current_new_customer_share': 0.45, 'reference_n': 10000, 'current_n': 10000, 'reference_bad_rate': 0.08, 'current_bad_rate': 0.1, 'sample_shift_checked': True}
- Guardrail: passed
- Updated State: `{"oot_ks_delta": -0.08, "score_psi": 0.03, "new_customer_share_delta": 0.25, "ins_ks": 0.42, "oos_ks": 0.38, "oot_ks": 0.3, "old_customer_ks_delta": -0.01, "new_customer_ks_delta": -0.12, "old_customer_n": 7000, "new_customer_n": 3000, "segment_localized": true, "segment_uniform": false, "reference_new_customer_share": 0.2, "current_new_customer_share": 0.45, "reference_n": 10000, "current_n": 10000, "reference_bad_rate": 0.08, "current_bad_rate": 0.1, "sample_shift_checked": true}`
- Status: running
- Next Action (实际下一轮): ask_human

### Round 3

- Adapter: ADAPTER: Host-Agent file exchange (no embedded LLM SDK)
- State: `{"oot_ks_delta": -0.08, "score_psi": 0.03, "new_customer_share_delta": 0.25, "ins_ks": 0.42, "oos_ks": 0.38, "oot_ks": 0.3, "old_customer_ks_delta": -0.01, "new_customer_ks_delta": -0.12, "old_customer_n": 7000, "new_customer_n": 3000, "segment_localized": true, "segment_uniform": false, "reference_new_customer_share": 0.2, "current_new_customer_share": 0.45, "reference_n": 10000, "current_n": 10000, "reference_bad_rate": 0.08, "current_bad_rate": 0.1, "sample_shift_checked": true}`
- Hypothesis: 样本代表性变化值得关注，但当前统计口径不足以确认根因或直接改变样本。
- Retrieved: K_HUMAN_SAMPLE_SHIFT_CHECKED, K_SAMPLE, DEMO_FAILED_TUNE
- Selected Action: `ask_human`
- Parameters: `{"question": "请确认分群 KS 的 3000/10000 新客样本与结构统计 45% 的分母、标签成熟度是否一致；明确业务样本口径后再讨论窗口候选。"}`
- Outcome (REAL): 请确认分群 KS 的 3000/10000 新客样本与结构统计 45% 的分母、标签成熟度是否一致；明确业务样本口径后再讨论窗口候选。
- Guardrail: passed
- Updated State: `{"oot_ks_delta": -0.08, "score_psi": 0.03, "new_customer_share_delta": 0.25, "ins_ks": 0.42, "oos_ks": 0.38, "oot_ks": 0.3, "old_customer_ks_delta": -0.01, "new_customer_ks_delta": -0.12, "old_customer_n": 7000, "new_customer_n": 3000, "segment_localized": true, "segment_uniform": false, "reference_new_customer_share": 0.2, "current_new_customer_share": 0.45, "reference_n": 10000, "current_n": 10000, "reference_bad_rate": 0.08, "current_bad_rate": 0.1, "sample_shift_checked": true}`
- Status: waiting_for_human
- Next Action (实际下一轮): 无；暂停或等待决策

## Scenario B


### Run

所有场景数据和训练结果均为合成演示；adapter 类型逐轮记录。

### Round 1

- Adapter: ADAPTER: Host-Agent file exchange (no embedded LLM SDK)
- State: `{"oot_ks_delta": -0.06, "score_psi": 0.04, "new_customer_share_delta": 0.02, "ins_ks": 0.65, "oos_ks": 0.36, "oot_ks": 0.34}`
- Hypothesis: 存在较大泛化差距，过拟合是候选原因，仍需排除泄漏和样本偏差。
- Retrieved: K_OVERFIT, DEMO_V18
- Selected Action: `inspect_overfitting`
- Parameters: `{}`
- Outcome (REAL): INS-OOS KS gap=0.290；这是泛化差距证据，不是过拟合因果证明。输入为模拟指标。
- Guardrail: passed
- Updated State: `{"oot_ks_delta": -0.06, "score_psi": 0.04, "new_customer_share_delta": 0.02, "ins_ks": 0.65, "oos_ks": 0.36, "oot_ks": 0.34, "ins_oos_ks_gap": 0.29, "ins_oot_ks_gap": 0.31, "overfitting_signal_confirmed": true}`
- Status: running
- Next Action (实际下一轮): tune_model

### Round 2

- Adapter: ADAPTER: Host-Agent file exchange (no embedded LLM SDK)
- State: `{"oot_ks_delta": -0.06, "score_psi": 0.04, "new_customer_share_delta": 0.02, "ins_ks": 0.65, "oos_ks": 0.36, "oot_ks": 0.34, "ins_oos_ks_gap": 0.29, "ins_oot_ks_gap": 0.31, "overfitting_signal_confirmed": true}`
- Hypothesis: 受控正则化实验可能缩小泛化差距；本次仅在模拟场景中检验。
- Retrieved: K_TUNE, DEMO_RTA_V9
- Selected Action: `tune_model`
- Parameters: `{"optimization_split": "OOS"}`
- Outcome (MOCK/DEMO): 合成场景事实：{'challenger_ready': True, 'selection_split': 'OOS', 'candidate_count': 1, 'challenger_ins_ks': 0.49, 'challenger_oos_ks': 0.4, 'challenger_id': 'synthetic_regularized_v1'}
- Guardrail: passed
- Updated State: `{"oot_ks_delta": -0.06, "score_psi": 0.04, "new_customer_share_delta": 0.02, "ins_ks": 0.65, "oos_ks": 0.36, "oot_ks": 0.34, "ins_oos_ks_gap": 0.29, "ins_oot_ks_gap": 0.31, "overfitting_signal_confirmed": true, "challenger_ready": true, "selection_split": "OOS", "candidate_count": 1, "challenger_ins_ks": 0.49, "challenger_oos_ks": 0.4, "challenger_id": "synthetic_regularized_v1"}`
- Status: running
- Next Action (实际下一轮): compare_challenger

### Round 3

- Adapter: ADAPTER: Host-Agent file exchange (no embedded LLM SDK)
- State: `{"oot_ks_delta": -0.06, "score_psi": 0.04, "new_customer_share_delta": 0.02, "ins_ks": 0.65, "oos_ks": 0.36, "oot_ks": 0.34, "ins_oos_ks_gap": 0.29, "ins_oot_ks_gap": 0.31, "overfitting_signal_confirmed": true, "challenger_ready": true, "selection_split": "OOS", "candidate_count": 1, "challenger_ins_ks": 0.49, "challenger_oos_ks": 0.4, "challenger_id": "synthetic_regularized_v1"}`
- Hypothesis: 冻结 Challenger 的 OOS 改善是否伴随可接受的时间外与客群表现仍需观察。
- Retrieved: K_COMPARE, DEMO_MULTI
- Selected Action: `compare_challenger`
- Parameters: `{}`
- Outcome (MOCK/DEMO): 合成场景事实：{'comparison_complete': True, 'champion_oot_ks': 0.34, 'challenger_oot_ks': 0.35, 'challenger_score_psi': 0.04, 'worst_segment_ks_delta': -0.01}
- Guardrail: passed
- Updated State: `{"oot_ks_delta": -0.06, "score_psi": 0.04, "new_customer_share_delta": 0.02, "ins_ks": 0.65, "oos_ks": 0.36, "oot_ks": 0.34, "ins_oos_ks_gap": 0.29, "ins_oot_ks_gap": 0.31, "overfitting_signal_confirmed": true, "challenger_ready": true, "selection_split": "OOS", "candidate_count": 1, "challenger_ins_ks": 0.49, "challenger_oos_ks": 0.4, "challenger_id": "synthetic_regularized_v1", "comparison_complete": true, "champion_oot_ks": 0.34, "challenger_oot_ks": 0.35, "challenger_score_psi": 0.04, "worst_segment_ks_delta": -0.01}`
- Status: running
- Next Action (实际下一轮): ask_human

### Round 4

- Adapter: ADAPTER: Host-Agent file exchange (no embedded LLM SDK)
- State: `{"oot_ks_delta": -0.06, "score_psi": 0.04, "new_customer_share_delta": 0.02, "ins_ks": 0.65, "oos_ks": 0.36, "oot_ks": 0.34, "ins_oos_ks_gap": 0.29, "ins_oot_ks_gap": 0.31, "overfitting_signal_confirmed": true, "challenger_ready": true, "selection_split": "OOS", "candidate_count": 1, "challenger_ins_ks": 0.49, "challenger_oos_ks": 0.4, "challenger_id": "synthetic_regularized_v1", "comparison_complete": true, "champion_oot_ks": 0.34, "challenger_oot_ks": 0.35, "challenger_score_psi": 0.04, "worst_segment_ks_delta": -0.01}`
- Hypothesis: Challenger 有小幅模拟整体增益，但统计不确定性和局部退化仍未解决。
- Retrieved: K_HUMAN_COMPARISON_COMPLETE, DEMO_MULTI
- Selected Action: `ask_human`
- Parameters: `{"question": "请评审冻结 Challenger：模拟 OOT KS +1pp、最差客群 -1pp；尚无置信区间和真实数据验证，不据此上线，也不再使用 OOT 调参。"}`
- Outcome (REAL): 请评审冻结 Challenger：模拟 OOT KS +1pp、最差客群 -1pp；尚无置信区间和真实数据验证，不据此上线，也不再使用 OOT 调参。
- Guardrail: passed
- Updated State: `{"oot_ks_delta": -0.06, "score_psi": 0.04, "new_customer_share_delta": 0.02, "ins_ks": 0.65, "oos_ks": 0.36, "oot_ks": 0.34, "ins_oos_ks_gap": 0.29, "ins_oot_ks_gap": 0.31, "overfitting_signal_confirmed": true, "challenger_ready": true, "selection_split": "OOS", "candidate_count": 1, "challenger_ins_ks": 0.49, "challenger_oos_ks": 0.4, "challenger_id": "synthetic_regularized_v1", "comparison_complete": true, "champion_oot_ks": 0.34, "challenger_oot_ks": 0.35, "challenger_score_psi": 0.04, "worst_segment_ks_delta": -0.01}`
- Status: waiting_for_human
- Next Action (实际下一轮): 无；暂停或等待决策
