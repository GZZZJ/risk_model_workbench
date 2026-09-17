# 辅助模型增量价值 Agentic Modeling Demo

## 最终定位：Level 2 — Agentic Runtime Demo

本 Demo 验证的是 Runtime 闭环：

```text
State → DecisionContract → ONE Semantic Action → Outcome → State Update → Re-decision
```

并包含确定性 Guardrail，以及“Counterfactual Outcome → different next Action”的运行时分支。默认 Orchestrator 是 `DemoFixtureOrchestratorAdapter`，它是 **MOCK / deterministic demo adapter**，只重放预定义 DecisionContract，**不用于证明 LLM 自主 reasoning**。

未实现：真实 LLM 自主 decision、动态语义 RAG、真实 covloss 训练、真实 CMI 计算、生产级 Agent Runtime。

```bash
python -m risk_model_workbench.agentic.demo --scenario auxiliary_model_incremental_value --output /tmp/auxiliary-agent-demo
python -m risk_model_workbench.agentic.demo --scenario auxiliary_model_incremental_value --counterfactual --output /tmp/auxiliary-agent-counterfactual
```

## Multiple Modeling Scenarios

| Scenario | Trigger | Problem Type |
| --- | --- | --- |
| `auxiliary_model_incremental_value` | 新模型评估 | 模型体系优化 |
| `segment_specific_modeling` | 监控异常 | 客群建模优化 |
| `label_definition_diagnosis` | 效果下降 | 训练目标诊断 |

三个案例不是三个 Workflow；它们复用同一 Runtime、State、DecisionContract、ActionRegistry、Guardrail、Reducer 与 Trace：

```text
Observation → Hypothesis → Candidate Actions → Experiment → Outcome → Memory Update → Re-plan
```

- `segment_specific_modeling`：主B卡 V18 的分客群监控触发。Round 1 仅可见局部客群退化与分群候选方向；KMeans 的失败结果只作为 Round 1 Outcome 写入 Updated State。Round 2 mock 决策据该新证据选择“规则生命周期分群 + KMeans”；其改善结果仍是 mock outcome。
- `label_definition_diagnosis`：RTA V8→V9 的效果下降诊断触发。`feature_shift_checked`、`sample_shift_checked`、`label_quality_suspected` 是待检验诊断候选，不是历史监控实测事实；下一轮只生成 `retrain_with_new_label` DecisionContract，确定性 Guardrail 要求人先确认标签，未执行训练。

两者与第一个案例一样均是 **COMPOSITE DEMO SCENARIO**：知识条目有真实来源，但端到端轨迹是 Level-2 架构演示，不是 historical replay。

## COMPOSITE DEMO SCENARIO

**All knowledge elements are grounded in historical documents, but the end-to-end trajectory is constructed for demonstrating the Agentic modeling architecture and is not a historical replay of one production model iteration.**

| Source | 历史事实 | 在 Demo 中的用途 |
| --- | --- | --- |
| A：`贷中模型组/00基础知识/贷中多模型决策体系.md`，`8.1 条件互信息（CMI）` 的“来自贷前建模”实例 | 度小满：KS 0.1644、Corr(**A卡**) 0.7312、CMI 0.001402；银联：KS 0.1615、Corr(**A卡**) 0.5085、CMI 0.002043。 | 说明较高单模型 KS 不必然意味着较高增量价值。它是 **Corr(A卡)**，绝非 Corr(B卡)。 |
| B：同文件，`4.3 LGB模型融合 / 相关性损失函数探索` | 在贷中 V16–V18 融合层的探索中，covloss 可行；squareloss、klloss、jsdloss、doublelogloss 有优化稳定性或效果问题。 | 作为历史正/负经验，说明为什么 mock 决策 fixture 不重复尝试已知失败 loss。 |

这两部分不是同一训练集、同一主模型、同一 Challenger 或同一次实验。它们只能作为可追溯的 Decision Knowledge transfer，不能称为 historical replay，也不能据此声称“度小满辅助模型相对主B卡经 covloss 重训后”的真实 Outcome。

## LIGHTWEIGHT KNOWLEDGE FIXTURE

当前不是生产级 RAG。Demo 将从本地团队知识库人工提取的 `Decision Knowledge` 与 `Historical Experience` 固化为可追溯 JSON fixture，每条保留：

```text
source_path
source_heading
fact_summary
```

`EvidenceRetriever` 读取两类 fixture，并支持预留的 `as_of` cutoff；它没有 embedding、向量库、query ranking 或动态语义 retrieval。每个 fixture 的 `initial_state.metadata.not_runtime_state=true` 仅说明场景，实际 Runtime State 始终由 `scenarios.initial_state()` 生成。生产形态可以替换为 metadata + embedding 的动态 retrieval。

## Orchestrator 边界

概念上的生产 adapter 契约是：

```python
decide(
    state: ModelingState,
    available_actions: list[SemanticActionSpec],
    retrieved_knowledge: dict,
) -> DecisionContract
```

当前 [HostAgentOrchestratorAdapter](/Users/guzijun/Desktop/AI攻坚/risk_model_workbench/src/risk_model_workbench/agentic/orchestrator.py:28) 已提供文件交换适配：Runtime 写出当前 context，外部 Host-Agent 回写 schema-validated `DecisionContract`。未接入任何 SDK。

未来替换 `DemoFixtureOrchestratorAdapter` 为 `HostAgentOrchestratorAdapter` 或真实 `LLMOrchestratorAdapter` 时，Runtime、StateReducer、ActionRegistry、Guardrail 与 Outcome Contract 不需要改变。

三个 Scenario 共用同一个 `DemoFixtureOrchestratorAdapter`。它是 deterministic mock decision layer；唯一用途是可重复验证：

```text
State → DecisionContract → Action → Outcome → Updated State → Re-decision
```

它不证明 LLM autonomous reasoning，也不证明 Retrieval 目前影响了默认 mock 的决策。

样例见：[orchestrator_context_example.json](orchestrator_context_example.json) 与 [decision_contract_example.json](decision_contract_example.json)。

## 三层能力边界

| 层 | 责任 |
| --- | --- |
| LLM / Agent Decision Layer | 识别当前问题、形成 Hypothesis、读取团队经验、比较候选 Action、选择下一实验方向、解释 Outcome、决定下一步。当前默认 mock 不具备这些自主能力。 |
| Traditional ML Tool Layer | 训练模型、计算 KS/AUC/Corr/CMI、分群分析、Challenger 比较。当前 Demo 未实际运行这些模型工具。 |
| Human Layer | 业务目标最终定义、标签重大调整、新外部数据源、成本/合规、生产上线、模型体系级决策。 |

## REAL / ADAPTER / FIXTURE / MOCK

| 能力 | 标记 | 当前事实 |
| --- | --- | --- |
| State、schema、单动作校验、StateReducer、Guardrail、每轮 Runtime 调用 | REAL | 本地确定性代码。 |
| Host-Agent 文件交换、既有 `rmw evaluate / compare` 映射 | ADAPTER | 入口存在，本 Demo 未加载真实版本产物。 |
| 知识提取与 retrieval 内容 | FIXTURE | 可追溯 JSON；非动态 RAG。 |
| 默认 Orchestrator、Decision、covloss Outcome、counterfactual、模型执行、指标计算 | MOCK | 不训练、不计算、不作 LLM 推理。 |

## 已知限制

- Round 2 确实重新调用 Orchestrator，但默认 `DemoFixtureOrchestratorAdapter` 对 scenario、iteration 和已更新 State 有预写分支；它只能验证接口与轨迹结构，不代表 LLM autonomous reasoning。
- 正常 Outcome 仅表述 covloss 的历史可行性，并要求当前候选继续验证区分度；不含“当前候选 KS 可接受”的真实测量。
- Counterfactual 是明确的 `COUNTERFACTUAL MOCK`，只用于验证状态变化能改变下一 action。
- 不提供生产并发、恢复、审批流、真实数据拉取或大版本主模型设计能力。
