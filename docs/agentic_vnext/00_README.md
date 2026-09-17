# RMW Agentic Core vNext — Implementation Spec

本目录用于指导 `risk_model_workbench` 从“预定义完整 Workflow + 确定性执行”重构为：

> **State → Orchestrator Decision → One Semantic Action → Deterministic Execution → Outcome → State Update → Next Decision**

核心目标只有一个：

**让模型 Agent 根据当前建模状态和上一步结果动态决定下一步，而不是在任务开始前把完整工作流全部确定。**

---

## 文件说明

1. `01_modeling_state_schema.md`
   - 定义 Agent 当前“世界状态”
   - 替代过去以完整 execution plan 为中心的全局运行方式

2. `02_decision_contract_schema.md`
   - 定义 Modeling Orchestrator 每轮允许输出什么
   - 每轮只能选择一个 Semantic Action

3. `03_action_spec_schema.md`
   - 定义 Semantic Action Registry
   - LLM 只选择语义动作，不直接生成任意 SQL / Python / CLI

4. `04_outcome_schema.md`
   - 定义 Action 执行后的机器可读结果
   - Tool 只返回事实，不决定下一步

5. `05_trajectory_schema.md`
   - 定义 Agent 每轮决策轨迹
   - 为后续 Trajectory Retrieval / Preference Learning 积累数据

6. `06_rmw_migration_map.md`
   - 定义现有 RMW 模块如何迁移
   - 明确保留 / 重构 / 降级 / 新增的部分

7. `07_codex_implementation_prompt.md`
   - 可直接交给 Codex 的实施提示词
   - 要求先审计当前代码，再分阶段改造

---

## 总体原则

### 1. Workflow 下沉到 Action 内部

过去：

```text
Workflow 控制 Agent
```

未来：

```text
Agent 选择 Semantic Action
        ↓
Action 内部可继续运行确定性 Workflow
```

例如：

```text
Orchestrator
→ inspect_feature_drift
→ ActionExecutor
→ prepare_data → calculate_psi → calculate_missing → save_artifact
→ Outcome
```

Action 内部的执行仍然可以是固定 DAG。

### 2. LLM 负责决策，不负责越权执行

LLM 可以：

- 总结当前 State
- 形成 Hypothesis
- 选择下一步 Semantic Action
- 给出该动作的业务参数
- 判断是否建议继续 / 停止

LLM 不可以：

- 绕过 Guardrail
- 直接调用任意 SQL
- 直接决定使用 OOT 调参
- 自行宣告实验成功
- 自行修改生产模型

### 3. 每轮只执行一个 Action

Orchestrator 可以给多个候选动作排序，但：

```text
selected_action
```

只能有一个。

这用于强制形成：

```text
Think → Act → Observe → Think
```

而不是重新生成一个隐藏的 Workflow。

### 4. Outcome 只陈述事实

例如：

```text
OOT KS +1.2pp
PSI -0.05
新客 KS -0.6pp
```

Tool / ActionExecutor 不应该返回：

```text
下一步建议调参
```

下一步必须重新交给 Orchestrator。

### 5. Guardrail 继续确定性执行

包括但不限于：

- OOT 禁止参与特征筛选 / 参数优化
- 数据泄漏
- SQL / 数据权限审批
- 标签修改审批
- 生产上线审批
- 最大实验预算
- 最小样本量

---

## MVP 推荐范围

第一版仅支持一个场景：

> **Model Refresh Agent：已有 Champion 模型出现效果衰退后，自主诊断并尝试 Challenger 优化。**

第一版建议开放的 Semantic Actions：

```text
inspect_model_performance
inspect_sample_shift
inspect_feature_drift
inspect_segment_performance
inspect_overfitting
inspect_feature_target_relation

filter_features
change_sample_window
tune_model
compare_challenger

retrieve_knowledge
retrieve_trajectory
ask_human
terminate
```

先证明：

1. 同一 Goal 在不同 State 下会选不同 Action
2. Outcome 会改变下一轮决策
3. 路径不是固定 A→B→C
4. Hypothesis 会被 supported / rejected
5. 历史经验会影响 Action 排序
6. Guardrail 无法被 LLM 绕过

达到以上六项，再扩展 RankNet / Focal / 改标签 / 分客群 / 特征演化等能力。
