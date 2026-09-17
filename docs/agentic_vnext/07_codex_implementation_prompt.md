# 07 — Codex Implementation Prompt

下面这段提示词建议在把本目录复制到 RMW repo 后，直接交给 Codex。

---

你当前位于 `risk_model_workbench` 项目仓库中。

我正在把当前项目从“预定义完整 Workflow + 确定性 Executor”升级为真正 Agentic 的自动化建模 Agent。

## 一、你必须先阅读的设计文件

请完整阅读：

```text
docs/agentic_vnext/00_README.md
docs/agentic_vnext/01_modeling_state_schema.md
docs/agentic_vnext/02_decision_contract_schema.md
docs/agentic_vnext/03_action_spec_schema.md
docs/agentic_vnext/04_outcome_schema.md
docs/agentic_vnext/05_trajectory_schema.md
docs/agentic_vnext/06_rmw_migration_map.md
```

这些文件是本次改造的设计约束。

如果当前代码实现与文档设计冲突：

1. 先指出冲突
2. 分析兼容方案
3. 不要自行改变核心设计目标

核心目标只有一个：

> 全局建模路径不再在任务开始时完整确定，而是每轮根据当前 `ModelingState` 动态选择一个 `Semantic Action`。

---

# 二、先做 Current Architecture Audit

第一轮不要改代码。

请扫描当前仓库，并重点阅读：

```text
Goal Planner
agent_plan
executor
embedded_advisor
Tool Registry
Approval
Agent State
Audit / Trace
Checkpoint / Resume
```

以及与：

```text
sample check
feature analysis
training
evaluation
comparison
report
```

相关的已有工具和 workflow。

请输出：

## A. Current Architecture

说明：

```text
用户目标如何进入
完整 Plan 如何生成
Executor 如何决定下一步 task
LLM 当前在哪里参与
Tool 如何被调用
Approval 如何工作
State 如何保存
Resume 如何工作
```

## B. Mapping to vNext

按：

```text
现有模块
当前职责
vNext 职责
处理方式
```

分类：

```text
KEEP
REUSE
REFACTOR
DEPRECATE
NEW
```

## C. Risks

重点检查：

1. 当前是否有任何模块强依赖 Full Plan
2. task_id / dependency / invocation hash 是否与全局 DAG 强绑定
3. resume 是否依赖完整 plan
4. approval 是否依赖 task graph
5. artifact 路径是否依赖 workflow step
6. 是否已有可复用的 schema / state store
7. 是否已有适合包装成 Semantic Action 的现成工具

## D. Phase 1 实施方案

只能规划：

```text
Schema + Persistence
```

暂时不要实现 Orchestrator。

等我确认后再编码。

---

# 三、实现原则

后续开始编码后，必须遵守：

## 1. 保留现有系统能力

不要推倒重写：

```text
Harness
Tool Registry
Approval
Audit
Artifact
Checkpoint
Resume
```

优先复用。

---

## 2. 不删除旧 Workflow

第一阶段必须：

```text
old runtime still works
```

新 Agentic Runtime 使用独立入口。

例如可以：

```text
rmw agent run ...
```

或项目当前风格下的等价命令。

旧模式和新模式短期并存。

---

## 3. Full Plan 不再控制 vNext

新 Runtime 中禁止：

```text
先生成完整 session DAG
再按 DAG 执行
```

允许：

```text
Semantic Action 内部使用小型 deterministic DAG
```

例如：

```text
inspect_feature_drift
    ↓
calculate PSI
    ↓
calculate missing shift
```

---

## 4. Orchestrator 每轮只能选一个 Action

必须通过 `DecisionContract` schema 校验。

禁止产生：

```text
selected_actions: [...]
```

禁止将多个未来步骤塞入 action parameters。

---

## 5. Tool 不得决定 next action

ActionOutcome 只能返回事实。

禁止：

```text
recommended_next_action
next_step
```

---

## 6. Guardrail 必须确定性执行

尤其：

```text
OOT feature selection
OOT hyperparameter tuning
SQL approval
target change approval
production change approval
```

不能交给 LLM 自我约束。

---

## 7. Schema 优先

优先使用：

```text
Pydantic v2
```

所有：

```text
State
Decision
Action
Outcome
Trajectory
```

都必须 schema 化。

---

# 四、推荐开发阶段

严格按照：

```text
Phase 0 Architecture Audit
Phase 1 Schema & Persistence
Phase 2 Semantic Action Registry
Phase 3 Orchestrator Dry Run
Phase 4 Diagnosis Runtime
Phase 5 Experiment Actions
Phase 6 Retrieval
Phase 7 Agentic Evaluation
```

执行。

不要跨阶段一次性完成。

每阶段完成后：

1. 跑现有 tests
2. 跑新增 tests
3. 给出 diff summary
4. 给出兼容性说明
5. 给出下一阶段建议
6. 停止，等待我确认

---

# 五、Phase 1 要求

当我批准 Phase 1 后，先实现：

```text
ModelingState
DecisionContract
ActionSpec
ActionOutcome
TrajectoryStep
```

以及：

```text
state store
decision store
outcome store
trajectory append store
```

第一阶段不需要调用真实 LLM。

需要：

```text
serialization round-trip tests
schema validation tests
backward compatibility tests
```

---

# 六、Phase 2 要求

只注册首批只读 Diagnosis Action：

```text
inspect_model_performance
inspect_sample_shift
inspect_feature_drift
inspect_segment_performance
```

优先包装现有工具。

如果现有工具缺能力：

先指出缺口。

不要为了完成 Action 大规模重写模型工具。

---

# 七、Phase 3 要求

实现：

```text
ModelingOrchestrator
```

但先 Dry Run：

```text
State
→ LLM
→ DecisionContract
```

不执行 selected action。

保存：

```text
decision JSON
```

用于人工检查。

---

# 八、Phase 4 要求

实现最小 AgentRuntime：

```text
load state
→ retrieve available actions
→ orchestrator decision
→ validate
→ guardrail
→ execute diagnosis action
→ outcome
→ reducer
→ new state
→ trajectory
→ repeat
```

此阶段禁止训练新模型。

---

# 九、Phase 5 要求

再加入：

```text
filter_features
change_sample_window
tune_model
compare_challenger
```

此阶段开始真正消费：

```text
training experiment budget
```

---

# 十、测试 Agentic 行为

必须新增至少三类 integration tests。

## Test A

State：

```text
KS严重下降
Score PSI高
```

期望 Agent 优先考虑：

```text
feature drift / sample shift diagnosis
```

## Test B

State：

```text
KS严重下降
Score PSI稳定
新客占比明显变化
```

期望：

```text
segment diagnosis
```

## Test C

State：

```text
INS显著优于OOS/OOT
```

期望：

```text
overfitting diagnosis
```

重点不是把 action name 写死成断言。

重点验证：

```text
相同 Goal + 不同 State
→ Decision path 不完全相同
```

以及：

```text
改变上一轮 Outcome
→ 下一轮 Decision 会变化
```

---

# 十一、明确禁止

不要：

1. 用多个“子 Agent”包装现有固定流程来冒充 Agentic
2. 继续在 Goal Planner 中生成完整 workflow
3. 用固定 if-else 替代 Orchestrator
4. 让 LLM 直接生成并运行 arbitrary SQL
5. 让 LLM 自己判断实验是否成功
6. 让 Tool 返回 next step
7. 为了兼容旧系统而偷偷保留新 Runtime 的全局 DAG
8. 第一版做 DPO / LoRA
9. 第一版重写全部 Tool
10. 第一轮直接修改代码

---

现在先执行：

> **Phase 0 — Current Architecture Audit**

只读代码并输出审计结果，不修改任何文件。
