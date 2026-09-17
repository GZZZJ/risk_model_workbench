# 06 — RMW Migration Map

## 总体原则

当前 RMW 已经具备很强的：

- Harness
- Tool Registry
- Approval
- Audit
- State / artifact persistence
- deterministic execution
- checkpoint / recovery

这些不应该推倒重做。

vNext 的核心变化只有一个：

> **全局 Next Action 的控制权从 Full Execution Plan 转移到 Modeling Orchestrator。**

---

# 一、现有组件迁移建议

## 1. Goal Planner

### 当前职责

大致：

```text
Natural Language Goal
→ identify workflow
→ produce request / plan
→ full execution path
```

### vNext

降级为：

```text
Natural Language Goal / Monitor Event
→ GoalSpec
→ Initial ModelingState
```

### 建议

```text
goal_planner.py
```

可保留兼容层，但新增：

```text
state_initializer.py
```

长期逐步让 `state_initializer` 成为新入口。

### 禁止

State Initializer 不允许输出：

```text
workflow
full task DAG
step sequence
```

---

# 二、agent_plan.py

## 当前

全局 Plan / DAG 是执行中心。

## vNext

Plan 不再控制整个 Agent session。

保留其能力，用于：

> **单个 Semantic Action 内部的确定性执行。**

例如：

```text
Action:
inspect_feature_drift
```

内部 Plan：

```text
load data
→ calculate PSI
→ calculate missing shift
→ save artifacts
```

完全允许。

### 迁移策略

不要立刻删除 `agent_plan.py`。

将其从：

```text
Global Agent Plan
```

逐步降级为：

```text
Action Execution Plan
```

---

# 三、executor.py

## 当前

可能负责：

```text
find next runnable task
→ execute
→ update state
```

这意味着它事实上掌握全局路径。

## vNext

拆成：

### AgentRuntime

负责：

```text
load ModelingState
retrieve context
call Orchestrator
validate DecisionContract
check termination
bind selected Action
validate Guardrail
execute Action
apply Outcome
update State
save Trajectory
loop
```

### ActionExecutor

负责：

```text
执行 selected Semantic Action 内部的固定 DAG / tools
```

---

# 四、embedded_advisor.py

## 当前

保留：

- LLM
- structured output
- diagnosis
- bounded tuning
- existing provider integration

## vNext

建议新建：

```text
modeling_orchestrator.py
```

Orchestrator 可以复用 embedded advisor 的底层 LLM client / structured generation。

但不要简单把 `embedded_advisor.py` 重命名后继续原逻辑。

新 Orchestrator 必须能：

```text
read current State
read available Actions
read knowledge retrieval
read trajectory retrieval

→ generate hypotheses
→ rank candidate actions
→ select ONE next action
→ propose termination
```

---

# 五、Tool Registry

## 保留

现有 Tool Registry 非常重要。

vNext 新增一层：

```text
Semantic Action Registry
        ↓
Tool Registry
```

区别：

### Action

业务 / 建模语义：

```text
inspect_feature_drift
```

### Tool

技术能力：

```text
calculate_psi
read_dataset
generate_report
train_lgb
```

一个 Action 可以绑定多个 Tool。

---

# 六、Approval

原样保留并强化。

建议三级权限：

## Level 0 — autonomous diagnosis

```text
read metrics
calculate PSI
segment analysis
retrieve knowledge
```

## Level 1 — sandbox experiment

```text
feature filtering
hyperparameter tuning
train challenger
```

## Level 2 — human approval

```text
change label definition
change sample business definition
new external source
production replacement
strategy change
```

---

# 七、Audit / Trace

原样保留。

新增记录：

```text
decision_contract.json
action_outcome.json
modeling_state.json
trajectory.jsonl
experiment_ledger.jsonl
```

建议目录：

```text
runs/<session_id>/
├── states/
├── decisions/
├── outcomes/
├── trajectories/
├── experiments/
├── artifacts/
└── approvals/
```

---

# 八、建议新增模块

```text
agent/
├── orchestrator.py
├── runtime.py
├── state.py
├── state_initializer.py
├── state_reducer.py
├── decision_contract.py
│
├── actions/
│   ├── registry.py
│   ├── specs.py
│   ├── diagnosis/
│   ├── experiments/
│   └── meta/
│
├── policies/
│   ├── guardrails.py
│   ├── action_policy.py
│   └── approval_policy.py
│
├── memory/
│   ├── knowledge_retriever.py
│   ├── trajectory_retriever.py
│   ├── trajectory_store.py
│   └── fingerprint.py
│
└── evaluation/
    ├── outcome.py
    ├── outcome_evaluator.py
    └── termination.py
```

---

# 九、AgentRuntime 参考伪代码

```python
def run_agent(session_id: str):

    state = state_store.load_current(session_id)

    while True:

        knowledge = knowledge_retriever.retrieve(state)

        trajectories = trajectory_retriever.retrieve(state)

        decision = orchestrator.decide(
            state=state,
            available_actions=action_registry.available_for(state),
            knowledge=knowledge,
            trajectories=trajectories,
        )

        decision_validator.validate(decision)

        decision_store.save(decision)

        if decision.termination.terminate:
            termination_result = termination_validator.validate(
                state=state,
                proposal=decision.termination,
            )

            if termination_result.allowed:
                return finalize_session(
                    state=state,
                    reason=decision.termination.reason,
                )

            # termination 被拒绝后重新让 orchestrator 决策
            state = state_reducer.apply_termination_rejection(
                state,
                termination_result,
            )
            continue

        action = action_registry.bind(
            decision.selected_action
        )

        policy_result = guardrails.validate(
            state=state,
            action=action,
        )

        if policy_result.denied:
            outcome = make_denied_outcome(
                state=state,
                action=action,
                policy_result=policy_result,
            )

        elif policy_result.requires_approval:
            approval = approval_manager.request(
                session_id=session_id,
                action=action,
            )

            if not approval.approved:
                outcome = make_denied_outcome(...)
            else:
                outcome = action_executor.run(
                    state=state,
                    action=action,
                )

        else:
            outcome = action_executor.run(
                state=state,
                action=action,
            )

        outcome_store.save(outcome)

        new_state = state_reducer.apply(
            state=state,
            decision=decision,
            outcome=outcome,
        )

        trajectory_store.append(
            state_before=state,
            decision=decision,
            outcome=outcome,
            state_after=new_state,
        )

        state_store.save(new_state)

        state = new_state
```

---

# 十、迁移时不要做的事情

1. 不要删除现有 deterministic Harness
2. 不要让 LLM 直接写 arbitrary Python/SQL 并执行
3. 不要一次性把所有现有 Workflow 删除
4. 不要把所有步骤包装成“多个 Agent”就称为 Agentic
5. 不要保留一个隐藏的 Full Plan 再让 Orchestrator 假装逐步选择
6. 不要让 Outcome Tool 返回 next_action
7. 不要允许 LLM 自行判断 OOT 可用于调参
8. 不要在第一版做 DPO

---

# 十一、推荐实施阶段

## Phase 0 — Current Architecture Audit

只读当前代码，输出：

```text
当前 Goal Planner
当前 Plan
当前 Executor
当前 Advisor
当前 Tool Registry
当前 Approval
当前 State/Audit
```

以及对应迁移点。

不修改代码。

---

## Phase 1 — Schema & Persistence

新增：

```text
ModelingState
DecisionContract
ActionSpec
ActionOutcome
TrajectoryStep
```

以及 JSON/YAML persistence。

保持旧 Workflow 正常运行。

---

## Phase 2 — Semantic Action Registry

先注册 4 个只读诊断 Action：

```text
inspect_model_performance
inspect_sample_shift
inspect_feature_drift
inspect_segment_performance
```

优先复用现有 Tool。

---

## Phase 3 — Orchestrator Dry Run

Orchestrator 根据 State 选择 Action。

但：

```text
只输出 DecisionContract
不真正执行
```

与人工判断对照。

这是很重要的阶段。

---

## Phase 4 — AgentRuntime + Diagnosis Loop

真正执行：

```text
State
→ Decision
→ Diagnostic Action
→ Outcome
→ State
```

先不训练新模型。

做到动态诊断。

---

## Phase 5 — Add Experiment Actions

加入：

```text
filter_features
change_sample_window
tune_model
compare_challenger
```

此时形成第一版完整 Model Refresh Agent。

---

## Phase 6 — Knowledge + Trajectory Retrieval

把已有：

```text
Decision Knowledge
Historical Trajectory
Failure Experiments
```

接入 Orchestrator。

---

## Phase 7 — Agentic Evaluation

至少构造三类测试 State：

### Case A
```text
KS↓ + PSI↑
```

预期倾向：

```text
inspect_feature_drift
```

### Case B
```text
KS↓ + PSI稳定 + 新客占比明显变化
```

预期倾向：

```text
inspect_segment_performance
```

### Case C
```text
INS很好 + OOS/OOT明显差
```

预期倾向：

```text
inspect_overfitting
```

验证：

> 同一个 Goal 在不同 State 下路径不同。

并测试 Outcome 后路径能否发生变化。
