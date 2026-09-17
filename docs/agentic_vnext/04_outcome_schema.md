# 04 — ActionOutcome Schema

## 目标

Action 执行完成以后，只返回：

> **发生了什么。**

Outcome 不负责决定下一步。

禁止：

```text
recommended_next_action = tune_model
```

下一步重新交给 Orchestrator。

---

## Pydantic v2 建议实现

```python
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


ExecutionStatus = Literal[
    "success",
    "partial_success",
    "failed",
    "denied",
    "waiting_for_approval",
]


class MetricDelta(BaseModel):
    before: Optional[float] = None
    after: Optional[float] = None
    delta: Optional[float] = None


class OutcomeEvidence(BaseModel):
    artifact_paths: List[str] = Field(default_factory=list)
    metric_refs: Dict[str, Any] = Field(default_factory=dict)
    logs: List[str] = Field(default_factory=list)


class GuardrailEvent(BaseModel):
    code: str
    status: Literal["passed", "blocked", "approval_required"]
    message: str


class ActionOutcome(BaseModel):
    schema_version: str = "1.0"

    outcome_id: str
    action_id: str
    action_type: str
    state_id_before: str

    execution_status: ExecutionStatus

    summary: str

    observations: Dict[str, Any] = Field(default_factory=dict)
    metric_deltas: Dict[str, MetricDelta] = Field(default_factory=dict)

    changed_entities: Dict[str, Any] = Field(default_factory=dict)
    side_effects: Dict[str, Any] = Field(default_factory=dict)

    guardrail_events: List[GuardrailEvent] = Field(default_factory=list)

    consumes_training_budget: bool = False
    created_model_version: Optional[str] = None

    evidence: OutcomeEvidence = Field(default_factory=OutcomeEvidence)

    error_code: Optional[str] = None
    error_message: Optional[str] = None
```

---

# 示例 1：诊断型 Outcome

```yaml
outcome_id: OUT_003
action_id: ACT_003
action_type: inspect_segment_performance
state_id_before: STATE_002

execution_status: success

summary: >
  模型衰退主要集中在新客群。

observations:
  old_customer:
    ks_current: 0.69
    ks_baseline: 0.70
    delta: -0.01

  new_customer:
    ks_current: 0.51
    ks_baseline: 0.63
    delta: -0.12

side_effects: {}

evidence:
  artifact_paths:
    - artifacts/segment_performance.csv
```

---

# 示例 2：实验型 Outcome

```yaml
outcome_id: OUT_007
action_id: ACT_007
action_type: filter_features
state_id_before: STATE_006

execution_status: success

summary: >
  移除高漂移特征后，模型稳定性明显改善，OOT KS小幅恢复。

metric_deltas:
  oot_ks:
    before: 0.270
    after: 0.282
    delta: 0.012

  score_psi:
    before: 0.180
    after: 0.110
    delta: -0.070

changed_entities:
  feature_count:
    before: 436
    after: 401

side_effects:
  new_customer_ks_delta: -0.006

consumes_training_budget: true
created_model_version: challenger_007
```

---

# Outcome Evaluator

可以有确定性 Evaluator 给 Outcome 添加：

```text
experiment_verdict
```

建议枚举：

```text
successful
partially_successful
neutral
failed
invalid
```

但它只能基于预定义指标计算。

例如：

```python
if ks_delta >= target and psi <= max_psi:
    verdict = "successful"
elif ks_delta > 0:
    verdict = "partially_successful"
else:
    verdict = "failed"
```

不能用 LLM 自己判。

---

# State Reducer

Reducer 根据：

```text
State + Decision + Outcome
```

生成 New State。

Reducer 做三类更新：

1. 原始数值更新
2. 实验历史追加
3. 根据 Outcome 证据更新 Hypothesis 状态

其中第 3 类可以允许 Orchestrator 提议，但最终变更必须有 Outcome Evidence。
