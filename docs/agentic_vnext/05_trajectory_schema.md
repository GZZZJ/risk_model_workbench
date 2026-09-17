# 05 — Trajectory Memory Schema

## 目标

每一轮 Agent 都形成：

```text
State Before
→ Retrieved Context
→ Hypothesis
→ Selected Action
→ Outcome
→ State After
```

这就是未来 RMW 自己积累的建模经验。

---

## Pydantic v2 建议实现

```python
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class StateFingerprint(BaseModel):
    business_scene: Optional[str] = None
    model_type: Optional[str] = None

    problem_types: List[str] = Field(default_factory=list)

    ks_decline_bucket: Optional[
        Literal["none", "mild", "moderate", "severe"]
    ] = None

    score_psi_bucket: Optional[
        Literal["unknown", "low", "medium", "high"]
    ] = None

    segment_shift_status: Optional[
        Literal["unknown", "absent", "suspected", "confirmed"]
    ] = None

    feature_drift_status: Optional[
        Literal["unknown", "absent", "suspected", "confirmed"]
    ] = None

    label_issue_status: Optional[
        Literal["unknown", "absent", "suspected", "confirmed"]
    ] = None

    goal_type: Optional[str] = None

    custom: Dict[str, Any] = Field(default_factory=dict)


class MemoryRef(BaseModel):
    source_type: Literal["knowledge", "trajectory", "experiment"]
    ref_id: str


class TrajectoryStep(BaseModel):
    schema_version: str = "1.0"

    trajectory_step_id: str
    session_id: str
    iteration: int

    state_before_id: str
    state_after_id: str

    state_before_fingerprint: StateFingerprint
    state_before_summary: str

    retrieved_memory: List[MemoryRef] = Field(default_factory=list)

    hypothesis: Optional[str] = None
    selected_action_type: Optional[str] = None
    selected_action_parameters: Dict[str, Any] = Field(default_factory=dict)

    outcome_id: Optional[str] = None
    outcome_summary: Optional[str] = None

    hypothesis_verdict: Optional[
        Literal["untested", "supported", "rejected", "inconclusive"]
    ] = None

    state_after_summary: str

    later_marked_useful: Optional[bool] = None
    human_feedback: Optional[str] = None

    metadata: Dict[str, Any] = Field(default_factory=dict)
```

---

# Trajectory Retrieval

不要只做：

```text
embedding(current_state_full_text)
```

建议先构造 `StateFingerprint`：

```yaml
business_scene: 贷中
model_type: B卡

problem_types:
  - performance_degradation

ks_decline_bucket: severe
score_psi_bucket: low

segment_shift_status: suspected
feature_drift_status: unknown

goal_type: recover_oot_performance
```

Retrieval：

```text
metadata filtering
    ↓
semantic similarity
    ↓
top-k
```

推荐第一版：

```text
top_k = 3
```

---

# Decision Knowledge 与 Trajectory Memory 分开

## Decision Knowledge

回答：

> 这种问题理论上有哪些方法？

来源：

```text
模型优化方法
SOP
方法论文档
团队总结
```

---

## Trajectory Memory

回答：

> 过去类似状态下实际做了什么，结果怎样？

来源：

```text
历史模型项目
Agent 自己未来的运行轨迹
```

---

# 失败轨迹

失败实验必须保留。

例如：

```yaml
state:
  sample_selection_bias: suspected

action:
  PSM augmentation

outcome:
  oot_auc_delta: -0.002

verdict:
  failed
```

以后 Retrieval 命中后：

> 降低 PSM 的优先级。

禁止直接：

> 永久禁止 PSM。

历史经验是 Prior，不是硬规则。

---

# Preference Learning 的未来接口

当前不做 DPO。

但建议保留未来字段：

```text
later_marked_useful
human_feedback
```

以后如果同一个 State 出现：

```text
Action A
Action B
```

并能被专家明确判断：

```text
A > B
```

才形成 Preference Pair。

当前阶段只积累 Trajectory。
