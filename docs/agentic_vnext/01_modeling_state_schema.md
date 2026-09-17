# 01 — ModelingState Schema

## 目标

`ModelingState` 描述：

> **当前建模环境、模型表现、已知证据、未知问题、假设、实验历史和约束。**

它不描述：

> Workflow 当前执行到第几步。

禁止出现以全局流程为中心的字段，例如：

```yaml
current_step: feature_selection
next_step: model_training
```

---

## Pydantic v2 建议实现

```python
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


HypothesisStatus = Literal["untested", "supported", "rejected"]
AnalysisStatus = Literal["unknown", "not_checked", "checked", "unavailable"]
SessionStatus = Literal[
    "running",
    "waiting_for_approval",
    "waiting_for_human",
    "terminated",
    "failed",
]


class SuccessCriteria(BaseModel):
    oot_ks_delta_min: Optional[float] = None
    oot_auc_delta_min: Optional[float] = None
    max_score_psi: Optional[float] = None
    max_segment_regression: Optional[float] = None
    custom: Dict[str, Any] = Field(default_factory=dict)


class GoalSpec(BaseModel):
    objective: str
    success_criteria: SuccessCriteria
    notes: Optional[str] = None


class TriggerEvidence(BaseModel):
    ks_decline_relative: Optional[float] = None
    auc_decline_relative: Optional[float] = None
    score_psi: Optional[float] = None
    feature_alert_count: Optional[int] = None
    custom: Dict[str, Any] = Field(default_factory=dict)


class TriggerSpec(BaseModel):
    type: Literal[
        "user_goal",
        "monitoring_alert",
        "scheduled_review",
        "manual_refresh",
        "other",
    ]
    evidence: TriggerEvidence = Field(default_factory=TriggerEvidence)


class BusinessContext(BaseModel):
    business_scene: str
    model_type: Optional[str] = None
    model_name: Optional[str] = None
    champion_version: Optional[str] = None
    target_definition: Optional[str] = None
    business_constraints: Dict[str, Any] = Field(default_factory=dict)


class DataContext(BaseModel):
    sample_window: Optional[str] = None
    train_size: Optional[int] = None
    oos_size: Optional[int] = None
    oot_size: Optional[int] = None
    bad_rate: Optional[float] = None
    feature_count: Optional[int] = None
    custom: Dict[str, Any] = Field(default_factory=dict)


class DatasetMetrics(BaseModel):
    auc: Optional[float] = None
    ks: Optional[float] = None
    lift: Optional[float] = None
    bad_rate: Optional[float] = None
    custom: Dict[str, Any] = Field(default_factory=dict)


class PerformanceState(BaseModel):
    ins: DatasetMetrics = Field(default_factory=DatasetMetrics)
    oos: DatasetMetrics = Field(default_factory=DatasetMetrics)
    oot: DatasetMetrics = Field(default_factory=DatasetMetrics)
    baseline: DatasetMetrics = Field(default_factory=DatasetMetrics)
    trend: Dict[str, Any] = Field(default_factory=dict)


class StabilityState(BaseModel):
    score_psi: Optional[float] = None
    high_psi_feature_count: Optional[int] = None
    missing_shift_feature_count: Optional[int] = None
    feature_drift_status: AnalysisStatus = "unknown"
    custom: Dict[str, Any] = Field(default_factory=dict)


class SegmentState(BaseModel):
    analysis_status: AnalysisStatus = "not_checked"
    degradation_localized_to: List[str] = Field(default_factory=list)
    metrics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class LabelQualityState(BaseModel):
    maturity_status: AnalysisStatus = "unknown"
    gray_sample_status: AnalysisStatus = "unknown"
    noise_status: AnalysisStatus = "unknown"
    custom: Dict[str, Any] = Field(default_factory=dict)


class Hypothesis(BaseModel):
    hypothesis_id: str
    description: str
    status: HypothesisStatus = "untested"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    supporting_evidence: List[str] = Field(default_factory=list)
    contradicting_evidence: List[str] = Field(default_factory=list)


class AttemptedAction(BaseModel):
    action_id: str
    action_type: str
    outcome_summary: Optional[str] = None
    verdict: Optional[str] = None


class ExperimentHistory(BaseModel):
    attempted_actions: List[AttemptedAction] = Field(default_factory=list)
    failed_action_types: List[str] = Field(default_factory=list)
    successful_action_types: List[str] = Field(default_factory=list)


class ModelingConstraints(BaseModel):
    oot_for_feature_selection: bool = False
    oot_for_hyperparameter_tuning: bool = False
    sql_requires_approval: bool = True
    target_change_requires_approval: bool = True
    sample_definition_change_requires_approval: bool = True
    production_change_requires_approval: bool = True
    custom: Dict[str, Any] = Field(default_factory=dict)


class BudgetState(BaseModel):
    max_agent_iterations: int = 20
    max_training_experiments: int = 8
    used_agent_iterations: int = 0
    used_training_experiments: int = 0


class ModelingState(BaseModel):
    schema_version: str = "1.0"

    state_id: str
    session_id: str
    iteration: int = 0

    goal: GoalSpec
    trigger: TriggerSpec

    business_context: BusinessContext
    data_context: DataContext = Field(default_factory=DataContext)

    performance: PerformanceState = Field(default_factory=PerformanceState)
    stability: StabilityState = Field(default_factory=StabilityState)
    segments: SegmentState = Field(default_factory=SegmentState)
    label_quality: LabelQualityState = Field(default_factory=LabelQualityState)

    hypotheses: List[Hypothesis] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)

    experiment_history: ExperimentHistory = Field(default_factory=ExperimentHistory)

    constraints: ModelingConstraints = Field(default_factory=ModelingConstraints)
    budget: BudgetState = Field(default_factory=BudgetState)

    status: SessionStatus = "running"

    last_action_id: Optional[str] = None
    last_outcome_id: Optional[str] = None

    artifacts: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

---

## State 更新规则

### 必须由确定性程序写入的字段

例如：

```text
performance.*
stability.score_psi
segments.metrics
data_context.train_size
budget.used_*
artifacts
```

这些字段来自 Tool / Harness 真实结果。

### 可以由 Orchestrator 提议、Reducer 落盘的字段

例如：

```text
hypotheses
open_questions
```

### 不允许 LLM 任意覆盖

```text
constraints
success_criteria
raw metrics
approval state
execution artifacts
```

---

## 核心原则

`ModelingState` 必须能够回答：

1. 现在发生了什么？
2. 当前有哪些证据？
3. 还有哪些未知？
4. 当前有哪些假设？
5. 已经试过什么？
6. 哪些方法失败过？
7. 还有多少实验预算？
8. 哪些动作受到业务 / 安全约束？
