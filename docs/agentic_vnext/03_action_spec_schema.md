# 03 — ActionSpec & Semantic Action Registry

## 目标

Orchestrator 不允许直接执行任意代码。

它只能选择注册在 `ActionRegistry` 中的 Semantic Action。

结构：

```text
LLM Decision
    ↓
Semantic Action
    ↓
Action Spec Validation
    ↓
Guardrail
    ↓
Deterministic Action Executor
```

---

## Pydantic v2 建议实现

```python
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


ActionCategory = Literal["diagnosis", "experiment", "meta"]


class ActionParameterSpec(BaseModel):
    name: str
    type: Literal[
        "string",
        "integer",
        "float",
        "boolean",
        "enum",
        "list",
        "dict",
    ]
    required: bool = False
    description: str
    allowed_values: Optional[List[Any]] = None
    default: Any = None


class ActionPrecondition(BaseModel):
    code: str
    description: str
    hard: bool = True


class ActionGuardrailBinding(BaseModel):
    guardrail_code: str
    description: str


class ActionExecutorBinding(BaseModel):
    executor_name: str
    workflow_name: Optional[str] = None
    tool_names: List[str] = Field(default_factory=list)


class ActionSpec(BaseModel):
    action_type: str
    version: str = "1.0"

    category: ActionCategory

    description: str
    intended_use: str

    parameters: List[ActionParameterSpec] = Field(default_factory=list)

    preconditions: List[ActionPrecondition] = Field(default_factory=list)
    guardrails: List[ActionGuardrailBinding] = Field(default_factory=list)

    executor: ActionExecutorBinding

    consumes_training_budget: bool = False
    requires_human_approval: bool = False

    expected_outcome_fields: List[str] = Field(default_factory=list)

    tags: List[str] = Field(default_factory=list)
```

---

# MVP Action Registry

第一版建议只注册以下 Action。

## Diagnosis

### `inspect_model_performance`

用途：

- KS/AUC/Lift 时序
- 与 Champion / baseline 比较
- 判断是单月异常还是持续衰退

建议 executor：

```text
model_performance_analyzer
```

---

### `inspect_sample_shift`

用途：

- 样本量
- 坏样本率
- 时间分布
- 客群占比
- 渠道结构
- vintage 变化

---

### `inspect_feature_drift`

用途：

- Feature PSI
- missing rate shift
- value range shift
- top feature drift

---

### `inspect_segment_performance`

参数示例：

```text
segment_dimension:
    customer_type
    channel
    credit_limit_band
    vintage
    product
```

注意：

Segment Dimension 应来自允许字段 / 项目配置。

不要允许 LLM 自由拼 SQL 字段。

---

### `inspect_overfitting`

用途：

```text
INS/OOS/OOT AUC/KS gap
```

---

### `inspect_feature_target_relation`

用途：

比较历史 / 当前：

```text
IV
WOE
单变量 AUC
bad rate by bins
feature importance
```

判断：

> 分布稳定但 Feature → Risk 映射是否变化。

---

## Experiment

### `filter_features`

参数：

```text
strategy:
  unstable_features
  low_value_features
  redundant_features
  custom_registered_strategy
```

第一版不要允许：

```text
features = LLM arbitrary list
```

建议让 LLM 选择 strategy，具体 Feature List 由确定性分析 Tool 生成。

---

### `change_sample_window`

参数：

```text
registered_window_candidate
```

候选窗口必须由项目配置 / State 提供。

不能让 LLM 随意输入任意日期。

---

### `tune_model`

仅在：

```text
当前 evidence 支持过拟合 / 欠拟合 / 参数空间仍有价值
```

时使用。

禁止 OOT tuning。

---

### `compare_challenger`

用途：

统一生成：

```text
Champion vs Challenger
KS
AUC
Lift
PSI
segment regression
```

---

## Meta

### `retrieve_knowledge`

### `retrieve_trajectory`

如果 Retrieval 直接作为 Orchestrator 前置上下文，也可以不暴露为显式 Action。

---

### `ask_human`

典型情况：

```text
修改标签口径
修改业务样本定义
信息缺失且无法通过 Tool 获得
```

---

### `terminate`

建议不要做普通 Action。

由 `DecisionContract.termination` 表示。

---

# Registry 示例

```python
ACTION_REGISTRY = {
    "inspect_feature_drift": ActionSpec(
        action_type="inspect_feature_drift",
        category="diagnosis",
        description="Inspect feature distribution and missingness drift.",
        intended_use="Determine whether model degradation is associated with feature drift.",
        executor=ActionExecutorBinding(
            executor_name="feature_drift_executor",
            workflow_name="feature_drift_dag",
            tool_names=[
                "calculate_feature_psi",
                "calculate_missing_shift",
            ],
        ),
        consumes_training_budget=False,
        requires_human_approval=False,
        expected_outcome_fields=[
            "high_psi_features",
            "missing_shift_features",
        ],
    ),
}
```

---

# 关键架构原则

Action 内部仍然允许是 Workflow。

例如：

```text
inspect_feature_drift
    ↓
load reference
    ↓
load current
    ↓
calculate PSI
    ↓
calculate missing
    ↓
save report
```

这没有问题。

Agentic 的关键是：

> **全局下一步是否由 Orchestrator 根据 State 动态决定。**
