# 02 — DecisionContract Schema

## 目标

每一轮 Orchestrator 只做一件事：

> **根据当前 State、知识、历史轨迹和可用 Action，选择下一步最值得执行的一个 Semantic Action。**

允许给出多个候选动作。

只允许一个：

```text
selected_action
```

---

## Pydantic v2 建议实现

```python
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


class DecisionHypothesis(BaseModel):
    hypothesis_id: Optional[str] = None
    description: str
    confidence: float = Field(ge=0.0, le=1.0)


class RetrievedEvidenceRef(BaseModel):
    source_type: Literal[
        "decision_knowledge",
        "trajectory",
        "experiment",
        "guardrail",
        "artifact",
    ]
    ref_id: str
    summary: Optional[str] = None


class CandidateAction(BaseModel):
    action_type: str
    reason: str
    priority: int = Field(ge=1)
    expected_information_gain: Optional[Literal["low", "medium", "high"]] = None
    expected_cost: Optional[Literal["low", "medium", "high"]] = None


class SelectedAction(BaseModel):
    action_type: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class TerminationProposal(BaseModel):
    terminate: bool = False
    reason: Optional[
        Literal[
            "goal_achieved",
            "no_promising_action",
            "experiment_budget_exhausted",
            "iteration_budget_exhausted",
            "human_decision_required",
            "guardrail_blocked",
            "system_failure",
            "user_stop",
        ]
    ] = None
    rationale: Optional[str] = None


class DecisionContract(BaseModel):
    schema_version: str = "1.0"

    decision_id: str
    state_id: str
    iteration: int

    state_summary: str

    primary_hypothesis: Optional[DecisionHypothesis] = None
    alternative_hypotheses: List[DecisionHypothesis] = Field(default_factory=list)

    uncertainties: List[str] = Field(default_factory=list)
    retrieved_evidence: List[RetrievedEvidenceRef] = Field(default_factory=list)

    candidate_actions: List[CandidateAction] = Field(default_factory=list)

    selected_action: Optional[SelectedAction] = None

    expected_evidence: Optional[str] = None
    decision_rationale: str

    approval_expected: bool = False

    termination: TerminationProposal = Field(default_factory=TerminationProposal)

    @model_validator(mode="after")
    def validate_action_or_termination(self):
        if self.termination.terminate:
            if self.selected_action is not None:
                raise ValueError(
                    "A terminating decision must not also select an action."
                )
        else:
            if self.selected_action is None:
                raise ValueError(
                    "A non-terminating decision must select exactly one action."
                )
        return self
```

---

## Orchestrator Prompt 的核心约束

建议明确写入 System / Developer Prompt：

```text
1. 每轮只选择一个 selected_action。
2. 优先选择能够最大程度降低关键不确定性的动作。
3. 在根因不清晰时，优先低成本诊断动作，再考虑高成本训练实验。
4. 不机械复制历史轨迹；历史经验仅作为证据。
5. 当前 session 已经失败的方案，应降低优先级，但不能自动永久禁止。
6. 不得绕过 Guardrail。
7. 不得自行修改 raw metrics。
8. 不得把未来多步 Action 打包成一个 selected_action。
9. 若认为任务应结束，只能提出 termination proposal，最终由 TerminationValidator 判断。
```

---

## 不建议的 Decision

错误：

```json
{
  "selected_action": {
    "action_type": "full_model_optimization",
    "parameters": {
      "steps": [
        "check_psi",
        "filter_features",
        "tune",
        "compare"
      ]
    }
  }
}
```

这相当于重新创建 Workflow。

正确：

```json
{
  "selected_action": {
    "action_type": "inspect_feature_drift",
    "parameters": {}
  }
}
```

执行完成后重新决策。
