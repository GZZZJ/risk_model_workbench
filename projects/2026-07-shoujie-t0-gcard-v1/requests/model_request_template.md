---
request_id: "2026-07-shoujie-t0-gcard-v1-baseline-request"
title: "首借T0 G卡新模型 baseline modeling request"
project: "2026-07-shoujie-t0-gcard-v1"
workflow: full_modeling
owner: ""

target_column: "target"
id_columns:
  - uid
  - "sample_date"
time_column: "sample_date"
period_column: "sample_month"
split_column: final_flag

sample_checks:
  - sample_check_profile
  - sample_check_stability

feature_selection:
  rounds:
    - metadata
    - prescreen
    - refine
  require_sql_approval: true

experiments:
  - name: baseline_all
    method: xgboost
    segment: all

evaluation:
  metrics:
    - auc
    - ks
    - decile_lift
  champions: []
  segments:
    - all

reports:
  outputs:
    - model_report.md
    - model_card.md
    - executive_summary.md
---

# 建模目标

填写模型要解决的问题、使用场景、候选策略动作和成功标准。

# 样本与切分

填写样本位置、标签定义、观察窗口、DEV/OOT/OOS 划分和待确认事项。

# 特征筛选要求

填写候选特征来源、默认筛选步骤以外的补充要求、禁止字段和泄漏风险。

# 建模实验要求

填写是否分 Y、是否分样本、是否加权、是否做排序优化，以及实验优先级。

# 评估与报告要求

填写重点比较维度、历史分对比对象、风险画像维度和报告输出要求。

# 补充说明

填写禁止事项、开放问题、特殊口径和给 Agent 的执行提醒。
