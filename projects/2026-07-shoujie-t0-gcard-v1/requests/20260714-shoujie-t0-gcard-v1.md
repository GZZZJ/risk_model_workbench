---
request_id: 20260714-shoujie-t0-gcard-v1
title: "首借 T0 G 卡新版本建模需求"
project: 2026-07-shoujie-t0-gcard-v1
workflow: full_modeling
owner: "辜子骏"
business_domain: preloan
scenario_profile: preloan_credit_card

data_source_mode: local_feather
sample_location: /root/notebook/首借G卡/五百维特征样本.feather
target_column: is_14_fq
id_columns:
  - uid
  - mdl_dte
time_column: mdl_dte
period_column: mdl_dte
split_column: final_flag
splits:
  dev:
    values: [DEV_INS]
  oos:
    values: [DEV_OOS]
  oot:
    values: [OOT]

sample_definition: 授信 T0 时点的首借候选用户；Y 为授信后 14 天内是否发起交易
sample_checks:
  - sample_check_001

feature_selection:
  rounds: [metadata, prescreen, refine]
  require_sql_approval: false

stage_steps:
  sample_check:
    - field_contract
    - key_uniqueness
    - monthly_label_distribution
    - segment_distribution
  feature_metadata:
    - feature_metadata_export
  feature_prescreen:
    - feature_quality_prescreen
  build_wide_sql:
    - wide_sql_generation
  feature_refine:
    - feature_availability_filter
    - missing_rate_filter
    - constant_value_filter
    - iv_filter
    - psi_filter
    - correlation_dedup
    - null_importance_filter
    - baseline_importance_filter
  train_baseline:
    - lightgbm_binary_training
  evaluate:
    - auc_ks
    - decile_lift
    - monthly_stability
    - score_psi
    - score_psi_bin_detail
    - segment_metrics
    - cross_gain_matrix
    - feature_gain_summary
  compare:
    - champion_challenger
  report:
    - model_report

step_params:
  missing_rate_filter:
    threshold: 0.90
  constant_value_filter:
    max_unique_values: 1
  iv_filter:
    min_iv: 0.005
  psi_filter:
    max_psi: 0.20
  correlation_dedup:
    method: spearman
    max_abs_corr: 0.80
  null_importance_filter:
    null_rounds: 20
    null_percentile: 75
    score_threshold: 1.0
  baseline_importance_filter:
    importance_type: gain
    keep_top_n: 500
  lightgbm_binary_training:
    early_stopping_rounds: 50
    max_auc_gap: 0.03
  decile_lift:
    bins: 10
  score_psi:
    bins: 10
    warn_psi: 0.20
  score_psi_bin_detail:
    bins: 10
  cross_gain_matrix:
    bins: 8

training:
  mode: llm_guided_tune
  tuning:
    max_rounds: 2
    candidates_per_round: 4
    max_trials: 8
    objective:
      primary_metric: valid_ks
      secondary_metric: valid_auc
    guardrails:
      max_train_valid_auc_gap: 0.03
    advisor:
      mode: host_agent
      fallback_to_heuristic: false

experiments:
  - name: main_lgbm
    method: lightgbm
    segment: all
    description: 全量样本二分类 LightGBM；客群仅用于评估切片

evaluation:
  metrics:
    - auc
    - ks
    - decile_lift
    - ranking_inversion
    - psi
    - business_risk
  champions:
    - v1_score
  comparison_dimensions:
    - sample_label
  risk_profile_dimensions:
    - sample_label
    - credit_limit

reports:
  model_display_name: 首借 T0 G 卡新模型
  score_labels:
    model_score: 首借 T0 G 卡新模型
    v1_score: 旧版首借 T0 G 卡
  sections:
    - sample_overview
    - data_quality
    - feature_screening
    - model_performance
    - champion_comparison
    - monthly_stability
    - score_stability
    - score_correlation
    - credit_limit_profile
    - top_features
    - limitations
    - next_action
  outputs:
    - model_report.xlsx
    - model_report.md
    - model_report.html
    - model_card.md
    - executive_summary.md
  targets:
    - name: main_lgbm
      experiment: main_lgbm
      train_dir: modeling/main_lgbm
      eval_dir: evaluation
      output_dir: reports
---

# 首借 T0 G 卡新版本建模需求

## 1. 结论与范围

本轮主线：

1. 使用新的首借样本和约 500 维候选特征。
2. 保持历史 Y 定义及 LightGBM 二分类训练方式不变。
3. 使用工作台 `llm_guided_tune` 完成有限轮次智能调参。
4. 评估结构对齐历史【模型文档】sheet。

## 2. 输入数据合同

### 2.1 本地 Feather 核验结果

| 项目 | 结果 |
| --- | --- |
| 文件 | `/root/notebook/首借G卡/五百维特征样本.feather` |
| 形状 | 1,653,812 行 × 518 列 |
| 时间范围 | 2025-06-01 至 2026-05-31 |
| 主键 | uid + mdl_dte（存在 1 组 8 行重复） |
| Y | is_14_fq，仅 0/1，正样本率 50.1% |
| final_flag | 不存在，需从 label + sample_split_tag 派生 |
| v1_score | string 类型，需转 float；88,453 空值 |
| credit_limit | 存在，float64，0 空值 |
| A 卡字段 | 不存在（a_card_score、a_card_level） |
| 候选特征 | 从 loan_org_typ_bnk_non_dps_min_cur_ovd_amt_day_diff 至文件末尾，500 列 |

### 2.2 final_flag 派生规则

| label | sample_split_tag | final_flag |
| --- | --- | --- |
| DEV | INS | DEV_INS |
| DEV | OOS | DEV_OOS |
| OOT | 任意 | OOT |

实际分布：
- DEV_INS: 1,087,440
- DEV_OOS: 465,854
- OOT: 100,518

### 2.3 禁止入模字段

- 主键: uid, mdl_dte
- 标签: is_14_fq, is_30_fq
- 切分: final_flag, label, sample_split_tag
- 辅助: sample_row_num, first_credit_time_by_btch, first_credit_law_type, sample_label
- 旧分数: v1_score
- 额度: credit_limit
- 随机: rand_flag0-rand_flag5

### 2.4 样本门禁

- is_14_fq 无空值，仅 0/1
- final_flag 派生后仅 DEV_INS、DEV_OOS、OOT
- 1组 uid+mdl_dte 重复（8行），需核对一致性后去重

## 3. 缺失字段披露

| 字段 | 状态 | 影响 |
| --- | --- | --- |
| a_card_score | 缺失 | A×G 交叉矩阵无法生成 |
| a_card_level | 缺失 | A 卡分客群评估无法生成 |
| credit_limit | 存在 | 额度画像可用 |

缺失 A 卡字段不阻断主模型训练，对应报告章节标记 `not_available`。
