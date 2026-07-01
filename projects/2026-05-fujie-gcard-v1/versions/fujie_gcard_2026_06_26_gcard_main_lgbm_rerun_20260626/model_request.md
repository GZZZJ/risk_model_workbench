---
request_id: 20260625-1740-model-request
title: "复借 G 卡主模型从0重跑"
project: 2026-05-fujie-gcard-v1
workflow: full_modeling
task_mode: "完整建模"
owner: "辜子骏"
business_domain: inloan_operation
scenario_profile: fujie_gcard_main_lgbm

# 数据口径修正（相对原始 request）：
# - data_source_mode 显式声明 local_feather（原为空，虽可自动解析但显式更可审计）
# - sample_location 去掉 "本地/" 前缀，使用合法绝对路径（原前缀会破坏文件路径解析）
# - feature_location 移除描述串，避免误触发 remote feature source；特征口径以本轮 run 注册产物为准
data_source_mode: local_feather
sample_location: /Users/guzijun/gcard_1pct_export/sample_50pct.feather
feature_location:
target_column: ftr_30d_ord_flag
id_columns:
  - uid
  - mdl_dte
time_column: mdl_dte
period_column: ds
split_column: final_flag
splits:
  dev:
    values:
      - DEV
  oos:
    values:
      - DEV-OOS
  oot:
    values:
      - OOT
      - OOT-OOS

sample_checks:
  - sample_check_profile
  - sample_check_stability
sample_definition: 可经营、当前未逾期用户、重资产订单；标签为观察日30天内是否发起

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
    - sql_review_gate
  feature_refine:
    - feature_availability_filter
    - missing_rate_filter
    - constant_value_filter
    - iv_filter
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
    - score_psi_bin_detail      # 稳定性·分箱 PSI component 明细（base/current 占比 + component）
    - segment_metrics
    - intent_zc_cross_risk       # 全量口径意愿×资产矩阵
    - intent_risk_segmented      # 意愿矩阵分客群（老户次新 e2e3 / 流失户 b2）
    - cross_gain_matrix
    - feature_gain_summary
  compare:
    - champion_challenger
  report:
    - model_report

step_params:
  feature_quality_prescreen:
    require_sql_approval: true
  sql_review_gate:
    block_on_high_risk: true
  missing_rate_filter:
    threshold: 0.9
  constant_value_filter:
    max_unique_values: 1
  iv_filter:
    min_iv: 0.005
  correlation_dedup:
    method: spearman
    max_abs_corr: 0.8
  null_importance_filter:
    null_rounds: 20
    null_percentile: 75
    score_threshold: 1.0
  baseline_importance_filter:
    importance_type: gain
    keep_top_n: 500
  lightgbm_binary_training:
    early_stopping_rounds: 50
    max_auc_gap: 0.02

feature_selection:
  rounds:
    - metadata
    - prescreen
    - refine
  require_sql_approval: true

candidate_targets:
  - ftr_30d_ord_flag
sample_variants:
  - all
  - e2e3
  - b2
experiments:
  - name: main_lgbm
    method: lightgbm
    segment: all
    description: 全客群复借G卡主模型，分客群只作为评估切片，不单独训练分客群模型

evaluation:
  metrics:
    - auc
    - ks
    - decile_lift
    - ranking_inversion
    - psi
  champions:
    - gcard_v2
    - gcard_v4
    - gcard_v5
    - gcard_v6
  comparison_dimensions:
    - split
    - month
    - segment
    - decile
  risk_profile_dimensions:
    - blue_customer_flag
    - zc_level

reports:
  sections:
    - sample_overview
    - feature_screening
    - modeling_plan
    - top_features
    - model_performance
    - champion_comparison
    - risk_profile
    - next_action
  outputs:
    - model_report.md
    - model_card.md
    - executive_summary.md
---

# 建模目标

基于最新复借G卡宽表数据口径，重新执行样本检查、特征收敛、LightGBM训练、评估、历史分对比和报告生成，形成可审计的新run产物。

# 样本与切分

使用复借G卡D01/D02宽表作为本轮建模数据源，字段合同如下：

- 主键：`uid`, `mdl_dte`
- 时间字段：`mdl_dte`
- 分区字段：`ds`
- 标签字段：`ftr_30d_ord_flag`
- 切分字段：`final_flag`
- 训练集：`DEV`
- OOS：`DEV-OOS`
- OOT：`OOT`, `OOT-OOS`

样本口径为可经营、当前未逾期用户、重资产订单；标签定义为观察日30天内是否发起。

数据源为本轮本地 feather（`/Users/guzijun/gcard_1pct_export/sample_50pct.feather`，489,743 行 × 2,857 列），已含 `mdl_dte`/`ds` 时间列与 `gcard_v2/v4/v5/v6` champion 分数列；时序切分为 2025-06~11 训练窗、2025-12~2026-01 OOT 窗。

# 特征筛选要求

本轮按复借G卡主模型专用链路执行：

1. 导出和注册特征元数据。
2. 进行特征质量初筛，真实取数或建表前必须先生成 SQL 并人工确认。
3. 生成宽表 SQL 并经过 SQL review gate。
4. 在宽表基础上执行精筛：可用性过滤、缺失率过滤、常量过滤、IV过滤、相关性去重、空标签重要性和基线模型重要性。

非特征字段必须从入模候选中排除，包括主键(`uid`/`mdl_dte`)、标签(`ftr_30d_ord_flag`)、切分字段(`final_flag`)、历史分(`gcard_v2`/`gcard_v4`/`gcard_v5`/`gcard_v6`)、画像切片(`blue_customer_flag`/`zc_level`)、时间/分区(`mdl_dte`/`ds`)以及随机列和报告辅助字段。

# 建模实验要求

实验描述：训练全客群 LightGBM 主模型 `main_lgbm`，分客群只用于评估切片，不在本轮拆分多个分客群模型。

训练前必须确认样本检查、特征清单和 SQL 审批状态；如缺少真实训练数据或特征清单，应停止并标记原因，不得继续产出伪完成结果。

# 评估与报告要求

重点比较维度：split, month, segment, decile

风险画像维度：blue_customer_flag, zc_level。

评估需包含 DEV/OOS/OOT、by 月、by 客群、十分箱 lift、PSI、排序倒挂检查，并与 `gcard_v2`、`gcard_v4`、`gcard_v5`、`gcard_v6` 做 champion/challenger 对比。

边界：OOT 仅覆盖 2025-12 与 2026-01 两月，月度趋势只能看 2 个点，PSI/漂移可算但趋势信息有限，报告中需注明。

# 补充说明

本次目标是从当前 risk_model_workbench 重新初始化并执行一版复借G卡主模型 run，
不复用旧 run 的完成状态。数据口径需与最近一次复借G卡宽表口径保持一致。

本地 feather 模式下，真实远端取数、profile、宽表 SQL 或 DP 拉数不实际触发；
SQL 证据仍需先生成并人工确认后再放行精筛与训练。缺失真实训练、评估或对比产物时，
只能标记 scaffold，不得把导入产物或占位结果当作本轮重跑证据。
vendor/feature-select-v2/scripts/code/ 只读，不作为本次修改范围。
