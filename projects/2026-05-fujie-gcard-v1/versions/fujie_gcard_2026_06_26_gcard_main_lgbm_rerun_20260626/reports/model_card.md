# Model Card — 复借 G 卡主模型 main_lgbm

**Run**: `2026-06-26-gcard-main-lgbm-rerun`　**日期**: 2026-06-26　**负责人**: 辜子骏

## 1. 模型概述

- **用途**: 预测可经营、当前未逾期的重资产用户在观察日后 30 天内是否发起复借订单（复借意愿）。
- **算法**: LightGBM（二分类，objective=binary，metric=auc）。
- **客群**: 全客群复借 G 卡主模型；分客群（B2 / E2E3）仅作评估切片，不单独训练。
- **入模特征**: 500（从 2,837 候选特征经缺失率 / 常量 / IV / 相关性去重 / 空重要性 / 基线重要性收敛）。
- **主要参数**: learning_rate=0.05, num_leaves=31, max_depth=5, min_child_samples=100, subsample=0.7, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0, random_seed=0。
- **best_iteration**: 303（early_stopping_rounds=50，验证集 DEV-OOS）。

## 2. 训练数据

- **来源**: 本地 feather `sample_50pct.feather`，复借 G 卡 D01/D02 宽表（59 表 join，2,837 特征）。
- **规模**: 489,743 行 × 2,857 列。
- **时间窗**: 2025-06 ~ 2026-01（8 个月）。
- **切分**（按 `final_flag`，时序切分）:
  - DEV (2025-06~11): 183,739 行，训练。
  - DEV-OOS (2025-06~11): 183,480 行，早停验证（同时间窗 holdout，保证 OOT 纯净）。
  - OOT (2025-12~2026-01): 61,173 行，时间外评估。
  - OOT-OOS (2025-12~2026-01): 61,351 行，时间外样本外评估。
- **标签**: `ftr_30d_ord_flag`（1 = 观察日后 30 天内发起订单）。整体正样本率 15.67%。
- **非特征列排除**: 主键(uid/mdl_dte)、标签、切分(final_flag)、历史分(gcard_v2/v4/v5/v6)、画像切片(blue_customer_flag/zc_level)、时间/分区(mdl_dte/ds)、随机列(rand_flag0~5)，杜绝信息泄漏。

## 3. 性能（model_score）

| 切分 | AUC | KS |
| --- | --- | --- |
| DEV | 0.945 | 0.749 |
| DEV-OOS | 0.932 | 0.716 |
| OOT | 0.934 | 0.733 |
| OOT-OOS | 0.930 | 0.719 |

- **auc_gap**: 0.013（DEV vs DEV-OOS，< 0.02 阈值）。
- **排序倒挂**: 十分箱 0 处倒挂（单调）。

## 4. 稳定性

- **月度 PSI**: 最大 0.052（2026-01 vs 基线），< 0.1 稳定线。
- **月度 AUC**: 0.928 ~ 0.948，跨 8 月无塌陷。
- **注意**: 业务正样本率同期从 17.7% 下行至 13.2%，模型排序能力在此漂移下保持稳定，但校准可能受影响。

## 5. Champion/Challenger 对比

相对最强历史版本 G卡V6，新模型在 OOT 基本持平（AUC −0.002）；
相对 gcard_v2/v4/v5，OOT AUC 提升 +0.006 ~ +0.011。
结论：新模型可作为 G卡V6 的并列 challenger。

## 6. 局限与注意事项

- **OOT 时间跨度短**: 仅 2 个月（2025-12 ~ 2026-01），长期时间外稳定性需后续月份验证。
- **同源瓶颈**: 训练口径与历史 G卡模型同源，特征域相近，相对 G卡V6 的提升空间有限。
- **非本模型口径**: MOB1/MOB3 还款表现、金额风险不在 30 天发起标签的验收范围内。
- **数据类型注意**: 源表中部分历史分（gcard_v2/v4/v6）为字符串型，已在打分阶段强转 numeric；任何下游复用须确保同样处理。
- **公平性**: 未针对受保护属性做专门公平性审计；当前风险画像维度为 blue_customer_flag / zc_level（客群/资产等级），用于评估切片而非受保护属性。

## 7. 监控建议

上线后持续监控：月度分数 PSI、月度 AUC/KS、分客群（B2/E2E3）稳定性、正样本率漂移对校准的影响。

## 8. 产物清单

- 模型: `modeling/main_lgbm/model.pkl`
- 全量打分: `modeling/main_lgbm/scores_all_splits.feather`
- 特征重要性: `modeling/main_lgbm/feature_importance.csv`
- 入模清单: `feature_selection/final_features.txt`（500）
- 评估明细: `evaluation/`（overall / monthly / segment / decile / psi / ranking_inversion / champion_challenger）
- WOE 解释: `reports/woe_top_features/`（Top20 变量）
