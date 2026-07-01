# 复借G卡模型报告缺失结果清单

本文件只记录当前已注册 run artifact 无法可靠还原的内容，不伪造指标。

## 不可补齐（3 项）

| # | 缺少字段/结果 | 原因 |
|---|---|---|
| 1 | 变量分布/分箱图 | 当前评分 feather 仅含模型分数和标签，不含原始特征值 |
| 2 | 变量中文描述、业务标签 | 需要业务知识和 D01/D02 特征字典 |
| 3 | MOB1/MOB3 历史风险 | 需要未来期还款表现数据，当前数据仅含 30 天发起标签和观察风险字段 |

## 已补齐 — 本轮 model_score

| 产出文件 | 内容 |
|---|---|
| `evaluation/decile_lift_bins.csv` | 分客群 x final_flag 十分位 lift，含 score 边界 |
| `evaluation/intent_zc_segment_distribution.csv` | 老户/流失户 DEV-OOS 意愿资产占比矩阵 |
| `evaluation/intent_zc_segment_ftr_rate.csv` | 老户/流失户 DEV-OOS 30天发起率矩阵 |
| `evaluation/intent_zc_segment_amount_risk.csv` | 老户/流失户 DEV-OOS 新增订单3期金额逾期率矩阵 |
| `evaluation/monthly_segment_metrics_oot_oos.csv` | 老户次新/流失户 OOT-OOS 按月 AUC/KS |
| `evaluation/segment_model_comparison.csv` | 分客群 vs 全客群同口径对比 |
| `evaluation/model_score_bin_distribution_by_month.csv` | 本轮模型按月分箱占比、发起率和 PSI 组件 |

## 已补齐 — 历史版本横向对比

覆盖 score_version：`model_score`、`gcard_v2`、`gcard_v4`、`gcard_v5`、`gcard_v6`。

| 产出文件 | 内容 |
|---|---|
| `evaluation/intent_zc_segment_distribution_by_version.csv` | 各版本老户/流失户 DEV-OOS 意愿资产占比矩阵 |
| `evaluation/intent_zc_segment_ftr_rate_by_version.csv` | 各版本老户/流失户 DEV-OOS 30天发起率矩阵 |
| `evaluation/intent_zc_segment_amount_risk_by_version.csv` | 各版本老户/流失户 DEV-OOS 新增订单3期金额逾期率矩阵 |
| `evaluation/decile_lift_bins_by_version.csv` | 各版本 sloping 分箱上下界 |
| `evaluation/monthly_segment_metrics_oos_by_version.csv` | 全客群/老户次新/老户/次新/流失户 DEV-OOS + OOT-OOS 按月版本横向效果 |
| `evaluation/monthly_segment_metrics_oot_oos_by_version.csv` | 老户次新/流失户 OOT-OOS 按月版本横向效果 |
| `evaluation/score_bin_distribution_by_month_by_version.csv` | 各版本按月稳定性分箱 |

后续继续通过 `jm report` 统一生成报告，避免人工改写 xlsx 中的计算结果。
