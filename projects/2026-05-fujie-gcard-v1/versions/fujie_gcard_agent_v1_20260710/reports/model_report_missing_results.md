# 复借G卡模型报告缺失结果清单

本文件只记录当前已注册 run artifact 无法可靠还原的内容，不伪造指标。

## 不可补齐（3 项）

| # | 缺少字段/结果 | 原因 |
|---|---|---|
| 1 | 变量分布/分箱图 | 当前评分 feather 仅含模型分数和标签，不含原始特征值 |
| 2 | 变量中文描述、业务标签 | 需要业务知识、特征字典或业务标签 |
| 3 | MOB1/MOB3 历史风险 | 需要未来期还款表现数据，当前数据仅含 30 天发起标签和观察风险字段 |

## 已补齐 — 本轮 model_score

| 产出文件 | 内容 |
|---|---|
| `evaluation/monthly_metrics.csv` | 全客群按月 AUC/KS |
| `evaluation/segment_metrics.csv` | 全客群/老户次新/老户/次新/流失户切片 AUC/KS |
| `evaluation/benchmark_uplift.csv` | 本轮模型 vs 历史版本整体提升量 |
| `evaluation/score_psi_by_month.csv` | 分数 PSI 稳定性 |
| `evaluation/decile_lift_all_model_score.csv` | 全客群本轮模型十分位 lift |
| `evaluation/decile_lift_e2e3_model_score.csv` | 老户次新本轮模型十分位 lift |
| `evaluation/decile_lift_b2_model_score.csv` | 流失户本轮模型十分位 lift |
| `evaluation/intent_zc_distribution.csv` | 全量观察口径意愿资产占比和发起率矩阵 |
| `evaluation/intent_zc_amount_risk.csv` | 全量观察口径意愿维度金额风险 |
| `evaluation/intent_zc_headcount_risk.csv` | 全量观察口径意愿资产人头风险矩阵 |



## 已补齐 — Top20 变量 WOE

| 产出文件 | 内容 |
|---|---|
| `reports/woe_top_features/woe_top20_summary.csv` | Top20 变量分箱、WOE、IV 和人群占比 |
| `reports/woe_top_features/images/` | Top20 变量 WOE 折线与人群占比柱图 |


后续继续通过 `rmw report` 统一生成报告，避免人工改写 xlsx 中的计算结果。
