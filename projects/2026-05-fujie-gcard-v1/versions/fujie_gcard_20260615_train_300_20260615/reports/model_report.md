# 复借G卡模型报告

生成日期：2026-06-16；Summary 更新日期：2026-06-24

## Summary（新版模型 vs G卡V6）

- 口径：新版模型（model_score） vs 旧版全客群模型（G卡V6）；分客群表仅为切片效果，不含分客群专属模型对比。
- 更新日期：2026-06-24；指标来源：当前 run 已注册 train/evaluation artifacts。
- 详情导航：模型描述、模型效果-每月效果、模型效果-模型sloping、模型稳定性、重要变量。

### 一、结论摘要

| 项目 | 摘要 |
| --- | --- |
| 核心结论 | OOT-OOS 全客群：本轮 KS 0.720 vs G卡V6 0.716，提升 0.4 个百分点；AUC 0.932 vs G卡V6 0.930，提升 0.2 个百分点。 |
| 对比对象 | G卡V6 作为旧版全客群模型；其余历史版本仍保留在明细 sheet 中供追溯。 |
| 解释边界 | 本页只聚焦 30天发起标签的 AUC/KS、OOS 月度表现、分客群切片、sloping、PSI；MOB/金额风险不是本页验收主口径。 |

### 二、模型与样本口径

| 项目 | 内容 |
| --- | --- |
| 标签字段 | ftr_30d_ord_flag |
| 训练/验证/OOS | ['DEV'] / ['OOT'] / ['DEV-OOS', 'OOT-OOS'] |
| 算法 | lightgbm |
| 入模特征数 | 300 |
| Best iteration | 995 |
| Valid AUC / KS | 0.937 / 0.737 |

### 三、整体效果对比

<div style="display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap">
<table>
<thead><tr><th>样本</th><th>样本数</th><th>本轮AUC</th><th>G卡V6 AUC</th><th>AUC提升</th></tr></thead>
<tbody>
<tr><td>DEV</td><td>3,600,000</td><td>0.940</td><td>0.936</td><td>0.3%</td></tr>
<tr><td>DEV-OOS</td><td>3,600,000</td><td>0.934</td><td>0.933</td><td>0.2%</td></tr>
<tr><td>OOT</td><td>1,200,000</td><td>0.937</td><td>0.936</td><td>0.1%</td></tr>
<tr><td>OOT-OOS</td><td>1,200,000</td><td>0.932</td><td>0.930</td><td>0.2%</td></tr>
</tbody>
</table>
<table>
<thead><tr><th>样本</th><th>样本数</th><th>本轮KS</th><th>G卡V6 KS</th><th>KS提升</th></tr></thead>
<tbody>
<tr><td>DEV</td><td>3,600,000</td><td>0.736</td><td>0.728</td><td>0.8%</td></tr>
<tr><td>DEV-OOS</td><td>3,600,000</td><td>0.722</td><td>0.717</td><td>0.4%</td></tr>
<tr><td>OOT</td><td>1,200,000</td><td>0.737</td><td>0.734</td><td>0.3%</td></tr>
<tr><td>OOT-OOS</td><td>1,200,000</td><td>0.720</td><td>0.716</td><td>0.4%</td></tr>
</tbody>
</table>
</div>

### 四、OOT-OOS 分客群切片效果

> 分客群结果是效果切片，不代表已训练老户次新/流失户专属模型。

<div style="display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap">
<table>
<thead><tr><th>客群</th><th>样本数</th><th>30天发起率</th><th>本轮AUC</th><th>G卡V6 AUC</th><th>AUC提升</th></tr></thead>
<tbody>
<tr><td>全客群</td><td>1,200,000</td><td>14.3%</td><td>0.932</td><td>0.930</td><td>0.2%</td></tr>
<tr><td>老户次新</td><td>372,417</td><td>40.4%</td><td>0.804</td><td>0.796</td><td>0.7%</td></tr>
<tr><td>老户</td><td>327,473</td><td>41.7%</td><td>0.807</td><td>0.801</td><td>0.6%</td></tr>
<tr><td>次新</td><td>44,944</td><td>31.0%</td><td>0.753</td><td>0.738</td><td>1.5%</td></tr>
<tr><td>流失户</td><td>827,583</td><td>2.6%</td><td>0.896</td><td>0.897</td><td>-0.2%</td></tr>
</tbody>
</table>
<table>
<thead><tr><th>客群</th><th>样本数</th><th>30天发起率</th><th>本轮KS</th><th>G卡V6 KS</th><th>KS提升</th></tr></thead>
<tbody>
<tr><td>全客群</td><td>1,200,000</td><td>14.3%</td><td>0.720</td><td>0.716</td><td>0.4%</td></tr>
<tr><td>老户次新</td><td>372,417</td><td>40.4%</td><td>0.454</td><td>0.443</td><td>1.2%</td></tr>
<tr><td>老户</td><td>327,473</td><td>41.7%</td><td>0.461</td><td>0.450</td><td>1.2%</td></tr>
<tr><td>次新</td><td>44,944</td><td>31.0%</td><td>0.379</td><td>0.360</td><td>1.9%</td></tr>
<tr><td>流失户</td><td>827,583</td><td>2.6%</td><td>0.640</td><td>0.639</td><td>0.2%</td></tr>
</tbody>
</table>
</div>

### 五、OOS 按月效果

<div style="display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap">
<table>
<thead><tr><th>样本月份</th><th>样本数</th><th>30天发起率</th><th>本轮AUC</th><th>G卡V6 AUC</th><th>AUC提升</th></tr></thead>
<tbody>
<tr><td>DEV-OOS 2025-06</td><td>600,000</td><td>18.8%</td><td>0.930</td><td>0.928</td><td>0.2%</td></tr>
<tr><td>DEV-OOS 2025-07</td><td>600,000</td><td>18.4%</td><td>0.934</td><td>0.932</td><td>0.2%</td></tr>
<tr><td>DEV-OOS 2025-08</td><td>600,000</td><td>18.6%</td><td>0.936</td><td>0.934</td><td>0.2%</td></tr>
<tr><td>DEV-OOS 2025-09</td><td>600,000</td><td>17.1%</td><td>0.935</td><td>0.935</td><td>0.1%</td></tr>
<tr><td>DEV-OOS 2025-10</td><td>600,000</td><td>15.9%</td><td>0.935</td><td>0.933</td><td>0.1%</td></tr>
<tr><td>DEV-OOS 2025-11</td><td>600,000</td><td>15.3%</td><td>0.934</td><td>0.933</td><td>0.2%</td></tr>
<tr><td>OOT-OOS 2025-12</td><td>600,000</td><td>14.7%</td><td>0.932</td><td>0.930</td><td>0.2%</td></tr>
<tr><td>OOT-OOS 2026-01</td><td>600,000</td><td>13.9%</td><td>0.931</td><td>0.929</td><td>0.2%</td></tr>
</tbody>
</table>
<table>
<thead><tr><th>样本月份</th><th>样本数</th><th>30天发起率</th><th>本轮KS</th><th>G卡V6 KS</th><th>KS提升</th></tr></thead>
<tbody>
<tr><td>DEV-OOS 2025-06</td><td>600,000</td><td>18.8%</td><td>0.708</td><td>0.702</td><td>0.6%</td></tr>
<tr><td>DEV-OOS 2025-07</td><td>600,000</td><td>18.4%</td><td>0.720</td><td>0.714</td><td>0.5%</td></tr>
<tr><td>DEV-OOS 2025-08</td><td>600,000</td><td>18.6%</td><td>0.725</td><td>0.720</td><td>0.4%</td></tr>
<tr><td>DEV-OOS 2025-09</td><td>600,000</td><td>17.1%</td><td>0.725</td><td>0.724</td><td>0.2%</td></tr>
<tr><td>DEV-OOS 2025-10</td><td>600,000</td><td>15.9%</td><td>0.726</td><td>0.723</td><td>0.3%</td></tr>
<tr><td>DEV-OOS 2025-11</td><td>600,000</td><td>15.3%</td><td>0.726</td><td>0.721</td><td>0.5%</td></tr>
<tr><td>OOT-OOS 2025-12</td><td>600,000</td><td>14.7%</td><td>0.721</td><td>0.716</td><td>0.5%</td></tr>
<tr><td>OOT-OOS 2026-01</td><td>600,000</td><td>13.9%</td><td>0.719</td><td>0.715</td><td>0.4%</td></tr>
</tbody>
</table>
</div>

### 六、Sloping 高分10%表现

| 客群 | 本轮高分10%发起率 | G卡V6高分10%发起率 | 发起率提升 | 本轮累计lift | G卡V6累计lift |
| --- | --- | --- | --- | --- | --- |
| 全客群 | 78.4% | 77.1% | 1.3% | 1 | 1 |
| 老户次新 | 92.2% | 90.9% | 1.3% | 1 | 1 |
| 流失户 | 16.2% | 16.0% | 0.2% | 1 | 1 |

### 七、模型稳定性

> PSI 为本轮模型分数月度汇总；分箱明细和 PSI component 见【模型稳定性】或待补口径说明。

| 月份 | PSI | 样本数 |
| --- | --- | --- |
| 2025-06 | 0 | 1,200,000 |
| 2025-07 | 0.4% | 1,200,000 |
| 2025-08 | 0.8% | 1,200,000 |
| 2025-09 | 1.7% | 1,200,000 |
| 2025-10 | 2.5% | 1,200,000 |
| 2025-11 | 3.1% | 1,200,000 |
| 2025-12 | 3.6% | 1,200,000 |
| 2026-01 | 4.6% | 1,200,000 |

### 八、Top 10 重要变量

> 变量明细与 WOE 图见【重要变量】和【Top变量WOE】。

| 排名 | feature | 特征中文名 | Gain | Split |
| --- | --- | --- | --- | --- |
| 1 | d360_apl_ord_ddf_mdl_ord_crt_dte_min | 近360天_发起订单距评分日的时间间隔_最小 | 2,331,466.470 | 458 |
| 2 | ord_apl_sum_prc_amt_90_day | 近90天发起动支订单的动支总金额总额 | 2,261,862.902 | 59 |
| 3 | ord_apl_max_prc_amt_180_day | 近180天发起动支订单的最大动支总金额总额 | 1,858,213.462 | 97 |
| 4 | unpaid_principal_future_light_add_heavy | 重加轻资产未来到期未还本金 | 1,851,464.808 | 185 |
| 5 | his_90_day_csh_apl_ord_cnt_his_rto | 近3月历史CASH订单总数/总订单数 | 1,488,565.267 | 64 |
| 6 | d180_apl_ord_days_cnt_all | 近180天_订单发起天数(去重)（所有业务类型口径统计） | 913,489.836 | 109 |
| 7 | his_360_day_csh_apl_ord_cnt | 近12月历史CASH订单总数 | 804,807.131 | 152 |
| 8 | cnt_event_result_1_uid_recent_days_120 | cnt_event_result_1_uid_recent_days_120 | 731,484.271 | 151 |
| 9 | his_180_day_csh_apl_ord_cnt_his_rto | 近6月历史CASH订单总数/总订单数 | 491,328.372 | 155 |
| 10 | d360_apl_ord_cnt_all | 近360天_发起订单数（所有业务类型口径统计） | 300,729.802 | 83 |

> 证据提示：本页不补造缺失结果；当前 run audit 仍应以 rmw run audit 输出为准。

## 一、模型描述

- 模型目标：预测配置标签字段 `ftr_30d_ord_flag`。
- 对比口径：新版模型 `model_score` vs 旧版全客群模型 G卡V6；分客群表仅作为效果切片，不代表已训练老户次新/流失户专属模型。
- 建模样本：训练集 DEV，验证集 OOT，OOS DEV-OOS、OOT-OOS。
- 算法：lightgbm；最终入模变量 300 个；best iteration 995。
- 验证集效果：AUC 0.937，KS 0.737。

## 二、变量筛选过程

- 当前 run 未登记独立 D01/D02 或 feature_refine 筛选过程产物；以下只展示训练阶段实际候选和入模特征准备结果。

| 步骤 | 处理说明 | 变量个数 | 来源 |
| --- | --- | --- | --- |
| 训练输入 | 读取本轮训练候选特征列表；当前 run 未登记独立 D01/D02 或 feature_refine 筛选过程产物 | 301 | modeling/main_lgbm/candidate_feature_list.txt |
| 训练预处理 | 按训练数据字段可用性、缺失哨兵、缺失率与常量字段规则保留变量；填充策略：train_median_fill_zero | 300 | modeling/main_lgbm/preprocessing.json |
| 训练剔除 | 训练预处理阶段剔除变量数 | 1 | modeling/main_lgbm/feature_drop_detail.csv |
| 最终入模 | LightGBM 实际入模变量数 | 300 | modeling/main_lgbm/actual_feature_list.txt |

- 训练预处理保留 300/301 个变量；剔除 1 个变量。
- 特征列表包含需复核提示：LEAKAGE-WARN: unpaid_principal_future_light_add_heavy - contains indicators that may depend on future information; review before production use

## 三、核心效果与旧版全客群模型对比

- 本节按 Summary 口径仅展示本轮模型与 G卡V6 的对比；其余历史版本保留在评估明细产物中供追溯，不作为本页主结论。
- OOT-OOS 全客群：本轮 KS 0.720 vs G卡V6 0.716，提升 0.4 个百分点；AUC 0.932 vs G卡V6 0.930，提升 0.2 个百分点。

<div style="display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap">
<table>
<thead><tr><th>样本</th><th>样本数</th><th>本轮AUC</th><th>G卡V6 AUC</th><th>AUC提升</th></tr></thead>
<tbody>
<tr><td>DEV</td><td>3,600,000</td><td>0.940</td><td>0.936</td><td>0.3%</td></tr>
<tr><td>DEV-OOS</td><td>3,600,000</td><td>0.934</td><td>0.933</td><td>0.2%</td></tr>
<tr><td>OOT</td><td>1,200,000</td><td>0.937</td><td>0.936</td><td>0.1%</td></tr>
<tr><td>OOT-OOS</td><td>1,200,000</td><td>0.932</td><td>0.930</td><td>0.2%</td></tr>
</tbody>
</table>
<table>
<thead><tr><th>样本</th><th>样本数</th><th>本轮KS</th><th>G卡V6 KS</th><th>KS提升</th></tr></thead>
<tbody>
<tr><td>DEV</td><td>3,600,000</td><td>0.736</td><td>0.728</td><td>0.8%</td></tr>
<tr><td>DEV-OOS</td><td>3,600,000</td><td>0.722</td><td>0.717</td><td>0.4%</td></tr>
<tr><td>OOT</td><td>1,200,000</td><td>0.737</td><td>0.734</td><td>0.3%</td></tr>
<tr><td>OOT-OOS</td><td>1,200,000</td><td>0.720</td><td>0.716</td><td>0.4%</td></tr>
</tbody>
</table>
</div>

## 四、模型效果

1、每月效果（OOS）

| 序号 | 结论 |
| --- | --- |
| 1 | OOT-OOS 老户次新切片：本轮全客群模型 KS 0.454 vs G卡V6 0.443，提升 1.2 个百分点。 |
| 2 | OOT-OOS 流失户切片：本轮全客群模型 KS 0.640 vs G卡V6 0.639，提升 0.2 个百分点。 |
| 3 | OOT-OOS 全客群：本轮 KS 0.720 vs G卡V6 0.716，提升 0.4 个百分点；AUC 0.932 vs G卡V6 0.930，提升 0.2 个百分点。 |
| 4 | 分客群结果是效果切片，不代表已训练老户次新/流失户专属模型；MOB/金额风险不是本页验收主口径。 |

OOT-OOS 分客群切片效果

<div style="display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap">
<table>
<thead><tr><th>客群</th><th>样本数</th><th>30天发起率</th><th>本轮AUC</th><th>G卡V6 AUC</th><th>AUC提升</th></tr></thead>
<tbody>
<tr><td>全客群</td><td>1,200,000</td><td>14.3%</td><td>0.932</td><td>0.930</td><td>0.2%</td></tr>
<tr><td>老户次新</td><td>372,417</td><td>40.4%</td><td>0.804</td><td>0.796</td><td>0.7%</td></tr>
<tr><td>老户</td><td>327,473</td><td>41.7%</td><td>0.807</td><td>0.801</td><td>0.6%</td></tr>
<tr><td>次新</td><td>44,944</td><td>31.0%</td><td>0.753</td><td>0.738</td><td>1.5%</td></tr>
<tr><td>流失户</td><td>827,583</td><td>2.6%</td><td>0.896</td><td>0.897</td><td>-0.2%</td></tr>
</tbody>
</table>
<table>
<thead><tr><th>客群</th><th>样本数</th><th>30天发起率</th><th>本轮KS</th><th>G卡V6 KS</th><th>KS提升</th></tr></thead>
<tbody>
<tr><td>全客群</td><td>1,200,000</td><td>14.3%</td><td>0.720</td><td>0.716</td><td>0.4%</td></tr>
<tr><td>老户次新</td><td>372,417</td><td>40.4%</td><td>0.454</td><td>0.443</td><td>1.2%</td></tr>
<tr><td>老户</td><td>327,473</td><td>41.7%</td><td>0.461</td><td>0.450</td><td>1.2%</td></tr>
<tr><td>次新</td><td>44,944</td><td>31.0%</td><td>0.379</td><td>0.360</td><td>1.9%</td></tr>
<tr><td>流失户</td><td>827,583</td><td>2.6%</td><td>0.640</td><td>0.639</td><td>0.2%</td></tr>
</tbody>
</table>
</div>

OOS 按月效果

<div style="display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap">
<table>
<thead><tr><th>样本月份</th><th>样本数</th><th>30天发起率</th><th>本轮AUC</th><th>G卡V6 AUC</th><th>AUC提升</th></tr></thead>
<tbody>
<tr><td>DEV-OOS 2025-06</td><td>600,000</td><td>18.8%</td><td>0.930</td><td>0.928</td><td>0.2%</td></tr>
<tr><td>DEV-OOS 2025-07</td><td>600,000</td><td>18.4%</td><td>0.934</td><td>0.932</td><td>0.2%</td></tr>
<tr><td>DEV-OOS 2025-08</td><td>600,000</td><td>18.6%</td><td>0.936</td><td>0.934</td><td>0.2%</td></tr>
<tr><td>DEV-OOS 2025-09</td><td>600,000</td><td>17.1%</td><td>0.935</td><td>0.935</td><td>0.1%</td></tr>
<tr><td>DEV-OOS 2025-10</td><td>600,000</td><td>15.9%</td><td>0.935</td><td>0.933</td><td>0.1%</td></tr>
<tr><td>DEV-OOS 2025-11</td><td>600,000</td><td>15.3%</td><td>0.934</td><td>0.933</td><td>0.2%</td></tr>
<tr><td>OOT-OOS 2025-12</td><td>600,000</td><td>14.7%</td><td>0.932</td><td>0.930</td><td>0.2%</td></tr>
<tr><td>OOT-OOS 2026-01</td><td>600,000</td><td>13.9%</td><td>0.931</td><td>0.929</td><td>0.2%</td></tr>
</tbody>
</table>
<table>
<thead><tr><th>样本月份</th><th>样本数</th><th>30天发起率</th><th>本轮KS</th><th>G卡V6 KS</th><th>KS提升</th></tr></thead>
<tbody>
<tr><td>DEV-OOS 2025-06</td><td>600,000</td><td>18.8%</td><td>0.708</td><td>0.702</td><td>0.6%</td></tr>
<tr><td>DEV-OOS 2025-07</td><td>600,000</td><td>18.4%</td><td>0.720</td><td>0.714</td><td>0.5%</td></tr>
<tr><td>DEV-OOS 2025-08</td><td>600,000</td><td>18.6%</td><td>0.725</td><td>0.720</td><td>0.4%</td></tr>
<tr><td>DEV-OOS 2025-09</td><td>600,000</td><td>17.1%</td><td>0.725</td><td>0.724</td><td>0.2%</td></tr>
<tr><td>DEV-OOS 2025-10</td><td>600,000</td><td>15.9%</td><td>0.726</td><td>0.723</td><td>0.3%</td></tr>
<tr><td>DEV-OOS 2025-11</td><td>600,000</td><td>15.3%</td><td>0.726</td><td>0.721</td><td>0.5%</td></tr>
<tr><td>OOT-OOS 2025-12</td><td>600,000</td><td>14.7%</td><td>0.721</td><td>0.716</td><td>0.5%</td></tr>
<tr><td>OOT-OOS 2026-01</td><td>600,000</td><td>13.9%</td><td>0.719</td><td>0.715</td><td>0.4%</td></tr>
</tbody>
</table>
</div>

补充：2026年2-4月外推验证（全客群）
| month | split | n_samples | bad_rate | model_auc | model_ks | v6_auc | v6_ks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-02 | DEV | 600000 | 0.120 | 0.938 | 0.743 | 0.937 | 0.739 |
| 2026-02 | OOT | 0 |  |  |  |  |  |
| 2026-02 | DEV-OOS | 600000 | 0.128 | 0.937 | 0.739 | 0.936 | 0.737 |
| 2026-02 | OOT-OOS | 0 |  |  |  |  |  |
| 2026-02 | DEV+OOT | 600000 | 0.120 | 0.938 | 0.743 | 0.937 | 0.739 |
| 2026-03 | DEV | 600000 | 0.111 | 0.942 | 0.752 | 0.941 | 0.748 |
| 2026-03 | OOT | 0 |  |  |  |  |  |
| 2026-03 | DEV-OOS | 600000 | 0.116 | 0.942 | 0.752 | 0.941 | 0.748 |
| 2026-03 | OOT-OOS | 0 |  |  |  |  |  |
| 2026-03 | DEV+OOT | 600000 | 0.111 | 0.942 | 0.752 | 0.941 | 0.748 |
| 2026-04 | DEV | 0 |  |  |  |  |  |
| 2026-04 | OOT | 600000 | 0.106 | 0.945 | 0.760 | 0.944 | 0.757 |
| 2026-04 | DEV-OOS | 0 |  |  |  |  |  |
| 2026-04 | OOT-OOS | 600000 | 0.111 | 0.945 | 0.758 | 0.943 | 0.757 |
| 2026-04 | DEV+OOT | 600000 | 0.106 | 0.945 | 0.760 | 0.944 | 0.757 |

2026年2-4月外推验证（分客群）
| month | split | segment | n_samples | bad_rate | model_auc | model_ks | v6_auc | v6_ks | ks_uplift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-02 | DEV+OOT | B2 | 440892 | 0.021 | 0.892 | 0.640 | 0.896 | 0.637 | 0.003 |
| 2026-02 | DEV | B2 | 440892 | 0.021 | 0.892 | 0.640 | 0.896 | 0.637 | 0.003 |
| 2026-02 | DEV-OOS | B2 | 436114 | 0.023 | 0.893 | 0.639 | 0.897 | 0.638 | 0.001 |
| 2026-02 | DEV+OOT | E2 | 11193 | 0.302 | 0.753 | 0.370 | 0.738 | 0.356 | 0.014 |
| 2026-02 | DEV | E2 | 11193 | 0.302 | 0.753 | 0.370 | 0.738 | 0.356 | 0.014 |
| 2026-02 | DEV-OOS | E2 | 11192 | 0.311 | 0.759 | 0.393 | 0.752 | 0.372 | 0.021 |
| 2026-02 | DEV+OOT | E3 | 147915 | 0.401 | 0.803 | 0.452 | 0.797 | 0.444 | 0.009 |
| 2026-02 | DEV | E3 | 147915 | 0.401 | 0.803 | 0.452 | 0.797 | 0.444 | 0.009 |
| 2026-02 | DEV-OOS | E3 | 152694 | 0.415 | 0.802 | 0.455 | 0.796 | 0.442 | 0.013 |
| 2026-03 | DEV+OOT | B2 | 447803 | 0.019 | 0.910 | 0.675 | 0.910 | 0.666 | 0.009 |
| 2026-03 | DEV | B2 | 447803 | 0.019 | 0.910 | 0.675 | 0.910 | 0.666 | 0.009 |
| 2026-03 | DEV-OOS | B2 | 445523 | 0.021 | 0.913 | 0.686 | 0.912 | 0.678 | 0.007 |
| 2026-03 | DEV+OOT | E2 | 8890 | 0.274 | 0.764 | 0.392 | 0.753 | 0.376 | 0.016 |
| 2026-03 | DEV | E2 | 8890 | 0.274 | 0.764 | 0.392 | 0.753 | 0.376 | 0.016 |
| 2026-03 | DEV-OOS | E2 | 9514 | 0.278 | 0.755 | 0.387 | 0.751 | 0.383 | 0.004 |
| 2026-03 | DEV+OOT | E3 | 143307 | 0.385 | 0.809 | 0.462 | 0.802 | 0.451 | 0.011 |
| 2026-03 | DEV | E3 | 143307 | 0.385 | 0.809 | 0.462 | 0.802 | 0.451 | 0.011 |
| 2026-03 | DEV-OOS | E3 | 144963 | 0.396 | 0.809 | 0.464 | 0.804 | 0.455 | 0.009 |
| 2026-04 | DEV+OOT | B2 | 453136 | 0.018 | 0.916 | 0.693 | 0.916 | 0.689 | 0.004 |
| 2026-04 | OOT | B2 | 453136 | 0.018 | 0.916 | 0.693 | 0.916 | 0.689 | 0.004 |
| 2026-04 | OOT-OOS | B2 | 450270 | 0.019 | 0.918 | 0.694 | 0.917 | 0.689 | 0.005 |
| 2026-04 | DEV+OOT | E2 | 7820 | 0.288 | 0.775 | 0.417 | 0.761 | 0.397 | 0.020 |
| 2026-04 | OOT | E2 | 7820 | 0.288 | 0.775 | 0.417 | 0.761 | 0.397 | 0.020 |
| 2026-04 | OOT-OOS | E2 | 8605 | 0.277 | 0.758 | 0.387 | 0.747 | 0.375 | 0.011 |
| 2026-04 | DEV+OOT | E3 | 139044 | 0.383 | 0.809 | 0.462 | 0.804 | 0.453 | 0.008 |
| 2026-04 | OOT | E3 | 139044 | 0.383 | 0.809 | 0.462 | 0.804 | 0.453 | 0.008 |
| 2026-04 | OOT-OOS | E3 | 141125 | 0.392 | 0.808 | 0.463 | 0.803 | 0.456 | 0.007 |

2、模型sloping

| 客群 | 本轮高分10%发起率 | G卡V6高分10%发起率 | 发起率提升 | 本轮累计lift | G卡V6累计lift |
| --- | --- | --- | --- | --- | --- |
| 全客群 | 78.4% | 77.1% | 1.3% | 1 | 1 |
| 老户次新 | 92.2% | 90.9% | 1.3% | 1 | 1 |
| 流失户 | 16.2% | 16.0% | 0.2% | 1 | 1 |

3、意愿交叉风险（DEV-OOS）

- 当前 run 仅有全量观察口径的意愿 x 资产评级结果，缺少老户/流失户和历史版本分层矩阵；完整缺失项见 `model_report_missing_results.md`。

当前可用全量观察：占比（意愿评级 x 资产评级）
| 意愿 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 低意愿 | 0.000 | 0.001 | 0.005 | 0.009 | 0.022 | 0.160 | 0.137 | 0.333 |
| 中意愿 | 0.008 | 0.025 | 0.046 | 0.038 | 0.069 | 0.129 | 0.020 | 0.333 |
| 高意愿 | 0.015 | 0.050 | 0.072 | 0.082 | 0.020 | 0.075 | 0.020 | 0.333 |
| sum | 0.023 | 0.076 | 0.122 | 0.129 | 0.110 | 0.363 | 0.176 | 1.000 |

当前可用全量观察：30天发起率（意愿评级 x 资产评级）
| 意愿 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 低意愿 | 0.001 | 0.003 | 0.002 | 0.002 | 0.003 | 0.003 | 0.003 | 0.003 |
| 中意愿 | 0.040 | 0.038 | 0.033 | 0.034 | 0.021 | 0.021 | 0.027 | 0.026 |
| 高意愿 | 0.397 | 0.433 | 0.424 | 0.455 | 0.400 | 0.447 | 0.505 | 0.440 |
| sum | 0.273 | 0.297 | 0.261 | 0.300 | 0.087 | 0.100 | 0.062 | 0.156 |

当前可用全量观察：人头风险率（意愿评级 x 资产评级）
| 意愿 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 低意愿 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 中意愿 | 0.001 | 0.001 | 0.001 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 |
| 高意愿 | 0.009 | 0.013 | 0.014 | 0.015 | 0.011 | 0.011 | 0.009 | 0.013 |
| sum | 0.006 | 0.009 | 0.008 | 0.010 | 0.002 | 0.002 | 0.001 | 0.004 |

当前可用全量观察：金额风险（仅意愿维度）
| intent_level | n_samples | total_principal | total_overdue | amount_overdue_rate | head_risk_count | head_risk_rate |
| --- | --- | --- | --- | --- | --- | --- |
| 低意愿 | 3200000 | 10575256.000 | 274864.594 | 0.026 | 60 | 0.000 |
| 中意愿 | 3200000 | 189100096.000 | 4099353.500 | 0.022 | 1064 | 0.000 |
| 高意愿 | 3200000 | 3410450432.000 | 111691512.000 | 0.033 | 40626 | 0.013 |

## 五、模型稳定性

- PSI 为本轮模型分数月度汇总；分箱明细和 PSI component 见【模型稳定性】或待补口径说明。

| 月份 | PSI | 样本数 |
| --- | --- | --- |
| 2025-06 | 0 | 1,200,000 |
| 2025-07 | 0.4% | 1,200,000 |
| 2025-08 | 0.8% | 1,200,000 |
| 2025-09 | 1.7% | 1,200,000 |
| 2025-10 | 2.5% | 1,200,000 |
| 2025-11 | 3.1% | 1,200,000 |
| 2025-12 | 3.6% | 1,200,000 |
| 2026-01 | 4.6% | 1,200,000 |

## 六、重要变量

- 变量明细与 WOE 图见【重要变量】和【Top变量WOE】。

| 排名 | feature | 特征中文名 | Gain | Split |
| --- | --- | --- | --- | --- |
| 1 | d360_apl_ord_ddf_mdl_ord_crt_dte_min | 近360天_发起订单距评分日的时间间隔_最小 | 2,331,466.470 | 458 |
| 2 | ord_apl_sum_prc_amt_90_day | 近90天发起动支订单的动支总金额总额 | 2,261,862.902 | 59 |
| 3 | ord_apl_max_prc_amt_180_day | 近180天发起动支订单的最大动支总金额总额 | 1,858,213.462 | 97 |
| 4 | unpaid_principal_future_light_add_heavy | 重加轻资产未来到期未还本金 | 1,851,464.808 | 185 |
| 5 | his_90_day_csh_apl_ord_cnt_his_rto | 近3月历史CASH订单总数/总订单数 | 1,488,565.267 | 64 |
| 6 | d180_apl_ord_days_cnt_all | 近180天_订单发起天数(去重)（所有业务类型口径统计） | 913,489.836 | 109 |
| 7 | his_360_day_csh_apl_ord_cnt | 近12月历史CASH订单总数 | 804,807.131 | 152 |
| 8 | cnt_event_result_1_uid_recent_days_120 | cnt_event_result_1_uid_recent_days_120 | 731,484.271 | 151 |
| 9 | his_180_day_csh_apl_ord_cnt_his_rto | 近6月历史CASH订单总数/总订单数 | 491,328.372 | 155 |
| 10 | d360_apl_ord_cnt_all | 近360天_发起订单数（所有业务类型口径统计） | 300,729.802 | 83 |

## 七、Top变量WOE

- Top20 WOE 图见 Excel sheet `Top变量WOE`，PNG 和汇总 CSV 见 `reports/woe_top_features/` 或训练产物目录。
| 排名 | 变量 | Gain | IV |
| --- | --- | --- | --- |
| 1 | d360_apl_ord_ddf_mdl_ord_crt_dte_min | 2331466.470 | 13.055 |
| 2 | ord_apl_sum_prc_amt_90_day | 2261862.902 | 6.893 |
| 3 | ord_apl_max_prc_amt_180_day | 1858213.462 | 7.641 |
| 4 | unpaid_principal_future_light_add_heavy | 1851464.808 | 8.741 |
| 5 | his_90_day_csh_apl_ord_cnt_his_rto | 1488565.267 | 5.911 |
| 6 | d180_apl_ord_days_cnt_all | 913489.836 | 12.893 |
| 7 | his_360_day_csh_apl_ord_cnt | 804807.131 | 10.003 |
| 8 | cnt_event_result_1_uid_recent_days_120 | 731484.271 | 8.412 |
| 9 | his_180_day_csh_apl_ord_cnt_his_rto | 491328.372 | 8.291 |
| 10 | d360_apl_ord_cnt_all | 300729.802 | 12.238 |
| 11 | ord_apl_sum_prc_amt_360_day | 269005.655 | 9.485 |
| 12 | dau_90d | 236275.718 | 6.741 |
| 13 | cnt_other_info_cash_uid_recent_days_30 | 234065.608 | 3.560 |
| 14 | cnt_avg_ord_span_30d | 144646.982 | 6.482 |
| 15 | stg_pln_pay_off_sum_prc_amt_1m | 130206.734 | 8.344 |
| 16 | ddf_lst_app_str_tim_to_mdl_tim_sec | 93691.185 | 6.685 |
| 17 | his_720_day_csh_apl_ord_cnt_his_rto | 88510.891 | 6.487 |
| 18 | clk_cnt_14d | 66581.064 | 5.354 |
| 19 | cnt_avb_lmt_2m_v2 | 60108.805 | 7.333 |
| 20 | d360_max_faqi_prc_amt_divide_rsk_avl_lmt_cash | 57697.036 | 10.144 |

## 八、待补充事项

- 证据提示：本页不补造缺失结果；当前 run audit 仍应以 `rmw run audit` 输出为准。
- 当前仍不可补齐：变量分布/分箱图、变量中文描述与业务标签、MOB1/MOB3 历史风险精确定义；这些需要原始特征值、业务字典或未来期还款表现数据。
- 详见 `model_report_missing_results.md`。
