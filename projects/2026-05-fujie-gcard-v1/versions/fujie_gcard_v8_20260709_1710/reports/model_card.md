# Model Card

## 基本信息

| Item | Value |
| --- | --- |
| Version | `fujie_gcard_v8_20260709_1710` |
| Scenario | 复借 G 卡主模型 |
| Experiment | `main_lgbm` |
| Algorithm | LightGBM binary |
| Target | `ftr_30d_ord_flag` |
| Split column | `final_flag` |
| Train split | `DEV` |
| Validation split | `DEV-OOS` |
| OOT splits | `OOT`, `OOT-OOS` |

## 数据与特征

- 样本文件: `/Users/guzijun/gcard_1pct_export/sample_50pct.feather`
- 全量打分样本: 489,743 行
- 特征元数据: 70/70 张表读取成功，候选字段 15,028 个
- 精筛输入变量: 2,837 个
- 精筛可用变量: 2,549 个
- D01 后保留: 1,708 个
- D04 Null Importance 后保留: 474 个
- 最终入模变量: 474 个

## 训练配置

- Training mode: `llm_guided_tune`
- Advisor path: host-agent unavailable, local heuristic fallback
- Trial count: 9
- Selected trial: 1
- Best iteration: 421
- Valid AUC: 0.932943
- Valid KS: 0.719264
- Train AUC: 0.954738
- Train KS: 0.769044
- AUC gap: 0.021794

## 分割表现

| Split | Samples | AUC | KS |
| --- | ---: | ---: | ---: |
| DEV | 183,739 | 0.954738 | 0.769044 |
| DEV-OOS | 183,480 | 0.932943 | 0.719264 |
| OOT | 61,173 | 0.934550 | 0.731755 |
| OOT-OOS | 61,351 | 0.930176 | 0.720092 |

## Champion Comparison

| Champion | DEV-OOS AUC uplift | DEV-OOS KS uplift | OOT AUC uplift | OOT KS uplift | OOT-OOS AUC uplift | OOT-OOS KS uplift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gcard_v2` | 0.007006 | 0.015444 | 0.006035 | 0.014200 | 0.008674 | 0.019716 |
| `gcard_v4` | 0.011453 | 0.025389 | 0.011453 | 0.022022 | 0.012703 | 0.032001 |
| `gcard_v5` | 0.009688 | 0.018672 | 0.006951 | 0.016981 | 0.009508 | 0.027394 |
| `gcard_v6` | -0.000625 | -0.000421 | -0.001260 | -0.002056 | 0.000274 | 0.004033 |

## Top Features By Gain

1. `d180_apl_ord_days_cnt`
2. `d360_apl_ord_ddf_mdl_ord_crt_dte_min`
3. `unpaid_principal_future_light_add_heavy`
4. `first_day_diff_event_result_1_all`
5. `his_360_day_csh_apl_ord_cnt_his_rto`

## 使用限制

- 本轮是本地 feather 全链路重跑，不是远端全量宽表重跑。
- 特征精筛最终使用 50,000 行采样完成，原因是 200,000/100,000 行在 D01 相关性计算中耗时过高。
- 不应直接声明模型全面优于 `gcard_v6`。
- MOB1/MOB3 历史风险缺未来期还款表现数据，不作为本轮准入结论。
- 发布前建议补充分群收益、线上资源成本和稳定性监控评审。

## Evidence

- `modeling/main_lgbm/metrics_train_valid.json`
- `modeling/main_lgbm/tuning_summary.json`
- `modeling/main_lgbm/scores_all_splits.feather`
- `feature_selection/stage_summary.json`
- `evaluation/evaluation_summary.json`
- `evaluation/benchmark_uplift.csv`
- `audit/artifact_manifest.json`
