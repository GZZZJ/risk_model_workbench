# Executive Summary

## 结论

`fujie_gcard_v8_20260709_1710` 已完成本地全链路重跑，strict audit verdict 为 `complete`。本轮模型相对 `gcard_v2`、`gcard_v4`、`gcard_v5` 在 DEV-OOS、OOT、OOT-OOS 均有提升；相对 `gcard_v6` 不构成全面提升，DEV-OOS 和 OOT 略低，OOT-OOS 小幅高于 `gcard_v6`。

## 本轮产物

- Version: `fujie_gcard_v8_20260709_1710`
- Workflow: `full_modeling`
- 样本来源: `/Users/guzijun/gcard_1pct_export/sample_50pct.feather`
- 全量打分样本: 489,743 行
- 最终入模变量: 474 个
- 主要报告: `reports/model_report.xlsx`, `reports/model_report.md`, `reports/model_report.html`
- 审计源: `version_state.yml`, `audit/artifact_manifest.json`

## 核心表现

| Split | Model AUC | Model KS | vs gcard_v6 AUC | vs gcard_v6 KS |
| --- | ---: | ---: | ---: | ---: |
| DEV-OOS | 0.932943 | 0.719264 | -0.000625 | -0.000421 |
| OOT | 0.934550 | 0.731755 | -0.001260 | -0.002056 |
| OOT-OOS | 0.930176 | 0.720092 | 0.000274 | 0.004033 |

## 关键判断

- 本轮模型可作为 `gcard_v2/v4/v5` 的有效 challenger。
- 相对 `gcard_v6` 的收益不稳定，不建议直接宣称替代；应进入更细粒度分群、业务收益和稳定性评审。
- 训练调参使用 `llm_guided_tune`，host-agent 不可用时由本地 heuristic fallback 生成候选；最终选择 trial 1，valid_auc=0.932943，valid_ks=0.719264，AUC gap=0.021794，guardrail passed。
- 特征精筛因 200,000/100,000 行 D01 相关性计算耗时过高，最终使用 50,000 行资源降级口径完成。该口径可审计，但不等同于全量宽表精筛。

## 风险与限制

- 本轮基于本地 feather，不是远端全量宽表执行结果。
- `feature_refine` 使用 50,000 行采样，若进入发布评审，建议用更高资源配置复核 D01/D02/Null Importance。
- MOB1/MOB3 历史风险缺少未来期还款表现数据，报告仅保留缺失说明。
- 评估记录了 `split/month/segment/decile` 字面字段缺失；当前已通过 `final_flag`、`mdl_dte`、客群字段和 decile 产物覆盖核心分析，但字段别名规则仍建议固化。
