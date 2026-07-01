# SQL Evidence Summary — 2026-06-26-gcard-main-lgbm-rerun

数据口径：本地 feather 模式（`data_source_mode: local_feather`）。
本轮 feather 已是 D01/D02 宽表的物化结果，真实远端 DP 拉数不触发；
以下 SQL 仅作为口径可审计证据，供 sql_review_gate 人工确认。

## 1. 数据源契约

- 本地 feather：`/Users/guzijun/gcard_1pct_export/sample_50pct.feather`
- 规模：489,743 行 × 2,857 列（= 2,837 候选特征 + 20 基础/非特征列）
- 标签：`ftr_30d_ord_flag`（正样本率 15.67%）
- 切分：`final_flag` ∈ {DEV 183,739 / DEV-OOS 183,480 / OOT 61,173 / OOT-OOS 61,351}
- 时间：`mdl_dte`（2025-06 ~ 2026-01，8 月）；分区 `ds`
- champion 分数列：`gcard_v2 / gcard_v4 / gcard_v5 / gcard_v6`（已在 feather 中，用于 champion/challenger）

## 2. 宽表构建 SQL（canonical，已注册到本 run）

- 文件：`queries/generated/06_build_d01_d02_wide_table_2837.sql`
- 目标表：`pdm_risk.pdm_risk_fujie_gcard_d01_d02_wide_feature_2837_v6_1`
- 口径：59 张特征表 join → 2,837 特征 + 基础列
- 基础/非特征列（共 20，精筛时必须排除）：
  `uid, mdl_dte, ds, blue_customer_flag, ftr_30d_ord_flag, ftr_30d_ord_amt,
  prc_amt_xz_30d_3m, ovd_amt_xz_30d_3m, final_flag, zc_level,
  gcard_v2, gcard_v4, gcard_v5, gcard_v6, rand_flag0~rand_flag5`
- 物化一致性校验：SQL 2837 特征 == feather 2837 候选特征列 ✅

## 3. 逐表拉数 SQL（prescreen dry-run，70 份）

- 目录：`queries/generated/ads_app_off_feature__dot__*.sql`
- 每份对应一张上游特征表，过滤口径统一：
  `ds = 'YYYYMMDD' and rand_flag0 < 0.1 and final_flag in ('DEV','OOT')`
- 覆盖 8 个分区：20250630 ~ 20260131
- 状态：`feature_prescreen = scaffold(sql_approval_required)`，本地模式下不实际执行 DP 拉数

## 4. 阶段状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| feature_metadata | done | 70 表 / 15,028 候选字段，元数据落 `data/profile/feature_metadata/` |
| feature_prescreen | scaffold | 本地模式 dry-run，仅生成 SQL 证据，不计算 remain_features |
| build_wide_sql | n/a (local) | 宽表已以 feather 形式存在，canonical SQL 已注册；remain_features 链路在本地模式下不适用 |

## 5. 闸门结论

SQL 证据已齐备且口径自洽。精筛（feature_refine）将直接基于本地 feather 的 2,837 候选特征执行
可用性 / 缺失率(0.9) / 常量 / IV(0.005) / 相关性去重(spearman 0.8) / 空重要性(20轮) / 基线重要性(top500)。
非特征列（含 champion 历史分）在精筛前显式排除，杜绝信息泄漏。
