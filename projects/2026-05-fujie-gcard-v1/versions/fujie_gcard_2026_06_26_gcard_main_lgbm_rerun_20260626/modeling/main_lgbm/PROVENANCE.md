# 模型产物溯源 — main_lgbm

本文件说明本 run 内模型文件（`model.pkl`、`scores_all_splits.feather`、`sample.pkl`）是如何产生的，便于在 run 内部直接追溯。

## 产物

| 文件 | 产生方式 |
|---|---|
| `model.pkl` | LightGBM 二分类模型，`rmw train --experiment main_lgbm` 训练得到，best_iter=303 |
| `scores_all_splits.feather` | 同一次训练对全量样本打分（model_score + champion 列），489,743 行 |
| `feature_selection/refine/sample.pkl` | 特征精筛阶段保留的样本元信息（特征矩阵 shape、最终特征名） |

## 训练命令

```bash
rmw train --project projects/2026-05-fujie-gcard-v1 \
  --run-id 2026-06-26-gcard-main-lgbm-rerun \
  --experiment main_lgbm
```

## 代码版本（workbench）

- 训练时 HEAD = `d835597`，工作树含 3 个 local-feather 修复（materialize 清 feature_map / feature_refine None 守卫 / train_lgb champion 转 numeric）。
- 这些修复随后提交为 `50dbb7f`（branch `fix/local-feather-refine-score`），即本模型所用代码状态的稳定指针。
- 字段见 `run_config.json: workbench_git_commit`。

## 输入数据指纹

- 本地 feather：`/Users/guzijun/gcard_1pct_export/sample_50pct.feather`
- 规模：489,743 行 × 2,857 列（= 2,837 候选特征 + 20 基础列）
- SHA-256：见 `run_config.json: input_feather.sha256`（`02b3bc98...`）
- 口径来源：`queries/generated/06_build_d01_d02_wide_table_2837.sql`（59 表 join，2,837 特征）

## 训练配方

- 算法：LightGBM，objective=binary，metric=auc；完整参数见 `run_config.json: params`。
- 切分：训练 DEV(183,739) / 早停 DEV-OOS(183,480) / 评估 OOT+OOT-OOS。
- 入模特征：500（来自 `../../feature_selection/final_features.txt`）。
- 结果：valid(DDEV-OOS) AUC 0.9325 / KS 0.7165；OOT AUC 0.9342；auc_gap 0.013。

## 数据类型注意

源 feather 中 champion 分数 `gcard_v2/v4/v6` 为 string 型，打分阶段已 `pd.to_numeric` 强转（见 `train_lgb.py`）。`scores_all_splits.feather` 中这些列为 float64。

## 复现路径

固定 `workbench_git_commit` 的代码 + `input_feather.sha256` 的输入 + `final_features.txt` 的特征 + `run_config.json: params` 的参数 → 可复现 `model.pkl`。
