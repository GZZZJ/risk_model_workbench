# Training Summary

| Item | Value |
| --- | --- |
| Status | done |
| Experiment | main_lgbm |
| Algorithm | lightgbm |
| Training mode | llm_guided_tune |
| Message | training completed |
| Input feather | /Users/guzijun/gcard_1pct_export/sample_50pct.feather |
| Feature list | /Users/guzijun/Desktop/AI攻坚/risk_model_workbench/projects/2026-05-fujie-gcard-v1/versions/fujie_gcard_v8_20260709_1710/feature_selection/final_features.txt |
| Candidate features | 474 |
| Actual features | 474 |
| Train values | ["DEV"] |
| Valid values | ["DEV-OOS"] |
| Train AUC | 0.954738 |
| Valid AUC | 0.932943 |
| Train KS | 0.769044 |
| Valid KS | 0.719264 |
| AUC gap | 0.0217943 |

## Parameters

| Parameter | Value |
| --- | --- |
| bagging_freq | 3 |
| colsample_bytree | 0.75 |
| learning_rate | 0.03 |
| max_depth | 7 |
| metric | auc |
| min_child_samples | 100 |
| num_leaves | 63 |
| objective | binary |
| reg_alpha | 0.1 |
| reg_lambda | 2 |
| subsample | 0.82 |
| verbose | -1 |

## Evidence

- train_metrics: `/Users/guzijun/Desktop/AI攻坚/risk_model_workbench/projects/2026-05-fujie-gcard-v1/versions/fujie_gcard_v8_20260709_1710/modeling/main_lgbm/train_metrics.json`
- training_status: `/Users/guzijun/Desktop/AI攻坚/risk_model_workbench/projects/2026-05-fujie-gcard-v1/versions/fujie_gcard_v8_20260709_1710/modeling/main_lgbm/training_status.json`
- feature_importance: `/Users/guzijun/Desktop/AI攻坚/risk_model_workbench/projects/2026-05-fujie-gcard-v1/versions/fujie_gcard_v8_20260709_1710/modeling/main_lgbm/feature_importance.csv`
