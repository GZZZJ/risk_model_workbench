# Training Summary

| Item | Value |
| --- | --- |
| Status | done |
| Experiment | main_lgbm |
| Algorithm | lightgbm |
| Training mode | llm_guided_tune |
| Message | training completed |
| Input feather | /Users/guzijun/gcard_1pct_export/sample_50pct.feather |
| Feature list | /private/tmp/rmw-agent-harness-p1-p2/projects/2026-05-fujie-gcard-v1/versions/fujie_gcard_agent_v1_20260710/feature_selection/final_features.txt |
| Candidate features | 500 |
| Actual features | 500 |
| Train values | ["DEV"] |
| Valid values | ["DEV-OOS"] |
| Train AUC | 0.947691 |
| Valid AUC | 0.93273 |
| Train KS | 0.753509 |
| Valid KS | 0.718788 |
| AUC gap | 0.0149612 |

## Parameters

| Parameter | Value |
| --- | --- |
| bagging_freq | 3 |
| colsample_bytree | 0.7 |
| learning_rate | 0.03 |
| max_depth | 5 |
| metric | auc |
| min_child_samples | 160 |
| num_leaves | 31 |
| objective | binary |
| reg_alpha | 0.2 |
| reg_lambda | 2 |
| subsample | 0.75 |
| verbose | -1 |

## Evidence

- train_metrics: `/private/tmp/rmw-agent-harness-p1-p2/projects/2026-05-fujie-gcard-v1/versions/fujie_gcard_agent_v1_20260710/modeling/main_lgbm/train_metrics.json`
- training_status: `/private/tmp/rmw-agent-harness-p1-p2/projects/2026-05-fujie-gcard-v1/versions/fujie_gcard_agent_v1_20260710/modeling/main_lgbm/training_status.json`
- feature_importance: `/private/tmp/rmw-agent-harness-p1-p2/projects/2026-05-fujie-gcard-v1/versions/fujie_gcard_agent_v1_20260710/modeling/main_lgbm/feature_importance.csv`
