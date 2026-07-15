# 首借T0 G卡新模型 项目 Workspace

场景：preloan_credit_card

创建日期：2026-07-14

本目录是一次独立建模项目的 workspace。所有配置、脚本、运行记录、模型、指标和报告都保存在本目录下，避免不同模型项目互相污染。

## 目录说明

- `project.yml`：项目总索引的 canonical 真源；`project.yaml` 是自动生成的兼容镜像。
- `configs/`：各步骤配置。
- `scripts/`：当前项目可执行脚本。
- `queries/`：DP 探查 SQL。
- `data/`：本地样本、抽样、加工和探查结果。
- `versions/`：模型版本的状态、日志、快照和产物。
- `runs/`：旧版本兼容目录；新建工作默认使用 `versions/`。
- `reports/`：最终报告。

## 建议执行顺序

1. 补齐 `project.yml`、`configs/feature_tables.txt`、`configs/feature_select.yaml`、`configs/refine_features.yaml` 中的样本表、特征表、标签列、切分列、分区和宽表名。
2. 导出特征表元数据：
   `python3 scripts/00_export_feature_metadata.py`
3. 生成 feature-select-v2 适配配置：
   `python3 scripts/02_feature_select.py`
4. 先只生成特征初筛取数 SQL，给使用者确认：
   `rmw feature prescreen --project <project> --version-id <version_id> --dry-run-sql --max-tables 1`
5. SQL 确认后执行特征初筛：
   `rmw feature prescreen --project <project> --version-id <version_id> --refresh-dp-cache --sql-approved`
6. 生成特征初筛后的宽表 SQL：
   `rmw build-wide-sql --project <project> --version-id <version_id>`
7. 先只生成宽表后收敛取数 SQL，给使用者确认：
   `python3 scripts/08_refine_wide_features.py --dry-run-sql`
8. SQL 确认后刷新本地 feather 并执行全局相关性、随机噪声、空标签重要性和基线重要性筛选：
   `python3 scripts/08_refine_wide_features.py --refresh-dp-cache --sql-approved`

所有从 DP 获取数据的步骤都会先写本地 feather，并在 `data/profile/dp_feather_datasets/` 记录 SQL、数据说明、存储位置和行列数；真实 feather 数据位于 `data/local/dp_feather/`，不进入 Git。
