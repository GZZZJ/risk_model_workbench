# risk-model-workbench

风险场景 AI 建模工作台。工作台提供可复用的本地建模流程：项目初始化、需求校验、执行计划、样本检查、特征筛选、模型训练、评估、对比、报告生成，以及跨会话连续性交接。

复借 G 卡是当前已承接的活跃案例项目，用于验证真实历史产物标准化、评估报告和连续性交接机制；工作台能力不与该项目强绑定。

标准入口是 `rmw` CLI。`jm` 是长期兼容别名；`agent.py` 和 `jingying-agent` 仅作为历史兼容入口保留。

<!-- RMW_CURRENT_STATE:START -->
```yaml
project: projects/2026-05-fujie-gcard-v1
active_version_id: fujie_gcard_v8_20260709_1710
objective: 复借G卡主模型从0重跑：全链路样本检查/特征收敛/LGBM训练/评估/对比/报告
workflow: full_modeling
status: done
```
<!-- RMW_CURRENT_STATE:END -->

详细规划见 [docs/legacy/AI经营建模Agent规划.md](docs/legacy/AI经营建模Agent规划.md)。

## RMW Agent

`RMW Agent` 使用内嵌 LangGraph/LangChain runtime 负责目标理解、诊断和有界
调参建议；RMW Harness 始终负责确定性执行、状态、门禁和审计证据。模型通过
统一的 LangChain gateway 接入，可以选择 OpenAI、Anthropic Claude 或兼容端点。

快速开始：

```bash
.venv/bin/rmw agent model status

# 可选：先把自然语言目标生成一份待确认的建模请求草案
.venv/bin/rmw agent request draft \
  --project projects/2026-05-fujie-gcard-v1 \
  --request-id fujie-gcard-goal-v1 \
  --objective "重跑复借G卡主模型，检查样本与特征稳定性，并生成模型报告"

rmw agent start \
  --project projects/2026-05-fujie-gcard-v1 \
  --request projects/2026-05-fujie-gcard-v1/requests/model_request_template.md \
  --version-id fujie_gcard_agent_v1_20260706 \
  --workflow full_modeling

rmw agent run \
  --project projects/2026-05-fujie-gcard-v1 \
  --version-id fujie_gcard_agent_v1_20260706

rmw agent status \
  --project projects/2026-05-fujie-gcard-v1 \
  --version-id fujie_gcard_agent_v1_20260706
```

如需使用 Claude 模型，仅安装轻量的 LangChain Provider 适配：

```bash
.venv/bin/python -m pip install -e ".[modeling,agent-anthropic]"
export RMW_AGENT_PROVIDER=anthropic
export RMW_AGENT_MODEL=<claude-model-name>
export ANTHROPIC_API_KEY=<secret>
.venv/bin/rmw agent model status
```

如果 Agent 因 SQL/DP action 暂停，先审查生成的 SQL 和 approval 记录，再显式批准：

```bash
rmw agent approve \
  --project projects/2026-05-fujie-gcard-v1 \
  --version-id fujie_gcard_agent_v1_20260706 \
  --approval-id <approval_id> \
  --approved-by <name> \
  --note "reviewed SQL evidence"
```

Agent 产物写入 version workspace，包括 `agent_plan.yml`、
`audit/agent_state.yml`、`audit/agent_trace.jsonl`、`audit/approvals.yml`
、`audit/advisor_requests/`、`audit/model_invocations/`，默认运行时还会写入
LangGraph checkpoint。
Advisor request/response 是内嵌推理层与 Harness 之间的持久化合同；
`rmw agent advisor list/show` 主要用于审计和旧版本兼容。
最终闭环仍以 `version_state.yml`、`audit/artifact_manifest.json` 和
`rmw version audit --strict` 为准。

## 当前状态

截至 2026-07-27：

- 通用工作台代码在 `src/risk_model_workbench/`。
- 项目模板在 `templates/project/`。
- 工作流定义在 `workflows/`。
- 当前活跃案例项目是 `projects/2026-05-fujie-gcard-v1/`。
- 当前项目断点文件是 `projects/2026-05-fujie-gcard-v1/project_state.yml`。
- 当前 active version 是 `fujie_gcard_v8_20260709_1710`。
- 当前目标是 `复借G卡主模型从0重跑：全链路样本检查/特征收敛/LGBM训练/评估/对比/报告`。
- `rmw project status` 显示项目状态为 `active`，active version 状态为 `done`，10 个阶段均已完成。
- `rmw version audit --strict` 的执行 verdict 为 `complete`；复现性口径为 `workspace_dependent`，原因是该版本依赖本地 workspace 证据。

active version 是本地全链路重跑证据。历史 imported run 和 legacy run 继续保留用于兼容读取与 lineage 审计，但不作为本轮闭环证据。

## 当前案例：复借 G 卡

- 活跃版本：`projects/2026-05-fujie-gcard-v1/versions/fujie_gcard_v8_20260709_1710/`
- 样本表：`ads_app_off_feature.ds29531_backtrack_fj_gcard_model_v6_1_sample`
- 本地建模样本：489,743 行；样本检查无重复主键。
- 训练产物：`modeling/main_lgbm/model.pkl`、`train_metrics.json`、`feature_importance.csv`、`run_config.json`
- 评估产物：`evaluation/overall_metrics.csv`、`monthly_metrics.csv`、`segment_metrics.csv`、`decile_lift_*.csv`、`score_psi_by_month.csv`
- 报告产物：`reports/model_report.xlsx`、`model_report.md`、`model_report.html`、`model_card.md`、`executive_summary.md`
- 核心效果：OOT AUC `0.9345`，OOT KS `0.7318`。

## Source Of Truth

继续任何建模任务前先看这些文件：

- `projects/2026-05-fujie-gcard-v1/project_state.yml`
- `projects/2026-05-fujie-gcard-v1/versions/<version_id>/version_state.yml`
- `projects/2026-05-fujie-gcard-v1/versions/<version_id>/audit/artifact_manifest.json`
- `projects/2026-05-fujie-gcard-v1/handoffs/` 下最新交接文档
- `projects/2026-05-fujie-gcard-v1/retrospectives/` 下最新复盘文档
- `projects/2026-05-fujie-gcard-v1/docs/lessons.md`

阶段状态以 active version 的 `version_state.yml` 和 registered artifacts 为准。目录里存在但未登记到 manifest 的文件不能直接当作阶段闭环证据；`runs/<run_id>/` 仅作为 legacy/lineage 兼容读取路径。

## 安装与检查

```bash
/opt/anaconda3/bin/python -m venv .venv
.venv/bin/python -m pip install -e ".[modeling,agent-openai]"

export RMW_AGENT_PROVIDER=openai
export RMW_AGENT_MODEL=<model-name>
export OPENAI_API_KEY=<secret>

.venv/bin/rmw doctor
.venv/bin/rmw project validate --project projects/2026-05-fujie-gcard-v1
.venv/bin/python -m pytest tests -q
```

`rmw doctor` 会检查规划文档、模型资产索引、vendored feature-select-v2、项目模板和核心 workflow 是否存在；复借 G 卡历史工作簿只作为 legacy/example 资料提示。

## 常用命令

查看项目连续性状态：

```bash
rmw project status --project projects/2026-05-fujie-gcard-v1
```

刷新项目断点文件：

```bash
rmw project status --project projects/2026-05-fujie-gcard-v1 --write-state
```

更新项目断点（新建或切换版本时使用 `--active-version-id`）：

```bash
rmw project update-state \
  --project projects/2026-05-fujie-gcard-v1 \
  --active-version-id fujie_gcard_v8_20260709_1710 \
  --objective "复借G卡主模型从0重跑：全链路样本检查/特征收敛/LGBM训练/评估/对比/报告" \
  --next-action "评审报告、模型卡和执行摘要，决定是否进入发布/准入评审"
```

查看版本状态与审计：

```bash
rmw version status \
  --project projects/2026-05-fujie-gcard-v1 \
  --version-id fujie_gcard_v8_20260709_1710 \
  --progress

rmw version audit \
  --project projects/2026-05-fujie-gcard-v1 \
  --version-id fujie_gcard_v8_20260709_1710 \
  --strict
```

写会话交接：

```bash
rmw handoff write --project projects/2026-05-fujie-gcard-v1
```

写显式复盘：

```bash
rmw retrospective write \
  --project projects/2026-05-fujie-gcard-v1 \
  --scope session \
  --note "本次会话完成连续性交接能力建设"
```

记录项目经验：

```bash
rmw lesson add \
  --project projects/2026-05-fujie-gcard-v1 \
  --title "SQL approval gate" \
  --kind guardrail \
  --body "Dry-run SQL must be reviewed before any DP pull."
```

`handoff write` 和 `retrospective write` 都是显式收尾动作；CLI 不猜测会话是否结束。

## 需求驱动流程

使用静态需求生成器创建需求文档：

```bash
open tools/model_request_builder/index.html
```

校验需求并生成执行计划：

```bash
rmw request validate \
  --project projects/2026-05-fujie-gcard-v1 \
  --request projects/2026-05-fujie-gcard-v1/requests/model_request_template.md

rmw plan create \
  --project projects/2026-05-fujie-gcard-v1 \
  --request projects/2026-05-fujie-gcard-v1/requests/model_request_template.md
```

把需求文档和执行计划绑定到一个新版本：

```bash
rmw version init \
  --project projects/2026-05-fujie-gcard-v1 \
  --workflow full_modeling \
  --version-id <version_id> \
  --request projects/2026-05-fujie-gcard-v1/requests/model_request_template.md \
  --plan projects/2026-05-fujie-gcard-v1/requests/2026-06-fujie-gcard-baseline.execution_plan.yml
```

不要覆盖已有版本。需要重跑时创建新的 `version_id`；`--force` 仅用于明确批准的覆盖场景。

## 项目与版本初始化

新建模型项目：

```bash
rmw init-project \
  --name 2026-xx-new-model \
  --display-name 新模型名称 \
  --scenario 业务场景 \
  --template generic
```

登记一个未绑定需求文档的版本：

```bash
rmw version init \
  --project projects/2026-05-fujie-gcard-v1 \
  --workflow full_modeling \
  --version-id <version_id>
```

`rmw run` 仅保留给历史 run 的兼容读取和迁移；新工作应使用 `rmw version`。需要迁移旧 run 时，使用：

```bash
rmw version migrate-run \
  --project projects/2026-05-fujie-gcard-v1 \
  --run-id 2026-06-imported-gcard-main-lgbm
```

迁移或导入的历史产物不代表本地重新执行了全链路。

## 特征筛选与 SQL Gate

导出特征表元数据：

```bash
rmw feature metadata --project projects/2026-05-fujie-gcard-v1 --version-id <version_id>
```

先生成特征初筛取数 SQL 给使用者确认，不拉数：

```bash
rmw feature prescreen \
  --project projects/2026-05-fujie-gcard-v1 \
  --version-id <version_id> \
  --dry-run-sql
```

确认 SQL 后执行特征初筛，数据先落本地 feather：

```bash
rmw feature prescreen \
  --project projects/2026-05-fujie-gcard-v1 \
  --version-id <version_id> \
  --refresh-dp-cache \
  --sql-approved
```

生成特征初筛后的宽表 SQL：

```bash
rmw build-wide-sql --project projects/2026-05-fujie-gcard-v1 --version-id <version_id>
```

先生成宽表后收敛取数 SQL 给使用者确认，不拉数：

```bash
rmw feature refine \
  --project projects/2026-05-fujie-gcard-v1 \
  --version-id <version_id> \
  --dry-run-sql
```

确认 SQL 后执行全局相关性、随机噪声、空标签重要性和基线重要性筛选：

```bash
rmw feature refine \
  --project projects/2026-05-fujie-gcard-v1 \
  --version-id <version_id> \
  --refresh-dp-cache \
  --sql-approved
```

任何 DP 或 `TMLSQLClient` 取数都必须先 dry-run 展示 SQL，得到使用者明确批准后才能带 `--sql-approved` 执行。

## 训练、评估和报告

在有本地 feather 训练数据和特征清单时执行 LightGBM 训练：

```bash
rmw train \
  --project projects/2026-05-fujie-gcard-v1 \
  --version-id <version_id> \
  --experiment main_lgbm \
  --input-feather <path/to/modeling_sample.feather> \
  --feature-list <path/to/feature_list.txt>
```

在有本地打分 feather 时执行标准评估：

```bash
rmw evaluate \
  --project projects/2026-05-fujie-gcard-v1 \
  --version-id <version_id> \
  --scores-feather <path/to/scores_all_splits.feather>
```

生成 champion/challenger 对比：

```bash
rmw compare \
  --project projects/2026-05-fujie-gcard-v1 \
  --version-id <version_id> \
  --champion gcard_v6
```

从标准训练和评估产物生成报告：

```bash
rmw report --project projects/2026-05-fujie-gcard-v1 --version-id <version_id>
```

如果本地 feather 训练数据或打分结果不可用，部分命令可能生成 scaffold artifact。scaffold artifact 不能当成真实建模证据。

## 当前特征筛选口径

- 特征初筛：在候选特征过多时先抽样检查基础质量和稳定性，默认关注缺失率、恒一/近恒一值占比、PSI 等规则，减少后续宽表规模。
- 宽表 SQL：初筛后把保留特征组装为宽表 SQL，尽量使用更完整的数据进入后续精筛。
- 特征精筛：在宽表样本上做可用性过滤、全局相关性去重、随机噪声重要性、Null Importance 和基线模型重要性筛选。
- 抽样：每张特征表使用 `feature_select.prescreen.sampling.where`，宽表后收敛使用 `configs/refine_features.yaml`。
- 执行策略：每个特征组或特征表单独筛选，支持 checkpoint 跳过已完成表。
- 宽表 join 主键：`uid`、`mdl_dte`。
- `ds` 作为底表保留字段和过滤字段，不作为特征表 join key。

## 目录说明

- `src/risk_model_workbench/`：通用工作台模块和 CLI 实现。
- `src/jingying_model_agent/`、`src/jingying_agent/`、`jingying_agent/`、`agent.py`：兼容层和历史入口。
- `projects/2026-05-fujie-gcard-v1/configs/`：项目配置。
- `projects/2026-05-fujie-gcard-v1/queries/`：SQL 草稿和生成 SQL。
- `projects/2026-05-fujie-gcard-v1/versions/`：版本 workspace、状态、审计、模型、评估和报告产物；新工作以此为准。
- `projects/2026-05-fujie-gcard-v1/runs/`：历史 run 兼容读取与 lineage 审计目录。
- `projects/2026-05-fujie-gcard-v1/legacy_scripts/`：历史项目脚本，仅作回溯和迁移参考。
- `tools/model_request_builder/`：静态模型需求生成器。
- `vendor/feature-select-v2/`：vendored feature selection 实现。

## 外部环境依赖

项目代码本身不依赖其他 Git 项目。实际执行取数、筛选和建模时，运行环境仍可能需要公司内部或三方 Python 包：

- `tmlpatch`：提供 `TMLSQLClient`
- `pandas`
- `numpy`
- `toad`：特征初筛质量规则优先使用；缺失时可通过脚本参数或配置使用 native selector
- `lightgbm`
- `xgboost`
- `pyarrow`

## 标准化边界

通用逻辑放在 `src/risk_model_workbench/`。项目特定口径放在项目配置、请求文档、version workspace 或 `legacy_scripts/`。

真实项目里的临时脚本和产物应先迁移或登记到标准 version，再判断哪些逻辑值得固化为通用 CLI。不要直接把一次性路径、样本口径或业务假设写进通用模块。

`vendor/feature-select-v2/scripts/code/` 视为只读，除非使用者明确要求修改。

## 拉取仓库

本仓库不依赖 Git submodule，直接 clone 即可：

```bash
git clone <repo-url> risk_model_workbench
```
