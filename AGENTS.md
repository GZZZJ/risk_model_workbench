# AGENTS.md

## Project Purpose

This repository is the `risk_model_workbench` local business modeling
workbench, also named `风险场景 AI 建模工作台`. The canonical entrypoint is the
`rmw` CLI. `jm` is a long-term compatibility alias.

The workbench is generic business modeling infrastructure. Fujie GCard is the
current active case project and regression example; reusable workbench behavior
must not depend on that project unless explicitly scoped as legacy/example.

<!-- RMW_CURRENT_STATE:START -->
```yaml
project: projects/2026-05-fujie-gcard-v1
active_version_id: fujie_gcard_v8_20260709_1710
objective: 复借G卡主模型从0重跑：全链路样本检查/特征收敛/LGBM训练/评估/对比/报告
workflow: full_modeling
status: done
```
<!-- RMW_CURRENT_STATE:END -->

## Current State

As of 2026-07-01:

- Active case project: `projects/2026-05-fujie-gcard-v1/`
- Active version: `fujie_gcard_v8_20260709_1710`
- Legacy active run: `20260625_211825_006138`
- Project checkpoint: `projects/2026-05-fujie-gcard-v1/project_state.yml`
- Current objective: `复借G卡主模型从0重跑：全链路样本检查/特征收敛/LGBM训练/评估/对比/报告`
- Active version workflow/status: `full_modeling` / `done`
- Historical standard runs have been copied into `versions/` with lineage back
  to their legacy `runs/<run_id>/` directories.

## Important Directories

- `src/risk_model_workbench/`: reusable modeling workbench code.
- `projects/`: concrete modeling project workspaces.
- `projects/2026-05-fujie-gcard-v1/`: current Fujie GCard case project.
- `projects/2026-05-fujie-gcard-v1/project_state.yml`: project-level
  continuity checkpoint.
- `projects/2026-05-fujie-gcard-v1/handoffs/`: explicit session handoffs.
- `projects/2026-05-fujie-gcard-v1/retrospectives/`: explicit session, stage,
  or project retrospectives.
- `projects/2026-05-fujie-gcard-v1/docs/lessons.md`: project lessons that may
  later be promoted into CLI guardrails, tests, or skills.
- `projects/2026-05-fujie-gcard-v1/versions/<version_id>/`: canonical version
  workspace.
- `projects/2026-05-fujie-gcard-v1/versions/<version_id>/version_state.yml`:
  version state source of truth.
- `projects/2026-05-fujie-gcard-v1/versions/<version_id>/audit/artifact_manifest.json`:
  registered artifact source of truth.
- `projects/2026-05-fujie-gcard-v1/runs/<run_id>/`: legacy run workspace,
  readable for compatibility.
- `workflows/`: reusable workflow definitions.
- `templates/project/`: project workspace template.
- `vendor/feature-select-v2/`: vendored feature selection implementation; treat
  as read-only unless explicitly asked.
- `.agents/skills/`: Codex skills.
- `.claude/skills/` and `.claude/agents/`: Claude Code extensions.

## Commands

- Install editable package: `pip install -e ".[modeling]"`
- Check environment: `rmw doctor`
- Validate project: `rmw project validate --project projects/2026-05-fujie-gcard-v1`
- Show project status: `rmw project status --project projects/2026-05-fujie-gcard-v1`
- List versions: `rmw version list --project projects/2026-05-fujie-gcard-v1`
- Show version state: `rmw version status --project projects/2026-05-fujie-gcard-v1 --version-id <version_id>`
- Audit version closure: `rmw version audit --project projects/2026-05-fujie-gcard-v1 --version-id <version_id>`
- Strict audit gate: `rmw version audit --project projects/2026-05-fujie-gcard-v1 --version-id <version_id> --strict`
- Write handoff: `rmw handoff write --project projects/2026-05-fujie-gcard-v1`
- Write retrospective: `rmw retrospective write --project projects/2026-05-fujie-gcard-v1`
- Add lesson: `rmw lesson add --project projects/2026-05-fujie-gcard-v1 --title <title> --body <body>`
- Promote lesson: `rmw lesson promote --project projects/2026-05-fujie-gcard-v1 --title <title> --target guardrail --rule-id <id>`
- List workbench rules: `rmw rules list`
- Run tests: `pytest tests -q`

`jm` remains compatible with legacy commands, and `rmw run status/audit` still
read migrated legacy runs by lineage. Prefer `rmw version ...` in documentation
and handoffs.

## Workflow Rules

- Use `rmw` for workflow actions. Do not run project scripts directly unless the
  user explicitly asks for legacy-script inspection or migration.
- Before resuming work, read `project_state.yml`, `version_state.yml`, and
  `audit/artifact_manifest.json`.
- Treat `versions/<version_id>/version_state.yml` and registered artifacts as
  source of truth for stage status.
- Use `rmw version audit` before declaring a stage or version closed.
- Use `rmw workflow validate` after changing workflow stage contracts.
- Reusable guardrails live in `docs/workbench_rules.yml`; ADR and glossary
  entries live under `docs/adr/` and `docs/glossary.md`.
- `handoff write` and `retrospective write` are explicit checkpoint actions; do
  not infer session completion from conversation state.
- When a modeling request Markdown file is provided, validate it with
  `rmw request validate`, create an execution plan with `rmw plan create`, and
  bind the request/plan into a new version with `rmw version init`.

## Host-Agent LLM Tuning

For binary LightGBM requests with `training.mode: llm_guided_tune`, the outer
Codex or ClaudeCode session is the preferred tuning advisor. The workbench
should not require a separate LLM API client for this workflow.

If `rmw train` returns `advisor_required`, read the emitted
`modeling/<experiment>/tuning_context_round_<n>.json`, write a structured plan to
`modeling/<experiment>/llm_tuning_plan_round_<n>.json`, then rerun `rmw train`.
Use 3-5 bounded candidates, explain each candidate, and let the workbench's
metric rules select the final trial. Do not optimize directly against OOT as the
primary tuning target.

Plan file shape:

```json
{
  "round": 1,
  "diagnosis": "brief diagnosis from metrics and trial history",
  "candidates": [
    {
      "name": "regularized_capacity",
      "params": {
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": 7,
        "min_child_samples": 120,
        "subsample": 0.82,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "bagging_freq": 3,
        "num_boost_round": 1200,
        "early_stopping_rounds": 80
      },
      "reason": "why this candidate should help"
    }
  ],
  "stop": false
}
```

## Safety Rules

- Never modify `vendor/feature-select-v2/scripts/code/` unless explicitly asked.
- Never commit raw data, local feather/parquet data exports, scored datasets, or
  secrets. Model binary artifacts such as `model.pkl` may be committed when they
  are intentional registered version artifacts for reproducibility and traceability.
- Before any DP or `TMLSQLClient` data pull, generate SQL first and require
  explicit approval before using `--sql-approved`.
- Do not overwrite previous versions. Create a new `version_id` or require explicit
  approval.
- Imported or scaffold artifacts are not local reproduction evidence.
- Reusable logic belongs under `src/risk_model_workbench/`; project-specific
  definitions belong in project config, request Markdown, or run workspaces.

## Done Means

- Relevant tests pass.
- CLI smoke command passes.
- New or changed workflow writes artifacts into `versions/<version_id>/`.
- `version_state.yml` and `audit/artifact_manifest.json` are updated when workflow
  artifacts change.
- `project_state.yml`, handoff, retrospective, or lessons are updated when the
  task changes project continuity state.
- The final response reports what changed and what was not verified.
