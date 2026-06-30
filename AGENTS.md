# AGENTS.md

## Project Purpose

This repository is the `risk_model_workbench` local business modeling
workbench, also named `风险场景 AI 建模工作台`. The canonical entrypoint is the
`rmw` CLI. `jm` is a long-term compatibility alias.

The workbench is generic business modeling infrastructure. Fujie GCard is the
current active case project and regression example; reusable workbench behavior
must not depend on that project unless explicitly scoped as legacy/example.

## Current State

As of 2026-06-09:

- Active case project: `projects/2026-05-fujie-gcard-v1/`
- Active run: `2026-06-imported-gcard-main-lgbm`
- Project checkpoint: `projects/2026-05-fujie-gcard-v1/project_state.yml`
- Current objective: `复借G卡主模型产物标准化与连续性交接机制建设`
- Run workflow/status: `imported_gcard_main_lgbm` / `imported`
- Run current stage: `feature_refine`
- `rmw project status` reports stage counts `done=6, pending=3, scaffold=1`.
- `rmw run audit` currently reports verdict `open` because
  `feature_metadata`, `feature_prescreen`, and `build_wide_sql` are pending,
  completed stages are imported evidence, and `feature_refine` is scaffold evidence.

The imported run contains real historical Fujie GCard model artifacts, but it is
not local end-to-end rerun evidence.

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
- `projects/2026-05-fujie-gcard-v1/runs/<run_id>/`: canonical run workspace.
- `projects/2026-05-fujie-gcard-v1/runs/<run_id>/run_state.yml`: run state
  source of truth.
- `projects/2026-05-fujie-gcard-v1/runs/<run_id>/audit/artifact_manifest.json`:
  registered artifact source of truth.
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
- Show run state: `rmw run status --project projects/2026-05-fujie-gcard-v1 --run-id <run_id>`
- Audit run closure: `rmw run audit --project projects/2026-05-fujie-gcard-v1 --run-id <run_id>`
- Strict audit gate: `rmw run audit --project projects/2026-05-fujie-gcard-v1 --run-id <run_id> --strict`
- Write handoff: `rmw handoff write --project projects/2026-05-fujie-gcard-v1 --run-id <run_id>`
- Write retrospective: `rmw retrospective write --project projects/2026-05-fujie-gcard-v1 --run-id <run_id>`
- Add lesson: `rmw lesson add --project projects/2026-05-fujie-gcard-v1 --title <title> --body <body>`
- Promote lesson: `rmw lesson promote --project projects/2026-05-fujie-gcard-v1 --title <title> --target guardrail --rule-id <id>`
- List workbench rules: `rmw rules list`
- Run tests: `pytest tests -q`

`jm` remains compatible with these commands, and `jm status` still exists as a
legacy alias for run state, but prefer `rmw run status` in documentation and
handoffs.

## Workflow Rules

- Use `rmw` for workflow actions. Do not run project scripts directly unless the
  user explicitly asks for legacy-script inspection or migration.
- Before resuming work, read `project_state.yml`, `run_state.yml`, and
  `audit/artifact_manifest.json`.
- Treat `runs/<run_id>/run_state.yml` and registered artifacts as source of
  truth for stage status.
- Use `rmw run audit` before declaring a stage or run closed.
- Use `rmw workflow validate` after changing workflow stage contracts.
- Reusable guardrails live in `docs/workbench_rules.yml`; ADR and glossary
  entries live under `docs/adr/` and `docs/glossary.md`.
- `handoff write` and `retrospective write` are explicit checkpoint actions; do
  not infer session completion from conversation state.
- When a modeling request Markdown file is provided, validate it with
  `rmw request validate`, create an execution plan with `rmw plan create`, and
  bind the request/plan into a new run with `rmw run init`.

## Safety Rules

- Never modify `vendor/feature-select-v2/scripts/code/` unless explicitly asked.
- Never commit raw data, local feather/parquet data exports, scored datasets, or
  secrets. Model binary artifacts such as `model.pkl` may be committed when they
  are intentional registered run artifacts for reproducibility and traceability.
- Before any DP or `TMLSQLClient` data pull, generate SQL first and require
  explicit approval before using `--sql-approved`.
- Do not overwrite previous runs. Create a new `run_id` or require explicit
  approval.
- Imported or scaffold artifacts are not local reproduction evidence.
- Reusable logic belongs under `src/risk_model_workbench/`; project-specific
  definitions belong in project config, request Markdown, or run workspaces.

## Done Means

- Relevant tests pass.
- CLI smoke command passes.
- New or changed workflow writes artifacts into `runs/<run_id>/`.
- `run_state.yml` and `audit/artifact_manifest.json` are updated when workflow
  artifacts change.
- `project_state.yml`, handoff, retrospective, or lessons are updated when the
  task changes project continuity state.
- The final response reports what changed and what was not verified.
