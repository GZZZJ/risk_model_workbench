---
name: risk-model-workbench
description: Orchestrate full-chain or ambiguous multi-stage risk modeling in the local Risk Model Workbench. Use when a request spans two or more modeling stages, initializes request/plan/version state, or needs cross-stage stop rules. Do not use for a clearly bounded single-stage training, feature-selection, evaluation, or reporting request; route those to the matching rmw child skill.
---

# Risk Model Workbench

Use the local `rmw` CLI for the 风险场景 AI 建模工作台. Do not reimplement
modeling logic in chat. `jm` is a long-term compatibility alias for existing
automation and handoffs.

## Route by scope

Keep this root skill for full-chain workflows, requests spanning more than one
modeling stage, and ambiguous modeling requests that require cross-stage stop
rules. For a clearly bounded single-stage request, load the matching thin skill
instead of duplicating its operating procedure here:

- Training, tuning, experiments, or Advisor candidates: `rmw-model-training`.
- Metadata, prescreen, wide SQL, refine, leakage, or feature convergence:
  `rmw-feature-selection`.
- Effectiveness, stability, slices, valid/OOS/OOT, or champion/challenger:
  `rmw-model-evaluation`.
- Model reports, model cards, executive summaries, or evidence packages:
  `rmw-report-generation`.

The root skill retains request/plan/version initialization, orchestration across
stages, and cross-stage stop decisions. Child skills never take ownership of a
full modeling chain.

## Source Of Truth

For an existing version, always read:

- `projects/<project>/project_state.yml`
- `projects/<project>/versions/<version_id>/version_state.yml`
- `projects/<project>/versions/<version_id>/audit/artifact_manifest.json`

Legacy `projects/<project>/runs/<run_id>/run_state.yml` remains readable after
migration, but new work should use `versions/<version_id>/`. Stage status comes
from `version_state.yml` or legacy `run_state.yml` plus registered artifacts,
not from loose files in the workspace.

## Preferred Commands

- `rmw doctor`
- `rmw project validate --project <project>`
- `rmw request validate --project <project> --request <request.md>`
- `rmw plan create --project <project> --request <request.md>`
- `rmw version init --project <project> --workflow full_modeling --version-id <version_id>`
- `rmw version list --project <project>`
- `rmw version status --project <project> --version-id <version_id>`
- `rmw version audit --project <project> --version-id <version_id>`
- `rmw version audit --project <project> --version-id <version_id> --strict`
- `rmw version audit --project <project> --version-id <version_id> --json`
- `rmw rules list`
- `rmw lesson promote --project <project> --title <title> --target guardrail --rule-id <id>`
- `rmw sample check --project <project> --version-id <version_id>`
- `rmw feature prescreen --project <project> --version-id <version_id> --dry-run-sql`
- `rmw feature refine --project <project> --version-id <version_id> --dry-run-sql`
- `rmw train --project <project> --version-id <version_id> --experiment main_lgbm`
- `rmw evaluate --project <project> --version-id <version_id>`
- `rmw compare --project <project> --version-id <version_id> --champion <score_column>`
- `rmw report --project <project> --version-id <version_id>`

For Fujie GCard migrated legacy/example baselines, use:

- `rmw run import-gcard-model-artifacts --project projects/2026-05-fujie-gcard-v1 --run-id 2026-06-imported-gcard-main-lgbm`
- `rmw version status --project projects/2026-05-fujie-gcard-v1 --version-id fujie_gcard_2026_06_imported_gcard_main_lgbm`

Use `rmw run ...` only for legacy compatibility checks or explicit legacy-run
inspection.

## Stop Rules

Stop and report before training if sample check artifacts are missing, feature
list is missing, leakage is detected, or SQL approval is required but absent.

Before any DP or `TMLSQLClient` data pull, generate SQL first and require
explicit approval before using `--sql-approved`.

If local feather training data or scored predictions are unavailable, the CLI
may create scaffold artifacts. Do not treat scaffold artifacts as real evidence.
When real artifacts exist, read metrics and reports from the version workspace
and artifact manifest, not from chat memory.

## Request-Driven Workflow

When the user provides a modeling request Markdown file, treat it as the task
contract. Validate it first, generate an execution plan, initialize a version,
then execute tasks from the plan. Do not invent tasks that contradict the
request.

If the user needs to create a request interactively, direct them to
`tools/model_request_builder/index.html`; the downloaded Markdown becomes the
request contract.

## Hardening Loop

When a project-specific script or notebook is useful across versions, turn the
reusable part into a CLI-backed module under `src/risk_model_workbench/`. Keep
project-specific data paths, business definitions, and one-off assumptions in
the project config, request Markdown, or version workspace. Preserve legacy
scripts under `legacy_scripts/` for traceability.
