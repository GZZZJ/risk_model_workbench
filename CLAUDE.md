# CLAUDE.md

This repository is the `risk_model_workbench` local business modeling
workbench, also named `风险场景 AI 建模工作台`. The canonical entrypoint is the
`rmw` CLI. `jm` is a long-term compatibility alias.

The workbench is generic business modeling infrastructure. Fujie GCard is the
current active case project and regression example, not the boundary of the
workbench's reusable capability.

## Current Project State

- Active case project: `projects/2026-05-fujie-gcard-v1/`
- Project checkpoint: `projects/2026-05-fujie-gcard-v1/project_state.yml`
- Active version: `fujie_gcard_action_runner_v1_20260710`
- Active objective: `ActionRunner Agent 建模闭环已完成，进入领域 Skill 验收、规则沉淀与更大样本泛化验证`
- Active workflow/status: `full_modeling` / `done`
- Current strict audit verdict: `complete`

The imported run contains real historical Fujie GCard training, evaluation, and
report artifacts, but it is not proof that the full workflow was rerun locally.
Treat imported or scaffold artifacts as evidence to review, not as local
reproduction evidence.

## Resume Checklist

Before continuing any modeling work:

1. Read `projects/2026-05-fujie-gcard-v1/project_state.yml`.
2. Read `versions/<version_id>/version_state.yml`.
3. Read `versions/<version_id>/audit/artifact_manifest.json`.
4. Review the latest handoff under `handoffs/` and retrospective under
   `retrospectives/` when present.
5. Run `rmw project status --project projects/2026-05-fujie-gcard-v1`.
6. Run `rmw version audit --project projects/2026-05-fujie-gcard-v1 --version-id <version_id> --strict`
   before treating a stage or version as closed.

Avoid relying on conversation memory.

## Operating Rules

- Use `rmw` instead of running project scripts directly unless the user
  explicitly asks for legacy-script inspection.
- Do not overwrite previous versions. Create a new `version_id` or get explicit
  approval.
- Before any DP or `TMLSQLClient` data pull, generate SQL first and require
  explicit approval before running with `--sql-approved`.
- Never modify `vendor/feature-select-v2/scripts/code/` unless explicitly
  asked.
- Never commit raw data, local feather/parquet exports, scored datasets, or secrets.
  Intentional registered model binaries may be committed for reproducibility.
- New reusable workflow logic belongs under `src/risk_model_workbench/`.
  Project-specific definitions belong in project config, request Markdown, or
  the version workspace.

## Useful Commands

```bash
rmw doctor
rmw project validate --project projects/2026-05-fujie-gcard-v1
rmw project status --project projects/2026-05-fujie-gcard-v1
rmw version status --project projects/2026-05-fujie-gcard-v1 --version-id fujie_gcard_action_runner_v1_20260710
rmw version audit --project projects/2026-05-fujie-gcard-v1 --version-id fujie_gcard_action_runner_v1_20260710 --strict
pytest tests -q
```

The same commands can still be run with `jm` for compatibility, but new docs
and handoffs should prefer `rmw`.

## Skill Routing

For every modeling request, you MUST invoke exactly one matching installed
project Skill before inspecting files, calling another tool, or selecting a
generic modeling Skill/Agent. These project routing rules take precedence over
generic modeling helpers:

- Full-chain, two-or-more-stage, or ambiguous workflow: `risk-model-workbench`.
- Training, tuning rounds, experiments, or Advisor candidates: `rmw-model-training`.
- Metadata, prescreen, wide SQL, refine, leakage, or convergence: `rmw-feature-selection`.
- Effectiveness, stability, slices, PSI, or champion/challenger: `rmw-model-evaluation`.
- Report, model card, executive summary, or evidence package: `rmw-report-generation`.

Missing inputs do not change routing. Invoke the matching single-stage Skill,
then apply its stop rules. Do not let a generic Skill or Agent capture an RMW
single-stage request, and do not let a child Skill capture a full chain.

For long or noisy tasks, use project subagents:

- sample-auditor
- feature-selector
- modeling-engineer
- evaluation-reviewer
- report-writer
