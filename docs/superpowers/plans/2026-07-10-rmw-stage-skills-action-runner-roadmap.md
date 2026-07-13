# RMW P1/P2 Stage Skills And ActionRunner Iteration Roadmap

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents are available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split Host-Agent domain knowledge into independently maintainable stage skills, then introduce a shared typed ActionRunner so skills, the Agent runtime, the CLI, and future adapters all use one deterministic execution path.

**Architecture:** P1 adds thin, non-executing stage skills above the existing `rmw` Tool/CLI boundary. P2 adds typed action contracts and application services below that boundary, migrates one stage at a time, and leaves `rmw` as a backward-compatible thin adapter. Harness policy, state, approval, trace, manifest, and audit remain cross-cutting control surfaces.

**Tech Stack:** Python 3, dataclasses/typing, argparse, PyYAML, pytest, existing `rmw` CLI, existing workflow/action/tool registries, Markdown skills under `.agents/skills/`, optional local MCP adapter only after the core contract is stable.

---

## 1. Scope And Fixed Decisions

### In scope

- P1 thin stage skills:
  - `rmw-model-training`
  - `rmw-feature-selection`
  - `rmw-model-evaluation`
  - `rmw-report-generation`
- Root routing changes in `risk-model-workbench`.
- Skill trigger, boundary, and output-quality evals.
- P2 typed `ActionRequest` / `ActionResult` and `ActionRunner`.
- CLI-to-application-service extraction by stage.
- Agent Executor migration from `cli.main(argv)` to typed action invocation.
- Machine-readable Host-Agent capabilities/tool schema.
- Synthetic and real-version dogfood.

### Explicitly out of scope

- Removing the `rmw` CLI.
- Embedding an LLM provider in RMW.
- Letting skills mutate `version_state.yml` or the artifact manifest directly.
- Multi-user service, daemon, RBAC, distributed queue, or Web console.
- Multi-Agent autonomous negotiation.
- A separate skill for every feature filter, metric, or report format.
- Committing raw data, scored feather/parquet files, local sample caches, or secrets.

## 2. Preconditions

P1 may start after the current CLI/tool registry is treated as the stable execution boundary.

P2 must not start until P0 closes these invariants:

- [ ] SQL remote operations follow `prepare -> approval -> execute`.
- [ ] Advisor responses are consumed deterministically and exactly once.
- [ ] Agent plans and tool invocations fail closed on tampering or registry drift.
- [ ] `review_ready`, `waiting_for_approval`, and scaffold evidence cannot unlock downstream stages.
- [ ] A new Agent-managed version can finish strict audit without Host-Agent bypassing the runtime.

If any P0 invariant is open, P2 work stops. Do not hide a P0 defect inside ActionRunner migration.

## 3. Target Layering

```text
Host-Agent
  -> root/full-chain skill or one stage skill
  -> typed Tool/Action contract
  -> Harness policy/state/approval/trace
  -> ActionRunner
  -> application service
  -> existing domain modules

Human / shell / CI
  -> rmw CLI adapter
  -> same ActionRunner

Optional local MCP adapter
  -> typed Tool/Action contract
  -> same Harness and ActionRunner
```

Rules:

1. Workflow composes Actions, not Skills.
2. Skills explain when and why to invoke actions; they do not implement algorithms.
3. Application services produce domain outcomes; they do not approve their own work.
4. ActionRunner owns common state transitions and artifact registration.
5. CLI preserves existing command names and exit-code behavior during migration.

## 4. Iteration Schedule

| Iteration | Priority | Deliverable | Dependency | Estimate | Release gate |
| --- | --- | --- | --- | ---: | --- |
| I1 | P1 | Stage-skill contract, trigger matrix, static validator | Current tool registry | 1–2 days | Contract tests pass |
| I2 | P1 | Training and feature-selection skills | I1 | 2–3 days | Skill evals show correct routing and stop rules |
| I3 | P1 | Evaluation/report skills and root router | I2 | 2–3 days | Full-chain and single-stage prompts route correctly |
| I4 | P2 | Typed action contracts and ActionRunner foundation | P0 complete, P1 contract frozen | 2–3 days | Runner/unit compatibility tests pass |
| I5 | P2 | Evaluation and compare migration | I4 | 2–3 days | Legacy/new semantic parity |
| I6 | P2 | Report migration | I5 | 2–3 days | Report artifact/audit parity |
| I7 | P2 | Training migration including Advisor pause/resume | I6 | 4–6 days | Advisor and training evidence parity |
| I8 | P2 | Feature-selection migration including SQL phases | I7, P0 SQL gate | 6–10 days | Local and approved-remote paths pass |
| I9 | P2 | Executor cutover, CLI thinning, Host capabilities schema | I5–I8 | 3–5 days | No stage depends on `cli.main(argv)` internally |
| I10 | P2 | Agent eval, synthetic dogfood, new Fujie version dogfood | I9 | 3–5 days plus model runtime | Strict audit complete |

Expected engineering effort: roughly 25–42 person-days, excluding real DP approval wait and full model runtime. Do not compress this by merging I7 and I8; they have different failure and approval semantics.

## 5. Release Gates

### Gate P1-A: Skill contract

- [ ] Each skill has a unique trigger boundary.
- [ ] Single-stage requests do not load the full-chain procedure by default.
- [ ] Full-chain requests still use `risk-model-workbench`.
- [ ] No skill runs project scripts directly.
- [ ] No skill adds `--sql-approved` without a recorded approval path.
- [ ] All skills read version state and manifest before acting.

### Gate P1-B: Skill quality

- [ ] Each skill has at least three realistic positive eval prompts.
- [ ] The shared trigger set includes near-miss negative prompts.
- [ ] With-skill outputs outperform the root-only/baseline workflow on boundary and completeness assertions.
- [ ] Host-Agent output contains executable `rmw` commands or a structured advisor response, not reimplemented modeling code.

### Gate P2-A: Per-action migration

- [ ] CLI command remains backward compatible.
- [ ] CLI and direct ActionRunner calls produce semantically equivalent `ActionResult` values.
- [ ] State, trace, and manifest are written once, by the Harness layer.
- [ ] Failure codes and exit codes remain compatible.
- [ ] Existing stage-specific and workflow tests pass.

### Gate P2-B: Runtime cutover

- [ ] Agent Executor no longer imports/calls `cli.main` for migrated stage actions.
- [ ] Tool Registry derives permissions and parameter validation.
- [ ] `rmw agent tools --json` contains typed parameter and result schemas.
- [ ] No skill or adapter bypasses policy or approval.

### Gate P2-C: Product proof

- [ ] Synthetic local Agent workflow completes.
- [ ] One new Fujie version completes through `rmw agent start/run/resume`.
- [ ] At least one real `advisor_required -> response -> consume -> resume` occurs.
- [ ] A controlled SQL canary covers prepare, approval, execute, and drift invalidation.
- [ ] Final `rmw version audit --strict` verdict is `complete`.

## 6. Quality Metrics

| Metric | Target |
| --- | ---: |
| Stage-skill trigger precision on agreed eval set | >= 90% |
| Full-chain routing accuracy | 100% |
| Direct script execution from skills | 0 |
| CLI compatibility regressions | 0 |
| Duplicate state/manifest writes per action attempt | 0 |
| Unregistered action execution | 0 |
| Approval/Advisor bypass | 0 |
| Semantic parity for migrated actions | 100% on fixtures |
| Agent dogfood strict audit | complete |

## 7. Test Strategy

Run after every task-sized commit:

```bash
pytest <changed-test-file> -q
```

Run at every iteration gate:

```bash
pytest tests -q
rmw workflow validate --workflow workflows/full_modeling.yml
rmw agent tools --json
rmw project validate --project projects/2026-05-fujie-gcard-v1
```

Run strict audit only against a new, non-overwritten dogfood version:

```bash
rmw version audit \
  --project projects/2026-05-fujie-gcard-v1 \
  --version-id <new_version_id> \
  --strict
```

## 8. Commit And Rollback Policy

- One logical commit per task or migrated action.
- Never combine Skill content changes with ActionRunner behavior changes.
- Keep legacy CLI behavior until the corresponding direct-runner parity gate passes.
- Roll back a migrated action by removing its handler registration; do not revert unrelated actions.
- Preserve old command templates and `jm` aliases until compatibility tests prove they are unused and a separate deprecation decision is approved.
- Do not stage ignored feather/sample files or ephemeral lock files.

## 9. Detailed Plans

- P1: `docs/superpowers/plans/2026-07-10-rmw-p1-stage-skills-implementation.md`
- P2: `docs/superpowers/plans/2026-07-10-rmw-p2-action-runner-implementation.md`

## 10. Final Handoff

After both detailed plans pass review:

1. Implement P1 first using `@subagent-driven-development`.
2. Review P1 skill eval output with the user before starting P2.
3. Reconfirm all P0 gates.
4. Implement P2 one migrated action at a time with author and verifier separated.
5. Stop at each release gate if evidence is incomplete.
