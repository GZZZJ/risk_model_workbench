# RMW Embedded Agent Implementation Plan

## Implementation Status (2026-08-23)

- M0 Python/runtime baseline: completed.
- M1 model gateway and reasoning contracts: completed.
- M2 LangGraph runtime, SQLite checkpoint, and HITL: completed.
- M3 embedded tuning/diagnosis/data-gap/product Advisor: completed.
- M4 deterministic embedded eval and existing Harness hardening: completed.
- M5 CLI, architecture/product docs, compatibility, and local verification:
  completed.

Local verification uses deterministic Fake Model trajectories. A real provider
call and a new real-data end-to-end modeling version remain deployment
validation items because no production model credential or new version/data
execution approval was supplied during implementation. These do not reintroduce
a Codex/Claude dependency; `rmw agent model status` reports the missing provider
configuration and reasoning tasks fail closed until it is configured.

## Objective

Turn RMW into a complete standalone modeling Agent that does not require
Codex, Claude Code, their command-line tools, or their Skills at runtime.

The existing deterministic Harness and modeling services remain in place. The
change adds an embedded reasoning layer and removes Host-Agent input as a
required execution path.

## Scope

### In scope

- Python baseline and reproducible local Python 3.12 environment.
- Provider-neutral model gateway and structured reasoning contracts.
- LangGraph orchestration with persistence and human interrupts.
- Embedded answers for tuning, failure diagnosis, data gaps, and product
  decisions.
- Bounded routing/reflection and action proposal through ActionRunner.
- Local configuration, CLI commands, traces, evals, and migration docs.

### Out of scope for the first release

- Autonomous SQL/DP approval.
- Runtime prompt/code/rule self-modification.
- General-purpose shell or arbitrary Python execution by the LLM.
- Multi-agent/A2A deployment.
- Mandatory cloud observability or vector databases.

## Milestones

### M0: Baseline and Python runtime

- Set project minimum Python to 3.11; use Python 3.12.2 for development.
- Create an isolated `.venv` from `/opt/anaconda3/bin/python`.
- Add bounded Agent dependencies and verify imports from this repository.
- Capture current Harness tests and eval metrics.

Acceptance:

- `python` resolves to the repository `.venv` during documented commands.
- package import path points to this repository.
- focused Harness tests pass under Python 3.12.

### M1: Model gateway and reasoning contracts

- Add `ModelGateway` protocol.
- Add LangChain production adapter and deterministic fake adapter.
- Define Pydantic request, decision, usage, and error contracts.
- Add provider/model/timeout/retry/token configuration without storing secrets.
- Record model metadata and validated structured output in local audit evidence.

Acceptance:

- gateway is provider-neutral and injectable;
- malformed output fails closed;
- tests require no network or model credentials;
- raw row-level data is rejected from model context.

### M2: LangGraph embedded runtime

- Build nodes for loading authoritative state, routing blockers, invoking the
  embedded Advisor, accepting/consuming validated responses, requesting human
  approval, executing safe actions, evaluating results, and stopping.
- Persist orchestration checkpoints in a workspace-local SQLite database.
- Bound graph steps, model calls, retries, and reflection cycles.
- Reconstruct graph state from RMW state when checkpoints are absent or stale.

Acceptance:

- one deterministic fake-model happy path reaches the same terminal state as
  the existing Harness;
- an Advisor blocker is resolved without Host-Agent files supplied by a human;
- SQL approval still interrupts;
- resume does not duplicate an ActionRunner execution.

### M3: Embedded Advisor capabilities

- Tuning plan generation with 3-5 bounded candidates.
- Failure diagnosis with retry/stop/escalate decisions.
- Data-gap and product-decision responses.
- Optional explicit user confirmation based on response risk.
- Remove Codex/Claude wording and runtime instructions from current product
  documentation while keeping historical ADRs and compatibility readers.

Acceptance:

- all four Advisor request types can be answered internally;
- current Advisor identity, validation, and one-time consumption tests remain
  valid;
- no current runtime state has `Host-Agent` as its only next safe action.

### M4: Agent evaluation and hardening

- Add golden trajectory tests and schema pass/fail datasets.
- Add prompt-injection, path traversal, stale-context, budget exhaustion,
  provider outage, crash recovery, and approval-bypass cases.
- Fix the current `crash_after_manifest` recovery expectation mismatch.
- Add decision quality checks for tuning and diagnosis, with deterministic
  rule checks as the primary judge.

Acceptance:

- policy escape count: 0;
- duplicate external execution count: 0;
- SQL approval bypass count: 0;
- standard recovery scenarios: 100%;
- happy-path strict audit completion: 100%;
- structured output valid or safe-stop: 100%;
- raw-data leakage cases: 0.

### M5: Product interface and release closure

- Add standalone Agent CLI configuration/status/run commands.
- Document provider setup, local-model alternatives, operation, recovery, and
  security boundaries.
- Run focused tests, full tests, CLI smoke tests, workflow validation, and Agent
  eval suite.
- Review final diff without changing unrelated project artifacts.

Acceptance:

- a user can start and resume a full Agent workflow without Codex/Claude Code;
- no unverified success claims;
- unresolved model-provider or real-data validation requirements are explicit.

## Delivery Rule

Milestones are implemented in order. A later milestone may start only after
the prior milestone has test evidence or a documented, user-visible blocker.
The Agent stays fail-closed throughout migration; temporary compatibility
paths may read existing Advisor artifacts but may not make Host-Agent input a
required dependency.
