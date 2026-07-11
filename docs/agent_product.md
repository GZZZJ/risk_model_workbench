# RMW Agent Product

`RMW Agent` is the local semi-autonomous Agent runtime for
`risk_model_workbench`. It turns a model request into a version-scoped,
auditable execution loop while preserving explicit approval for high-risk
actions.

## User Flow

1. Create or receive a Markdown model request.
2. Start an Agent-managed version:

   ```bash
   rmw agent start \
     --project <project> \
     --request <request.md> \
     --version-id <version_id> \
     --workflow full_modeling
   ```

3. Run or resume the Agent:

   ```bash
   rmw agent run --project <project> --version-id <version_id>
   rmw agent resume --project <project> --version-id <version_id>
   ```

4. Inspect state:

   ```bash
   rmw agent status --project <project> --version-id <version_id>
   rmw version audit --project <project> --version-id <version_id> --strict
   ```

5. When a SQL/DP action pauses for approval, review the generated SQL and
   approval record, then approve explicitly:

   ```bash
   rmw agent approve \
     --project <project> \
     --version-id <version_id> \
     --approval-id <approval_id> \
     --approved-by <name> \
     --note "reviewed SQL evidence"
   ```

## State Machine

Agent states:

- `draft`: Agent plan and state are initialized but not executing.
- `running`: a safe task is executing or pending execution.
- `waiting_for_approval`: a high-risk SQL/DP action needs explicit approval.
- `waiting_for_advisor`: Host-Agent input is required, such as a tuning plan.
- `waiting_for_user`: an accepted Advisor decision needs explicit human
  confirmation.
- `reconciliation_required`: an external operation outcome cannot be proved
  locally and must not be retried automatically.
- `blocked`: no task can safely run.
- `failed`: a task failed with a non-recoverable failure.
- `stopped`: an explicit Host-Agent or human decision stopped execution.
- `done`: all tasks completed and strict evidence can close.
- `done_with_gaps`: execution finished with scaffold/imported/incomplete
  evidence or strict audit did not reach `complete`.

Task states:

- `pending`
- `running`
- `review_ready`
- `paused`
- `done`
- `scaffold`
- `failed`
- `skipped`
- `stopped`
- `reconciliation_required`

`review_ready`, `scaffold`, and every waiting or reconciliation state are not
strict success. Only real `done` dependencies may unlock real downstream work.

## Gates

Default policy:

- `read_only`: automatic.
- `writes_run`: automatic for version-local writes.
- `dp_sql_pull`: requires approval.
- `external_data`: blocked by default.
- `--force`: blocked.
- `--skip-split-check`: blocked by default.

The Agent writes approval requests to `audit/approvals.yml`. Approval is bound
to the exact command hash, so changing the command requires a new approval.

## Host-Agent Boundary

The local runtime does not call an LLM API. When it needs judgment, it writes an
advisor request under `audit/advisor_requests/` and pauses. Codex or Claude Code
can inspect the context, write the required plan or fix inputs, then the user
can resume the Agent.

Advisor exchange is a local file protocol:

1. The Agent writes `audit/advisor_requests/<request_id>.json`.
2. The Host-Agent or human expert writes a matching response JSON.
3. The user accepts the response:

   ```bash
   rmw agent advisor accept \
     --project <project> \
     --version-id <version_id> \
     --response <response.json>
   ```

4. `rmw agent resume` verifies that the response matches the request before
   execution continues.

Useful inspection commands:

```bash
rmw agent advisor list --project <project> --version-id <version_id>
rmw agent advisor show --project <project> --version-id <version_id> --request-id <request_id>
```

Advisor request and response shapes are documented in
`schemas/advisor_request.schema.yml` and `schemas/advisor_response.schema.yml`.
The first supported Advisor types are:

- `tuning_plan_required`
- `failure_diagnosis_required`
- `data_gap_decision_required`
- `product_decision_required`

## Evidence Files

An Agent-managed version includes:

- `agent_plan.yml`: bound version-scoped plan.
- `audit/agent_state.yml`: Agent state source of truth.
- `audit/agent_trace.jsonl`: append-only observation, decision, action, and
  result log.
- `audit/approvals.yml`: approval ledger for high-risk actions.
- `audit/advisor_requests/*.json`: structured Host-Agent handoff requests.
- `audit/advisor_responses/*.json`: accepted Advisor responses.

Completion still depends on `version_state.yml`, `audit/artifact_manifest.json`,
workflow contracts, and `rmw version audit --strict`.

The normative Host-Agent/Harness boundary, state-transition table, version
compatibility policy, and completion invariants are defined in
[ADR 0003](adr/0003-host-agent-harness-contract.md).

## Harness Evaluation

The product gate is a test-backed scenario suite declared in
`evals/agent_harness/scenarios.yml`. Each scenario binds its expected state
transitions, blockers, artifacts, audit verdict, and forbidden runner calls to
a real pytest fixture. Passing is necessary but not sufficient: the fixture
must emit evidence derived by common predicates from actual state snapshots,
audit verdicts, workspace artifacts, trace rows, recovery reports, validation
errors, and recorded runner calls. Rates only credit the intersection of
expected and observed evidence. Free-string observations, missing evidence, an
invalid audit verdict, or an observed forbidden runner call fail the scenario.

The callable API is `risk_model_workbench.agent.eval.evaluate_harness_suite`.
The CLI exposes the same suite as:

```bash
rmw agent eval --suite harness --json
```

The JSON result reports exact numerators and denominators:

- `transition_conformance_rate`: conforming transitions / asserted transitions.
- `recovery_success_rate`: successfully reconciled crash scenarios / crash scenarios.
- `trace_coverage_rate`: expected correlated events present / expected correlated events.
- `strict_audit_complete_rate`: strict-complete terminal scenarios / terminal happy-path scenarios.
- `policy_escape_count` and `duplicate_external_execution_count`: count plus the
  scenario IDs responsible for the count.

Count metrics fail closed: a failed guard fixture is reported as a potential
escape or forbidden execution until the scenario passes. Host-Agent answer
quality is intentionally excluded. Runtime tests use golden Advisor responses;
recommendation quality belongs to the domain Skill evaluations.

## Optional MCP Action Adapter

Codex and Claude may expose the same local typed action protocol through
`risk_model_workbench.adapters.mcp.MCPActionAdapter`. The adapter is not an
execution engine or server: it publishes MCP-safe capability schemas, validates
parameters, delegates once to an injected policy-gated ActionRunner, and
returns the structured ActionResult.

The human interface remains `rmw`. MCP owns no Agent state, approval, Advisor,
retry, recovery, shell, or artifact behavior. Those controls remain in the
Harness and shared application layer. The decision and boundaries are recorded
in [ADR 0004](adr/0004-mcp-adapter-decision.md).
