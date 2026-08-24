# RMW Agent Product

`RMW Agent` is the standalone embedded Agent runtime for
`risk_model_workbench`. It turns a model request into a version-scoped,
auditable execution loop while preserving explicit approval for high-risk
actions. The implementation uses LangGraph/LangChain. Optional provider
adapters can use OpenAI or Anthropic Claude without introducing another Agent
runtime.

## User Flow

1. Create a Python 3.12 environment and install the Agent provider adapter:

   ```bash
   /opt/anaconda3/bin/python -m venv .venv
   .venv/bin/python -m pip install -e ".[modeling,agent-openai]"
   ```

2. Configure a model without writing its credential into project files:

   ```bash
   export RMW_AGENT_PROVIDER=openai
   export RMW_AGENT_MODEL=<model-name>
   export OPENAI_API_KEY=<secret>
   .venv/bin/rmw agent model status
   ```

   `RMW_AGENT_BASE_URL` may point the OpenAI adapter at a compatible local or
   private endpoint. LangSmith tracing is not enabled by RMW and should remain
   off unless the data-governance review explicitly permits it.

   To use Claude through the same LangGraph runtime:

   ```bash
   .venv/bin/python -m pip install -e ".[modeling,agent-anthropic]"
   export RMW_AGENT_PROVIDER=anthropic
   export RMW_AGENT_MODEL=<claude-model-name>
   export ANTHROPIC_API_KEY=<secret>
   .venv/bin/rmw agent model status
   ```

3. Create or receive a Markdown model request.
4. Start an Agent-managed version:

   ```bash
   rmw agent start \
     --project <project> \
     --request <request.md> \
     --version-id <version_id> \
     --workflow full_modeling
   ```

5. Run or resume the Agent:

   ```bash
   rmw agent run --project <project> --version-id <version_id>
   rmw agent resume --project <project> --version-id <version_id>
   ```

6. Inspect state:

   ```bash
   rmw agent status --project <project> --version-id <version_id>
   rmw version audit --project <project> --version-id <version_id> --strict
   ```

7. When a SQL/DP action pauses for approval, review the generated SQL and
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
- `waiting_for_advisor`: the embedded reasoning layer must produce and validate
  a structured decision, such as a tuning plan.
- `waiting_for_user`: an accepted Advisor decision needs explicit human
  confirmation.
- `reconciliation_required`: an external operation outcome cannot be proved
  locally and must not be retried automatically.
- `blocked`: no task can safely run.
- `failed`: a task failed with a non-recoverable failure.
- `stopped`: an explicit embedded-Advisor or human decision stopped execution.
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

## Embedded Reasoning Boundary

When deterministic execution needs judgment, the Harness writes an immutable
Advisor request and bounded Context Pack. LangGraph invokes the configured model
provider through structured output, writes model invocation evidence, submits
the answer through the existing Advisor validator, and consumes it exactly once
before execution continues. Provider adapters cannot call ActionRunner or
receive RMW tools directly.

The model may recommend `continue`, `retry`, `stop`, or explicit user
confirmation. It cannot call ActionRunner directly, bypass approval, mutate
Agent state, or write the artifact manifest. Missing credentials, provider
failure, malformed output, stale context, and exhausted retry budgets all fail
closed locally; they do not hand control to an external coding Agent.

The Advisor file commands remain available for audit inspection and reading
legacy workspaces. Manual `advisor accept` is a compatibility and expert-review
path, not a required runtime step.

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
- `audit/advisor_requests/*.json`: durable structured reasoning requests.
- `audit/advisor_responses/*.json`: accepted Advisor responses.
- `audit/model_invocations/*.json`: provider/model, prompt and response hashes,
  token usage, decision identity, and fail-closed evidence.
- `audit/agent_graph.sqlite`: local LangGraph orchestration checkpoints; never
  the business source of truth.

Completion still depends on `version_state.yml`, `audit/artifact_manifest.json`,
workflow contracts, and `rmw version audit --strict`.

The embedded runtime decision and state-ownership boundary is defined in
[ADR 0005](adr/0005-embedded-langgraph-agent-runtime.md). The original external
Host-Agent contract remains in [ADR 0003](adr/0003-host-agent-harness-contract.md)
as compatibility history.

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
escape or forbidden execution until the scenario passes. Embedded model-answer
quality is evaluated separately with golden structured answers, parameter
bounds, safe-stop cases, and trajectory tests; an LLM judge is supplementary,
not the source of truth.

## Optional MCP Action Adapter

External clients may expose the same local typed action protocol through
`risk_model_workbench.adapters.mcp.MCPActionAdapter`. MCP is optional and is not
required by the standalone runtime. The adapter is not an
execution engine or server: it publishes MCP-safe capability schemas, validates
parameters, delegates once to an injected policy-gated ActionRunner, and
returns the structured ActionResult.

The human interface remains `rmw`. MCP owns no Agent state, approval, Advisor,
retry, recovery, shell, or artifact behavior. Those controls remain in the
Harness and shared application layer. The decision and boundaries are recorded
in [ADR 0004](adr/0004-mcp-adapter-decision.md).
