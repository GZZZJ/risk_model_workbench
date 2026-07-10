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
- `blocked`: no task can safely run.
- `failed`: a task failed with a non-recoverable failure.
- `done`: all tasks completed and strict evidence can close.
- `done_with_gaps`: execution finished with scaffold/imported/incomplete
  evidence or strict audit did not reach `complete`.

Task states:

- `pending`
- `running`
- `paused`
- `done`
- `scaffold`
- `failed`
- `skipped`

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
