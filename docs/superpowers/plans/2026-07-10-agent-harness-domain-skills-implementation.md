# RMW Agent Harness and Domain Skills Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build a trustworthy local RMW Agent harness with an external Host-Agent intelligence layer, add independently maintainable domain skills, and make the rmw CLI a thin compatibility adapter over a shared ActionRunner.

**Architecture:** Codex or Claude remains the Host-Agent for judgment. RMW owns typed action contracts, deterministic state transitions, policy gates, approval, execution, recovery, trace, and artifact evidence. Domain skills contain orchestration knowledge only; CLI, Agent Runtime, and optional MCP adapters share the same ActionRunner and Python domain services.

**Tech Stack:** Python 3, argparse, dataclasses, PyYAML, JSON/YAML schemas, pytest, local filesystem locks and atomic replace, existing rmw workflows and version workspaces.

---

## 0. Approved Decisions and Non-Goals

### Product boundary

- Host-Agent: intent interpretation, ambiguous diagnosis, tuning proposals, and human-facing recommendations.
- RMW Harness: request validation, typed plans, tool policy, state transitions, approval, deterministic execution, recovery, trace, and strict audit.
- Human: SQL/DP approval and product decisions that explicitly require confirmation.
- Domain services: deterministic feature selection, training, evaluation, comparison, and reporting.

### Non-goals

- Do not embed a general LLM provider in RMW.
- Do not remove or break existing rmw CLI commands.
- Do not introduce a web daemon, multi-tenant service, RBAC, or distributed queue.
- Do not allow a Skill or Host-Agent to write version_state.yml or artifact_manifest.json directly.
- Do not allow arbitrary shell generation or a free-form ReAct loop inside RMW.
- Do not auto-retry DP writes with an unknown remote outcome.
- Do not retrofit Agent artifacts into the completed v8 workspace. Create a new version for dogfood.
- Do not commit scored feather datasets, sample caches, secrets, or local credentials.

### Release train

| Priority | Release meaning | Estimated effort | Exit condition |
| --- | --- | ---: | --- |
| P0 | Trustworthy control loop | 8-14 person-days | Synthetic Agent E2E reaches strict audit complete; policy escape metrics are zero |
| P1 | Stable daily local use | 10-16 person-days | Crash recovery works; Host context is deterministic; domain Skills pass evals; real local-feather dogfood completes |
| P2 | Maintainable execution architecture | 15-25 person-days | Executor no longer calls cli.main; CLI/Agent parity passes; post-migration dogfood completes; Agent eval suite and optional adapter gate are established |

Estimates exclude LightGBM runtime, DP queue time, and human approval wait time.

### Dependency graph

    P0.0 Contract freeze
      -> P0.1 Typed plan and tool integrity
      -> P0.2 Attempt-scoped ActionResult and minimum atomic store
      -> { P0.3 SQL two-phase gate || P0.4 Advisor reducer }
      -> P0.5 Audit closure and synthetic E2E
      -> P1.1 Atomic workspace store
      -> P1.2 Recovery and reconciliation
      -> { P1.3 Host context pack || P1.4 Domain Skills }
      -> P1.5 Local-only artifact retention
      -> P1.6 Real local-feather dogfood
      -> P2.1 ActionRunner foundation
      -> P2.2 Evaluation and report migration
      -> P2.3 Training and feature-selection migration
      -> P2.4 Post-migration dogfood
      -> P2.5 Agent eval productization
      -> P2.6 Optional MCP decision gate

## 1. Planned File Structure

### New contract and harness files

- Create docs/adr/0003-host-agent-harness-contract.md
  - Records the external Host-Agent boundary, state invariants, compatibility policy, and fail-closed rules.
- Create schemas/agent_plan.schema.yml
  - Defines typed invocation tasks and dependency constraints for Agent Plan v2.
- Create schemas/action_result.schema.yml
  - Defines attempt-scoped semantic results.
- Create schemas/approval_record.schema.yml
  - Defines SQL/config/invocation-bound approval subjects and consumption state.
- Create schemas/advisor_consumption.schema.yml
  - Defines deterministic response-consumption receipts.
- Create schemas/artifact_retention.schema.yml
  - Defines repository, local-only, external, and legacy artifact availability semantics.
- Create src/risk_model_workbench/harness/invocation.py
  - Canonical ActionInvocation model, hashing, parameter validation, and argv rendering.
- Create src/risk_model_workbench/agent/transitions.py
  - Production state-transition validator and reducer shared by all Agent paths.
- Create src/risk_model_workbench/agent/advisor_reducer.py
  - Applies typed Advisor decisions to Agent state.
- Create src/risk_model_workbench/agent/workspace_store.py
  - P0 provides atomic receipts and a single-runner lock; P1 adds revisions, compare-and-swap, and cross-file recovery.
- Create src/risk_model_workbench/agent/recovery.py
  - Reconciles interrupted attempts and unknown external outcomes.
- Create src/risk_model_workbench/agent/context_pack.py
  - Builds bounded, hashed, redacted context for external Host-Agents.

### New application layer files

- Create src/risk_model_workbench/application/__init__.py
- Create src/risk_model_workbench/application/context.py
  - VersionContext and stable workspace paths.
- Create src/risk_model_workbench/application/action_runner.py
  - Shared ActionRunner used by CLI and Agent Runtime.
- Create src/risk_model_workbench/application/handlers/__init__.py
- Create src/risk_model_workbench/application/handlers/evaluate.py
- Create src/risk_model_workbench/application/handlers/report.py
- Create src/risk_model_workbench/application/handlers/train.py
- Create src/risk_model_workbench/application/handlers/feature_selection.py

### New project-local Skills

- Create .agents/skills/rmw-model-training/SKILL.md
- Create .agents/skills/rmw-model-training/evals/evals.json
- Create .agents/skills/rmw-feature-selection/SKILL.md
- Create .agents/skills/rmw-feature-selection/evals/evals.json
- Create .agents/skills/rmw-model-evaluation/SKILL.md
- Create .agents/skills/rmw-model-evaluation/evals/evals.json
- Create .agents/skills/rmw-report-generation/SKILL.md
- Create .agents/skills/rmw-report-generation/evals/evals.json
- Modify .agents/skills/risk-model-workbench/SKILL.md
  - Keeps full-chain routing and delegates explicit single-stage requests to the relevant domain skill.

### New tests and evals

- Create tests/test_agent_contract.py
- Create tests/test_agent_plan_integrity.py
- Create tests/test_agent_action_result.py
- Create tests/test_agent_sql_approval_flow.py
- Create tests/test_agent_advisor_consumption.py
- Create tests/test_agent_workspace_store.py
- Create tests/test_agent_recovery.py
- Create tests/test_agent_context_pack.py
- Create tests/test_context_snapshot.py
- Create tests/test_clean_clone_audit.py
- Create tests/test_action_runner.py
- Create tests/test_cli_action_parity.py
- Create tests/agent_scenarios/test_local_happy_path.py
- Create tests/agent_scenarios/test_policy_fail_closed.py
- Create tests/agent_scenarios/test_crash_recovery.py
- Create evals/agent_harness/scenarios.yml
- Create src/risk_model_workbench/agent/eval.py

Project-owned Skills stay under .agents/skills so their source and evals are versioned with the workbench. Install or synchronize them to /Users/guzijun/.Codex/skills only after their project-local evals are approved.

## Chunk 1: P0 - Trustworthy Control Loop

### Task 1: Freeze the Host-Agent Harness Contract

**Priority:** P0.0

**Files:**

- Create: docs/adr/0003-host-agent-harness-contract.md
- Create: tests/test_agent_contract.py
- Modify: docs/adr/0002-agent-runtime.md
- Modify: docs/agent_product.md

- [ ] **Step 1: Write failing state-invariant tests**

Add tests for these rules:

    def test_unapproved_dp_task_is_never_runnable():
        ...

    def test_unconsumed_advisor_response_does_not_resume_task():
        ...

    def test_review_ready_is_not_a_terminal_success():
        ...

    def test_scaffold_does_not_unlock_real_downstream_work():
        ...

Run:

    pytest tests/test_agent_contract.py -q

Expected: FAIL because the state-transition validator does not exist.

- [ ] **Step 2: Write ADR 0003**

Document:

- Plan v2 and Agent State v2 are used only for newly initialized Agent versions.
- Existing v1 workspaces remain readable.
- Policy and tool metadata are derived from the registry, never trusted from plan copies.
- Every action execution has an attempt_id and one semantic ActionResult.
- SQL review-ready and Advisor waiting states are non-terminal.
- Every blocker has one explicit next safe action.
- done requires no unconsumed approval/Advisor blocker and strict version audit complete.

- [ ] **Step 3: Add a transition table to the ADR**

Minimum Agent states:

    draft
    running
    waiting_for_approval
    waiting_for_advisor
    waiting_for_user
    reconciliation_required
    blocked
    failed
    stopped
    done
    done_with_gaps

Minimum task states:

    pending
    running
    review_ready
    paused
    done
    scaffold
    failed
    skipped
    stopped
    reconciliation_required

- [ ] **Step 4: Add a contract conformance helper**

Initially keep the helper in tests so later tasks can implement against it. The helper must reject illegal transitions and report a stable failure code.

- [ ] **Step 5: Run contract tests**

Run:

    pytest tests/test_agent_contract.py -q

Expected: tests that only check the documented matrix PASS; implementation-dependent tests remain xfail with a linked task reference.

- [ ] **Step 6: Commit**

    git add docs/adr/0003-host-agent-harness-contract.md docs/adr/0002-agent-runtime.md docs/agent_product.md tests/test_agent_contract.py
    git commit -m "docs: define host-agent harness contract"

### Task 2: Introduce Typed ActionInvocation and Plan Integrity

**Priority:** P0.1

**Files:**

- Create: schemas/agent_plan.schema.yml
- Create: src/risk_model_workbench/harness/invocation.py
- Create: src/risk_model_workbench/agent/transitions.py
- Create: tests/test_agent_plan_integrity.py
- Modify: src/risk_model_workbench/harness/tools.py
- Modify: src/risk_model_workbench/agent/plan.py
- Modify: src/risk_model_workbench/agent/policy.py
- Modify: src/risk_model_workbench/agent/state.py
- Modify: src/risk_model_workbench/cli.py
- Modify: tests/test_agent_plan.py
- Modify: tests/test_agent_policy.py
- Modify: tests/test_agent_state_trace.py
- Modify: tests/test_rename_compatibility.py

- [ ] **Step 1: Write failing integrity tests**

Cover:

- Duplicate task IDs.
- Missing dependencies.
- Cyclic DAG.
- Unknown tool.
- Project/version mismatch.
- Tampered permission.
- Tampered argv.
- Blocked flags.
- Registry digest drift.
- Mutation of the caller-owned params object after invocation construction.
- Attempted mutation of the invocation's internal parameter snapshot.
- Registry digest stability across process restarts and callable object identity changes.

The runner call count must remain zero for every invalid case.

Run:

    pytest tests/test_agent_plan_integrity.py -q

Expected: FAIL because Agent Plan v1 trusts copied command and permission fields.

- [ ] **Step 2: Add ActionInvocation**

Implement the minimal public model:

    @dataclass(frozen=True)
    class ActionInvocation:
        tool_name: str
        params: Mapping[str, CanonicalValue]
        project: str
        version_id: str

        def canonical_payload(self) -> dict[str, object]:
            ...

        def digest(self) -> str:
            ...

At construction time, deep-copy and recursively freeze the caller-provided JSON-like params into one canonical immutable snapshot: mappings become immutable sorted mappings, sequences become tuples, and only JSON scalar values are accepted. Reject unsupported objects. `canonical_payload`, `digest`, policy evaluation, and `render_argv` must all read that same frozen snapshot; none may revisit the caller-owned object.

The digest must use sorted canonical JSON and SHA256. Tests must prove that mutating the original params after construction does not change the digest or rendered argv, and that direct mutation through `invocation.params` is rejected.

- [ ] **Step 3: Extend ToolSpec**

Add:

    params_schema: dict[str, object]
    execution_semantics: read_only | idempotent_write | non_idempotent_write | external_unknown
    approval_type: str
    render_argv: Callable[[ActionInvocation], list[str]]

Keep command_template as display-only compatibility metadata.

- [ ] **Step 4: Bind Agent Plan v2**

Each task must store:

    invocation:
      tool_name: evaluate
      params: {}
      project: projects/example
      version_id: example_v1
    invocation_hash: ...

Do not store permission as an authoritative input. It may appear under derived_metadata for inspection.

- [ ] **Step 5: Validate and hash the plan**

Add:

    validate_agent_plan(plan, registry) -> list[str]
    canonical_plan_hash(plan) -> str
    registry_digest(registry) -> str

Persist plan_hash and registry_digest in Agent State v2.

`registry_digest` must serialize only stable ToolSpec metadata: tool name, parameter schema, execution semantics, approval type, blocked flags, and compatibility command metadata. Never serialize a Callable, its `repr`, object identity, module load address, or other process-local value.

- [ ] **Step 6: Make policy derive from ToolSpec**

Change evaluate_task_policy to accept ActionInvocation plus the registry. Never accept task.permission or task.requires_approval as authority.

- [ ] **Step 7: Implement the production transition reducer**

Add:

    validate_transition(current_state, event, target_state) -> list[str]
    apply_transition(state, event, payload) -> dict[str, object]

All state writes in executor, approval, Advisor, and recovery paths must go through this reducer. Convert the temporary xfail contract cases from Task 1 to passing tests.

- [ ] **Step 8: Add v1 compatibility and drift handling**

- Existing v1 plans, v1 Agent State, and legacy manifests remain readable and status-inspectable.
- New agent start emits v2 only.
- Running a v1 plan requires an explicit compatibility path and must never gain permissions beyond its registered tool.
- Registry drift fails closed and exposes an inspectable diff.
- Add rmw agent plan rebind with a dry-run preview. Rebind is prohibited while a task is running or an approval/Advisor response is pending.
- Keep jm alias parity for status, audit, and legacy-compatible stage commands.

- [ ] **Step 9: Run tests**

    pytest tests/test_agent_plan_integrity.py tests/test_agent_plan.py tests/test_agent_policy.py tests/test_agent_state_trace.py tests/test_harness_registry.py tests/test_rename_compatibility.py -q

Expected: PASS; invalid plans fail before a runner is invoked.

- [ ] **Step 10: Commit**

    git add schemas/agent_plan.schema.yml src/risk_model_workbench/harness/invocation.py src/risk_model_workbench/agent/transitions.py src/risk_model_workbench/harness/tools.py src/risk_model_workbench/agent/plan.py src/risk_model_workbench/agent/policy.py src/risk_model_workbench/agent/state.py src/risk_model_workbench/cli.py tests/test_agent_plan_integrity.py tests/test_agent_plan.py tests/test_agent_policy.py tests/test_agent_state_trace.py tests/test_rename_compatibility.py
    git commit -m "feat: validate typed agent invocations"

### Task 3: Make ActionResult Attempt-Scoped and Add the Minimum Atomic Store

**Priority:** P0.2

**Files:**

- Create: schemas/action_result.schema.yml
- Create: src/risk_model_workbench/agent/workspace_store.py
- Create: tests/test_agent_action_result.py
- Create: tests/test_agent_workspace_store.py
- Modify: src/risk_model_workbench/harness/runtime.py
- Modify: src/risk_model_workbench/harness/errors.py
- Modify: src/risk_model_workbench/agent/executor.py
- Modify: src/risk_model_workbench/agent/state.py
- Modify: src/risk_model_workbench/cli.py
- Modify: tests/test_harness_runtime.py
- Modify: tests/test_agent_executor.py

- [ ] **Step 1: Write stale-result and exit-code precedence tests**

Cover:

- Exit code zero plus next_required_action=approval must pause.
- Exit code non-zero plus a valid semantic scaffold result must not be read as unknown.
- An old stage last_result cannot satisfy a new attempt.
- Missing result receipt is a failure, not success.
- Two concurrent runners cannot execute the same attempt.
- Two consumers cannot consume the same approval or Advisor response.

Run:

    pytest tests/test_agent_action_result.py -q

Expected: FAIL because Executor currently checks process exit code before semantic stage state.

- [ ] **Step 2: Extend ActionResult**

Minimum model:

    @dataclass(frozen=True)
    class ActionResult:
        schema_version: int
        attempt_id: str
        task_id: str
        action_id: str
        invocation_hash: str
        project: str
        version_id: str
        status: str
        failure_code: str = ""
        next_required_action: str = "none"
        retryable: bool = False
        scaffold: bool = False
        artifacts: list[dict[str, object]] = field(default_factory=list)
        message: str = ""
        created_at: str = ""

- [ ] **Step 3: Write result receipts**

Write one immutable JSON receipt per attempt under:

    audit/action_results/<attempt_id>.json

Register the receipt as audit evidence.
Validate it against action_result.schema.yml, bind it to the current task/invocation/project/version, and create it with exclusive-create semantics so a second result cannot overwrite the first.

- [ ] **Step 4: Add the minimum atomic store**

Implement before SQL approval or Advisor consumption:

- Same-directory temporary write.
- Flush and fsync.
- os.replace.
- Parent-directory fsync.
- One exclusive workspace runner lock.
- Atomic create-if-absent for approval and Advisor consumption receipts.

The P0 guarantee is scoped to one local workspace and filesystem. Cross-file revisions and crash reconciliation are added in P1.

- [ ] **Step 5: Change Executor precedence**

Executor must:

1. Start an attempt and persist attempt_id.
2. Execute the rendered invocation.
3. Load the receipt for the same attempt_id.
4. Reduce semantic result to state.
5. Use process exit code only as diagnostic evidence.

- [ ] **Step 6: Keep CLI compatibility**

Existing CLI exit codes remain stable. CLI handlers emit ActionResult receipts through the harness rather than requiring Agent Runtime to infer from stage last_result.

- [ ] **Step 7: Run tests**

    pytest tests/test_agent_action_result.py tests/test_agent_workspace_store.py tests/test_agent_executor.py tests/test_harness_runtime.py tests/test_cli_smoke.py -q

Expected: PASS; no stale result can close an attempt.

- [ ] **Step 8: Commit**

    git add schemas/action_result.schema.yml src/risk_model_workbench/agent/workspace_store.py src/risk_model_workbench/harness/runtime.py src/risk_model_workbench/harness/errors.py src/risk_model_workbench/agent/executor.py src/risk_model_workbench/agent/state.py src/risk_model_workbench/cli.py tests/test_agent_action_result.py tests/test_agent_workspace_store.py tests/test_agent_executor.py tests/test_harness_runtime.py
    git commit -m "feat: add attempt-scoped action results"

### Task 4: Implement SQL Prepare, Approval, and Execute

**Priority:** P0.3

**Files:**

- Create: schemas/approval_record.schema.yml
- Create: tests/test_agent_sql_approval_flow.py
- Modify: src/risk_model_workbench/planning/execution_plan.py
- Modify: src/risk_model_workbench/harness/actions.py
- Modify: src/risk_model_workbench/harness/tools.py
- Modify: src/risk_model_workbench/agent/approvals.py
- Modify: src/risk_model_workbench/agent/policy.py
- Modify: src/risk_model_workbench/agent/executor.py
- Modify: src/risk_model_workbench/agent/state.py
- Modify: src/risk_model_workbench/data/sql_evidence.py
- Modify: src/risk_model_workbench/dp_feather.py
- Modify: src/risk_model_workbench/cli.py
- Modify: workflows/feature_selection.yml
- Modify: workflows/full_modeling.yml

- [ ] **Step 1: Write failing two-phase flow tests**

Cover:

    prepare -> review_ready
    execute -> waiting_for_approval
    approve -> execute exactly once
    reject -> remain blocked
    SQL hash drift -> old approval invalid
    config hash drift -> old approval invalid
    unknown remote outcome -> reconciliation_required

Also assert that train cannot run after review_ready.

Run:

    pytest tests/test_agent_sql_approval_flow.py -q

Expected: FAIL because the current plan only contains dry-run tasks and treats dry-run exit code zero as completion.

- [ ] **Step 2: Split tools at plan time**

Create fixed tool pairs:

- feature_prescreen_prepare / feature_prescreen_execute
- build_wide_sql_prepare / build_wide_sql_execute
- feature_refine_prepare / feature_refine_execute

Do not mutate the command after approval.

- [ ] **Step 3: Define approval subject**

Minimum subject:

    project
    version_id
    task_id
    invocation_hash
    config_snapshot_hash
    sql_evidence_manifest_hash
    sql_files: [{path, sha256}]
    operation_id

- [ ] **Step 4: Add approval lifecycle**

Statuses:

    pending
    approved
    rejected
    revoked
    consumed

Approval must be one-time consumable. Any subject drift revokes it.

- [ ] **Step 5: Preserve local-feather behavior**

For local_feather workflows:

- Do not create remote execute tasks.
- Produce an explicit by-design completion or skip artifact accepted by the workflow contract.
- Do not label local-feather completion as SQL-approved remote execution.

- [ ] **Step 6: Add DP intent and receipt**

Before external execution write:

    audit/external_operations/<operation_id>.intent.json

After confirmed success write:

    audit/external_operations/<operation_id>.receipt.json

When success cannot be proven, enter reconciliation_required.

- [ ] **Step 7: Run tests**

    pytest tests/test_agent_sql_approval_flow.py tests/test_data_pull_engine.py tests/test_sql_evidence.py tests/test_request_plan.py -q
    rmw workflow validate --workflow feature_selection
    rmw workflow validate --workflow full_modeling

Expected: PASS; approval escape count is zero.

- [ ] **Step 8: Commit**

    git add schemas/approval_record.schema.yml src/risk_model_workbench/planning/execution_plan.py src/risk_model_workbench/harness/actions.py src/risk_model_workbench/harness/tools.py src/risk_model_workbench/agent/approvals.py src/risk_model_workbench/agent/policy.py src/risk_model_workbench/agent/executor.py src/risk_model_workbench/agent/state.py src/risk_model_workbench/data/sql_evidence.py src/risk_model_workbench/dp_feather.py src/risk_model_workbench/cli.py workflows/feature_selection.yml workflows/full_modeling.yml tests/test_agent_sql_approval_flow.py
    git commit -m "feat: close sql approval execution loop"

### Task 5: Consume Advisor Responses Deterministically

**Priority:** P0.4

**Files:**

- Create: schemas/advisor_consumption.schema.yml
- Create: src/risk_model_workbench/agent/advisor_reducer.py
- Create: tests/test_agent_advisor_consumption.py
- Modify: schemas/advisor_request.schema.yml
- Modify: schemas/advisor_response.schema.yml
- Modify: src/risk_model_workbench/agent/advisor.py
- Modify: src/risk_model_workbench/agent/executor.py
- Modify: src/risk_model_workbench/agent/state.py
- Modify: src/risk_model_workbench/modeling/llm_tuning.py
- Modify: src/risk_model_workbench/cli.py
- Modify: tests/test_agent_advisor.py
- Modify: tests/test_llm_tuning.py

- [ ] **Step 1: Write failing decision tests**

Cover:

- continue with valid outputs.
- retry with a bounded attempt budget.
- stop to a terminal stopped state.
- needs_user_confirmation to waiting_for_user.
- rejected response to a new request path.
- Duplicate consumption.
- Stale attempt, round, context, or invocation hash.
- Missing output file.
- Absolute path, dot-dot path, and symlink escape.
- Tuning plan with invalid experiment, round, candidate count, or parameter bounds.

Run:

    pytest tests/test_agent_advisor_consumption.py -q

Expected: FAIL because resume currently checks only whether a request is answered.

- [ ] **Step 2: Bind request identity to context**

Request ID and request hash must include:

    task_id
    attempt_id
    request_type
    invocation_hash
    context_hash
    round

- [ ] **Step 3: Write a minimum immutable context manifest**

Before creating the request, enumerate the exact context files, record workspace-relative path, SHA256, size, and missing status, then write:

    audit/advisor_contexts/<request_id>.json

The P0 context_hash is the canonical hash of this immutable manifest. P1 enriches it with bounded summaries, redaction rules, capabilities, and token budgeting; P1 must not change the P0 identity semantics.

- [ ] **Step 4: Separate accept and consume**

- accept: schema validation plus immutable storage.
- consume: revalidate against current plan/state/context and apply exactly one state transition.

Write receipt:

    audit/advisor_consumptions/<request_id>.json

- [ ] **Step 5: Implement the reducer**

Public function:

    def consume_advisor_response(
        workspace: Path,
        request_id: str,
        current_state: dict[str, object],
    ) -> AdvisorConsumptionResult:
        ...

- [ ] **Step 6: Add explicit user confirmation**

Add CLI:

    rmw agent confirm --project ... --version-id ... --request-id ... --confirmed-by ...
    rmw agent reject --project ... --version-id ... --request-id ... --reason ...

Do not allow resume while waiting_for_user is unresolved.

- [ ] **Step 7: Expose a public tuning-plan validator**

Refactor the existing private normalization logic in modeling/llm_tuning.py into a public validator used by both training and Advisor consumption.

- [ ] **Step 8: Run tests**

    pytest tests/test_agent_advisor_consumption.py tests/test_agent_advisor.py tests/test_agent_executor.py tests/test_llm_tuning.py -q

Expected: PASS; stale acceptance, repeat consumption, and path escape counts are zero.

- [ ] **Step 9: Commit**

    git add schemas/advisor_request.schema.yml schemas/advisor_response.schema.yml schemas/advisor_consumption.schema.yml src/risk_model_workbench/agent/advisor.py src/risk_model_workbench/agent/advisor_reducer.py src/risk_model_workbench/agent/executor.py src/risk_model_workbench/agent/state.py src/risk_model_workbench/modeling/llm_tuning.py src/risk_model_workbench/cli.py tests/test_agent_advisor.py tests/test_agent_advisor_consumption.py tests/test_llm_tuning.py
    git commit -m "feat: consume advisor decisions deterministically"

### Task 6: Add Agent Closure Audit and Synthetic E2E

**Priority:** P0.5

**Files:**

- Create: tests/agent_scenarios/test_local_happy_path.py
- Create: tests/agent_scenarios/test_policy_fail_closed.py
- Modify: src/risk_model_workbench/project_state.py
- Modify: src/risk_model_workbench/run_evidence.py
- Modify: src/risk_model_workbench/agent/executor.py
- Modify: src/risk_model_workbench/versioning.py
- Modify: src/risk_model_workbench/cli.py
- Modify: tests/test_harness_hardening.py
- Modify: tests/test_version_workflow.py

- [ ] **Step 1: Write failing closure tests**

Strict audit must fail when:

- Agent plan/state/trace is missing for an Agent-managed version.
- Approval is pending or approved but unconsumed.
- Advisor response is accepted but unconsumed.
- A task is review_ready, waiting_for_user, or reconciliation_required.
- A real stage closes with scaffold evidence.

- [ ] **Step 2: Extend audit evidence**

Add a dedicated managed_by=agent version marker. Keep source_type for artifact/workspace provenance; do not overload it. rmw agent start must set managed_by=agent, while ordinary rmw version init remains unmanaged unless explicitly requested.

For managed_by=agent require:

    agent_plan.yml
    audit/agent_state.yml
    audit/agent_trace.jsonl
    action result receipts
    relevant approval/advisor receipts

Do not apply these requirements to non-Agent versions. Add separate fixtures for managed Agent v2, Agent v1 compatibility, and ordinary workbench versions.

- [ ] **Step 3: Build a synthetic local project fixture**

The fixture must complete:

    request
    -> agent start
    -> local safe stages
    -> controlled Advisor pause
    -> accepted and consumed golden response
    -> resume
    -> report
    -> strict audit complete

No direct stage CLI may be invoked from the test outside Agent Runtime.

- [ ] **Step 4: Run P0 gate**

    pytest tests/test_agent_contract.py tests/test_agent_plan_integrity.py tests/test_agent_action_result.py tests/test_agent_sql_approval_flow.py tests/test_agent_advisor_consumption.py tests/agent_scenarios -q
    test -z "$(rg -n 'pytest\.mark\.(xfail|skip|skipif)|pytest\.skip\(|unittest\.skip' tests/test_agent_contract.py tests/test_agent_plan_integrity.py tests/test_agent_action_result.py tests/test_agent_sql_approval_flow.py tests/test_agent_advisor_consumption.py)"
    pytest tests -q
    rmw agent tools --json
    rmw workflow validate --workflow full_modeling

Expected:

- All tests PASS.
- No P0 contract test remains xfail or skipped.
- approval escape = 0.
- unregistered/tampered tool execution = 0.
- stale Advisor consumption = 0.
- path escape = 0.
- synthetic version strict audit = complete.

- [ ] **Step 5: Commit**

    git add tests/agent_scenarios/test_local_happy_path.py tests/agent_scenarios/test_policy_fail_closed.py src/risk_model_workbench/project_state.py src/risk_model_workbench/run_evidence.py src/risk_model_workbench/agent/executor.py src/risk_model_workbench/versioning.py src/risk_model_workbench/cli.py tests/test_harness_hardening.py tests/test_version_workflow.py
    git commit -m "test: add agent closure integration gate"

## Chunk 2: P1 - Recovery, Host Integration, and Domain Skills

### Task 7: Extend the Workspace Store with Revisions and Cross-File Safety

**Priority:** P1.1

**Files:**

- Modify: src/risk_model_workbench/agent/workspace_store.py
- Modify: tests/test_agent_workspace_store.py
- Modify: src/risk_model_workbench/agent/state.py
- Modify: src/risk_model_workbench/agent/approvals.py
- Modify: src/risk_model_workbench/agent/advisor.py
- Modify: src/risk_model_workbench/agent/trace.py
- Modify: src/risk_model_workbench/registry.py
- Modify: src/risk_model_workbench/state.py

- [ ] **Step 1: Write failing persistence tests**

Cover:

- Temp file plus os.replace atomicity.
- fsync of file and parent directory.
- Workspace single-runner lock.
- Revision compare-and-swap.
- Interrupted write preserves the previous valid document.
- Two writers cannot silently lose an update.

- [ ] **Step 2: Extend WorkspaceStore**

Minimum API:

    class WorkspaceStore:
        def read_yaml(self, relative_path: str) -> VersionedDocument:
            ...

        def write_yaml(self, relative_path: str, payload: dict, expected_revision: int) -> int:
            ...

        def append_jsonl(self, relative_path: str, event: dict) -> None:
            ...

        def lock(self) -> ContextManager[None]:
            ...

- [ ] **Step 3: Migrate remaining Agent documents**

The P0 store already protects attempt and consumption receipts. Migrate:

- agent_state.yml
- approvals.yml
- Advisor requests/responses/receipts
- agent_trace.jsonl

Do not migrate version state and manifest until Agent files pass fault-injection tests.

- [ ] **Step 4: Migrate state and manifest writes**

Use a shared transaction_id in version state and artifact manifest updates so recovery can detect divergence.

- [ ] **Step 5: Run tests**

    pytest tests/test_agent_workspace_store.py tests/test_agent_state_trace.py tests/test_run_evidence.py tests/test_version_workflow.py -q

Expected: PASS; no corrupt YAML/JSON after injected failure.

- [ ] **Step 6: Commit**

    git add src/risk_model_workbench/agent/workspace_store.py src/risk_model_workbench/agent/state.py src/risk_model_workbench/agent/approvals.py src/risk_model_workbench/agent/advisor.py src/risk_model_workbench/agent/trace.py src/risk_model_workbench/registry.py src/risk_model_workbench/state.py tests/test_agent_workspace_store.py
    git commit -m "feat: add atomic agent workspace store"

### Task 8: Add Attempt Journal and Recovery

**Priority:** P1.2

**Files:**

- Create: src/risk_model_workbench/agent/recovery.py
- Create: tests/test_agent_recovery.py
- Create: tests/agent_scenarios/test_crash_recovery.py
- Modify: src/risk_model_workbench/agent/executor.py
- Modify: src/risk_model_workbench/agent/state.py
- Modify: src/risk_model_workbench/harness/actions.py
- Modify: src/risk_model_workbench/harness/runtime.py
- Modify: src/risk_model_workbench/cli.py

- [ ] **Step 1: Write four crash-point tests**

Inject crashes:

1. After execution intent, before command.
2. After command, before ActionResult commit.
3. After manifest update, before version state update.
4. After result commit, before Agent state transition.

- [ ] **Step 2: Add transition journal**

Each attempt records:

    attempt_id
    transition_id
    invocation_hash
    intent_status
    result_status
    transaction_id
    started_at
    finished_at

- [ ] **Step 3: Declare recovery policy per tool**

Reuse the single ToolSpec execution_semantics field introduced in Task 2:

    read_only
    idempotent_write
    non_idempotent_write
    external_unknown

Only read-only and explicitly idempotent writes may retry automatically.

- [ ] **Step 4: Add reconcile on run/resume**

On startup:

- Resolve completed local actions from their receipts.
- Requeue safe unstarted attempts.
- Keep unknown DP operations in reconciliation_required.
- Reject concurrent resume when another runner holds the workspace lock.

- [ ] **Step 5: Add CLI diagnostics**

    rmw agent diagnose --project ... --version-id ...
    rmw agent reconcile --project ... --version-id ... --operation-id ...

Reconcile must require explicit evidence and user identity text; it must never infer remote success.

The structured reconciliation result is:

    outcome: confirmed_succeeded | confirmed_failed | abandoned
    evidence_path: workspace-relative path
    evidence_sha256: string
    operator_identity: string
    note: string

Reconciliation may update execution state but may not synthesize missing model, evaluation, or report artifacts.

- [ ] **Step 6: Run tests**

    pytest tests/test_agent_recovery.py tests/agent_scenarios/test_crash_recovery.py tests/test_agent_executor.py -q

Expected: PASS; external operations are never duplicated.

- [ ] **Step 7: Commit**

    git add src/risk_model_workbench/agent/recovery.py src/risk_model_workbench/agent/executor.py src/risk_model_workbench/agent/state.py src/risk_model_workbench/harness/actions.py src/risk_model_workbench/harness/runtime.py src/risk_model_workbench/cli.py tests/test_agent_recovery.py tests/agent_scenarios/test_crash_recovery.py
    git commit -m "feat: recover interrupted agent attempts"

### Task 9: Build Versioned Host-Agent Context Packs

**Priority:** P1.3

**Files:**

- Create: src/risk_model_workbench/agent/context_pack.py
- Create: tests/test_agent_context_pack.py
- Create: tests/test_context_snapshot.py
- Modify: src/risk_model_workbench/agent/advisor.py
- Modify: src/risk_model_workbench/context_snapshot.py
- Modify: src/risk_model_workbench/agent/plan.py
- Modify: src/risk_model_workbench/cli.py
- Modify: schemas/advisor_request.schema.yml

- [ ] **Step 1: Write failing context tests**

Cover:

- Deterministic file order and SHA256.
- Bounded total bytes.
- Workspace-relative paths only.
- Sensitive-key redaction.
- Path and file-type allowlist.
- Rejection of feather, parquet, pickle, credential files, and symlinks.
- Rejection or redaction of sample-level identifiers and secret-like content.
- Maximum per-file bytes and maximum total bytes.
- Missing files explicitly recorded.
- Context hash changes when any included file changes.

- [ ] **Step 2: Define ContextPack**

Minimum fields:

    version
    project
    version_id
    task_id
    attempt_id
    request_type
    files: [{path, sha256, size, summary}]
    constraints
    allowed_tools
    output_contract
    context_hash

All summaries must be generated by deterministic local code. Context Pack creation must not call an LLM.

- [ ] **Step 3: Add CLI**

    rmw agent advisor context --project ... --version-id ... --request-id ... --json
    rmw agent capabilities --json

- [ ] **Step 4: Replace glob-only Advisor context**

Advisor requests point to one immutable context pack plus its hash. Keep file globs only as informational provenance.

Allowed inputs are limited to version state, Agent state, plan, manifest summaries, bounded metrics/config JSON/YAML, and explicitly approved text reports. Raw datasets, row-level samples, feather/parquet/pickle, secrets, local credential files, and symlinks are always excluded.

- [ ] **Step 5: Run tests**

    pytest tests/test_agent_context_pack.py tests/test_agent_advisor.py tests/test_context_snapshot.py -q

Expected: PASS; the same workspace state produces the same context hash.

- [ ] **Step 6: Commit**

    git add src/risk_model_workbench/agent/context_pack.py src/risk_model_workbench/agent/advisor.py src/risk_model_workbench/context_snapshot.py src/risk_model_workbench/agent/plan.py src/risk_model_workbench/cli.py schemas/advisor_request.schema.yml tests/test_agent_context_pack.py tests/test_context_snapshot.py
    git commit -m "feat: add host-agent context packs"

### Task 10: Create Four Thin Domain Skills

**Priority:** P1.4

**Required method:** Use @skill-creator. Run with-skill and baseline evals before accepting each Skill.

**Files:**

- Create: .agents/skills/rmw-model-training/SKILL.md
- Create: .agents/skills/rmw-model-training/evals/evals.json
- Create: .agents/skills/rmw-feature-selection/SKILL.md
- Create: .agents/skills/rmw-feature-selection/evals/evals.json
- Create: .agents/skills/rmw-model-evaluation/SKILL.md
- Create: .agents/skills/rmw-model-evaluation/evals/evals.json
- Create: .agents/skills/rmw-report-generation/SKILL.md
- Create: .agents/skills/rmw-report-generation/evals/evals.json
- Modify: .agents/skills/risk-model-workbench/SKILL.md

- [ ] **Step 1: Define non-overlapping trigger boundaries**

- risk-model-workbench: full-chain or ambiguous multi-stage modeling request.
- rmw-model-training: explicit training, tuning, candidate, or experiment request.
- rmw-feature-selection: explicit metadata, prescreen, wide SQL, refine, leakage, or feature convergence request.
- rmw-model-evaluation: explicit model effectiveness, stability, slicing, or champion/challenger request.
- rmw-report-generation: explicit model report, model card, executive summary, or evidence packaging request.

- [ ] **Step 2: Draft training Skill and evals**

The Skill may:

- Read request/config/state/manifest.
- Generate bounded Host-Agent tuning responses.
- Call registered rmw training and Advisor commands.

It may not:

- Implement LightGBM training itself.
- Change Agent state or manifest directly.
- Optimize directly on OOT.

Run at least three with-skill and baseline prompts through @skill-creator.

- [ ] **Step 3: Draft feature-selection Skill and evals**

The Skill must enforce:

- Registered SQL prepare before remote pull.
- Explicit approval bound to the prepared SQL before the registered execute action.
- Leakage stop rules.
- Version workspace and artifact contract checks.
- No direct vendor code mutation.

- [ ] **Step 4: Draft evaluation Skill and evals**

The Skill must distinguish:

- Model metrics from business risk slices.
- Valid/OOS/OOT roles.
- Missing dimensions from zero-valued results.
- Evaluation from champion/challenger comparison.

- [ ] **Step 5: Draft reporting Skill and evals**

The Skill must:

- Read registered evidence rather than loose files.
- Surface missing requirements.
- Avoid inventing unavailable MOB or future-performance results.
- Keep report generation deterministic through rmw report.

- [ ] **Step 6: Update root routing**

The root Skill owns full workflow and cross-stage stop rules. It must not duplicate every child Skill body.

- [ ] **Step 7: Run trigger and output evals**

For every Skill:

- 3 realistic positive prompts.
- 3 near-miss negative prompts.
- One missing-input case.
- One explicit single-stage case.
- One full-chain case that should stay with the root Skill.

Generate the skill eval viewer and obtain human review before acceptance.

Run structural validation:

    python /Users/guzijun/.agents/skills/skill-creator/scripts/quick_validate.py .agents/skills/rmw-model-training
    python /Users/guzijun/.agents/skills/skill-creator/scripts/quick_validate.py .agents/skills/rmw-feature-selection
    python /Users/guzijun/.agents/skills/skill-creator/scripts/quick_validate.py .agents/skills/rmw-model-evaluation
    python /Users/guzijun/.agents/skills/skill-creator/scripts/quick_validate.py .agents/skills/rmw-report-generation

For each Skill, use @skill-creator to write paired with_skill and without_skill outputs under:

    .agents/skills/<skill-name>-workspace/iteration-1/

Then run:

    cd /Users/guzijun/.agents/skills/skill-creator
    python -m scripts.aggregate_benchmark /Users/guzijun/Desktop/AI攻坚/risk_model_workbench/.agents/skills/<skill-name>-workspace/iteration-1 --skill-name <skill-name>
    python eval-viewer/generate_review.py /Users/guzijun/Desktop/AI攻坚/risk_model_workbench/.agents/skills/<skill-name>-workspace/iteration-1 --skill-name <skill-name> --benchmark /Users/guzijun/Desktop/AI攻坚/risk_model_workbench/.agents/skills/<skill-name>-workspace/iteration-1/benchmark.json --static /Users/guzijun/Desktop/AI攻坚/risk_model_workbench/.agents/skills/<skill-name>-workspace/iteration-1/review.html

Acceptance thresholds:

- Structural validation: 100%.
- Safety and approval assertions: 100%.
- Output assertions: at least 90%.
- Trigger precision and recall: at least 85%.
- Full-chain prompts must not be incorrectly captured by a child Skill.
- Human review status recorded as approved in each Skill eval directory.

- [ ] **Step 8: Commit**

    git add .agents/skills/risk-model-workbench .agents/skills/rmw-model-training .agents/skills/rmw-feature-selection .agents/skills/rmw-model-evaluation .agents/skills/rmw-report-generation
    git commit -m "feat: add risk modeling domain skills"

### Task 11: Define Local-Only Artifact Retention and Clean-Clone Audit

**Priority:** P1.5

**Files:**

- Create: schemas/artifact_retention.schema.yml
- Create: tests/test_clean_clone_audit.py
- Modify: src/risk_model_workbench/registry.py
- Modify: src/risk_model_workbench/manifest.py
- Modify: src/risk_model_workbench/run_evidence.py
- Modify: src/risk_model_workbench/project_state.py
- Modify: src/risk_model_workbench/cli.py
- Modify: .gitignore

- [ ] **Step 1: Write failing clean-clone tests**

Build a temporary Git repository fixture whose manifest registers:

- Required committed report artifacts.
- An optional local-only scores feather.
- An ignored sample cache.

Commit only allowed artifacts, construct a clean checkout with git archive HEAD, and assert:

- Strict workflow closure remains complete when every required contract artifact exists.
- The missing optional local-only file appears as an explicit availability warning.
- A missing required artifact remains incomplete.
- An ignored file cannot be registered without an explicit storage class.
- An unignored but untracked file is not misclassified as repository-retained.
- A legacy manifest without retention metadata remains readable and is labeled legacy_workspace rather than repository-retained.

Run:

    pytest tests/test_clean_clone_audit.py -q

Expected: FAIL because the current manifest treats registered local presence as implicit retention semantics.

- [ ] **Step 2: Define artifact retention metadata**

Minimum fields:

    storage_class: repository | workspace_only | local_only | external | legacy_workspace
    contract_role: required | optional
    sha256: string
    size_bytes: integer
    retention_reason: string
    regeneration: optional command or evidence description
    external_reference: optional opaque reference

Rules:

- A required workflow-contract artifact must exist locally for execution closure.
- A repository artifact counts as reproducible only when git ls-files confirms it is tracked.
- An untracked generated artifact is workspace_only until it is committed; not being ignored is insufficient.
- A local_only optional artifact may be absent in a clean clone only when hash, size, reason, and provenance are recorded.
- A local_only optional artifact cannot satisfy a required artifact pattern.
- An external artifact cannot be treated as locally reproduced evidence.
- Missing optional artifacts are warnings, never silently discarded.

- [ ] **Step 3: Update manifest registration**

Require callers that register ignored paths to declare storage_class. Default new generated artifacts to workspace_only. Promote to repository only when a repository-aware audit confirms the path is tracked. Treat legacy entries without this field as legacy_workspace and expose a migration warning.

- [ ] **Step 4: Extend audit output**

Add:

    availability_summary:
      repository_present: ...
      workspace_only_present: ...
      local_only_present: ...
      local_only_missing: ...
      external_references: ...
      legacy_workspace: ...
      required_missing: ...

Also expose separate execution_verdict and reproducibility_verdict values. Existing verdict remains backward compatible during the migration window and must document which dimension it represents.

Keep the existing strict verdict contract: required_missing is blocking; local_only_missing is non-blocking but visible.

- [ ] **Step 5: Reconcile .gitignore documentation**

Make comments and rules agree about sample.pkl. Do not start tracking an existing cache as part of this task.

- [ ] **Step 6: Run tests**

    pytest tests/test_clean_clone_audit.py tests/test_run_evidence.py tests/test_harness_hardening.py tests/test_version_workflow.py -q

Expected: PASS; clean-clone strict audit no longer depends on prohibited scored datasets.

- [ ] **Step 7: Commit**

    git add schemas/artifact_retention.schema.yml src/risk_model_workbench/registry.py src/risk_model_workbench/manifest.py src/risk_model_workbench/run_evidence.py src/risk_model_workbench/project_state.py src/risk_model_workbench/cli.py .gitignore tests/test_clean_clone_audit.py
    git commit -m "feat: define local-only artifact retention"

### Task 12: Run a Real Local-Feather Agent Dogfood Version

**Priority:** P1.6

**Files:**

- Create: projects/2026-05-fujie-gcard-v1/requests/20260710-agent-harness-dogfood.md
- Create at execution time: projects/2026-05-fujie-gcard-v1/versions/fujie_gcard_agent_v1_20260710/
- Create at execution time: projects/2026-05-fujie-gcard-v1/handoffs/<timestamp>-fujie_gcard_agent_v1_20260710.md
- Modify at execution time: projects/2026-05-fujie-gcard-v1/project_state.yml
- Modify at execution time: projects/2026-05-fujie-gcard-v1/versions/index.yml

- [ ] **Step 1: Create a new request and version**

Do not overwrite or mutate v8. Use fujie_gcard_agent_v1_20260710. If that ID exists when execution begins, stop and obtain approval for a new ID rather than overwriting it.

Create and validate:

    rmw request validate --project projects/2026-05-fujie-gcard-v1 --request projects/2026-05-fujie-gcard-v1/requests/20260710-agent-harness-dogfood.md
    rmw agent start --project projects/2026-05-fujie-gcard-v1 --request projects/2026-05-fujie-gcard-v1/requests/20260710-agent-harness-dogfood.md --version-id fujie_gcard_agent_v1_20260710 --workflow full_modeling

- [ ] **Step 2: Disable heuristic fallback for one controlled tuning round**

The run must produce:

    advisor_required
    -> Host-Agent context pack
    -> tuning response
    -> accept
    -> consume
    -> resume

Set:

    training.tuning.advisor.mode: host_agent
    training.tuning.advisor.fallback_to_heuristic: false

Use the exact interaction:

    rmw agent run --project projects/2026-05-fujie-gcard-v1 --version-id fujie_gcard_agent_v1_20260710
    rmw agent advisor list --project projects/2026-05-fujie-gcard-v1 --version-id fujie_gcard_agent_v1_20260710 --json
    rmw agent advisor context --project projects/2026-05-fujie-gcard-v1 --version-id fujie_gcard_agent_v1_20260710 --request-id <request_id> --json
    rmw agent advisor accept --project projects/2026-05-fujie-gcard-v1 --version-id fujie_gcard_agent_v1_20260710 --response <response.json>
    rmw agent resume --project projects/2026-05-fujie-gcard-v1 --version-id fujie_gcard_agent_v1_20260710

- [ ] **Step 3: Run only through Agent commands**

Allowed orchestration commands:

    rmw agent start
    rmw agent run
    rmw agent advisor ...
    rmw agent confirm
    rmw agent resume
    rmw agent status
    rmw version audit --strict

Do not invoke sample, feature, train, evaluate, compare, or report stage commands directly.

- [ ] **Step 4: Verify required evidence**

Require:

- agent_plan.yml
- audit/agent_state.yml
- audit/agent_trace.jsonl
- context pack
- Advisor request/response/consumption receipt
- attempt-scoped ActionResults
- version state and manifest
- model and report artifacts

- [ ] **Step 5: Enforce data safety**

- Do not commit scores_all_splits.feather.
- Do not commit sample.pkl.
- Apply the Task 11 retention metadata to every unavailable local-only artifact and verify the clean-clone fixture before committing the manifest.

- [ ] **Step 6: Run acceptance**

    rmw agent status --project projects/2026-05-fujie-gcard-v1 --version-id fujie_gcard_agent_v1_20260710 --json
    rmw version audit --project projects/2026-05-fujie-gcard-v1 --version-id fujie_gcard_agent_v1_20260710 --strict
    pytest tests -q

Expected: Agent status done; strict audit complete; full tests PASS.

- [ ] **Step 7: Write handoff and commit only permitted evidence**

Use rmw handoff write. Stage only:

- The approved request and execution plan.
- Project state, version index, and final handoff.
- The new version state, manifest, configs, summaries, reports, model card, and intentional model binary.
- Audit receipts required by the Agent closure contract.

Before commit run:

    git status --short
    git diff --cached --check
    git diff --cached --stat
    git diff --cached --name-only | rg '(\.feather$|\.parquet$|sample\.pkl$|\.progress\.lock$)'

Expected: the final command prints no paths. Also inspect every staged file larger than 10 MiB and confirm it is allowed by repository policy.

## Chunk 3: P2 - Shared ActionRunner, Eval, and Optional Adapters

### Task 13: Introduce the Shared ActionRunner

**Priority:** P2.1

**Files:**

- Create: src/risk_model_workbench/application/__init__.py
- Create: src/risk_model_workbench/application/context.py
- Create: src/risk_model_workbench/application/action_runner.py
- Create: src/risk_model_workbench/application/handlers/__init__.py
- Create: tests/test_action_runner.py
- Modify: src/risk_model_workbench/harness/actions.py
- Modify: src/risk_model_workbench/harness/tools.py

- [ ] **Step 1: Write runner contract tests**

Expected public API:

    result = runner.run(
        invocation=ActionInvocation(...),
        context=VersionContext(...),
        attempt_id="attempt_001",
    )

Assert:

- Registered handler only.
- Policy checked before handler.
- One ActionResult receipt.
- Trace correlation IDs.
- No arbitrary shell.

- [ ] **Step 2: Implement VersionContext**

Centralize:

    project_dir
    version_id
    workspace
    runtime_config_dir
    manifest_path
    version_state_path

- [ ] **Step 3: Implement handler registry**

Map action_id to a Python callable. Use fake test handlers at this stage. Keep legacy CLI rendering available and keep the production Agent Executor on cli.main until Tasks 14 and 15 provide every production handler.

- [ ] **Step 4: Prove the runner contract without switching production**

Verify policy, receipt, trace, and registry behavior with fake handlers. A missing production handler must fail closed rather than fall back to arbitrary shell.

- [ ] **Step 5: Run tests**

    pytest tests/test_action_runner.py tests/test_agent_plan_integrity.py -q

Expected: PASS; ActionRunner contract is stable while the existing Agent execution path remains unchanged.

- [ ] **Step 6: Commit**

    git add src/risk_model_workbench/application/__init__.py src/risk_model_workbench/application/context.py src/risk_model_workbench/application/action_runner.py src/risk_model_workbench/application/handlers/__init__.py src/risk_model_workbench/harness/actions.py src/risk_model_workbench/harness/tools.py tests/test_action_runner.py
    git commit -m "refactor: add shared action runner"

### Task 14: Migrate Evaluation and Report to Thin CLI Adapters

**Priority:** P2.2

**Files:**

- Create: src/risk_model_workbench/application/handlers/evaluate.py
- Create: src/risk_model_workbench/application/handlers/report.py
- Create: tests/test_cli_action_parity.py
- Modify: src/risk_model_workbench/cli.py
- Modify: src/risk_model_workbench/evaluation/run.py
- Modify: src/risk_model_workbench/reporting/render.py
- Modify: tests/test_report_render.py
- Modify: tests/test_metrics.py

- [ ] **Step 1: Capture CLI parity fixtures**

For evaluate, compare, and report, record:

- Exit code.
- Semantic ActionResult.
- Registered artifact paths.
- Stage status.
- Decision log entries.

- [ ] **Step 2: Extract evaluation handler**

Move orchestration out of cmd_evaluate and cmd_compare. CLI functions only parse args, build invocation, call ActionRunner, and render output.

- [ ] **Step 3: Extract reporting handler**

Move orchestration out of cmd_report while keeping report domain rendering in reporting modules.

- [ ] **Step 4: Verify parity**

    pytest tests/test_cli_action_parity.py tests/test_metrics.py tests/test_report_render.py tests/test_cli_smoke.py -q

Expected: old CLI commands remain compatible and produce equivalent artifacts.

- [ ] **Step 5: Commit**

    git add src/risk_model_workbench/application/handlers/evaluate.py src/risk_model_workbench/application/handlers/report.py src/risk_model_workbench/cli.py src/risk_model_workbench/evaluation/run.py src/risk_model_workbench/reporting/render.py tests/test_cli_action_parity.py tests/test_metrics.py tests/test_report_render.py
    git commit -m "refactor: move evaluation and reporting behind action runner"

### Task 15: Migrate Sample Check, Training, and Feature Selection

**Priority:** P2.3

**Files:**

- Create: src/risk_model_workbench/application/handlers/sample_check.py
- Create: src/risk_model_workbench/application/handlers/train.py
- Create: src/risk_model_workbench/application/handlers/feature_selection.py
- Create: tests/test_sample_check_action_parity.py
- Modify: src/risk_model_workbench/application/action_runner.py
- Modify: src/risk_model_workbench/agent/executor.py
- Modify: src/risk_model_workbench/cli.py
- Modify: src/risk_model_workbench/modeling/experiment.py
- Modify: src/risk_model_workbench/modeling/llm_tuning.py
- Modify: src/risk_model_workbench/feature_selection/refine.py
- Modify: src/risk_model_workbench/feature_selection/metadata.py
- Modify: src/risk_model_workbench/data/pull_engine.py
- Modify: tests/test_train_hardening.py
- Modify: tests/test_feature_pipeline_flow.py
- Modify: tests/test_agent_sql_approval_flow.py
- Modify: tests/test_agent_advisor_consumption.py
- Modify: tests/test_agent_executor.py
- Modify: tests/test_cli_action_parity.py

- [ ] **Step 1: Write sample-check parity tests and extract its handler**

Cover the `sample_check` action used by `full_modeling`, including valid input, missing/scaffold input, semantic receipt generation, state/manifest updates, and legacy CLI output. Register `sample_check.py` in ActionRunner and prove CLI and direct ActionRunner parity before switching Agent Executor.

- [ ] **Step 2: Write training parity tests**

Cover:

- Normal training.
- Advisor-required pause.
- Consumed tuning plan.
- Scaffold/missing input.
- Evaluation artifact handoff.

- [ ] **Step 3: Extract training handler**

The handler calls modeling services and returns semantic results. It does not own Agent state transitions.

- [ ] **Step 4: Write feature-selection parity tests**

Cover local feather and controlled DP prepare/execute paths.

- [ ] **Step 5: Extract feature-selection handler**

Keep:

- Metadata.
- Prescreen.
- Wide SQL.
- Refine.

as typed actions under one domain handler package. Do not turn D01/D02/D03 into separate Skills.

- [ ] **Step 6: Switch Agent Executor only after all handlers exist**

Before switching, assert that every action emitted by every supported workflow has a registered production handler. At minimum, `full_modeling` must resolve `sample_check`, feature selection, training, evaluation, and reporting through the handlers delivered by Tasks 14-15. Fail the registry-coverage test if any workflow action is unresolved.

Replace the default cli.main(argv) path with ActionRunner only after that coverage test passes. Keep an injectable test runner. Add a static test that rejects imports or calls to cli.main from agent/executor.py.

- [ ] **Step 7: Verify CLI and Agent compatibility**

    pytest tests/test_sample_check_action_parity.py tests/test_train_hardening.py tests/test_feature_pipeline_flow.py tests/test_agent_sql_approval_flow.py tests/test_agent_advisor_consumption.py tests/test_agent_executor.py tests/test_cli_action_parity.py tests/test_cli_smoke.py -q

Expected: PASS; no policy, approval, trace, or artifact registration bypass.

- [ ] **Step 8: Commit**

    git add src/risk_model_workbench/application/action_runner.py src/risk_model_workbench/application/handlers/sample_check.py src/risk_model_workbench/application/handlers/train.py src/risk_model_workbench/application/handlers/feature_selection.py src/risk_model_workbench/agent/executor.py src/risk_model_workbench/cli.py src/risk_model_workbench/modeling/experiment.py src/risk_model_workbench/modeling/llm_tuning.py src/risk_model_workbench/feature_selection/refine.py src/risk_model_workbench/feature_selection/metadata.py src/risk_model_workbench/data/pull_engine.py tests/test_sample_check_action_parity.py tests/test_train_hardening.py tests/test_feature_pipeline_flow.py tests/test_agent_sql_approval_flow.py tests/test_agent_advisor_consumption.py tests/test_agent_executor.py tests/test_cli_action_parity.py tests/test_cli_smoke.py
    git commit -m "refactor: move sample check training and feature selection behind action runner"

### Task 16: Repeat Dogfood Through ActionRunner

**Priority:** P2.4

**Files:**

- Create at execution time: projects/2026-05-fujie-gcard-v1/versions/fujie_gcard_action_runner_v1_20260710/
- Create at execution time: projects/2026-05-fujie-gcard-v1/handoffs/<timestamp>-fujie_gcard_action_runner_v1_20260710.md
- Modify at execution time: projects/2026-05-fujie-gcard-v1/project_state.yml
- Modify at execution time: projects/2026-05-fujie-gcard-v1/versions/index.yml

- [ ] **Step 1: Re-run the synthetic suite after migration**

Assert every invocation is handled by ActionRunner and no Agent execution path imports or calls cli.main.

- [ ] **Step 2: Create a second real dogfood version**

Use fujie_gcard_action_runner_v1_20260710. If it already exists, stop and obtain approval for a replacement ID. Reuse an approved request/config shape, but do not reuse P1 Agent state or receipts.

- [ ] **Step 3: Exercise both migrated high-complexity paths**

Require:

- One Advisor-required training decision.
- One feature-selection mode transition. For local feather, verify the explicit by-design path; use a stub SQL canary for prepare/approval/execute.
- Evaluation and reporting through the shared runner.

- [ ] **Step 4: Compare P1 and P2 evidence**

Compare:

- Final artifact contracts.
- Model/evaluation metrics within deterministic tolerance.
- Agent transitions.
- Trace completeness.
- CLI parity fixtures.

Do not require bitwise model equality when underlying libraries are nondeterministic; require configured seed and metric tolerances.

- [ ] **Step 5: Run acceptance**

    rmw agent status --project projects/2026-05-fujie-gcard-v1 --version-id fujie_gcard_action_runner_v1_20260710 --json
    rmw version audit --project projects/2026-05-fujie-gcard-v1 --version-id fujie_gcard_action_runner_v1_20260710 --strict
    pytest tests/test_action_runner.py tests/test_cli_action_parity.py tests/agent_scenarios -q

Expected: done, strict audit complete, and no direct cli.main execution.

- [ ] **Step 6: Write handoff and commit only permitted evidence**

Use explicit staging. Apply the artifact-retention contract from Task 11.

### Task 17: Productize Agent Eval

**Priority:** P2.5

**Files:**

- Create: evals/agent_harness/scenarios.yml
- Create: src/risk_model_workbench/agent/eval.py
- Modify: tests/agent_scenarios/test_local_happy_path.py
- Modify: tests/agent_scenarios/test_policy_fail_closed.py
- Modify: tests/agent_scenarios/test_crash_recovery.py
- Modify: src/risk_model_workbench/cli.py
- Modify: docs/agent_product.md

- [ ] **Step 1: Define scenario schema**

Each scenario includes:

    name
    fixture
    expected_transitions
    expected_blockers
    expected_artifacts
    expected_audit_verdict
    forbidden_runner_calls

- [ ] **Step 2: Add core scenarios**

- Local happy path.
- SQL approve, reject, and drift.
- Advisor continue, retry, stop, and user confirmation.
- Stale response and path escape.
- Plan tamper and cycle.
- Scaffold closure rejection.
- Four crash points.
- Concurrent resume.

- [ ] **Step 3: Add eval command**

    rmw agent eval --suite harness --json

Output metrics:

    policy_escape_count
    transition_conformance_rate
    recovery_success_rate
    duplicate_external_execution_count
    trace_coverage_rate
    strict_audit_complete_rate

Define denominators in the JSON output:

- transition_conformance_rate = conforming transitions / asserted transitions.
- recovery_success_rate = successfully reconciled crash scenarios / crash scenarios.
- trace_coverage_rate = expected correlated events present / expected correlated events.
- strict_audit_complete_rate = strict-complete terminal scenarios / terminal happy-path scenarios.
- Count metrics always report both count and scenario IDs.

- [ ] **Step 4: Keep Host quality separate**

Runtime tests use golden Advisor responses. Codex/Claude recommendation quality is outside the Harness eval; it is covered by the Task 10 domain Skill evals for schema pass rate, context completeness, parameter-bound compliance, and human usefulness.

- [ ] **Step 5: Run P2 gate**

    rmw agent eval --suite harness --json
    pytest tests -q

Expected:

- policy_escape_count = 0
- transition_conformance_rate = 1.0
- recovery_success_rate = 1.0
- duplicate_external_execution_count = 0
- trace_coverage_rate = 1.0
- strict_audit_complete_rate = 1.0
- full tests PASS

- [ ] **Step 6: Commit**

    git add evals/agent_harness/scenarios.yml src/risk_model_workbench/agent/eval.py tests/agent_scenarios/test_local_happy_path.py tests/agent_scenarios/test_policy_fail_closed.py tests/agent_scenarios/test_crash_recovery.py src/risk_model_workbench/cli.py docs/agent_product.md
    git commit -m "test: add agent harness evaluation suite"

### Task 18: Apply the Optional MCP Adapter Gate

**Priority:** P2.6, conditional

**Files:**

- Create: docs/adr/0004-mcp-adapter-decision.md
- Create only if gate passes: src/risk_model_workbench/adapters/mcp.py
- Create only if gate passes: tests/test_mcp_action_adapter.py
- Modify only if gate passes: docs/agent_product.md

- [ ] **Step 1: Evaluate the gate**

Proceed only if at least one is true:

- Codex/Claude CLI integration has repeated parameter/schema failures that typed MCP calls would prevent.
- More than one Host-Agent needs the same local tool protocol.
- Human CLI remains stable and ActionRunner parity is complete.

Record the evidence and decision in docs/adr/0004-mcp-adapter-decision.md. If none is true, set the ADR decision to deferred and stop implementation.

- [ ] **Step 2: Keep MCP thin**

The adapter may:

- Expose capabilities.
- Parse typed parameters.
- Call ActionRunner.
- Return ActionResult.

It may not:

- Own state.
- Bypass policy.
- Implement retries.
- Execute arbitrary shell.

- [ ] **Step 3: Verify parity**

    pytest tests/test_mcp_action_adapter.py tests/test_cli_action_parity.py tests/test_action_runner.py -q

Expected: MCP and CLI return equivalent semantic results for the same invocation.

- [ ] **Step 4: Commit if implemented**

If deferred:

    git add docs/adr/0004-mcp-adapter-decision.md
    git commit -m "docs: defer mcp action adapter"

If implemented:

    git add docs/adr/0004-mcp-adapter-decision.md src/risk_model_workbench/adapters/mcp.py tests/test_mcp_action_adapter.py docs/agent_product.md
    git commit -m "feat: add thin mcp action adapter"

## 2. Acceptance Matrix

| Capability | P0 | P1 | P2 |
| --- | --- | --- | --- |
| Plan/tool tamper rejected before execution | Required | Required | Required |
| SQL prepare/approval/execute exact binding | Required | Required | Required |
| Advisor decision consumed exactly once | Required | Required | Required |
| Synthetic Agent strict audit complete | Required | Required | Required |
| Minimum atomic receipts and single-runner lock | Required | Required | Required |
| Atomic versioned Agent documents and cross-file safety |  | Required | Required |
| Crash recovery and reconciliation |  | Required | Required |
| Deterministic Host context pack |  | Required | Required |
| Four domain Skills pass evals |  | Required | Required |
| Clean-clone audit handles optional local-only artifacts explicitly |  | Required | Required |
| Real local-feather Agent dogfood |  | Required | Required |
| Executor independent from cli.main |  |  | Required |
| CLI and Agent share ActionRunner |  |  | Required |
| Post-ActionRunner real dogfood |  |  | Required |
| Agent eval command and metrics |  |  | Required |
| MCP adapter |  |  | Conditional |

## 3. Iteration Operating Rules

- Use a dedicated codex/ branch or worktree for every priority chunk.
- Do not work directly on a dirty main worktree containing uncommitted modeling artifacts.
- At most one task is in progress per worktree.
- Every task begins with a failing test.
- Every task ends with targeted tests, then the priority gate.
- Author and reviewer must be separate agents or contexts.
- Preserve legacy CLI behavior until parity tests prove a safe migration.
- New Agent schemas are versioned; never silently reinterpret old workspace state.
- Do not close a priority if any security metric is non-zero.
- Do not use scaffold/imported evidence as real dogfood proof.

## 4. Final Verification Commands

P0:

    pytest tests/test_agent_contract.py tests/test_agent_plan_integrity.py tests/test_agent_action_result.py tests/test_agent_sql_approval_flow.py tests/test_agent_advisor_consumption.py tests/agent_scenarios -q
    pytest tests -q
    rmw workflow validate --workflow full_modeling
    rmw agent tools --json
    jm agent tools --json

P1:

    pytest tests/test_agent_workspace_store.py tests/test_agent_recovery.py tests/test_agent_context_pack.py tests/agent_scenarios -q
    pytest tests -q
    rmw agent status --project <project> --version-id <dogfood_version> --json
    rmw version audit --project <project> --version-id <dogfood_version> --strict

P2:

    pytest tests/test_action_runner.py tests/test_cli_action_parity.py tests/agent_scenarios -q
    rmw agent eval --suite harness --json
    pytest tests -q

## 5. Execution Handoff

When this plan is approved:

1. Create a clean codex/ worktree for P0.
2. Execute Chunk 1 with @subagent-driven-development.
3. Use a fresh implementation subagent per task.
4. Run a separate spec/code review and verifier pass for every task.
5. Stop at each P0/P1/P2 gate for evidence review; do not ask whether to continue between tasks inside an approved chunk unless blocked.
