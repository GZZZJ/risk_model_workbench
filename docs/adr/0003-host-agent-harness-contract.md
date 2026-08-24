# ADR 0003: External Host-Agent and Deterministic Harness Contract

- status: superseded by ADR 0005
- date: 2026-07-10

## Context

RMW deliberately depends on Codex or Claude Code for judgment. The workbench
itself must nevertheless remain deterministic, inspectable, and safe when the
Host-Agent proposes an action or supplies an Advisor response. The existing
local Agent runtime established durable plans, state, trace, and approval
records, but its plan fields, state transitions, semantic results, and
Host-Agent handoffs were not yet one versioned, fail-closed contract.

This ADR freezes that boundary before the control loop is hardened.

## Decision

RMW is a domain workflow Harness with an external Host-Agent intelligence
layer.

- Codex or Claude interprets intent, diagnoses ambiguity, proposes bounded
  tuning choices, and explains decisions to a human.
- RMW validates requests, binds typed plans, derives policy from its tool
  registry, executes deterministic actions, persists state, enforces approval,
  records trace and receipts, and performs strict audit.
- A human remains the authority for SQL/DP execution and any Advisor response
  that explicitly requires confirmation.
- Domain services implement deterministic modeling behavior. A Skill may guide
  the Host-Agent, but it may not write Harness state or manifests directly.

The Harness fails closed. Copied permission or command data in a plan is not an
authority. Policy and execution metadata are derived from the registered tool
at execution time.

## Version and Compatibility Policy

- Newly initialized Agent versions use Agent Plan v2 and Agent State v2.
- Existing v1 plans, state, manifests, and workspaces remain readable and
  status-inspectable.
- A v1 compatibility path may not grant more permission than the current tool
  registry.
- Schemas are versioned. RMW never silently reinterprets an old document as a
  new schema.
- Tool-registry or invocation drift blocks execution until an explicit safe
  rebind is reviewed.

## Execution Identity and Evidence

Every execution has an immutable `attempt_id`, a typed invocation, and exactly
one semantic ActionResult receipt bound to the task, project, version, and
invocation hash. Process exit status is diagnostic evidence; it is not the
source of truth for workflow meaning.

Approval is bound to the exact invocation, configuration snapshot, and SQL
evidence. Advisor acceptance and Advisor consumption are separate operations;
an accepted response is not effective until it is revalidated and consumed
exactly once.

## State Vocabulary

### Agent states

- `draft`: plan and state exist but execution has not started.
- `running`: a safe action is executing or may be selected.
- `waiting_for_approval`: a bound high-risk action requires human approval.
- `waiting_for_advisor`: a bound Host-Agent response is required.
- `waiting_for_user`: an Advisor decision requires explicit human confirmation.
- `reconciliation_required`: an external outcome cannot be proved locally.
- `blocked`: no safe automatic action is available.
- `failed`: execution ended because of a non-recoverable failure.
- `stopped`: an explicit stop decision ended the run safely.
- `done`: every required task and strict audit are complete.
- `done_with_gaps`: execution ended with explicit non-complete evidence and may
  not be represented as strict success.

### Task states

- `pending`: eligible only after real dependencies are complete.
- `running`: one attempt is active.
- `review_ready`: prepared evidence awaits review and is not execution success.
- `paused`: an approval, Advisor, or user decision blocks the task.
- `done`: the required action and evidence are complete.
- `scaffold`: placeholder or incomplete evidence exists.
- `failed`: the task ended unsuccessfully.
- `skipped`: a workflow-approved non-execution path was selected.
- `stopped`: an explicit stop decision ended the task.
- `reconciliation_required`: an external result cannot be proved locally.

## Transition Contract

### Agent transition table

| Current state | Event | Permitted target | Required evidence |
| --- | --- | --- | --- |
| `draft` | start | `running` | valid bound plan and state |
| `running` | run registered safe task | `running` | policy pass and attempt identity |
| `running` | request approval | `waiting_for_approval` | immutable approval subject |
| `waiting_for_approval` | consume or reject approval | `running` or `blocked` | current one-time record |
| `running` | request Advisor | `waiting_for_advisor` | immutable request and context manifest |
| `waiting_for_advisor` | consume response | `running`, `waiting_for_user`, or `stopped` | current accepted response and consumption receipt |
| `waiting_for_user` | confirm or reject | `running` or `stopped` | explicit user identity and decision receipt |
| `running` | unknown external outcome | `reconciliation_required` | external operation intent without proven receipt |
| `reconciliation_required` | reconcile | `running`, `failed`, or `stopped` | explicit operator evidence |
| `running` | block | `blocked` | one persisted `next_safe_action` |
| `blocked` | resolve, fail, or stop | `running`, `failed`, or `stopped` | completion receipt for the recorded safe action |
| `running` | unrecoverable error | `failed` | semantic failure receipt |
| `running` | explicit stop | `stopped` | stop decision receipt |
| `running` | strict close | `done` | all required tasks `done`, no blocker, strict audit complete |
| `running` | incomplete close | `done_with_gaps` | explicit gap evidence; never strict success |
| `failed` | none | none | terminal; create a new attempt/version rather than mutate history |
| `stopped` | none | none | terminal |
| `done` | none | none | terminal strict success |
| `done_with_gaps` | none | none | terminal non-strict outcome |

### Task transition table

| Current state | Event | Permitted target |
| --- | --- | --- |
| `pending` | start, wait for a bound decision, skip, or stop | `running`, `paused`, `skipped`, or `stopped` |
| `running` | prepare complete | `review_ready` |
| `running` | wait for a bound decision | `paused` |
| `running` | complete, scaffold, fail, skip, or stop | `done`, `scaffold`, `failed`, `skipped`, or `stopped` |
| `running` | unknown external outcome | `reconciliation_required` |
| `review_ready` | approval confirmed, request approval, revise, or stop | `done`, `paused`, `pending`, or `stopped` |
| `paused` | resume, fail, or stop | `pending`, `failed`, or `stopped` |
| `reconciliation_required` | confirm success, confirm failure, or abandon | `done`, `failed`, or `stopped` |
| `done` | none | none; terminal |
| `scaffold` | none | none; terminal non-strict evidence |
| `failed` | none | none; terminal |
| `skipped` | none | none; terminal by workflow contract only |
| `stopped` | none | none; terminal |

Any transition not listed above is rejected with stable failure code
`illegal_transition` or `illegal_task_transition`. In particular,
`waiting_for_approval`, `waiting_for_advisor`, `waiting_for_user`, and
`reconciliation_required` are non-terminal blocker states and cannot run a new
action until their one safe resolution path is completed.

### Persisted blocker contract

Every blocker document contains exactly one non-empty `next_safe_action`
object:

```yaml
blocker_type: approval | advisor | user | reconciliation | dependency
blocker_id: stable workspace-scoped identifier
next_safe_action:
  action: registered action or explicit human decision
  required_evidence: receipt or artifact needed to clear the blocker
  command_hint: optional display-only CLI hint
```

The Harness rejects a blocker without this object, rejects multiple competing
safe actions, and clears the object atomically only when the required evidence
has been consumed. `command_hint` is never executable authority.

## Invariants

1. An unapproved DP task is never runnable.
2. An accepted but unconsumed Advisor response never resumes a task.
3. `review_ready` is not a terminal success and cannot unlock real downstream
   work.
4. `scaffold` cannot satisfy a real dependency or strict closure.
5. Every blocker exposes one explicit next safe action.
6. `done` requires no pending or merely approved-but-unconsumed approval, no
   unconsumed Advisor response, no unresolved user confirmation, and a complete
   strict version audit.
7. Unknown external outcomes are never retried automatically; they enter
   `reconciliation_required`.
8. A consumed SQL approval is revalidated at the external client boundary;
   the client receives the immutable SQL evidence text, never an unchecked
   regenerated string.
9. Batch workers receive the exact approval ID, subject hash, consumption
   receipt, parent operation ID, and attempt ID. One unknown sub-operation
   makes the whole action `reconciliation_required`.

## Consequences

The external Host-Agent may evolve independently, while RMW retains stable
execution semantics. P0 implementation tasks must add the typed invocation,
production transition reducer, attempt-scoped result receipts, two-phase SQL
approval, and deterministic Advisor consumption required by this contract.

Existing CLI commands remain supported. Their internal implementation may be
made thinner later, but they cannot bypass this Harness contract.
