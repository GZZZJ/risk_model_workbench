# ADR 0002: Local Agent Runtime

- status: accepted
- date: 2026-07-06

## Context

The workbench already has stable `rmw` actions, workflow contracts, version
state, artifact manifests, and read-only auditors. Before this ADR, Codex or
Claude Code acted as the orchestration layer directly: they validated requests,
created plans, initialized workspaces, selected commands, and decided when to
pause.

That model is useful, but it leaves important product behavior in soft prompts:
task execution order, permission gates, approval state, retry boundaries, and
Host-Agent handoffs are not first-class local artifacts.

## Decision

Add `RMW Agent` as a local, semi-autonomous CLI Agent runtime exposed through
`rmw agent ...`.

- `rmw agent start` creates a version-scoped Agent workspace from a model
  request and execution plan.
- `rmw agent run` and `rmw agent resume` execute safe local tasks
  deterministically through existing `rmw` commands.
- `rmw agent approve` records approval for high-risk SQL/DP actions but does
  not execute the action by itself.
- `rmw agent status` exposes Agent state, latest audit, manifest summary, and
  recent trace events.
- `rmw agent advisor list/show/accept` standardizes Host-Agent consultation
  through local request and response JSON files.
- `rmw agent tools --json` exports the Agent-visible tool schema from the
  existing action and tool registries.

The runtime remains local and conservative:

- It does not embed an LLM provider or require API keys.
- It does not add a web service or background daemon.
- It does not bypass SQL/DP approval gates.
- It does not treat scaffold or imported artifacts as complete local evidence.
- New Agent-managed work uses `versions/<version_id>/`; legacy
  `runs/<run_id>/` remains readable for compatibility only.

## Consequences

Agent orchestration now has durable local artifacts:

- `agent_plan.yml`
- `audit/agent_state.yml`
- `audit/agent_trace.jsonl`
- `audit/approvals.yml`
- `audit/advisor_requests/*.json`
- `audit/advisor_responses/*.json`

Codex or Claude Code remains the Host-Agent intelligence layer for ambiguous
diagnosis, tuning advice, and human-facing judgment. The repository owns the
deterministic execution loop, policy gates, state transitions, and audit
evidence. Advisor responses must match an existing request before
`rmw agent resume` may continue from `waiting_for_advisor`.
