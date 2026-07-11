# ADR 0004: Add a Thin MCP Action Adapter

- status: accepted
- date: 2026-07-12

## Context

RMW supports Codex and Claude Code as external Host-Agents. Both need the same
typed local action protocol while the human-facing `rmw` CLI remains the stable
operator interface. Task 18 permits an MCP adapter only when it is another
adapter over the shared ActionRunner, not a second execution architecture.

## Gate Evidence

### Repeated CLI parameter or schema failures

Not established. This condition is not used to justify the decision.

### More than one Host-Agent needs the same local protocol

Satisfied. ADR 0003 and the project extension layout explicitly support both
Codex and Claude Code. Maintaining separate parameter encodings for the same
local modeling actions would duplicate integration behavior and increase drift
risk. A shared typed protocol therefore has current product value.

### Stable CLI and ActionRunner parity

Satisfied for the adapter boundary. Production write actions resolve through
the shared handler registry, the Agent Executor no longer imports or calls
`cli.main`, and CLI/direct ActionRunner parity tests cover the migrated action
families. The harness eval command also provides a fail-closed regression gate.

The second condition alone is sufficient under the approved Task 18 gate; the
parity evidence additionally makes a thin implementation safe. Completion of a
long-running real modeling version remains a release evidence item, not a
reason to create state or orchestration inside MCP.

## Decision

Add a transport-neutral `MCPActionAdapter` in
`src/risk_model_workbench/adapters/mcp.py`. It exposes MCP-safe capability
metadata, validates typed parameters against the canonical tool schemas,
constructs an `ActionInvocation`, calls an injected policy-gated ActionRunner
exactly once, and returns the resulting `ActionResult` payload.

The adapter does not start a daemon or select an MCP SDK/transport. Codex or
Claude integration may register this core with a local MCP transport later
without changing execution semantics.

## Hard Boundaries

The adapter may:

- expose typed capabilities;
- parse and validate typed parameters;
- construct `ActionInvocation` and `VersionContext` values;
- call an injected ActionRunner;
- return `ActionResult` as structured data.

The adapter may not:

- own or reduce Agent/version state;
- create an allow-all policy or bypass the runner's injected policy;
- retry an action;
- render or execute CLI arguments, shell commands, or subprocesses;
- implement approval, Advisor, recovery, or artifact rules.

## Consequences

- Codex and Claude can share one typed local action surface.
- Human operators continue to use `rmw`; MCP does not replace the CLI.
- CLI, Agent Runtime, and MCP converge on the same ActionRunner receipts,
  policy decisions, handlers, traces, and artifacts.
- Transport lifecycle, authentication, and remote MCP hosting remain out of
  scope; this is a local adapter core only.
