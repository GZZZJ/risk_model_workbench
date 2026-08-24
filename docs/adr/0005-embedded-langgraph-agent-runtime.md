# ADR 0005: Embedded LangGraph Agent Runtime

- status: accepted
- date: 2026-08-23
- supersedes: ADR 0003 for the intelligence-layer boundary

## Context

RMW already owns a deterministic Agent Harness: typed plans, policy gates,
approval records, version-scoped state, ActionRunner dispatch, recovery,
semantic receipts, traces, and strict audit. It currently delegates every
non-deterministic judgment to Codex or Claude Code through the Advisor file
protocol. That makes the workbench safe but prevents it from operating as a
complete standalone Agent.

The target product must:

- understand a modeling objective and manage a bounded execution loop;
- make structured planning, routing, diagnosis, and tuning decisions;
- pause for explicit human approval on SQL/DP and other high-risk actions;
- recover without duplicating external effects;
- remain usable without Codex, Claude Code, their CLIs, or their Skills;
- preserve all existing version, manifest, approval, and audit contracts.

The local development runtime is Python 3.12.2. Current LangChain v1 and
LangGraph require Python 3.10 or newer, so the project can no longer promise
Python 3.9 compatibility.

## Decision

RMW will embed a LangGraph-based reasoning runtime above the existing Harness.

1. LangGraph owns the cognitive control graph: context assembly, structured
   reasoning, routing, bounded reflection, Advisor response generation, and
   human interrupts.
2. The RMW Harness remains the only authority for executable plans, policy,
   approval, task transitions, receipts, recovery, artifacts, and completion.
3. ActionRunner remains the only production action dispatch boundary. An LLM
   may propose an action but may not execute shell, SQL, filesystem, or modeling
   code directly.
4. Model access is hidden behind a local `ModelGateway` protocol. The first
   production adapter uses LangChain model abstractions and structured output;
   tests use a deterministic fake gateway.
5. The existing Advisor request/response protocol remains the durable domain
   contract. The embedded runtime answers pending requests and submits its
   response through the same validation and one-time consumption path.
6. LangGraph checkpoints store orchestration state only. Version state,
   `agent_state.yml`, approvals, receipts, and the artifact manifest remain the
   business source of truth.
7. Raw sample rows, DataFrames, credentials, and model binaries are forbidden
   in prompts and graph checkpoints. Context packs expose bounded summaries and
   registered evidence only.
8. SQL/DP execution, unknown external outcomes, destructive changes, and
   decisions explicitly marked for confirmation remain human-in-the-loop.

## State Ownership

| State | Owner | Purpose |
| --- | --- | --- |
| Graph thread/checkpoint | LangGraph | Current reasoning node, bounded messages, decision metadata, budgets |
| Agent plan/state | RMW Harness | Executable task graph and legal task transitions |
| Version state/manifest | RMW workspace | Authoritative business progress and evidence |
| Approval/consumption receipts | RMW Harness | Human authority and exactly-once evidence |
| Context pack | RMW Harness | Immutable, redacted evidence supplied to the model |
| Prompt/model configuration | Embedded runtime | Versioned reasoning behavior; never executable authority |

On resume, the runtime always reloads authoritative RMW state. A stale graph
checkpoint may be discarded and reconstructed; it may not overwrite workspace
state.

## Technology Choices

### Adopted from Agentic Design Patterns

- LangGraph for stateful orchestration, loops, persistence, and HITL.
- LangChain model interfaces and structured output for provider portability.
- Pydantic schemas for model inputs and outputs.
- Planning, routing, tool use, memory separation, bounded reflection,
  exception recovery, guardrails, and trajectory evaluation.
- MCP remains an optional interoperability adapter, not a runtime dependency.

### Deferred or Replaced

- CrewAI is not used as the primary runtime because RMW needs explicit state,
  transition, approval, and receipt control rather than role-oriented task
  delegation.
- Google ADK is an alternative only if the deployment standardizes on
  Vertex/Gemini; combining it with LangGraph would duplicate orchestration.
- A2A and multi-agent collaboration are deferred until evaluation proves a
  single graph cannot meet quality or context-isolation requirements.
- A vector database is not introduced initially. Repository and version
  evidence are small, structured, and better retrieved deterministically.
- Runtime self-modification is rejected. Improvements go through offline eval,
  human review, versioned prompts, and regression tests.
- LangSmith is optional and disabled by default because modeling context may be
  sensitive; local traces and eval artifacts are the initial observability
  source.

## Non-Functional Requirements

- **Safety:** zero policy escapes and zero SQL approval bypasses.
- **Reliability:** idempotent resume; no duplicate external execution.
- **Auditability:** every model decision binds request identity, context hash,
  model metadata, structured response, and validation outcome.
- **Privacy:** no raw row-level data or secrets in prompts/checkpoints/traces.
- **Availability:** deterministic CLI and Harness remain operational when the
  model provider is unavailable; judgment tasks fail closed with a local
  blocker rather than requesting a Host Agent.
- **Cost:** bounded model calls, retries, tokens, graph steps, and tuning
  candidates per request.
- **Maintainability:** provider-neutral gateway and deterministic fake model;
  graph nodes remain thin and domain behavior stays under RMW services.

## Failure Modes

| Failure | Required behavior |
| --- | --- |
| Missing model credentials/provider | Persist `model_unavailable`; do not execute a guessed action |
| Invalid structured model output | Retry within budget, then fail closed with validation evidence |
| Stale context/request identity | Reject response through the existing Advisor validator |
| Graph checkpoint loss | Rebuild orchestration state from RMW workspace |
| Crash around external action | Use existing attempt journal/reconciliation; never blindly replay |
| Excessive loop/tool calls | Stop at configured step/call/token budget |
| Sensitive context detected | Redact or reject before model invocation and record the reason |

## Consequences

### Positive

- RMW can complete judgment-bearing workflows without Codex or Claude Code.
- Existing safety and audit investment is preserved.
- Model providers and local models can be changed without changing domain
  execution contracts.
- The embedded Agent is testable offline with deterministic trajectories.

### Negative

- Python 3.9 compatibility is dropped.
- A model provider, credentials, and cost/latency management become runtime
  responsibilities.
- There are two persistence layers, requiring a strict ownership rule.
- LLM decision quality becomes part of the product evaluation surface.

## Alternatives Considered

### Rewrite the workbench as a generic LangChain Agent

Rejected because it would weaken version state, policy, approval, recovery, and
artifact audit contracts already implemented by RMW.

### Keep Codex/Claude as the permanent Host Agent

Rejected because the target product must be independently deployable and must
not require an external coding-agent session.

### Use CrewAI as the primary runtime

Rejected for the first implementation because explicit durable state and
fine-grained safety control matter more than role-based multi-agent ergonomics.

### Build a custom orchestration engine

Rejected because LangGraph already provides the required graph, persistence,
interrupt, and bounded-loop primitives. RMW should only retain its domain
state machine and execution semantics.
