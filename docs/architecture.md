# Architecture

`risk_model_workbench` is the 风险场景 AI 建模工作台 and a standalone local
modeling Agent. An embedded LangGraph/LangChain reasoning layer interprets user
goals and bounded Advisor requests. The existing RMW Harness provides stable
actions, workflow definitions, version state, approval, recovery, artifact
registration, and strict audit.

The workbench is intentionally split into five layers:

1. User-facing goal/request layer: a natural-language objective converted into
   a reviewed Markdown request draft, or a request generated from
   `tools/model_request_builder/index.html`.
2. Embedded reasoning layer: the LangGraph runtime routes deterministic
   execution, model decisions, and human interrupts. LangChain provides a
   provider-neutral gateway for OpenAI, Anthropic Claude, and compatible
   endpoints; every provider returns structured decisions through the same RMW
   Advisor contract.
3. Deterministic Harness layer: typed plans, policy, ActionRunner, approvals,
   receipts, recovery, version state, and strict audit remain executable
   authority.
4. Atomic capability layer: reusable modules under `src/risk_model_workbench/`
   implement sample checks, feature selection integration, training,
   evaluation, reporting, artifact registry, and version state.
5. Project workspace layer: every modeling attempt writes code snapshots,
   configs, intermediate outputs, final artifacts, and decisions under one
   `projects/<project>/versions/<version_id>/` directory.

Project-specific scripts are allowed during exploration, but once they become
repeatable they should be converted into CLI-backed modules. Legacy scripts
remain under `legacy_scripts/` as provenance, while the workbench API stays in
`src/risk_model_workbench/`.

Legacy `projects/<project>/runs/<run_id>/` workspaces remain readable for
compatibility. Standard legacy runs can be migrated into versions with
`rmw version migrate-standard-runs`.

The Fujie GCard `main_lgbm` versions are the current real-project case for this
architecture. They prove that external project outputs and historical standard
runs can be normalized into version workspaces and then used to harden generic
train, evaluate, and report capabilities without making GCard the default
workbench behavior.

LangGraph checkpoints are orchestration cursors only. `agent_state.yml`,
`version_state.yml`, approval/consumption receipts, and
`audit/artifact_manifest.json` remain authoritative and can reconstruct a lost
graph checkpoint. The detailed decision is recorded in
[ADR 0005](adr/0005-embedded-langgraph-agent-runtime.md).
