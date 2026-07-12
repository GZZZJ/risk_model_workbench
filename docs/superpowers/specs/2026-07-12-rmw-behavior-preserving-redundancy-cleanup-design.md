# RMW Behavior-Preserving Redundancy Cleanup Design

Date: 2026-07-12

## Purpose

Reduce duplicated configuration, contracts, compatibility implementations, and
workspace assets without removing public behavior or weakening audit evidence.
The current Fujie GCard v8 workspace is the regression baseline.

## Compatibility Contract

The cleanup must preserve:

- the `rmw`, `jm`, and `jingying-agent` console entrypoints and `agent.py`;
- legacy `jingying_agent.*` and `jingying_model_agent.*` imports;
- migrated legacy run lookup through `rmw run status/audit` lineage;
- CLI argument defaults, exit codes, core stdout/stderr fields, state
  transitions, manifest registration, and required artifact paths;
- editable-install and wheel-install behavior;
- any historical Python module path referenced by committed pickle artifacts.

Compatibility modules remain permanent forwarding shims when their paths form
part of this contract. The obsolete implementation behind a shim may be
removed only after package, import, and mutation-parity tests pass.

## Canonical Configuration

- Project canonical path: `project.yml`; legacy mirror: `project.yaml`.
- Stage canonical path: `configs/<name>.yaml`; legacy mirror:
  `configs/<name>.yml`.
- A single existing variant remains readable.
- Two valid, semantically equivalent mappings resolve to the canonical path.
- Two valid but different mappings fail with `ConfigConflictError`.
- Duplicate keys, multiple YAML documents, or non-mapping roots fail with
  `ConfigFormatError`.
- Empty YAML remains `{}`; anchors are resolved; environment-looking strings
  remain literal; unknown fields remain accepted as today.
- Writers update canonical files. Compatibility mirrors are generated from the
  canonical content and must not diverge.

## State Documentation

`project_state.yml` and the active `version_state.yml` remain authoritative.
README, AGENTS, and CLAUDE keep a manually edited, machine-readable current
state block with `project`, `active_version_id`, `objective`, `workflow`, and
`status`. Tests fail if the three blocks disagree with authoritative state.

## Contract Architecture

Shared stage contracts move to a registry loaded by workflow definitions.
Loading a workflow must produce the same effective contract as before.
Action and tool command declarations gain one internal source of truth for the
stable display template and executable argv template. Existing public
`ActionSpec` and `ToolSpec` fields, constructors, serialization, repr, and
equality semantics remain unchanged.

## Code Migration

The ActionRunner handlers are compared against the dormant CLI implementations
before old code is removed. Missing behavior is ported first, including monthly
and configured-segment sample distributions. Root compatibility modules are
reduced to forwarding shims rather than deleting public paths.

## Asset Governance

The workbench gains a read-only cleanup-plan command. It reports stable
candidate IDs, hashes, sizes, lineage, retention role, reason, and proposed
action. It does not delete, archive, copy, recommit, or rewrite history.
Active required artifacts and raw data default to retain. Harness workspaces,
legacy run/version duplicates, and tracked score Feather files remain pending
human review.

## Verification and Rollback

Each implementation batch is an independent commit. The full test suite,
package install checks, CLI compatibility tests, project validation, and Fujie
GCard v8 strict audit must be green before and after every material batch. A
failure stops the cleanup; no pre-existing failure is accepted as baseline.

## Non-Goals

- No legacy run or version deletion.
- No Git history rewrite.
- No Feather or raw-data movement.
- No report-engine split or unrelated helper abstraction.
- No v8 manifest retention backfill without separate approval.
