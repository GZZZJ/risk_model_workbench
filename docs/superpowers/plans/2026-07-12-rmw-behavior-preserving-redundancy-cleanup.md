# RMW Behavior-Preserving Redundancy Cleanup Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove redundant RMW implementation and duplicated facts without reducing public compatibility, registered evidence, or v8 audit closure.

**Architecture:** Freeze public behavior with characterization tests, converge configuration and documented state, centralize workflow/action contracts, port any missing handler behavior before replacing legacy implementations with shims, and add a read-only asset cleanup plan. Destructive storage work remains a separately approved change.

**Tech Stack:** Python 3, argparse, dataclasses, PyYAML, JSON/YAML schemas, pytest, Git worktrees, existing RMW state and audit contracts.

---

## Chunk 1: Baseline and Facts

### Task 0: Freeze the compatibility baseline

**Files:**
- Create: `tests/test_compatibility_contract.py`
- Create: `tests/fixtures/compatibility/cli_surface.json`

- [ ] Capture the normalized parser surface, console aliases, legacy imports,
      public registry object shapes, and committed pickle module references.
- [ ] Add disposable-project mutation characterization for request, plan,
      version initialization, and migrated stage commands.
- [ ] Verify editable and wheel installs preserve the compatibility surface.
- [ ] Run `pytest tests -q`, `rmw doctor`, project validation, and v8 strict
      audit. Stop on any failure.

### Task 1: Converge configuration and documented state

**Files:**
- Modify: `src/risk_model_workbench/config.py`
- Modify: `src/risk_model_workbench/paths.py`
- Modify: `src/risk_model_workbench/project/create.py`
- Modify: `templates/project/`, current project config mirrors, and the three
  entrypoint documents
- Test: `tests/test_config.py`, `tests/test_project_schema.py`, and a new
  documented-state consistency test

- [ ] Write failing tests for strict YAML format errors, canonical/legacy-only
      reads, equal dual files, divergent dual files, and canonical-only writes.
- [ ] Implement strict mapping loading and a centralized variant resolver.
- [ ] Generate compatibility mirrors from canonical content and synchronize
      the current project and template.
- [ ] Add parseable current-state blocks and consistency tests.
- [ ] Re-run config, request-materialization, project-creation, and full tests.

## Chunk 2: Contracts and Runtime

### Task 2: Centralize stage and command contracts

**Files:**
- Create: `workflows/stage_contracts.yml`
- Modify: `src/risk_model_workbench/workflow_contracts.py`
- Modify: `src/risk_model_workbench/harness/actions.py`
- Modify: `src/risk_model_workbench/harness/tools.py`
- Test: `tests/test_harness_registry.py`, `tests/test_cli_action_parity.py`

- [ ] Snapshot effective workflow contracts and public ActionSpec/ToolSpec
      objects before changing implementation.
- [ ] Load shared stage contracts and require every workflow stage to reference
      a standard contract or declare `closure_required: false`.
- [ ] Introduce one internal display/argv declaration while retaining existing
      public dataclass behavior and rendered output.
- [ ] Validate every workflow and run registry/parity tests.

### Task 3: Retire dormant implementations behind permanent shims

**Files:**
- Modify: `src/risk_model_workbench/cli.py`
- Modify: `src/risk_model_workbench/application/handlers/`
- Modify: `jingying_agent/`
- Modify: `src/risk_model_workbench/project.py`
- Test: ActionRunner compatibility and feature/sample parity tests

- [ ] Write failing parity tests for every behavior present only in dormant CLI
      code, including monthly and segment sample outputs.
- [ ] Port the missing behavior into production handlers and move tests off
      private legacy helpers.
- [ ] Remove `_legacy_cmd_*` code only when state, artifacts, failure codes, and
      output parity are green.
- [ ] Replace root historical implementations and `project.py` with forwarding
      shims; retain all promised package/module paths.
- [ ] Run targeted parity, full tests, package build/install, and v8 audit.

## Chunk 3: Asset Governance and Closure

### Task 4: Add a non-destructive cleanup plan

**Files:**
- Create: `schemas/artifact_cleanup_plan.schema.yml`
- Modify: `src/risk_model_workbench/registry.py`
- Modify: `src/risk_model_workbench/cli.py`
- Create: `tests/test_artifact_cleanup_plan.py`

- [ ] Test deterministic IDs, duplicate grouping, active/required/raw retain
      rules, Harness manual review, lineage archive candidates, tracked Feather
      review, output path confinement, and idempotence.
- [ ] Implement `rmw project cleanup-plan` with human, JSON, and explicit
      project-contained output modes.
- [ ] Verify the command performs no deletion, copying, manifest backfill, or
      Git mutation.

### Task 5: Verify and hand off

- [ ] Run `pytest tests -q`, doctor, project validate/status, every workflow
      validator, v8 strict audit, and `python -m build`.
- [ ] Install editable and wheel packages in temporary environments and test
      all console aliases and legacy imports.
- [ ] Dispatch independent code-reviewer and verifier agents; address findings
      and rerun affected gates.
- [ ] Write the explicit RMW handoff with final commands, remaining retention
      warnings, and the separately approved destructive-cleanup boundary.
