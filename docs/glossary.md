# Workbench Glossary

This glossary defines shared terms for `risk_model_workbench` harness work.

## Version

A concrete model-building workspace under
`projects/<project>/versions/<version_id>/`. A version contains request,
configuration snapshots, stage outputs, final artifacts, audit evidence, and
handoff-ready reports.

## Legacy Run

A pre-version workflow workspace under `projects/<project>/runs/<run_id>/`.
Legacy runs remain readable and auditable, but new work should initialize a
version.

## Stage

A named workflow step tracked in `version_state.yml` or legacy
`run_state.yml`.

## Stage Contract

Workflow YAML requirements that define the artifact evidence needed to close a
stage. Contracts are validated by `rmw workflow validate` and enforced by
`rmw version audit`; legacy `rmw run audit` remains compatible.

## Artifact Manifest

`audit/artifact_manifest.json`, the registered artifact inventory for a version
or legacy run.
Loose files are not closure evidence until registered.

## Imported Evidence

Artifacts copied from historical or external executions. They can be real
historical evidence, but they are not local reproduction evidence.

## Scaffold Evidence

Placeholder artifacts generated when local data, predictions, or execution
inputs are unavailable. Scaffold evidence is useful for continuity, but cannot
close a stage under strict audit.

## Guardrail

A rule enforced by CLI behavior, audit, tests, or repository policy.

## Proposed Rule

A lesson promoted into `docs/workbench_rules.yml` that still needs an
implementation, test, or explicit decision before it is `enforced`.
