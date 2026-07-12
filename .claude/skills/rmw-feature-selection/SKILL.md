---
name: rmw-feature-selection
description: Select and converge risk-model features in the local Risk Model Workbench. Always use this project skill for a request scoped to rmw feature metadata, prescreen, wide-table SQL generation or approved execution, refine, leakage checks, feature availability, or convergence, including an explicit request to rerun only refine. Use it even when project, version, sample source, SQL approval, or feature inputs are missing and the correct action is to stop. Do not use for full-chain or ambiguous multi-stage modeling requests; route those to risk-model-workbench.
---

# RMW Feature Selection

Own only feature intake and convergence. Use the registered `rmw feature`
actions; never run vendored feature-selection code directly or modify
`vendor/feature-select-v2/`.

## Route first

- Continue here for an explicit metadata, prescreen, wide SQL, refine, leakage,
  availability, or final-feature request.
- Delegate training, evaluation, and reporting to their corresponding RMW
  skills.
- Delegate full-chain or ambiguous multi-stage work to `risk-model-workbench`.

## Establish the contract

Read the project checkpoint, version state, manifest, request/plan, runtime
feature configuration, and registered sample evidence. Verify that all paths
belong to the named version. Treat loose project files and imported/scaffold
artifacts as context, not local reproduction evidence.

Stop on label leakage, post-outcome/future information, target-derived fields,
unresolved time semantics, missing mandatory columns, or incompatible sample
roles. Record the offending fields and reason through the supported RMW action;
do not silently drop evidence or directly edit state/manifest files.

## Local inputs

For declared local Feather data, use the registered sequence as applicable:

```bash
rmw feature metadata --project <project> --version-id <version_id>
rmw feature prescreen --project <project> --version-id <version_id>
rmw feature refine --project <project> --version-id <version_id>
```

Check the resulting version workspace paths and manifest entries after each
stage. Do not call project scripts or the vendor entrypoint.

## Remote DP safety gate

Remote pulls require a two-step, content-bound approval flow:

1. Prepare SQL with the registered action and no data pull:

```bash
rmw feature prescreen --project <project> --version-id <version_id> --dry-run-sql
rmw feature refine --project <project> --version-id <version_id> --dry-run-sql
```

2. Present the exact prepared SQL and its recorded identity to the user. Execute
   only after explicit approval of that prepared SQL, using the corresponding
   registered action with `--sql-approved`.

Never infer approval, reuse approval for changed SQL, or call `TMLSQLClient`
directly. If SQL changes, prepare and approve again. `--refresh-dp-cache` is a
new remote action and requires the same gate.

## Close the stage

Use version status and strict audit. Report input mode, leakage decisions,
prepared/approved SQL identity where relevant, converged feature count,
registered artifacts, and any missing contract. Feature selection alone does
not authorize training.
