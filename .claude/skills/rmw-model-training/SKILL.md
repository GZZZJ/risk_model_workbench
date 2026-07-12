---
name: rmw-model-training
description: Train or tune one risk model experiment in the local Risk Model Workbench. Always use this project skill for a request scoped solely to rmw train, main_lgbm, LightGBM tuning, comparing tuning rounds, continuing the same experiment, bounded candidate parameters, or an RMW training Advisor response, including when evaluation and reporting are explicitly excluded. Use it even when project, version, features, or other required inputs are missing and the correct action is to stop and request evidence. Do not use for full-chain or ambiguous multi-stage modeling requests; route those to risk-model-workbench.
---

# RMW Model Training

Own only the training stage. Keep the external Host-Agent as the reasoning layer
and use registered `rmw` commands as the execution layer. Do not implement a
LightGBM trainer in the conversation or edit state/manifest files directly.

## Route first

- Continue here when the user explicitly requests training, tuning, candidates,
  an experiment, or a training Advisor response.
- Delegate feature convergence to `rmw-feature-selection`, effectiveness or
  champion/challenger work to `rmw-model-evaluation`, and evidence packaging to
  `rmw-report-generation`.
- Delegate a full workflow, several stages, or an ambiguous modeling request to
  `risk-model-workbench`.

## Establish evidence

Resolve `<project>` and `<version_id>`, then read the project checkpoint,
`version_state.yml`, and `audit/artifact_manifest.json`. Check registered sample
checks, final features, training configuration, input availability, and stage
preconditions. Loose files and scaffold/import artifacts are not local training
evidence.

Stop and report the exact missing prerequisite when required input, sample
checks, final features, configuration, or leakage clearance is absent. Never
mark a stage complete by editing state or the manifest.

## Execute through RMW

Preview when paths or prerequisites are uncertain:

```bash
rmw train --project <project> --version-id <version_id> --experiment <experiment> --plan-only
```

Run the registered action only after its prerequisites are satisfied:

```bash
rmw train --project <project> --version-id <version_id> --experiment <experiment>
```

Optional `--input-feather`, `--feature-list`, `--score-output`, and `--config`
must remain inside the declared project/version contract. Do not use
`--skip-split-check` unless the user explicitly accepts the resulting evidence
limitation.

## Host-Agent tuning protocol

For `training.mode: llm_guided_tune`, allow `rmw train` to emit an Advisor
request. Inspect it and its immutable context pack with:

```bash
rmw agent advisor show --project <project> --version-id <version_id> --request-id <request_id> --json
rmw agent advisor context --project <project> --version-id <version_id> --request-id <request_id> --json
```

Return 3–5 bounded candidates in the response schema requested by the context
pack. Give each candidate a name, permitted parameters, and a reason. Diagnose
from registered train/valid history; use OOS as configured for selection and
reserve OOT for final generalization evidence. Never optimize directly against
OOT as the primary tuning target. Submit only through:

```bash
rmw agent advisor accept --project <project> --version-id <version_id> --response <response.json>
```

Rerun `rmw train`; the workbench validates bounds and selects by configured
metric rules. Do not bypass the Advisor protocol or mutate its request files.

## Close the stage

Run version status and strict audit. Report the experiment, evidence status,
selected candidate or blocker, registered outputs, and any scaffold/import or
split-check limitation. Training alone does not authorize evaluation or report
generation. State explicitly that only registered `rmw` actions may update
`version_state.yml` or `artifact_manifest.json`; never edit either file
directly.
