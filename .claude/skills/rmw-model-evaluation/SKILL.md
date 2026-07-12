---
name: rmw-model-evaluation
description: Evaluate or compare trained risk models in the local Risk Model Workbench. Use for explicit model effectiveness, discrimination, calibration, stability, slicing, PSI, valid/OOS/OOT interpretation, or champion/challenger requests, even when a requested dimension or registered score evidence is missing and the correct result is a documented gap. Do not use for full-chain or ambiguous multi-stage modeling requests; route those to risk-model-workbench.
---

# RMW Model Evaluation

Own evaluation and champion/challenger comparison, not training or reporting.
Read registered evidence and execute only through `rmw evaluate` and
`rmw compare`.

## Route first

- Continue here for explicit effectiveness, stability, slice, dataset-role, or
  model comparison work.
- Delegate training, feature selection, and report production to their domain
  skills.
- Delegate full-chain or ambiguous multi-stage work to
  `risk-model-workbench`.

## Validate evidence and roles

Read the project checkpoint, version state, manifest, evaluation configuration,
registered model outputs, and scored data contract. Stop if the trained model,
score column, label, sample role, or required time/slice dimension is missing.
Do not treat an imported/scaffold artifact as local evaluation evidence.

Keep dataset roles explicit:

- `valid` supports fitting decisions such as early stopping according to config.
- `OOS` is held-out selection/generalization evidence and may support configured
  candidate comparison.
- `OOT` is future-period final generalization and stability evidence; do not tune
  against it.

Distinguish model metrics (for example AUC/KS/calibration/lift) from business
risk slices (for example product, channel, customer segment, month, or MOB).
A missing dimension is unavailable evidence, not a zero-valued result. Never
fill an absent slice, MOB, PSI, or metric with `0`.

## Evaluate and compare

Run evaluation for one version:

```bash
rmw evaluate --project <project> --version-id <version_id>
```

Use `--scores-feather` or `--output-dir` only when declared by the version
contract. Evaluation describes one model's evidence. Champion/challenger is a
separate explicit action:

```bash
rmw compare --project <project> --version-id <version_id> --champion <score_column>
```

Pass each intended score column explicitly; do not reinterpret an evaluation
summary as a comparison. Apply configured metric direction, eligibility, and
slice gates rather than selecting a winner from one convenient number.

## Close the stage

Run version status and strict audit. Report dataset roles, metric evidence,
business slices, missing dimensions, comparison eligibility and outcome, and
registered artifacts. State clearly whether the result is evaluation only or a
champion/challenger decision.
