---
name: rmw-report-generation
description: Generate deterministic risk-model reports from registered Risk Model Workbench evidence. Always use this project skill for an explicit rmw report, model card, executive summary, named report target, or audit evidence package, including packages that must flag scaffold evidence or missing MOB and future-performance results. Use it even when project, version, manifest, evaluation artifacts, or requested evidence is missing and the report must surface the gap. Do not use for full-chain or ambiguous multi-stage modeling requests; route those to risk-model-workbench.
---

# RMW Report Generation

Own evidence packaging only. Generate reports through `rmw report`; do not
recompute training/evaluation logic or write an authoritative report from loose
workspace files.

## Route first

- Continue here for an explicit report, model card, executive summary, audit
  package, or configured report target.
- Delegate feature, training, evaluation, or comparison work to its domain
  skill when the requested evidence does not yet exist.
- Delegate a full-chain or ambiguous multi-stage request to
  `risk-model-workbench`.

## Build an evidence inventory

Read the project checkpoint, version state, `audit/artifact_manifest.json`,
report configuration, and only the artifacts registered for the named version.
Check artifact source/status so imported and scaffold outputs are labelled
correctly. Loose files may help diagnose a gap but may not substantiate a model
claim.

Before generation, enumerate required sections and classify their evidence as
available, missing, scaffold, imported, or not applicable. Surface missing
requirements rather than substituting plausible values. In particular, never
invent unavailable MOB, future-performance/OOT, stability, slice, or comparison
results; missing is not zero.

## Generate deterministically

Run the registered action:

```bash
rmw report --project <project> --version-id <version_id>
```

For a configured named output, use:

```bash
rmw report --project <project> --version-id <version_id> --report-target <target>
```

Do not hand-edit report outputs and call them authoritative. If required
evidence is missing, stop or allow the CLI's explicitly labelled scaffold path
according to the report contract; never remove its limitation label.

## Close the stage

Run version status and strict audit. Report generated targets, evidence sources,
missing sections, scaffold/import limitations, and registered output paths.
State explicitly that the report remains deterministic because it is generated
by `rmw report` from registered evidence, not hand-written or recomputed.
Report generation does not retroactively prove incomplete upstream stages.
