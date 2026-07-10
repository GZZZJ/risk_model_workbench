# Request-Driven Workflow

Users can provide a Markdown model request with YAML front matter. The request
acts as the task contract for the local RMW Agent runtime and for the external
Host-Agent layer such as Codex or Claude Code.

For non-technical users, use the static request builder:

```bash
open tools/model_request_builder/index.html
```

The page lets users select sample, split, feature-selection, modeling,
evaluation, risk-profile, and report requirements, then download Markdown.
Current support boundaries are documented in
`docs/model_request_builder_support_audit.md`.

Standard Agent flow:

1. Validate the project config.
2. Validate the model request.
3. Generate `execution_plan.yml`.
4. Initialize a version and copy the request plus plan into the version workspace.
5. Bind `execution_plan.yml` into version-scoped `agent_plan.yml`.
6. Execute safe tasks through `rmw agent run`.
7. Pause for SQL/DP approval, Host-Agent advisor input, missing data, or failed audit evidence.
8. Register artifacts and decisions.
9. Record missing reusable capabilities in `audit/improvement_candidates.md`.

The local Agent runtime owns deterministic execution, policy gates, state, and
trace artifacts. Codex or Claude Code remains the Host-Agent intelligence layer
for ambiguous diagnosis, tuning advice, and product judgment. The version
workspace remains the source of truth.

Recommended commands:

```bash
rmw project validate --project <project>
rmw request validate --project <project> --request <request.md>
rmw plan create --project <project> --request <request.md>
rmw agent start --project <project> --request <request.md> --version-id <version_id> --workflow full_modeling
rmw agent run --project <project> --version-id <version_id>
rmw agent status --project <project> --version-id <version_id>
rmw version audit --project <project> --version-id <version_id> --strict
```

When the Agent pauses for Host-Agent advice, inspect and answer the Advisor
request with the local JSON protocol:

```bash
rmw agent advisor list --project <project> --version-id <version_id>
rmw agent advisor show --project <project> --version-id <version_id> --request-id <request_id>
rmw agent advisor accept --project <project> --version-id <version_id> --response <response.json>
rmw agent resume --project <project> --version-id <version_id>
```

`jm` remains a compatible CLI alias for existing automation, but new workflow
docs should prefer `rmw`.

After a real project finishes, review its version workspace before changing the
generic workbench:

1. Keep one-off business assumptions in the request or project config.
2. Promote repeated code into a CLI command or shared module.
3. Add or update tests for the promoted behavior.
4. Import historical artifacts into a standard run if they came from an older
   layout.

For the current Fujie GCard legacy/example baseline, historical outputs can be normalized with:

```bash
rmw run import-gcard-model-artifacts \
  --project projects/2026-05-fujie-gcard-v1 \
  --run-id 2026-06-imported-gcard-main-lgbm
```
