# Workflow Standard

Every modeling workflow must use a `version_id` under
`projects/<project>/versions/<version_id>/`.

`version_state.yml` and `audit/artifact_manifest.json` are the source of truth.
Legacy `runs/<run_id>/run_state.yml` workspaces remain readable for historical
compatibility only.
