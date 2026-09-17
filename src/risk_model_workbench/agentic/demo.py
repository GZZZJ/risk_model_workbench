"""Run the auxiliary-model incremental-value Agentic Modeling Demo."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from .demo_executor import DemoActionExecutor
from .demo_runtime import DemoRuntime, render_steps, write_json
from .orchestrator import DemoFixtureOrchestratorAdapter, HostAgentFileAdapter
from .scenarios import SCENARIOS, initial_state
from .schemas import ModelingState


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, default=SCENARIOS[0])
    parser.add_argument("--counterfactual", action="store_true", help="COUNTERFACTUAL MOCK: covloss lowers corr but severely degrades KS")
    parser.add_argument("--adapter", choices=["demo", "host"], default="demo")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, help="Host-Agent request/response directory")
    parser.add_argument("--resume", action="store_true", help="Resume a cleanly paused host exchange; not crash recovery")
    args = parser.parse_args(argv)
    if args.adapter == "host" and not args.decisions:
        parser.error("--adapter host requires --decisions")
    if args.resume and args.adapter != "host":
        parser.error("--resume is only for clean Host-Agent pauses")
    output = args.output.resolve()
    config = {"scenario": args.scenario, "counterfactual": args.counterfactual, "adapter": args.adapter}
    if args.resume:
        if json.loads((output / "run_config.json").read_text()) != config:
            parser.error("resume options differ from the original session")
        state = ModelingState.model_validate_json((output / "current.json").read_text())
    else:
        if output.exists() and any(output.iterdir()):
            parser.error("output directory is not empty; use a new directory")
        session_id = re.sub(r"[^A-Za-z0-9_-]", "_", output.name) or "demo"
        state = initial_state(args.scenario, session_id)
        write_json(output / "run_config.json", config)
    adapter = DemoFixtureOrchestratorAdapter() if args.adapter == "demo" else HostAgentFileAdapter(args.decisions)
    runtime = DemoRuntime(adapter, executor=DemoActionExecutor(args.counterfactual))
    try:
        steps = runtime.run(state, output)
    except (ValueError, OSError) as exc:
        print(f"Decision/execution rejected: {exc}")
        return 1
    all_steps = steps
    if args.resume and (output / "trajectory.jsonl").exists():
        from .schemas import TrajectoryStep
        all_steps = [TrajectoryStep.model_validate_json(line) for line in (output / "trajectory.jsonl").read_text().splitlines() if line]
    (output / "demo_run.md").write_text(render_steps(all_steps), encoding="utf-8")
    for step in steps:
        print(f"{step.state_before.iteration + 1}: {step.decision.selected_action.action_type} -> {step.outcome.execution_status} [{step.outcome.implementation}]")
    print(runtime.pending or f"status: {(steps[-1].state_after if steps else state).status}")
    print(f"Adapter: {adapter.label}\nReport: {output / 'demo_run.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
