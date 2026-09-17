"""Small, traceable two-corpus retrieval with optional historical cutoff."""
from __future__ import annotations
import json
from pathlib import Path
from .schemas import ModelingState

FIXTURES = Path(__file__).parent / "fixtures"


class EvidenceRetriever:
    def __init__(self, fixture_path: Path | None = None):
        self.fixture_path = fixture_path

    def retrieve(self, state: ModelingState) -> dict:
        """Current Knowledge Demo has no cutoff; replay callers may pass as_of."""
        scenario_id = state.business_context.get("scenario_id", "auxiliary_model_incremental_value")
        fixture_path = self.fixture_path or FIXTURES / f"{scenario_id}.json"
        self.data = json.loads(fixture_path.read_text(encoding="utf-8"))
        as_of = state.business_context.get("as_of")
        records = self.data["decision_knowledge"] + self.data["historical_experience"] + self.data.get("negative_evidence", [])
        if as_of:
            records = [r for r in records if not r.get("recorded_at") or r["recorded_at"] <= as_of]
        return {"scenario_id": scenario_id, "scenario_type": self.data.get("scenario_type", "COMPOSITE DEMO SCENARIO"), "mode": state.business_context.get("knowledge_mode"), "as_of": as_of,
                "decision_knowledge": [r for r in records if r["kind"] == "decision_knowledge"],
                "historical_experience": [r for r in records if r["kind"] == "historical_experience" and r["id"] not in {x["id"] for x in self.data.get("negative_evidence", [])}],
                "negative_evidence": [r for r in records if r["id"] in {x["id"] for x in self.data.get("negative_evidence", [])}]}
