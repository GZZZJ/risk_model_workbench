# LLM Tuning Decisions

- round 1: Baseline valid KS=0.71696 and valid AUC=0.93250 with train-valid AUC gap=0.01322, inside the 0.03 guardrail. Explore three bounded capacity/regularization trade-offs on validation metrics only; reserve OOT for final generalization assessment.
