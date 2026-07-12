# LLM Tuning Decisions

- round 1: Baseline valid AUC/KS are strong with a controlled train-valid AUC gap (0.0132). Evaluate three bounded alternatives around regularization, sampling, and capacity; choose by configured valid KS then valid AUC, not OOT.
