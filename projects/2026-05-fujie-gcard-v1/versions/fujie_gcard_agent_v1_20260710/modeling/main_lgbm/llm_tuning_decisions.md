# LLM Tuning Decisions

- round 1: Baseline validation is already strong (KS 0.7170, AUC 0.9325) with a controlled train-validation AUC gap of 0.0132. Explore modest regularization and capacity changes while keeping OOT outside the primary tuning objective.
