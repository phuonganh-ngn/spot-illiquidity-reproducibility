# Phase-7 protocol amendment 001: feature tensor layout

This amendment was written before development fitting and before external outcome evaluation. It clarifies implementation details without changing sample dates, targets, estimators, model families, prediction timing, scoring, tuning boundaries or decision criteria.

For every information set, the base variables form a six-step chronological input sequence ordered t-5, t-4, ..., t. The legacy Spot-only sequence uses the six frozen spot variables; the matched legacy Spot-plus-Derivatives sequence adds the four frozen derivatives variables at each of those six steps.

In enhanced information sets, UTC-slot and weekday indicators are static features measured at t and are appended to the flattened sequence for tabular learners. For Enhanced Spot-plus-Derivatives, the eight already-defined change/shock/interaction variables are also static values measured at t. Their underlying rolling denominators use only observations through t-1. No static covariate uses information after the close of interval t.

For the GRU, only the six-step base-variable sequence is processed recurrently. The same static features are passed to its final prediction head alongside the last hidden state. This ensures that all model families receive the same underlying information set while preserving the intended sequence architecture.
