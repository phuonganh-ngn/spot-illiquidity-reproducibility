# Manuscript revision recommendations (do not edit manuscript in Phase 13)

## Abstract
Keep the original Phase 6 result and its scope explicit. If the target-scaled sensitivity changes the neural/category macro-average materially, state that Phase 6 averages were sensitive to neural training scale and that only 2/18 original tasks met the frozen rule. Do not frame these retrospective results as confirmation or generalizable derivatives value.

## Section 5.1 — Matched Multi-Model Forecasting
Add a short retrospective-sensitivity paragraph explaining the training-only target standardization intervention, unchanged selected MLP/GRU configurations, matched controls, unchanged splits/seeds, inverse transformation, and original log1p-scale scoring. Preserve the frozen original results separately.

## Section 6.1 — Historical Multi-Model Forecasting Benchmark
Retain the historical table and 2/18 registered outcome. Add a distinct subsection titled “Retrospective Neural Target-Scaling Sensitivity.” Report all BTC/ETH × MLP/GRU × S0/S1 absolute MAEs, matched unscaled replay, fold behavior, and the unchanged mean/tree references. Distinguish historical and sensitivity ensembles/category averages.

## Section 6.2 — Generalization Across Assets and Liquidity Proxies
No Phase 13 result changes Phase 7–8. Keep the failed broad generalization rule, mixed proxy and temporal results, and the inspected/reused status of Binance periods.

## Section 7 — Discussion
Explain that `log1p` is approximately identity at Amihud values around 10⁻⁵ and that target scaling is a neural conditioning choice. Interpret any improvement as sensitivity of model fitting, not new evidence of market transmission. Separate favorable point estimates from baseline dominance and uncertainty.

## Section 8 — Limitations
State the diagnostic is retrospective on an already-inspected test period; no independent confirmation, causal identification, execution-cost savings or trading profit follows. Retain the negative Bybit result and missing stronger Bybit classical comparator. Do not conflate Phase 6 MLP/GRU with the separate QML residual pipeline.

## Section 9 — Conclusion
Keep the conclusion calibrated to the frozen overall record. If the positive category average weakens/disappears after target scaling, explicitly describe it as model-conditioning sensitive and not robust against the strongest saved classical baseline. Phase 9 remains `NO_EXTERNAL_QML_REPLICATION`.
