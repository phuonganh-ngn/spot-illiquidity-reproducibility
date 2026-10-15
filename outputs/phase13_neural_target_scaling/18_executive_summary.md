# Phase 13 executive summary

**Status:** COMPLETE. This is a retrospective target-scaling sensitivity on the already-inspected Phase 6 test period. It did not change any frozen Phase 6 result or protocol.

The run used only the originally development-selected BTC/ETH 24-hour Amihud MLP and GRU configurations, both frozen information sets, original folds/test origins, and exact Phase 6 seed formulas. The treatment fit target StandardScaler on training labels only, inverse-transformed outputs, and scored all forecasts on the original log1p target scale. The unscaled paired replay and original saved outputs are separately reported.

The historical Phase 6 result remains **2/18 tasks passed**. Historical category-average reductions were 18.5648% for BTC and 13.6448% for ETH; their exact reconstruction is checked before reporting retrospective category values. The target-scaled category sensitivity is BTC -0.9744% and ETH 0.8081% (retrospective; not a new registered result).

Prespecified descriptive classifications:
NEURAL_SCALING_RESPONSE: CLEAR_ABSOLUTE_IMPROVEMENT
NEURAL_COMPETITIVENESS: MIXED_ACROSS_ASSETS
DERIVATIVES_INFORMATION_SENSITIVITY: MIXED_ACROSS_ASSETS
ENSEMBLE_RELIABILITY: SUBSTANTIALLY_IMPROVED

See `05_mlp_sensitivity_results.csv`, `06_gru_sensitivity_results.csv`, `07_neural_prediction_diagnostics.csv`, `09_strong_baseline_comparison.csv`, `10_ensemble_sensitivity.csv`, `11_category_average_sensitivity.csv`, and `12_exploratory_uncertainty.md`. All Phase 6, Phase 9, Phase 12, certified panel, code and manuscript hashes are rechecked in `16_source_manifest_after.json`.

No manuscript was edited, no market data were downloaded, and no causal, independent-confirmation, trading-value, or quantum-advantage claim is supported by this sensitivity.
