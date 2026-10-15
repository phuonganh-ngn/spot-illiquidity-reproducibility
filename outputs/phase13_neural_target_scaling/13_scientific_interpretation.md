# Phase 13 scientific interpretation

This is a retrospective sensitivity analysis on Phase 6 data whose final period was previously inspected. It cannot establish independent confirmation or revise the frozen Phase 6 2/18 result.

## Main results
### BTC 24-hour Amihud
MLP: historical Phase 6 S0/S1 MAE = 0.035047456 / 0.014367342; unscaled replay raw S0/S1 = 0.043807385 / 0.021793846; target-scaled S0/S1 = 7.8090769e-06 / 7.9756417e-06; scaled Δ(S0−S1) = -1.6656484e-07 (positive favors derivatives). Target-scaled forecasts did not beat the strongest saved classical test oracle under S0 and did not beat under S1.
GRU: historical Phase 6 S0/S1 MAE = 0.034905188 / 0.015861983; unscaled replay raw S0/S1 = 0.038343403 / 0.030465863; target-scaled S0/S1 = 6.9201742e-06 / 7.3903388e-06; scaled Δ(S0−S1) = -4.7016457e-07 (positive favors derivatives). Target-scaled forecasts did not beat the strongest saved classical test oracle under S0 and did not beat under S1.

### ETH 24-hour Amihud
MLP: historical Phase 6 S0/S1 MAE = 0.0094316601 / 0.0079834132; unscaled replay raw S0/S1 = 0.017565074 / 0.021198425; target-scaled S0/S1 = 1.5506742e-05 / 1.5745954e-05; scaled Δ(S0−S1) = -2.3921163e-07 (positive favors derivatives). Target-scaled forecasts did not beat the strongest saved classical test oracle under S0 and did not beat under S1.
GRU: historical Phase 6 S0/S1 MAE = 0.0053601748 / 0.00091320479; unscaled replay raw S0/S1 = 0.011599751 / 0.02489445; target-scaled S0/S1 = 1.4617092e-05 / 1.45624e-05; scaled Δ(S0−S1) = 5.4691888e-08 (positive favors derivatives). Target-scaled forecasts did not beat the strongest saved classical test oracle under S0 and did not beat under S1.

## Prespecified descriptive classifications

- NEURAL_SCALING_RESPONSE: **CLEAR_ABSOLUTE_IMPROVEMENT**
- NEURAL_COMPETITIVENESS: **MIXED_ACROSS_ASSETS**
- DERIVATIVES_INFORMATION_SENSITIVITY: **MIXED_ACROSS_ASSETS**
- ENSEMBLE_RELIABILITY: **SUBSTANTIALLY_IMPROVED**

The complete figures and metrics are reported in the machine-readable result files. The original equal-weight ensemble remains frozen; the component replacement is a retrospective sensitivity only.

## Answers to research questions

RQ1–RQ2. Compare historical forecasts with target-scaled forecasts and the matched unscaled replay in `05_mlp_sensitivity_results.csv`, `06_gru_sensitivity_results.csv`, and `07_neural_prediction_diagnostics.csv`. A reduction in extreme forecasts is descriptive, not evidence of incremental derivatives information.
RQ3. Compare target-scaled neural test MAEs with training-mean, persistence, and unchanged classical forecasts on the same origins in `09_strong_baseline_comparison.csv`. The family labeled best classical is a retrospective test oracle, not deployable selection.
RQ4. The target-scaled derivatives point pattern is summarized by each model’s matched S0−S1 difference. These are point estimates; bootstrap intervals are post-hoc exploratory and do not update the frozen decision.
RQ5. The historical category means are reconstructed exactly from retained predictions before the sensitivity version is calculated. Updated averages remain retrospective and reflect changes to the neural and ensemble categories only.
RQ6. The ensemble sensitivity retains six equal weights and changes only the two neural component forecasts. The historical ensemble reconstruction check is available in `10_ensemble_sensitivity.csv`.
RQ7. The original Phase 6 positive task-level record remains 2/18 and is not replaced. Neural scaling can explain sensitivity of the neural category contribution; it does not establish broad or operational forecasting superiority.
RQ8. Add a clearly labeled retrospective subsection and retain original Phase 6 results in their own table. If the category statistic changes materially, revise abstract/conclusion wording so it is not presented as robust deployable gain.
RQ9. This fixed sensitivity addresses the registered target-scaling concern. Do not add more neural architectures or search scales based on these results; another experiment is not indicated unless an integrity or unresolved implementation discrepancy is found.
