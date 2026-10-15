# Phase 14 scientific interpretation

This is a **retrospective reanalysis**, not a fresh holdout. The Phase 6 final test period was examined during earlier phases. The new procedure protects model selection within Phase 14 from direct test tuning, but does not erase that historical exposure.

## Primary results

BTC: development chose HistGradientBoosting for S0 (dev MAE 4.4486613e-06) and HistGradientBoosting for S1 (dev MAE 4.7281103e-06). Test MAE was 6.1228799e-06 vs 6.0009129e-06; Delta_best=1.2196702e-07 (1.992%). Primary 42-bar CI [-1.436159e-08,2.4133173e-07], raw p=0.06793, Holm p=0.06793; result INCONCLUSIVE.

ETH: development chose Ridge for S0 (dev MAE 1.4685191e-05) and ExtraTrees for S1 (dev MAE 1.5276743e-05). Test MAE was 1.4906541e-05 vs 1.414604e-05; Delta_best=7.6050104e-07 (5.102%). Primary 42-bar CI [1.8482546e-07,1.1749992e-06], raw p=0.004995, Holm p=0.00999; result FAVORABLE.

Development selected BTC HGB for both information sets, while ETH selected Ridge for SpotOnly and ExtraTrees for SpotPlusDerivatives. ETH’s selected S1 system was not better in pooled development validation than the selected S0 model, yet had lower retrospective test MAE. That temporal reversal is material and argues against presenting the test result as an established transferable gain. BTC’s selected systems are both HGB; the observed test reduction is small and its 42-bar interval includes zero.

Across within-family comparisons, derivatives point estimates are mixed: the tree/boosting families often improve, while neural and some linear/tree comparisons worsen; XGBoost closely reproduces the training-mean reference under this frozen target scale/grid. The within-family comparisons are descriptive and are not the primary endpoint. The primary endpoint compares independently development-selected S0 and S1 systems.

Target-scaled MLP/GRU predictions are numerically reasonable relative to Phase 6, but none beats the strongest test-period classical tree/boosting reference in either information set. MLP fits hit their frozen 400-iteration limit without convergence in all four final asset/information cells. GRU fits used the fixed 20 epochs. No test-directed change was made.

The results do not restore the historical category-average interpretation: Phase 6’s +18.5648% BTC and +13.6448% ETH category averages, Phase 13’s −0.9744% and +0.8081% sensitivity averages, and Phase 14’s selected-system test deltas are different estimands. The Phase 14 ETH signal is a favorable retrospective, development-selected comparison; BTC remains inconclusive. Phase 7–8’s broad generalization failure remains intact, and Phase 9’s `NO_EXTERNAL_QML_REPLICATION` is untouched. No causal, trading-value, universal derivatives-information, or quantum-advantage conclusion follows.

No further experiment is required to answer the current retrospective benchmark question. A future claim of generalizable incremental information requires a prospectively frozen evaluation on genuinely unseen data.
