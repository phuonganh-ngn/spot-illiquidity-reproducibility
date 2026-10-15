# Phase 14 limitations and integrity

- Evidence type: retrospective reanalysis; the historical Phase 6 test period was previously viewed.
- Model selection used only pooled Phase 14 F1–F3 development predictions; the selected model registry was SHA-256 frozen before the test panel was parsed.
- The primary interval remains the historical Phase 6 test calendar; 1,417 matching origins per asset are scored, ending at 2024-03-02 20:00 UTC because the next six-bar target must be observed before the frozen sample end.
- The target constructor, 4-hour rows, six-step history, S0/S1 feature sets and target-window purges come from the frozen Phase 6 code. No feature was added and no data were downloaded.
- MLP and GRU used train-only target scaling. Every Phase 14 model was scored without output clipping on the same log1p scale.
- MLP reached its fixed 400-iteration maximum without convergence in all four final asset/information-set fits. No retry or extension was made. XGBoost forecasts were effectively at the training-mean level under the locked grid; this is reported as a limitation, not tuned away.
- Bootstrap intervals are nominal, not Holm-adjusted. Holm applies to the two primary BTC/ETH raw p-values separately for each block length. These retrospective p-values do not provide independent confirmation.
- Phase 6/7/8/9/10/12/13 artifacts, the certified data panel, original Phase 6 source code, and the manuscript remain unchanged by the source-hash audit. No QML model or Bybit result was touched.
- No causal effect, execution-cost saving, trading profit, broad external validity, or quantum computational advantage is established.
