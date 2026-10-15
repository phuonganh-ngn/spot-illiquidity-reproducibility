# Limitations and integrity

- This is a retrospective sensitivity run after the Phase 6 final test was inspected. It is not independent confirmation and cannot change the original frozen task-level decision (2/18).
- Only BTC and ETH 24-hour Amihud are re-estimated; no inference extends to other horizons or targets.
- The development-selected architectures and configurations are preserved; this is not a neural model search.
- StandardScaler uses training labels only. Negative forecast outputs are retained on the `log1p` prediction scale as required. The separate control replay’s lower-zero transform is reported only to compare the historic implementation.
- Any block-bootstrap intervals are nominal post-hoc exploratory intervals and do not constitute new confirmatory significance tests.
- Best classical reference based on final-test MAE is an oracle-descriptive comparison; operational model selection still requires development selection.
- No causal effect, market-wide transmission, execution-cost improvement, trading profitability, or quantum computational advantage is established.
- Source hashes and output checks are reported in the final manifest.
