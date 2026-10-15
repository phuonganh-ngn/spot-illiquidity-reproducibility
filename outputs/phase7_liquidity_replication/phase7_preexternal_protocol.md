# Phase 7 pre-external validation protocol

This protocol tests whether the Stage-6 finding that derivatives features add predictive information for 24-hour future spot Amihud illiquidity generalizes beyond BTC/ETH and beyond the original Amihud construction. It is a forecasting and measurement-validation exercise, not causal identification. The protocol is written before development cross-validation and before any post-2024 outcome values or forecast scores are examined.

## Protected evidence and samples

Stage 1, Stage 3, the original Phase-5 analysis, Stage 6 and their frozen inputs/results remain unchanged. BTC and ETH are checked against the immutable Phase-5 4-hour records. The new 4-hour panel uses only cached raw archives and adds SOL, BNB and XRP, selected solely by the previously frozen outcome-blind data-eligibility rules. No asset is selected using target distributions, correlations, forecasts or model results.

The sole development sample is 2021-12-01 00:00 through 2023-06-30 20:00 UTC, matching the prior Stage-6 development boundary. The pre-existing Stage-6 final-test interval, 2023-07-01 through 2024-03-03, is not used for Phase-7 feature, transformation or hyperparameter selection because its outcomes have already been reported. The new external evaluation begins 2024-03-04 00:00 UTC, after the verified archive/timestamp break, and ends with labels whose last 4-hour interval is no later than 2026-09-30 20:00 UTC. A 24-hour target at feature time t uses only intervals t+1 through t+6. The six terminal external feature rows are ineligible when their future target would exceed cached raw history.

The external period is evaluated once after the fitted feature/model specification is frozen and hashed. No new market data are downloaded. No result from the external sample may change model, feature, transform, threshold or asset choices.

## Outcomes and measurement

All outcomes cover the same 24-hour future window, six 4-hour intervals beginning strictly after t:

1. `future_amihud_24h_mean`: mean of the six 4-hour values `1e6 * abs(log(close_t/close_{t-1})) / spot_quote_volume_USDT_t`. Higher means greater measured price impact per USDT of turnover and therefore greater illiquidity. This reproduces the Stage-6 target definition.
2. `future_kyle_abs_lambda_24h`: absolute OLS slope with intercept of 288 future 5-minute log returns on signed Binance spot quote flow (`2*taker_buy_quote_volume - quote_volume`). This is a signed-flow price-impact proxy, not a quoted spread.
3. `future_corwin_schultz_spread_24h`: mean of five adjacent-4-hour-bar Corwin-Schultz high-low spread estimates entirely within the future window. This is a range-based spread proxy; it is not a transaction-level quoted spread and may reflect volatility as well as trading costs.

Amihud, Kyle price impact and Corwin-Schultz are kept as distinct measures. No one proxy is renamed as another or substituted when missing. A model is scored only on rows where its target and the common comparison feature set are observed; Spot-only and Spot-plus-Derivatives use the identical row set for each asset/target/specification.

The development-only transform candidates are raw nonnegative target, the original Stage-6 `log1p(y)`, and dimensionless `log1p(y/s)` where s is the training-fold median of positive y. Predictions are inverted to original units before operational-scale scoring. A transform is chosen per asset/outcome using pooled development expanding-fold validation loss averaged across the prespecified independent model families and both information sets; it is then shared by both information sets and frozen. Folds estimate s from their training portion only. Negative inverse predictions are clipped at zero only for nonnegative outcomes; this rule applies identically in development and external scoring.

## Information sets

All raw market features end at the close of 4-hour interval t and use at most six intervals, t-5 through t. They are constructed independently by asset.

The legacy Spot-only set contains the six frozen Phase-6 spot series: spot return, realized volatility, downside semivariance, Amihud, spot taker imbalance, and quote-volume growth. The matched legacy Spot-plus-Derivatives set adds the four frozen derivatives series: funding state, OI log change, futures taker imbalance, and basis mean. This is the direct out-of-period/multi-asset comparison to the prior benchmark.

The enhanced Spot-only set adds only the predefined UTC-slot and weekday indicators to the same six-lag spot history. The matched enhanced Spot-plus-Derivatives set further adds the already-defined backward-looking funding/basis changes, 180-interval past-only OI and flow shocks, and four prespecified derivatives interactions. No future or centered rolling variables are eligible. All transformations and feature rules are fixed before external evaluation. Feature scaling, where used, is fit on each training fold only. Rows with missing predictors are dropped, never imputed.

## Development design and model families

Three expanding-origin development folds are fixed: train through 2022-09-30, validate 2022-10-01 through 2022-12-31; train through 2022-12-31, validate 2023-01-01 through 2023-03-31; and train through 2023-03-31, validate 2023-04-01 through 2023-06-30. Training rows are purged whenever their six-step target window reaches the validation block. Validation rows are scored only when all six future 4-hour outcome intervals are inside the corresponding development period. These are date-ordered folds; no random split is used.

The fixed model set is a training-mean benchmark, persistence benchmark, Ridge, RBF-SVR, ExtraTrees, HistGradientBoosting, shallow MLP, and compact GRU. The six learned models span five independent family categories: linear, kernel, bagged trees, boosting, and neural (MLP/GRU are one category). The exact small configuration grids and seeds are recorded in `phase7_protocol.json`; tuning is limited to development folds. There is no ensemble because its predictions are not an independent model family and Phase 6 already included an equal-weight ensemble.

For each learned model, asset, outcome, transform and information set, choose the configuration with lowest pooled development OOF MAE on the selected transformed target. Ties are resolved by simpler configuration order. Report development-selected configurations and fold scores, but treat them only as tuning diagnostics. Refit once on eligible development rows and evaluate the new external period once.

## External scoring and uncertainty

Primary accuracy is MAE on the frozen development-selected transformed target. Secondary measures are MAE/RMSE on original units and, for Amihud, stress-event warning metrics. Incremental value is the paired loss difference `loss(Spot+Derivatives) - loss(Spot-only)` on exactly matched asset/date rows; negative values favor the added derivatives information.

Paired uncertainty uses a moving date-block bootstrap of 2,000 replications, seed 724051, with 42 consecutive 4-hour intervals per block. The same sampled date indices are used for both information sets and all assets at a given date. Report percentile confidence intervals and the bootstrap probability that incremental loss is below zero. Holm correction is applied across the six learned families within each outcome, with the five assets retained as distinct replications and also shown separately; no causal p-value language is used. Family counts collapse MLP and GRU into one neural category.

The stress warning threshold for each asset/outcome is fixed at the 90th percentile of the development target. The forecast alarm cutoff is the 90th percentile of development OOF predictions. Report external precision, recall, F1 and lift over event prevalence; thresholds are not recalibrated on external labels.

Temporal stability is reported by fixed external calendar segments: 2024-03-04 through 2024-12-31, calendar 2025, and 2026-01-01 through 2026-09-30. These are descriptive comparisons, not break tests. Asset replication is reported separately and summarized only when sign and metric behavior agree; no asset is removed after external evaluation.

## Negative controls and interpretation

An impossible-time diagnostic uses future derivatives values after the 24-hour target window to predict the earlier target. It is reported only as a timing/persistence diagnostic, never as a deployable predictor and never to select models. A second check compares derivative incremental value after a fixed 42-interval circular displacement, with the displacement seeded and fixed before external scoring; this is a placebo sensitivity, not the primary estimate. Strong placebos weaken interpretation of incremental value.

All claims are predictive. Results do not identify causal effects or prove that derivatives markets transmit liquidity shocks. Same-venue Binance spot and futures share traders, infrastructure and USDT settlement; this can create mechanical co-movement. Kyle and Corwin-Schultz remain proxies with distinct measurement error. Post-2024 OI is used only under the fixed piecewise timestamp map established by the outcome-blind data audit; failed OI days remain missing.

## Frozen decision criteria

Call the Stage-6 Amihud forecasting finding externally replicated only if derivatives reduce external transformed MAE in both BTC and ETH, improve in at least four of five eligible assets, and the paired 42-interval bootstrap intervals are compatible with improvement for BTC and ETH without a material external-period sign reversal. Call the result generalizable across liquidity measurement only if the same direction is supported by at least one distinct proxy (Kyle or Corwin-Schultz) in BTC and ETH and is not driven by one learned model category. If only Amihud improves, classify it as Amihud-specific predictive value. If families or assets conflict, report model/asset sensitivity. No threshold may be relaxed after external outcomes are read.

This study does not run causal estimators, hypothesis regressions, Granger tests, Transfer Entropy, HMM, GAN, DRL, trading strategies or portfolio optimization.
