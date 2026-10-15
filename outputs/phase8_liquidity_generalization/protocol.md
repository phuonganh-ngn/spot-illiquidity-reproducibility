# Phase 8 — Liquidity-signal validation and generalization audit

## Status and scope

This phase audits and consolidates the already executed Phase-7 out-of-period evaluation of the Phase-6 24-hour future Amihud finding, then adds matched secondary diagnostics that were missing from the earlier report. It is predictive validation, not causal identification. Stage 3, Phase 5, Phase 6, and Phase 7 artifacts are immutable inputs.

The post-2024 period and its forecast scores have already been inspected in Phase 7. This is therefore a reproducible reanalysis of previously inspected external data, not a fresh or untouched confirmation. No new market data will be downloaded. No asset, target, proxy, feature, model family, transform, sample boundary, threshold, or selection rule will be changed in response to the existing external scores.

## Frozen source design reused from Phase 7

Use the existing cached Binance 4-hour UTC panel, SHA-256 `5a339c5c4a8778ceb74344a10c67806c2323ffac80556be18bf7a9c99de8dcf6`, and the already frozen Phase-7 development and external protocols. The sole primary outcome remains mean 24-hour future Amihud, over intervals `t+1` through `t+6`. The external feature-date range remains 2024-03-04 00:00 UTC through 2026-09-29 20:00 UTC, with labels ending no later than 2026-09-30 20:00 UTC.

The fixed asset set is BTC, ETH, SOL, BNB, and XRP. SOL, BNB, and XRP were mechanically retained under the outcome-blind, pre-estimation coverage and timestamp-semantic eligibility protocol. No additional assets are eligible in this phase.

The matched information-set comparison uses the frozen legacy six-lag Spot Only features and the same six spot features plus the four frozen derivatives features. The enhanced calendar/innovation feature pair remains a separate robustness comparison. Frozen families are Ridge, RBF-SVR, ExtraTrees, HistGradientBoosting, MLP, and GRU; MLP and GRU remain one neural category for equal-category aggregation. Development-selected target transforms and model configurations are reused. No external retuning is permitted.

## Measurement rules

Amihud remains `1e6 × |4-hour log return| / 4-hour Binance spot quote volume in USDT`; larger values indicate greater price impact per turnover and therefore measured illiquidity. Recheck the stored series against its certified reconstruction and preserve zero/missing/tail behavior.

The two distinct 24-hour outcomes already built from cached raw data remain separate measurement checks: (1) absolute OLS price-impact slope of 288 subsequent 5-minute log returns on signed spot quote flow, and (2) mean of five adjacent-4-hour Corwin–Schultz high-low estimates wholly inside the future window. Neither proxy is relabeled as a quoted spread or substituted for Amihud. Reuse the existing archive-integrity and offline-rebuild records; do not rebuild or redownload raw market data in this audit.

## New fixed calculations

1. **Development-selected best-family dominance.** Within each asset, target, and legacy information set, select the learned family with the lowest development-only pooled transformed OOF MAE in the existing frozen development output. Compare those two frozen family forecasts on identical external dates. Report transformed MAE and the paired difference `loss(Spot+Derivatives) - loss(Spot Only)` with paired moving-date-block bootstrap intervals at 42 and 84 four-hour intervals. No family is selected using external scores.
2. **Matched impossible-time controls.** For Amihud only, fit fixed `Ridge(alpha=1)` models with training-only standard scaling on eligible development observations. Compare a Spot Only model using six spot lags through `t` with a matched model adding the legacy derivatives history shifted so its source rows are exactly `t+7…t+12` (after the target window), and separately a 42-interval forward displacement with no wraparound. Both models use the same development and external dates, target transform selected in Phase 7, and transformed MAE. Report paired `future-derivatives model loss - Spot Only loss`; a supported negative value is a temporal-contamination warning, not forward predictive evidence. Bootstrap complete date blocks at lengths 42 and 84, seed 724051.
3. **Stress-warning utility.** Recompute the frozen q90 event and development-OOF alarm rules for the existing Amihud forecasts. Report precision, recall, F1, and lift. For each asset and feature pair, compare Spot+Derivatives minus Spot Only after averaging MLP/GRU within the frozen neural category and weighting the five categories equally. Use paired 42/84-interval date-block bootstrap for uncertainty. Thresholds remain fixed at their development values.
4. **Temporal stability and cross-asset replication.** Reuse the three frozen external calendar segments (2024-03–12, 2025, 2026 partial) and all five mechanically eligible assets. Preserve the original Phase-7 all-conditions Amihud and cross-proxy decision rules exactly. Do not promote a finding based on one asset, model family, proxy, or segment.

The paired bootstrap uses 2,000 replications, deterministic seeds derived from 724051, and resamples complete UTC date blocks. Missing paired observations are never filled. For diagnostics, a 95% interval excluding zero is reported as uncertainty evidence only; it does not establish causal identification.

## Final classification rule

The Phase-6 Amihud forecasting finding is classified using the frozen Phase-7 rule. Cross-measure generalization requires the frozen BTC/ETH and model-category criteria for at least one of Kyle or Corwin–Schultz. This phase additionally reports whether a development-selected Spot+Derivatives model dominates the development-selected Spot Only model and whether impossible-time controls flag temporal dependence. The overall result must retain the Phase-7 external-replication classification; best-family wins, stress-warning metrics, or exploratory diagnostics cannot override a failed replication rule.

Allowed conclusion language is limited to predictive value, forecast performance, stress-warning utility, temporal stability, and proxy/asset generalization. No causal, structural, or universal derivatives-predictability claim is permitted.
