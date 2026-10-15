# Phase 13 locked experiment protocol

## Research question and status
This retrospective sensitivity experiment tests whether Phase 6 BTC/ETH 24-hour Amihud MLP and GRU forecasts depend on omitting training-target standardization, and whether the SpotPlusDerivatives versus SpotOnly point pattern persists. Phase 6 final-test outcomes have already been inspected; this is not independent confirmation and cannot replace the frozen 2/18 task decision.

## Frozen scope
Only BTC and ETH, target `future_amihud`, six four-hour future bars, and the original Phase 6 `SPOT_ONLY` / `SPOT_PLUS_ALL_DERIVATIVES` information sets are included. Exactly the development-selected model configuration for each asset × information set × neural family is used. The source frozen registry is recorded in `01_frozen_configuration_registry.json`. No model selection, tuning, new features, market data, target redefinition, architecture change, altered split, or new seed is allowed.

## Data, target, and timing
Use only `outputs/phase5_dynamic_discovery/dataset_4h.jsonl` after verifying its frozen Phase 6 source SHA-256. The target is the existing Phase 6 mean Amihud over t+1,…,t+6, transformed as `log1p(max(y,0))`. Six input rows t−5,…,t are ordered oldest to newest and use the original complete-case rule. Final test origins are 2023-07-01 00:00 through 2024-03-03 20:00 UTC. Training labels must end strictly before the relevant validation/test boundary. Development folds are the three frozen expanding windows in the source registry; final training uses only labels ending strictly before 2023-07-01 00:00 UTC.

## Single intervention and matched replay
For each fitting unit, the treatment fits `StandardScaler` to that unit’s eligible training `y_log` values only, trains on `(y_log−mu_train)/sigma_train`, and inverse-transforms predictions using the same training-only `mu_train` and population `sigma_train`. Degenerate sigma is a recorded fitting failure. The control replay fits the same selected architecture on unscaled `y_log`. Both branches use identical rows, predictors, predictor scaling, architecture, optimizer, training schedule, fixed seed, and raw/unclipped log-scale forecast scoring. Neither branch truncates negative model forecasts or clips to the observed target range. For audit comparability, the control replay also records the original Phase 6 lower-zero prediction transform as a separate derived score; this transformed score is not the paired scaling contrast.

## Models and seeds
MLP and GRU only. MLP settings, including original selected width/config, Adam, alpha 0.001, learning rate 0.001, max 400 iterations, no early stopping, shuffle, tolerance 1e−5, and no-iteration-change 30, remain fixed. GRU settings, including original selected hidden width, single GRU layer + linear output, sequence length six, Adam lr 0.005, weight decay 0.001, full batch, 20 epochs, and no early stopping/gradient clipping, remain fixed. Development and final seeds follow the exact deterministic Phase 6 formulas, recorded in the configuration registry.

## Outcomes
Primary metric is MAE on the original `log1p` target scale, after inverse target-scaler transformation. Report historical saved MAE, unscaled replay MAE (raw output), target-scaled replay MAE, and separately the frozen Phase 6 clipped-output control replay. Also report relative change, training-mean, persistence, OLS, Ridge, RBF-SVR, ExtraTrees, HistGradientBoosting, test count, prediction distribution, fold metrics, warnings/convergence, training loss curves, and deterministic ensemble/category sensitivity. A test-selected strongest classical reference is explicitly an oracle-descriptive comparison.

## Retrospective uncertainty
If prediction series are complete and matched, compute paired moving-block bootstrap intervals with fixed seed, 1,000 draws, block lengths 42 and 84. These are post-hoc exploratory intervals only. No new frozen PASS/FAIL status is created, and no inference supersedes the registered Phase 6 correction or decision.

## Prespecified diagnostics
An extreme absolute error is counted descriptively when it exceeds ten times the maximum eligible training `y_log` for that fit. Out-of-observed-range counts use the evaluation subset's observed target minimum and maximum and remain descriptive only.

## Prespecified descriptive classifications
Scaling response is CLEAR_ABSOLUTE_IMPROVEMENT if scaled MAE is below historical saved MAE in all four asset × family cells; MIXED_IMPROVEMENT if in one to three; NO_MEANINGFUL_IMPROVEMENT if in none; any failed fit is separately reported. Competitiveness is COMPETITIVE_WITH_STRONG_CLASSICAL only if every scaled neural cell beats its within-information-set best saved classical test-MAE oracle; COMPETITIVE_WITH_SIMPLE_BASELINE_ONLY if all four beat training-mean but the stronger-classical criterion fails; STILL_INFERIOR_TO_SIMPLE_BASELINE if all four are worse than training-mean; otherwise MIXED_ACROSS_ASSETS. Derivatives point pattern PERSISTS only if all four scaled S0-minus-S1 MAE differences are positive; DISAPPEARS if all are zero within 1e−15; REVERSES if all are negative; otherwise MIXED_ACROSS_ASSETS. Ensemble is SUBSTANTIALLY_IMPROVED if all four asset × information-set reconstructed ensembles lower MAE by at least 50% versus the saved ensemble; PARTIALLY_IMPROVED if one to three cells improve or all improve by less than 50%; UNCHANGED if all differences are within 1e−15; WORSENED if all four increase; otherwise MIXED (reported under PARTIALLY_IMPROVED with the cell pattern stated). These are descriptive conventions frozen before fitting, not inferential decision rules.

## Integrity and limits
Hash protocol/config/training plan/source manifest before any fit. Hash all outputs and recheck source hashes after the run. Preserve negative and failed fits. No manuscript is edited. Report any failure or reproduction discrepancy. This experiment can diagnose target-scaling sensitivity in this previously inspected test sample; it cannot establish independent generalization, causality, trading value, or change the historical Phase 6 decision.
