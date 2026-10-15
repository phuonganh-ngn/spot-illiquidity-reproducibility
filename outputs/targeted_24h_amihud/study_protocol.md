# Targeted 24-hour Amihud study protocol

## Research question and claim scope

This study tests whether the joint state of BTC/ETH perpetual futures markets adds predictive information for the next 24 hours of Binance spot Amihud beyond a matched spot-only information set. It is predictive, not causal. Results from Binance 2023–2026 are historically reused because those dates were examined in Phases 6–7; they are exploratory rather than fresh confirmation. Stage 3 and Phases 5–7 remain unchanged.

## Data, target, and timing

Use only the certified 4-hour UTC Phase-5/Phase-7 panels, BTC and ETH, and no new downloads. The sole target is the arithmetic mean of spot Amihud over intervals t+1 through t+6. Spot Amihud is 1e6 times the absolute 4-hour spot log return divided by 4-hour Binance spot quote volume in USDT. Inputs comprise intervals t-5 through t; interval-t inputs become available only at its end. The future label begins at t+1 and requires all six intervals. S0 and S1 use identical complete-case rows. S0 contains the six frozen spot variables, six lags, and the same calendar controls as S1. S1 adds six lags of the four frozen derivatives variables, fixed backward-looking derivative innovations, and four prespecified economic interactions. No imputation or winsorization is used.

## Amihud transform and feature construction

Re-audit the numerator, quote-volume denominator, units, zeros, near-zero behavior, target-window timestamps, tails, and numerical scale before fitting models. Compare four target transforms: raw y; log1p(y); log(y+1e-8); and log(10000*y+1e-8). Epsilon=1e-8 and c=10000 are fixed from the certified 1e6 Amihud unit scaling and numerical stability, not selected on the evaluation period. Select one shared BTC/ETH transform using fixed Ridge(alpha=1), training-fold-only StandardScaler, and expanding validation folds within 2021-12 through 2022-09. Rank transformations by mean asset/fold MAE divided by training-fold transformed-target IQR; tie order is T1,T2,T3,T4. Freeze the selected transform before any main model-family or S0/S1 comparison.

Base histories are t-5..t. OIShock and FlowShock use the current observation minus the median of exactly 180 preceding four-hour values, divided by 1.4826 times their preceding-window MAD; zero MAD yields missing. FundingChange and BasisChange are one-step differences. Only Funding×OIShock, OIShock×FlowShock, Funding×Basis, and FlowShock×Basis are added. Calendar controls are five UTC-slot indicators (slot 0 reference), six weekday indicators (Monday reference), and a fixed funding-settlement-window indicator for interval starts at 00:00, 08:00, or 16:00 UTC.

## Samples and validation

The development period is 2021-12-01 through 2023-06-30 20:00 UTC. It has been used in earlier project work and is not pristine. Nested expanding walk-forward outer blocks are: 2022-10-01–2022-12-31; 2023-01-01–2023-03-31; and 2023-04-01–2023-06-30. For each outer block, tune only on three expanding inner folds formed by dividing the 180 calendar days preceding the block into consecutive 60-day validation windows. Purge at least six 4-hour intervals at fold boundaries, require all target labels to remain inside each validation block, and never randomly shuffle dates.

The 2023-07–2024-03 Stage-6 period and the 2024-03–2026-09 Phase-7 period can be scored only as EXPLORATORY_REUSE. Train final selected models on development data only. No period currently available is a fresh external confirmation.

## Models and tuning

Nine classical families are frozen: ElasticNet; RBF-SVR; ExtraTrees; HistGradientBoosting; XGBoost; LightGBM; CatBoost; compact MLP; and a compact GRU. Each gets exactly 20 predefined configurations, shared between S0 and S1. For linear, kernel, and neural families, StandardScaler and RobustScaler are compared with parameters fitted inside each training fold. Tree/boosting models use unscaled features. The GRU receives a real six-step sequence plus current static/calendar features. Selection uses transformed-target MAE on inner folds only; outer predictions are paired.

## Residual learning and QML

Generate strict forward OOF predictions from the development-selected strongest spot-only model, then construct observed-minus-predicted residuals. Train derivatives-only residual learners using chronological folds; no in-sample spot residual is permitted. Compare regularized linear, RBF, boosting, MLP, quantum kernel, QNN, and compact hybrid QNN.

QML uses PennyLane CPU simulation, fixed angle encoding, shallow circuits, training-only scaling/PCA to 4/6/8 dimensions, and five frozen seeds for QNN/hybrid methods. Quantum kernels use a deterministic 128-landmark Nyström approximation and are matched to RBF models on the same compressed rows/features/landmarks. Validate each implementation on frozen linear, nonlinear-interaction, and null synthetic fixtures before empirical use. No hardware or quantum-advantage claim is made.

## Inference, multiplicity, controls, and regimes

Primary loss is MAE on the frozen transformed target; report original-scale MAE, RMSE, median absolute error, and rank correlation as secondary metrics. Use paired moving-block bootstrap with 1,000 replications, block lengths 42 and 84 four-hour intervals, and seed 20261007. Use six-lag Bartlett HAC forecast-loss tests as supporting evidence. Apply Holm within each asset across the nine family-level direct comparisons; define separate prespecified residual and QML families. Report model-family results and best-model dominance separately; all best-model selection uses development data. Construct a derivative-only future-information negative control at t+7..t+12 and compare against the same SpotOnly benchmark. Regime thresholds are learned on development only and are exploratory. Component/drop-one ablations use development-selected parameters without retuning.

## Decision rules

Use the specified labels NO_INCREMENTAL_INFORMATION, LINEAR_INCREMENTAL_INFORMATION, NONLINEAR_INCREMENTAL_INFORMATION, RESIDUAL_INCREMENTAL_INFORMATION, JOINT_DERIVATIVES_STATE_INFORMATION, MODEL_SENSITIVE_SIGNAL, QUANTUM_INCREMENTAL_VALUE, and INCONCLUSIVE. A robust incremental finding requires a positive median family delta, improvement in at least three independent families, at least two Holm-adjusted paired-bootstrap supports, at least one non-neural supporting family, residual-learning support, competitive best-model performance on the reused historical periods, and no serious negative-control contradiction. Quantum incremental value requires stable multi-seed QML improvement over matched classical nonlinear models with paired uncertainty; replication requires both assets. Historical reuse cannot be described as independent confirmation.

Exact configuration grids, feature formulas, timestamps, fold boundaries, and seeds are in the machine-readable protocol and registries. No model result was used to define these rules.
