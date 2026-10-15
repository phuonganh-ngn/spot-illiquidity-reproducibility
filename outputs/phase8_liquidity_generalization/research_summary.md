# Phase 8 — Liquidity-signal validation and generalization

This is a predictive reanalysis of cached Binance data. The Phase-7 post-2024 results were already inspected, so this work is not a new untouched confirmation. No market data were downloaded, and Stage 3 / Phase 5 / Phase 6 / Phase 7 artifacts remain unchanged.

## Measurement and sample

The Phase-7 panel contains 54,780 rows across BTC, ETH, SOL, BNB and XRP. The Amihud formula independently reproduces to maximum absolute error 9.99e-17; the 24-hour target reproduces to 9.58e-17. The Kyle price-impact proxy is built from 5-minute signed spot quote flow; Corwin–Schultz is a distinct high-low range spread proxy. Both were constructed by the cached Phase-7 builder, and 6,195 raw archive hashes were reverified. Kyle was not independently re-estimated from all 5-minute bars in this Phase-8 summary.

The external sample begins after the empirically audited archive-layout boundary and uses the frozen piecewise OI timestamp map. This is semantically usable under the prior audit, with that harmonization retained as a measurement limitation. SOL, BNB and XRP were included by pre-estimation coverage/timestamp rules, not forecast results.

## Forecast generalization

The frozen Phase-7 rule remains `NOT_EXTERNALLY_REPLICATED_UNDER_FROZEN_CRITERIA`. For legacy Amihud, the equal-category Spot+Derivatives improvement criterion is met in 3/5 assets, below the required 4/5; BTC and ETH also fail the requirement that at least 3/5 categories improve in both assets, and the BTC/ETH paired intervals do not both exclude zero. The two distinct liquidity proxies do not pass the frozen BTC/ETH and category criteria.

The best-family comparison selects one family per information set using development OOF loss only, then compares the already saved external forecasts:

BTC: development-selected HistGradientBoosting vs HistGradientBoosting; S1−S0 MAE +0.00148961, CI [+0.00036475, +0.00260874].
ETH: development-selected Ridge vs RBF_SVR; S1−S0 MAE -0.00238859, CI [-0.00676514, +0.00189664].
SOL: development-selected Ridge vs Ridge; S1−S0 MAE +4.46456e-05, CI [+3.84192e-05, +5.16151e-05].
BNB: development-selected HistGradientBoosting vs HistGradientBoosting; S1−S0 MAE +0.00741516, CI [+0.0040849, +0.010776].
XRP: development-selected HistGradientBoosting vs HistGradientBoosting; S1−S0 MAE -0.000383312, CI [-0.00271831, +0.00218729].

This winner-selected secondary comparison does not override equal-category or cross-asset replication criteria.

## Stress warning, time stability, and negative controls

Stress warnings use the frozen development q90 event/alarm thresholds and equal-weight five categories:

BTC: equal-category F1 delta +0.0086 (S1−S0), 42-step block CI [-0.0010, +0.0193].
ETH: equal-category F1 delta +0.0732 (S1−S0), 42-step block CI [+0.0540, +0.0911].

The matched impossible-time tests found 0/10 cases where future/displaced derivatives information significantly reduced past-target error under the fixed paired block-bootstrap screen. Details below; these are persistence/temporal-contamination diagnostics, not forecastable forward gains:

BTC future_derivatives_t_plus_7_to_t_plus_12: ΔMAE -2.72615e-05, CI [-0.000710827, +0.000671694].
BTC forward_displacement_42_intervals: ΔMAE +0.000862035, CI [+0.000436148, +0.00132298].
ETH future_derivatives_t_plus_7_to_t_plus_12: ΔMAE -0.000649499, CI [-0.0024056, +0.00119175].
ETH forward_displacement_42_intervals: ΔMAE -0.000538066, CI [-0.00191926, +0.000910638].
SOL future_derivatives_t_plus_7_to_t_plus_12: ΔMAE +3.58969e-05, CI [+2.95541e-05, +4.24911e-05].
SOL forward_displacement_42_intervals: ΔMAE +4.47363e-05, CI [+3.76575e-05, +5.20876e-05].
BNB future_derivatives_t_plus_7_to_t_plus_12: ΔMAE +0.0116983, CI [+0.0080465, +0.0155577].
BNB forward_displacement_42_intervals: ΔMAE +0.0039362, CI [+0.0011874, +0.00666684].
XRP future_derivatives_t_plus_7_to_t_plus_12: ΔMAE +0.00575405, CI [-0.00337828, +0.0141053].
XRP forward_displacement_42_intervals: ΔMAE +0.00668551, CI [-0.000136685, +0.0133375].

Fixed 2024-03–2024-12, 2025 and 2026-partial windows show materially changing signs across assets, models, and targets; the full machine-readable segment results are in `temporal_stability.json`.

## Defensible interpretation

The narrow Phase-6 24-hour Amihud result does not generalize robustly under the already frozen Phase-7 multi-condition rule. Added derivatives can change individual model-family stress warnings and selected best-family losses, but those secondary results are asset/model dependent and do not establish cross-proxy replication. No causal claim is supported. The strongest next test would reserve a genuinely unseen venue or future period under a new preregistered protocol; the already inspected Binance 2024–2026 period cannot serve as fresh confirmation.

Overall classification: **NO_ROBUST_GENERALIZATION**. Frozen Phase-7 classification: **NOT_EXTERNALLY_REPLICATED_UNDER_FROZEN_CRITERIA**.
