# Phase 6 benchmark protocol

The confirmatory comparison is Spot Only versus Spot Plus All Derivatives for BTC and ETH, estimated separately. The data are 4-hour UTC observations; development ends 2023-06-30 20:00 UTC and the final test begins 2023-07-01 00:00 UTC. The source panel hash is `e50637340052f505c585bbce70c86ac7f4d3f607b6ef127a24c1317d1d99402d`.

The tasks are continuous forecasts of future realized volatility, downside semivariance, and Amihud illiquidity over 4-hour, 8-hour, and 24-hour windows. Inputs are six intervals of history through t, and every outcome begins at t+1. Positive skewed targets use a fixed log1p training transform. Primary scoring is MAE on the log1p target; original-scale MAE/RMSE and volatility QLIKE are secondary.

The primary information sets use the same complete rows. Spot Only includes six spot features at lags t-5 through t; Spot Plus All Derivatives adds funding, OI log change, futures taker imbalance, and basis over the same history. Single-component derivative additions are fixed-hyperparameter secondary ablations. Missing values are not imputed. Scaling is trained inside each temporal fold.

Model families are OLS/Ridge, RBF-SVR, ExtraTrees, HistGradientBoosting, a compact MLP, a compact GRU, and a fixed equal-weight OOF ensemble. Each tunable family receives three prespecified configurations on the same three expanding-origin folds. Larger sequence models and additional boosting libraries are excluded because the per-asset development sample is about 3.2k usable rows and the chosen families already cover the relevant model classes.

Final-test uncertainty uses paired 42-interval moving blocks and 2,000 seeded replications. Holm correction is within each asset × target × horizon model-family comparison. Robust incremental value requires at least four of six family categories to improve, two independent categories to pass Holm-adjusted paired inference, at least 1% average relative log-MAE reduction, and no two categories with supported deterioration. Replication and broader classifications follow the frozen JSON criteria.

The final test is reserved for one evaluation after development tuning and selected configurations are saved. The protocol JSON is frozen and hashed before any final-test feature or target value is parsed. Predictive improvement will not be described as causal evidence.
