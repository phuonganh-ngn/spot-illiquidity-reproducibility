# Table 2

Source-derived values; CSV retains full precision.

| model_family | class | information_sets | registered_specification_note |
| --- | --- | --- | --- |
| Ridge | regularized linear | S0 and S1 | Phase 14 development-selected config per asset/info set; PCA/scaling is inherited only where registered. |
| ExtraTrees | bagged trees | S0 and S1 | 200 trees; selected min_samples_leaf from locked three-setting grid. |
| HistGradientBoosting | boosted trees | S0 and S1 | learning_rate .05; max_iter 100; max_leaf_nodes selected from locked grid. |
| CatBoost | boosted trees | S0 and S1 | Three fixed development-only settings; no post-result expansion. |
| XGBoost | boosted trees | S0 and S1 | Three fixed development-only settings; poor/mean-like forecasts retained. |
| MLP | one-hidden-layer neural | S0 and S1 | Phase 13 selected configuration; train-only feature and target scaling; same seed protocol. |
| GRU | one-layer temporal neural | S0 and S1 | Sequence length 6, hidden size 16, linear head, fixed 20 epochs, target scaling train-only. |
| Training mean / persistence | simple references | S0 and S1 aligned | Reference forecasts; no learner tuning. |
