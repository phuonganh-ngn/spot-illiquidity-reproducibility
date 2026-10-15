Table 2. Final historical test results for all fixed Phase 14 model families, scored on identical 1,417 origins per asset. MAE is on Z=log1p(future Amihud); lower is better. S0 and S1 may have different configurations, selected only on development folds. Retrospective results.

| Asset | Model                | SpotOnly MAE           | Spot+Derivatives MAE   | Delta MAE               | Relative change (%) | Convergence status   |
| ----- | -------------------- | ---------------------- | ---------------------- | ----------------------- | ------------------- | -------------------- |
| BTC   | CatBoost             | 6.115194611907135e-06  | 6.072439895269089e-06  | 4.275471663804651e-08   | 0.6991554537740651  | NO_REPORTED_WARNING  |
| BTC   | ExtraTrees           | 6.054120321400544e-06  | 5.9941818515897895e-06 | 5.993846981075426e-08   | 0.9900442447250248  | NO_REPORTED_WARNING  |
| BTC   | GRU                  | 6.920174228557576e-06  | 7.390338797425146e-06  | -4.701645688675705e-07  | -6.7941146182033405 | FIXED_20_EPOCHS      |
| BTC   | HistGradientBoosting | 6.122879881677108e-06  | 6.000912862245633e-06  | 1.2196701943147494e-07  | 1.9919877866045475  | NO_REPORTED_WARNING  |
| BTC   | MLP                  | 7.809076863142212e-06  | 7.975641700336123e-06  | -1.6656483719391173e-07 | -2.132964499044378  | MLP_MAX_ITER_WARNING |
| BTC   | Persistence          | 1.4938068204822818e-05 | 1.4938068204822818e-05 | 0.0                     | 0.0                 | NO_REPORTED_WARNING  |
| BTC   | Ridge                | 6.8953886880448365e-06 | 7.252772314189808e-06  | -3.5738362614497124e-07 | -5.182936630745701  | NO_REPORTED_WARNING  |
| BTC   | TrainingMean         | 7.37342659430194e-06   | 7.37342659430194e-06   | 0.0                     | 0.0                 | NO_REPORTED_WARNING  |
| BTC   | XGBoost              | 7.373426318226414e-06  | 7.373426318226414e-06  | 0.0                     | 0.0                 | NO_REPORTED_WARNING  |
| ETH   | CatBoost             | 1.4196955129428606e-05 | 1.4096944026024283e-05 | 1.0001110340432274e-07  | 0.7044545995430497  | NO_REPORTED_WARNING  |
| ETH   | ExtraTrees           | 1.4029787840184666e-05 | 1.414603971596146e-05  | -1.1625187577679366e-07 | -0.8286075106839499 | NO_REPORTED_WARNING  |
| ETH   | GRU                  | 1.4617091602796658e-05 | 1.4562399714574023e-05 | 5.469188822263487e-08   | 0.3741639562015934  | FIXED_20_EPOCHS      |
| ETH   | HistGradientBoosting | 1.4147248211607276e-05 | 1.4014506758191438e-05 | 1.327414534158372e-07   | 0.9382846150032762  | NO_REPORTED_WARNING  |
| ETH   | MLP                  | 1.550674216715923e-05  | 1.5745953793212944e-05 | -2.392116260537127e-07  | -1.5426298024115226 | MLP_MAX_ITER_WARNING |
| ETH   | Persistence          | 3.4735682882179743e-05 | 3.4735682882179743e-05 | 0.0                     | 0.0                 | NO_REPORTED_WARNING  |
| ETH   | Ridge                | 1.490654075941996e-05  | 1.4626612368426665e-05 | 2.7992839099329555e-07  | 1.8778896828655507  | NO_REPORTED_WARNING  |
| ETH   | TrainingMean         | 1.5363822811930615e-05 | 1.5363822811930615e-05 | 0.0                     | 0.0                 | NO_REPORTED_WARNING  |
| ETH   | XGBoost              | 1.5363822884904733e-05 | 1.5363822884904733e-05 | 0.0                     | 0.0                 | NO_REPORTED_WARNING  |
