# Table 1

Source-derived values; CSV retains full precision.

| experiment | venue_assets | frequency | dates_and_split | sample_size | target | evidence_status |
| --- | --- | --- | --- | --- | --- | --- |
| Phase 6 historical benchmark | Binance BTC/ETH | 4h UTC | 2021-12-01 to 2024-03-03 calendar; test origins 2023-07-01 to 2024-03-02 20:00 UTC | 1,417 scored origins/asset | 24h Amihud plus other frozen targets/horizons | FROZEN_HISTORICAL |
| Phase 13 neural scaling | Binance BTC/ETH | 4h UTC | Same Phase 6 final test origins | 1,417/asset | 24h Amihud; log1p target | RETROSPECTIVE_DIAGNOSTIC |
| Phase 14 strong baseline | Binance BTC/ETH | 4h UTC | Dev 2021-12-01 to 2023-06-30 20:00; test 2023-07-01 to 2024-03-02 20:00 UTC | 1,417/asset | 24h future Amihud; MAE(log1p) | RETROSPECTIVE_REANALYSIS |
| Phase 7–8 reused generalization | Binance BTC/ETH/SOL/BNB/XRP | 4h UTC | 2024-03-04 to 2026-09-29 20:00 UTC; previously inspected | Asset/proxy-specific; see source CSV | Amihud, Kyle lambda, Corwin–Schultz | SAME_VENUE_ROBUSTNESS |
| Phase 9A external QML | Bybit BTC/ETH | 4h UTC | 2024-10-06 20:00 to 2026-10-06 20:00; holdout 2026-03-01 20:00 to 2026-10-06 20:00 UTC | 1,315 holdout origins/asset | 24h future Amihud; QML residual correction | FROZEN_EXTERNAL_QML_TEST |
