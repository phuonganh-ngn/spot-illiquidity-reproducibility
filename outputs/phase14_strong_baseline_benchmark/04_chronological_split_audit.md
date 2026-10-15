# Phase 14 chronological split audit

Development date range: 2021-12-01T00:00:00Z through 2023-06-30T20:00:00Z. Final test feature-origin range: 2023-07-01T00:00:00Z through 2024-03-03T20:00:00Z. Split timestamp: 2023-07-01T00:00:00+00:00.

The exact original expanding validation folds are F1: 2022-10-01 to 2023-01-01; F2: 2023-01-01 to 2023-04-01; F3: 2023-04-01 to 2023-07-01. For each fold, training origins require target_end < validation_start; validation origins require origin < validation_end and target_end < validation_end. Final model training requires origin < 2023-07-01 and target_end < 2023-07-01. Test prediction origins require origin >= 2023-07-01 and target_end < 2024-03-04.

All S0/S1 models use the same complete-case rows because Phase 6 requires all ten certified raw features to be finite over t−5…t, even for S0, and all six future Amihud bars to be present. No gaps are imputed. Development-only timestamps and fold sample counts:

- BTC F1: train n=1684 (2021-12-01T20:00:00+00:00–2022-09-29T20:00:00+00:00, last target end 2022-09-30T20:00:00+00:00); validation n=534 (2022-10-01T00:00:00+00:00–2022-12-30T20:00:00+00:00, last target end 2022-12-31T20:00:00+00:00).
- BTC F2: train n=2224 (2021-12-01T20:00:00+00:00–2022-12-30T20:00:00+00:00, last target end 2022-12-31T20:00:00+00:00); validation n=509 (2023-01-01T00:00:00+00:00–2023-03-30T20:00:00+00:00, last target end 2023-03-31T20:00:00+00:00).
- BTC F3: train n=2739 (2021-12-01T20:00:00+00:00–2023-03-30T20:00:00+00:00, last target end 2023-03-31T20:00:00+00:00); validation n=509 (2023-04-01T00:00:00+00:00–2023-06-29T20:00:00+00:00, last target end 2023-06-30T20:00:00+00:00).
- ETH F1: train n=1705 (2021-12-02T00:00:00+00:00–2022-09-29T20:00:00+00:00, last target end 2022-09-30T20:00:00+00:00); validation n=534 (2022-10-01T00:00:00+00:00–2022-12-30T20:00:00+00:00, last target end 2022-12-31T20:00:00+00:00).
- ETH F2: train n=2245 (2021-12-02T00:00:00+00:00–2022-12-30T20:00:00+00:00, last target end 2022-12-31T20:00:00+00:00); validation n=509 (2023-01-01T00:00:00+00:00–2023-03-30T20:00:00+00:00, last target end 2023-03-31T20:00:00+00:00).
- ETH F3: train n=2760 (2021-12-02T00:00:00+00:00–2023-03-30T20:00:00+00:00, last target end 2023-03-31T20:00:00+00:00); validation n=526 (2023-04-01T00:00:00+00:00–2023-06-29T20:00:00+00:00, last target end 2023-06-30T20:00:00+00:00).

Test values are excluded from selection. Phase 14 selection is locked before the full-panel loader is called.
