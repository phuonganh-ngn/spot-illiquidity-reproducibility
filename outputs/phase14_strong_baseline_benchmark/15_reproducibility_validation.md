# Phase 14 reproducibility validation

Overall: **PASS**.

- Locked protocol SHA-256: `f36f1be8a3b6768a7d873c8a77087ed9d6b48df185974de97f7151b1b3add603`.
- Frozen model registry SHA-256: `d3db7b62c523ca894ad5b73bf5fb0daa8676edfc039585e69293421ab2135843`.
- Development selection SHA-256: `086e74619225256873ad376c5dcb7ca21e74db29e29be9936b067a09410bbf8c`; it records `test_values_parsed=false`.
- Development fits: 204/204 passed; every candidate/reference pooled development MAE and origin count reconstructed from 228 saved fold prediction files.
- Final test fits: 28/28 passed; 51012 forecast rows cover 9 models × 2 information sets × 2 assets × 1,417 origins.
- Final MAEs reconstructed from the saved prediction matrix within 5e−14.
- Phase 14 test origin vectors and target values exactly match the retained Phase 13 final-test targets for BTC and ETH (`{'BTC': {'identical_test_origin_vector': True, 'identical_target_log1p': True, 'n_phase14': 1417, 'n_phase13': 1417}, 'ETH': {'identical_test_origin_vector': True, 'identical_target_log1p': True, 'n_phase14': 1417, 'n_phase13': 1417}}`).
- Feature scalers were fit on training data only; target scalers were used only for MLP/GRU and fit on training labels only. Inverse transformation precedes scoring. No prediction clipping is used.
- Frozen Phase 6 fold boundaries, strict six-bar purging, asset universe, target constructor, features, and test boundary are unchanged.
- Bootstrap uses fixed declared seeds, contiguous 42/84-row moving blocks, 1,000 draws, centered two-sided p-values, add-one finite-resample correction, and Holm across the two asset hypotheses.
- Current hashes of 704 pre-run source artifacts match the pre-run inventory; 283 paths independently checked against Phase 13’s source audit also match. No prior Phase 6–13 output, market data, QML/Bybit result, or manuscript was modified.
- New market-data downloads: none. Quantum model fitting: none. Manuscript edits: none.

The project had already inspected the test interval in earlier phases; this is reproducibility of the retrospective analysis, not independent confirmation.
