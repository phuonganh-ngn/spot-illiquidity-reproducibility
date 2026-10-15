# Phase 9 protocol — external venue replication

This protocol was frozen before model fitting or inspection of any external-venue forecast result. The two primary candidates are Bybit and OKX. Every candidate that passes the same data-quality gate is included; no venue can be selected from predictive performance.

The primary assets are BTC and ETH, the frequency is 4-hour UTC, and the outcome is the frozen 24-hour future Amihud mean over t+1 through t+6, using `1e6 × |4-hour log return| / positive USDT quote volume`, transformed as `log1p`. The information sets and registered transformations are copied from the earlier frozen targeted study.

The primary external configurations are BTC Hybrid QNN, 4 qubits, and ETH variational QNN, 6 qubits. Both use the exact five logged seeds, training-only StandardScaler/PCA, two StronglyEntanglingLayers, Y-angle encoding, Pauli-Z readout, 40 epochs, batch size 64, Adam at 0.01, exact CPU statevector simulation, and the matched one-hidden-layer MLP from the frozen artifacts. No architecture, qubit, feature, transform, or hyperparameter search is allowed on either venue.

The SpotOnly family and parameters are frozen from prior development evidence: BTC CatBoost and ETH XGBoost. Training-period residual labels will use chronological out-of-fold SpotOnly predictions only. The final holdout is the last 30% of the longest common contiguous valid sample after a 70% chronological training split, computed mechanically before fitting. At least 18 months of common data are required.

A venue must provide semantically comparable spot price and USDT quote volume, plus funding, OI, aggressive futures flow, and basis/premium series for both assets. The acceptance floor is 95% coverage for required 4-hour series with known UTC timestamps, no forward-filled observations, no unresolved contract migration, and no material duplicate blocks. All candidates meeting these rules are retained.

The primary endpoints are transformed-scale MAE improvement over SpotOnly and improvement over the paired MLP residual correction. Paired moving-block bootstrap uses 1,000 replications at block lengths 42 and 84, seed base 20261007. Holm correction is applied separately to BTC/ETH for each of the two primary endpoints. No causal or hardware quantum advantage claim is permitted.
