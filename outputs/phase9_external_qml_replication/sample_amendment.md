# Phase 9A sample amendment

This amendment replaces the original “longest common eligible sample” rule with the **latest complete common 24-month interval** for the primary Bybit external replication. It is made before panel construction and before any predictive model evaluation. No SpotOnly, residual, MLP, QNN, bootstrap, MAE, model selection, or exploratory forecast has been run or inspected.

The reason is computational feasibility: constructing frozen futures taker flow from transaction-level Bybit archives is I/O-intensive. All existing raw files are preserved. Only files required by the mechanically frozen common 24-month primary window will be parsed or downloaded from this point forward.

The primary venue remains Bybit; assets remain BTC and ETH; target frequency remains 4-hour UTC; target remains the frozen 24-hour future spot Amihud; the shared train/holdout rule remains a common chronological 70/30 calendar split. Feature definitions, transformations, model settings, seeds, and evaluation rules are unchanged.

The exact calendar endpoints will be frozen mechanically from timestamp coverage and complete future-label availability, without inspecting outcome behavior or forecast results.
