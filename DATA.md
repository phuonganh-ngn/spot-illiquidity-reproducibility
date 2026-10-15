# Data and provenance

The package includes the compact, frozen panels used by the research scripts. It does not contain raw Binance or Bybit trade archives. The local project contained roughly 155 GB of raw and intermediate market data; those files are unsuitable for a normal Git repository and are not needed to verify the saved results.

Bundled inputs include:

- Clean daily BTC/ETH spot and derivatives panels used in the directional analyses.
- The frozen four-hour discovery panel used by the historical multi-model benchmark.
- The four-hour multi-asset panel used for the cross-asset and liquidity-proxy evaluation.
- The Amihud study panel used for the residual-learning analysis.
- The frozen BTC and ETH Bybit four-hour panels used in the external QML evaluation.

Their paths and SHA-256 values are listed in `source_manifest.csv`. The original source paths are relative to the research project; no local home-directory paths are included.

## Rebuilding from raw archives

Raw archives are not bundled. The original ingestion and panel-construction scripts remain under `src/`, and the historical source manifests and data protocols remain beside the archived results where applicable. A raw rebuild requires obtaining the exact source archives for the frozen date windows, checking their schema and timestamp semantics, and verifying them against the recorded manifests before running panel construction. Do not treat a newly downloaded archive as byte-identical to the archived input without that verification.

The Bybit analysis used the frozen common 24-month interval. Binance post-2024 observations were already inspected in the original project and must not be described as a fresh holdout.
