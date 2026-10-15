# Phase 7 data and liquidity-measurement audit

Panel SHA-256: `5a339c5c4a8778ceb74344a10c67806c2323ffac80556be18bf7a9c99de8dcf6`.

The five-asset panel was rebuilt from cached Stage-1 BTC/ETH files and cached official Binance archives for SOL, BNB and XRP. No market-data download was made. The cached archive integrity check, all 6,195 archive hashes/CRCs, Phase-5 parity checks, Amihud reconstruction, and protected Stage-3/Phase-5/Phase-6 artifact hashes pass.

The clean base series span 2021-12-01 through 2026-09-30 at 4-hour UTC frequency. Each asset has 10,590 calendar rows in that window. The development interval is 2021-12-01 through 2023-06-30; the previously inspected Stage-6 test interval through 2024-03-03 is excluded from new tuning. The new external evaluation is reserved from 2024-03-04. A 24-hour target has 5,646 candidate feature dates per asset in the external interval, with 5,640 labels after the six terminal rows are removed because future source history ends at 2026-09-30.

All five assets pass the pre-specified coverage rules. SOL/BNB/XRP were selected only from coverage and timestamp-semantic checks, not from correlations, target distributions, or forecast performance. Candidate OI mapping uses the fixed, previously documented 0-second mapping through 2024-03-03 and +300-second mapping from 2024-03-04. Days failing the expected-offset match are missing; they are never retimed, interpolated, or filled.

BTC/ETH values in the rebuilt panel match the frozen Phase-5 records on 9,888 shared asset-intervals. Ten shared series have identical missingness and maximum numerical differences below 1e-12: funding state, OI log change, futures taker imbalance, spot return, realized volatility, downside semivariance, Amihud, spot taker imbalance, spot quote-volume growth, and basis.

Amihud is independently reproduced as 1e6 × absolute 4-hour log return divided by Binance spot quote volume in USDT for the same 4-hour interval. The maximum reconstruction difference is 9.993633524885137e-17 and missingness is identical. Higher Amihud indicates greater price impact per turnover (measured illiquidity). It is not a quoted bid-ask spread.

Two distinct future liquidity proxies are constructed over the same 24-hour window t+1 through t+6. The Kyle-type price-impact proxy is the absolute intercept-adjusted slope of 288 following 5-minute log returns on signed Binance spot quote flow. The Corwin-Schultz proxy is the mean of five adjacent-4-hour high-low estimates contained in that window. Kyle is a price-impact measure, while Corwin-Schultz is a range-based spread estimate that can also reflect volatility; neither is treated as interchangeable with Amihud or as a directly observed quote spread.

All three external target series have 99.8% or better observed coverage before forecast-feature completeness; their external labels contain no negative values. External target values were not inspected for distributional selection. The development-only measurement audit retains zero/tail behavior and records a fixed preselected transform comparison.

Known limitations are the same-venue Binance spot/futures trading population, shared USDT settlement environment, the fixed piecewise post-2024 OI timestamp harmonization, and proxy measurement error in Kyle and Corwin-Schultz. These are interpretation limitations, not reasons to alter the sample after evaluation.
