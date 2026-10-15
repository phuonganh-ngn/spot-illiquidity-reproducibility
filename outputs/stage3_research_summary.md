# Stage 3 research summary

## Scope and interpretation

The primary analysis uses BTC and ETH on the 824-date UTC clean sample, 2021-12-01 through 2024-03-03. There are 1,648 maximum asset-days, while the treatment-time/calendar dimension is 824 dates. The four V2 theory-matched channels yield 48 confirmatory tests, corrected in 16 separate three-horizon Holm families. The timestamp-harmonized 2024–2026 panel is reported separately as robustness only.

These are conditional directional associations and predictive information measures. They are not structural causal effects. Binance spot and USDⓈ-M perpetuals share venue infrastructure, trader populations and USDT settlement. Unobserved endogeneity and exchange-specific mechanical linkage remain possible.

## Confirmatory findings

The primary HGB blocked TS-DML results surviving within-family Holm correction are: BTC C1 FORWARD h3 (θ=35.69, Holm p=0.0468), BTC C1 FORWARD h7 (θ=54.65, Holm p=0.0388), BTC C4 FORWARD h1 (θ=-3.43e-05, Holm p=0.00247), BTC C4 FORWARD h3 (θ=-2.341e-05, Holm p=0.00247), BTC C4 FORWARD h7 (θ=-2.788e-05, Holm p=4.68e-05), BTC C1 REVERSE h7 (θ=-0.0005864, Holm p=0.0418), ETH C4 REVERSE h7 (θ=53.78, Holm p=0.0348).

Channel-level classifications are:

```text
channel_id asset final_classification                          time_variation
        C1   BTC         INCONCLUSIVE          NO_DEMONSTRATED_TIME_VARIATION
        C1   ETH         INCONCLUSIVE          NO_DEMONSTRATED_TIME_VARIATION
        C2   BTC  NO_ROBUST_DIRECTION          NO_DEMONSTRATED_TIME_VARIATION
        C2   ETH  NO_ROBUST_DIRECTION          NO_DEMONSTRATED_TIME_VARIATION
        C3   BTC  NO_ROBUST_DIRECTION DESCRIPTIVE_ONLY_NOT_IN_ENDPOINT_FAMILY
        C3   ETH  NO_ROBUST_DIRECTION DESCRIPTIVE_ONLY_NOT_IN_ENDPOINT_FAMILY
        C4   BTC  NO_ROBUST_DIRECTION          NO_DEMONSTRATED_TIME_VARIATION
        C4   ETH  NO_ROBUST_DIRECTION          NO_DEMONSTRATED_TIME_VARIATION
```

No channel meets the frozen strong-direction criteria. C1 funding–RV is `INCONCLUSIVE`: the continuous TE estimator is unsuitable because funding has a repeated-value atom above 5%, while the VAR benchmark fails its preregistered stationarity/continuity gates. C2 OI–RV, C3 OI–downside and C4 futures-flow–Amihud are `NO_ROBUST_DIRECTION` after learner/timing/placebo requirements.

1. **Funding → spot RV:** BTC has HGB blocked associations at h3 and h7 surviving the C1/BTC/forward Holm family; ETH does not. BTC's sign does not survive the required forward-only and conservative-timing checks, so it is not a robust leading result.
2. **Spot RV → funding:** BTC has one h7 Holm-surviving estimate, which is insufficient for the two-horizon rule. The reverse direction is not established.
3. **OI growth → spot RV:** no confirmatory horizon survives Holm for BTC or ETH.
4. **Spot RV → OI growth:** no confirmatory horizon survives Holm.
5. **OI growth ↔ downside:** neither direction establishes a robust relationship; no confirmatory result survives Holm.
6. **Futures imbalance → Amihud:** BTC has negative HGB blocked coefficients at all three horizons surviving Holm, consistent directionally with lower measured Amihud, but forward-only/timing robustness fails. ETH has no robust forward result.
7. **Spot Amihud → futures imbalance:** ETH's h7 coefficient survives the reverse-family Holm correction, but the two-horizon, learner/timing and placebo requirements do not pass.
8. The isolated statistical rejections are therefore not sufficient to establish a stable channel leader.

### Multiple testing, falsification, and robustness

There are 7 of 48 confirmatory primary tests with Holm p < 0.05. Primary-window past-outcome falsifications reject for 3 registered channel tests after their frozen placebo-family correction: P_REV_BTC_FLOW_AMIHUD_k1, P_REV_BTC_FLOW_AMIHUD_k3, P_FWD_ETH_FLOW_AMIHUD_k7. These failures are retained as identification warnings; they are not evidence for the opposite direction.

Ridge and HGB forward-only estimates are reported for all 48 tests. The required sign stability fails for the isolated funding/RV and futures-flow/Amihud candidates. Conservative timing also changes signs for the BTC C1-forward and C4-forward candidates. Date-block bootstrap intervals accompany all confirmatory estimates; qualitative agreement is required by the frozen rule and does not rescue failed timing or placebo checks.

### Linear and nonlinear directional methods

All 16 VAR/Granger direction tests are `LINEAR_DIAGNOSTIC_FAIL`; no Granger direction is counted as corroboration. Pairwise samples use the longest consecutive complete segment, with a minimum of 400 dates and ADF/KPSS admissibility. Missing segments and the KPSS results prevent an admissible VAR.

The primary continuous kNN TE family has 12 estimable directions and 4 unsuitable/unavailable directions. Funding/RV is excluded by the >5% exact-value atom rule. Among estimable TE tests, 0 survives Holm correction. Observed and surrogate-adjusted TE values are kept in nats; no net direction is computed because the frozen protocol does not permit comparing raw TE across different target variables and no common normalization was preregistered.

## BTC–ETH and temporal stability

BTC–ETH contrasts use common dates and shared date-block bootstrap draws. The paired contrasts are mixed: some C1 forward HAC differences survive their own Holm family, but the h7 bootstrap interval includes zero, and there is no broad channel-level heterogeneity claim. Separate asset significance is not interpreted as an asset difference.

All fixed 365-day and 540-day rolling estimates are reported. The endpoint comparison uses the first and last scheduled non-overlapping 365-day windows. 0 core endpoint comparisons meet the frozen stability criterion after correction. Although some rolling and filtered coefficients move substantially, the endpoint tests do not establish systematic direction changes. The state-space model passed its numerical diagnostics for 16 pair/direction fits, but several estimated state variances lie at the lower bound; filtered movement is not treated as evidence of a time-varying relationship.

## Secondary and timestamp-harmonized analyses

The 100 secondary rows include the 60 originally registered non-confirmatory combinations plus funding-sum/basis extensions; the 72 h14/h30 results remain persistence-only. Neither set upgrades a failed primary channel.

The separate harmonized extended sample estimates cover 48 core rows from 2021-12-01 through 2026-09-30. The fraction of the 48 matched clean/extended core coefficients with the same sign is 81.2%. These results are robustness only and do not replace clean-sample findings; changes around the archive/timestamp harmonization period remain a measurement limitation.

## Overall conclusion

Paper-level classification: **INCONCLUSIVE**.

The strongest defensible statement is that some BTC-specific funding/RV and futures-flow/Amihud conditional associations appear in blocked TS-DML, but their sensitivity to forward-only fitting, conservative timing and falsification prevents a stable derivatives-leading or spot-leading conclusion. OI channels show no robust direction. The results support continuing with a carefully qualified predictive/directional paper, not a causal claim.

Unsupported claims include that derivatives structurally cause spot liquidity or risk, that spot stress structurally causes derivative positioning, that one market universally leads the other, or that the relationship is demonstrably time-varying. The next research step should focus on paper framing and a separately justified external identification or independent-venue validation design; no further model or asset expansion is run here.
