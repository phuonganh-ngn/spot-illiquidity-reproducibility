# Stage 3 validation report

**Overall validation: PASS.** Checked at 2026-10-07T05:44:09+00:00.

- **Stage-2 original and V2 protocol hashes: PASS.** 8 original-manifest rows; 12 V2-manifest rows.
- **Confirmatory universe and Holm families: PASS.** 48 unique tests; 16 families × 3 horizons.
- **Clean sample and panel key: PASS.** 824 UTC dates; 1,648 unique BTC/ETH asset-days.
- **Variable allowlist / future-label exclusion: PASS.** 11 predetermined controls; no future-label or centered variable.
- **Temporal folds and date grouping: PASS.** Four contiguous blocks per primary fit; no row-level random folds.
- **Date-block bootstrap: PASS.** 96 intervals; 999 fixed-seed replications; adequacy passed.
- **Placebo, Granger and Transfer Entropy coverage: PASS.** 108 placebos; 16 Granger directions; 48 TE configurations.
- **Pre-secondary clean-output lock: PASS.** 34 locked items verified; locked robustness source preserved separately.
- **Raw data immutability and source lineage: PASS.** 9771 raw files; 4886 source-manifest rows verified.
- **Offline deterministic reproduction: PASS.** Clean primary outputs match original lock; secondary/extended and reports reproduced byte-identically.
- **Required output inventory: PASS.** 29 required tables/reports/figures present.

## Interpretation boundary

The estimates are conditional directional associations and predictive information measures, not structural causal effects. TS-DML does not remove unobserved endogeneity. Granger predictability and Transfer Entropy do not establish structural causality. The primary paper-level classification is **INCONCLUSIVE**: no channel satisfies the frozen directional robustness criteria.

## Lock follow-up

The original clean-sample lock is preserved. One locked derived summary table is expanded with a deterministic robustness audit; the pre-enrichment bytes are retained separately and both hashes are recorded in `stage3_lock_followup_manifest.csv`. No primary estimate or classification is changed.
