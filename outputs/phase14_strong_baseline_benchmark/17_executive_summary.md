# Phase 14 executive summary

**Status: COMPLETE — retrospective reanalysis.** The original Phase 6–13 outputs, data and manuscript remain unchanged. The new benchmark used fixed BTC/ETH 24-hour Amihud targets, the frozen Phase 6 feature sets and chronology, training-only preprocessing, six model families plus references, and a development-only selection step frozen before test data were parsed.

BTC selected HGB under both S0 and S1. Test MAE fell from 6.1229e−6 to 6.0009e−6 (1.99%), but the 42-bar nominal interval includes zero and Holm p=0.0679: **inconclusive**. ETH selected Ridge under S0 and ExtraTrees under S1. MAE fell from 1.4907e−5 to 1.4146e−5 (5.10%); the 42-bar nominal interval is positive and Holm p=0.0100: **favorable retrospective evidence**. The 84-bar sensitivity agrees qualitatively for both assets.

Target-scaled MLP/GRU models did not beat the strongest test-period classical tree/boosting reference. MLPs still failed to converge by the fixed iteration cap. Within-family derivatives effects were mixed. Thus Phase 14 does not support a universal derivatives advantage; the ETH result is a single-asset retrospective selected-system result and is not independent confirmation. Phase 6 remains 2/18, Phase 7–8 broad generalization remains failed, and Phase 9 remains `NO_EXTERNAL_QML_REPLICATION`.

See `12_scientific_interpretation.md`, `07_primary_best_system_comparison.csv`, and manuscript-ready tables under `tables/`. No further model search is warranted for this retrospective question; prospective evidence would require a future frozen study on genuinely unseen data.
