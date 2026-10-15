# Phase 14 source audit

Status: **PASS**. The frozen Phase 6 target/feature registries, chronological fold definitions, Phase 13 selected neural configurations, Phase 6 source code, certified 4-hour panel, and prior Phase 10/12/13 audit records were verified before any Phase 14 fitting.

The certified panel is `outputs/phase5_dynamic_discovery/dataset_4h.jsonl` (SHA-256 `e50637340052f505c585bbce70c86ac7f4d3f607b6ef127a24c1317d1d99402d`). Phase 6 protocol hash: `69affa8f08f3943e9c0914dc276bdd5821410f5d401825e0299487ef13a01028`; selected-model registry hash: `93d8d2d15d0a03ccebd2cea4ce115a8d4dbfabe3d8dfbde8f9dac845757a6313`. Phase 13 final output manifest checksum verified: `4314bc939b04160d182ce3ab95bfd8717a5f3b748ff03999ffc4b8307374c3fc`; all 299 files it covers match their recorded hashes. Phase 13 source audit flags all monitored sources unchanged.

The development-only loader parsed only timestamps before 2023-07-01T00:00:00+00:00; it contained 6546 complete BTC/ETH origins for this task. Fold sizes and target-end purges are recorded in `04_chronological_split_audit.md`. No test outcome was used to define the Phase 14 registry.

Frozen features match Phase 6 exactly: S0 has 6 variables over six rows; S1 uses the same rows plus the four registered derivatives variables. The source implementation uses a future arithmetic mean of `spot_amihud[t+1:t+7]`, log1p target, oldest-to-newest t−5…t inputs, and requires target windows to end strictly before each development boundary.

Available libraries were verified before locking: CatBoost 1.2.10, XGBoost 3.4.1, scikit-learn 1.9.1, PyTorch 2.14.1+cu130. No data download or model fitting occurred during this audit.

Historical data/test outcomes were already inspected in prior phases. All Phase 14 findings must therefore be labeled retrospective, even though the Phase 14 model-selection choices are based only on its chronological development folds.
