# Phase 7 validation report

The data build was reproduced from the 6,195 cached monthly archives without network requests. SHA-256 and ZIP CRC checks passed for every archive. The resulting 54,780-row five-asset panel has SHA-256 `5a339c5c4a8778ceb74344a10c67806c2323ffac80556be18bf7a9c99de8dcf6`, identical to the frozen panel. The independent Phase-5 BTC/ETH parity check passed and the reconstructed Amihud formula differs by at most 9.99e-17. The Phase-7 data audit passed.

The offline development reproduction reran all 8,640 development fits with the same frozen folds, transforms, feature sets and learners. The selected-model file and all 559,440 OOF rows are byte-identical. The gzip containers differ only in volatile header metadata, so their compressed-file hashes differ although their decompressed payloads match exactly. The complete development JSON matches after excluding output path, runtime, and the hash of that differently packaged gzip file. The reproduction runner changes only the output directory.

The external reproduction produced 2,149,446 rows. Its compressed prediction file is byte-identical to the primary external artifact, and all compared numerical sections—paired scores, block-bootstrap summaries, category aggregation, fixed-period summaries, stress metrics, baselines and negative controls—match exactly.

The original Stage 1/3/Phase 5/Phase 6 snapshot remains intact: all 9,947 protected file hashes pass. The candidate archive integrity, Stage-6 protocol/model hashes, frozen Phase-7 inputs, development freeze and final external freeze all pass.

The external evaluation had one incomplete technical attempt because a `StandardScaler` import was missing in the final negative-control routine. The retry used the unchanged frozen data, transforms, models and seeds. No first-attempt metrics were inspected before retry; both forecast artifacts are byte-identical. This execution deviation is recorded in the external evaluation JSON and is not concealed.

The fixed future-information negative controls ran, but their saved output contains raw MAE only and lacks a matched Spot-only benchmark. They are therefore **not sufficient to certify the negative-control test as passed**. The primary forecast conclusion remains that the frozen Amihud replication criteria and cross-measure generalization criteria fail. No causal conclusion is supported.

Overall engineering reproduction: **PASS**. Research result: **NOT_EXTERNALLY_REPLICATED_UNDER_FROZEN_CRITERIA**.
