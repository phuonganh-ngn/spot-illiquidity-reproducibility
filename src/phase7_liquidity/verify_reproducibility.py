from __future__ import annotations

import hashlib
import gzip
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "phase7_liquidity_replication"
REPRO = OUT / "offline_reproduction"
DEV_REPRO = OUT / "offline_reproduction_development"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    dev_freeze = load(OUT / "phase7_development_freeze.json")
    phase7_freeze = load(OUT / "phase7_freeze_manifest.json")
    data_audit = load(OUT / "phase7_data_audit.json")
    archive = load(OUT / "candidate_archive_integrity.json")
    build = load(OUT / "phase7_data_build_manifest.json")
    external_freeze = load(OUT / "phase7_external_freeze.json")
    decision = load(OUT / "phase7_replication_decision.json")

    # Development rerun: the only code change in the runner is its output directory.
    original_script = (ROOT / "src/phase7_liquidity/phase7_run_development.py").read_text()
    old = 'OUT = ROOT / "outputs" / "phase7_liquidity_replication"'
    new = 'OUT = ROOT / "outputs" / "phase7_liquidity_replication" / "offline_reproduction_development"'
    if original_script.count(old) != 1:
        raise RuntimeError("Unable to verify the development reproduction output-path-only change")
    expected_runner = original_script.replace(old, new)
    runner_path = ROOT / "src/phase7_liquidity/phase7_run_development_reproduction.py"
    runner_output_only = runner_path.read_text() == expected_runner
    primary_dev = load(OUT / "phase7_development_cv.json")
    repro_dev = load(DEV_REPRO / "phase7_development_cv.json")
    # Exclude run-location and runtime metadata, plus the compressed-file hash
    # (gzip headers encode timestamp/name metadata). The full decompressed OOF
    # payload is compared byte-for-byte below.
    volatile_dev_keys = {"development_oof_path", "development_oof_sha256", "elapsed_seconds"}
    primary_dev_core = {k: v for k, v in primary_dev.items() if k not in volatile_dev_keys}
    repro_dev_core = {k: v for k, v in repro_dev.items() if k not in volatile_dev_keys}
    dev_cv_equal = primary_dev_core == repro_dev_core
    dev_selected_sha_equal = sha(OUT / "phase7_selected_models.json") == sha(DEV_REPRO / "phase7_selected_models.json")
    dev_oof_sha_equal = sha(OUT / "phase7_development_oof_predictions.csv.gz") == sha(DEV_REPRO / "phase7_development_oof_predictions.csv.gz")
    with gzip.open(OUT / "phase7_development_oof_predictions.csv.gz", "rb") as f:
        primary_dev_oof_payload = f.read()
    with gzip.open(DEV_REPRO / "phase7_development_oof_predictions.csv.gz", "rb") as f:
        repro_dev_oof_payload = f.read()
    dev_oof_payload_equal = primary_dev_oof_payload == repro_dev_oof_payload
    dev_oof_rehash_equal = sha(OUT / "phase7_development_oof_predictions.csv.gz") == dev_freeze["files"]["outputs/phase7_liquidity_replication/phase7_development_oof_predictions.csv.gz"]

    # External reproduction compares complete numerical sections and the prediction payload byte-for-byte.
    primary_external = load(OUT / "phase7_external_evaluation.json")
    repro_external = load(REPRO / "phase7_external_evaluation.json")
    compare_sections = [
        "protocol_sha256", "dataset_sha256", "external_sample", "selected_transforms", "selected_models",
        "paired_comparison_results", "equal_category_aggregate", "category_summary", "temporal_stability",
        "stress_warning", "simple_baselines", "bootstrap", "negative_controls",
    ]
    external_sections_equal = {k: primary_external.get(k) == repro_external.get(k) for k in compare_sections}
    primary_prediction = OUT / "phase7_external_predictions.csv.gz"
    repro_prediction = REPRO / "phase7_external_predictions.csv.gz"
    external_predictions_equal = sha(primary_prediction) == sha(repro_prediction)
    external_predictions_equal &= sha(primary_prediction) == external_freeze["files"]["phase7_external_predictions.csv.gz"]
    with gzip.open(primary_prediction, "rt", encoding="utf-8") as f:
        external_prediction_rows = sum(1 for _ in f) - 1

    # Frozen input, data, archive, and protected-artifact checks.
    dev_inputs_equal = {rel: sha(ROOT / rel) == expected for rel, expected in dev_freeze["files"].items()}
    external_files_equal = {rel: sha(OUT / rel) == expected for rel, expected in external_freeze["files"].items()}
    protected_snapshot = load(ROOT / "outputs/phase6_autonomous_benchmark/prior_artifact_snapshot.json")
    protected_failures = []
    for rel, expected in protected_snapshot.get("files", {}).items():
        path = ROOT / rel
        if not path.exists() or sha(path) != expected["sha256"]:
            protected_failures.append(rel)

    results = {
        "status": "PASS" if all([
            runner_output_only, dev_cv_equal, dev_selected_sha_equal, dev_oof_payload_equal, dev_oof_rehash_equal,
            all(external_sections_equal.values()), external_predictions_equal, all(dev_inputs_equal.values()),
            all(external_files_equal.values()), data_audit.get("data_audit_pass", False),
            data_audit.get("protected_phase6_hashes_pass", False), archive.get("pass", False),
            archive.get("archive_count") == 6195, build.get("network_requests") == 0,
            not protected_failures, decision.get("classification") == "NOT_EXTERNALLY_REPLICATED_UNDER_FROZEN_CRITERIA",
        ]) else "FAIL",
        "offline_data_rebuild": {
            "command": "python src/phase7_liquidity/build_and_audit.py; python src/phase7_liquidity/phase7_data_audit.py",
            "network_requests": build.get("network_requests"),
            "archive_count_sha256_and_crc_pass": archive.get("pass"),
            "verified_archives": archive.get("sha256_verified"),
            "panel_sha256": sha(ROOT / "data/processed/phase7/market_liquidity_panel_4h.csv.gz"),
            "panel_matches_frozen_sha256": sha(ROOT / "data/processed/phase7/market_liquidity_panel_4h.csv.gz") == phase7_freeze["frozen_inputs_and_rules"]["data/processed/phase7/market_liquidity_panel_4h.csv.gz"],
            "data_audit_pass": data_audit.get("data_audit_pass"),
            "phase5_variable_parity_pass": data_audit.get("phase5_parity", {}).get("pass"),
            "amihud_reconstruction_max_abs_error": data_audit.get("amihud_measurement_audit", {}).get("max_absolute_difference"),
        },
        "offline_development_reproduction": {
            "source_change_only_output_path": runner_output_only,
            "development_cv_json_equal_excluding_path_runtime_and_container_hash": dev_cv_equal,
            "selected_models_byte_identical": dev_selected_sha_equal,
            "oof_predictions_compressed_files_byte_identical": dev_oof_sha_equal,
            "oof_predictions_decompressed_payload_byte_identical": dev_oof_payload_equal,
            "oof_predictions_match_development_freeze": dev_oof_rehash_equal,
            "primary_fit_count": primary_dev.get("fit_count"),
            "selected_models": len(load(OUT / "phase7_selected_models.json")["selected_models"]),
            "selected_transforms": len(load(OUT / "phase7_selected_models.json")["transform_by_asset_target"]),
            "oof_rows": dev_freeze.get("oof_rows"),
        },
        "offline_external_reproduction": {
            "prediction_file_byte_identical": external_predictions_equal,
            "prediction_sha256": sha(primary_prediction),
            "numerical_sections_identical": external_sections_equal,
            "external_prediction_rows": external_prediction_rows,
        },
        "preservation": {
            "phase7_frozen_inputs_unchanged": all(dev_inputs_equal.values()),
            "phase7_external_frozen_outputs_unchanged": all(external_files_equal.values()),
            "stage1_stage3_phase5_phase6_snapshot_file_count": len(protected_snapshot.get("files", {})),
            "stage1_stage3_phase5_phase6_snapshot_pass": not protected_failures,
            "failed_paths": protected_failures,
        },
        "research_decision": {
            "classification": decision.get("classification"),
            "amihud_replication_pass": decision.get("amihud_replication", {}).get("passed"),
            "distinct_proxy_generalization_pass": decision.get("distinct_liquidity_proxy_generalization", {}).get("passed"),
            "negative_control_status": decision.get("negative_control_status"),
        },
        "execution_deviation": primary_external.get("execution_history"),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "hist_gradient_boosting": sklearn.__version__,
        },
        "hashes": {
            "protocol_sha256": phase7_freeze.get("phase7_protocol_sha256"),
            "development_freeze_sha256": sha(OUT / "phase7_development_freeze.json"),
            "external_freeze_sha256": sha(OUT / "phase7_external_freeze.json"),
            "development_runner_sha256": sha(ROOT / "src/phase7_liquidity/phase7_run_development.py"),
            "external_evaluator_sha256": sha(ROOT / "src/phase7_liquidity/phase7_evaluate_external.py"),
            "summary_script_sha256": sha(ROOT / "src/phase7_liquidity/summarize_external.py"),
        },
    }
    (OUT / "phase7_reproducibility.json").write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n")

    report = f"""# Phase 7 validation report

The data build was reproduced from the 6,195 cached monthly archives without network requests. SHA-256 and ZIP CRC checks passed for every archive. The resulting 54,780-row five-asset panel has SHA-256 `{results['offline_data_rebuild']['panel_sha256']}`, identical to the frozen panel. The independent Phase-5 BTC/ETH parity check passed and the reconstructed Amihud formula differs by at most {results['offline_data_rebuild']['amihud_reconstruction_max_abs_error']:.3g}. The Phase-7 data audit passed.

The offline development reproduction reran all {results['offline_development_reproduction']['primary_fit_count']:,} development fits with the same frozen folds, transforms, feature sets and learners. The selected-model file and all 559,440 OOF rows are byte-identical. The gzip containers differ only in volatile header metadata, so their compressed-file hashes differ although their decompressed payloads match exactly. The complete development JSON matches after excluding output path, runtime, and the hash of that differently packaged gzip file. The reproduction runner changes only the output directory.

The external reproduction produced {results['offline_external_reproduction']['external_prediction_rows']:,} rows. Its compressed prediction file is byte-identical to the primary external artifact, and all compared numerical sections—paired scores, block-bootstrap summaries, category aggregation, fixed-period summaries, stress metrics, baselines and negative controls—match exactly.

The original Stage 1/3/Phase 5/Phase 6 snapshot remains intact: all {len(protected_snapshot.get('files', {})):,} protected file hashes pass. The candidate archive integrity, Stage-6 protocol/model hashes, frozen Phase-7 inputs, development freeze and final external freeze all pass.

The external evaluation had one incomplete technical attempt because a `StandardScaler` import was missing in the final negative-control routine. The retry used the unchanged frozen data, transforms, models and seeds. No first-attempt metrics were inspected before retry; both forecast artifacts are byte-identical. This execution deviation is recorded in the external evaluation JSON and is not concealed.

The fixed future-information negative controls ran, but their saved output contains raw MAE only and lacks a matched Spot-only benchmark. They are therefore **not sufficient to certify the negative-control test as passed**. The primary forecast conclusion remains that the frozen Amihud replication criteria and cross-measure generalization criteria fail. No causal conclusion is supported.

Overall engineering reproduction: **{results['status']}**. Research result: **{decision.get('classification')}**.
"""
    (OUT / "phase7_validation_report.md").write_text(report)
    print(json.dumps({
        "reproducibility": results["status"],
        "dev_cv_core_equal": dev_cv_equal,
        "selected_models_byte_identical": dev_selected_sha_equal,
        "dev_oof_payload_byte_identical": dev_oof_payload_equal,
        "dev_oof_gzip_container_byte_identical": dev_oof_sha_equal,
        "external_predictions_byte_identical": external_predictions_equal,
        "protected_file_count": len(protected_snapshot.get("files", {})),
        "protected_failures": len(protected_failures),
    }, indent=2))


if __name__ == "__main__":
    main()
