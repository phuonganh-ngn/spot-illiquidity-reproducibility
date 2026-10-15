from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "phase9_external_qml_replication"
START = pd.Timestamp("2024-10-06T20:00:00Z")
END = pd.Timestamp("2026-10-06T20:00:00Z")
BAR = pd.Timedelta(hours=4)
TARGET_END_OFFSET = pd.Timedelta(hours=28)
PURGE_BARS = 6
ASSETS = ("BTC", "ETH")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def feature_arrays(g: pd.DataFrame, registry: dict) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    s0 = registry["registry"]["spot_only"]["base_variables"]
    d = registry["registry"]["spot_plus_derivatives"]["adds_base_variables"]
    static = registry["registry"]["spot_plus_derivatives"]["static_derivative_features"]
    cal = [*[f"utc_slot_{i}" for i in range(1, 6)], *[f"weekday_{i}" for i in range(1, 7)], "funding_schedule_window"]
    g = g.sort_values("period_start_utc").reset_index(drop=True)
    x0, x1 = {}, {}
    for lag in range(5, -1, -1):
        for col in s0:
            x0[f"{col}__lag{lag}"] = pd.to_numeric(g[col], errors="coerce").shift(lag)
        for col in d:
            x1[f"{col}__lag{lag}"] = pd.to_numeric(g[col], errors="coerce").shift(lag)
    for col in cal:
        x0[col] = pd.to_numeric(g[col], errors="coerce")
        x1[col] = pd.to_numeric(g[col], errors="coerce")
    for col in static:
        x1[col] = pd.to_numeric(g[col], errors="coerce")
    a, b = pd.DataFrame(x0), pd.DataFrame(x1)
    y = np.log1p(pd.to_numeric(g.future_amihud_24h_mean, errors="coerce").to_numpy(float))
    valid = np.isfinite(a.to_numpy(float)).all(axis=1) & np.isfinite(b.to_numpy(float)).all(axis=1) & np.isfinite(y)
    return a, b, valid


def main() -> None:
    require(not (OUT / "pre_model_freeze_manifest.json").exists(), "Pre-model freeze already exists; refusing to overwrite.")
    required_result_names = [
        "spot_oof_predictions.parquet", "residual_training_data.parquet",
        "holdout_spotonly_predictions.parquet", "holdout_mlp_corrected_predictions.parquet",
        "holdout_qml_corrected_predictions.parquet", "primary_metrics.json", "delta_q_results.json",
        "delta_i_results.json", "bootstrap_42.json", "bootstrap_84.json", "primary_holm.json",
        "negative_controls.json", "btc_decision.json", "eth_decision.json", "paper_level_decision.json",
        "primary_seed_checkpoints.jsonl", "placebo_seed_checkpoints.jsonl", "negative_control_predictions.parquet",
    ]
    existing = [name for name in required_result_names if (OUT / name).exists()]
    require(not existing, f"Model-result artifacts already exist before freeze: {existing}")

    amendment_hash = (OUT / "sample_amendment.sha256").read_text().split()[0]
    require(sha256(OUT / "sample_amendment.json") == amendment_hash, "Sample amendment JSON hash mismatch")
    amendment_md_hash = (OUT / "sample_amendment.sha256").read_text().splitlines()[1].split()[0]
    require(sha256(OUT / "sample_amendment.md") == amendment_md_hash, "Sample amendment Markdown hash mismatch")
    sample_hash = (OUT / "sample_window.sha256").read_text().split()[0]
    require(sha256(OUT / "sample_window.json") == sample_hash, "Sample window hash mismatch")

    sample = json.loads((OUT / "sample_window.json").read_text(encoding="utf-8"))
    require(pd.Timestamp(sample["T_start"]) == START and pd.Timestamp(sample["T_end"]) == END, "Unexpected frozen sample dates")
    require(sample["expected_4h_timestamps"] == 4381, "Unexpected sample timestamp count")
    semantic = json.loads((OUT / "semantic_parity.json").read_text(encoding="utf-8"))
    semantic_statuses = {k: v.get("status") for k, v in semantic["variables"].items()}
    require(all(v == "PASS" for v in semantic_statuses.values()), f"Semantic parity gate failed: {semantic_statuses}")

    raw_integrity = json.loads((OUT / "raw_integrity_report.json").read_text(encoding="utf-8"))
    require(raw_integrity.get("status") == "PASS", "Raw integrity gate failed")
    require(raw_integrity.get("required_archive_count") == 2924, "Unexpected number of required raw archives")
    require(raw_integrity.get("all_required_files_have_sha256"), "One or more raw archives lack SHA-256")
    raw_hash_verification = json.loads((OUT / "raw_hash_verification.json").read_text(encoding="utf-8"))
    require(raw_hash_verification.get("status") == "PASS", "Required-archive SHA-256 verification failed")
    require(raw_hash_verification.get("required_file_count") == 2924, "Raw SHA-256 check did not cover all required archives")
    require(raw_hash_verification.get("verified_file_count") == 2924, "Not all required raw archives passed SHA-256 verification")

    registry = json.loads((OUT / "feature_registry_bybit_frozen.json").read_text(encoding="utf-8"))
    target = json.loads((OUT / "target_registry.json").read_text(encoding="utf-8"))
    require(target.get("target_transform") == "T2_log1p", "Target transform differs from the frozen study")
    require(target.get("amihud_formula") == "1e6 * abs(4-hour log return) / positive 4-hour spot quote volume in USDT", "Target formula differs from frozen definition")

    panels = {}
    audits = {}
    valid_dates = {}
    for asset in ASSETS:
        path = OUT / f"bybit_{asset.lower()}_4h_panel.parquet"
        require(path.is_file(), f"Missing frozen {asset} panel")
        p = pd.read_parquet(path)
        p["period_start_utc"] = pd.to_datetime(p.period_start_utc, utc=True)
        expected_grid = pd.date_range(START, END, freq=BAR)
        require(len(p) == 4381, f"{asset} panel row count differs from frozen 24-month calendar")
        require(p.period_start_utc.is_unique, f"{asset} panel has duplicate 4h timestamps")
        require(pd.DatetimeIndex(p.period_start_utc).equals(expected_grid), f"{asset} panel calendar is incomplete or misaligned")
        _, _, valid = feature_arrays(p, registry)
        valid_dates[asset] = pd.DatetimeIndex(p.loc[valid, "period_start_utc"])
        panels[asset] = p
        audits[asset] = json.loads((OUT / "panel_audit.json").read_text())["assets"][asset]["audit"]
        require(audits[asset]["duplicate_period_start"] == 0, f"{asset} panel audit reports duplicates")
        for variable in ("spot_return", "spot_realized_volatility", "spot_amihud", "future_amihud_24h_mean",
                         "funding_rate_state", "oi_log_change", "basis_mean", "futures_taker_imbalance"):
            coverage = audits[asset]["variable_coverage"][variable]["coverage_pct"]
            require(coverage >= 95.0, f"{asset} required feature {variable} coverage is only {coverage:.2f}%")

    common_dates = valid_dates["BTC"].intersection(valid_dates["ETH"]).sort_values()
    require(len(common_dates) > 0, "No complete common BTC/ETH dates")
    elapsed = END - START
    unaligned_cut = START + elapsed * 0.70
    cut_ns = int(round(unaligned_cut.value / BAR.value)) * BAR.value
    cut = pd.Timestamp(cut_ns, tz="UTC")
    require(START < cut < END and (cut - START) % BAR == pd.Timedelta(0), "70/30 calendar split did not map to a 4h boundary")

    primary_train_end = cut - pd.Timedelta(hours=24)
    train_dates = common_dates[common_dates < primary_train_end]
    purge_dates = common_dates[(common_dates >= primary_train_end) & (common_dates < cut)]
    holdout_dates = common_dates[common_dates >= cut]
    require(len(train_dates) > 0 and len(holdout_dates) > 0, "Empty training or holdout sample")
    require(all((d + TARGET_END_OFFSET) <= cut for d in train_dates), "A training target extends into the holdout")
    require(all(d >= cut for d in holdout_dates), "A holdout origin precedes T_cut")

    warmup_n = int(np.ceil(0.50 * len(train_dates)))
    remaining = np.arange(warmup_n, len(train_dates))
    blocks = [b for b in np.array_split(remaining, 5) if len(b)]
    folds = []
    train_list = train_dates.to_list()
    for i, block in enumerate(blocks, 1):
        first = train_dates[int(block[0])]
        last = train_dates[int(block[-1])]
        eligible_train = train_dates[train_dates < first - pd.Timedelta(hours=24)]
        fold_purge = train_dates[(train_dates >= first - pd.Timedelta(hours=24)) & (train_dates < first)]
        require(len(eligible_train) > 0 and len(fold_purge) <= PURGE_BARS, f"OOF fold {i} failed horizon purge")
        folds.append({
            "fold": f"OOF{i}",
            "training_first": eligible_train[0].isoformat(),
            "training_last_inclusive": eligible_train[-1].isoformat(),
            "training_rows": int(len(eligible_train)),
            "purge_first": fold_purge[0].isoformat(),
            "purge_last_inclusive": fold_purge[-1].isoformat(),
            "purge_rows": int(len(fold_purge)),
            "validation_first": first.isoformat(),
            "validation_last_inclusive": last.isoformat(),
            "validation_rows": int(len(block)),
        })
    split = {
        "venue": "Bybit",
        "assets": list(ASSETS),
        "T_start": START.isoformat(),
        "T_cut": cut.isoformat(),
        "T_end_inclusive": END.isoformat(),
        "split_rule": "One common 70/30 chronological calendar split from T_start to T_end; mapped to nearest 4h UTC boundary before model fitting.",
        "train_interval": {"start_inclusive": START.isoformat(), "end_exclusive": cut.isoformat()},
        "purge_interval": {"start_inclusive": primary_train_end.isoformat(), "end_exclusive": cut.isoformat(), "expected_4h_rows": 6},
        "holdout_interval": {"start_inclusive": cut.isoformat(), "end_inclusive": END.isoformat()},
        "purge_rule": "A training origin is eligible only when origin + 28h target end <= T_cut; final six origins before T_cut are purged.",
        "calendar_timestamps_per_asset": 4381,
        "common_complete_rows": int(len(common_dates)),
        "common_complete_first": common_dates[0].isoformat(),
        "common_complete_last": common_dates[-1].isoformat(),
        "usable_observations": {
            "BTC": {"training_after_purge": int(np.sum(valid_dates["BTC"] < primary_train_end)),
                    "purge": int(np.sum((valid_dates["BTC"] >= primary_train_end) & (valid_dates["BTC"] < cut))),
                    "holdout": int(np.sum(valid_dates["BTC"] >= cut))},
            "ETH": {"training_after_purge": int(np.sum(valid_dates["ETH"] < primary_train_end)),
                    "purge": int(np.sum((valid_dates["ETH"] >= primary_train_end) & (valid_dates["ETH"] < cut))),
                    "holdout": int(np.sum(valid_dates["ETH"] >= cut))},
            "common_training_after_purge": int(len(train_dates)),
            "common_purge": int(len(purge_dates)),
            "common_holdout": int(len(holdout_dates)),
        },
        "oof_schedule": {"rule": "50% initial expanding warm-up; five equal chronological validation blocks; six 4h origins embargoed at each fold boundary.",
                         "initial_warmup_rows": warmup_n, "folds": folds},
        "holdout_predictions_unopened": True,
        "model_fitting_performed": False,
    }
    write_json(OUT / "temporal_split.json", split)
    (OUT / "temporal_split.sha256").write_text(f"{sha256(OUT / 'temporal_split.json')}  temporal_split.json\n", encoding="utf-8")

    env = {"python": platform.python_version(), "platform": platform.platform(), "packages": {}}
    for package in ["numpy", "pandas", "scikit-learn", "catboost", "xgboost", "torch", "pennylane", "pyarrow", "threadpoolctl"]:
        try:
            env["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            env["packages"][package] = None
    env["qml_backend"] = "PennyLane default.qubit exact CPU statevector; shots=null"
    env["model_fitting_performed"] = False
    write_json(OUT / "software_environment.json", env)

    qml_fixture = json.loads((OUT / "qml_fixture_validation.json").read_text(encoding="utf-8"))
    require(qml_fixture.get("status") == "VALID", "Existing deterministic QML fixture validation is not VALID")

    code_files = [
        "src/phase9a_download_24m.py", "src/phase9a_build_24m.py",
        "src/phase9a_freeze_before_models.py", "src/phase9a_external_qml_estimate.py",
        "src/phase9a_verify_raw_hashes.py",
        "src/phase9a_prefetch_asset.py",
        "src/targeted_24h_amihud/qml_models.py", "src/targeted_24h_amihud/classical_models.py",
        "src/phase9_external_qml_estimate.py", "src/phase9_external_qml_replication/offline_guard/sitecustomize.py",
    ]
    missing_code = [p for p in code_files if not (ROOT / p).is_file()]
    require(not missing_code, f"Missing code required by frozen pipeline: {missing_code}")
    artifact_files = [
        "sample_amendment.json", "sample_amendment.md", "sample_window.json",
        "sample_amendment.sha256", "sample_window.sha256", "raw_inventory_before_amendment.json",
        "phase9_protocol.md", "phase9_protocol.json", "phase9_protocol.sha256", "volume_parity_probe.json",
        "required_download_manifest.json", "raw_integrity_report.json", "raw_hash_verification.json", "semantic_parity.json",
        "feature_registry_bybit_frozen.json", "target_registry.json", "temporal_split.json",
        "frozen_btc_hybrid4q.json", "frozen_eth_qnn6q.json", "matched_mlp_config.json",
        "matched_classical_models.json", "seed_registry.json", "qml_fixture_validation.json", "software_environment.json",
        "panel_audit.json", "flow_parallel_validation.json", "prefetch_btc_data.json", "prefetch_eth_data.json",
        "bybit_btc_4h_panel.parquet", "bybit_eth_4h_panel.parquet",
    ]
    hashes = {name: sha256(OUT / name) for name in artifact_files}
    hashes.update({code: sha256(ROOT / code) for code in code_files})
    prior_before = json.loads((OUT / "prior_artifact_hashes_before.json").read_text(encoding="utf-8"))
    prior_changes = []
    prior_records = prior_before.get("artifacts") if isinstance(prior_before, dict) else None
    require(isinstance(prior_records, list), "Immutable prior-artifact hash snapshot has an unexpected schema")
    for record in prior_records:
        rel, expected = record["file"], record["sha256"]
        p = ROOT / rel
        if not p.is_file() or sha256(p) != expected:
            prior_changes.append(rel)
    require(not prior_changes, f"Immutable prior artifacts changed: {prior_changes[:10]}")

    manifest = {
        "status": "FROZEN_BEFORE_MODEL_RESULTS",
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "protocol_amendment_sha256": amendment_hash,
        "sample_window_sha256": sample_hash,
        "forecast_results_viewed": False,
        "model_fitting_performed": False,
        "panel_constructed_after_sample_amendment": True,
        "venue": "Bybit",
        "sample_window": {"start": START.isoformat(), "end_inclusive": END.isoformat(), "expected_4h_origins_per_asset": 4381},
        "temporal_split_sha256": sha256(OUT / "temporal_split.json"),
        "semantic_parity": semantic_statuses,
        "qml_fixture_status": qml_fixture["status"],
        "primary_models": {
            "BTC": {"configuration_file": "frozen_btc_hybrid4q.json", "model": "Hybrid QNN 4 qubits", "seeds": json.loads((OUT / "seed_registry.json").read_text())["BTC"]},
            "ETH": {"configuration_file": "frozen_eth_qnn6q.json", "model": "QNN 6 qubits", "seeds": json.loads((OUT / "seed_registry.json").read_text())["ETH"]},
            "matched_classical": "Exact frozen one-hidden-layer MLP from matched_mlp_config.json",
            "spot_only": json.loads((OUT / "matched_classical_models.json").read_text())["spot_only"],
        },
        "target": target,
        "feature_registry_sha256": sha256(OUT / "feature_registry_bybit_frozen.json"),
        "hyperparameter_search_on_bybit": False,
        "panel_hashes": {a: sha256(OUT / f"bybit_{a.lower()}_4h_panel.parquet") for a in ASSETS},
        "input_hashes": hashes,
        "prior_stage_immutability": "PASS",
        "required_model_result_files_absent_at_freeze": required_result_names,
        "fold_schedule": split["oof_schedule"],
    }
    write_json(OUT / "pre_model_freeze_manifest.json", manifest)
    manifest_hash = sha256(OUT / "pre_model_freeze_manifest.json")
    (OUT / "pre_model_freeze_manifest.sha256").write_text(f"{manifest_hash}  pre_model_freeze_manifest.json\n", encoding="utf-8")
    validation = {
        "status": "PASS",
        "protocol_amendment_hashes": "PASS",
        "sample_window_hash": "PASS",
        "sample_dates_and_calendar": "PASS",
        "raw_integrity": raw_integrity["status"],
        "semantic_parity": semantic_statuses,
        "panel_rows_unique_and_complete": {a: True for a in ASSETS},
        "common_calendar_split": "PASS",
        "horizon_purge": "PASS",
        "qml_fixture": qml_fixture["status"],
        "prior_stage_immutability": "PASS",
        "model_results_absent_before_freeze": True,
        "pre_model_freeze_manifest_sha256": manifest_hash,
        "model_fitting_performed": False,
    }
    write_json(OUT / "pre_model_freeze_validation.json", validation)
    print(json.dumps({"status": validation["status"], "T_start": START.isoformat(), "T_cut": cut.isoformat(),
                      "T_end": END.isoformat(), "common_rows": len(common_dates),
                      "BTC_holdout": split["usable_observations"]["BTC"]["holdout"],
                      "ETH_holdout": split["usable_observations"]["ETH"]["holdout"],
                      "freeze_sha256": manifest_hash, "model_fitting_performed": False}, indent=2))


if __name__ == "__main__":
    main()
