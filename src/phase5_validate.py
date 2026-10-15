#!/usr/bin/env python3
"""Offline integrity and construction checks for Phase 5 artifacts.

This validator does not fit any discovery or prediction model and does not
read outcome results. It writes only new Phase-5 JSON artifacts.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "phase5_dynamic_discovery"
DATASET = OUT / "dataset_4h.jsonl"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")) + "\n"
    path.write_text(payload, encoding="utf-8")


def feature_definitions():
    common = {
        "frequency": "4-hour UTC",
        "sample": "2021-12-01 00:00 UTC through 2024-03-03 20:00 UTC interval starts",
        "availability": "Computed only from observations timestamped within or before the interval; no future interval values.",
        "missingness": "No interpolation, forward-fill (except the explicitly stated funding state), or synthetic observations.",
    }
    rows = [
        {
            "variable": "funding_rate_state", "layer": "derivatives", "raw_or_derived": "derived state",
            "formula": "last_funding_rate at the greatest settlement calc_time <= interval_end; valid only when interval_end - calc_time is in [0, 12 hours]",
            "unit": "native decimal funding rate per settlement interval",
            "semantics": "Most recently settled funding rate available by the four-hour interval end. Repeated values reflect settlement cadence; no jitter.",
            "availability_rule": "as-of interval end", "method_caveat": "Repeated/discrete values are retained only for methods that validly support them; prior daily TE tie/atom gate is unchanged.",
        },
        {
            "variable": "oi_log_change", "layer": "derivatives", "raw_or_derived": "derived",
            "formula": "log(oi_close_t / oi_close_{t-1}); oi_close is the final valid sum_open_interest snapshot in an interval, retained only if all 48 five-minute snapshots are valid",
            "unit": "log change per four hours", "semantics": "Change in the provider-reported open-interest quantity; not a notional level.",
            "availability_rule": "interval close; requires adjacent valid intervals", "method_caveat": "The source's historical timestamp mapping is the Stage-1 certified mapping; this build applies that mapping and drops unmapped snapshots.",
        },
        {
            "variable": "futures_taker_imbalance", "layer": "derivatives", "raw_or_derived": "derived",
            "formula": "(2 * sum(taker_buy_base_volume) - sum(base_volume)) / sum(base_volume), requiring positive denominator and all 48 five-minute bars",
            "unit": "dimensionless, nominally [-1, 1]", "semantics": "Binance USDⓈ-M perpetual kline taker-buy base-volume share transformed to signed imbalance.",
            "availability_rule": "complete four-hour interval", "method_caveat": "The formula uses the certified kline fields; not a separate aggressor sell-volume feed.",
        },
        {
            "variable": "spot_return", "layer": "spot", "raw_or_derived": "derived",
            "formula": "log(spot_close_t / spot_close_{t-1}); spot_close is the last valid five-minute close in the four-hour interval",
            "unit": "log return per four hours", "semantics": "Binance BTCUSDT or ETHUSDT spot return.",
            "availability_rule": "interval close", "method_caveat": "Same-venue Binance spot represents venue-specific conditions, not global spot.",
        },
        {
            "variable": "spot_realized_volatility", "layer": "spot", "raw_or_derived": "derived",
            "formula": "sqrt(sum(r_5m_j^2 for the 48 within-interval five-minute close-to-close log returns))",
            "unit": "unannualized four-hour realized volatility", "semantics": "Intraday realized volatility, requiring all 48 interval returns and complete bars.",
            "availability_rule": "interval close", "method_caveat": "The first five-minute return uses the immediately preceding cached five-minute close.",
        },
        {
            "variable": "spot_downside_semivariance", "layer": "spot", "raw_or_derived": "derived",
            "formula": "sum(min(r_5m_j, 0)^2 for the 48 within-interval five-minute close-to-close log returns)",
            "unit": "squared log-return per four hours", "semantics": "Realized downside semivariance from negative five-minute returns only.",
            "availability_rule": "interval close", "method_caveat": "Requires all 48 interval returns and complete bars.",
        },
        {
            "variable": "spot_amihud", "layer": "spot", "raw_or_derived": "derived",
            "formula": "1e6 * abs(spot_return) / spot_quote_volume, requiring positive four-hour quote_volume",
            "unit": "scaled absolute four-hour log return per USDT quote-volume unit", "semantics": "Same-pair Binance spot Amihud proxy; fixed readability multiplier 1e6.",
            "availability_rule": "interval close", "method_caveat": "USDT quote volume is not asserted to equal USD volume.",
        },
        {
            "variable": "spot_taker_imbalance", "layer": "spot", "raw_or_derived": "derived",
            "formula": "(2 * sum(taker_buy_base_volume) - sum(base_volume)) / sum(base_volume), requiring positive denominator and all 48 five-minute bars",
            "unit": "dimensionless, nominally [-1, 1]", "semantics": "Binance spot kline taker-buy base-volume share transformed to signed imbalance.",
            "availability_rule": "complete four-hour interval", "method_caveat": "The formula uses the certified kline fields; not a separate aggressor sell-volume feed.",
        },
        {
            "variable": "spot_quote_volume_growth", "layer": "spot", "raw_or_derived": "derived",
            "formula": "log(spot_quote_volume_t) - log(spot_quote_volume_{t-1}); both four-hour quote volumes must be positive",
            "unit": "log change per four hours", "semantics": "Change in Binance BTCUSDT or ETHUSDT spot quote volume.",
            "availability_rule": "interval close; requires adjacent positive-volume intervals", "method_caveat": "USDT quote units are retained as reported; no USD conversion is implied.",
        },
        {
            "variable": "basis_mean", "layer": "derivatives", "raw_or_derived": "derived secondary state",
            "formula": "mean(mark_price_close_j / index_price_close_j - 1 for 48 matched five-minute observations)",
            "unit": "raw, nonannualized ratio", "semantics": "Four-hour mean of the contemporaneous Binance futures mark/index premium.",
            "availability_rule": "complete matched interval", "method_caveat": "Retained as an optional secondary feature; it is not a funding rate or annualized futures basis.",
        },
    ]
    return {"status": "FROZEN_FEATURE_DEFINITIONS", "common": common,
            "variables": rows, "no_new_raw_variables": True,
            "no_model_estimation": True}


def close(a, b, atol=2e-10, rtol=2e-8):
    return np.isclose(a, b, atol=atol, rtol=rtol, equal_nan=True)


def main():
    checks = {}
    errors = []

    required = [
        "phase5_protocol.json", "dataset_4h.jsonl", "dataset_validation.json",
        "pcmci_results.json", "dynotears_results.json", "notears_mlp_results.json",
        "neural_granger_results.json", "var_lingam_results.json",
        "bootstrap_stability.json", "cross_method_consensus.json",
        "cross_asset_replication.json", "reverse_time_checks.json",
        "holdout_confirmation.json", "final_edges.json", "method_diagnostics.json",
        "reproducibility.json", "offline_reproduction.json",
        "research_summary.md", "phase5_run.log",
    ]
    missing = [x for x in required if not (OUT / x).is_file()]
    checks["required_artifacts_present"] = {"pass": not missing, "missing": missing}
    if missing:
        errors.append(f"Missing required Phase-5 outputs: {missing}")

    # Persist the formula registry independently of the frozen dataset/protocol.
    definitions = feature_definitions()
    write_json(OUT / "variable_definitions.json", definitions)
    checks["feature_definition_registry"] = {"pass": len(definitions["variables"]) == 10,
                                             "count": len(definitions["variables"])}

    raw_lines = DATASET.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in raw_lines]
    df = pd.DataFrame(rows)
    df["period_start_utc"] = pd.to_datetime(df["period_start_utc"], utc=True)
    df["period_end_utc"] = pd.to_datetime(df["period_end_utc"], utc=True)
    df["funding_last_event_utc"] = pd.to_datetime(df["funding_last_event_utc"], utc=True, format="mixed")
    start = pd.Timestamp("2021-12-01T00:00:00Z")
    split = pd.Timestamp("2023-07-01T00:00:00Z")
    last_start = pd.Timestamp("2024-03-03T20:00:00Z")
    keys_unique = not df.duplicated(["asset", "period_start_utc"]).any()
    dates = sorted(df.period_start_utc.unique())
    dates_ok = len(dates) == 4944 and dates[0] == start and dates[-1] == last_start
    per_asset = df.groupby("asset").size().to_dict()
    split_counts = {}
    for asset, group in df.groupby("asset"):
        split_counts[asset] = {
            "discovery": int((group.period_start_utc < split).sum()),
            "confirmation": int((group.period_start_utc >= split).sum()),
        }
    split_ok = all(v == {"discovery": 3462, "confirmation": 1482}
                   for v in split_counts.values())
    end_ok = bool((df.period_end_utc <= pd.Timestamp("2024-03-04T00:00:00Z")).all())
    split_label_ok = bool(((df.period_start_utc < split) == df.discovery_sample).all())
    checks["panel_calendar"] = {
        "pass": len(df) == 9888 and keys_unique and dates_ok and per_asset == {"BTC": 4944, "ETH": 4944} and split_ok and end_ok and split_label_ok,
        "rows": int(len(df)), "asset_rows": {k: int(v) for k, v in per_asset.items()},
        "unique_periods_per_asset": int(len(dates)), "unique_keys": keys_unique,
        "first_period_start": dates[0].isoformat(), "last_period_start": dates[-1].isoformat(),
        "split_counts": split_counts, "period_end_within_boundary": end_ok,
        "split_labels_correct": split_label_ok,
    }
    if not checks["panel_calendar"]["pass"]:
        errors.append("4-hour panel calendar, uniqueness, or fixed split did not validate")

    # Construction checks are limited to mechanically testable formulas.
    formula_checks = {}
    imbalances = ["futures_taker_imbalance", "spot_taker_imbalance"]
    for col in imbalances:
        x = pd.to_numeric(df[col], errors="coerce")
        formula_checks[f"{col}_bounds"] = bool(x.dropna().between(-1 - 1e-9, 1 + 1e-9).all())
    funding_age = pd.to_numeric(df.funding_age_hours, errors="coerce")
    formula_checks["funding_asof_age"] = bool(funding_age.notna().all() and funding_age.between(0, 12).all() and (df.funding_last_event_utc <= df.period_end_utc).all())

    max_err = {"spot_return": 0.0, "spot_amihud": 0.0, "spot_quote_volume_growth": 0.0,
               "oi_log_change": 0.0, "futures_taker_imbalance": 0.0, "spot_taker_imbalance": 0.0}
    all_formula_ok = True
    for asset, g in df.sort_values("period_start_utc").groupby("asset", sort=False):
        g = g.sort_values("period_start_utc")
        def compare(name, actual, expected, valid):
            nonlocal all_formula_ok
            actual = np.asarray(actual, dtype=float)
            expected = np.asarray(expected, dtype=float)
            valid = np.asarray(valid, dtype=bool) & np.isfinite(actual) & np.isfinite(expected)
            if not valid.any():
                all_formula_ok = False
                return
            err = np.abs(actual[valid] - expected[valid])
            max_err[name] = max(max_err[name], float(err.max()))
            if not close(actual[valid], expected[valid]).all():
                all_formula_ok = False

        close_px = pd.to_numeric(g.spot_raw_close, errors="coerce").to_numpy()
        prev_close = np.roll(close_px, 1); prev_close[0] = np.nan
        exp_ret = np.log(close_px / prev_close)
        actual_ret = pd.to_numeric(g.spot_return, errors="coerce").to_numpy()
        compare("spot_return", actual_ret, exp_ret, np.isfinite(exp_ret))

        quote = pd.to_numeric(g.spot_raw_quote_volume, errors="coerce").to_numpy()
        expected_amihud = 1e6 * np.abs(actual_ret) / quote
        actual_amihud = pd.to_numeric(g.spot_amihud, errors="coerce").to_numpy()
        compare("spot_amihud", actual_amihud, expected_amihud,
                np.isfinite(expected_amihud) & (quote > 0))

        prev_q = np.roll(quote, 1); prev_q[0] = np.nan
        exp_qgrowth = np.log(quote / prev_q)
        actual_qgrowth = pd.to_numeric(g.spot_quote_volume_growth, errors="coerce").to_numpy()
        compare("spot_quote_volume_growth", actual_qgrowth, exp_qgrowth,
                np.isfinite(exp_qgrowth) & (quote > 0) & (prev_q > 0))

        oi = pd.to_numeric(g.oi_close, errors="coerce").to_numpy()
        prev_oi = np.roll(oi, 1); prev_oi[0] = np.nan
        exp_oigrowth = np.log(oi / prev_oi)
        actual_oigrowth = pd.to_numeric(g.oi_log_change, errors="coerce").to_numpy()
        compare("oi_log_change", actual_oigrowth, exp_oigrowth,
                np.isfinite(exp_oigrowth) & (oi > 0) & (prev_oi > 0))

        for market, var, prefix in [
            ("futures", "futures_taker_imbalance", "futures_raw_"),
            ("spot", "spot_taker_imbalance", "spot_raw_"),
        ]:
            total = pd.to_numeric(g[f"{prefix}base_volume"], errors="coerce").to_numpy()
            buy = pd.to_numeric(g[f"{prefix}taker_buy_base_volume"], errors="coerce").to_numpy()
            exp = (2 * buy - total) / total
            actual = pd.to_numeric(g[var], errors="coerce").to_numpy()
            compare(var, actual, exp, np.isfinite(exp) & (total > 0))
    checks["formula_reconstruction"] = {
        "pass": all_formula_ok and all(formula_checks.values()),
        "max_absolute_error": max_err, "bounds_and_timing": formula_checks,
        "note": "Volatility, downside, basis, and their intrabar formulas are verified by source-row completeness and aggregation code review; raw 5-minute return vectors are not duplicated into the output JSONL.",
    }
    if not checks["formula_reconstruction"]["pass"]:
        errors.append("A reconstructed formula, imbalance bound, or funding as-of check failed")

    dataset_validation = read_json(OUT / "dataset_validation.json")
    repro = read_json(OUT / "reproducibility.json")
    protocol = read_json(OUT / "phase5_protocol.json")
    method_names = {
        "pcmci_results.json": "PCMCI+",
        "dynotears_results.json": "DYNOTEARS / temporal NOTEARS",
        "notears_mlp_results.json": "NOTEARS-MLP",
        "neural_granger_results.json": "Neural Granger / cMLP",
        "var_lingam_results.json": "VAR-LiNGAM",
    }
    method_pass = True
    method_statuses = {}
    for filename, name in method_names.items():
        obj = read_json(OUT / filename)
        ok = obj.get("method") == name and obj.get("status") == "FAIL" and obj.get("fit_attempted") is False
        method_statuses[name] = {"status": obj.get("status"), "fit_attempted": obj.get("fit_attempted"), "valid_unavailable_record": ok}
        method_pass &= ok
    no_fit = repro.get("fit_count") == 0 and all(x["fit_attempted"] is False for x in method_statuses.values())
    no_artifact_outputs = not any(p.suffix.lower() in {".png", ".svg", ".pdf", ".csv", ".html"} for p in OUT.iterdir())
    checks["method_execution_state"] = {
        "pass": method_pass and no_fit and no_artifact_outputs,
        "methods": method_statuses, "fit_count": repro.get("fit_count"),
        "no_figures_or_presentation_tables": no_artifact_outputs,
        "bootstrap_status": read_json(OUT / "bootstrap_stability.json").get("status"),
        "holdout_status": read_json(OUT / "holdout_confirmation.json").get("status"),
        "final_classification": read_json(OUT / "final_edges.json").get("phase_classification"),
    }
    if not checks["method_execution_state"]["pass"]:
        errors.append("Method availability / no-fit / no-figure audit failed")

    protocols = protocol.get("lag_set") == [1, 2, 3, 4, 5, 6] and protocol.get("stable_edge_rules") == {"stable": 0.7, "very_stable": 0.85, "weak_consensus": 2, "moderate_consensus": 3, "strong_consensus": 4}
    checks["frozen_protocol"] = {"pass": bool(protocols), "sha256": sha(OUT / "phase5_protocol.json"),
                                 "lag_set": protocol.get("lag_set"), "created_before_fits": protocol.get("status") == "FROZEN_BEFORE_METHOD_FITS"}
    if not checks["frozen_protocol"]["pass"]:
        errors.append("Frozen Phase-5 protocol integrity check failed")

    # Check all cached inputs and the prior-stage files frozen by the Phase-5 build.
    source_hash_results = []
    for source in dataset_validation["source_file_hashes"]:
        p = ROOT / source["path"]
        ok = p.is_file() and sha(p) == source["sha256"]
        source_hash_results.append({"path": source["path"], "pass": ok})
    prev_hash_results = []
    for rel, expected in repro["previous_stage_artifacts_sha256"].items():
        p = ROOT / rel
        ok = p.is_file() and sha(p) == expected
        prev_hash_results.append({"path": rel, "pass": ok})
    hashes_ok = all(x["pass"] for x in source_hash_results + prev_hash_results)
    checks["immutability"] = {"pass": hashes_ok, "cached_inputs": source_hash_results,
                               "previous_stage_artifacts": prev_hash_results}
    if not hashes_ok:
        errors.append("One or more source/prior-stage hashes changed")

    offline_ok = (repro.get("status") == "PASS" and repro.get("offline_rebuild_byte_identical") is True
                  and repro.get("cached_intraday_sources_unchanged") is True
                  and repro.get("network_data_downloads") == 0)
    offline_run = read_json(OUT / "offline_reproduction.json")
    offline_ok = offline_ok and offline_run.get("status") == "PASS" and offline_run.get("byte_identical") is True and offline_run.get("cached_source_hashes_unchanged") is True and offline_run.get("network_data_downloads") == 0 and offline_run.get("model_fits") == 0
    checks["offline_reproduction"] = {"pass": offline_ok,
                                      "dataset_sha256": repro.get("dataset_sha256"),
                                      "offline_rebuild_byte_identical": repro.get("offline_rebuild_byte_identical"),
                                      "independent_verify_only_pass": offline_run.get("status") == "PASS",
                                      "network_data_downloads": repro.get("network_data_downloads")}
    if not offline_ok:
        errors.append("Offline reproduction recorded failure")

    # Write the validation result after all checks; status concerns artifact
    # integrity and data mechanics, not availability of discovery estimators.
    result = {
        "phase": "PHASE_5_DYNAMIC_STRUCTURE_DISCOVERY",
        "validation_status": "PASS" if not errors else "FAIL",
        "data_and_artifact_validation": "PASS" if not errors else "FAIL",
        "discovery_estimation_status": "NOT_RUN_VALIDATED_METHODS_UNAVAILABLE",
        "errors": errors,
        "checks": checks,
        "input_dataset_sha256": sha(DATASET),
        "protocol_sha256": sha(OUT / "phase5_protocol.json"),
    }
    write_json(OUT / "phase5_validation.json", result)
    print(f"PHASE5_VALIDATION={result['validation_status']} rows={len(df)} errors={len(errors)}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
