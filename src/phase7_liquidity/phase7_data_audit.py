from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "phase7_liquidity_replication"
PANEL_PATH = ROOT / "data" / "processed" / "phase7" / "market_liquidity_panel_4h.csv.gz"
P5_PATH = ROOT / "outputs" / "phase5_dynamic_discovery" / "dataset_4h.jsonl"
DEV_END = pd.Timestamp("2023-06-30T20:00:00Z")
OLD_TEST_END = pd.Timestamp("2024-03-03T20:00:00Z")
EXTERNAL_START = pd.Timestamp("2024-03-04T00:00:00Z")
SAMPLE_END = pd.Timestamp("2026-09-30T20:00:00Z")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def clean_number(x):
    return float(x) if np.isfinite(x) else None


def summarise(s: pd.Series) -> dict:
    x = pd.to_numeric(s, errors="coerce").to_numpy(float)
    x = x[np.isfinite(x)]
    probs = [0.001, 0.005, 0.01, 0.5, 0.9, 0.99, 0.995, 0.999]
    q = np.quantile(x, probs) if len(x) else np.full(len(probs), np.nan)
    return {
        "n": int(len(x)),
        "mean": clean_number(np.mean(x)) if len(x) else None,
        "sd": clean_number(np.std(x, ddof=1)) if len(x) > 1 else None,
        "zero_count": int(np.sum(x == 0)),
        "negative_count": int(np.sum(x < 0)),
        "nonfinite_count": int(np.sum(~np.isfinite(pd.to_numeric(s, errors="coerce").to_numpy(float)))),
        "quantiles": {str(p): clean_number(v) for p, v in zip(probs, q)},
    }


def main():
    panel = pd.read_csv(PANEL_PATH, parse_dates=["period_start_utc"], low_memory=False)
    panel["period_start_utc"] = pd.to_datetime(panel.period_start_utc, utc=True)
    p5 = pd.read_json(P5_PATH, lines=True)
    p5["period_start_utc"] = pd.to_datetime(p5.period_start_utc, utc=True)

    clean = (panel.asset.isin(["BTC", "ETH"])) & panel.period_start_utc.between(
        pd.Timestamp("2021-12-01T00:00:00Z"), OLD_TEST_END
    )
    p5clean = p5[p5.period_start_utc.between(pd.Timestamp("2021-12-01T00:00:00Z"), OLD_TEST_END)]
    p7clean = panel.loc[clean]
    keys = ["asset", "period_start_utc"]
    p5clean = p5clean.set_index(keys).sort_index()
    p7clean = p7clean.set_index(keys).sort_index()
    common = p5clean.index.intersection(p7clean.index)
    shared = [
        "funding_rate_state", "oi_log_change", "futures_taker_imbalance", "spot_return",
        "spot_realized_volatility", "spot_downside_semivariance", "spot_amihud",
        "spot_taker_imbalance", "spot_quote_volume_growth", "basis_mean",
    ]
    comparisons = {}
    for col in shared:
        x = pd.to_numeric(p5clean.loc[common, col], errors="coerce")
        y = pd.to_numeric(p7clean.loc[common, col], errors="coerce")
        both = x.notna() & y.notna()
        comparisons[col] = {
            "phase5_valid": int(x.notna().sum()),
            "phase7_valid": int(y.notna().sum()),
            "missingness_identical": bool(x.notna().equals(y.notna())),
            "max_absolute_difference": clean_number((x[both] - y[both]).abs().max()) if both.any() else None,
            "mean_absolute_difference": clean_number((x[both] - y[both]).abs().mean()) if both.any() else None,
        }
    parity_pass = len(common) == 9_888 and all(x["missingness_identical"] and (x["max_absolute_difference"] or 0) < 2e-12 for x in comparisons.values())
    write_json(OUT / "phase5_variable_parity.json", {
        "comparison": "rebuilt Phase-7 BTC/ETH 4-hour rows vs frozen Stage-3/Phase-5 Phase-5 dataset",
        "sample_start": "2021-12-01T00:00:00Z",
        "sample_end": "2024-03-03T20:00:00Z",
        "phase5_rows": int(len(p5clean)),
        "phase7_rows": int(len(p7clean)),
        "common_rows": int(len(common)),
        "expected_rows": 9_888,
        "comparisons": comparisons,
        "pass": bool(parity_pass),
    })
    if not parity_pass:
        raise RuntimeError("Frozen Phase-5 parity check failed")

    panel = panel.sort_values(["asset", "period_start_utc"]).reset_index(drop=True)
    key_dups = int(panel.duplicated(["asset", "period_start_utc"]).sum())
    all_dates = pd.date_range("2021-10-01T00:00:00Z", SAMPLE_END, freq="4h")
    expected_per_asset = len(all_dates)
    assets = sorted(panel.asset.unique().tolist())
    asset_audit = {}
    coverage_vars = [
        "spot_return", "spot_rv", "spot_quote_volume", "funding_rate_state", "oi_log_change",
        "futures_taker_imbalance", "basis_mean", "spot_amihud", "future_amihud_24h_mean",
        "future_kyle_abs_lambda_24h", "future_corwin_schultz_spread_24h",
    ]
    for asset in assets:
        g = panel[panel.asset == asset].copy()
        years = {}
        for year, gy in g[(g.period_start_utc >= "2021-12-01T00:00:00Z")].groupby(g.period_start_utc.dt.year):
            years[str(int(year))] = {
                v: {"valid": int(gy[v].notna().sum()), "expected": int(len(gy)), "coverage_pct": clean_number(100 * gy[v].notna().mean())}
                for v in coverage_vars
            }
        sample = g[g.period_start_utc.between("2021-12-01T00:00:00Z", SAMPLE_END)]
        ext = sample[sample.period_start_utc >= EXTERNAL_START]
        dev = sample[sample.period_start_utc <= DEV_END]
        old_test = sample[(sample.period_start_utc > DEV_END) & (sample.period_start_utc <= OLD_TEST_END)]
        assetsummary = {
            "panel_rows": int(len(g)),
            "expected_all_calendar_rows": expected_per_asset,
            "date_first": g.period_start_utc.min().isoformat(),
            "date_last": g.period_start_utc.max().isoformat(),
            "main_period_rows": int(len(sample)),
            "development_rows": int(len(dev)),
            "prior_phase6_test_rows": int(len(old_test)),
            "new_external_rows": int(len(ext)),
            "full_sample_feature_coverage_pct": {v: clean_number(100 * sample[v].notna().mean()) for v in coverage_vars},
            "external_feature_target_coverage_pct": {v: clean_number(100 * ext[v].notna().mean()) for v in coverage_vars},
            "external_target_validity_only": {
                v: {"valid": int(ext[v].notna().sum()), "expected": int(len(ext)), "coverage_pct": clean_number(100 * ext[v].notna().mean()), "zero_count": int((ext[v] == 0).sum()), "negative_count": int((ext[v] < 0).sum())}
                for v in ["future_amihud_24h_mean", "future_kyle_abs_lambda_24h", "future_corwin_schultz_spread_24h"]
            },
            "annual_coverage": years,
        }
        # Inspect outcome distributions only through the pre-external development window.
        dev_by_target = {
            v: summarise(dev[v])
            for v in ["spot_amihud", "future_amihud_24h_mean", "future_kyle_abs_lambda_24h", "future_corwin_schultz_spread_24h"]
        }
        assets_audit = {**assetsummary, "development_target_distributions": dev_by_target}
        asset_audit[asset] = assets_audit

    # Independently reproduce the source Amihud definition from the built OHLCV fields.
    expected_amihud = 1e6 * panel.spot_return.abs() / panel.spot_quote_volume.where(panel.spot_quote_volume > 0)
    valid_a = expected_amihud.notna() & panel.spot_amihud.notna()
    amihud_reproduction = {
        "formula": "1e6 * abs(log(close_t / close_{t-1})) / spot_quote_volume_USDT_t",
        "max_absolute_difference": clean_number((expected_amihud[valid_a] - panel.loc[valid_a, "spot_amihud"]).abs().max()),
        "missingness_identical": bool(expected_amihud.notna().equals(panel.spot_amihud.notna())),
        "volume_unit": "Binance spot quote volume in USDT accumulated over the 4-hour interval",
        "amihud_scale": 1e6,
        "meaning": "price movement per unit of spot quote turnover; larger values indicate greater measured illiquidity",
    }

    # Integrity references: source 5m files, candidate archive SHA/CRC check and frozen Phase-6 hashes.
    build_manifest = json.loads((OUT / "phase7_data_build_manifest.json").read_text())
    external_audit = json.loads((OUT / "external_data_audit.json").read_text())
    archive_integrity = json.loads((OUT / "candidate_archive_integrity.json").read_text())
    amendment = OUT / "asset_eligibility_amendment_001.md"
    external_audit["asset_eligibility_amendment_001_sha256"] = sha(amendment)
    write_json(OUT / "external_data_audit.json", external_audit)
    candidate_eligibility = {
        a: bool(external_audit.get("candidate_assets", {}).get(a, {}).get("eligibility", {}).get("eligible", False))
        for a in ["SOL", "BNB", "XRP"]
    }
    candidate_eligibility_pass = all(candidate_eligibility.values())
    archive_integrity_pass = bool(archive_integrity.get("pass")) and int(archive_integrity.get("archive_count", 0)) == 6195
    panel_build_hash_pass = build_manifest.get("panel_sha256") == sha(PANEL_PATH)

    frozen_phase6 = json.loads((ROOT / "outputs" / "phase6_autonomous_benchmark" / "prior_artifact_snapshot.json").read_text())
    protected = {}
    for rel, expected in frozen_phase6.get("files", {}).items():
        path = ROOT / rel
        if path.exists():
            actual = sha(path)
            protected[rel] = {"expected_sha256": expected["sha256"], "actual_sha256": actual, "pass": actual == expected["sha256"]}
    # These files are known frozen Phase-6 protocol/results; their checksums provide a direct guard.
    phase6_refs = {
        "outputs/phase6_autonomous_benchmark/benchmark_protocol.json": "69affa8f08f3943e9c0914dc276bdd5821410f5d401825e0299487ef13a01028",
        "outputs/phase6_autonomous_benchmark/selected_models.json": "93d8d2d15d0a03ccebd2cea4ce115a8d4dbfabe3d8dfbde8f9dac845757a6313",
        "outputs/phase5_dynamic_discovery/dataset_4h.jsonl": "e50637340052f505c585bbce70c86ac7f4d3f607b6ef127a24c1317d1d99402d",
    }
    for rel, expected in phase6_refs.items():
        path = ROOT / rel
        actual = sha(path) if path.exists() else None
        protected[rel] = {"expected_sha256": expected, "actual_sha256": actual, "pass": actual == expected}

    report = {
        "dataset": {"path": str(PANEL_PATH.relative_to(ROOT)), "sha256": sha(PANEL_PATH), "rows": int(len(panel)), "assets": assets, "unique_asset_date_duplicates": key_dups, "date_min": panel.period_start_utc.min().isoformat(), "date_max": panel.period_start_utc.max().isoformat()},
        "samples": {"development": {"start": "2021-12-01T00:00:00Z", "end": DEV_END.isoformat()}, "prior_phase6_test": {"start": "2023-07-01T00:00:00Z", "end": OLD_TEST_END.isoformat()}, "new_external": {"start": EXTERNAL_START.isoformat(), "end": SAMPLE_END.isoformat()}},
        "amihud_measurement_audit": amihud_reproduction,
        "liquidity_target_semantics": {
            "future_amihud_24h_mean": "mean of six 4-hour Amihud values at t+1 through t+6; Amihud uses 1e6*absolute 4-hour log return divided by 4-hour Binance spot quote volume in USDT",
            "future_kyle_abs_lambda_24h": "absolute OLS slope of 288 subsequent 5-minute log returns on signed spot quote order flow (2*taker-buy quote volume - total quote volume), t+1 through t+6; this is a price-impact proxy, not a quoted spread",
            "future_corwin_schultz_spread_24h": "mean of five two-adjacent-4-hour-bar Corwin-Schultz high-low spread estimates wholly inside t+1 through t+6; nonnegative range-based spread proxy, not a transaction-level quoted spread",
        },
        "asset_audit": asset_audit,
        "phase5_parity": json.loads((OUT / "phase5_variable_parity.json").read_text()),
        "candidate_source_manifest_sha256": build_manifest["candidate_manifest_sha256"],
        "asset_eligibility_protocol_sha256": build_manifest["asset_eligibility_protocol_sha256"],
        "asset_eligibility_amendment_sha256": sha(amendment),
        "protected_phase6_artifact_checks": protected,
        "protected_phase6_hashes_pass": all(x["pass"] for x in protected.values()),
        "candidate_eligibility": candidate_eligibility,
        "candidate_eligibility_pass": candidate_eligibility_pass,
        "archive_integrity_pass": archive_integrity_pass,
        "panel_build_hash_pass": panel_build_hash_pass,
        "dataset_key_unique": key_dups == 0,
        "data_audit_pass": key_dups == 0 and parity_pass and candidate_eligibility_pass and archive_integrity_pass and panel_build_hash_pass and amihud_reproduction["missingness_identical"] and (amihud_reproduction["max_absolute_difference"] or 0) < 2e-12 and all(x["pass"] for x in protected.values()),
        "note": "No forecast fit, model selection, correlation-driven selection, or new network request was performed by this audit. External outcome validation is limited to coverage/validity counts; external target distributions and forecast scores were not inspected.",
    }
    write_json(OUT / "phase7_data_audit.json", report)
    if not report["data_audit_pass"]:
        raise RuntimeError("Phase-7 data audit failed; see phase7_data_audit.json")
    print(json.dumps({"data_audit_pass": report["data_audit_pass"], "dataset_sha256": report["dataset"]["sha256"], "assets": assets, "amihud_reproduction": amihud_reproduction, "protected_phase6_hashes_pass": report["protected_phase6_hashes_pass"]}, indent=2))


if __name__ == "__main__":
    main()
