from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "phase9_external_qml_replication"
PANEL_PATH = OUT / "external_panel.csv.gz"
RAW_MANIFEST_PATH = OUT / "raw_manifest.json"
ASSETS = ("BTC", "ETH")
SPOT = ["spot_return", "spot_realized_volatility", "spot_downside_semivariance",
        "spot_amihud", "spot_taker_imbalance", "spot_quote_volume_growth"]
DER = ["funding_rate_state", "oi_log_change", "futures_taker_imbalance", "basis_mean"]
STATIC_DER = ["funding_change", "basis_change", "oi_shock_30d_robust", "flow_shock_30d_robust",
              "funding_x_oi_shock", "oi_shock_x_flow_shock", "funding_x_basis", "flow_x_basis"]
CAL = [*[f"utc_slot_{i}" for i in range(1, 6)], *[f"weekday_{i}" for i in range(1, 7)],
       "funding_schedule_window"]
TARGET_RAW = "future_amihud_24h_mean"
TARGET = "future_amihud_24h_log1p"
INTERVAL = pd.Timedelta(hours=4)
PURGE = 6


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def code_manifest() -> dict:
    sources = [
        "src/phase9_external_replication_download.py",
        "src/phase9_external_replication_build.py",
        "src/phase9_prefit_freeze.py",
        "src/phase9_external_qml_estimate.py",
        "src/targeted_24h_amihud/qml_models.py",
        "src/targeted_24h_amihud/classical_models.py",
        "src/phase9_external_qml_replication/offline_guard/sitecustomize.py",
        "src/phase9_finalize_reproducibility.py",
    ]
    missing = [p for p in sources if not (ROOT / p).is_file()]
    if missing:
        raise RuntimeError(f"Missing source files required for frozen reproducibility: {missing}")
    return {"status": "FROZEN_BEFORE_EXTERNAL_FORECASTS",
            "files": {p: sha256(ROOT / p) for p in sources}}


def write_json(name: str, obj) -> None:
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def feature_arrays(g: pd.DataFrame):
    g = g.sort_values("period_start_utc").reset_index(drop=True)
    x0 = pd.DataFrame(index=g.index)
    for lag in range(5, -1, -1):
        for col in SPOT:
            x0[f"{col}__lag{lag}"] = g[col].shift(lag)
    for col in CAL:
        x0[col] = g[col]
    xd = pd.DataFrame(index=g.index)
    for lag in range(5, -1, -1):
        for col in DER:
            xd[f"{col}__lag{lag}"] = g[col].shift(lag)
    for col in STATIC_DER + CAL:
        xd[col] = g[col]
    needed = [TARGET_RAW, *x0.columns, *xd.columns]
    valid = np.isfinite(g[needed].to_numpy(dtype=float)).all(axis=1)
    return g, x0.to_numpy(dtype=float), xd.to_numpy(dtype=float), valid


def longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    best_start = best_end = cur_start = 0
    best_len = 0
    for i, value in enumerate(mask):
        if value and (i == 0 or not mask[i - 1]):
            cur_start = i
        if value:
            length = i - cur_start + 1
            if length > best_len:
                best_len, best_start, best_end = length, cur_start, i + 1
    return best_start, best_end


def pct_count(series: pd.Series, expected: int) -> dict:
    good = np.isfinite(pd.to_numeric(series, errors="coerce").to_numpy(dtype=float))
    return {"valid_intervals": int(good.sum()), "expected_intervals": int(expected),
            "coverage_pct": 100.0 * float(good.mean()) if expected else None}


def instrument_metadata_audit(asset: str) -> dict:
    result = {}
    for category in ("spot", "linear"):
        path = ROOT / "data" / "raw" / "phase9_external" / "bybit" / "api" / asset / "metadata" / f"instruments_{category}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("retCode") != 0:
            raise RuntimeError(f"Bybit {category} instrument metadata API error for {asset}")
        records = [r for r in payload.get("result", {}).get("list", []) if r.get("symbol") == asset + "USDT"]
        if len(records) != 1:
            raise RuntimeError(f"Expected exactly one Bybit {category} instrument record for {asset}USDT")
        row = records[0]
        result[category] = {k: row.get(k) for k in ["symbol", "status", "baseCoin", "quoteCoin", "settleCoin",
                                                     "contractType", "launchTime", "deliveryTime", "fundingInterval"]}
    spot = result["spot"]
    linear = result["linear"]
    result["eligibility"] = {
        "spot_symbol_and_quote_match": spot.get("symbol") == asset + "USDT" and spot.get("baseCoin") == asset and spot.get("quoteCoin") == "USDT",
        "linear_perpetual_symbol_and_settlement_match": linear.get("symbol") == asset + "USDT" and linear.get("baseCoin") == asset and linear.get("quoteCoin") == "USDT" and linear.get("settleCoin") == "USDT" and linear.get("contractType") == "LinearPerpetual",
        "linear_launch_precedes_requested_start": int(linear.get("launchTime", 0)) <= int(pd.Timestamp("2022-11-10T00:00:00Z").timestamp() * 1000),
        "same_symbol_in_daily_archives": True,
    }
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel_hash = sha256(PANEL_PATH)
    prior_source_hashes = json.loads((OUT / "frozen_source_hashes.json").read_text())
    prior_source_errors = []
    for filename, expected_hash in prior_source_hashes.items():
        source_path = ROOT / "outputs" / "targeted_24h_amihud" / filename
        if not source_path.exists() or sha256(source_path) != expected_hash:
            prior_source_errors.append(filename)
    if prior_source_errors:
        raise RuntimeError(f"Immutable targeted-study source hash mismatch: {prior_source_errors}")
    panel = pd.read_csv(PANEL_PATH, parse_dates=["period_start_utc", "interval_end_utc", "funding_last_event_utc"],
                        low_memory=False)
    panel["period_start_utc"] = pd.to_datetime(panel.period_start_utc, utc=True)
    panel["interval_end_utc"] = pd.to_datetime(panel.interval_end_utc, utc=True)
    panel["funding_last_event_utc"] = pd.to_datetime(panel.funding_last_event_utc, utc=True)
    if panel.duplicated(["asset", "period_start_utc"]).any():
        raise RuntimeError("External panel has duplicate asset × 4-hour keys")
    if set(panel.asset.unique()) != set(ASSETS):
        raise RuntimeError("Panel asset universe is not BTC/ETH")

    validation = json.loads((OUT / "external_panel_validation.json").read_text())
    raw_manifest = json.loads(RAW_MANIFEST_PATH.read_text())
    raw_count = 0
    raw_hash_errors = []
    for row in raw_manifest["files"]:
        raw_path = ROOT / row["path"]
        raw_count += 1
        if not raw_path.exists() or sha256(raw_path) != row["sha256"]:
            raw_hash_errors.append(row["path"])
    if raw_hash_errors:
        raise RuntimeError(f"Raw source hash mismatch: {raw_hash_errors[:5]}")

    asset_audit = {}
    asset_valid = {}
    timestamps = {}
    metadata = {asset: instrument_metadata_audit(asset) for asset in ASSETS}
    for asset, g0 in panel.groupby("asset", sort=False):
        g = g0.sort_values("period_start_utc").reset_index(drop=True)
        expected = len(g)
        grid = pd.date_range(g.period_start_utc.min(), g.period_start_utc.max(), freq=INTERVAL)
        exact_grid = g.period_start_utc.equals(grid)
        _, _, _, valid = feature_arrays(g)
        asset_valid[asset] = pd.Series(valid, index=g.period_start_utc)
        named = {
            "spot_price_close": pct_count(g.spot_close, expected),
            "spot_quote_volume": pct_count(g.spot_quote_volume, expected),
            "spot_amihud": pct_count(g.spot_amihud, expected),
            "spot_realized_volatility": pct_count(g.spot_realized_volatility, expected),
            "spot_downside_semivariance": pct_count(g.spot_downside_semivariance, expected),
            "spot_taker_imbalance": pct_count(g.spot_taker_imbalance, expected),
            "spot_trade_count": pct_count(g.spot_trade_count, expected),
            "funding_rate_state": pct_count(g.funding_rate_state, expected),
            "oi_log_change": pct_count(g.oi_log_change, expected),
            "futures_taker_imbalance": pct_count(g.futures_taker_imbalance, expected),
            "basis_mean": pct_count(g.basis_mean, expected),
            "future_amihud_24h_mean": pct_count(g.future_amihud_24h_mean, expected),
            "complete_joint_model_rows": {"valid_intervals": int(valid.sum()), "expected_intervals": expected,
                                           "coverage_pct": 100.0 * float(valid.mean())},
        }
        funding_events = g.funding_last_event_utc.dropna().drop_duplicates().sort_values()
        funding_gap_hours = funding_events.diff().dropna().dt.total_seconds().div(3600)
        oi_age = pd.to_numeric(g.oi_snapshot_age_hours, errors="coerce").dropna()
        timestamps[asset] = {
            "panel_interval_label": "period_start_utc; API 5m candle open times are milliseconds UTC; daily public-trade timestamp is execution time (spot archive milliseconds; futures archive fractional Unix seconds)",
            "interval_end_rule": "period_start_utc + 4 hours; aggregates available at interval end",
            "panel_grid_exact_4h": bool(exact_grid),
            "duplicate_4h_timestamps": int(g.period_start_utc.duplicated().sum()),
            "first_interval_utc": g.period_start_utc.min().isoformat(),
            "last_interval_utc": g.period_start_utc.max().isoformat(),
            "funding_unique_settlement_events": int(funding_events.nunique()),
            "funding_gap_hours_counts": {str(float(k)): int(v) for k, v in funding_gap_hours.value_counts().sort_index().items()},
            "oi_snapshot_age_hours_quantiles": {str(q): float(oi_age.quantile(q)) for q in [0, .5, .95, 1]} if len(oi_age) else {},
            "five_minute_source_coverage": validation.get("asset_diagnostics", {}).get(asset, {}).get("five_minute_timestamps", {}),
            "open_interest_source_timestamps": validation.get("asset_diagnostics", {}).get(asset, {}).get("open_interest_timestamps", {}),
            "funding_source_timestamps": validation.get("asset_diagnostics", {}).get(asset, {}).get("funding_timestamps", {}),
        }
        def distribution(s: pd.Series) -> dict:
            a = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
            return {"n": int(len(a)), "min": float(a.min()) if len(a) else None,
                    "max": float(a.max()) if len(a) else None,
                    "quantiles": {str(q): float(np.quantile(a, q)) for q in [.001, .01, .5, .99, .999] if len(a)}}
        asset_audit[asset] = {
            "grid_rows": expected,
            "first_date": g.period_start_utc.min().isoformat(),
            "last_date": g.period_start_utc.max().isoformat(),
            "coverage": named,
            "timestamp_audit": timestamps[asset],
            "amihud_value_diagnostics": {
                "formula": "1e6 × abs(4-hour log return) / 4-hour USDT quote volume",
                "zero_denominators": int((g.spot_quote_volume == 0).sum()),
                "negative_denominators": int((g.spot_quote_volume < 0).sum()),
                "negative_amihud": int((g.spot_amihud < 0).sum()),
                "zero_amihud": int((g.spot_amihud == 0).sum()),
                "future_target_quantiles": {str(q): float(g.future_amihud_24h_mean.quantile(q))
                                            for q in [.001, .01, .5, .99, .999] if g.future_amihud_24h_mean.notna().any()},
                "log1p_max_relative_difference": float(
                    np.max(np.abs(np.log1p(g.future_amihud_24h_mean.dropna()) - g.future_amihud_24h_mean.dropna()) /
                           np.maximum(np.abs(g.future_amihud_24h_mean.dropna()), 1e-30))) if g.future_amihud_24h_mean.notna().any() else None,
            },
            "derivative_semantic_diagnostics": {
                "funding_rate_state": distribution(g.funding_rate_state),
                "oi_log_change": distribution(g.oi_log_change),
                "futures_taker_imbalance": distribution(g.futures_taker_imbalance),
                "futures_imbalance_out_of_bounds_count": int(((g.futures_taker_imbalance < -1-1e-10) | (g.futures_taker_imbalance > 1+1e-10)).sum()),
                "spot_taker_imbalance": distribution(g.spot_taker_imbalance),
                "spot_imbalance_out_of_bounds_count": int(((g.spot_taker_imbalance < -1-1e-10) | (g.spot_taker_imbalance > 1+1e-10)).sum()),
                "basis_mean": distribution(g.basis_mean),
                "basis_abs_gt_5pct_count": int((g.basis_mean.abs() > .05).sum()),
                "nonpositive_spot_quote_volume_count": int((g.spot_quote_volume <= 0).sum()),
            },
        }

    common_dates = sorted(set(asset_valid["BTC"].index[asset_valid["BTC"].to_numpy()]) &
                          set(asset_valid["ETH"].index[asset_valid["ETH"].to_numpy()]))
    if not common_dates:
        raise RuntimeError("No complete common BTC/ETH model rows")
    common = panel[panel.period_start_utc.isin(common_dates)].drop_duplicates("period_start_utc").sort_values("period_start_utc")
    full_grid = pd.date_range(common.period_start_utc.min(), common.period_start_utc.max(), freq=INTERVAL)
    common_mask = common.period_start_utc.reset_index(drop=True).equals(full_grid)
    if not common_mask:
        both = np.isin(full_grid.asi8, common.period_start_utc.array.asi8)
        rs, re = longest_true_run(both)
        run_dates = full_grid[rs:re]
        if len(run_dates) == 0:
            raise RuntimeError("No contiguous common valid period")
        common_dates = list(run_dates)
        common = panel[panel.period_start_utc.isin(common_dates)].sort_values(["asset", "period_start_utc"])
    n = int(len(common_dates))
    months = (common_dates[-1] - common_dates[0]).total_seconds() / (86400 * 30.4375)
    if months < 18:
        raise RuntimeError(f"Bybit common valid period is too short: {months:.1f} months")
    boundary = int(np.floor(.70 * n))
    train_end_ix = boundary - PURGE
    if train_end_ix < int(.50 * boundary):
        raise RuntimeError("Not enough training dates after overlap purge")
    train_dates = pd.DatetimeIndex(common_dates[:train_end_ix])
    purge_dates = pd.DatetimeIndex(common_dates[train_end_ix:boundary])
    holdout_dates = pd.DatetimeIndex(common_dates[boundary:])
    if len(holdout_dates) == 0:
        raise RuntimeError("Empty untouched holdout")

    # Frozen chronological OOF schedule: 50% initial warm-up followed by five
    # equal validation blocks, with six 4-hour labels purged at each boundary.
    oof_n = len(train_dates)
    warmup_n = int(np.ceil(.50 * oof_n))
    remaining = np.arange(warmup_n, oof_n)
    blocks = [b for b in np.array_split(remaining, 5) if len(b)]
    oof_folds = []
    for j, block in enumerate(blocks, 1):
        first = int(block[0])
        fit_end = first - PURGE
        if fit_end <= 0:
            raise RuntimeError("Chronological OOF fold has no training rows after purge")
        oof_folds.append({"fold": f"OOF{j}", "train_first": train_dates[0].isoformat(),
                          "train_last_inclusive": train_dates[fit_end - 1].isoformat(),
                          "validation_first": train_dates[first].isoformat(),
                          "validation_last_inclusive": train_dates[int(block[-1])].isoformat(),
                          "train_rows": fit_end, "validation_rows": int(len(block)),
                          "purge_rows": PURGE, "validation_index_start": first,
                          "validation_index_end_exclusive": int(block[-1]) + 1})

    model_info = json.loads((OUT / "matched_classical_models.json").read_text())
    residual_prior = json.loads((ROOT / "outputs" / "targeted_24h_amihud" / "residual_learning_results.json").read_text())
    strong_cfg = {}
    for asset, family in (("BTC", "ElasticNet"), ("ETH", "XGBoost")):
        record = next(r for r in residual_prior["fit_records"]
                      if r.get("asset") == asset and r.get("family") == family and r.get("fold") == "F3")
        strong_cfg[asset] = {"family": family, "config": record["selected_config"],
                             "selection_basis": "F3 inner chronological development selection in the immutable targeted-study residual artifacts; no Phase-9 observations used"}
    model_info["strongest_previously_frozen_classical_residual_learner"] = strong_cfg
    write_json("matched_classical_models.json", model_info)

    split = {
        "venue": "Bybit",
        "asset_universe": list(ASSETS),
        "frequency": "4-hour UTC",
        "rule": "Longest contiguous common BTC/ETH period with all frozen feature fields and complete 24-hour future target; chronological 70/30 split; six 4-hour rows immediately before holdout start purged from training because labels overlap the holdout boundary.",
        "common_valid_rows_before_purge": n,
        "common_start_utc": common_dates[0].isoformat(),
        "common_end_utc": common_dates[-1].isoformat(),
        "total_months_approx": months,
        "train_span_start_utc": common_dates[0].isoformat(),
        "train_span_end_utc_inclusive": train_dates[-1].isoformat(),
        "training_rows_used_after_purge": len(train_dates),
        "purge_rows": PURGE,
        "purge_start_utc": purge_dates[0].isoformat() if len(purge_dates) else None,
        "purge_end_utc": purge_dates[-1].isoformat() if len(purge_dates) else None,
        "holdout_start_utc": holdout_dates[0].isoformat(),
        "holdout_end_utc_inclusive": holdout_dates[-1].isoformat(),
        "holdout_rows": len(holdout_dates),
        "holdout_fraction_of_common_period": len(holdout_dates) / n,
        "external_holdout_untouched_before_freeze": True,
        "oof_rule": "50% initial chronological warm-up; five equal expanding validation blocks; six observations purged between fit labels and each validation block; no in-sample residuals.",
        "oof_folds": oof_folds,
    }
    write_json("temporal_split.json", split)

    # Evaluate the venue only on the preregistered, outcome-blind data gates.
    coverage_gate = True
    gates = {}
    for asset in ASSETS:
        c = asset_audit[asset]["coverage"]
        checks = {
            "spot_price_close_ge_95pct": c["spot_price_close"]["coverage_pct"] >= 95,
            "spot_quote_volume_ge_95pct": c["spot_quote_volume"]["coverage_pct"] >= 95,
            "spot_trade_count_ge_95pct": c["spot_trade_count"]["coverage_pct"] >= 95,
            "funding_ge_95pct": c["funding_rate_state"]["coverage_pct"] >= 95,
            "oi_ge_95pct": c["oi_log_change"]["coverage_pct"] >= 95,
            "futures_taker_imbalance_ge_95pct": c["futures_taker_imbalance"]["coverage_pct"] >= 95,
            "basis_ge_95pct": c["basis_mean"]["coverage_pct"] >= 95,
            "spot_and_futures_imbalance_bounds": asset_audit[asset]["derivative_semantic_diagnostics"]["futures_imbalance_out_of_bounds_count"] == 0 and asset_audit[asset]["derivative_semantic_diagnostics"]["spot_imbalance_out_of_bounds_count"] == 0,
            "positive_spot_quote_volume": asset_audit[asset]["derivative_semantic_diagnostics"]["nonpositive_spot_quote_volume_count"] == 0,
            "target_rows_exist": c["future_amihud_24h_mean"]["valid_intervals"] > 0,
            "utc_grid_exact": timestamps[asset]["panel_grid_exact_4h"],
            "instrument_metadata_semantics": all(metadata[asset]["eligibility"].values()),
            "5m_source_coverage_ge_95pct": all(
                z.get("unique_rows", 0) / max(1, z.get("expected_rows", 1)) >= .95
                for z in timestamps[asset]["five_minute_source_coverage"].values()),
            "no_5m_duplicate_or_off_grid_timestamps": all(
                z.get("duplicate_rows") == 0 and z.get("off_grid_timestamps") == 0
                for z in timestamps[asset]["five_minute_source_coverage"].values()),
            "no_oi_or_funding_timestamp_duplicates": timestamps[asset]["open_interest_source_timestamps"].get("duplicate_rows", 0) == 0 and timestamps[asset]["funding_source_timestamps"].get("duplicate_rows", 0) == 0,
        }
        gates[asset] = checks
        coverage_gate &= all(checks.values())
    raw_archive_counts = {}
    for asset in ASSETS:
        raw_archive_counts[asset] = {}
        for market, directory in (("spot", "spot"), ("futures", "trading")):
            files = list((ROOT / "data" / "raw" / "phase9_external" / "bybit" / "archive" / directory / (asset + "USDT")).glob("*.csv.gz"))
            raw_archive_counts[asset][market] = len(files)
            coverage_gate &= len(files) / 1428 >= .95

    semantic = {
        "Bybit_vs_Binance": {
            "spot_amihud": {"binance": "1e6 × abs(4h log return) / 4h USDT quote volume", "bybit": "same; V5 spot kline turnover is USDT quote notional; daily close-to-close return at 4-hour boundaries", "timestamp": "5m kline open timestamps in Unix milliseconds UTC; aggregate only complete 48-bar windows", "status": "PASS" if all(gates[a]["spot_price_close_ge_95pct"] and gates[a]["spot_quote_volume_ge_95pct"] for a in ASSETS) else "FAIL"},
            "funding_rate_state": {"binance": "settlement funding rate; last event at/before interval end, maximum age 12h", "bybit": "fundingRate at fundingRateTimestamp; same as-of rule and decimal rate per settlement", "status": "PASS" if all(gates[a]["funding_ge_95pct"] for a in ASSETS) else "FAIL", "caveat": "Settlement cadence can vary by symbol/regime; event spacing is retained in the audit and no synthetic settlement values are created."},
            "oi_log_change": {"binance": "log change of 4-hour open-interest level", "bybit": "log change of openInterest; Bybit documents BTCUSDT linear units in BTC and says openInterest is both-side sum; log change is invariant to a constant scale", "timestamp": "UTC millisecond snapshot, as-of interval end with age capped at 4h", "status": "PASS" if all(gates[a]["oi_ge_95pct"] for a in ASSETS) else "FAIL", "caveat": "Absolute OI levels are not compared; both-side provider convention is documented."},
            "futures_taker_imbalance": {"binance": "(taker buy base volume − taker sell base volume)/(buy + sell)", "bybit": "same ratio using homeNotional base quantity and archived Buy/Sell aggressor side", "bounds": "[-1,1] when denominator is positive", "status": "PASS" if all(gates[a]["futures_taker_imbalance_ge_95pct"] for a in ASSETS) else "FAIL", "caveat": "Bybit public archive has fractional Unix-second trade timestamps; floor to UTC milliseconds only for 4h bin assignment; raw files unchanged."},
            "basis_mean": {"binance": "mean(mark/index − 1) over 48 five-minute observations", "bybit": "same nonannualized ratio from linear perpetual mark and index 5m closes", "status": "PASS" if all(gates[a]["basis_ge_95pct"] for a in ASSETS) else "FAIL"},
            "overall": "PASS" if coverage_gate else "FAIL",
        },
        "OKX": {"status": "FAIL_REQUIRED_OI_HISTORY", "detail": "The official contract OI-history endpoint was probed for BTC-USDT-SWAP and ETH-USDT-SWAP in one-day 4h windows. Queries on 2024-07-01, 2024-10-01, 2025-01-01, 2025-07-01, and 2026-01-01 returned zero records for both assets; 2026-10-01 returned five records each. These probes do not provide the minimum 18-month historical OI series. The API changelog says the endpoint was added 2024-06-13. No OKX forecast outcome or model fit was inspected.", "probe_artifact": "okx_oi_probe.json"},
        "source_urls": {
            "bybit_kline": "https://bybit-exchange.github.io/docs/v5/market/kline",
            "bybit_open_interest": "https://bybit-exchange.github.io/docs/v5/market/open-interest",
            "bybit_funding": "https://bybit-exchange.github.io/docs/v5/market/funding-history",
            "bybit_public_trade_side": "https://bybit-exchange.github.io/docs/v5/market/recent-trade",
            "bybit_archive": "https://public.bybit.com/",
            "okx_contract_oi_history": "https://www.okx.com/docs-v5/en/#trading-data-rest-api-get-contract-open-interest-history",
        },
        "status": "PASS" if coverage_gate else "FAIL",
    }
    write_json("semantic_parity.json", semantic)

    write_json("amihud_audit.json", {
        "status": "COMPLETE_DATA_AUDIT_ONLY",
        "formula": "1e6 × abs(4-hour spot log return) / 4-hour spot quote volume in USDT; denominator <=0 is missing, no epsilon or clipping",
        "target": "arithmetic mean of spot Amihud at t+1,...,t+6; all six future intervals required",
        "transformation": "frozen T2_log1p from the immutable targeted study; no external transform selection",
        "assets": {a: asset_audit[a]["amihud_value_diagnostics"] | {"spot_amihud_coverage": asset_audit[a]["coverage"]["spot_amihud"], "quote_volume_coverage": asset_audit[a]["coverage"]["spot_quote_volume"]} for a in ASSETS},
        "no_predictive_results_used": True,
    })

    venue_audit = {
        "audit_basis": "schema, coverage, timestamps, units, duplicates and source definitions only; no forecast results inspected",
        "raw_period_requested_utc": {"start": "2022-11-10T00:00:00Z", "end_exclusive": "2026-10-08T00:00:00Z"},
        "raw_file_count": raw_count,
        "raw_hash_verification": "PASS" if not raw_hash_errors else "FAIL",
        "raw_manifest_sha256": sha256(RAW_MANIFEST_PATH),
        "external_panel_sha256": panel_hash,
        "bybit": {"asset_audit": asset_audit, "timestamp_audit": timestamps,
                  "instrument_metadata": metadata,
                  "archive_daily_file_counts": raw_archive_counts,
                  "archive_coverage_pct": {a: {m: 100.0 * n / 1428 for m, n in raw_archive_counts[a].items()} for a in ASSETS},
                  "eligibility_gates": gates,
                  "status": "ELIGIBLE" if coverage_gate else "INELIGIBLE"},
        "okx": semantic["OKX"],
        "okx_oi_probe_sha256": sha256(OUT / "okx_oi_probe.json"),
        "venue_selection": "Bybit is the only venue proceeding to the preregistered replication because OKX fails historical OI coverage and Bybit passes all frozen data/semantic gates." if coverage_gate else "NO_ELIGIBLE_EXTERNAL_VENUE",
        "no_performance_data_used_for_venue_selection": True,
    }
    write_json("venue_data_audit.json", venue_audit)
    write_json("data_validation.json", {"status": "PASS_DATA_ONLY_AUDIT" if coverage_gate else "FAIL",
                "external_panel_sha256": panel_hash, "venue_audit_sha256": sha256(OUT / "venue_data_audit.json"),
                "semantic_parity_sha256": sha256(OUT / "semantic_parity.json"), "venue_eligible": bool(coverage_gate),
                "holdout_forecast_results_read": False})
    if not coverage_gate:
        raise RuntimeError("NO_ELIGIBLE_EXTERNAL_VENUE; stopping before any forecasting")

    # Freeze exact specifications, raw data, panel hash and the mechanical split
    # before loading any forecast results.
    protocol = json.loads((OUT / "phase9_protocol.json").read_text())
    protocol["data_freeze"] = {"raw_manifest_sha256": venue_audit["raw_manifest_sha256"],
                               "external_panel_sha256": panel_hash,
                               "semantic_parity_sha256": sha256(OUT / "semantic_parity.json"),
                               "venue_audit_sha256": sha256(OUT / "venue_data_audit.json"),
                               "temporal_split_sha256": sha256(OUT / "temporal_split.json"),
                               "venue_selection": "Bybit only; based solely on frozen data eligibility gates"}
    protocol["strong_classical_models"] = strong_cfg
    protocol["qml_configs"] = {asset: json.loads((OUT / f"frozen_{'btc_hybrid4q' if asset == 'BTC' else 'eth_qnn6q'}.json").read_text()) for asset in ASSETS}
    protocol["spot_only_configs"] = model_info["spot_only"]
    protocol["matched_classical_residual_configs"] = model_info["matched_residual_model"]["per_asset"]
    protocol["training_residual_oof"] = split["oof_rule"]
    protocol["temporal_split"] = split
    protocol["target_transform"] = {"name": "T2_log1p", "basis": "Exact transform selected in the immutable targeted study; no external selection."}
    protocol["negative_control"] = "At placebo anchor t, the target is Amihud mean over t−6..t−1, matched SpotOnly inputs are shifted back to information available through t−7, and the residual learner receives derivative features dated t+7..t+12. Training uses only venue-training anchors; placebo target windows are wholly inside holdout. A positive supported gain warns of temporal persistence/confounding."
    protocol["frozen_seeds"] = {a: json.loads((OUT / f"frozen_{'btc_hybrid4q' if a == 'BTC' else 'eth_qnn6q'}.json").read_text())["seeds"] for a in ASSETS}
    protocol["seed_rules"] = {"base": 20261007, "spot_oof": "stable_seed(20261007, asset, SpotOnlyFamily, phase9_oof_fold_id)", "spot_final": "stable_seed(20261007, asset, SpotOnlyFamily, phase9_final)", "strong_classical_residual": "stable_seed(20261007, asset, family, phase9_strong_residual)", "paired_bootstrap": "stable_seed(20261007, asset, endpoint, block_length)", "QML_and_MLP": "the exact five seed integers in each immutable frozen asset configuration; MLP uses the paired QML seed"}
    protocol["bootstrap"] = {"replications": 1000, "block_lengths_4h_intervals": [42, 84], "primary": 42, "method": "paired moving temporal blocks on identical holdout dates"}
    protocol["inference"] = {"primary_metric": "MAE on log1p future 24h Amihud", "paired_bootstrap_two_sided_p": "plus-one corrected from moving-block bootstrap", "Holm_families": ["Delta_FULL BTC/ETH", "Delta_QML_MLP BTC/ETH"], "loss_differential": "Bartlett/Newey-West HAC, 6 lags, supporting only"}
    execution_code = code_manifest()
    write_json("execution_code_manifest.json", execution_code)
    protocol["execution_code_manifest_sha256"] = sha256(OUT / "execution_code_manifest.json")
    write_json("phase9_protocol.json", protocol)
    (OUT / "phase9_protocol.md").write_text(
        "# Phase 9 protocol freeze\n\n"
        "The confirmatory external replication uses Bybit only. Venue inclusion was determined from source schema, semantics, coverage, timestamp integrity and missingness before any forecast was fit. OKX was excluded because the tested historical 4-hour contract OI endpoint did not provide the required long history.\n\n"
        f"The frozen common BTC/ETH interval is {split['common_start_utc']} through {split['common_end_utc_inclusive']}. The chronological 70/30 boundary is {split['holdout_start_utc']}; the holdout ends {split['holdout_end_utc_inclusive']}. Six 4-hour observations immediately before the boundary are purged to prevent future-target overlap. The holdout was not used for model or feature selection.\n\n"
        "The target is the frozen T2 log1p transform of the arithmetic mean of the next six 4-hour Amihud observations. BTC uses the frozen Hybrid QNN with 4 qubits; ETH uses the frozen variational QNN with 6 qubits. Architecture, PCA dimension, scaler, optimizer, epochs, batches, seeds and matched MLPs are recovered from the immutable targeted-study artifacts. No external-venue tuning is permitted.\n\n"
        "The SpotOnly model is the prior development-selected asset-specific CatBoost/XGBoost configuration. Residual training targets are constructed only from chronological training-period out-of-fold SpotOnly forecasts. Six labels are purged at each OOF boundary. The primary endpoint is transformed-scale MAE(SpotOnly) minus MAE(SpotOnly + QML residual correction); the matched MLP comparison is separate. Paired moving-block bootstrap uses 1,000 draws at 42 and 84 intervals, with centered bootstrap p-values. The matched temporal placebo predicts t−6..t−1 using SpotOnly information through t−7 and derivatives from t+7..t+12. Holm correction is applied across BTC/ETH separately for each primary endpoint.\n\n"
        "No forecast result was read or generated before this freeze. All historical prior-stage files remain unchanged.\n",
        encoding="utf-8")
    protocol_hash = sha256(OUT / "phase9_protocol.json")
    (OUT / "phase9_protocol.sha256").write_text(f"{protocol_hash}  phase9_protocol.json\n", encoding="utf-8")
    write_json("target_transform_selection.json", {"selected_transform": "T2_log1p", "basis": "Exact transform frozen in the immutable targeted 24h Amihud study; no external target transformation comparison or selection.", "sha256_source": sha256(ROOT / "outputs" / "targeted_24h_amihud" / "target_transform_selection.json")})
    software = {
        "python": __import__("sys").version,
        "numpy": np.__version__, "pandas": pd.__version__,
        "scikit_learn": __import__("sklearn").__version__,
        "torch": __import__("torch").__version__, "pennylane": __import__("pennylane").__version__,
        "catboost": __import__("catboost").__version__, "xgboost": __import__("xgboost").__version__,
        "backend": "PennyLane default.qubit exact CPU statevector; shots=None",
        "cuda_available": bool(__import__("torch").cuda.is_available()),
        "forecast_results_read_before_freeze": False,
    }
    write_json("software_environment.json", software)
    write_json("qml_environment.json", {k: software[k] for k in ["python", "numpy", "pandas", "scikit_learn", "torch", "pennylane", "backend", "cuda_available"]} | {
        "qnn_architecture": {a: json.loads((OUT / f"frozen_{'btc_hybrid4q' if a == 'BTC' else 'eth_qnn6q'}.json").read_text()) for a in ASSETS},
        "simulator": "PennyLane default.qubit exact statevector", "shots": None})
    frozen_files = ["phase9_protocol.md", "phase9_protocol.json", "venue_eligibility_protocol.json",
                    "semantic_parity.json", "venue_data_audit.json", "raw_manifest.json", "external_panel.csv.gz",
                    "temporal_split.json", "frozen_btc_hybrid4q.json", "frozen_eth_qnn6q.json",
                    "matched_classical_models.json", "feature_registry.json", "target_registry.json",
                    "target_transform_selection.json", "software_environment.json", "qml_environment.json",
                    "data_validation.json", "amihud_audit.json", "frozen_source_hashes.json",
                    "execution_code_manifest.json", "qml_fixture_validation.json", "okx_oi_probe.json"]
    manifest = {name: sha256(OUT / name) for name in frozen_files}
    manifest["phase9_protocol.json"] = protocol_hash
    write_json("frozen_artifact_hashes.json", {"status": "FROZEN_BEFORE_EXTERNAL_FORECASTS", "files": manifest})
    print(json.dumps({"status": "FROZEN_BEFORE_EXTERNAL_FORECASTS", "panel_sha256": panel_hash,
                      "protocol_sha256": protocol_hash, "common_dates": n, "train_used": len(train_dates),
                      "purge": len(purge_dates), "holdout": len(holdout_dates),
                      "holdout_start": holdout_dates[0].isoformat(), "holdout_end": holdout_dates[-1].isoformat(),
                      "venue": "Bybit", "okx": "FAIL_REQUIRED_OI_HISTORY"}, indent=2))


if __name__ == "__main__":
    main()
