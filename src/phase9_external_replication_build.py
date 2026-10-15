from __future__ import annotations

import datetime as dt
import concurrent.futures
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "outputs" / "phase9_external_qml_replication"
RAW = ROOT / "data" / "raw" / "phase9_external" / "bybit"
OUT_PANEL = PHASE / "external_panel.csv.gz"
ASSETS = ("BTC", "ETH")
START = pd.Timestamp("2022-11-10T00:00:00Z")
END_EXCLUSIVE = pd.Timestamp("2026-10-08T00:00:00Z")
FOUR_H = pd.Timedelta(hours=4)
FIVE_M = pd.Timedelta(minutes=5)
N5 = 48


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jwrite(name: str, value) -> None:
    path = PHASE / name
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_api_chunks(asset: str, dataset: str) -> pd.DataFrame:
    files = sorted((RAW / "api" / asset / dataset).glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No raw API chunks for {asset}/{dataset}")
    rows = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("retCode") != 0:
            raise RuntimeError(f"API error in cached raw payload {path}")
        rows.extend(data.get("result", {}).get("list", []))
    if dataset in ("spot_klines_5m",):
        cols = ["open_time_ms", "open", "high", "low", "close", "base_volume", "quote_volume"]
    elif dataset in ("mark_klines_5m", "index_klines_5m"):
        cols = ["open_time_ms", "open", "high", "low", "close"]
    elif dataset == "open_interest_4h":
        cols = ["open_interest", "single_open_interest", "timestamp_ms"]
        normalized = []
        for row in rows:
            normalized.append([row.get("openInterest"), row.get("singleOpenInterest"), row.get("timestamp")])
        rows = normalized
    elif dataset == "funding_history":
        cols = ["symbol", "funding_rate", "funding_timestamp_ms"]
        normalized = []
        for row in rows:
            normalized.append([row.get("symbol"), row.get("fundingRate"), row.get("fundingRateTimestamp")])
        rows = normalized
    else:
        raise KeyError(dataset)
    frame = pd.DataFrame(rows, columns=cols)
    if dataset.startswith("spot_") or dataset.endswith("klines_5m"):
        frame["open_time_ms"] = pd.to_numeric(frame.open_time_ms, errors="coerce")
        frame["timestamp"] = pd.to_datetime(frame.open_time_ms, unit="ms", utc=True, errors="coerce")
        frame = frame.drop(columns="open_time_ms")
        numeric = [c for c in frame.columns if c != "timestamp"]
        frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    elif dataset == "open_interest_4h":
        frame["timestamp_ms"] = pd.to_numeric(frame.timestamp_ms, errors="coerce")
        frame["timestamp"] = pd.to_datetime(frame.timestamp_ms, unit="ms", utc=True, errors="coerce")
        frame["open_interest"] = pd.to_numeric(frame.open_interest, errors="coerce")
        frame["single_open_interest"] = pd.to_numeric(frame.single_open_interest, errors="coerce")
    else:
        frame["funding_timestamp_ms"] = pd.to_numeric(frame.funding_timestamp_ms, errors="coerce")
        frame["timestamp"] = pd.to_datetime(frame.funding_timestamp_ms, unit="ms", utc=True, errors="coerce")
        frame["funding_rate"] = pd.to_numeric(frame.funding_rate, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return frame


def aggregate_5m_to_4h(kline: pd.DataFrame, asset: str) -> pd.DataFrame:
    x = kline.set_index("timestamp").sort_index()
    duplicate_count = int(x.index.duplicated(keep="last").sum())
    x = x[~x.index.duplicated(keep="last")]
    grid = pd.date_range(START, END_EXCLUSIVE, freq=FIVE_M, inclusive="left")
    x = x.reindex(grid)
    price_cols = ["open", "high", "low", "close"]
    vol_cols = ["base_volume", "quote_volume"]
    valid = np.isfinite(x[price_cols + vol_cols].to_numpy(dtype=float)).all(axis=1)
    valid &= (x[price_cols].to_numpy(dtype=float) > 0).all(axis=1)
    valid &= (x[vol_cols].to_numpy(dtype=float) >= 0).all(axis=1)
    z = x.copy()
    z.loc[~valid, :] = np.nan
    group = grid.floor("4h")
    counts = pd.Series(valid.astype(np.int16), index=grid).groupby(group).sum()
    g = z.groupby(group)
    out = pd.DataFrame(index=counts.index)
    out["spot_open"] = g.open.first()
    out["spot_high"] = g.high.max()
    out["spot_low"] = g.low.min()
    out["spot_close"] = g.close.last()
    out["spot_base_volume"] = g.base_volume.sum(min_count=N5)
    out["spot_quote_volume"] = g.quote_volume.sum(min_count=N5)
    out["spot_valid_5m_bars"] = counts
    complete = counts.eq(N5)
    r5 = np.log(z.close / z.close.shift(1))
    rg = r5.groupby(group)
    rcount = rg.count()
    rv = np.sqrt(rg.apply(lambda s: np.square(s).sum(min_count=N5)))
    downside = rg.apply(lambda s: np.square(s.clip(upper=0)).sum(min_count=N5))
    out["spot_realized_volatility"] = rv.where(complete & rcount.eq(N5))
    out["spot_downside_semivariance"] = downside.where(complete & rcount.eq(N5))
    out.loc[~complete, ["spot_open", "spot_high", "spot_low", "spot_close", "spot_base_volume", "spot_quote_volume"]] = np.nan
    out["spot_return"] = np.log(out.spot_close / out.spot_close.shift(1))
    out["spot_amihud"] = 1e6 * out.spot_return.abs() / out.spot_quote_volume.where(out.spot_quote_volume > 0)
    out["spot_quote_volume_growth"] = np.log(out.spot_quote_volume.where(out.spot_quote_volume > 0)).diff()
    out["_spot_5m_duplicate_count"] = duplicate_count
    return out


def aggregate_basis(mark: pd.DataFrame, index: pd.DataFrame) -> tuple[pd.Series, dict]:
    m = mark.set_index("timestamp").close.sort_index()
    i = index.set_index("timestamp").close.sort_index()
    md = int(m.index.duplicated(keep="last").sum())
    ix = int(i.index.duplicated(keep="last").sum())
    m = m[~m.index.duplicated(keep="last")]
    i = i[~i.index.duplicated(keep="last")]
    grid = pd.date_range(START, END_EXCLUSIVE, freq=FIVE_M, inclusive="left")
    basis = (m.reindex(grid) / i.reindex(grid).where(i.reindex(grid) > 0) - 1).replace([np.inf, -np.inf], np.nan)
    group = grid.floor("4h")
    count = basis.groupby(group).count()
    mean = basis.groupby(group).mean().where(count.eq(N5))
    return mean, {"mark_duplicate_timestamps": md, "index_duplicate_timestamps": ix,
                  "matched_valid_5m_rows": int(basis.notna().sum()), "basis_formula": "mean((mark_close/index_close)-1) across 48 five-minute intervals"}


def aggregate_trade_day(path: Path, asset: str, market: str, day: dt.date) -> tuple[pd.DataFrame, dict]:
    # Source archive bytes remain untouched; parsing is in-memory from the gzip stream.
    if market == "spot":
        required = {"id", "timestamp", "price", "volume", "side"}
        schema = list(pd.read_csv(path, compression="gzip", nrows=0).columns)
        if not required.issubset(schema):
            raise RuntimeError(f"Unexpected Bybit spot archive schema: {path.name}: {schema}")
        frame = pd.read_csv(path, compression="gzip", usecols=list(required), low_memory=False)
        ts_ms = pd.to_numeric(frame.timestamp, errors="coerce")
        base = pd.to_numeric(frame.volume, errors="coerce")
        price = pd.to_numeric(frame.price, errors="coerce")
        side = frame.side.astype(str).str.lower()
        time_unit = "ms"
        exec_id = frame.id
    else:
        required = {"timestamp", "symbol", "side", "homeNotional", "price", "trdMatchID"}
        schema = list(pd.read_csv(path, compression="gzip", nrows=0).columns)
        if not required.issubset(schema):
            raise RuntimeError(f"Unexpected Bybit futures archive schema: {path.name}: {schema}")
        frame = pd.read_csv(path, compression="gzip", usecols=list(required), low_memory=False)
        symbols = sorted(frame.symbol.astype(str).unique().tolist())
        if symbols != [asset + "USDT"]:
            raise RuntimeError(f"Futures archive symbol mismatch in {path.name}: {symbols}")
        ts_sec = pd.to_numeric(frame.timestamp, errors="coerce")
        ts_ms = np.floor(ts_sec * 1000.0)
        # homeNotional is the base-currency quantity in the archive schema.
        # Preserve raw size unchanged but do not assume it is a base-volume unit.
        base = pd.to_numeric(frame["homeNotional"], errors="coerce")
        price = pd.to_numeric(frame.price, errors="coerce")
        side = frame.side.astype(str).str.lower()
        time_unit = "seconds_fractional"
        exec_id = frame.trdMatchID
    invalid = (~np.isfinite(np.asarray(ts_ms, dtype=float)) | ~np.isfinite(np.asarray(base, dtype=float)) |
               ~np.isfinite(np.asarray(price, dtype=float)) | (np.asarray(base, dtype=float) < 0) |
               (np.asarray(price, dtype=float) <= 0) | (~side.isin(["buy", "sell"])))
    invalid_n = int(invalid.sum())
    if invalid_n:
        raise RuntimeError(f"Invalid trade rows ({invalid_n}) in {path}")
    ts_ms = np.asarray(ts_ms, dtype=np.int64)
    bins = (ts_ms // (4 * 60 * 60 * 1000)) * (4 * 60 * 60 * 1000)
    day_start = int(pd.Timestamp(day, tz="UTC").timestamp() * 1000)
    outside = (ts_ms < day_start) | (ts_ms >= day_start + 24 * 60 * 60 * 1000)
    outside_n = int(outside.sum())
    if outside_n:
        raise RuntimeError(f"Trade timestamps outside archive UTC date ({outside_n}) in {path}")
    work = pd.DataFrame({"bin_ms": bins, "base": np.asarray(base, dtype=float),
                         "quote": np.asarray(base, dtype=float) * np.asarray(price, dtype=float),
                         "buy": side.eq("buy").to_numpy(), "sell": side.eq("sell").to_numpy(),
                         "exec_id": exec_id.astype(str).to_numpy()})
    work["buy_base"] = work.base.where(work.buy, 0.0)
    work["sell_base"] = work.base.where(work.sell, 0.0)
    work["buy_quote"] = work.quote.where(work.buy, 0.0)
    work["sell_quote"] = work.quote.where(work.sell, 0.0)
    agg = work.groupby("bin_ms", sort=True).agg(base_volume=("base", "sum"), quote_from_trades=("quote", "sum"),
                                                 buy_base=("buy_base", "sum"), sell_base=("sell_base", "sum"),
                                                 buy_quote=("buy_quote", "sum"), sell_quote=("sell_quote", "sum"),
                                                 trade_count=("exec_id", "size"))
    agg.index = pd.to_datetime(agg.index, unit="ms", utc=True)
    denom = agg.buy_base + agg.sell_base
    agg[f"{market}_taker_imbalance"] = ((agg.buy_base - agg.sell_base) / denom.where(denom > 0))
    agg[f"{market}_trade_count"] = agg.trade_count.astype("int64")
    agg = agg.drop(columns="trade_count")
    diag = {"file": str(path.relative_to(ROOT)), "rows": int(len(frame)), "schema": schema,
            "duplicate_execution_ids": int(work.exec_id.duplicated().sum()), "invalid_rows": invalid_n,
            "timestamps_outside_utc_file_date": outside_n, "timestamp_representation": time_unit,
            "taker_side_values": sorted(side.unique().tolist())}
    return agg, diag


def aggregate_all_trades(asset: str) -> tuple[pd.DataFrame, dict]:
    days = pd.date_range(START.floor("D"), END_EXCLUSIVE.floor("D"), freq="D", inclusive="left")
    all_frames = []
    file_missing = {"spot": [], "futures": []}
    diagnostics = {"spot": [], "futures": []}
    for market in ("spot", "futures"):
        folder = "spot" if market == "spot" else "trading"
        jobs = []
        for n, stamp in enumerate(days, start=1):
            day = stamp.date()
            d = day.isoformat()
            fname = f"{asset}USDT_{d}.csv.gz" if market == "spot" else f"{asset}USDT{d}.csv.gz"
            path = RAW / "archive" / folder / f"{asset}USDT" / fname
            if not path.exists():
                file_missing[market].append(d)
                continue
            jobs.append((n, path, day))
        parsed = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(aggregate_trade_day, path, asset, market, day): (n, path, day)
                       for n, path, day in jobs}
            for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                n, path, day = futures[future]
                frame, diag = future.result()
                parsed[n] = (frame, diag)
                if done % 180 == 0 or done == len(futures):
                    print(f"{asset} {market} archive parse: {done}/{len(futures)} UTC files", flush=True)
        rows = []
        processing_summary = {"files_processed": len(parsed), "trade_rows": 0, "duplicate_execution_ids": 0,
                              "invalid_rows": 0, "timestamps_outside_file_date": 0, "schema_signatures": {},
                              "timestamp_representations": {}}
        for n in sorted(parsed):
            frame, diag = parsed[n]
            rows.append(frame)
            processing_summary["trade_rows"] += int(diag["rows"])
            processing_summary["duplicate_execution_ids"] += int(diag["duplicate_execution_ids"])
            processing_summary["invalid_rows"] += int(diag["invalid_rows"])
            processing_summary["timestamps_outside_file_date"] += int(diag["timestamps_outside_utc_file_date"])
            signature = ",".join(diag["schema"])
            processing_summary["schema_signatures"][signature] = processing_summary["schema_signatures"].get(signature, 0) + 1
            unit = diag["timestamp_representation"]
            processing_summary["timestamp_representations"][unit] = processing_summary["timestamp_representations"].get(unit, 0) + 1
            if len(diagnostics[market]) < 4 or n % 365 == 0:
                diagnostics[market].append(diag)
        z = pd.concat(rows).sort_index() if rows else pd.DataFrame()
        if not z.empty and z.index.duplicated().any():
            raise RuntimeError(f"Duplicate aggregate timestamps in {asset} {market} trade files")
        z = z.add_prefix("spot_trade_" if market == "spot" else "futures_trade_")
        all_frames.append(z)
        diagnostics[market] = {"sample_and_periodic_file_diagnostics": diagnostics[market],
                               "processing_summary": processing_summary}
    out = pd.concat(all_frames, axis=1).sort_index()
    return out, {"missing_daily_files": file_missing, "markets": diagnostics,
                 "expected_daily_archive_files_per_market": int(len(days))}


def asof_state(values: pd.DataFrame, query_times: pd.DatetimeIndex, value_col: str, max_age: pd.Timedelta) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = values.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    t = x.timestamp.astype("int64").to_numpy()
    q = query_times.astype("int64").to_numpy()
    pos = np.searchsorted(t, q, side="right") - 1
    val = np.full(len(q), np.nan)
    age = np.full(len(q), np.nan)
    event = np.full(len(q), np.datetime64("NaT"), dtype="datetime64[ns]")
    ok = pos >= 0
    ix = pos[ok]
    val[ok] = pd.to_numeric(x[value_col], errors="coerce").to_numpy(dtype=float)[ix]
    age_ns = q[ok] - t[ix]
    age[ok] = age_ns / 3.6e12
    event[ok] = x.timestamp.to_numpy(dtype="datetime64[ns]")[ix]
    too_old = age > max_age / pd.Timedelta(hours=1)
    val[too_old] = np.nan
    return val, age, pd.to_datetime(event, utc=True).to_numpy()


def build_asset(asset: str) -> tuple[pd.DataFrame, dict]:
    spot5 = read_api_chunks(asset, "spot_klines_5m")
    mark5 = read_api_chunks(asset, "mark_klines_5m")
    index5 = read_api_chunks(asset, "index_klines_5m")
    oi = read_api_chunks(asset, "open_interest_4h")
    funding = read_api_chunks(asset, "funding_history")
    spot4 = aggregate_5m_to_4h(spot5, asset)
    basis, basis_diag = aggregate_basis(mark5, index5)
    trades, trade_diag = aggregate_all_trades(asset)
    starts = pd.date_range(START, END_EXCLUSIVE, freq=FOUR_H, inclusive="left")
    out = pd.DataFrame(index=starts)
    out.index.name = "period_start_utc"
    out = out.join(spot4, how="left")
    out["basis_mean"] = basis.reindex(starts)
    out = out.join(trades, how="left")
    # Bybit open-interest timestamps are UTC snapshot timestamps on the 4h grid.
    oi = oi.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    oi_aligned = asof_state(oi, starts + FOUR_H, "open_interest", FOUR_H)
    out["oi_close"] = oi_aligned[0]
    out["oi_snapshot_age_hours"] = oi_aligned[1]
    out["oi_log_change"] = np.log(out.oi_close / out.oi_close.shift(1))
    # Funding is a settlement observation, carried forward at most 12h, matching the frozen as-of rule.
    funding = funding.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    f = asof_state(funding, starts + FOUR_H, "funding_rate", pd.Timedelta(hours=12))
    out["funding_rate_state"] = f[0]
    out["funding_age_hours"] = f[1]
    out["funding_last_event_utc"] = pd.to_datetime(f[2], utc=True)
    out["spot_taker_imbalance"] = out.spot_trade_spot_taker_imbalance
    out["futures_taker_imbalance"] = out.futures_trade_futures_taker_imbalance
    out["spot_trade_count"] = out.spot_trade_spot_trade_count
    out["spot_quote_volume_from_trades"] = out.spot_trade_spot_quote_from_trades
    out["asset"] = asset
    # Frozen targeted-study derivative transformations; all rolling statistics use prior-only windows.
    out["funding_change"] = out.funding_rate_state.diff()
    out["basis_change"] = out.basis_mean.diff()
    for source, name in (("oi_log_change", "oi_shock_30d_robust"),
                         ("futures_taker_imbalance", "flow_shock_30d_robust")):
        past = out[source].shift(1)
        med = past.rolling(180, min_periods=180).median()
        mad = past.rolling(180, min_periods=180).apply(lambda x: float(np.median(np.abs(x - np.median(x)))), raw=True)
        out[name] = (out[source] - med) / (1.4826 * mad.where(mad > 0))
    out["funding_x_oi_shock"] = out.funding_rate_state * out.oi_shock_30d_robust
    out["oi_shock_x_flow_shock"] = out.oi_shock_30d_robust * out.flow_shock_30d_robust
    out["funding_x_basis"] = out.funding_rate_state * out.basis_mean
    out["flow_x_basis"] = out.futures_taker_imbalance * out.basis_mean
    slot = out.index.hour // 4
    weekday = out.index.dayofweek
    for k in range(1, 6):
        out[f"utc_slot_{k}"] = (slot == k).astype("int8")
    for k in range(1, 7):
        out[f"weekday_{k}"] = (weekday == k).astype("int8")
    out["funding_schedule_window"] = out.index.hour.isin([0, 8, 16]).astype("int8")
    future = pd.concat([out.spot_amihud.shift(-k).rename(f"f{k}") for k in range(1, 7)], axis=1)
    out["future_amihud_24h_mean"] = future.mean(axis=1).where(future.notna().all(axis=1))
    out["future_amihud_24h_log1p"] = np.log1p(out.future_amihud_24h_mean)
    out = out.replace([np.inf, -np.inf], np.nan).reset_index()
    out["interval_end_utc"] = out.period_start_utc + FOUR_H
    expected_5m = len(pd.date_range(START, END_EXCLUSIVE, freq=FIVE_M, inclusive="left"))
    def timestamp_audit(x: pd.DataFrame, expected: int, interval: pd.Timedelta) -> dict:
        ts = pd.DatetimeIndex(x.timestamp.dropna().sort_values().unique())
        delta = ts.to_series().diff().dropna()
        return {"expected_rows": expected, "unique_rows": int(len(ts)), "missing_rows_from_expected_grid": int(expected-len(ts)),
                "duplicate_rows": int(x.timestamp.duplicated().sum()),
                "first_timestamp_utc": ts.min().isoformat() if len(ts) else None,
                "last_timestamp_utc": ts.max().isoformat() if len(ts) else None,
                "off_grid_timestamps": int(sum((ts.asi8 % interval.value) != 0)),
                "unexpected_interval_steps": int((delta != interval).sum())}
    diag = {"asset": asset, "spot_5m_rows": len(spot5), "mark_5m_rows": len(mark5), "index_5m_rows": len(index5),
            "oi_rows": len(oi), "funding_rows": len(funding), "basis": basis_diag, "trade_archives": trade_diag,
            "five_minute_timestamps": {"spot": timestamp_audit(spot5, expected_5m, FIVE_M),
                                       "mark": timestamp_audit(mark5, expected_5m, FIVE_M),
                                       "index": timestamp_audit(index5, expected_5m, FIVE_M)},
            "open_interest_timestamps": timestamp_audit(oi, len(starts), FOUR_H),
            "funding_timestamps": {"unique_rows": int(funding.timestamp.nunique()),
                                   "duplicate_rows": int(funding.timestamp.duplicated().sum()),
                                   "first_timestamp_utc": funding.timestamp.min().isoformat() if len(funding) else None,
                                   "last_timestamp_utc": funding.timestamp.max().isoformat() if len(funding) else None,
                                   "timestamp_deltas_hours": {str(float(k)): int(v) for k, v in funding.timestamp.diff().dropna().dt.total_seconds().div(3600).value_counts().sort_index().items()}},
            "duplicate_oi_timestamps": int(read_api_chunks(asset, "open_interest_4h").timestamp.duplicated().sum()),
            "duplicate_funding_timestamps": int(read_api_chunks(asset, "funding_history").timestamp.duplicated().sum())}
    return out, diag


def main() -> None:
    PHASE.mkdir(parents=True, exist_ok=True)
    panels = []
    diagnostics = {}
    for asset in ASSETS:
        frame, diag = build_asset(asset)
        panels.append(frame)
        diagnostics[asset] = diag
    panel = pd.concat(panels, ignore_index=True).sort_values(["asset", "period_start_utc"]).reset_index(drop=True)
    if panel.duplicated(["asset", "period_start_utc"]).any():
        raise RuntimeError("duplicate asset × 4h key")
    panel.to_csv(OUT_PANEL, index=False, compression={"method": "gzip", "mtime": 0}, date_format="%Y-%m-%dT%H:%M:%SZ")
    panel_hash = sha256(OUT_PANEL)
    expected_grid = pd.date_range(START, END_EXCLUSIVE, freq=FOUR_H, inclusive="left")
    validation = {"status": "DATA_PANEL_BUILT", "source": "cached official Bybit raw API payloads and raw public trade archives only",
                  "raw_data_modified": False, "panel_path": str(OUT_PANEL.relative_to(ROOT)), "panel_sha256": panel_hash,
                  "requested_start_utc": START.isoformat(), "requested_end_exclusive_utc": END_EXCLUSIVE.isoformat(),
                  "expected_intervals_per_asset": len(expected_grid), "asset_rows": {}, "asset_diagnostics": diagnostics,
                  "key_unique": True, "one_common_calendar_grid": True, "target_window": "future Amihud mean over exactly t+1..t+6; all six required",
                  "feature_registry_source": "outputs/phase9_external_qml_replication/feature_registry.json"}
    for asset, g in panel.groupby("asset", sort=False):
        required = ["spot_return", "spot_realized_volatility", "spot_downside_semivariance", "spot_amihud",
                    "spot_taker_imbalance", "spot_quote_volume_growth", "funding_rate_state", "oi_log_change",
                    "futures_taker_imbalance", "basis_mean", "future_amihud_24h_mean"]
        ok = np.isfinite(g[required].to_numpy(dtype=float)).all(axis=1)
        validation["asset_rows"][asset] = {"grid_rows": len(g), "complete_required_rows": int(ok.sum()),
                                            "complete_coverage_of_requested_grid_pct": 100 * float(ok.mean()),
                                            "first_grid_date": g.period_start_utc.min().isoformat(),
                                            "last_grid_date": g.period_start_utc.max().isoformat(),
                                            "first_complete_date": g.loc[ok, "period_start_utc"].min().isoformat() if ok.any() else None,
                                            "last_complete_date": g.loc[ok, "period_start_utc"].max().isoformat() if ok.any() else None}
    jwrite("external_panel_validation.json", validation)
    semantic = {
        "venue": "Bybit",
        "comparison_venue": "Binance USDⓈ-M linear perpetual and Binance spot",
        "variables": {
            "spot_amihud": {"binance_definition": "1e6 * abs(4h log return) / 4h USDT quote volume", "bybit_definition": "same formula; 5m Bybit spot close returns aggregated to 4h; 5m turnover aggregated in USDT", "timestamp": "5m candle open timestamp; represented within UTC four-hour start; interval values available at interval end", "parity": "PASS_CONDITIONAL_ON_5M_COVERAGE"},
            "funding_rate_state": {"binance_definition": "last settlement funding rate at or before 4h interval end, maximum age 12 hours", "bybit_definition": "fundingRate at fundingRateTimestamp; same as-of carry for at most 12h", "unit": "decimal funding rate per settlement", "parity": "PASS_IF_HISTORICAL_INTERVAL_AUDIT_CONFIRMS"},
            "oi_log_change": {"binance_definition": "log change in 4h close open interest", "bybit_definition": "log change in Bybit openInterest; docs state BTCUSDT linear OI unit is BTC and the field is sum of both sides", "unit": "BTC for BTCUSDT; ETH for ETHUSDT; log change scale invariant", "timestamp": "documented UTC millisecond timestamp, sampled on 4h grid and aligned as-of interval end", "parity": "PASS_IF_GRID_AND_UNIT_STABILITY_AUDIT_CONFIRMS"},
            "futures_taker_imbalance": {"binance_definition": "(taker buy base volume - taker sell base volume)/(buy+sell)", "bybit_definition": "(base size of Buy-side taker trades - Sell-side taker trades)/(total base size); Bybit API defines side as side of taker", "unit": "dimensionless [-1,1]", "parity": "PASS_IF_ARCHIVE_SIDE_VALIDATION_CONFIRMS"},
            "basis_mean": {"binance_definition": "mean((mark close/index close)-1) over 48 five-minute intervals", "bybit_definition": "same formula from Bybit linear-perpetual mark-price and index-price five-minute candles", "unit": "nonannualized ratio", "parity": "PASS_IF_COMPLETE_MATCHED_5M_COVERAGE"}
        },
        "known_difference": "Same-venue Bybit spot/futures ecosystem rather than Binance; OI base-coin level units differ but only the frozen log change enters the model.",
        "status": "PROVISIONAL_PENDING_EMPIRICAL_COVERAGE_AND_TIMESTAMP_AUDIT"
    }
    jwrite("semantic_parity.json", semantic)
    print(f"Panel built: {len(panel)} asset × 4h rows; sha256={panel_hash}", flush=True)


if __name__ == "__main__":
    main()
