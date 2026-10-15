from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "phase9_external_qml_replication"
RAW = ROOT / "data" / "raw" / "phase9_external" / "bybit"
INTERMEDIATE = OUT / "intermediate" / "bybit_4h_flow"
START = pd.Timestamp("2024-10-06T20:00:00Z")
END = pd.Timestamp("2026-10-06T20:00:00Z")
FOUR_H = pd.Timedelta(hours=4)
FIVE_M = pd.Timedelta(minutes=5)
LABEL_END_EXCLUSIVE = END + pd.Timedelta(hours=28)
CHUNK_ROWS = 250_000
FLOW_WORKERS = min(4, max(1, (os.cpu_count() or 4) // 8))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_api_rows(asset: str, dataset: str, min_ms: int, max_ms: int) -> pd.DataFrame:
    files = sorted((RAW / "api" / asset / dataset).glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No API payloads for {asset}/{dataset}")
    rows = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("retCode") != 0:
            raise RuntimeError(f"Bybit API error in cached payload: {path}")
        rows.extend(payload.get("result", {}).get("list", []))
    if dataset == "spot_klines_5m":
        columns = ["timestamp_ms", "open", "high", "low", "close", "base_volume", "quote_volume"]
    elif dataset in ("mark_klines_5m", "index_klines_5m"):
        columns = ["timestamp_ms", "open", "high", "low", "close"]
    elif dataset == "open_interest_4h":
        columns = ["timestamp_ms", "open_interest", "single_open_interest"]
        rows = [[r.get("timestamp"), r.get("openInterest"), r.get("singleOpenInterest")] for r in rows]
    elif dataset == "funding_history":
        columns = ["timestamp_ms", "funding_rate"]
        rows = [[r.get("fundingRateTimestamp"), r.get("fundingRate")] for r in rows]
    else:
        raise KeyError(dataset)
    x = pd.DataFrame(rows, columns=columns)
    x["timestamp_ms"] = pd.to_numeric(x.timestamp_ms, errors="coerce")
    x = x.dropna(subset=["timestamp_ms"])
    x["timestamp_ms"] = x.timestamp_ms.astype("int64")
    x = x[(x.timestamp_ms >= min_ms) & (x.timestamp_ms < max_ms)]
    x["timestamp"] = pd.to_datetime(x.timestamp_ms, unit="ms", utc=True)
    x = x.drop(columns="timestamp_ms")
    for col in x.columns:
        if col != "timestamp":
            x[col] = pd.to_numeric(x[col], errors="coerce")
    return x.sort_values("timestamp").reset_index(drop=True)


def build_spot_4h(asset: str) -> tuple[pd.DataFrame, dict]:
    grid_start = START - FOUR_H
    grid = pd.date_range(grid_start, LABEL_END_EXCLUSIVE, freq=FIVE_M, inclusive="left")
    x = read_api_rows(asset, "spot_klines_5m", int(grid[0].timestamp() * 1000),
                      int(LABEL_END_EXCLUSIVE.timestamp() * 1000))
    duplicates = int(x.timestamp.duplicated(keep="last").sum())
    x = x.drop_duplicates("timestamp", keep="last").set_index("timestamp").reindex(grid)
    values = x[["open", "high", "low", "close", "base_volume", "quote_volume"]].to_numpy(float)
    valid = np.isfinite(values).all(axis=1)
    valid &= (values[:, :4] > 0).all(axis=1)
    valid &= (values[:, 4:] >= 0).all(axis=1)
    z = x.copy()
    z.loc[~valid, :] = np.nan
    groups = grid.floor("4h")
    counts = pd.Series(valid.astype(np.int16), index=grid).groupby(groups).sum()
    g = z.groupby(groups)
    out = pd.DataFrame(index=counts.index)
    out["spot_open"] = g.open.first()
    out["spot_high"] = g.high.max()
    out["spot_low"] = g.low.min()
    out["spot_close"] = g.close.last()
    out["spot_base_volume"] = g.base_volume.sum(min_count=48)
    out["spot_quote_volume"] = g.quote_volume.sum(min_count=48)
    out["spot_valid_5m_bars"] = counts
    complete = counts.eq(48)
    r5 = np.log(z.close / z.close.shift(1))
    rcount = r5.groupby(groups).count()
    out["spot_realized_volatility"] = r5.pow(2).groupby(groups).sum(min_count=48).pow(0.5).where(complete & rcount.eq(48))
    out["spot_downside_semivariance"] = r5.clip(upper=0).pow(2).groupby(groups).sum(min_count=48).where(complete & rcount.eq(48))
    incomplete = ~complete
    out.loc[incomplete, ["spot_open", "spot_high", "spot_low", "spot_close", "spot_base_volume", "spot_quote_volume"]] = np.nan
    out["spot_return"] = np.log(out.spot_close / out.spot_close.shift(1))
    out["spot_amihud"] = 1e6 * out.spot_return.abs() / out.spot_quote_volume.where(out.spot_quote_volume > 0)
    out["spot_quote_volume_growth"] = np.log(out.spot_quote_volume.where(out.spot_quote_volume > 0)).diff()
    return out, {
        "asset": asset,
        "expected_5m_timestamps": int(len(grid)),
        "unique_source_5m_timestamps": int(x.index.nunique()),
        "missing_5m_rows_after_grid_reindex": int(x.close.isna().sum()),
        "duplicate_5m_timestamps": duplicates,
        "first_timestamp_utc": grid[0].isoformat(),
        "last_timestamp_utc": grid[-1].isoformat(),
        "timestamp_semantics": "5m candle startTime, UTC milliseconds; closed candles only through the maximum cached timestamp."
    }


def build_basis(asset: str, starts: pd.DatetimeIndex) -> tuple[pd.Series, dict]:
    end = starts.max() + FOUR_H
    lo, hi = int(starts.min().timestamp() * 1000), int(end.timestamp() * 1000)
    mark = read_api_rows(asset, "mark_klines_5m", lo, hi)
    index = read_api_rows(asset, "index_klines_5m", lo, hi)
    md = int(mark.timestamp.duplicated(keep="last").sum())
    idup = int(index.timestamp.duplicated(keep="last").sum())
    m = mark.drop_duplicates("timestamp", keep="last").set_index("timestamp").close
    i = index.drop_duplicates("timestamp", keep="last").set_index("timestamp").close
    grid = pd.date_range(starts.min(), end, freq=FIVE_M, inclusive="left")
    basis = (m.reindex(grid) / i.reindex(grid).where(i.reindex(grid) > 0) - 1).replace([np.inf, -np.inf], np.nan)
    grouped = basis.groupby(grid.floor("4h"))
    count = grouped.count()
    mean = grouped.mean().where(count.eq(48)).reindex(starts)
    return mean, {"mark_duplicates": md, "index_duplicates": idup, "expected_5m_rows": int(len(grid)),
                  "matched_valid_5m_rows": int(basis.notna().sum()), "complete_4h_basis_rows": int(mean.notna().sum()),
                  "formula": "mean(mark_close/index_close - 1) over 48 matched five-minute bars"}


def asof(values: pd.DataFrame, queries: pd.DatetimeIndex, column: str, max_age: pd.Timedelta):
    x = values.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    # pandas 3 preserves millisecond/microsecond precision inferred from the
    # provider's epoch-ms values. Normalize both sides to ns before search;
    # otherwise ms and ns integers compare as if snapshots were decades old.
    t = x.timestamp.dt.as_unit("ns").astype("int64").to_numpy()
    q = queries.as_unit("ns").asi8
    pos = np.searchsorted(t, q, side="right") - 1
    value = np.full(len(q), np.nan)
    age_hours = np.full(len(q), np.nan)
    event = np.full(len(q), np.datetime64("NaT"), dtype="datetime64[ns]")
    ok = pos >= 0
    ix = pos[ok]
    value[ok] = x[column].to_numpy(float)[ix]
    age_hours[ok] = (q[ok] - t[ix]) / 3.6e12
    event[ok] = x.timestamp.to_numpy(dtype="datetime64[ns]")[ix]
    value[age_hours > max_age / pd.Timedelta(hours=1)] = np.nan
    return value, age_hours, pd.to_datetime(event, utc=True)


def raw_path(asset: str, market: str, day: pd.Timestamp) -> Path:
    d = day.strftime("%Y-%m-%d")
    folder = "spot" if market == "spot" else "trading"
    name = f"{asset}USDT_{d}.csv.gz" if market == "spot" else f"{asset}USDT{d}.csv.gz"
    return RAW / "archive" / folder / f"{asset}USDT" / name


def file_hash_manifest() -> dict[str, dict]:
    manifest = json.loads((OUT / "required_download_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("download_failures"):
        raise RuntimeError("Required in-window download failures exist.")
    result = {}
    for row in manifest["required_files"]:
        path = ROOT / row["path"]
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing/empty required archive: {path}")
        if row.get("size_bytes") is not None and path.stat().st_size != int(row["size_bytes"]):
            raise RuntimeError(f"Raw archive size changed after inventory/download: {path}")
        with path.open("rb") as stream:
            if stream.read(2) != b"\x1f\x8b":
                raise RuntimeError(f"Invalid gzip header: {path}")
        digest = row.get("sha256") or sha256(path)
        result[row["path"]] = {"sha256": digest, "size_bytes": path.stat().st_size}
    return result


def aggregate_trade_file(asset: str, market: str, day: pd.Timestamp, lo: pd.Timestamp,
                         hi: pd.Timestamp) -> tuple[dict, dict]:
    path = raw_path(asset, market, day)
    if market == "spot":
        usecols = ["timestamp", "price", "volume", "side"]
        qty_name = "volume"
        representation = "unix_milliseconds"
        dtype = {"timestamp": "string", "price": "float64", "volume": "float64", "side": "string"}
    else:
        usecols = ["timestamp", "symbol", "price", "homeNotional", "side"]
        qty_name = "homeNotional"
        representation = "fractional_unix_seconds"
        dtype = {"timestamp": "string", "symbol": "string", "price": "float64",
                 "homeNotional": "float64", "side": "string"}
    header = pd.read_csv(path, compression="gzip", nrows=0).columns.tolist()
    if not set(usecols).issubset(header):
        raise RuntimeError(f"Unexpected archive schema in {path}: {header}")
    day_ms = int(day.timestamp() * 1000)
    lo_ms, hi_ms = int(lo.timestamp() * 1000), int(hi.timestamp() * 1000)
    accum = defaultdict(lambda: {"buy": 0.0, "sell": 0.0, "n": 0, "quote": 0.0})
    diag = {"path": path.relative_to(ROOT).as_posix(), "rows": 0, "rows_in_window": 0,
            "invalid_rows": 0, "outside_archive_day": 0, "side_values": set(),
            "timestamp_representation": representation, "schema": header}
    for chunk in pd.read_csv(path, compression="gzip", usecols=usecols, dtype=dtype,
                             chunksize=CHUNK_ROWS, low_memory=False):
        diag["rows"] += int(len(chunk))
        if market == "futures" and not chunk.symbol.eq(asset + "USDT").all():
            raise RuntimeError(f"Futures symbol mismatch in {path}")
        raw_t = pd.to_numeric(chunk.timestamp, errors="coerce").to_numpy(float)
        qty = chunk[qty_name].to_numpy(float)
        price = chunk.price.to_numpy(float)
        side = chunk.side.str.lower().to_numpy(dtype=str)
        valid_timestamp = np.isfinite(raw_t)
        ts_ms = np.zeros(len(raw_t), dtype=np.int64)
        ts_ms[valid_timestamp] = np.floor(
            raw_t[valid_timestamp] if market == "spot" else raw_t[valid_timestamp] * 1000.0
        ).astype("int64")
        invalid = (~valid_timestamp | ~np.isfinite(qty) | ~np.isfinite(price) |
                   (qty < 0) | (price <= 0) | ~np.isin(side, ["buy", "sell"]))
        if invalid.any():
            diag["invalid_rows"] += int(invalid.sum())
            raise RuntimeError(f"Invalid trade rows in {path}: {int(invalid.sum())}")
        outside = (ts_ms < day_ms) | (ts_ms >= day_ms + 86_400_000)
        diag["outside_archive_day"] += int(outside.sum())
        if outside.any():
            raise RuntimeError(f"Trade timestamps outside archive UTC date: {path}")
        keep = (ts_ms >= lo_ms) & (ts_ms < hi_ms)
        if not keep.any():
            continue
        diag["rows_in_window"] += int(keep.sum())
        q, p, s = qty[keep], price[keep], side[keep]
        bins = (ts_ms[keep] // 14_400_000) * 14_400_000
        chunk_agg = pd.DataFrame({
            "bin": bins, "buy": np.where(s == "buy", q, 0.0),
            "sell": np.where(s == "sell", q, 0.0), "quote": q * p,
            "n": np.ones(len(q), dtype=np.int32),
        }).groupby("bin", sort=False).agg(buy=("buy", "sum"), sell=("sell", "sum"),
                                            quote=("quote", "sum"), n=("n", "sum"))
        for bin_ms, row in chunk_agg.iterrows():
            a = accum[int(bin_ms)]
            a["buy"] += float(row.buy)
            a["sell"] += float(row.sell)
            a["quote"] += float(row.quote)
            a["n"] += int(row.n)
        diag["side_values"].update(s.tolist())
    diag["side_values"] = sorted(diag["side_values"])
    return dict(accum), diag


def _aggregate_trade_task(args):
    return aggregate_trade_file(*args)


def build_flow(asset: str, hashes: dict[str, dict]) -> tuple[pd.DataFrame, dict]:
    days = pd.date_range(START.floor("D"), END.floor("D"), freq="D")
    months = sorted(set((d.year, d.month) for d in days))
    partitions = []
    diagnostics = {"asset": asset, "files_opened": 0, "rows_read": 0, "rows_in_window": 0,
                   "raw_bytes_processed": 0, "cached_months_reused": 0,
                   "schema_signatures": {}, "timestamp_representations": {},
                   "invalid_rows": 0, "outside_archive_day_rows": 0,
                   "archive_parse_checks": [], "monthly_partitions": []}
    for year, month in months:
        month_dates = [d for d in days if d.year == year and d.month == month]
        expected_sources = []
        for day in month_dates:
            for market in ("spot", "futures"):
                p = raw_path(asset, market, day)
                rel = p.relative_to(ROOT).as_posix()
                if rel not in hashes:
                    raise RuntimeError(f"Raw archive is outside the frozen manifest or missing: {rel}")
                expected_sources.append((rel, hashes[rel]))
        source_hashes = dict(expected_sources)
        part_path = INTERMEDIATE / asset / f"{year:04d}-{month:02d}.parquet"
        meta_path = part_path.with_suffix(".metadata.json")
        if part_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("source_hashes") == source_hashes and sha256(part_path) == meta.get("partition_sha256"):
                part = pd.read_parquet(part_path)
                partitions.append(part)
                diagnostics["cached_months_reused"] += 1
                diagnostics["archive_parse_checks"].extend(meta["streaming_diagnostics"].get("archive_checks", []))
                diagnostics["monthly_partitions"].append({"partition": str(part_path.relative_to(ROOT)),
                                                          "cache_reused": True, "rows": int(len(part)),
                                                          "raw_bytes_processed": 0})
                continue
        accumulated = {"spot": defaultdict(lambda: {"buy": 0.0, "sell": 0.0, "quote": 0.0, "n": 0}),
                      "futures": defaultdict(lambda: {"buy": 0.0, "sell": 0.0, "quote": 0.0, "n": 0})}
        md = {"files": 0, "rows": 0, "rows_in_window": 0, "bytes": 0,
              "schemas": {}, "units": {}, "invalid": 0, "outside": 0, "archive_checks": []}
        tasks = []
        task_sources = []
        for day in month_dates:
            lo = max(START, day)
            hi = min(END + FOUR_H, day + pd.Timedelta(days=1))
            for market in ("spot", "futures"):
                p = raw_path(asset, market, day)
                rel = p.relative_to(ROOT).as_posix()
                tasks.append((asset, market, day, lo, hi))
                task_sources.append((rel, market))
        with ProcessPoolExecutor(max_workers=min(FLOW_WORKERS, len(tasks))) as pool:
            task_results = pool.map(_aggregate_trade_task, tasks, chunksize=1)
            for (flow, file_diag), (rel, market) in zip(task_results, task_sources):
                market_acc = accumulated[market]
                for timestamp, values in flow.items():
                    row = market_acc[timestamp]
                    for col in ("buy", "sell", "quote", "n"):
                        row[col] += values[col]
                md["files"] += 1
                md["rows"] += file_diag["rows"]
                md["rows_in_window"] += file_diag["rows_in_window"]
                md["bytes"] += (ROOT / file_diag["path"]).stat().st_size
                md["invalid"] += file_diag["invalid_rows"]
                md["outside"] += file_diag["outside_archive_day"]
                signature = ",".join(file_diag["schema"])
                md["schemas"][signature] = md["schemas"].get(signature, 0) + 1
                md["units"][file_diag["timestamp_representation"]] = md["units"].get(file_diag["timestamp_representation"], 0) + 1
                md["archive_checks"].append({
                    "path": file_diag["path"],
                    "file_sha256": hashes[rel]["sha256"],
                    "file_size_bytes": hashes[rel]["size_bytes"],
                    "readable_gzip_and_full_csv_parse": True,
                    "rows_parsed": file_diag["rows"],
                    "rows_in_primary_time_window": file_diag["rows_in_window"],
                    "invalid_rows": file_diag["invalid_rows"],
                    "timestamp_representation": file_diag["timestamp_representation"],
                    "schema": file_diag["schema"],
                    "timestamps_within_archive_utc_day": file_diag["outside_archive_day"] == 0,
                })
        rows = []
        for day in month_dates:
            for slot in range(6):
                timestamp = day + slot * FOUR_H
                if timestamp < START or timestamp > END:
                    continue
                stamp = int(timestamp.timestamp() * 1000)
                row = {"period_start_utc": timestamp, "asset": asset}
                for market in ("spot", "futures"):
                    a = accumulated[market].get(stamp)
                    den = (a["buy"] + a["sell"]) if a else 0.0
                    row[f"{market}_taker_imbalance"] = (a["buy"] - a["sell"]) / den if den > 0 else np.nan
                    row[f"{market}_trade_count"] = int(a["n"]) if a else 0
                    row[f"{market}_trade_base_volume"] = den if a else np.nan
                    row[f"{market}_trade_quote_volume"] = a["quote"] if a else np.nan
                rows.append(row)
        part = pd.DataFrame(rows).sort_values("period_start_utc").reset_index(drop=True)
        if part.period_start_utc.duplicated().any():
            raise RuntimeError(f"Duplicate 4h flow timestamps for {asset} {year:04d}-{month:02d}")
        part_path.parent.mkdir(parents=True, exist_ok=True)
        part.to_parquet(part_path, engine="pyarrow", compression="zstd", index=False, version="2.6")
        metadata = {"asset": asset, "year_month": f"{year:04d}-{month:02d}",
                    "source_hashes": source_hashes, "partition_sha256": sha256(part_path),
                    "rows": int(len(part)), "raw_bytes_processed": md["bytes"],
                    "streaming_diagnostics": md}
        write_json(meta_path, metadata)
        partitions.append(part)
        diagnostics["files_opened"] += md["files"]
        diagnostics["rows_read"] += md["rows"]
        diagnostics["rows_in_window"] += md["rows_in_window"]
        diagnostics["raw_bytes_processed"] += md["bytes"]
        diagnostics["invalid_rows"] += md["invalid"]
        diagnostics["outside_archive_day_rows"] += md["outside"]
        diagnostics["archive_parse_checks"].extend(md["archive_checks"])
        for key, value in md["schemas"].items():
            diagnostics["schema_signatures"][key] = diagnostics["schema_signatures"].get(key, 0) + value
        for key, value in md["units"].items():
            diagnostics["timestamp_representations"][key] = diagnostics["timestamp_representations"].get(key, 0) + value
        diagnostics["monthly_partitions"].append({"partition": str(part_path.relative_to(ROOT)),
                                                  "cache_reused": False, "rows": int(len(part)),
                                                  "raw_bytes_processed": md["bytes"]})
        print(f"{asset} flow {year:04d}-{month:02d}: {md['files']} files, {md['rows']:,} rows, {md['bytes']/1e9:.2f} GB", flush=True)
    flow = pd.concat(partitions, ignore_index=True).sort_values("period_start_utc").reset_index(drop=True)
    if flow.period_start_utc.duplicated().any():
        raise RuntimeError(f"Duplicate cached 4h flow rows for {asset}")
    return flow.set_index("period_start_utc"), diagnostics


def build_asset(asset: str, hashes: dict[str, dict]) -> tuple[pd.DataFrame, dict]:
    started = time.perf_counter()
    spot, spot_diag = build_spot_4h(asset)
    starts = pd.date_range(START - FOUR_H, END + pd.Timedelta(hours=24), freq=FOUR_H)
    basis, basis_diag = build_basis(asset, starts)
    oi = read_api_rows(asset, "open_interest_4h", int((START - pd.Timedelta(hours=12)).timestamp() * 1000),
                       int((END + FOUR_H).timestamp() * 1000))
    funding = read_api_rows(asset, "funding_history", int((START - pd.Timedelta(hours=24)).timestamp() * 1000),
                            int((END + FOUR_H).timestamp() * 1000))
    oi_dup = int(oi.timestamp.duplicated(keep="last").sum())
    funding_dup = int(funding.timestamp.duplicated(keep="last").sum())
    funding_events = funding.drop_duplicates("timestamp", keep="last").sort_values("timestamp").timestamp
    funding_gaps = funding_events.diff().dropna().dt.total_seconds().div(3600)
    oi_val, oi_age, oi_event = asof(oi, starts + FOUR_H, "open_interest", FOUR_H)
    funding_val, funding_age, funding_event = asof(funding, starts + FOUR_H, "funding_rate", pd.Timedelta(hours=12))
    base = spot.reindex(starts).copy()
    base["basis_mean"] = basis.reindex(starts)
    base["oi_close"] = oi_val
    base["oi_snapshot_age_hours"] = oi_age
    base["oi_snapshot_utc"] = oi_event
    base["funding_rate_state"] = funding_val
    base["funding_age_hours"] = funding_age
    base["funding_last_event_utc"] = funding_event
    base["oi_log_change"] = np.log(base.oi_close / base.oi_close.shift(1))
    oi_change = pd.to_numeric(base.oi_log_change, errors="coerce").dropna()
    base["funding_change"] = base.funding_rate_state.diff()
    base["basis_change"] = base.basis_mean.diff()
    flow, flow_diag = build_flow(asset, hashes)
    base = base.join(flow, how="left")
    # Rolling shock formulas are copied exactly; transaction archives before START are never opened.
    for source, name in (("oi_log_change", "oi_shock_30d_robust"),
                         ("futures_taker_imbalance", "flow_shock_30d_robust")):
        past = base[source].shift(1)
        median = past.rolling(180, min_periods=180).median()
        mad = past.rolling(180, min_periods=180).apply(lambda x: float(np.median(np.abs(x - np.median(x)))), raw=True)
        base[name] = (base[source] - median) / (1.4826 * mad.where(mad > 0))
    base["funding_x_oi_shock"] = base.funding_rate_state * base.oi_shock_30d_robust
    base["oi_shock_x_flow_shock"] = base.oi_shock_30d_robust * base.flow_shock_30d_robust
    base["funding_x_basis"] = base.funding_rate_state * base.basis_mean
    base["flow_x_basis"] = base.futures_taker_imbalance * base.basis_mean
    slot, weekday = base.index.hour // 4, base.index.dayofweek
    for k in range(1, 6):
        base[f"utc_slot_{k}"] = (slot == k).astype("int8")
    for k in range(1, 7):
        base[f"weekday_{k}"] = (weekday == k).astype("int8")
    base["funding_schedule_window"] = base.index.hour.isin([0, 8, 16]).astype("int8")
    fut = pd.concat([base.spot_amihud.shift(-k).rename(f"f{k}") for k in range(1, 7)], axis=1)
    base["future_amihud_24h_mean"] = fut.mean(axis=1).where(fut.notna().all(axis=1))
    base["future_amihud_24h_log1p"] = np.log1p(base.future_amihud_24h_mean)
    panel = base.loc[(base.index >= START) & (base.index <= END)].copy()
    panel.index.name = "period_start_utc"
    panel = panel.reset_index()
    if "asset" in panel.columns:
        if not panel["asset"].eq(asset).all():
            raise RuntimeError(f"Cached flow asset label mismatch while building {asset}")
    else:
        panel.insert(1, "asset", asset)
    panel["interval_end_utc"] = panel.period_start_utc + FOUR_H
    panel["predictor_available_at_utc"] = panel.interval_end_utc
    panel["target_start_utc"] = panel.period_start_utc + FOUR_H
    panel["target_end_utc"] = panel.period_start_utc + pd.Timedelta(hours=28)
    panel["overlapping_24h_target"] = True
    diag = {"asset": asset, "spot_kline": spot_diag, "basis": basis_diag,
            "open_interest": {"rows": int(len(oi)), "duplicate_timestamps": oi_dup,
                              "missing_after_asof": int(np.isnan(oi_val).sum()),
                              "max_age_hours": float(np.nanmax(oi_age)),
                              "oi_log_change_quantiles": {str(q): float(oi_change.quantile(q)) for q in [0, .001, .01, .5, .99, .999, 1]} if len(oi_change) else {},
                              "abs_oi_log_change_exceedances": {str(t): int((oi_change.abs() > t).sum()) for t in [.10, .25, .50]},
                              "largest_abs_oi_log_change": float(oi_change.abs().max()) if len(oi_change) else None},
            "funding": {"rows": int(len(funding)), "duplicate_timestamps": funding_dup,
                        "missing_after_asof": int(np.isnan(funding_val).sum()),
                        "max_age_hours": float(np.nanmax(funding_age)),
                        "unique_settlement_events": int(funding_events.nunique()),
                        "settlement_gap_hours_counts": {str(float(k)): int(v) for k, v in funding_gaps.value_counts().sort_index().items()},
                        "settlement_utc_hour_counts": {str(int(k)): int(v) for k, v in funding_events.dt.hour.value_counts().sort_index().items()}},
            "trade_archives": flow_diag, "build_wall_seconds": time.perf_counter() - started}
    return panel, diag


def audit_panel(panel: pd.DataFrame) -> dict:
    registry = json.loads((OUT / "feature_registry_bybit_frozen.json").read_text())
    spot = registry["registry"]["spot_only"]["base_variables"]
    der = registry["registry"]["spot_plus_derivatives"]["adds_base_variables"]
    static = registry["registry"]["spot_plus_derivatives"]["static_derivative_features"]
    cal = [*[f"utc_slot_{i}" for i in range(1, 6)], *[f"weekday_{i}" for i in range(1, 7)], "funding_schedule_window"]
    g = panel.sort_values("period_start_utc").reset_index(drop=True)
    x0, xd = {}, {}
    for col in spot:
        for lag in range(6):
            x0[f"{col}__lag{lag}"] = g[col].shift(lag)
    for col in der:
        for lag in range(6):
            xd[f"{col}__lag{lag}"] = g[col].shift(lag)
    for col in cal:
        x0[col] = g[col]
        xd[col] = g[col]
    for col in static:
        xd[col] = g[col]
    x0m, xdm = pd.DataFrame(x0), pd.DataFrame(xd)
    y = pd.to_numeric(g.future_amihud_24h_log1p, errors="coerce").to_numpy(float)
    valid = np.isfinite(x0m.to_numpy(float)).all(axis=1) & np.isfinite(xdm.to_numpy(float)).all(axis=1) & np.isfinite(y)
    variables = ["spot_return", "spot_realized_volatility", "spot_downside_semivariance", "spot_amihud",
                 "spot_taker_imbalance", "spot_quote_volume_growth", "funding_rate_state", "oi_log_change",
                 "futures_taker_imbalance", "basis_mean", "funding_change", "basis_change",
                 "oi_shock_30d_robust", "flow_shock_30d_robust", "funding_x_oi_shock",
                 "oi_shock_x_flow_shock", "funding_x_basis", "flow_x_basis",
                 "future_amihud_24h_mean", "future_amihud_24h_log1p"]
    coverage = {}
    for col in variables:
        v = pd.to_numeric(g[col], errors="coerce").to_numpy(float)
        coverage[col] = {"valid": int(np.isfinite(v).sum()), "expected": int(len(g)),
                         "coverage_pct": float(np.isfinite(v).mean() * 100),
                         "infinite": int(np.isinf(v).sum())}
    numeric = g.select_dtypes(include=[np.number]).to_numpy(float)
    return {"asset": str(g.asset.iloc[0]), "rows": int(len(g)), "expected_timestamps": 4381,
            "first_date": g.period_start_utc.min().isoformat(), "last_date": g.period_start_utc.max().isoformat(),
            "duplicate_period_start": int(g.period_start_utc.duplicated().sum()),
            "variable_coverage": coverage, "complete_common_model_rows": int(valid.sum()),
            "complete_common_model_first_date": g.loc[valid, "period_start_utc"].min().isoformat() if valid.any() else None,
            "complete_common_model_last_date": g.loc[valid, "period_start_utc"].max().isoformat() if valid.any() else None,
            "spot_only_input_dimension": len(x0m.columns), "spot_plus_derivatives_input_dimension": len(xdm.columns),
            "rows_with_nonfinite_numeric": int((~np.isfinite(numeric)).any(axis=1).sum()),
            "impossible_values": {"spot_close_nonpositive": int((g.spot_close <= 0).sum()),
                                  "spot_quote_volume_negative": int((g.spot_quote_volume < 0).sum()),
                                  "basis_abs_gt_5pct": int((g.basis_mean.abs() > 0.05).sum()),
                                  "futures_taker_imbalance_out_of_bounds": int(((g.futures_taker_imbalance < -1-1e-10) | (g.futures_taker_imbalance > 1+1e-10)).sum()),
                                  "spot_taker_imbalance_out_of_bounds": int(((g.spot_taker_imbalance < -1-1e-10) | (g.spot_taker_imbalance > 1+1e-10)).sum())},
            "target_definition_check": "future_amihud_24h_mean[t] is mean of spot_amihud[t+1:t+6], all six future bars required; target_end_utc=t+28h.",
            "no_imputation": True}


def main() -> None:
    started = time.perf_counter()
    hashes = file_hash_manifest()
    assets, details = {}, {}
    for asset in ("BTC", "ETH"):
        print(f"Building frozen 24-month Bybit panel: {asset}", flush=True)
        panel, detail = build_asset(asset, hashes)
        panel_audit = audit_panel(panel)
        path = OUT / f"bybit_{asset.lower()}_4h_panel.parquet"
        panel.to_parquet(path, engine="pyarrow", compression="zstd", index=False, version="2.6")
        assets[asset] = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                         "size_bytes": path.stat().st_size, "audit": panel_audit}
        details[asset] = detail
    complete_archives = list((RAW / "archive").rglob("*.csv.gz"))
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sample_window_sha256": sha256(OUT / "sample_window.json"),
        "sample_window": {"T_start": START.isoformat(), "T_end_inclusive": END.isoformat(),
                          "expected_forecast_origins_per_asset": 4381},
        "assets": assets,
        "per_asset_input_diagnostics": details,
        "construction": {
            "transaction_files_in_primary_window": 2924,
            "transaction_files_opened_or_cache_verified": 2924,
            "transaction_files_skipped_outside_primary_window": max(0, len(complete_archives) - 2924),
            "raw_bytes_actually_processed": int(sum(details[a]["trade_archives"]["raw_bytes_processed"] for a in details)),
            "raw_bytes_actually_processed_GB": sum(details[a]["trade_archives"]["raw_bytes_processed"] for a in details) / 1e9,
            "intermediate_partition_root": str(INTERMEDIATE.relative_to(ROOT)),
            "parser": f"At most {FLOW_WORKERS} day-archive workers per month; each streams gzip CSV using only required columns in 250,000-row chunks and immediately aggregates to 4h; tick-level frames are released after each chunk.",
            "parallel_worker_count": FLOW_WORKERS,
            "cache_identity": "Every monthly 4h flow Parquet partition records SHA-256 and size for each required raw source file.",
            "wall_seconds": time.perf_counter() - started
        },
        "predictive_models_run": False,
        "forecast_results_viewed": False
    }
    write_json(OUT / "panel_audit.json", audit)
    parsed = [check for asset in details.values() for check in asset["trade_archives"]["archive_parse_checks"]]
    manifest = json.loads((OUT / "required_download_manifest.json").read_text(encoding="utf-8"))
    required = manifest["required_files"]
    required_paths = [row["path"] for row in required]
    parsed_paths = [row["path"] for row in parsed]
    integrity = {
        "required_archive_count": len(required),
        "successfully_full_parsed_archive_count": len(parsed),
        "unique_required_paths": len(set(required_paths)),
        "unique_parsed_paths": len(set(parsed_paths)),
        "missing_parse_records": sorted(set(required_paths) - set(parsed_paths)),
        "duplicate_required_paths": len(required_paths) - len(set(required_paths)),
        "duplicate_parsed_paths": len(parsed_paths) - len(set(parsed_paths)),
        "all_gzip_streams_and_csv_files_read_to_eof": len(parsed) == len(required) and set(parsed_paths) == set(required_paths),
        "all_rows_schema_checked_and_timestamp_checked": all(
            row["readable_gzip_and_full_csv_parse"] and row["timestamps_within_archive_utc_day"]
            and row["invalid_rows"] == 0 for row in parsed
        ),
        "files": parsed,
        "manifest_status_counts": manifest.get("required_status_counts", {}),
        "additional_download_bytes": manifest.get("additional_download_bytes", 0),
        "all_required_files_have_sha256": all(bool(row.get("sha256")) for row in required),
    }
    integrity["status"] = "PASS" if (
        integrity["all_gzip_streams_and_csv_files_read_to_eof"]
        and integrity["all_rows_schema_checked_and_timestamp_checked"]
        and integrity["duplicate_required_paths"] == 0
        and integrity["duplicate_parsed_paths"] == 0
        and integrity["all_required_files_have_sha256"]
    ) else "FAIL"
    write_json(OUT / "raw_integrity_report.json", integrity)
    bytes_this_invocation = int(sum(details[a]["trade_archives"]["raw_bytes_processed"] for a in details))
    rows_this_invocation = int(sum(details[a]["trade_archives"]["rows_read"] for a in details))
    files_this_invocation = int(sum(details[a]["trade_archives"]["files_opened"] for a in details))
    prefetches = {}
    for asset in ("BTC", "ETH"):
        prefetch_path = OUT / f"prefetch_{asset.lower()}_data.json"
        if prefetch_path.exists():
            prefetches[asset] = json.loads(prefetch_path.read_text(encoding="utf-8"))
    prefetch_bytes = int(sum(p.get("raw_bytes_processed", 0) for p in prefetches.values()))
    prefetch_rows = int(sum(p.get("raw_rows_parsed", 0) for p in prefetches.values()))
    prefetch_files = int(sum(p.get("raw_files_parsed", 0) for p in prefetches.values()))
    total_build_bytes = bytes_this_invocation + prefetch_bytes
    total_build_rows = rows_this_invocation + prefetch_rows
    total_build_files = files_this_invocation + prefetch_files
    audit["construction"].update({
        "raw_bytes_this_final_build_invocation": bytes_this_invocation,
        "raw_rows_this_final_build_invocation": rows_this_invocation,
        "transaction_files_opened_this_final_build_invocation": files_this_invocation,
        "prefetch_raw_bytes_processed": prefetch_bytes,
        "prefetch_raw_rows_parsed": prefetch_rows,
        "prefetch_files_parsed": prefetch_files,
        "prefetch_assets": sorted(prefetches),
        "raw_bytes_actually_processed": total_build_bytes,
        "raw_bytes_actually_processed_GB": total_build_bytes / 1e9,
        "raw_rows_actually_parsed": total_build_rows,
        "transaction_files_actually_parsed": total_build_files,
    })
    write_json(OUT / "panel_audit.json", audit)
    print(json.dumps({"rows": {a: assets[a]["audit"]["rows"] for a in assets},
                      "common_complete_rows": {a: assets[a]["audit"]["complete_common_model_rows"] for a in assets},
                      "raw_GB_processed": audit["construction"]["raw_bytes_actually_processed_GB"],
                      "wall_seconds": audit["construction"]["wall_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
