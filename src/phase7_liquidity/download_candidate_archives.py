#!/usr/bin/env python3
"""Fetch predeclared non-BTC/ETH candidate archives into a new cache namespace.

BTC/ETH files are reused from Stage 1. This downloader is limited to SOL, BNB,
and XRP and never changes the Stage-1 source manifest or any prior output.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "phase7_liquidity_replication"
RAW = ROOT / "data" / "raw" / "phase7_assets"
CATALOGUE = OUT / "candidate_archive_catalogue.json"
ASSETS = ("SOLUSDT", "BNBUSDT", "XRPUSDT")
REQUIRED_DATASETS = {"spot_5m", "futures_5m", "mark_5m", "index_5m", "funding_monthly", "metrics_daily"}
THREAD_LOCAL = threading.local()


def session() -> requests.Session:
    s = getattr(THREAD_LOCAL, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": "LiquidityReplicationAudit/1.0"})
        THREAD_LOCAL.session = s
    return s


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_one(job: dict) -> dict:
    url = "https://data.binance.vision/" + job["key"]
    identity = hashlib.sha256(url.encode()).hexdigest()
    folder = RAW / job["symbol"] / job["dataset"]
    raw_path = folder / (identity + ".raw")
    meta_path = folder / (identity + ".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if not raw_path.exists() or sha256(raw_path.read_bytes()) != meta["sha256"]:
            raise RuntimeError(f"Cached raw hash mismatch: {raw_path}")
        return {**job, **meta, "cache_hit": True}
    if raw_path.exists():
        raise RuntimeError(f"Raw file exists without metadata; refusing overwrite: {raw_path}")
    response = None
    error = None
    for attempt in range(5):
        try:
            response = session().get(url, timeout=(20, 120))
            if response.status_code not in {429, 500, 502, 503, 504}:
                break
            error = f"HTTP_{response.status_code}"
        except requests.RequestException as exc:
            error = f"{type(exc).__name__}: {exc}"
        if attempt < 4:
            time.sleep(min(2 ** attempt, 20))
    status = response.status_code if response is not None else 0
    if status != 200:
        return {**job, "url": url, "http_status": status, "error": error,
                "retrieval_timestamp": datetime.now(timezone.utc).isoformat(),
                "raw_file_path": None, "sha256": None, "file_size": 0, "cache_hit": False}
    body = response.content
    if not body:
        raise RuntimeError(f"Empty HTTP 200 archive: {url}")
    folder.mkdir(parents=True, exist_ok=True)
    digest = sha256(body)
    with raw_path.open("xb") as f:
        f.write(body)
    meta = {
        "url": url,
        "http_status": status,
        "retrieval_timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_file_path": str(raw_path.relative_to(ROOT)),
        "sha256": digest,
        "file_size": len(body),
        "catalogue_last_modified": job.get("last_modified"),
        "catalogue_size": job.get("catalogue_size"),
    }
    with meta_path.open("x", encoding="utf-8") as f:
        json.dump(meta, f, sort_keys=True, indent=2)
        f.write("\n")
    return {**job, **meta, "cache_hit": False}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cat = json.loads(CATALOGUE.read_text(encoding="utf-8"))
    jobs = []
    for symbol in ASSETS:
        for dataset, record in cat["asset_objects"][symbol].items():
            if dataset not in REQUIRED_DATASETS:
                continue
            for obj in record["objects"]:
                key = obj["key"]
                if not key.endswith(".zip"):
                    continue
                if dataset == "metrics_daily":
                    match = re.search(r"(\d{4}-\d{2}-\d{2})\.zip$", key)
                    date = match.group(1) if match else ""
                    keep = "2021-10-01" <= date <= "2026-09-30"
                    period = date
                else:
                    match = re.search(r"(\d{4}-\d{2})\.zip$", key)
                    period = match.group(1) if match else ""
                    keep = "2021-10" <= period <= "2026-09"
                if keep:
                    jobs.append({"symbol": symbol, "dataset": dataset, "period": period,
                                 "key": key, "last_modified": obj["last_modified"],
                                 "catalogue_size": obj["size"]})
    # Metrics archives have no candidate observations before the exact
    # research-window start in the frozen catalogue; their rolling transforms
    # therefore receive a 30-day warm-up loss rather than fabricated history.
    expected = 3 * (len(REQUIRED_DATASETS - {"metrics_daily"}) * 60 + 1765)
    if len(jobs) != expected:
        raise RuntimeError(f"Candidate download plan mismatch: expected {expected}, got {len(jobs)}")
    print(f"frozen candidate download jobs={len(jobs)}; assets={ASSETS}; no model results inspected", flush=True)

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        future_map = {pool.submit(fetch_one, job): job for job in jobs}
        for n, future in enumerate(concurrent.futures.as_completed(future_map), 1):
            rows.append(future.result())
            if n % 100 == 0 or n == len(future_map):
                print(f"downloaded/cached {n}/{len(future_map)}", flush=True)

    rows.sort(key=lambda x: (x["symbol"], x["dataset"], x["period"]))
    manifest = OUT / "candidate_source_manifest.json"
    manifest.write_text(json.dumps({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Binance Data Vision monthly archives and daily futures metrics",
        "assets": list(ASSETS), "expected_job_count": expected,
        "pre_sample_warmup_start": "2021-10-01",
        "research_window": {"start": "2021-12-01", "end": "2026-09-30"},
        "retrieved_job_count": len(rows), "http_status_counts": {
            str(s): sum(x.get("http_status") == s for x in rows) for s in sorted({x.get("http_status") for x in rows})
        },
        "downloaded_bytes": sum(x.get("file_size", 0) for x in rows),
        "files": rows,
        "phase1_manifest_modified": False,
        "forecast_values_or_model_results_read": False,
    }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    failed = [x for x in rows if x.get("http_status") != 200]
    print(f"manifest={manifest.relative_to(ROOT)} bytes={sum(x.get('file_size',0) for x in rows)} failed={len(failed)}", flush=True)
    if failed:
        print(json.dumps(failed[:25], indent=2), flush=True)
        raise RuntimeError(f"Candidate archive retrieval had {len(failed)} non-200 responses")


if __name__ == "__main__":
    main()
