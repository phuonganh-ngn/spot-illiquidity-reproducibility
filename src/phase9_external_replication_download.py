from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "outputs" / "phase9_external_qml_replication"
RAW = ROOT / "data" / "raw" / "phase9_external" / "bybit"
API = "https://api.bybit.com"
ARCHIVE = "https://public.bybit.com"
START = dt.datetime(2022, 11, 10, tzinfo=dt.timezone.utc)
END_EXCLUSIVE = dt.datetime(2026, 10, 8, tzinfo=dt.timezone.utc)
ASSETS = ("BTC", "ETH")
LOCK = threading.Lock()
COUNT = 0
PROGRESS = PHASE / "download_progress.jsonl"
API_WORKERS = 4
THREAD_LOCAL = threading.local()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> tuple[str, int, bool]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return sha256(path), path.stat().st_size, False
    temp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.part")
    with temp.open("xb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(temp, path)
        created = True
    except FileExistsError:
        created = False
    finally:
        temp.unlink(missing_ok=True)
    return sha256(path), path.stat().st_size, created


def log_record(record: dict) -> None:
    with LOCK:
        with PROGRESS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def request_json(session: requests.Session, path: str, params: dict) -> tuple[bytes, dict]:
    url = API + path
    last_error = None
    for attempt in range(8):
        try:
            response = session.get(url, params=params, timeout=(10, 25))
            if response.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(30, 1.5 ** attempt))
                continue
            response.raise_for_status()
            payload = response.content
            decoded = json.loads(payload)
            if decoded.get("retCode") != 0:
                raise RuntimeError(f"Bybit retCode={decoded.get('retCode')}: {decoded.get('retMsg')}")
            meta = {"url": response.url, "status": response.status_code, "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
            return payload, meta
        except Exception as e:
            last_error = e
            if attempt == 7:
                break
            time.sleep(min(30, 1.5 ** attempt))
    raise RuntimeError(f"Request failed after retries: {url} {params}: {last_error}")


def save_api(session: requests.Session, asset: str, dataset: str, path: str, params: dict, stamp: str) -> None:
    global COUNT
    dest = RAW / "api" / asset / dataset / f"{dataset}_{stamp}.json"
    if dest.exists():
        # Existing complete payloads are immutable and may be reused.
        json.loads(dest.read_text(encoding="utf-8"))
        request = requests.Request("GET", API + path, params=params).prepare()
        log_record({"path": str(dest.relative_to(ROOT)), "sha256": sha256(dest), "size_bytes": dest.stat().st_size,
                    "source": request.url, "source_dataset": dataset, "symbol": asset + "USDT",
                    "market_type": "spot" if dataset.startswith("spot") else "linear_perpetual",
                    "retrieval_timestamp_utc": None, "reused": True})
        with LOCK:
            COUNT += 1
        return
    payload, meta = request_json(session, path, params)
    digest, size, created = atomic_bytes(dest, payload)
    record = {"path": str(dest.relative_to(ROOT)), "sha256": digest, "size_bytes": size,
              "source": meta["url"], "source_dataset": dataset, "symbol": asset + "USDT",
              "market_type": "spot" if dataset.startswith("spot") else "linear_perpetual",
              "retrieval_timestamp_utc": meta["retrieved_utc"], "created": created}
    log_record(record)
    with LOCK:
        COUNT += 1
        if COUNT % 100 == 0:
            print(f"API payloads saved/reused: {COUNT}", flush=True)


def save_metadata(session: requests.Session) -> None:
    for asset in ASSETS:
        for category in ("spot", "linear"):
            params = {"category": category, "symbol": asset + "USDT"}
            payload, meta = request_json(session, "/v5/market/instruments-info", params)
            dest = RAW / "api" / asset / "metadata" / f"instruments_{category}.json"
            digest, size, created = atomic_bytes(dest, payload)
            log_record({"path": str(dest.relative_to(ROOT)), "sha256": digest, "size_bytes": size,
                        "source": meta["url"], "source_dataset": "instrument_metadata_" + category,
                        "symbol": asset + "USDT", "market_type": category,
                        "retrieval_timestamp_utc": meta["retrieved_utc"], "created": created})


def download_api() -> None:
    global COUNT
    session = requests.Session()
    session.headers.update({"User-Agent": "Phase9-reproducible-research/1.0"})
    save_metadata(session)
    start_ms = int(START.timestamp() * 1000)
    end_ms = int(END_EXCLUSIVE.timestamp() * 1000)
    five_min = 5 * 60_000
    four_hour = 4 * 60 * 60_000

    # Fixed range requests remain below the exchange's 1000-row kline cap.
    kline_kinds = (
        ("spot_klines_5m", "/v5/market/kline", {"category": "spot", "interval": "5"}),
        ("mark_klines_5m", "/v5/market/mark-price-kline", {"category": "linear", "interval": "5"}),
        ("index_klines_5m", "/v5/market/index-price-kline", {"category": "linear", "interval": "5"}),
    )
    jobs = []
    for asset in ASSETS:
        for dataset, endpoint, common in kline_kinds:
            cursor = start_ms
            while cursor < end_ms:
                stop = min(end_ms, cursor + 900 * five_min)
                params = common | {"symbol": asset + "USDT", "start": cursor, "end": stop - 1, "limit": 1000}
                jobs.append((asset, dataset, endpoint, params, f"{cursor}_{stop}"))
                cursor = stop

        cursor = start_ms
        while cursor < end_ms:
            stop = min(end_ms, cursor + 180 * four_hour)
            params = {"category": "linear", "symbol": asset + "USDT", "intervalTime": "4h",
                      "startTime": cursor, "endTime": stop - 1, "limit": 200}
            jobs.append((asset, "open_interest_4h", "/v5/market/open-interest", params, f"{cursor}_{stop}"))
            cursor = stop

        # A 45-day window contains at most 135 8-hour settlements at the observed 8-hour schedule.
        cursor = start_ms
        while cursor < end_ms:
            stop = min(end_ms, cursor + 45 * 24 * 60 * 60_000)
            params = {"category": "linear", "symbol": asset + "USDT", "startTime": cursor,
                      "endTime": stop - 1, "limit": 200}
            jobs.append((asset, "funding_history", "/v5/market/funding/history", params, f"{cursor}_{stop}"))
            cursor = stop

    def run_job(job):
        asset, dataset, endpoint, params, stamp = job
        if not hasattr(THREAD_LOCAL, "session"):
            THREAD_LOCAL.session = requests.Session()
            THREAD_LOCAL.session.headers.update({"User-Agent": "Phase9-reproducible-research/1.0"})
        save_api(THREAD_LOCAL.session, asset, dataset, endpoint, params, stamp)

    print(f"API payload chunks scheduled: {len(jobs)}; workers={API_WORKERS}", flush=True)
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=API_WORKERS) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        for n, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            try:
                future.result()
            except Exception as e:
                errors.append({"job": n, "error": f"{type(e).__name__}: {e}"})
            if n % 100 == 0 or n == len(futures):
                print(f"API payload chunks processed: {n}/{len(futures)}; errors={len(errors)}", flush=True)
    if errors:
        (PHASE / "api_download_errors.json").write_text(json.dumps(errors, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"{len(errors)} API chunks failed; see api_download_errors.json")


def download_archive_file(job: tuple[str, str, str, str]) -> dict:
    asset, market, day, url = job
    folder = "spot" if market == "spot" else "trading"
    filename = url.rsplit("/", 1)[1]
    dest = RAW / "archive" / folder / (asset + "USDT") / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        digest = sha256(dest)
        return {"path": str(dest.relative_to(ROOT)), "sha256": digest, "size_bytes": dest.stat().st_size,
                "source": url, "source_dataset": market + "_public_trades_daily", "symbol": asset + "USDT",
                "market_type": market, "date_utc": day, "retrieval_timestamp_utc": None, "reused": True}
    temp = dest.with_name(dest.name + f".{os.getpid()}.{threading.get_ident()}.part")
    h = hashlib.sha256()
    size = 0
    try:
        with requests.get(url, stream=True, timeout=(20, 180), headers={"User-Agent": "Phase9-reproducible-research/1.0"}) as response:
            response.raise_for_status()
            response.raw.decode_content = False
            with temp.open("xb") as f:
                for block in response.raw.stream(1024 * 1024, decode_content=False):
                    if not block:
                        continue
                    f.write(block)
                    h.update(block)
                    size += len(block)
                f.flush()
                os.fsync(f.fileno())
        try:
            os.link(temp, dest)
            created = True
        except FileExistsError:
            created = False
        digest = sha256(dest)
        rec = {"path": str(dest.relative_to(ROOT)), "sha256": digest, "size_bytes": dest.stat().st_size,
               "source": url, "source_dataset": market + "_public_trades_daily", "symbol": asset + "USDT",
               "market_type": market, "date_utc": day, "retrieval_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "created": created}
        return rec
    finally:
        temp.unlink(missing_ok=True)


def download_archives() -> None:
    jobs = []
    n_days = (END_EXCLUSIVE.date() - START.date()).days
    for asset in ASSETS:
        for offset in range(n_days):
            day = START.date() + dt.timedelta(days=offset)
            day_s = day.isoformat()
            jobs.append((asset, "spot", day_s, f"{ARCHIVE}/spot/{asset}USDT/{asset}USDT_{day_s}.csv.gz"))
            jobs.append((asset, "futures", day_s, f"{ARCHIVE}/trading/{asset}USDT/{asset}USDT{day_s}.csv.gz"))
    print(f"Archive daily files scheduled: {len(jobs)}", flush=True)
    errors = []
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(download_archive_file, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            try:
                rec = future.result()
                log_record(rec)
            except Exception as e:
                errors.append({"job": list(job), "error": f"{type(e).__name__}: {e}"})
            done += 1
            if done % 100 == 0 or done == len(jobs):
                print(f"Archive files processed: {done}/{len(jobs)}; errors={len(errors)}", flush=True)
    if errors:
        (PHASE / "download_errors.json").write_text(json.dumps(errors, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"{len(errors)} archive files failed; see download_errors.json")


def finalize_manifest() -> None:
    rows = []
    if PROGRESS.exists():
        for line in PROGRESS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    unique = {}
    for row in rows:
        previous = unique.get(row["path"])
        if previous and previous.get("sha256") != row.get("sha256"):
            raise RuntimeError(f"Raw source changed during reproduction: {row['path']}")
        if previous and previous.get("retrieval_timestamp_utc") and not row.get("retrieval_timestamp_utc"):
            row["retrieval_timestamp_utc"] = previous["retrieval_timestamp_utc"]
        unique[row["path"]] = row
    manifest = {"source_policy": "Bybit official V5 public market data API and official public.bybit.com trade archives; raw payloads/files preserved unchanged; no model results used for venue eligibility.",
                "download_start_utc": START.isoformat(), "download_end_exclusive_utc": END_EXCLUSIVE.isoformat(),
                "file_count": len(unique), "files": [unique[k] for k in sorted(unique)]}
    (PHASE / "raw_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-only", action="store_true")
    parser.add_argument("--archive-only", action="store_true")
    args = parser.parse_args()
    PHASE.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    if not args.archive_only:
        download_api()
    if not args.api_only:
        download_archives()
    finalize_manifest()
    print(f"Raw manifest records: {json.loads((PHASE / 'raw_manifest.json').read_text())['file_count']}", flush=True)


if __name__ == "__main__":
    main()
