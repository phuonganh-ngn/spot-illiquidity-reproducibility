from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "phase9_external_qml_replication"
MANIFEST = OUT / "required_download_manifest.json"
LOG = OUT / "download_24m_log.jsonl"
LOCK = threading.Lock()
WORKERS = 8


def append_log(record: dict) -> None:
    with LOCK:
        with LOG.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


def download_one(item: dict) -> dict:
    rel = Path(item["path"])
    dest = ROOT / rel
    if dest.exists():
        raise RuntimeError(f"Refusing to overwrite a raw file already present: {rel}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_name(dest.name + f".{os.getpid()}.{threading.get_ident()}.part")
    headers = {"User-Agent": "Phase9-Bybit-24m-replication/1.0"}
    last_error = None
    for attempt in range(7):
        try:
            h = hashlib.sha256()
            n = 0
            with requests.get(item["source_url"], stream=True, timeout=(20, 180), headers=headers) as resp:
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"retryable HTTP {resp.status_code}")
                resp.raise_for_status()
                with temp.open("xb") as stream:
                    for block in resp.iter_content(chunk_size=1024 * 1024):
                        if not block:
                            continue
                        stream.write(block)
                        h.update(block)
                        n += len(block)
                    stream.flush()
                    os.fsync(stream.fileno())
                expected = resp.headers.get("Content-Length")
                if expected is not None and n != int(expected):
                    raise IOError(f"Content-Length mismatch: got {n}, expected {expected}")
            with temp.open("rb") as stream:
                magic = stream.read(2)
            if n <= 0 or magic != b"\x1f\x8b":
                raise IOError(f"Invalid gzip payload: size={n}, magic={magic.hex()}")
            try:
                os.link(temp, dest)
            except FileExistsError as exc:
                raise RuntimeError(f"Destination appeared during download; preserving both: {dest}") from exc
            temp.unlink(missing_ok=True)
            return {
                "path": rel.as_posix(),
                "source_url": item["source_url"],
                "asset": item["asset"],
                "market_type": item["market_type"],
                "archive_date_utc": item["archive_date_utc"],
                "retrieval_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "size_bytes": n,
                "sha256": h.hexdigest(),
                "integrity_status": "DOWNLOADED_HASHED_GZIP_HEADER_PASS_FULL_PARSE_PENDING",
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            temp.unlink(missing_ok=True)
            if isinstance(exc, requests.HTTPError) and getattr(exc.response, "status_code", None) == 404:
                break
            if attempt < 6:
                time.sleep(min(30, 1.7**attempt))
    return {
        "path": rel.as_posix(),
        "source_url": item["source_url"],
        "asset": item["asset"],
        "market_type": item["market_type"],
        "archive_date_utc": item["archive_date_utc"],
        "retrieval_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "error": last_error,
        "integrity_status": "DOWNLOAD_FAIL",
    }


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    todo = [r for r in manifest["required_files"] if r["integrity_status"] in (
        "MISSING", "PRESENT_CORRUPT_OR_UNVERIFIED")]
    # Reuse only present files which had a positive size and gzip header at inventory time.
    for item in todo:
        p = ROOT / item["path"]
        if p.exists():
            raise RuntimeError(f"Corrupt existing raw file needs isolated handling; will not overwrite: {p}")
    print(f"Frozen-window archive downloads only: {len(todo)}; workers={WORKERS}", flush=True)
    done = 0
    failures = []
    total_bytes = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(download_one, item) for item in todo]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            append_log(result)
            done += 1
            total_bytes += int(result.get("size_bytes", 0))
            if result.get("error"):
                failures.append(result)
            if done % 25 == 0 or done == len(todo):
                print(f"24m files processed: {done}/{len(todo)}; new_GB={total_bytes/1e9:.2f}; failures={len(failures)}", flush=True)
    log_rows = []
    if LOG.exists():
        for line in LOG.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if row.get("path"):
                    log_rows.append(row)
            except Exception:
                continue
    latest = {r["path"]: r for r in log_rows}
    for row in manifest["required_files"]:
        if row["path"] in latest and latest[row["path"]].get("integrity_status", "").startswith("DOWNLOADED"):
            saved = latest[row["path"]]
            row.update({k: saved.get(k) for k in ("size_bytes", "sha256", "retrieval_timestamp_utc", "integrity_status")})
        elif row["integrity_status"] == "PRESENT_VALID_HASH_FROM_DOWNLOAD_LOG" or row["integrity_status"] == "PRESENT_VALID_REHASHED":
            row["integrity_status"] = "PRESENT_VALID_HASH_FROM_DOWNLOAD_LOG" if row.get("sha256") else "PRESENT_VALID_REHASHED"
        elif row["path"] in latest and latest[row["path"]].get("error"):
            row["integrity_status"] = "DOWNLOAD_FAIL"
            row["download_error"] = latest[row["path"]]["error"]
    manifest["completed_download_count"] = sum(1 for r in latest.values() if r.get("integrity_status", "").startswith("DOWNLOADED"))
    manifest["additional_download_bytes"] = sum(int(r.get("size_bytes", 0)) for r in latest.values() if r.get("integrity_status", "").startswith("DOWNLOADED"))
    manifest["download_failures"] = failures
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(f"{len(failures)} required archive downloads failed; see {LOG}")
    print(f"Completed {manifest['completed_download_count']} missing-window files; {manifest['additional_download_bytes']/1e9:.2f} GB downloaded.", flush=True)


if __name__ == "__main__":
    main()
