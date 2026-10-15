from __future__ import annotations

import concurrent.futures
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "phase9_external_qml_replication"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_one(row: dict) -> dict:
    path = ROOT / row["path"]
    result = {"path": row["path"], "expected_sha256": row.get("sha256"),
              "expected_size_bytes": row.get("size_bytes"), "exists": path.is_file()}
    if not path.is_file():
        result.update({"actual_sha256": None, "actual_size_bytes": None, "sha256_match": False,
                       "gzip_header_valid": False, "status": "MISSING"})
        return result
    result["actual_size_bytes"] = path.stat().st_size
    with path.open("rb") as stream:
        result["gzip_header_valid"] = stream.read(2) == b"\x1f\x8b"
    result["actual_sha256"] = sha256_file(path)
    result["sha256_match"] = result["actual_sha256"] == result["expected_sha256"]
    result["status"] = "PASS" if result["sha256_match"] and result["gzip_header_valid"] and result["actual_size_bytes"] == result["expected_size_bytes"] else "FAIL"
    return result


def main() -> None:
    t0 = time.perf_counter()
    manifest = json.loads((OUT / "required_download_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("download_failures"):
        raise RuntimeError("The frozen-window download manifest records failures")
    rows = manifest["required_files"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        checks = list(pool.map(check_one, rows))
    paths = [row["path"] for row in checks]
    passed = sum(row["status"] == "PASS" for row in checks)
    payload = {
        "status": "PASS" if passed == len(rows) and len(set(paths)) == len(paths) else "FAIL",
        "required_file_count": len(rows), "verified_file_count": passed,
        "unique_required_paths": len(set(paths)), "duplicate_required_paths": len(paths) - len(set(paths)),
        "sha256_mismatches": [row["path"] for row in checks if not row.get("sha256_match", False)],
        "gzip_header_failures": [row["path"] for row in checks if not row.get("gzip_header_valid", False)],
        "size_mismatches": [row["path"] for row in checks if row.get("actual_size_bytes") != row.get("expected_size_bytes")],
        "total_raw_bytes_verified": int(sum(row.get("actual_size_bytes") or 0 for row in checks)),
        "worker_count": 4, "wall_seconds": time.perf_counter() - t0,
        "files": checks,
    }
    (OUT / "raw_hash_verification.json").write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ["status", "required_file_count", "verified_file_count",
                                           "total_raw_bytes_verified", "worker_count", "wall_seconds"]}, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit("Raw archive SHA-256 verification failed; do not proceed to model freeze.")


if __name__ == "__main__":
    main()
