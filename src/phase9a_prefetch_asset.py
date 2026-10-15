from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from phase9a_build_24m import OUT, ROOT, INTERMEDIATE, audit_panel, asof, build_asset, sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True, choices=("BTC", "ETH"))
    args = parser.parse_args()
    asset = args.asset
    manifest = json.loads((OUT / "required_download_manifest.json").read_text(encoding="utf-8"))
    hashes = {}
    for row in manifest["required_files"]:
        if row["asset"] != asset:
            continue
        path = ROOT / row["path"]
        if not path.is_file() or path.stat().st_size != int(row["size_bytes"]):
            raise RuntimeError(f"Missing/changed in-window raw archive: {row['path']}")
        with path.open("rb") as stream:
            if stream.read(2) != b"\x1f\x8b":
                raise RuntimeError(f"Invalid gzip header: {row['path']}")
        hashes[row["path"]] = {"sha256": row["sha256"], "size_bytes": int(row["size_bytes"])}
    expected = 731 * 2
    if len(hashes) != expected:
        raise RuntimeError(f"Expected {expected} {asset} archives, found {len(hashes)}")

    # Exercise the provider-ms / pandas-subsecond-unit alignment path before
    # accepting market features. This catches silent ms-vs-ns as-of failures.
    fixture_time = pd.Timestamp("2025-01-01T00:00:00Z")
    fixture = pd.DataFrame({"timestamp": pd.to_datetime([fixture_time], utc=True), "value": [7.0]})
    aligned, age, _ = asof(fixture, pd.DatetimeIndex([fixture_time]), "value", pd.Timedelta(hours=4))
    asof_fixture_pass = bool(aligned[0] == 7.0 and age[0] == 0.0)
    if not asof_fixture_pass:
        raise RuntimeError("Datetime precision as-of fixture failed")

    started = time.perf_counter()
    panel, detail = build_asset(asset, hashes)
    audit = audit_panel(panel)
    panel_path = OUT / f"bybit_{asset.lower()}_4h_panel.parquet"
    panel.to_parquet(panel_path, engine="pyarrow", compression="zstd", index=False, version="2.6")

    metadata = []
    for path in sorted((INTERMEDIATE / asset).glob("*.metadata.json")):
        metadata.append(json.loads(path.read_text(encoding="utf-8")))
    raw_bytes = sum(int(item["raw_bytes_processed"]) for item in metadata)
    raw_rows = sum(int(item["streaming_diagnostics"]["rows"]) for item in metadata)
    raw_files = sum(int(item["streaming_diagnostics"]["files"]) for item in metadata)
    coverage = audit["variable_coverage"]
    required_coverage = ["spot_return", "spot_realized_volatility", "spot_amihud", "future_amihud_24h_mean",
                         "funding_rate_state", "oi_log_change", "basis_mean", "futures_taker_imbalance"]
    coverage_pass = all(coverage[name]["coverage_pct"] >= 95.0 for name in required_coverage)
    cache_pass = detail["trade_archives"]["cached_months_reused"] == 25 and detail["trade_archives"]["files_opened"] == 0
    payload = {
        "asset": asset,
        "status": "PASS" if len(metadata) == 25 and len(panel) == 4381 and coverage_pass and asof_fixture_pass else "FAIL",
        "coverage_pass": coverage_pass,
        "required_coverage": {name: coverage[name]["coverage_pct"] for name in required_coverage},
        "asof_precision_fixture": "PASS" if asof_fixture_pass else "FAIL",
        "all_trade_months_reused_from_verified_cache": cache_pass,
        "prefetch_cache_reused": all(part.get("source_hashes") is not None for part in metadata),
        "monthly_partition_count": len(metadata),
        "panel_rows": len(panel),
        "panel_sha256": sha256(panel_path),
        "panel_audit": audit,
        "raw_bytes_processed": raw_bytes,
        "raw_rows_parsed": raw_rows,
        "raw_files_parsed": raw_files,
        "raw_bytes_from_monthly_partition_metadata": raw_bytes,
        "detail": detail,
        "prefetch_wall_seconds": time.perf_counter() - started,
        "predictive_models_run": False,
        "forecast_results_viewed": False,
    }
    write_json(OUT / f"prefetch_{asset.lower()}_data.json", payload)
    print(json.dumps({k: payload[k] for k in ("asset", "status", "monthly_partition_count", "panel_rows",
                                              "raw_bytes_processed", "raw_rows_parsed", "raw_files_parsed",
                                              "prefetch_wall_seconds")}, indent=2), flush=True)
    if payload["status"] != "PASS":
        raise SystemExit(f"{asset} prefetch validation failed")


if __name__ == "__main__":
    main()
