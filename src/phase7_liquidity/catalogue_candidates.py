#!/usr/bin/env python3
"""Read-only Binance archive catalogue audit for predeclared extension assets.

This script does not download market observations and does not inspect forecast
performance. It records object availability for SOL, BNB and XRP over the full
study span before any asset-specific model fit.
"""
from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "phase7_liquidity_replication"
BUCKET = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
ASSETS = ("SOLUSDT", "BNBUSDT", "XRPUSDT")
START_MONTH = "2021-12"
END_MONTH = "2026-09"
NAMESPACE = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
DATASETS = {
    "spot_5m": "data/spot/monthly/klines/{symbol}/5m/",
    "futures_5m": "data/futures/um/monthly/klines/{symbol}/5m/",
    "mark_5m": "data/futures/um/monthly/markPriceKlines/{symbol}/5m/",
    "index_5m": "data/futures/um/monthly/indexPriceKlines/{symbol}/5m/",
    "premium_5m": "data/futures/um/monthly/premiumIndexKlines/{symbol}/5m/",
    "funding_monthly": "data/futures/um/monthly/fundingRate/{symbol}/",
    "metrics_daily": "data/futures/um/daily/metrics/{symbol}/",
}


def list_prefix(prefix: str) -> list[dict]:
    marker = ""
    results: list[dict] = []
    while True:
        params = {"prefix": prefix, "max-keys": 1000}
        if marker:
            params["marker"] = marker
        response = requests.get(BUCKET + "?" + urlencode(params), timeout=(15, 90))
        response.raise_for_status()
        xml = ET.fromstring(response.content)
        for item in xml.findall("s:Contents", NAMESPACE):
            results.append({
                "key": item.findtext("s:Key", namespaces=NAMESPACE),
                "last_modified": item.findtext("s:LastModified", namespaces=NAMESPACE),
                "size": int(item.findtext("s:Size", namespaces=NAMESPACE)),
            })
        if xml.findtext("s:IsTruncated", namespaces=NAMESPACE) != "true":
            break
        marker = xml.findtext("s:NextMarker", namespaces=NAMESPACE) or results[-1]["key"]
        time.sleep(0.05)
    return results


def month_key(key: str) -> str | None:
    name = key.rsplit("/", 1)[-1]
    # Monthly candle/funding archives end in YYYY-MM.zip.
    for token in name.removesuffix(".zip").split("-")[-2:]:
        pass
    stem = name.removesuffix(".zip")
    candidate = stem[-7:]
    if len(candidate) == 7 and candidate[4] == "-" and candidate[:4].isdigit() and candidate[5:].isdigit():
        return candidate
    return None


def day_key(key: str) -> str | None:
    stem = key.rsplit("/", 1)[-1].removesuffix(".zip")
    candidate = stem[-10:]
    if len(candidate) == 10 and candidate[4] == "-" and candidate[7] == "-":
        return candidate
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    expected_months = []
    y, m = 2021, 12
    while (y, m) <= (2026, 9):
        expected_months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    expected_days = set(str(x.date()) for x in __import__("pandas").date_range("2021-12-01", "2026-09-30", freq="D"))

    catalogue: dict[str, dict] = {}
    for symbol in ASSETS:
        catalogue[symbol] = {}
        for dataset, template in DATASETS.items():
            prefix = template.format(symbol=symbol)
            objects = list_prefix(prefix)
            catalogue[symbol][dataset] = {"prefix": prefix, "objects": objects}
            print(f"catalogued {symbol} {dataset}: {len(objects)} objects", flush=True)

    audit = []
    for symbol in ASSETS:
        for dataset, rec in catalogue[symbol].items():
            keys = [x["key"] for x in rec["objects"] if x["key"].endswith(".zip")]
            if dataset == "metrics_daily":
                observed = {d for k in keys if (d := day_key(k)) is not None and "2021-12-01" <= d <= "2026-09-30"}
                expected = expected_days
            else:
                observed = {m for k in keys if (m := month_key(k)) is not None and START_MONTH <= m <= END_MONTH}
                expected = set(expected_months)
            absent = sorted(expected - observed)
            audit.append({
                "symbol": symbol,
                "dataset": dataset,
                "scope": "5m spot/perpetual candles, funding settlements, or 5m OI snapshots",
                "expected_periods": len(expected),
                "catalogued_periods": len(observed),
                "catalogue_coverage_pct": 100 * len(observed & expected) / len(expected),
                "first_period": min(observed) if observed else None,
                "last_period": max(observed) if observed else None,
                "missing_periods": absent,
                "catalogue_only_not_content_validated": True,
            })

    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Binance Data Vision public S3-compatible object catalogue",
        "source_endpoint": BUCKET,
        "assets_predeclared_before_forecast_evaluation": list(ASSETS),
        "span": {"start": "2021-12-01", "end": "2026-09-30"},
        "expected_months": len(expected_months),
        "expected_days": len(expected_days),
        "catalogue_audit": audit,
        "asset_objects": catalogue,
        "forecast_values_or_model_results_read": False,
    }
    path = OUT / "candidate_archive_catalogue.json"
    path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
