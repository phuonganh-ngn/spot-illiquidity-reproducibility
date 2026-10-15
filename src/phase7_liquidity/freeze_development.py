from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "phase7_liquidity_replication"
DEV_LAST = pd.Timestamp("2023-06-30T20:00:00Z")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    freeze = json.loads((OUT / "phase7_freeze_manifest.json").read_text())
    protocol_path = OUT / "phase7_protocol.json"
    dev_path = OUT / "phase7_development_cv.json"
    selected_path = OUT / "phase7_selected_models.json"
    oof_path = OUT / "phase7_development_oof_predictions.csv.gz"
    audit_path = OUT / "phase7_data_audit.json"
    if (OUT / "phase7_external_evaluation.json").exists():
        raise RuntimeError("External evaluation already exists; development freeze must precede it")
    if not freeze.get("protocol_frozen") or sha(protocol_path) != freeze["phase7_protocol_sha256"]:
        raise RuntimeError("Protocol freeze failed")
    audit = json.loads(audit_path.read_text())
    if not audit.get("data_audit_pass") or sha(audit_path) != freeze["data_audit_sha256"]:
        raise RuntimeError("Frozen data audit failed or changed")
    if sha(ROOT / "data/processed/phase7/market_liquidity_panel_4h.csv.gz") != freeze["frozen_inputs_and_rules"]["data/processed/phase7/market_liquidity_panel_4h.csv.gz"]:
        raise RuntimeError("Frozen panel changed")

    dev = json.loads(dev_path.read_text())
    selected = json.loads(selected_path.read_text())
    if dev.get("protocol_sha256") != freeze["phase7_protocol_sha256"] or selected.get("protocol_sha256") != freeze["phase7_protocol_sha256"]:
        raise RuntimeError("Development outputs use the wrong protocol")
    if dev.get("external_outcome_values_parsed") is not False:
        raise RuntimeError("Development run reports external outcome access")
    if dev.get("development_oof_sha256") != sha(oof_path):
        raise RuntimeError("OOF prediction hash does not match development report")
    if len(selected.get("selected_models", [])) != 360 or len(selected.get("transform_by_asset_target", [])) != 15:
        raise RuntimeError("Unexpected selected model / transform count")
    oof = pd.read_csv(oof_path, usecols=["asset", "target", "infoset", "family", "date", "y_raw", "prediction_raw"], parse_dates=["date"])
    if oof.duplicated(["asset", "target", "infoset", "family", "date"]).any():
        raise RuntimeError("Duplicate development OOF keys")
    if oof["date"].max() > DEV_LAST or oof[["y_raw", "prediction_raw"]].isna().any().any():
        raise RuntimeError("Invalid or out-of-window development OOF values")

    files = [
        "data/processed/phase7/market_liquidity_panel_4h.csv.gz",
        "outputs/phase7_liquidity_replication/phase7_protocol.json",
        "outputs/phase7_liquidity_replication/phase7_preexternal_protocol.md",
        "outputs/phase7_liquidity_replication/phase7_protocol_amendment_001.md",
        "outputs/phase7_liquidity_replication/phase7_protocol_amendment_002.md",
        "outputs/phase7_liquidity_replication/phase7_data_audit.json",
        "outputs/phase7_liquidity_replication/phase7_development_cv.json",
        "outputs/phase7_liquidity_replication/phase7_selected_models.json",
        "outputs/phase7_liquidity_replication/phase7_development_oof_predictions.csv.gz",
        "src/phase7_liquidity/phase7_run_development.py",
    ]
    manifest = {
        "status": "DEVELOPMENT_FROZEN_PRE_EXTERNAL",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": freeze["phase7_protocol_sha256"],
        "development_outcome_values_only": True,
        "external_outcomes_parsed": False,
        "fit_count": dev["fit_count"],
        "selected_model_count": len(selected["selected_models"]),
        "selected_transform_count": len(selected["transform_by_asset_target"]),
        "oof_rows": len(oof),
        "oof_first_date": oof["date"].min().isoformat(),
        "oof_last_date": oof["date"].max().isoformat(),
        "files": {name: sha(ROOT / name) for name in files},
    }
    path = OUT / "phase7_development_freeze.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"development_frozen": True, "manifest": str(path.relative_to(ROOT)), **{k: manifest[k] for k in ("fit_count", "selected_model_count", "selected_transform_count", "oof_rows", "oof_first_date", "oof_last_date")}}, indent=2))


if __name__ == "__main__":
    main()
