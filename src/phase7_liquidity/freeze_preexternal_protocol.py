from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "phase7_liquidity_replication"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    audit = json.loads((OUT / "phase7_data_audit.json").read_text())
    if not audit.get("data_audit_pass") or not audit.get("protected_phase6_hashes_pass"):
        raise RuntimeError("Data/integrity audit must pass before protocol freeze")
    result_names = [
        "phase7_development_cv.json", "phase7_selected_models.json", "phase7_external_evaluation.json",
        "phase7_incremental_value.json", "phase7_research_summary.md",
    ]
    existing = [name for name in result_names if (OUT / name).exists()]
    if existing:
        raise RuntimeError(f"Model/result artifacts already exist; refusing to freeze retrospectively: {existing}")
    files = [
        OUT / "phase7_preexternal_protocol.md",
        OUT / "phase7_protocol.json",
        OUT / "phase7_protocol_amendment_001.md",
        OUT / "phase7_protocol_amendment_002.md",
        OUT / "phase7_data_audit.json",
        OUT / "phase5_variable_parity.json",
        OUT / "asset_eligibility_protocol.json",
        OUT / "asset_eligibility_amendment_001.md",
        OUT / "candidate_source_manifest.json",
        ROOT / "data" / "processed" / "phase7" / "market_liquidity_panel_4h.csv.gz",
    ]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(missing)
    manifest = {
        "protocol_frozen": True,
        "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "phase7_protocol_sha256": sha(OUT / "phase7_protocol.json"),
        "phase7_protocol_markdown_sha256": sha(OUT / "phase7_preexternal_protocol.md"),
        "phase7_protocol_amendment_001_sha256": sha(OUT / "phase7_protocol_amendment_001.md"),
        "phase7_protocol_amendment_002_sha256": sha(OUT / "phase7_protocol_amendment_002.md"),
        "data_audit_sha256": sha(OUT / "phase7_data_audit.json"),
        "frozen_inputs_and_rules": {str(p.relative_to(ROOT)): sha(p) for p in files},
        "development_sample_only_for_selection": "2021-12-01T00:00:00Z through 2023-06-30T20:00:00Z feature intervals, with six-step labels fully inside the development window",
        "previously_seen_stage6_test_not_reused_for_selection": "2023-07-01 through 2024-03-03",
        "new_external_test": "2024-03-04T00:00:00Z through eligible 24-hour labels ending no later than 2026-09-30T20:00:00Z",
        "external_outcome_distributions_or_forecast_scores_read": False,
        "network_used_for_new_market_data": False,
    }
    (OUT / "phase7_freeze_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"protocol_frozen": True, "protocol_sha256": manifest["phase7_protocol_sha256"], "input_count": len(files)}, indent=2))


if __name__ == "__main__":
    main()
