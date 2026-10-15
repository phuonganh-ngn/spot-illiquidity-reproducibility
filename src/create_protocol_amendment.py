#!/usr/bin/env python3
"""Create and validate the pre-estimation Stage 2A protocol amendment.

This script reads only the frozen protocol and hypothesis registry, never data
values or estimation outputs. It does not fit models.
"""
from __future__ import annotations
import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
ORIGINAL_FILES = [
    "modeling_protocol.md",
    "hypothesis_registry.csv",
    "primary_variable_registry.csv",
    "multiple_testing_families.csv",
    "crossfit_plan.csv",
    "timing_rules.md",
    "directional_classification_rules.md",
    "stage3_execution_plan.md",
    "protocol_manifest.csv",
]
NEW_FILES = [
    "protocol_amendment_001.md",
    "primary_hypothesis_registry_v2.csv",
    "multiple_testing_families_v2.csv",
]
FORBIDDEN_RESULT_PREFIXES = (
    "table_stage3", "stage3_crossfit", "stage3_model", "stage3_validation",
    "stage3_research", "fig_channel", "fig_direction",
    "directional_evidence", "channel_classification",
)
CHANNELS = {
    "C1": {"name": "POSITIONING / RISK", "pair": "FUNDING<->RV", "forward_exposure": "funding_mean", "forward_outcome": "spot_realized_volatility", "reverse_exposure": "spot_realized_volatility", "reverse_outcome": "funding_mean"},
    "C2": {"name": "LEVERAGE / RISK", "pair": "OI<->RV", "forward_exposure": "oi_log_change_1d", "forward_outcome": "spot_realized_volatility", "reverse_exposure": "spot_realized_volatility", "reverse_outcome": "oi_log_change"},
    "C3": {"name": "LEVERAGE / TAIL RISK", "pair": "OI<->DOWNSIDE", "forward_exposure": "oi_log_change_1d", "forward_outcome": "spot_downside_semivariance", "reverse_exposure": "spot_downside_semivariance", "reverse_outcome": "oi_log_change"},
    "C4": {"name": "ORDER FLOW / LIQUIDITY", "pair": "FLOW<->AMIHUD", "forward_exposure": "futures_taker_imbalance", "forward_outcome": "spot_amihud", "reverse_exposure": "spot_amihud", "reverse_outcome": "futures_taker_imbalance"},
}
HORIZONS = {"1", "3", "7"}
ASSETS = {"BTC", "ETH"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def verify_original_hashes():
    manifest_path = OUT / "protocol_manifest.csv"
    rows = load_csv(manifest_path)
    if not rows:
        raise RuntimeError("Original protocol manifest is empty")
    checked = 0
    for row in rows:
        rel = row.get("file", "")
        expected = row.get("SHA256", "")
        path = ROOT / rel
        if not path.exists() or sha256(path) != expected:
            raise RuntimeError(f"Original protocol hash mismatch: {rel}")
        checked += 1
    # The requested immutable list must all be represented by the original manifest.
    represented = {row["file"] for row in rows}
    missing = {f"outputs/{name}" for name in ORIGINAL_FILES[:-1]} - represented
    if missing:
        raise RuntimeError(f"Original manifest omits required frozen files: {sorted(missing)}")
    return checked, rows


def no_stage3_results_preexisting():
    found = [p.name for p in OUT.iterdir() if p.is_file() and p.name.startswith(FORBIDDEN_RESULT_PREFIXES)]
    if found:
        raise RuntimeError(f"Stage 3 result artefacts already exist; hard stop: {found}")
    return True


def channel_for(row):
    if row.get("asset") not in ASSETS or row.get("direction") not in {"FWD", "REV"}:
        return None
    if row.get("horizon", "").rstrip(".0") not in HORIZONS:
        return None
    for cid, c in CHANNELS.items():
        if row.get("pair") == c["pair"]:
            exp = c["forward_exposure"] if row["direction"] == "FWD" else c["reverse_exposure"]
            out = c["forward_outcome"] if row["direction"] == "FWD" else c["reverse_outcome"]
            if row.get("exposure") == exp and row.get("outcome", "").startswith(out):
                return cid
    return None


def main():
    original_count, original_manifest_rows = verify_original_hashes()
    no_stage3_results_preexisting()
    original = load_csv(OUT / "hypothesis_registry.csv")
    old_primary = [r for r in original if r.get("kind") == "DIRECTIONAL_DML" and r.get("priority") == "PRIMARY"]
    if len(old_primary) != 108:
        raise RuntimeError(f"Expected 108 original primary directional tests; found {len(old_primary)}")

    # Freeze amendment language before creating any estimates or loading any data.
    amendment = """# Protocol Amendment 001 — Confirmatory Universe Narrowing

**Stage:** 2A, before estimation  
**Status:** frozen before model estimation

The original Stage-2 protocol registered 108 primary directional tests. Before any estimation, the confirmatory primary universe is narrowed to 48 theory-matched directional tests across four predefined economic channels. All remaining originally registered combinations are retained and reported as secondary/exploratory analyses.

No data, model result, coefficient, p-value, correlation, forecast performance, or statistical significance was inspected or used to define this amendment.

No sample dates, estimators, timing rules, horizons, variables, inference rules, cross-fitting rules, falsification rules or robustness requirements are changed.

## Confirmatory channels

Each channel includes BTC and ETH, forward and reverse directions, and primary horizons h = 1, 3, 7. Each therefore contributes 2 × 2 × 3 = 12 tests; four channels contribute 48.

| ID | Channel | Forward | Reverse |
|---|---|---|---|
| C1 | Positioning / risk | Funding mean → future spot realized volatility | Spot realized volatility → future funding |
| C2 | Leverage / risk | OI log change → future spot realized volatility | Spot realized volatility → future OI log change |
| C3 | Leverage / tail risk | OI log change → future spot downside semivariance | Spot downside semivariance → future OI log change |
| C4 | Order flow / liquidity | Futures taker imbalance → future spot Amihud illiquidity | Spot Amihud illiquidity → future futures taker imbalance |

The primary status applies only to these 48 exact asset × direction × horizon rows. All other originally registered primary directional combinations remain in the analysis registry as `SECONDARY`; they may be reported as secondary/exploratory and cannot upgrade or replace a primary channel. The existing Stage-2 protocol, registries, timing, sample, estimators, controls, cross-fitting, inference, falsification and robustness rules remain in force. Horizons 14 and 30 remain extended/persistence analyses only and cannot rescue a failed primary direction.

## Multiplicity

The 48 tests are assigned to 16 predeclared Holm families: channel × asset × direction. Each family contains exactly the h1, h3, h7 tests. No global 48-test Holm family is used. Placebos, learner/timing robustness, Granger, Transfer Entropy and secondary analyses retain separate reporting families; this amendment does not alter their original rules.

## Freeze validation

The amendment generator verified the original Stage-2 manifest hashes before writing these files, found no pre-existing Stage-3 estimation-result artefacts by filename, and verified the test counts programmatically. No model was run. The protocol is frozen as V2 only when `protocol_manifest_v2.csv` records matching hashes for all original Stage-2 artifacts and this amendment and both V2 registries.
"""
    (OUT / "protocol_amendment_001.md").write_text(amendment, encoding="utf-8")

    # V2 registry is the complete set of the original 108 registered primary directional tests.
    v2_rows = []
    for row in old_primary:
        cid = channel_for(row)
        horizon = row["horizon"].rstrip(".0")
        if cid:
            channel = CHANNELS[cid]
            direction = "FORWARD" if row["direction"] == "FWD" else "REVERSE"
            family = f"{cid}_{row['asset']}_{direction}_{row['pair'].replace('<->', '_').replace('FUNDING','FUNDING').replace('FLOW','FLOW')}"
            v2_rows.append({
                "test_id": row["test_id"], "channel_id": cid, "channel_name": channel["name"],
                "asset": row["asset"], "direction": direction, "exposure": row["exposure"],
                "outcome": row["outcome"], "horizon": horizon,
                "status": "PRIMARY_CONFIRMATORY", "holm_family": family, "confirmatory": "True",
            })
        else:
            v2_rows.append({
                "test_id": row["test_id"], "channel_id": "", "channel_name": "",
                "asset": row["asset"], "direction": "FORWARD" if row["direction"] == "FWD" else "REVERSE",
                "exposure": row["exposure"], "outcome": row["outcome"], "horizon": horizon,
                "status": "SECONDARY", "holm_family": "", "confirmatory": "False",
            })
    fieldnames = ["test_id", "channel_id", "channel_name", "asset", "direction", "exposure", "outcome", "horizon", "status", "holm_family", "confirmatory"]
    with (OUT / "primary_hypothesis_registry_v2.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(v2_rows)

    primary = [r for r in v2_rows if r["confirmatory"] == "True"]
    if len(primary) != 48:
        raise RuntimeError(f"Confirmatory test count must be 48; found {len(primary)}")
    by_channel = {}
    for r in primary: by_channel[r["channel_id"]] = by_channel.get(r["channel_id"], 0) + 1
    if by_channel != {cid: 12 for cid in CHANNELS}:
        raise RuntimeError(f"Channel counts wrong: {by_channel}")
    if len(v2_rows) != 108 or len({r["test_id"] for r in v2_rows}) != 108:
        raise RuntimeError("V2 registry must retain all 108 unique original tests")
    if sum(r["status"] == "SECONDARY" for r in v2_rows) != 60:
        raise RuntimeError("Expected 60 original tests labeled SECONDARY")

    family_rows = []
    for row in primary:
        family_rows.append({
            "family_id": row["holm_family"], "channel_id": row["channel_id"],
            "channel_name": row["channel_name"], "asset": row["asset"],
            "direction": row["direction"], "test_id": row["test_id"],
            "horizon": row["horizon"], "correction": "Holm", "alpha": "0.05",
            "family_size": "3", "confirmatory": "True", "status": "PREDEFINED_NOT_ESTIMATED",
        })
    with (OUT / "multiple_testing_families_v2.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["family_id", "channel_id", "channel_name", "asset", "direction", "test_id", "horizon", "correction", "alpha", "family_size", "confirmatory", "status"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(family_rows)
    family_counts = {}
    for r in family_rows: family_counts[r["family_id"]] = family_counts.get(r["family_id"], 0) + 1
    if len(family_counts) != 16 or set(family_counts.values()) != {3}:
        raise RuntimeError(f"Expected 16 Holm families of 3 tests; got {family_counts}")

    # Reverify protected hashes, then hash all original Stage-2 artifacts + amendment/registries.
    verify_original_hashes()
    freeze_time = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest_files = [OUT / f for f in ORIGINAL_FILES] + [OUT / f for f in NEW_FILES]
    manifest_path = OUT / "protocol_manifest_v2.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["file", "SHA256", "creation_timestamp", "stage", "status", "protocol_frozen_v2"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for path in manifest_files:
            w.writerow({"file": path.relative_to(ROOT).as_posix(), "SHA256": sha256(path), "creation_timestamp": freeze_time, "stage": "Stage 2A protocol freeze", "status": "IMMUTABLE_FROZEN", "protocol_frozen_v2": "YES"})

    # Final gate: hashes, counts, pre-estimation status and all source protocols intact.
    verify_original_hashes()
    v2_manifest_rows = load_csv(manifest_path)
    for row in v2_manifest_rows:
        if sha256(ROOT / row["file"]) != row["SHA256"]:
            raise RuntimeError(f"V2 manifest hash mismatch: {row['file']}")
    no_stage3_results_preexisting()
    print("STAGE2A_GATE=PASS")
    print(f"ORIGINAL_PROTOCOL_FILES_HASHED={original_count}")
    print("ORIGINAL_PRIMARY_DIRECTIONAL_TESTS=108")
    print("CONFIRMATORY_PRIMARY_TESTS=48")
    print("SECONDARY_RETAINED_TESTS=60")
    print("HOLM_FAMILIES=16; TESTS_PER_FAMILY=3")
    print("PREEXISTING_STAGE3_RESULT_FILES=0")
    print("ESTIMATION_PERFORMED=NO")
    print("PROTOCOL_FROZEN_V2=YES")
    for r in v2_manifest_rows: print(f"HASH {r['file']} {r['SHA256']}")

if __name__ == "__main__":
    main()
