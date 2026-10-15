from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "phase9_external_qml_replication"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def write_json(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    raw_manifest = read_json("raw_manifest.json")
    raw_failures = []
    for rec in raw_manifest["files"]:
        path = ROOT / rec["path"]
        if not path.is_file() or path.stat().st_size != rec["size_bytes"] or sha256(path) != rec["sha256"]:
            raw_failures.append(rec["path"])

    prior_snapshot = read_json("prior_artifact_hashes_before.json")
    prior_failures = []
    for rec in prior_snapshot["artifacts"]:
        path = ROOT / rec["file"]
        if not path.is_file() or path.stat().st_size != rec["size_bytes"] or sha256(path) != rec["sha256"]:
            prior_failures.append(rec["file"])

    frozen = read_json("frozen_artifact_hashes.json")
    frozen_failures = []
    for rel, expected in frozen["files"].items():
        path = OUT / rel
        if not path.is_file() or sha256(path) != expected:
            frozen_failures.append(rel)
    code = read_json("execution_code_manifest.json")
    code_failures = []
    for rel, expected in code["files"].items():
        path = ROOT / rel
        if not path.is_file() or sha256(path) != expected:
            code_failures.append(rel)

    split = read_json("temporal_split.json")
    panel_hash = sha256(OUT / "external_panel.csv.gz")
    protocol_hash = sha256(OUT / "phase9_protocol.json")
    primary_hashes = read_json("forecast_prediction_hashes.json")
    offline = read_json("offline_rerun_payload.json")
    offline_status = {
        "network_disabled": offline.get("network_disabled") is True,
        "prediction_payload_matches_primary": offline.get("prediction_payload_matches_primary") is True,
        "prediction_hash_matches_primary": offline.get("canonical_prediction_payload_sha256") == primary_hashes.get("canonical_prediction_payload_sha256"),
        "inference_payload_matches_primary": offline.get("inference_payload_matches_primary") is True,
        "inference_hash_matches_primary": offline.get("canonical_inference_payload_sha256") == primary_hashes.get("canonical_inference_payload_sha256"),
        "protocol_hash_matches": offline.get("protocol_sha256") == protocol_hash,
        "panel_hash_matches": offline.get("panel_sha256") == panel_hash,
    }
    status = "PASS" if (not raw_failures and not prior_failures and not frozen_failures and
                         not code_failures and all(offline_status.values())) else "FAIL"
    result = {
        "status": status,
        "phase": "Phase 9 external QML replication",
        "verified_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "raw_sources": {"manifest_file_count": raw_manifest["file_count"], "verified_file_count": len(raw_manifest["files"]),
                        "sha256_and_size_match": not raw_failures, "failures": raw_failures[:25]},
        "previous_study_artifacts": {"verified_file_count": len(prior_snapshot["artifacts"]),
                                     "unchanged": not prior_failures, "failures": prior_failures[:25]},
        "frozen_phase9_artifacts": {"verified_file_count": len(frozen["files"]),
                                    "unchanged": not frozen_failures, "failures": frozen_failures[:25]},
        "execution_code": {"verified_file_count": len(code["files"]), "unchanged": not code_failures,
                           "failures": code_failures},
        "protocol_sha256": protocol_hash,
        "panel_sha256": panel_hash,
        "offline_rerun": offline_status | {
            "primary_prediction_hash": primary_hashes.get("canonical_prediction_payload_sha256"),
            "offline_prediction_hash": offline.get("canonical_prediction_payload_sha256"),
            "network_guard": "sitecustomize blocks socket.create_connection, socket.connect and socket.connect_ex",
        },
        "split": {k: split[k] for k in ["common_start_utc", "common_end_utc", "training_rows_used_after_purge",
                                          "holdout_start_utc", "holdout_end_utc_inclusive", "holdout_rows"]},
    }
    write_json("reproducibility.json", result)
    if status != "PASS":
        raise RuntimeError("Phase 9 reproducibility validation failed; see reproducibility.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
