#!/usr/bin/env python3
"""Check source-to-bundle provenance for archived research artifacts."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_manifest() -> int:
    manifest = ROOT / "source_manifest.csv"
    checked = 0
    errors: list[str] = []

    with manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            path = ROOT / row["package_path"]
            if not path.is_file():
                errors.append(f"missing copied artifact: {row['package_path']}")
            elif path.stat().st_size != int(row["bytes"]):
                errors.append(f"size mismatch: {row['package_path']}")
            elif sha256(path) != row["sha256"]:
                errors.append(f"source hash mismatch: {row['package_path']}")
            else:
                checked += 1

    if errors:
        raise SystemExit("Source-manifest errors:\n" + "\n".join(errors))
    print(f"Source provenance: {checked} copied artifacts verified")
    return checked


def main() -> None:
    verify_source_manifest()


if __name__ == "__main__":
    main()
