#!/usr/bin/env python3
"""Check package checksums and source-to-bundle provenance."""

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


def verify_checksums() -> int:
    checksum_file = ROOT / "SHA256SUMS"
    checked = 0
    errors: list[str] = []
    listed_paths: set[str] = set()

    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative_path = line.split("  ", 1)
        listed_paths.add(relative_path)
        path = (ROOT / relative_path).resolve()
        if ROOT.resolve() not in path.parents:
            errors.append(f"path escapes package: {relative_path}")
            continue
        if not path.is_file():
            errors.append(f"missing file: {relative_path}")
        elif sha256(path) != expected:
            errors.append(f"checksum mismatch: {relative_path}")
        else:
            checked += 1

    actual_paths = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.name != "SHA256SUMS"
        and "__pycache__" not in path.parts
        and "build" not in path.parts
        and ".git" not in path.parts
    }
    for extra in sorted(actual_paths - listed_paths):
        errors.append(f"unlisted file: {extra}")
    for missing in sorted(listed_paths - actual_paths):
        errors.append(f"checksum entry has no file: {missing}")

    if errors:
        raise SystemExit("Bundle checksum errors:\n" + "\n".join(errors))
    print(f"Package checksums: {checked} files verified")
    return checked


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
    verify_checksums()
    verify_source_manifest()


if __name__ == "__main__":
    main()
