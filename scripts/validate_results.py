#!/usr/bin/env python3
"""Check headline values against archived tables without refitting models."""

from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_rows(relative_path: str) -> list[dict[str, str]]:
    with (ROOT / relative_path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def close(actual: str | float, expected: float, tolerance: float = 1e-10) -> bool:
    return math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=tolerance)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")
    print(f"PASS: {message}")


def main() -> None:
    tasks = read_rows("evidence/neural_and_generalization_audit/phase6_all_tasks_summary.csv")
    passed = [row for row in tasks if row["frozen_task_rule_pass"].lower() == "true"]
    passed_ids = {row["task_id"] for row in passed}
    require(len(tasks) == 18, "the historical benchmark contains 18 registered tasks")
    require(len(passed) == 2, "the original registered decision remains 2 of 18")
    require(
        passed_ids == {"BTC__future_amihud__h6", "ETH__future_amihud__h6"},
        "the two passing tasks are BTC and ETH 24-hour Amihud",
    )

    categories = read_rows("tables/supplement_02_category_sensitivity.csv")
    macro = {row["asset"]: row for row in categories if row["category"] == "SIX_CATEGORY_MACRO_AVERAGE"}
    require(close(macro["BTC"]["historical_relative_reduction_pct_reconstructed"], 18.56477335524347),
            "BTC original registered category average matches the archived value")
    require(close(macro["ETH"]["historical_relative_reduction_pct_reconstructed"], 13.644812507260148),
            "ETH original registered category average matches the archived value")
    require(close(macro["BTC"]["target_scaled_sensitivity_relative_reduction_pct"], -0.9744155837036568),
            "BTC retrospective scaling sensitivity matches the archived value")
    require(close(macro["ETH"]["target_scaled_sensitivity_relative_reduction_pct"], 0.8080744892608575),
            "ETH retrospective scaling sensitivity matches the archived value")

    primary = {row["asset"]: row for row in read_rows("tables/table_05_selected_systems.csv")}
    require(primary["BTC"]["selected_s0_model"] == "HistGradientBoosting"
            and primary["BTC"]["selected_s1_model"] == "HistGradientBoosting",
            "BTC development-selected systems are both HistGradientBoosting")
    require(close(primary["BTC"]["relative_reduction_pct"], 1.991987786604532),
            "BTC retrospective selected-system difference matches the archived value")
    require(float(primary["BTC"]["block42_ci95_nominal"].strip("[]").split(",")[0]) < 0,
            "BTC nominal 42-bar interval includes zero")
    require(primary["ETH"]["selected_s0_model"] == "Ridge"
            and primary["ETH"]["selected_s1_model"] == "ExtraTrees",
            "ETH selected-system comparison changes model family")
    require(close(primary["ETH"]["relative_reduction_pct"], 5.1017942776422505),
            "ETH retrospective selected-system difference matches the archived value")
    require(float(primary["ETH"]["block42_ci95_nominal"].strip("[]").split(",")[0]) > 0,
            "ETH nominal 42-bar interval excludes zero")

    bybit = {row["asset"]: row for row in read_rows("tables/table_08_external_qml.csv")}
    require(bybit["BTC"]["n_holdout"] == "1315" and bybit["ETH"]["n_holdout"] == "1315",
            "Bybit holdout contains 1,315 scored origins for each asset")
    require(close(bybit["BTC"]["delta_i_holm_p"], 0.0579420579420579),
            "BTC Bybit incremental-information Holm p-value matches the record")
    require(close(bybit["ETH"]["delta_i_holm_p"], 0.0299700299700299),
            "ETH Bybit incremental-information Holm p-value matches the record")
    require(float(bybit["ETH"]["delta_i_mean"]) < 0,
            "ETH Bybit complete-forecast Delta_I is adverse to QML")

    print("Archived result checks passed. No model was trained and no new test was run.")


if __name__ == "__main__":
    main()
