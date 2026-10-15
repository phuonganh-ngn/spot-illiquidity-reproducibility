#!/usr/bin/env python3
"""Rebuild the two paper figures from archived CSV results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_interval(value: str) -> tuple[float, float]:
    low, high = value.strip("[]").split(",", 1)
    return float(low), float(high)


def category_figure(output: Path) -> None:
    records = rows(ROOT / "tables/supplement_02_category_sensitivity.csv")
    macro = {
        row["asset"]: row
        for row in records
        if row["category"] == "SIX_CATEGORY_MACRO_AVERAGE"
    }
    assets = ["BTC", "ETH"]
    original = [float(macro[a]["historical_relative_reduction_pct_reconstructed"]) for a in assets]
    sensitivity = [float(macro[a]["target_scaled_sensitivity_relative_reduction_pct"]) for a in assets]

    fig, ax = plt.subplots(figsize=(10, 6.1))
    x = np.arange(len(assets))
    width = 0.34
    groups = [
        ax.bar(x - width / 2, original, width, label="Original registered category result", color="#2878B5"),
        ax.bar(x + width / 2, sensitivity, width, label="Retrospective scaling sensitivity", color="#F28E2B"),
    ]
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(x, assets)
    ax.set_ylabel("Category-average relative MAE reduction (%)")
    ax.set_title("Original registered category results and scaling sensitivity")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)

    for group in groups:
        for bar in group:
            value = bar.get_height()
            offset = (0, 5) if value >= 0 else (0, -5)
            vertical = "bottom" if value >= 0 else "top"
            ax.annotate(
                f"{value:.2f}%",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=offset,
                textcoords="offset points",
                ha="center",
                va=vertical,
                fontsize=10,
            )

    fig.text(
        0.5,
        0.015,
        "Separate evidence statuses; not independent replications or identical trained models.",
        ha="center",
        va="bottom",
        fontsize=9.7,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output / "figure2_category_sensitivity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def selected_system_figure(output: Path) -> None:
    records = {row["asset"]: row for row in rows(ROOT / "tables/table_05_selected_systems.csv")}
    assets = ["BTC", "ETH"]
    means = [float(records[a]["delta_best_mae_s0_minus_s1"]) / 1e-6 for a in assets]
    intervals = [read_interval(records[a]["block42_ci95_nominal"]) for a in assets]
    lows = [lo / 1e-6 for lo, _ in intervals]
    highs = [hi / 1e-6 for _, hi in intervals]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    y = np.array([1, 0])
    colors = ["#2878B5", "#F28E2B"]
    for index in range(len(assets)):
        ax.errorbar(
            means[index],
            y[index],
            xerr=[[means[index] - lows[index]], [highs[index] - means[index]]],
            fmt="o",
            color=colors[index],
            ecolor=colors[index],
            capsize=5,
            markersize=8,
            linewidth=2,
        )

    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_yticks(y, assets)
    ax.set_xlabel(r"$\Delta_{best}$ (MAE S0 $-$ MAE S1), $10^{-6}$")
    ax.set_title("Selected-system MAE differences\nNominal 42-bar confidence intervals", pad=12)
    ax.grid(axis="x", alpha=0.22)
    ax.set_axisbelow(True)
    fig.text(
        0.5,
        0.015,
        "Zero lies within the BTC interval but outside the ETH interval. ETH changes both information set and selected model family.",
        ha="center",
        va="bottom",
        fontsize=9.3,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(output / "figure3_selected_system_delta.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "build/figures",
        help="Directory for newly rendered figures (default: build/figures).",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    category_figure(args.output_dir)
    selected_system_figure(args.output_dir)
    print(f"Figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
