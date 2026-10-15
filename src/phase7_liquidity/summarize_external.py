from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "phase7_liquidity_replication"
ASSETS = ["BTC", "ETH", "SOL", "BNB", "XRP"]
TARGETS = [
    "future_amihud_24h_mean",
    "future_kyle_abs_lambda_24h",
    "future_corwin_schultz_spread_24h",
]
SEGMENTS = [
    ("2024-03_to_2024-12", pd.Timestamp("2024-03-04T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    ("2025", pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    ("2026_partial", pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-10-01T00:00:00Z")),
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe_float(x):
    if pd.isna(x):
        return None
    return float(x)


def main() -> None:
    ext_path = OUT / "phase7_external_evaluation.json"
    pred_path = OUT / "phase7_external_predictions.csv.gz"
    freeze_path = OUT / "phase7_external_freeze.json"
    freeze = json.loads(freeze_path.read_text())
    for name, expected in freeze["files"].items():
        if sha(OUT / name) != expected:
            raise RuntimeError(f"Frozen external output changed: {name}")
    evaluation = json.loads(ext_path.read_text())
    if not evaluation.get("external_outcome_read_after_protocol_freeze"):
        raise RuntimeError("External evaluation provenance flags are invalid")
    if evaluation["protocol_sha256"] != freeze["protocol_sha256"]:
        raise RuntimeError("External evaluation protocol hash mismatch")

    aggregates = evaluation["equal_category_aggregate"]
    agg_index = {(x["asset"], x["target"], x["feature_pair"]): x for x in aggregates}
    paired = evaluation["paired_comparison_results"]
    family_index = {(x["asset"], x["target"], x["feature_pair"], x["family"]): x for x in paired}

    # Fixed calendar-segment criterion: equal weight over the five independent categories,
    # with MLP and GRU averaged inside the single neural category.
    pred = pd.read_csv(pred_path, usecols=["asset", "target", "feature_pair", "information_set", "family", "date", "actual_transformed", "prediction_transformed"], parse_dates=["date"])
    segment_rows = []
    for asset in ASSETS:
        for seg_name, seg_start, seg_end in SEGMENTS:
            p = pred[(pred.asset == asset) & (pred.target == "future_amihud_24h_mean") & (pred.feature_pair == "legacy") & (pred.date >= seg_start) & (pred.date < seg_end)]
            if p.empty:
                segment_rows.append({"asset": asset, "segment": seg_name, "n_dates": 0, "equal_category_delta": None, "spot_only_mae": None, "delta_over_spot_mae": None})
                continue
            s = p[p.information_set == "spot_only"].copy()
            q = p[p.information_set == "spot_plus_derivatives"].copy()
            key = ["asset", "target", "feature_pair", "family", "date"]
            m = s.merge(q, on=key, suffixes=("_spot", "_plus"), validate="one_to_one")
            m["loss_spot"] = (m.prediction_transformed_spot - m.actual_transformed_spot).abs()
            m["loss_plus"] = (m.prediction_transformed_plus - m.actual_transformed_plus).abs()
            m["delta"] = m.loss_plus - m.loss_spot
            m["category"] = m.family.map({"Ridge": "linear", "RBF_SVR": "kernel", "ExtraTrees": "bagged_trees", "HistGradientBoosting": "boosting", "MLP": "neural", "GRU": "neural"})
            by_family = m.groupby(["category", "family"], as_index=False).agg(delta=("delta", "mean"), spot_mae=("loss_spot", "mean"))
            by_category = by_family.groupby("category", as_index=False).agg(delta=("delta", "mean"), spot_mae=("spot_mae", "mean"))
            delta = float(by_category.delta.mean())
            spot_mae = float(by_category.spot_mae.mean())
            segment_rows.append({"asset": asset, "segment": seg_name, "n_dates": int(m.date.nunique()), "equal_category_delta": delta, "spot_only_mae": spot_mae, "delta_over_spot_mae": delta / spot_mae if spot_mae else None, "categories_improving": int((by_category.delta < 0).sum())})

    primary_segments = [x for x in segment_rows if x["asset"] in ("BTC", "ETH")]
    segment_ok = True
    for asset in ("BTC", "ETH"):
        ar = [x for x in primary_segments if x["asset"] == asset]
        improved = sum(x["equal_category_delta"] is not None and x["equal_category_delta"] < 0 for x in ar)
        severe = any(x["delta_over_spot_mae"] is not None and x["delta_over_spot_mae"] > 0.05 for x in ar)
        segment_ok &= improved >= 2 and not severe

    amihud_rows = [agg_index[(a, "future_amihud_24h_mean", "legacy")] for a in ASSETS]
    btc = agg_index[("BTC", "future_amihud_24h_mean", "legacy")]
    eth = agg_index[("ETH", "future_amihud_24h_mean", "legacy")]
    amihud_conditions = {
        "btc_and_eth_equal_category_delta_below_zero": btc["equal_category_mean_delta"] < 0 and eth["equal_category_mean_delta"] < 0,
        "btc_and_eth_bootstrap_ci_upper_below_zero": btc["bootstrap_ci95"][1] < 0 and eth["bootstrap_ci95"][1] < 0,
        "at_least_4_of_5_assets_improve": sum(x["equal_category_mean_delta"] < 0 for x in amihud_rows) >= 4,
        "at_least_3_of_5_categories_improve_for_btc_and_eth": btc["categories_improving"] >= 3 and eth["categories_improving"] >= 3,
        "at_least_2_of_3_segments_improve_without_material_deterioration": bool(segment_ok),
    }
    amihud_pass = all(amihud_conditions.values())

    proxy_conditions = {}
    for target in ("future_kyle_abs_lambda_24h", "future_corwin_schultz_spread_24h"):
        b = agg_index[("BTC", target, "legacy")]
        e = agg_index[("ETH", target, "legacy")]
        proxy_conditions[target] = {
            "btc_and_eth_delta_below_zero": b["equal_category_mean_delta"] < 0 and e["equal_category_mean_delta"] < 0,
            "btc_and_eth_bootstrap_ci_upper_below_zero": b["bootstrap_ci95"][1] < 0 and e["bootstrap_ci95"][1] < 0,
            "at_least_3_of_5_categories_improve_for_btc_and_eth": b["categories_improving"] >= 3 and e["categories_improving"] >= 3,
        }
    distinct_proxy_pass = any(all(c.values()) for c in proxy_conditions.values())
    if amihud_pass and distinct_proxy_pass:
        classification = "EXTERNAL_REPLICATION_AND_CROSS_MEASURE_GENERALIZATION"
    elif amihud_pass:
        classification = "AM IHUD_SPECIFIC_PREDICTIVE_VALUE".replace(" ", "")
    else:
        classification = "NOT_EXTERNALLY_REPLICATED_UNDER_FROZEN_CRITERIA"

    family_summary = []
    for asset in ("BTC", "ETH"):
        rows = [family_index[(asset, "future_amihud_24h_mean", "legacy", f)] for f in ["Ridge", "RBF_SVR", "ExtraTrees", "HistGradientBoosting", "MLP", "GRU"]]
        family_summary.append({"asset": asset, "families_improving": sum(x["delta_plus_minus_spot"] < 0 for x in rows), "families_total": 6, "families_holm_significant_improving": sum(x["delta_plus_minus_spot"] < 0 and x["holm_p_within_asset_target_feature_pair"] < 0.05 for x in rows), "per_family": [{"family": x["family"], "delta": x["delta_plus_minus_spot"], "bootstrap_ci95": x["bootstrap_ci95"], "holm_p": x["holm_p_within_asset_target_feature_pair"]} for x in rows]})

    stress_summary = []
    stress_rows = [x for x in evaluation["stress_warning"] if x["target"] == "future_amihud_24h_mean" and x["feature_pair"] == "legacy"]
    for asset in ASSETS:
        ar = [x for x in stress_rows if x["asset"] == asset]
        result = {"asset": asset}
        for info in ("spot_only", "spot_plus_derivatives"):
            vals = [x for x in ar if x["information_set"] == info]
            metrics = {}
            for key in ("precision", "recall", "f1", "lift", "event_prevalence"):
                finite = [x[key] for x in vals if x[key] is not None and np.isfinite(x[key])]
                metrics[key] = float(np.mean(finite)) if finite else None
            result[info] = metrics
        stress_summary.append(result)

    counts = pred.groupby(["asset", "target", "feature_pair", "information_set", "family"]).agg(n=("date", "nunique"), first=("date", "min"), last=("date", "max")).reset_index()
    sample_summary = {
        "start_date": str(counts["first"].min().date()),
        "end_date": str(counts["last"].max().date()),
        "unique_prediction_rows": int(len(pred)),
        "per_asset_target_pair_date_counts": [
            {"asset": r.asset, "target": r.target, "feature_pair": r.feature_pair, "min_model_dates": int(r.min_n), "max_model_dates": int(r.max_n), "first_model_date": str(r.first.date()), "last_model_date": str(r.last.date())}
            for r in pred.groupby(["asset", "target", "feature_pair"]).agg(min_n=("date", "nunique"), max_n=("date", "nunique"), first=("date", "min"), last=("date", "max")).reset_index().itertuples(index=False)
        ],
    }
    decision = {
        "protocol_sha256": freeze["protocol_sha256"],
        "external_freeze_sha256": sha(freeze_path),
        "external_predictions_sha256": sha(pred_path),
        "external_evaluation_sha256": sha(ext_path),
        "classification": classification,
        "amihud_replication": {"passed": amihud_pass, "conditions": amihud_conditions, "legacy_asset_results": [{"asset": x["asset"], "delta": x["equal_category_mean_delta"], "bootstrap_ci95": x["bootstrap_ci95"], "categories_improving": x["categories_improving"]} for x in amihud_rows], "legacy_segments": segment_rows},
        "distinct_liquidity_proxy_generalization": {"passed": distinct_proxy_pass, "proxy_conditions": proxy_conditions},
        "family_sensitivity_amihud_legacy_btc_eth": family_summary,
        "stress_warning_amihud_legacy_equal_category_means": stress_summary,
        "sample": sample_summary,
        "negative_control_status": "LIMITED_DIAGNOSTIC_ONLY; outputs contain fixed-Ridge raw MAE for impossible-time features but no paired spot-only incremental-loss benchmark, so they cannot establish whether future derivatives add predictive value.",
        "measurement_note": "Kyle absolute signed-flow price-impact slope and Corwin-Schultz range-based spread are distinct proxies. Neither reproduces Amihud incremental value in both BTC and ETH under the frozen legacy comparison.",
        "interpretation": "Forecasting/predictive association only; no causal or structural interpretation.",
    }
    decision_path = OUT / "phase7_replication_decision.json"
    decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True, allow_nan=False) + "\n")

    btc_a, eth_a = btc, eth
    seg_lines = []
    for x in segment_rows:
        if x["asset"] in ("BTC", "ETH"):
            seg_lines.append(f"{x['asset']} {x['segment']}: delta {x['equal_category_delta']:.6g}; Spot-only MAE {x['spot_only_mae']:.6g}; relative change {100*x['delta_over_spot_mae']:.2f}%.")
    segment_text = " ".join(seg_lines)
    text = f"""# Phase 7 external liquidity replication

The frozen development protocol was applied to cached, same-venue Binance data. The new external evaluation uses feature dates from 2024-03-04 through 2026-09-29 20:00 UTC, with future 24-hour labels ending by 2026-09-30 20:00 UTC. The model-specific complete-case samples begin on 2024-03-05 04:00 UTC because of the frozen lagged-feature and availability rules; there are 5,480–5,526 scored feature dates per asset/target comparison. The output contains {sample_summary['unique_prediction_rows']:,} prediction rows across five assets, three distinct liquidity targets, legacy/enhanced comparisons and six learned model families.

The confirmatory Amihud comparison does not meet the frozen replication criteria. In the legacy feature specification, the equal-category paired transformed-MAE difference (Spot+Derivatives minus Spot-only; negative favors added derivatives) is {btc_a['equal_category_mean_delta']:.6g} for BTC (42-interval block-bootstrap 95% CI {btc_a['bootstrap_ci95'][0]:.6g} to {btc_a['bootstrap_ci95'][1]:.6g}) and {eth_a['equal_category_mean_delta']:.6g} for ETH (CI {eth_a['bootstrap_ci95'][0]:.6g} to {eth_a['bootstrap_ci95'][1]:.6g}). Only {sum(x['equal_category_mean_delta']<0 for x in amihud_rows)} of five eligible assets improve; BTC has {btc_a['categories_improving']}/5 and ETH {eth_a['categories_improving']}/5 model categories improving. The multi-condition primary rule therefore fails even though BTC's aggregate interval is below zero.

The result is model-sensitive. For Amihud in BTC, the six individual model families split {family_summary[0]['families_improving']}/6 in favor of added derivatives and only {family_summary[0]['families_holm_significant_improving']}/6 both improve and survive the within-comparison Holm adjustment. ETH splits {family_summary[1]['families_improving']}/6, with {family_summary[1]['families_holm_significant_improving']}/6 Holm-surviving improvements. Large neural-family gains coexist with deterioration in several other families, so a best-model result would overstate robustness.

The two distinct liquidity proxies do not generalize the BTC/ETH result. For Kyle price impact, BTC's legacy equal-category delta is {agg_index[('BTC','future_kyle_abs_lambda_24h','legacy')]['equal_category_mean_delta']:.6g} and ETH's is {agg_index[('ETH','future_kyle_abs_lambda_24h','legacy')]['equal_category_mean_delta']:.6g}. For Corwin-Schultz, the corresponding deltas are {agg_index[('BTC','future_corwin_schultz_spread_24h','legacy')]['equal_category_mean_delta']:.6g} and {agg_index[('ETH','future_corwin_schultz_spread_24h','legacy')]['equal_category_mean_delta']:.6g}. Signs and uncertainty do not satisfy the frozen two-asset and category-consistency rule for either proxy. Thus the evidence is not general across liquidity measurement.

Fixed external subperiods show a common but not fully stable pattern. {segment_text} Both BTC and ETH improve in two of three segments, and the 2025 deterioration is below 1% of segment Spot-only MAE, so the frozen segment condition passes. The overall Amihud replication still fails the other required conditions.

The enhanced Amihud comparison is also mixed: its equal-category delta is {agg_index[('BTC','future_amihud_24h_mean','enhanced')]['equal_category_mean_delta']:.6g} for BTC with {agg_index[('BTC','future_amihud_24h_mean','enhanced')]['categories_improving']}/5 categories improving, and {agg_index[('ETH','future_amihud_24h_mean','enhanced')]['equal_category_mean_delta']:.6g} for ETH with {agg_index[('ETH','future_amihud_24h_mean','enhanced')]['categories_improving']}/5 categories improving. This calendar/shock/interaction-augmented specification does not replace the legacy replication criterion and does not resolve its model sensitivity.

Amihud stress-warning metrics are mixed across assets. Equal-category average F1 changes from {stress_summary[0]['spot_only']['f1']:.3f} to {stress_summary[0]['spot_plus_derivatives']['f1']:.3f} for BTC and from {stress_summary[1]['spot_only']['f1']:.3f} to {stress_summary[1]['spot_plus_derivatives']['f1']:.3f} for ETH, while BNB moves from {stress_summary[3]['spot_only']['f1']:.3f} to {stress_summary[3]['spot_plus_derivatives']['f1']:.3f}. For SOL, no external observations crossed the development q90 event threshold, so its precision/recall/lift are not informative. These are fixed-threshold descriptive warnings, not evidence of causal transmission.

The impossible-time diagnostics ran with fixed Ridge settings, but the stored diagnostics contain their raw MAE only and no matched Spot-only benchmark. They are therefore insufficient to establish whether post-outcome derivative information adds predictive value. This limitation is explicitly retained rather than calling the negative controls a pass.

The defensible conclusion is that the Stage-6 BTC/ETH Amihud forecasting result is **not externally replicated under the frozen multi-condition rule**, and the apparent added predictive value is sensitive to asset, model family, feature specification and liquidity proxy. This is predictive evidence only; it does not identify a causal effect. The three liquidity measures capture different concepts: Amihud price impact per turnover, Kyle signed-flow price impact and a Corwin-Schultz range-based spread proxy.

The first external scoring attempt completed all forecast predictions but stopped in its final diagnostic because `StandardScaler` was missing from the negative-control routine. Its incomplete prediction file was archived and not inspected for performance. The fixed code was rerun without changing any protocol/model choice; the final prediction artifact has the same SHA-256 as that incomplete attempt, demonstrating byte-identical forecast reproduction. Thus the frozen external forecast pipeline was executed twice for engineering recovery, with no interim result inspection or result-guided choice; the first run produced no complete report. The evaluation summary and negative-control outputs were then completed successfully.
"""
    (OUT / "phase7_research_summary.md").write_text(text)
    print(json.dumps({"decision": classification, "amihud_pass": amihud_pass, "distinct_proxy_pass": distinct_proxy_pass, "amihud_conditions": amihud_conditions, "sample_start": sample_summary["start_date"], "sample_end": sample_summary["end_date"], "prediction_rows": sample_summary["unique_prediction_rows"], "summary": "phase7_research_summary.md"}, indent=2))


if __name__ == "__main__":
    main()
