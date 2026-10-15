from __future__ import annotations

import gzip
import hashlib
import json
import math
import platform
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "phase8_liquidity_generalization"
PANEL = ROOT / "data" / "processed" / "phase7" / "market_liquidity_panel_4h.csv.gz"
PRED = ROOT / "outputs" / "phase7_liquidity_replication" / "phase7_external_predictions.csv.gz"
DEV_LAST = pd.Timestamp("2023-06-30T20:00:00Z")
DEV_END_EXCL = pd.Timestamp("2023-07-01T00:00:00Z")
EXT_START = pd.Timestamp("2024-03-04T00:00:00Z")
EXT_LAST = pd.Timestamp("2026-09-29T20:00:00Z")
SAMPLE_END_EXCL = pd.Timestamp("2026-10-01T00:00:00Z")
TARGET = "future_amihud_24h_mean"
TARGETS = [TARGET, "future_kyle_abs_lambda_24h", "future_corwin_schultz_spread_24h"]
ASSETS = ["BTC", "ETH", "SOL", "BNB", "XRP"]
FAMILIES = ["Ridge", "RBF_SVR", "ExtraTrees", "HistGradientBoosting", "MLP", "GRU"]
CATEGORY = {
    "Ridge": "linear",
    "RBF_SVR": "kernel",
    "ExtraTrees": "bagged_trees",
    "HistGradientBoosting": "boosting",
    "MLP": "neural",
    "GRU": "neural",
}
SPOT = ["spot_return", "spot_realized_volatility", "spot_downside_semivariance", "spot_amihud", "spot_taker_imbalance", "spot_quote_volume_growth"]
DERIV = ["funding_rate_state", "oi_log_change", "futures_taker_imbalance", "basis_mean"]
INFO = {
    "legacy_spot_only": "spot_only",
    "legacy_spot_plus_derivatives": "spot_plus_derivatives",
    "enhanced_spot_only": "spot_only",
    "enhanced_spot_plus_derivatives": "spot_plus_derivatives",
}
SEED = 724051
N_BOOT = 2000


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    return x


def dump(name: str, obj):
    (OUT / name).write_text(json.dumps(clean(obj), indent=2, sort_keys=True, allow_nan=False) + "\n")


def tree_fingerprint(records: dict[str, str]) -> str:
    body = "".join(f"{p}\0{h}\n" for p, h in sorted(records.items()))
    return hashlib.sha256(body.encode()).hexdigest()


def verify_hashes():
    freeze = json.loads((OUT / "freeze_manifest.json").read_text())
    for rel, expected in freeze["inputs"].items():
        path = ROOT / rel
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError(f"Frozen Phase-8 input hash mismatch: {rel}")
    protocol_hashes = {
        "outputs/phase8_liquidity_generalization/protocol.md": sha(OUT / "protocol.md"),
        "outputs/phase8_liquidity_generalization/protocol.json": sha(OUT / "protocol.json"),
    }
    if protocol_hashes != freeze["protocol_files"]:
        raise RuntimeError("Phase-8 protocol changed after its freeze")
    # Phase-7 frozen forecast outputs are checked independently below.
    p7out = ROOT / "outputs" / "phase7_liquidity_replication"
    p7freeze = json.loads((p7out / "phase7_external_freeze.json").read_text())
    for rel, expected in p7freeze["files"].items():
        if sha(p7out / rel) != expected:
            raise RuntimeError(f"Phase-7 external frozen output changed: {rel}")
    if sha(PANEL) != freeze["inputs"]["data/processed/phase7/market_liquidity_panel_4h.csv.gz"]:
        raise RuntimeError("Phase-7 panel hash mismatch")
    return freeze, p7freeze


def protected_file_snapshot() -> dict[str, str]:
    audit = json.loads((ROOT / "outputs/phase7_liquidity_replication/phase7_data_audit.json").read_text())
    checks = audit["protected_phase6_artifact_checks"]
    out = {}
    for rel, rec in checks.items():
        path = ROOT / rel
        if not path.is_file():
            raise RuntimeError(f"Protected prior-stage input missing: {rel}")
        actual = sha(path)
        expected = rec.get("expected_sha256")
        if expected and actual != expected:
            raise RuntimeError(f"Protected prior-stage hash mismatch: {rel}")
        out[rel] = actual
    return out


def verify_candidate_archives():
    base = ROOT / "outputs/phase7_liquidity_replication/candidate_archive_integrity.json"
    integrity = json.loads(base.read_text())
    records = integrity["files"]
    checked = 0
    for row in records:
        p = ROOT / row["path"]
        if not p.is_file() or sha(p) != row["sha256_expected"]:
            raise RuntimeError(f"Cached Phase-7 raw archive changed: {row['path']}")
        checked += 1
    if checked != 6195:
        raise RuntimeError(f"Expected 6195 cached archives, verified {checked}")
    return {"archive_count": checked, "sha256_verified": checked, "manifest_sha256": sha(base), "pass": True}


def moving_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    if n <= block:
        raise ValueError(f"series length {n} must exceed block length {block}")
    n_blocks = (n + block - 1) // block
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    return (starts[:, None] + np.arange(block)[None, :]).reshape(-1)[:n]


def bootstrap_mean(x: np.ndarray, lengths=(42, 84), seed=SEED):
    x = np.asarray(x, float)
    n = len(x)
    if n <= max(lengths) or not np.isfinite(x).all():
        raise ValueError("paired bootstrap requires a finite series longer than every block")
    out = {}
    for j, block in enumerate(lengths):
        rng = np.random.default_rng(seed + j * 1009)
        vals = np.empty(N_BOOT, float)
        for b in range(N_BOOT):
            ids = moving_block_indices(n, block, rng)
            vals[b] = x[ids].mean()
        out[str(block)] = {
            "block_length_4h": block,
            "replications": N_BOOT,
            "ci95_percentile": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
            "probability_delta_below_zero": float((vals < 0).mean()),
            "probability_delta_above_zero": float((vals > 0).mean()),
            "bootstrap_mean": float(vals.mean()),
        }
    return out


def matched_prediction_metrics(pred: pd.DataFrame):
    """Recreate paired transformed MAE deltas from Phase-7 saved predictions."""
    rows = []
    by_asset_target = []
    for asset in ASSETS:
        for target in TARGETS:
            for pair, left, right in [
                ("legacy", "spot_only", "spot_plus_derivatives"),
                ("enhanced", "spot_only", "spot_plus_derivatives"),
            ]:
                x = pred[(pred.asset == asset) & (pred.target == target) & (pred.feature_pair == pair)]
                if x.empty:
                    raise RuntimeError(f"Missing prediction rows: {asset}/{target}/{pair}")
                cols = ["date", "family", "actual_transformed", "actual_raw", "prediction_transformed", "prediction_raw", "alarm_threshold_raw_dev_oof_q90"]
                s = x[x.information_set == left][cols].rename(columns={c: f"{c}_s0" for c in cols if c not in ("date", "family")})
                d = x[x.information_set == right][cols].rename(columns={c: f"{c}_s1" for c in cols if c not in ("date", "family")})
                m = s.merge(d, on=["date", "family"], validate="one_to_one")
                if m.date.nunique() != x[x.information_set == left].date.nunique() or m.date.nunique() != x[x.information_set == right].date.nunique():
                    raise RuntimeError("Spot and Spot+Derivatives predictions are not date-matched")
                if not np.allclose(m.actual_transformed_s0, m.actual_transformed_s1, rtol=0, atol=1e-12, equal_nan=True):
                    raise RuntimeError("Matched information sets have different targets")
                m["loss_s0"] = (m.prediction_transformed_s0 - m.actual_transformed_s0).abs()
                m["loss_s1"] = (m.prediction_transformed_s1 - m.actual_transformed_s1).abs()
                m["delta_s1_minus_s0"] = m.loss_s1 - m.loss_s0
                m["category"] = m.family.map(CATEGORY)
                family_rows = []
                for family, g in m.groupby("family", sort=False):
                    family_rows.append({
                        "asset": asset, "target": target, "feature_pair": pair, "family": family,
                        "n_dates": int(g.date.nunique()),
                        "spot_only_transformed_mae": float(g.loss_s0.mean()),
                        "spot_plus_derivatives_transformed_mae": float(g.loss_s1.mean()),
                        "delta_plus_minus_spot": float(g.delta_s1_minus_s0.mean()),
                    })
                fam_df = pd.DataFrame(family_rows)
                category_rows = []
                for cat, g in fam_df.groupby(fam_df.family.map(CATEGORY)):
                    category_rows.append({
                        "category": cat,
                        "spot_only_transformed_mae": float(g.spot_only_transformed_mae.mean()),
                        "spot_plus_derivatives_transformed_mae": float(g.spot_plus_derivatives_transformed_mae.mean()),
                        "delta_plus_minus_spot": float(g.delta_plus_minus_spot.mean()),
                        "families": g.family.tolist(),
                    })
                category_delta = float(np.mean([z["delta_plus_minus_spot"] for z in category_rows]))
                category_s0 = float(np.mean([z["spot_only_transformed_mae"] for z in category_rows]))
                category_s1 = float(np.mean([z["spot_plus_derivatives_transformed_mae"] for z in category_rows]))
                rows.extend(family_rows)
                by_asset_target.append({
                    "asset": asset, "target": target, "feature_pair": pair,
                    "n_unique_dates": int(m.date.nunique()),
                    "first_date": m.date.min().isoformat(), "last_date": m.date.max().isoformat(),
                    "equal_category_spot_only_transformed_mae": category_s0,
                    "equal_category_spot_plus_derivatives_transformed_mae": category_s1,
                    "equal_category_delta_plus_minus_spot": category_delta,
                    "categories_improving": int(sum(z["delta_plus_minus_spot"] < 0 for z in category_rows)),
                    "category_details": category_rows,
                })
    return rows, by_asset_target


def select_best_model_dominance(pred: pd.DataFrame, selected: dict):
    results = []
    rows = selected["selected_models"]
    for asset in ASSETS:
        best = {}
        for info, label in [("legacy_spot_only", "spot_only"), ("legacy_spot_plus_derivatives", "spot_plus_derivatives")]:
            candidates = [x for x in rows if x["asset"] == asset and x["target"] == TARGET and x["infoset"] == info]
            if len(candidates) != len(FAMILIES):
                raise RuntimeError(f"Expected {len(FAMILIES)} development-selected models for {asset}/{info}")
            best[label] = min(candidates, key=lambda x: (x["dev_transformed_oof_mae"], FAMILIES.index(x["family"])))
        s0, s1 = best["spot_only"], best["spot_plus_derivatives"]
        pred_asset = pred[(pred.asset == asset) & (pred.target == TARGET) & (pred.feature_pair == "legacy")]
        a = pred_asset[(pred_asset.information_set == "spot_only") & (pred_asset.family == s0["family"])].copy()
        b = pred_asset[(pred_asset.information_set == "spot_plus_derivatives") & (pred_asset.family == s1["family"])].copy()
        m = a.merge(b, on=["date"], suffixes=("_s0", "_s1"), validate="one_to_one")
        if len(m) == 0 or not np.allclose(m.actual_transformed_s0, m.actual_transformed_s1, atol=1e-12, rtol=0):
            raise RuntimeError("Development-selected best models do not have paired external targets")
        d = (m.prediction_transformed_s1 - m.actual_transformed_s1).abs().to_numpy() - (m.prediction_transformed_s0 - m.actual_transformed_s0).abs().to_numpy()
        results.append({
            "asset": asset, "target": TARGET, "feature_pair": "legacy",
            "selection_rule": "minimum development-only transformed OOF MAE separately by information set",
            "spot_only_selected_family": s0["family"], "spot_only_development_oof_mae": s0["dev_transformed_oof_mae"],
            "spot_plus_selected_family": s1["family"], "spot_plus_development_oof_mae": s1["dev_transformed_oof_mae"],
            "n_unique_dates": int(m.date.nunique()), "first_date": pd.to_datetime(m.date).min().isoformat(), "last_date": pd.to_datetime(m.date).max().isoformat(),
            "spot_only_external_transformed_mae": float(np.mean(np.abs(m.prediction_transformed_s0-m.actual_transformed_s0))),
            "spot_plus_external_transformed_mae": float(np.mean(np.abs(m.prediction_transformed_s1-m.actual_transformed_s1))),
            "delta_plus_minus_spot": float(d.mean()), "paired_block_bootstrap": bootstrap_mean(d, seed=SEED + ASSETS.index(asset) * 1000),
        })
    return results


def confusion(y: np.ndarray, alarm: np.ndarray) -> dict:
    y = np.asarray(y, bool); alarm = np.asarray(alarm, bool)
    tp = int(np.sum(y & alarm)); fp = int(np.sum(~y & alarm)); fn = int(np.sum(y & ~alarm)); tn = int(np.sum(~y & ~alarm))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    prevalence = float(y.mean()) if len(y) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "lift": precision / prevalence if prevalence > 0 else None, "event_prevalence": prevalence, "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn}


def stress_bootstrap(pred: pd.DataFrame, selected: dict):
    stress_thresholds = selected["stress_thresholds"]
    results = []
    for asset in ASSETS:
        for pair, sname, dname in [
            ("legacy", "legacy_spot_only", "legacy_spot_plus_derivatives"),
            ("enhanced", "enhanced_spot_only", "enhanced_spot_plus_derivatives"),
        ]:
            x = pred[(pred.asset == asset) & (pred.target == TARGET) & (pred.feature_pair == pair)].copy()
            x = x[x.family.isin(FAMILIES)].copy()
            if x.empty:
                raise RuntimeError(f"Missing Amihud stress rows for {asset}/{pair}")
            q = stress_thresholds[f"{asset}__{TARGET}"]
            event_q90 = float(q["development_target_q90"])
            x["event"] = x.actual_raw >= event_q90
            x["alarm"] = x.prediction_raw >= x.alarm_threshold_raw_dev_oof_q90
            family_metrics = []
            for (info, family), g in x.groupby(["information_set", "family"], sort=False):
                family_metrics.append({"information_set": info, "family": family, "category": CATEGORY[family], **confusion(g.event.to_numpy(), g.alarm.to_numpy())})
            agg_by_info = {}
            for info in ["spot_only", "spot_plus_derivatives"]:
                cats = []
                for cat in ["linear", "kernel", "bagged_trees", "boosting", "neural"]:
                    vals = [z for z in family_metrics if z["information_set"] == info and z["category"] == cat]
                    if vals:
                        cats.append({metric: float(np.mean([z[metric] for z in vals if z.get(metric) is not None])) if any(z.get(metric) is not None for z in vals) else None for metric in ["precision", "recall", "f1", "lift"]} | {"category": cat, "family_count": len(vals)})
                agg_by_info[info] = {metric: float(np.mean([z[metric] for z in cats if z[metric] is not None])) if any(z[metric] is not None for z in cats) else None for metric in ["precision", "recall", "f1", "lift"]}
                agg_by_info[info]["categories"] = cats
            # Date-wise family alarms and outcomes; bootstrap whole date blocks, preserving all models.
            dates = pd.DatetimeIndex(sorted(pd.to_datetime(x.date, utc=True).unique()))
            if len(dates) < 1000:
                raise RuntimeError("Stress-warning external sample unexpectedly small")
            per_info_family = {}
            for info in ["spot_only", "spot_plus_derivatives"]:
                for fam in FAMILIES:
                    z = x[(x.information_set == info) & (x.family == fam)].set_index(pd.to_datetime(x[(x.information_set == info) & (x.family == fam)].date, utc=True)).reindex(dates)
                    if z.event.isna().any() or z.alarm.isna().any():
                        raise RuntimeError("Stress warning model/date grid is incomplete")
                    yv = z.event.to_numpy(bool)
                    av = z.alarm.to_numpy(bool)
                    per_info_family[(info, fam)] = np.vstack([yv & av, (~yv) & av, yv & (~av)]).astype(np.int64)
            observed_delta = {
                metric: (agg_by_info["spot_plus_derivatives"][metric] - agg_by_info["spot_only"][metric])
                if agg_by_info["spot_plus_derivatives"][metric] is not None and agg_by_info["spot_only"][metric] is not None else None
                for metric in ["precision", "recall", "f1", "lift"]
            }
            boot = {}
            for j, block in enumerate((42, 84)):
                rng = np.random.default_rng(SEED + ASSETS.index(asset)*10000 + (0 if pair == "legacy" else 5000) + j*101)
                vals = {metric: np.empty(N_BOOT, float) for metric in observed_delta}
                n = len(dates)
                n_full, remainder = divmod(n, block)
                starts_n = n_full + int(remainder > 0)
                # Prefix sums make the block bootstrap fast while still sampling whole date blocks.
                prefix = {}
                for key, counts in per_info_family.items():
                    prefix[key] = np.cumsum(np.pad(counts, ((0, 0), (1, 0))), axis=1)
                for b in range(N_BOOT):
                    starts = rng.integers(0, n - block + 1, size=starts_n)
                    scores = {}
                    for info in ["spot_only", "spot_plus_derivatives"]:
                        scores[info] = {}
                        family_scores = {metric: [] for metric in observed_delta}
                        family_categories = []
                        for fam in FAMILIES:
                            pr = prefix[(info, fam)]
                            full_starts = starts[:n_full]
                            total = (pr[:, full_starts + block] - pr[:, full_starts]).sum(axis=1) if n_full else np.zeros(3, dtype=np.int64)
                            if remainder:
                                final_start = starts[-1]
                                total += pr[:, final_start + remainder] - pr[:, final_start]
                            tp, fp, fn = total.astype(float)
                            precision = tp / (tp + fp) if tp + fp else 0.0
                            recall = tp / (tp + fn) if tp + fn else 0.0
                            f1 = 2*tp / (2*tp + fp + fn) if 2*tp + fp + fn else 0.0
                            prevalence = (tp + fn) / (n_full*block + remainder if remainder else n_full*block)
                            lift = precision / prevalence if prevalence > 0 else np.nan
                            family_scores["precision"].append(precision)
                            family_scores["recall"].append(recall)
                            family_scores["f1"].append(f1)
                            family_scores["lift"].append(lift)
                            family_categories.append(CATEGORY[fam])
                        for metric, fvals in family_scores.items():
                            cat_means = []
                            for cat in ["linear", "kernel", "bagged_trees", "boosting", "neural"]:
                                v = [fvals[i] for i, c in enumerate(family_categories) if c == cat]
                                if v and np.isfinite(v).all():
                                    cat_means.append(float(np.mean(v)))
                            scores[info][metric] = float(np.mean(cat_means)) if cat_means else np.nan
                    for metric in observed_delta:
                        vals[metric][b] = (scores["spot_plus_derivatives"][metric] - scores["spot_only"][metric]) if np.isfinite(scores["spot_plus_derivatives"][metric]) and np.isfinite(scores["spot_only"][metric]) else np.nan
                for metric, arr in vals.items():
                    boot[str(block)] = boot.get(str(block), {})
                    finite = arr[np.isfinite(arr)]
                    boot[str(block)][metric] = ({
                        "ci95_percentile": [float(np.quantile(finite, .025)), float(np.quantile(finite, .975))],
                        "probability_delta_above_zero": float((finite > 0).mean()),
                        "valid_replications": int(len(finite)), "replications": N_BOOT,
                    } if len(finite) and observed_delta[metric] is not None else None)
            results.append({
                "asset": asset, "target": TARGET, "feature_pair": pair,
                "event_threshold_development_q90": event_q90,
                "n_unique_external_dates": int(len(dates)), "equal_category_spot_only": agg_by_info["spot_only"],
                "equal_category_spot_plus_derivatives": agg_by_info["spot_plus_derivatives"],
                "delta_plus_minus_spot": observed_delta, "paired_date_block_bootstrap": boot,
                "family_level": family_metrics,
            })
    return results


def transform_target(y: np.ndarray, name: str):
    y = np.maximum(np.asarray(y, float), 0.0)
    if name == "raw_nonnegative":
        return y, 1.0
    if name == "log1p_y":
        return np.log1p(y), 1.0
    if name == "log1p_y_over_train_median_positive":
        pos = y[y > 0]
        scale = float(np.median(pos)) if len(pos) else 1.0
        return np.log1p(y / max(scale, np.finfo(float).tiny)), scale
    raise ValueError(f"Unknown Phase-7 target transform {name}")


def fit_placebo_pair(panel: pd.DataFrame, selected: dict):
    trans = {(r["asset"], r["target"]): r["transform"] for r in selected["transform_by_asset_target"]}
    output = []
    data_diagnostics = []
    for ai, asset in enumerate(ASSETS):
        g = panel[panel.asset == asset].sort_values("period_start_utc").reset_index(drop=True)
        dates = pd.DatetimeIndex(pd.to_datetime(g.period_start_utc, utc=True))
        if len(dates) > 1 and not np.all(np.diff(dates.asi8) == pd.Timedelta(hours=4).value):
            raise RuntimeError(f"{asset} panel is not a continuous 4-hour grid")
        n = len(g)
        spot = g[SPOT].to_numpy(float)
        deriv = g[DERIV].to_numpy(float)
        y = pd.to_numeric(g[TARGET], errors="coerce").to_numpy(float)
        ends = pd.to_datetime(g.future_window_end_utc_exclusive, utc=True)
        spot_seq = np.full((n, 6*len(SPOT)), np.nan, float)
        for i in range(5, n):
            spot_seq[i] = spot[i-5:i+1].reshape(-1)
        for label, offset in [("future_derivatives_t_plus_7_to_t_plus_12", 12), ("forward_displacement_42_intervals", 42)]:
            deriv_seq = np.full((n, 6*len(DERIV)), np.nan, float)
            valid_end = n - offset
            for i in range(5, valid_end):
                deriv_seq[i] = deriv[i+offset-5:i+offset+1].reshape(-1)
            finite_y = np.isfinite(y) & (y >= 0)
            dev = (dates <= DEV_LAST) & (ends <= DEV_END_EXCL) & finite_y
            # The shifted derivative history must end within development for training.
            shift_in_dev = np.zeros(n, bool)
            shift_in_dev[:valid_end] = dates[offset:offset+valid_end] <= DEV_LAST
            dev &= shift_in_dev
            ext = (dates >= EXT_START) & (dates <= EXT_LAST) & (ends <= SAMPLE_END_EXCL) & finite_y
            ext &= np.arange(n) < valid_end
            mask_train = dev & np.isfinite(spot_seq).all(axis=1) & np.isfinite(deriv_seq).all(axis=1)
            mask_test = ext & np.isfinite(spot_seq).all(axis=1) & np.isfinite(deriv_seq).all(axis=1)
            if mask_train.sum() < 1000 or mask_test.sum() < 1000:
                raise RuntimeError(f"Insufficient matched impossible-time sample {asset}/{label}: {mask_train.sum()}/{mask_test.sum()}")
            tr_idx = np.where(mask_train)[0]; te_idx = np.where(mask_test)[0]
            trans_name = trans[(asset, TARGET)]
            yt, scale = transform_target(y[tr_idx], trans_name)
            yte = transform_target(y[te_idx], trans_name)[0]
            x0tr, x0te = spot_seq[tr_idx], spot_seq[te_idx]
            x1tr, x1te = np.concatenate([x0tr, deriv_seq[tr_idx]], axis=1), np.concatenate([x0te, deriv_seq[te_idx]], axis=1)
            preds = {}
            for which, xtr, xte in [("spot_only", x0tr, x0te), ("spot_plus_future_derivatives", x1tr, x1te)]:
                scaler = StandardScaler().fit(xtr)
                model = Ridge(alpha=1.0).fit(scaler.transform(xtr), yt)
                p = np.maximum(model.predict(scaler.transform(xte)), 0.0)
                preds[which] = p
            delta = np.abs(preds["spot_plus_future_derivatives"]-yte) - np.abs(preds["spot_only"]-yte)
            date_delta = pd.Series(delta, index=dates[te_idx]).groupby(level=0).mean().sort_index()
            grid = pd.date_range(EXT_START, EXT_LAST, freq="4h")
            full = date_delta.reindex(grid).to_numpy(float)
            # Block resampling retains gaps on the original UTC grid and ignores missing paired rows.
            bts = {}
            for j, block in enumerate((42, 84)):
                rng = np.random.default_rng(SEED + ai*10000 + offset*10 + j)
                valid = np.isfinite(full)
                ngrid = len(full)
                vals = np.empty(N_BOOT, float)
                for b in range(N_BOOT):
                    ids = moving_block_indices(ngrid, block, rng)
                    z = full[ids]
                    z = z[np.isfinite(z)]
                    vals[b] = z.mean() if len(z) else np.nan
                bts[str(block)] = {
                    "block_length_4h": block, "replications": N_BOOT,
                    "ci95_percentile": [float(np.nanquantile(vals, .025)), float(np.nanquantile(vals, .975))],
                    "probability_future_derivatives_improve_past_target": float(np.nanmean(vals < 0)),
                    "valid_external_dates": int(valid.sum()),
                }
            output.append({
                "asset": asset, "target": TARGET, "placebo": label, "source_rows": f"t+7:t+12" if offset == 12 else f"t+37:t+42",
                "model": "Ridge(alpha=1.0), StandardScaler fit on development only",
                "transform": trans_name, "target_scale_training_median": scale,
                "train_dates": [dates[tr_idx].min().isoformat(), dates[tr_idx].max().isoformat()],
                "test_dates": [dates[te_idx].min().isoformat(), dates[te_idx].max().isoformat()],
                "n_train_dates": int(len(tr_idx)), "n_external_dates": int(len(te_idx)),
                "spot_only_transformed_mae": float(np.mean(np.abs(preds["spot_only"]-yte))),
                "spot_plus_future_derivatives_transformed_mae": float(np.mean(np.abs(preds["spot_plus_future_derivatives"]-yte))),
                "delta_future_derivatives_minus_spot_only": float(delta.mean()),
                "paired_block_bootstrap": bts,
                "interpretation": "impossible-time diagnostic only; negative incremental loss signals temporal persistence/confounding concern and is not forward predictive evidence",
            })
            data_diagnostics.append({"asset": asset, "placebo": label, "train_rows": int(len(tr_idx)), "external_rows": int(len(te_idx)), "complete_case_rows_match_model_pair": True})
    return output, data_diagnostics


def main():
    started = time.time()
    freeze, p7freeze = verify_hashes()
    protected_before = protected_file_snapshot()
    archive_check = verify_candidate_archives()
    p7out = ROOT / "outputs/phase7_liquidity_replication"
    p7audit = json.loads((p7out / "phase7_data_audit.json").read_text())
    ext_audit = json.loads((p7out / "external_data_audit.json").read_text())
    old_decision = json.loads((p7out / "phase7_replication_decision.json").read_text())
    selected = json.loads((p7out / "phase7_selected_models.json").read_text())
    evaluation = json.loads((p7out / "phase7_external_evaluation.json").read_text())
    dev_cv = json.loads((p7out / "phase7_development_cv.json").read_text())
    pred = pd.read_csv(PRED, parse_dates=["date"])
    pred["date"] = pd.to_datetime(pred.date, utc=True)
    if pred.duplicated(["asset", "target", "feature_pair", "information_set", "family", "date"]).any():
        raise RuntimeError("External forecast predictions have duplicate model/date keys")
    if set(pred.asset.unique()) != set(ASSETS) or set(pred.target.unique()) != set(TARGETS):
        raise RuntimeError("External forecast asset/target universe differs from the frozen registry")
    if not evaluation.get("raw_market_data_downloaded") is False:
        raise RuntimeError("Phase-7 evaluation provenance indicates market data download")

    # Validate liquidity construction and future-window alignment from the cached panel.
    panel = pd.read_csv(PANEL, usecols=["period_start_utc", "asset", "spot_return", "spot_quote_volume", "spot_amihud", "corwin_schultz_4h", TARGET, "future_kyle_abs_lambda_24h", "future_corwin_schultz_spread_24h", "future_window_start_utc", "future_window_end_utc_exclusive"])
    panel["period_start_utc"] = pd.to_datetime(panel.period_start_utc, utc=True)
    panel["future_window_start_utc"] = pd.to_datetime(panel.future_window_start_utc, utc=True)
    panel["future_window_end_utc_exclusive"] = pd.to_datetime(panel.future_window_end_utc_exclusive, utc=True)
    if panel.duplicated(["asset", "period_start_utc"]).any():
        raise RuntimeError("Phase-7 panel has duplicate asset/date keys")
    a_calc = 1e6 * panel.spot_return.abs() / panel.spot_quote_volume.where(panel.spot_quote_volume > 0)
    a_mask = a_calc.notna() | panel.spot_amihud.notna()
    a_diff = (a_calc - panel.spot_amihud).abs()
    panel = panel.sort_values(["asset", "period_start_utc"])
    shifted = panel.groupby("asset", sort=False)["spot_amihud"]
    expected_future_a = sum(shifted.shift(-k) for k in range(1, 7)) / 6.0
    mask_fa = expected_future_a.notna() | panel[TARGET].notna()
    future_a_err = (expected_future_a - panel[TARGET]).abs()
    cs_series = panel.groupby("asset", sort=False)["corwin_schultz_4h"]
    expected_cs = sum(cs_series.shift(-k) for k in range(2, 7)) / 5.0
    cs_target = "future_corwin_schultz_spread_24h"
    mask_cs = expected_cs.notna() | panel[cs_target].notna()
    cs_err = (expected_cs - panel[cs_target]).abs()
    label_rows = panel[TARGET].notna()
    window_ok = (panel.future_window_start_utc == panel.period_start_utc + pd.Timedelta(hours=4)) & (panel.future_window_end_utc_exclusive == panel.period_start_utc + pd.Timedelta(hours=28))
    extpanel = panel[(panel.period_start_utc >= EXT_START) & (panel.period_start_utc <= EXT_LAST)]
    semantics = {
        "amihud": {
            "definition": "1e6 * abs(4-hour log return) / positive Binance spot quote turnover in USDT",
            "larger_means": "greater measured price impact / illiquidity",
            "max_abs_reconstruction_error": float(a_diff[a_mask].max()),
            "missingness_identical": bool(a_calc.notna().equals(panel.spot_amihud.notna())),
            "nonpositive_quote_volume_missing": bool(panel.loc[panel.spot_quote_volume <= 0, "spot_amihud"].isna().all()),
            "numerical_tail_quantiles_external": {str(q): float(extpanel.spot_amihud.quantile(q)) for q in [.001, .01, .5, .99, .999]},
            "zero_count_external": int((extpanel.spot_amihud == 0).sum()),
            "negative_count_external": int((extpanel.spot_amihud < 0).sum()),
        },
        "future_amihud": {
            "definition": "arithmetic mean of six 4-hour Amihud observations t+1:t+6",
            "max_abs_reconstruction_error": float(future_a_err[mask_fa].max()),
            "missingness_identical": bool(expected_future_a.notna().equals(panel[TARGET].notna())),
        },
        "future_kyle_abs_lambda_24h": {
            "definition": "absolute intercept-adjusted OLS slope of 288 following 5-minute spot log returns on signed spot USDT quote flow (2*taker-buy quote volume - total quote volume)",
            "unit": "return per USDT signed quote flow",
            "nonnegative": bool((panel.future_kyle_abs_lambda_24h.dropna() >= 0).all()),
            "coverage_external_pct": float(extpanel.future_kyle_abs_lambda_24h.notna().mean()*100),
            "construction_evidence": "verified deterministic Phase-7 builder source and 6,195 cached archive hashes; Phase-7 offline panel rebuild and full source audit passed; not independently reconstructed from 5-minute bars in this reanalysis",
        },
        "future_corwin_schultz_spread_24h": {
            "definition": "mean of five consecutive 4-hour Corwin-Schultz high-low estimates based on adjacent bars wholly within t+1:t+6",
            "max_abs_reconstruction_error": float(cs_err[mask_cs].max()),
            "missingness_identical": bool(expected_cs.notna().equals(panel[cs_target].notna())),
            "coverage_external_pct": float(extpanel[cs_target].notna().mean()*100),
            "interpretation_limit": "range-based spread proxy can also reflect volatility; not an observed bid-ask quote",
        },
        "future_window": {
            "labels_with_metadata": int(label_rows.sum()),
            "start_equals_t_plus_4h_on_valid_labels": bool(window_ok[label_rows].all()),
            "exclusive_end_equals_t_plus_28h_on_valid_labels": bool(window_ok[label_rows].all()),
            "terminal_unlabelled_rows_with_missing_window_metadata": int((~label_rows & panel.future_window_start_utc.isna() & panel.future_window_end_utc_exclusive.isna()).sum()),
            "no_future_labels_after_sample_end": bool((panel.loc[panel[TARGET].notna(), "future_window_end_utc_exclusive"] <= SAMPLE_END_EXCL).all()),
        },
        "panel_rows": int(len(panel)), "panel_unique_asset_dates": True,
        "asset_audit_pass": bool(p7audit.get("candidate_eligibility_pass") and ext_audit.get("selection_was_performance_blind")),
        "post_2024_timestamp_audit": "valid under frozen per-date OI offset mapping; unresolved concern is piecewise source timestamp harmonization, retained as measurement limitation",
    }
    if semantics["amihud"]["max_abs_reconstruction_error"] > 1e-12 or semantics["future_amihud"]["max_abs_reconstruction_error"] > 1e-12 or semantics["future_corwin_schultz_spread_24h"]["max_abs_reconstruction_error"] > 1e-12:
        raise RuntimeError("Liquidity target reconstruction exceeded tolerance")
    if not semantics["amihud"]["missingness_identical"] or not semantics["future_amihud"]["missingness_identical"] or not semantics["future_corwin_schultz_spread_24h"]["missingness_identical"]:
        raise RuntimeError("Liquidity reconstruction changed missingness")
    if not all(semantics["future_window"][k] for k in ["start_equals_t_plus_4h_on_valid_labels", "exclusive_end_equals_t_plus_28h_on_valid_labels", "no_future_labels_after_sample_end"]):
        raise RuntimeError("Future target timing validation failed")
    dump("liquidity_measurement_audit.json", semantics)

    family_rows, forecast_summary = matched_prediction_metrics(pred)
    existing = {(x["asset"], x["target"], x["feature_pair"]): x for x in evaluation["equal_category_aggregate"]}
    for r in forecast_summary:
        ref = existing[(r["asset"], r["target"], r["feature_pair"])]
        if not math.isclose(r["equal_category_delta_plus_minus_spot"], ref["equal_category_mean_delta"], abs_tol=5e-10, rel_tol=1e-8):
            raise RuntimeError(f"Forecast delta does not reproduce Phase-7 summary: {r['asset']}/{r['target']}/{r['feature_pair']}")
        r["frozen_phase7_bootstrap_ci95"] = ref["bootstrap_ci95"]
        r["frozen_phase7_bootstrap_probability_improvement"] = ref["bootstrap_probability_improvement"]
    dump("external_forecast_summary.json", {"prediction_sha256": sha(PRED), "classification_source": "frozen Phase-7 equal-category evaluation", "rows": family_rows, "equal_category": forecast_summary})

    dominance = select_best_model_dominance(pred, selected)
    dump("best_model_dominance.json", {"selection_source": "phase7 development OOF only", "external_scores_used_for_selection": False, "results": dominance})

    stress = stress_bootstrap(pred, selected)
    # Exact check against stored Phase-7 per-family stress statistics.
    stored_stress = {(x["asset"], x["target"], x["feature_pair"], x["family"], x["information_set"]): x for x in evaluation["stress_warning"]}
    for r in stress:
        z = pred[(pred.asset == r["asset"]) & (pred.target == TARGET) & (pred.feature_pair == r["feature_pair"])]
        q = selected["stress_thresholds"][f"{r['asset']}__{TARGET}"]["development_target_q90"]
        for info in ["spot_only", "spot_plus_derivatives"]:
            for fam in FAMILIES:
                w = z[(z.information_set == info) & (z.family == fam)]
                comp = confusion((w.actual_raw.to_numpy() >= q), (w.prediction_raw.to_numpy() >= w.alarm_threshold_raw_dev_oof_q90.to_numpy()))
                ref = stored_stress[(r["asset"], TARGET, r["feature_pair"], fam, info)]
                if not math.isclose(comp["f1"], ref["f1"], abs_tol=1e-12):
                    raise RuntimeError("Recomputed frozen stress-warning F1 does not match Phase-7")
    dump("stress_warning_utility.json", {"rule": "fixed development q90 event and per-model development-OOF q90 alarm; five equal-weight categories", "results": stress})

    # Matched fixed-model impossible-time controls; both information sets share dates and training rows.
    feature_cols = ["period_start_utc", "asset", *SPOT, *DERIV, TARGET, "future_window_end_utc_exclusive"]
    full_panel = pd.read_csv(PANEL, usecols=feature_cols, parse_dates=["period_start_utc", "future_window_end_utc_exclusive"])
    full_panel["period_start_utc"] = pd.to_datetime(full_panel.period_start_utc, utc=True)
    full_panel["future_window_end_utc_exclusive"] = pd.to_datetime(full_panel.future_window_end_utc_exclusive, utc=True)
    neg_results, neg_data = fit_placebo_pair(full_panel, selected)
    dump("matched_negative_controls.json", {"status": "COMPLETE", "reanalysis_status": "previously inspected external period; secondary diagnostic", "results": neg_results, "sample_diagnostics": neg_data})

    # Frozen period stability by model/category across all assets and three outcome definitions.
    seg_defs = [
        ("2024-03_to_2024-12", EXT_START, pd.Timestamp("2025-01-01T00:00:00Z")),
        ("2025", pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
        ("2026_partial", pd.Timestamp("2026-01-01T00:00:00Z"), SAMPLE_END_EXCL),
    ]
    stability = []
    for asset in ASSETS:
        for target in TARGETS:
            for pair in ["legacy", "enhanced"]:
                x = pred[(pred.asset == asset) & (pred.target == target) & (pred.feature_pair == pair)]
                x = x[x.family.isin(FAMILIES)]
                for seg, start, end in seg_defs:
                    z = x[(x.date >= start) & (x.date < end)]
                    if z.empty:
                        continue
                    family_delta = []
                    for fam, g in z.groupby("family"):
                        m = g.pivot(index="date", columns="information_set", values=["actual_transformed", "prediction_transformed"])
                        d = (m[("prediction_transformed", "spot_plus_derivatives")] - m[("actual_transformed", "spot_plus_derivatives")]).abs() - (m[("prediction_transformed", "spot_only")] - m[("actual_transformed", "spot_only")]).abs()
                        family_delta.append({"family": fam, "category": CATEGORY[fam], "delta_plus_minus_spot": float(d.mean())})
                    cats = []
                    for cat in ["linear", "kernel", "bagged_trees", "boosting", "neural"]:
                        v = [q["delta_plus_minus_spot"] for q in family_delta if q["category"] == cat]
                        if v: cats.append({"category": cat, "delta_plus_minus_spot": float(np.mean(v))})
                    stability.append({
                        "asset": asset, "target": target, "feature_pair": pair, "segment": seg,
                        "n_unique_dates": int(z.date.nunique()),
                        "equal_category_delta_plus_minus_spot": float(np.mean([x["delta_plus_minus_spot"] for x in cats])),
                        "categories_improving": int(sum(x["delta_plus_minus_spot"] < 0 for x in cats)),
                        "category_deltas": cats, "family_deltas": family_delta,
                    })
    dump("temporal_stability.json", {"segments": [x[0] for x in seg_defs], "primary_target": TARGET, "results": stability})

    legacy_amihud = [x for x in forecast_summary if x["target"] == TARGET and x["feature_pair"] == "legacy"]
    conds = old_decision["amihud_replication"]["conditions"]
    asset_rep = {
        "selection": "BTC/ETH core plus SOL/BNB/XRP mechanically selected before estimation by the frozen outcome-blind eligibility protocol",
        "asset_order": ASSETS,
        "legacy_amihud_results": [{k: r[k] for k in ["asset", "n_unique_dates", "equal_category_delta_plus_minus_spot", "categories_improving", "frozen_phase7_bootstrap_ci95"]} for r in legacy_amihud],
        "frozen_phase7_rule_conditions": conds,
        "frozen_phase7_rule_passed": bool(old_decision["amihud_replication"]["passed"]),
        "assets_with_improving_equal_category_loss": int(sum(r["equal_category_delta_plus_minus_spot"] < 0 for r in legacy_amihud)),
        "required_assets_improving": 4,
    }
    dump("asset_replication.json", asset_rep)

    proxy_results = []
    for target in ["future_kyle_abs_lambda_24h", "future_corwin_schultz_spread_24h"]:
        rs = [x for x in forecast_summary if x["target"] == target and x["feature_pair"] == "legacy"]
        btc, eth = (next(x for x in rs if x["asset"] == a) for a in ["BTC", "ETH"])
        improving_both = btc["equal_category_delta_plus_minus_spot"] < 0 and eth["equal_category_delta_plus_minus_spot"] < 0
        ci_both = all(x["frozen_phase7_bootstrap_ci95"][1] < 0 for x in [btc, eth])
        cats_both = btc["categories_improving"] >= 3 and eth["categories_improving"] >= 3
        proxy_results.append({
            "target": target,
            "btc": {"delta": btc["equal_category_delta_plus_minus_spot"], "ci95": btc["frozen_phase7_bootstrap_ci95"], "categories_improving": btc["categories_improving"]},
            "eth": {"delta": eth["equal_category_delta_plus_minus_spot"], "ci95": eth["frozen_phase7_bootstrap_ci95"], "categories_improving": eth["categories_improving"]},
            "btc_eth_direction_condition": improving_both,
            "btc_eth_bootstrap_condition": ci_both,
            "btc_eth_category_condition": cats_both,
            "frozen_proxy_generalization_pass": bool(improving_both and ci_both and cats_both),
            "all_asset_legacy_results": [{"asset": x["asset"], "delta": x["equal_category_delta_plus_minus_spot"], "ci95": x["frozen_phase7_bootstrap_ci95"]} for x in rs],
        })
    dump("proxy_generalization.json", {"proxies": proxy_results, "any_distinct_proxy_passes": any(x["frozen_proxy_generalization_pass"] for x in proxy_results), "proxy_semantics": semantics})

    nc_flags = []
    for r in neg_results:
        c = r["paired_block_bootstrap"]["42"]
        ci = c["ci95_percentile"]
        nc_flags.append({"asset": r["asset"], "placebo": r["placebo"], "delta": r["delta_future_derivatives_minus_spot_only"], "ci95": ci, "supported_future_info_gain_on_past_target": ci[1] < 0})
    supported_n = sum(x["supported_future_info_gain_on_past_target"] for x in nc_flags)
    old_decision["amihud_replication"]["passed"] = False if not all(conds.values()) else old_decision["amihud_replication"]["passed"]
    overall_class = "NO_ROBUST_GENERALIZATION" if not old_decision["amihud_replication"]["passed"] else ("MULTIPROXY_GENERALIZATION" if any(x["frozen_proxy_generalization_pass"] for x in proxy_results) else "AMIHUD_SPECIFIC_PREDICTIVE_VALUE")
    final = {
        "phase6_finding": "target-specific 24-hour future Amihud predictive signal, previously reported; not a causal result",
        "phase7_frozen_external_classification": old_decision["classification"],
        "overall_classification": overall_class,
        "amihud_external_replication_pass": bool(old_decision["amihud_replication"]["passed"]),
        "distinct_proxy_generalization_pass": bool(any(x["frozen_proxy_generalization_pass"] for x in proxy_results)),
        "development_selected_best_model_dominance": {r["asset"]: (r["delta_plus_minus_spot"] < 0) for r in dominance},
        "supported_matched_impossible_time_controls": supported_n,
        "total_matched_impossible_time_controls": len(neg_results),
        "interpretation": "Existing Phase-7 external replication criteria remain authoritative. New best-family, stress-warning, and paired negative-control diagnostics are secondary reanalysis on previously inspected dates.",
        "causal_claim_supported": False,
        "quantum_or_other_algorithm_claim": "not relevant to this validation objective",
    }
    dump("final_classification.json", final)

    # Verify all source objects that were meant to remain immutable before closing.
    post_protected = protected_file_snapshot()
    if protected_before != post_protected:
        raise RuntimeError("Stage 3 / Phase 5 / Phase 6 protected files changed during Phase 8")
    for rel, expected in freeze["inputs"].items():
        if sha(ROOT / rel) != expected:
            raise RuntimeError(f"Frozen Phase-8 input changed during analysis: {rel}")
    if verify_candidate_archives() != archive_check:
        raise RuntimeError("Cached raw archive integrity changed during analysis")
    p7after = json.loads((p7out / "phase7_external_freeze.json").read_text())
    for rel, expected in p7after["files"].items():
        if sha(p7out / rel) != expected:
            raise RuntimeError(f"Phase-7 frozen forecast output changed during Phase 8: {rel}")

    data_validation = {
        "status": "PASS", "phase7_panel_hash": sha(PANEL), "external_prediction_hash": sha(PRED),
        "panel_rows": int(len(panel)), "prediction_rows": int(len(pred)), "panel_duplicate_asset_dates": 0,
        "prediction_duplicate_keys": 0, "assets": ASSETS, "targets": TARGETS,
        "external_prediction_start": pred.date.min().isoformat(), "external_prediction_end": pred.date.max().isoformat(),
        "phase7_asset_eligibility_pass": bool(p7audit.get("candidate_eligibility_pass")),
        "phase7_asset_selection_performance_blind": bool(ext_audit.get("selection_was_performance_blind")),
        "all_frozen_phase7_predictions_verified": True,
        "archive_integrity": archive_check,
        "amihud_reconstruction_max_abs_error": semantics["amihud"]["max_abs_reconstruction_error"],
        "future_amihud_reconstruction_max_abs_error": semantics["future_amihud"]["max_abs_reconstruction_error"],
        "future_corwin_schultz_reconstruction_max_abs_error": semantics["future_corwin_schultz_spread_24h"]["max_abs_reconstruction_error"],
        "offline_only": True,
    }
    dump("data_validation.json", data_validation)
    dump("input_integrity.json", {
        "protected_prior_stage_files": len(protected_before),
        "protected_prior_stage_tree_fingerprint": tree_fingerprint(protected_before),
        "protected_prior_stage_files_unchanged": True,
        "frozen_phase8_inputs_unchanged": True,
        "phase7_external_freeze_and_predictions_unchanged": True,
        "cached_raw_archive_hashes_verified": archive_check,
    })
    dump("reproducibility.json", {
        "status": "PASS", "network_used": False, "raw_data_downloaded": False,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "script_sha256": sha(Path(__file__)), "phase8_protocol_sha256": freeze["protocol_files"]["outputs/phase8_liquidity_generalization/protocol.md"],
        "input_manifest_sha256": sha(OUT / "freeze_manifest.json"),
        "prior_stage_protected_files_unchanged": True, "phase7_frozen_evaluation_unchanged": True,
        "external_predictions_reused": True, "external_period_previously_inspected": True,
        "bootstrap": {"replications": N_BOOT, "block_lengths_4h": [42, 84], "seed_base": SEED},
        "deterministic_outputs": "All analysis JSON and Markdown are deterministically regenerated; elapsed runtime is excluded from byte-equivalence comparison.",
    })

    stress_lines = []
    for r in stress:
        if r["feature_pair"] == "legacy" and r["asset"] in ["BTC", "ETH"]:
            f1 = r["delta_plus_minus_spot"]["f1"]
            ci = r["paired_date_block_bootstrap"]["42"]["f1"]["ci95_percentile"]
            stress_lines.append(f"{r['asset']}: equal-category F1 delta {f1:+.4f} (S1−S0), 42-step block CI [{ci[0]:+.4f}, {ci[1]:+.4f}].")
    best_lines = []
    for r in dominance:
        ci = r["paired_block_bootstrap"]["42"]["ci95_percentile"]
        best_lines.append(f"{r['asset']}: development-selected {r['spot_only_selected_family']} vs {r['spot_plus_selected_family']}; S1−S0 MAE {r['delta_plus_minus_spot']:+.6g}, CI [{ci[0]:+.6g}, {ci[1]:+.6g}].")
    nc_lines = []
    for r in neg_results:
        ci = r["paired_block_bootstrap"]["42"]["ci95_percentile"]
        nc_lines.append(f"{r['asset']} {r['placebo']}: ΔMAE {r['delta_future_derivatives_minus_spot_only']:+.6g}, CI [{ci[0]:+.6g}, {ci[1]:+.6g}].")
    report = [
        "# Phase 8 — Liquidity-signal validation and generalization",
        "",
        "This is a predictive reanalysis of cached Binance data. The Phase-7 post-2024 results were already inspected, so this work is not a new untouched confirmation. No market data were downloaded, and Stage 3 / Phase 5 / Phase 6 / Phase 7 artifacts remain unchanged.",
        "",
        "## Measurement and sample",
        "",
        f"The Phase-7 panel contains {len(panel):,} rows across BTC, ETH, SOL, BNB and XRP. The Amihud formula independently reproduces to maximum absolute error {semantics['amihud']['max_abs_reconstruction_error']:.3g}; the 24-hour target reproduces to {semantics['future_amihud']['max_abs_reconstruction_error']:.3g}. The Kyle price-impact proxy is built from 5-minute signed spot quote flow; Corwin–Schultz is a distinct high-low range spread proxy. Both were constructed by the cached Phase-7 builder, and 6,195 raw archive hashes were reverified. Kyle was not independently re-estimated from all 5-minute bars in this Phase-8 summary.",
        "",
        "The external sample begins after the empirically audited archive-layout boundary and uses the frozen piecewise OI timestamp map. This is semantically usable under the prior audit, with that harmonization retained as a measurement limitation. SOL, BNB and XRP were included by pre-estimation coverage/timestamp rules, not forecast results.",
        "",
        "## Forecast generalization",
        "",
        f"The frozen Phase-7 rule remains `{old_decision['classification']}`. For legacy Amihud, the equal-category Spot+Derivatives improvement criterion is met in {asset_rep['assets_with_improving_equal_category_loss']}/5 assets, below the required 4/5; BTC and ETH also fail the requirement that at least 3/5 categories improve in both assets, and the BTC/ETH paired intervals do not both exclude zero. The two distinct liquidity proxies do not pass the frozen BTC/ETH and category criteria.",
        "",
        "The best-family comparison selects one family per information set using development OOF loss only, then compares the already saved external forecasts:",
        "", *best_lines,
        "",
        "This winner-selected secondary comparison does not override equal-category or cross-asset replication criteria.",
        "",
        "## Stress warning, time stability, and negative controls",
        "",
        "Stress warnings use the frozen development q90 event/alarm thresholds and equal-weight five categories:",
        "", *stress_lines,
        "",
        f"The matched impossible-time tests found {supported_n}/{len(neg_results)} cases where future/displaced derivatives information significantly reduced past-target error under the fixed paired block-bootstrap screen. Details below; these are persistence/temporal-contamination diagnostics, not forecastable forward gains:",
        "", *nc_lines,
        "",
        "Fixed 2024-03–2024-12, 2025 and 2026-partial windows show materially changing signs across assets, models, and targets; the full machine-readable segment results are in `temporal_stability.json`.",
        "",
        "## Defensible interpretation",
        "",
        "The narrow Phase-6 24-hour Amihud result does not generalize robustly under the already frozen Phase-7 multi-condition rule. Added derivatives can change individual model-family stress warnings and selected best-family losses, but those secondary results are asset/model dependent and do not establish cross-proxy replication. No causal claim is supported. The strongest next test would reserve a genuinely unseen venue or future period under a new preregistered protocol; the already inspected Binance 2024–2026 period cannot serve as fresh confirmation.",
        "",
        f"Overall classification: **{overall_class}**. Frozen Phase-7 classification: **{old_decision['classification']}**.",
    ]
    (OUT / "research_summary.md").write_text("\n".join(report) + "\n")
    lines = [
        "PHASE 8 — LIQUIDITY-SIGNAL VALIDATION AND GENERALIZATION",
        f"Data validation: {data_validation['status']}",
        f"Assets: {', '.join(ASSETS)}; rows: {len(panel)}; external forecast rows: {len(pred)}",
        f"Amihud reconstruction max error: {semantics['amihud']['max_abs_reconstruction_error']:.3g}",
        f"Kyle/Corwin-Schultz cross-proxy generalization: {any(x['frozen_proxy_generalization_pass'] for x in proxy_results)}",
        f"Frozen Phase-7 classification: {old_decision['classification']}",
        f"Development-selected best-family results saved: {len(dominance)} assets",
        f"Stress-warning comparisons saved: {len(stress)} asset × feature-pair cells",
        f"Matched impossible-time comparisons saved: {len(neg_results)}",
        f"Supported impossible-time future-information gains: {supported_n}/{len(neg_results)}",
        f"Overall classification: {overall_class}",
        "Offline/local-only: PASS; previous-stage artifacts unchanged: PASS",
    ]
    (OUT / "run.log").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
