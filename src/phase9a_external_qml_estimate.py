from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "phase9_external_qml_replication"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "targeted_24h_amihud"))
from phase9_external_qml_estimate import (  # noqa: E402
    BASE_SEED, bootstrap_delta, fit_mlp_residual, fit_qml_one, holm, metrics,
)
from classical_models import fit_predict, stable_seed  # noqa: E402

START = pd.Timestamp("2024-10-06T20:00:00Z")
END = pd.Timestamp("2026-10-06T20:00:00Z")
BAR = pd.Timedelta(hours=4)
TARGET_END = pd.Timedelta(hours=28)
ASSETS = ("BTC", "ETH")
RESULT_NAMES = [
    "spot_oof_predictions.parquet", "residual_training_data.parquet",
    "holdout_spotonly_predictions.parquet", "holdout_mlp_corrected_predictions.parquet",
    "holdout_qml_corrected_predictions.parquet", "primary_metrics.json", "delta_q_results.json",
    "delta_i_results.json", "bootstrap_42.json", "bootstrap_84.json", "primary_holm.json",
    "negative_controls.json", "btc_decision.json", "eth_decision.json", "paper_level_decision.json",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def verify_frozen_inputs() -> dict:
    manifest_path = OUT / "pre_model_freeze_manifest.json"
    digest_path = OUT / "pre_model_freeze_manifest.sha256"
    if not manifest_path.exists() or not digest_path.exists():
        raise RuntimeError("Pre-model freeze manifest is absent; model execution is prohibited.")
    expected_manifest_hash = digest_path.read_text().split()[0]
    if sha256(manifest_path) != expected_manifest_hash:
        raise RuntimeError("Pre-model freeze manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN_BEFORE_MODEL_RESULTS" or manifest.get("model_fitting_performed") is not False:
        raise RuntimeError("Pre-model manifest does not certify a pre-result freeze")
    if manifest.get("forecast_results_viewed") is not False:
        raise RuntimeError("Pre-model manifest says results were viewed")
    for name, expected in manifest["input_hashes"].items():
        path = (ROOT / name) if name.startswith("src/") else (OUT / name)
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Frozen input hash mismatch: {name}")
    prior = json.loads((OUT / "prior_artifact_hashes_before.json").read_text(encoding="utf-8"))
    prior_records = prior.get("artifacts") if isinstance(prior, dict) else None
    if not isinstance(prior_records, list):
        raise RuntimeError("Immutable prior-artifact hash snapshot has an unexpected schema")
    for record in prior_records:
        rel, expected = record["file"], record["sha256"]
        path = ROOT / rel
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Immutable prior-stage artifact changed: {rel}")
    if json.loads((OUT / "qml_fixture_validation.json").read_text()).get("status") != "VALID":
        raise RuntimeError("Frozen QML fixture validation did not pass")
    return manifest


def make_features(panel: pd.DataFrame, registry: dict) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame]:
    g = panel.sort_values("period_start_utc").reset_index(drop=True).copy()
    spot = registry["registry"]["spot_only"]["base_variables"]
    der = registry["registry"]["spot_plus_derivatives"]["adds_base_variables"]
    static = registry["registry"]["spot_plus_derivatives"]["static_derivative_features"]
    cal = [*[f"utc_slot_{i}" for i in range(1, 6)], *[f"weekday_{i}" for i in range(1, 7)], "funding_schedule_window"]
    x0, xd = {}, {}
    for lag in range(5, -1, -1):
        for col in spot:
            x0[f"{col}__lag{lag}"] = pd.to_numeric(g[col], errors="coerce").shift(lag)
        for col in der:
            xd[f"{col}__lag{lag}"] = pd.to_numeric(g[col], errors="coerce").shift(lag)
    for col in cal:
        x0[col] = pd.to_numeric(g[col], errors="coerce")
        xd[col] = pd.to_numeric(g[col], errors="coerce")
    for col in static:
        xd[col] = pd.to_numeric(g[col], errors="coerce")
    x0_df, xd_df = pd.DataFrame(x0), pd.DataFrame(xd)
    y_raw = pd.to_numeric(g.future_amihud_24h_mean, errors="coerce").to_numpy(float)
    y = np.log1p(y_raw)
    valid = np.isfinite(x0_df.to_numpy(float)).all(axis=1) & np.isfinite(xd_df.to_numpy(float)).all(axis=1) & np.isfinite(y)
    g["target_raw"] = y_raw
    g["target_transformed"] = y
    g["model_valid"] = valid
    return x0_df, xd_df, valid, g


def _df_for_output(rows: list[dict]) -> pd.DataFrame:
    out = pd.DataFrame(rows)
    if "date" in out:
        out["date"] = pd.to_datetime(out.date, utc=True)
    return out


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False, version="2.6")


def _load_checkpoint(path: Path, run_tag: str, task: str) -> dict[tuple[str, int], dict]:
    done = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("run_tag") != run_tag or row.get("task") != task:
            continue
        key = (row["asset"], int(row["seed"]))
        if key in done:
            raise RuntimeError(f"Duplicate seed checkpoint for {task}: {key}")
        for keyname, predname in (("qml_sha256", "qml_residual"), ("mlp_sha256", "mlp_residual")):
            digest = hashlib.sha256(np.asarray(row[predname], dtype="<f8").tobytes()).hexdigest()
            if digest != row[keyname]:
                raise RuntimeError(f"Checkpoint prediction hash mismatch: {task}/{key}/{keyname}")
        done[key] = row
    return done


def _prediction_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def _fit_seed_models(asset: str, xtrain: np.ndarray, ytrain: np.ndarray, xtest: np.ndarray,
                     cfg: dict, run_tag: str, task: str, checkpoint_path: Path,
                     checkpoint: dict) -> tuple[np.ndarray, np.ndarray, list[dict], list[dict]]:
    q_preds, m_preds, q_meta, m_meta = [], [], [], []
    for seed in [int(s) for s in cfg["seeds"]]:
        item = checkpoint.get((asset, seed))
        if item is not None:
            q = np.asarray(item["qml_residual"], float)
            m = np.asarray(item["mlp_residual"], float)
            qm, mm = item.get("qml_metadata", {}), item.get("mlp_metadata", {})
        else:
            with threadpool_limits(limits=1):
                q, qm = fit_qml_one(asset, xtrain, ytrain, xtest, seed, cfg)
                m, mm = fit_mlp_residual(xtrain, ytrain, xtest, cfg, seed)
            q, m = np.asarray(q, float), np.asarray(m, float)
            if not np.isfinite(q).all() or not np.isfinite(m).all():
                raise RuntimeError(f"Nonfinite {task} residual forecast for {asset}, seed={seed}")
            item = {"run_tag": run_tag, "task": task, "asset": asset, "seed": seed,
                    "qml_residual": q.tolist(), "mlp_residual": m.tolist(),
                    "qml_sha256": _prediction_hash(q), "mlp_sha256": _prediction_hash(m),
                    "qml_metadata": qm, "mlp_metadata": mm}
            with checkpoint_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
                f.flush()
            checkpoint[(asset, seed)] = item
            print(f"{run_tag} {task} {asset} seed={seed} complete", flush=True)
        if len(q) != len(xtest) or len(m) != len(xtest):
            raise RuntimeError(f"Checkpoint prediction length mismatch: {task}/{asset}/{seed}")
        q_preds.append(q)
        m_preds.append(m)
        q_meta.append(qm)
        m_meta.append(mm)
    return np.vstack(q_preds), np.vstack(m_preds), q_meta, m_meta


def build_primary(run_tag: str, out_root: Path) -> dict:
    freeze = verify_frozen_inputs()
    split = json.loads((OUT / "temporal_split.json").read_text(encoding="utf-8"))
    registry = json.loads((OUT / "feature_registry_bybit_frozen.json").read_text(encoding="utf-8"))
    configs = {a: json.loads((OUT / f"frozen_{a.lower()}_{'hybrid4q' if a == 'BTC' else 'qnn6q'}.json").read_text(encoding="utf-8")) for a in ASSETS}
    base_configs = json.loads((OUT / "matched_classical_models.json").read_text(encoding="utf-8"))["spot_only"]
    panels, features, targets, valid_dates = {}, {}, {}, {}
    for asset in ASSETS:
        p = pd.read_parquet(OUT / f"bybit_{asset.lower()}_4h_panel.parquet")
        p["period_start_utc"] = pd.to_datetime(p.period_start_utc, utc=True)
        x0, xd, valid, g = make_features(p, registry)
        panels[asset], features[asset], targets[asset] = g, (x0, xd), (g.target_raw.to_numpy(float), g.target_transformed.to_numpy(float))
        valid_dates[asset] = pd.DatetimeIndex(g.loc[valid, "period_start_utc"])
    common = valid_dates["BTC"].intersection(valid_dates["ETH"]).sort_values()
    if len(common) != split["common_complete_rows"] or common[0].isoformat() != split["common_complete_first"] or common[-1].isoformat() != split["common_complete_last"]:
        raise RuntimeError("Common complete panel changed after pre-model freeze")
    cut = pd.Timestamp(split["T_cut"])
    train_dates = common[common < cut - pd.Timedelta(hours=24)]
    purge_dates = common[(common >= cut - pd.Timedelta(hours=24)) & (common < cut)]
    holdout_dates = common[common >= cut]
    if len(train_dates) != split["usable_observations"]["common_training_after_purge"] or len(purge_dates) != split["usable_observations"]["common_purge"] or len(holdout_dates) != split["usable_observations"]["common_holdout"]:
        raise RuntimeError("Frozen temporal split counts do not match the panel")
    date_to_pos = {d: i for i, d in enumerate(pd.date_range(START, END, freq=BAR))}
    common_pos = np.asarray([date_to_pos[d] for d in common], dtype=int)
    train_pos = np.asarray([date_to_pos[d] for d in train_dates], dtype=int)
    hold_pos = np.asarray([date_to_pos[d] for d in holdout_dates], dtype=int)
    if not len(train_pos) or not len(hold_pos):
        raise RuntimeError("Empty training or holdout rows")

    out_root.mkdir(parents=True, exist_ok=True)
    seed_checkpoint_path = out_root / "primary_seed_checkpoints.jsonl"
    seed_checkpoint = _load_checkpoint(seed_checkpoint_path, run_tag, "primary")
    all_oof, all_resid = [], []
    hold_spot, hold_mlp, hold_qml = [], [], []
    primary_by_asset = {}
    elapsed = {"spot_oof_seconds": 0.0, "spot_final_seconds": 0.0, "qml_mlp_seconds": 0.0}
    for asset in ASSETS:
        x0df, xddf = features[asset]
        x0, xd = x0df.to_numpy(float), xddf.to_numpy(float)
        yraw, y = targets[asset]
        dti = pd.DatetimeIndex(panels[asset].period_start_utc)
        loc = {d: i for i, d in enumerate(dti)}
        train_i = np.asarray([loc[d] for d in train_dates], dtype=int)
        hold_i = np.asarray([loc[d] for d in holdout_dates], dtype=int)
        oof_records = []
        family = base_configs[asset]["family"]
        model_cfg = base_configs[asset]["config"]
        for fold in split["oof_schedule"]["folds"]:
            vf, vl = pd.Timestamp(fold["validation_first"]), pd.Timestamp(fold["validation_last_inclusive"])
            tl = pd.Timestamp(fold["training_last_inclusive"])
            tr = np.asarray([loc[d] for d in train_dates[train_dates <= tl]], dtype=int)
            va_dates = train_dates[(train_dates >= vf) & (train_dates <= vl)]
            va = np.asarray([loc[d] for d in va_dates], dtype=int)
            if len(tr) != fold["training_rows"] or len(va) != fold["validation_rows"]:
                raise RuntimeError(f"Frozen OOF fold count mismatch: {asset}/{fold['fold']}")
            if dti[tr[-1]] + TARGET_END > va_dates[0] + BAR:
                raise RuntimeError(f"OOF target purge failure: {asset}/{fold['fold']}")
            seed = stable_seed(BASE_SEED, asset, family, "phase9_oof", fold["fold"])
            t0 = time.perf_counter()
            with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pred, fit_seconds, fit_warnings = fit_predict(family, model_cfg, x0[tr], y[tr], x0[va], seed=seed)
            elapsed["spot_oof_seconds"] += fit_seconds
            if not np.isfinite(pred).all():
                raise RuntimeError(f"Nonfinite SpotOnly OOF prediction: {asset}/{fold['fold']}")
            for idx, value in zip(va, pred):
                oof_records.append({"asset": asset, "fold": fold["fold"], "date": dti[idx],
                                    "target_raw": float(yraw[idx]), "target_transformed": float(y[idx]),
                                    "spotonly_oof_prediction": float(value), "residual": float(y[idx] - value),
                                    "spotonly_seed": int(seed)})
            print(f"{run_tag} SpotOnly OOF {asset} {fold['fold']}: train={len(tr)} validation={len(va)}", flush=True)
        oof_records.sort(key=lambda r: r["date"])
        if len({r["date"] for r in oof_records}) != len(oof_records):
            raise RuntimeError(f"Duplicate OOF predictions: {asset}")
        oof_frame = pd.DataFrame(oof_records)
        if len(oof_frame) < 100:
            raise RuntimeError(f"Insufficient chronological OOF residuals: {asset}")
        oof_pos = np.asarray([loc[d] for d in pd.DatetimeIndex(oof_frame.date)], dtype=int)
        residual_x = xd[oof_pos]
        residual_y = oof_frame.residual.to_numpy(float)

        seed_spot = stable_seed(BASE_SEED, asset, family, "phase9_final")
        t0 = time.perf_counter()
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            p0, spot_seconds, spot_warnings = fit_predict(family, model_cfg, x0[train_i], y[train_i], x0[hold_i], seed=seed_spot)
        elapsed["spot_final_seconds"] += spot_seconds
        if not np.isfinite(p0).all():
            raise RuntimeError(f"Nonfinite holdout SpotOnly prediction: {asset}")

        cfg = configs[asset]
        t0 = time.perf_counter()
        qmat, mmat, qmeta, mmeta = _fit_seed_models(asset, residual_x, residual_y, xd[hold_i], cfg,
                                                      run_tag, "primary", seed_checkpoint_path, seed_checkpoint)
        elapsed["qml_mlp_seconds"] += time.perf_counter() - t0
        qensemble, mens = qmat.mean(axis=0), mmat.mean(axis=0)
        yt, yr, dates = y[hold_i], yraw[hold_i], dti[hold_i]
        p_q, p_m = p0 + qensemble, p0 + mens
        dq = np.abs(yt - p_m) - np.abs(yt - p_q)
        di = np.abs(yt - p0) - np.abs(yt - p_q)
        seed_delta = [float(np.mean(np.abs(yt - (p0 + mmat[i])) - np.abs(yt - (p0 + qmat[i])))) for i in range(5)]
        boot_q = bootstrap_delta(dates, dq, asset, "delta_q")
        boot_i = bootstrap_delta(dates, di, asset, "delta_i")
        primary_by_asset[asset] = {
            "asset": asset, "venue": "Bybit", "n_holdout": int(len(dates)),
            "first_holdout_origin_utc": dates.min().isoformat(), "last_holdout_origin_utc": dates.max().isoformat(),
            "spotonly_model_family": family, "spotonly_model_config": model_cfg, "spotonly_seed": int(seed_spot),
            "spotonly_mae": metrics(yr, yt, p0), "spotonly_plus_mlp_mae": metrics(yr, yt, p_m),
            "spotonly_plus_qml_mae": metrics(yr, yt, p_q),
            "delta_q_positive_favors_qml": float(dq.mean()), "delta_i_positive_favors_incremental_information": float(di.mean()),
            "delta_q_bootstrap": boot_q, "delta_i_bootstrap": boot_i,
            "seed_deltas_qml_vs_mlp": seed_delta, "seeds_favoring_qml": int(sum(x > 0 for x in seed_delta)),
            "seed_ensemble": "mean of five frozen residual predictions added to the common SpotOnly prediction",
            "qml_config": cfg, "qml_fit_metadata": qmeta, "matched_mlp_fit_metadata": mmeta,
            "oof_residual_rows": int(len(oof_frame)), "oof_first": oof_frame.date.min().isoformat(),
            "oof_last": oof_frame.date.max().isoformat(),
            "delta_q_hac": _hac_loss_delta(dq), "delta_i_hac": _hac_loss_delta(di),
        }
        for r in oof_records:
            all_oof.append(r)
            ix = loc[r["date"]]
            rec = {"asset": asset, "fold": r["fold"], "date": r["date"],
                   "target_raw": r["target_raw"], "target_transformed": r["target_transformed"],
                   "spotonly_oof_prediction": r["spotonly_oof_prediction"], "residual": r["residual"]}
            rec.update({col: float(xd[ix, j]) for j, col in enumerate(xddf.columns)})
            all_resid.append(rec)
        for j, d in enumerate(dates):
            base = {"asset": asset, "date": d, "target_raw": float(yr[j]), "target_transformed": float(yt[j]),
                    "spotonly_prediction": float(p0[j])}
            hold_spot.append(base)
            mlp_row = base | {f"mlp_residual_seed_{k+1}": float(mmat[k, j]) for k in range(5)}
            mlp_row["corrected_prediction"] = float(p_m[j])
            hold_mlp.append(mlp_row)
            qml_row = base | {f"qml_residual_seed_{k+1}": float(qmat[k, j]) for k in range(5)}
            qml_row["corrected_prediction"] = float(p_q[j])
            hold_qml.append(qml_row)
        print(f"{run_tag} primary fits complete for {asset}; holdout dates={len(dates)}", flush=True)

    write_parquet(out_root / "spot_oof_predictions.parquet", _df_for_output(all_oof))
    write_parquet(out_root / "residual_training_data.parquet", _df_for_output(all_resid))
    write_parquet(out_root / "holdout_spotonly_predictions.parquet", _df_for_output(hold_spot))
    write_parquet(out_root / "holdout_mlp_corrected_predictions.parquet", _df_for_output(hold_mlp))
    write_parquet(out_root / "holdout_qml_corrected_predictions.parquet", _df_for_output(hold_qml))
    return {"by_asset": primary_by_asset, "predictions": {"spotonly": hold_spot, "mlp": hold_mlp, "qml": hold_qml},
            "oof_rows": all_oof, "residual_rows": all_resid, "timing": elapsed, "freeze_hash": sha256(OUT / "pre_model_freeze_manifest.json")}


def _hac_loss_delta(delta: np.ndarray, max_lag: int = 6) -> dict:
    x = np.asarray(delta, float)
    n = len(x)
    mu = float(x.mean())
    z = x - mu
    var = float(np.dot(z, z) / n)
    for lag in range(1, min(max_lag, n - 1) + 1):
        gamma = float(np.dot(z[lag:], z[:-lag]) / n)
        var += 2.0 * (1.0 - lag / (max_lag + 1)) * gamma
    se = float(np.sqrt(max(0.0, var) / n))
    stat = mu / se if se > 0 else 0.0
    p = float(2.0 * norm.sf(abs(stat)))
    return {"mean_loss_delta": mu, "hac_se": se, "hac_lags": max_lag, "z_statistic": float(stat), "p_two_sided_normal": p}


def build_placebo(run_tag: str, out_root: Path) -> dict:
    verify_frozen_inputs()
    split = json.loads((OUT / "temporal_split.json").read_text(encoding="utf-8"))
    registry = json.loads((OUT / "feature_registry_bybit_frozen.json").read_text(encoding="utf-8"))
    configs = {a: json.loads((OUT / f"frozen_{a.lower()}_{'hybrid4q' if a == 'BTC' else 'qnn6q'}.json").read_text(encoding="utf-8")) for a in ASSETS}
    base_configs = json.loads((OUT / "matched_classical_models.json").read_text(encoding="utf-8"))["spot_only"]
    cut = pd.Timestamp(split["T_cut"])
    checkpoint_path = out_root / "placebo_seed_checkpoints.jsonl"
    checkpoint = _load_checkpoint(checkpoint_path, run_tag, "placebo")
    results, all_rows = {}, []
    for asset in ASSETS:
        p = pd.read_parquet(OUT / f"bybit_{asset.lower()}_4h_panel.parquet")
        p["period_start_utc"] = pd.to_datetime(p.period_start_utc, utc=True)
        x0df, xddf, _, g = make_features(p, registry)
        x0, xd = x0df.to_numpy(float), xddf.to_numpy(float)
        dates = pd.DatetimeIndex(g.period_start_utc)
        loc = {d: i for i, d in enumerate(dates)}
        spot_amihud = pd.to_numeric(g.spot_amihud, errors="coerce").to_numpy(float)
        past_raw = np.full(len(g), np.nan)
        for i in range(6, len(g)):
            w = spot_amihud[i - 6:i]
            if np.isfinite(w).all():
                past_raw[i] = float(np.mean(w))
        past_y = np.log1p(past_raw)
        x0_past = np.full_like(x0, np.nan)
        x0_past[7:] = x0[:-7]
        xd_future = np.full_like(xd, np.nan)
        xd_future[:-12] = xd[12:]
        date_ns = dates.asi8
        common_calendar_mask = (dates >= START) & (dates <= END)
        base_valid = np.isfinite(x0_past).all(axis=1) & np.isfinite(past_y) & np.isfinite(xd_future).all(axis=1)
        train_mask = common_calendar_mask & base_valid & ((dates + pd.Timedelta(hours=52)) <= cut)
        hold_mask = common_calendar_mask & base_valid & (dates >= cut + pd.Timedelta(hours=24)) & (dates <= END - pd.Timedelta(hours=48))
        train_i = np.flatnonzero(train_mask)
        hold_i = np.flatnonzero(hold_mask)
        if len(train_i) < 250 or len(hold_i) < 100:
            raise RuntimeError(f"Temporal placebo has insufficient complete observations for {asset}: train={len(train_i)}, holdout={len(hold_i)}")
        train_dates = dates[train_i]
        warmup = int(np.ceil(0.50 * len(train_i)))
        remaining = np.arange(warmup, len(train_i))
        blocks = [b for b in np.array_split(remaining, 5) if len(b)]
        oof_x, oof_resid = [], []
        family = base_configs[asset]["family"]
        model_cfg = base_configs[asset]["config"]
        for j, block in enumerate(blocks, 1):
            va = train_i[block]
            first_date = dates[va[0]]
            # The lead-12 derivative feature's interval closes 52 hours after the anchor.
            tr = train_i[dates[train_i] + pd.Timedelta(hours=52) <= first_date]
            if len(tr) < 100:
                continue
            seed = stable_seed(BASE_SEED, asset, family, "phase9a_placebo_oof", j)
            with threadpool_limits(limits=1):
                pva, _, _ = fit_predict(family, model_cfg, x0_past[tr], past_y[tr], x0_past[va], seed=seed)
            oof_x.extend(xd_future[va])
            oof_resid.extend((past_y[va] - pva).tolist())
            print(f"{run_tag} placebo SpotOnly OOF {asset} fold={j}: train={len(tr)} validation={len(va)}", flush=True)
        xtrain, ytrain = np.asarray(oof_x, float), np.asarray(oof_resid, float)
        if len(ytrain) < 100:
            raise RuntimeError(f"Insufficient placebo OOF residuals for {asset}")
        seed_spot = stable_seed(BASE_SEED, asset, family, "phase9a_placebo_final")
        with threadpool_limits(limits=1):
            p0, _, _ = fit_predict(family, model_cfg, x0_past[train_i], past_y[train_i], x0_past[hold_i], seed=seed_spot)
        qmat, mmat, _, _ = _fit_seed_models(asset, xtrain, ytrain, xd_future[hold_i], configs[asset],
                                               run_tag, "placebo", checkpoint_path, checkpoint)
        ytest = past_y[hold_i]
        qcorr, mcorr = p0 + qmat.mean(axis=0), p0 + mmat.mean(axis=0)
        delta_q = np.abs(ytest - mcorr) - np.abs(ytest - qcorr)
        delta_placebo = np.abs(ytest - p0) - np.abs(ytest - qcorr)
        dates_test = dates[hold_i]
        boot = bootstrap_delta(dates_test, delta_placebo, asset, "delta_placebo")
        boot_q = bootstrap_delta(dates_test, delta_q, asset, "delta_placebo_qml_mlp")
        results[asset] = {
            "n_holdout": int(len(hold_i)), "first_anchor": dates_test[0].isoformat(), "last_anchor": dates_test[-1].isoformat(),
            "past_target": "log1p(mean spot Amihud at t-6 through t-1)",
            "baseline_information": "SpotOnly features through t-7; same frozen family and hyperparameters, refit on the placebo past target.",
            "impossible_derivative_information": "Frozen derivatives feature vector at t+12, containing source history t+7 through t+12.",
            "spotonly_mae": float(np.mean(np.abs(ytest - p0))),
            "spotonly_plus_qml_mae": float(np.mean(np.abs(ytest - qcorr))),
            "spotonly_plus_mlp_mae": float(np.mean(np.abs(ytest - mcorr))),
            "delta_placebo_positive_warns": float(delta_placebo.mean()),
            "delta_qml_mlp_positive_favors_qml": float(delta_q.mean()),
            "delta_placebo_bootstrap": boot, "delta_placebo_qml_mlp_bootstrap": boot_q,
            "seed_deltas_qml_vs_mlp": [float(np.mean(np.abs(ytest - (p0 + mmat[i])) - np.abs(ytest - (p0 + qmat[i])))) for i in range(5)],
            "holdout_origin_dates": [d.isoformat() for d in dates_test],
            "delta_placebo_by_date": delta_placebo.tolist(),
        }
        for i, idx in enumerate(hold_i):
            all_rows.append({"asset": asset, "date": dates[idx], "past_target_transformed": float(ytest[i]),
                             "spotonly_prediction": float(p0[i]), "qml_corrected_prediction": float(qcorr[i]),
                             "mlp_corrected_prediction": float(mcorr[i]), "delta_placebo": float(delta_placebo[i])})
        print(f"{run_tag} placebo complete {asset}; holdout dates={len(hold_i)}", flush=True)
    raw_p = [results[a]["delta_placebo_bootstrap"]["42"]["p_two_sided"] for a in ASSETS]
    adj_p = holm(raw_p)
    for asset, adj in zip(ASSETS, adj_p):
        r = results[asset]
        r["delta_placebo_holm_p_block42"] = float(adj)
        ci = r["delta_placebo_bootstrap"]["42"]["ci95"]
        r["material_placebo_warning"] = bool(r["delta_placebo_positive_warns"] > 0 and ci[0] > 0 and adj < .05)
    payload = {"status": "COMPLETE", "by_asset": results,
               "familywise_rule": "Holm across BTC and ETH for positive impossible-time QML correction, based on paired moving-block bootstrap with 42 four-hour blocks.",
               "interpretation": "A supported positive correction from derivatives observed after an already-realized target is a temporal-confounding warning, not predictive evidence."}
    write_parquet(out_root / "negative_control_predictions.parquet", _df_for_output(all_rows))
    return payload


def save_results(result: dict, placebo: dict, run_tag: str, out_root: Path) -> dict:
    by = result["by_asset"]
    qps = [by[a]["delta_q_bootstrap"]["42"]["p_two_sided"] for a in ASSETS]
    ips = [by[a]["delta_i_bootstrap"]["42"]["p_two_sided"] for a in ASSETS]
    qholm, iholm = holm(qps), holm(ips)
    for a, qh, ih in zip(ASSETS, qholm, iholm):
        by[a]["delta_q_holm_p_block42"] = float(qh)
        by[a]["delta_i_holm_p_block42"] = float(ih)
    delta_q = {a: {"mean": by[a]["delta_q_positive_favors_qml"], "bootstrap": by[a]["delta_q_bootstrap"],
                   "holm_p_block42": by[a]["delta_q_holm_p_block42"], "hac": by[a]["delta_q_hac"]} for a in ASSETS}
    delta_i = {a: {"mean": by[a]["delta_i_positive_favors_incremental_information"], "bootstrap": by[a]["delta_i_bootstrap"],
                   "holm_p_block42": by[a]["delta_i_holm_p_block42"], "hac": by[a]["delta_i_hac"]} for a in ASSETS}
    dump(out_root / "primary_metrics.json", {"run_tag": run_tag, "primary_metric": "transformed-target MAE on frozen log1p future 24h Amihud",
          "by_asset": {a: {k: v for k, v in by[a].items() if k not in ["delta_q_bootstrap", "delta_i_bootstrap"]} | {
              "delta_q_bootstrap": by[a]["delta_q_bootstrap"], "delta_i_bootstrap": by[a]["delta_i_bootstrap"]} for a in ASSETS},
          "holdout_dates_identical_for_SpotOnly_MLP_QML": True, "ensemble_rule": "mean of five frozen seed residual predictions"})
    dump(out_root / "delta_q_results.json", {"definition": "MAE(SpotOnly+matched MLP correction)-MAE(SpotOnly+QML correction); positive favors QML.", "by_asset": delta_q})
    dump(out_root / "delta_i_results.json", {"definition": "MAE(SpotOnly)-MAE(SpotOnly+QML residual correction); positive favors incremental predictive information.", "by_asset": delta_i})
    dump(out_root / "bootstrap_42.json", {"block_length_4h": 42, "by_asset": {a: {"delta_q": by[a]["delta_q_bootstrap"]["42"], "delta_i": by[a]["delta_i_bootstrap"]["42"]} for a in ASSETS}})
    dump(out_root / "bootstrap_84.json", {"block_length_4h": 84, "by_asset": {a: {"delta_q": by[a]["delta_q_bootstrap"]["84"], "delta_i": by[a]["delta_i_bootstrap"]["84"]} for a in ASSETS}})
    dump(out_root / "primary_holm.json", {"correction": "Holm across exactly the two primary asset hypotheses, separately for Delta_Q and Delta_I; primary p-values use 42-bar paired moving-block bootstrap.",
          "delta_q": [{"asset": a, "raw_p": qps[i], "holm_p": qholm[i]} for i, a in enumerate(ASSETS)],
          "delta_i": [{"asset": a, "raw_p": ips[i], "holm_p": iholm[i]} for i, a in enumerate(ASSETS)]})
    dump(out_root / "negative_controls.json", placebo)
    decisions = {}
    for asset in ASSETS:
        r = by[asset]
        qci = r["delta_q_bootstrap"]["42"]["ci95"]
        ici = r["delta_i_bootstrap"]["42"]["ci95"]
        q_support = r["delta_q_positive_favors_qml"] > 0 and qci[0] > 0 and r["delta_q_holm_p_block42"] < .05
        i_support = r["delta_i_positive_favors_incremental_information"] > 0 and ici[0] > 0 and r["delta_i_holm_p_block42"] < .05
        seed_pass = r["seeds_favoring_qml"] >= 4
        placebo_pass = not placebo["by_asset"][asset]["material_placebo_warning"]
        passed = bool(q_support and i_support and seed_pass and placebo_pass)
        decisions[asset] = {"status": "PASS" if passed else "FAIL", "primary_model": "Hybrid QNN 4 qubits" if asset == "BTC" else "QNN 6 qubits",
            "delta_q_supported": q_support, "delta_i_supported": i_support, "seeds_favoring_qml": r["seeds_favoring_qml"],
            "seed_requirement_pass": seed_pass, "material_placebo_warning": not placebo_pass,
            "requirements": {"delta_q_positive_and_42block_ci_positive_and_holm_lt_0_05": q_support,
                             "delta_i_positive_and_42block_ci_positive_and_holm_lt_0_05": i_support,
                             "at_least_four_of_five_seeds_favor_qml": seed_pass,
                             "no_material_impossible_time_placebo": placebo_pass}}
        dump(out_root / f"{asset.lower()}_decision.json", decisions[asset])
    if all(d["status"] == "PASS" for d in decisions.values()):
        paper = "EXTERNALLY_REPLICATED_QML_INCREMENTAL_VALUE"
    elif sum(d["status"] == "PASS" for d in decisions.values()) == 1:
        paper = "ASSET_SPECIFIC_EXTERNAL_QML_SIGNAL"
    elif any(by[a]["delta_q_positive_favors_qml"] > 0 and not (by[a]["delta_i_bootstrap"]["42"]["ci95"][0] > 0) for a in ASSETS):
        paper = "ALGORITHMIC_QML_GAIN_WITHOUT_INCREMENTAL_INFORMATION"
    else:
        paper = "NO_EXTERNAL_QML_REPLICATION"
    dump(out_root / "paper_level_decision.json", {"classification": paper, "by_asset": decisions,
          "quantum_hardware_advantage": "NOT_TESTED / NOT CLAIMED"})
    dump(out_root / "model_run_timing.json", {"wall_seconds": result["timing"], "primary_total_wall_seconds": float(sum(result["timing"].values())),
          "note": "Fit wall time only; excludes raw download and 4h panel construction."})
    return {"by_asset": by, "decisions": decisions, "paper": paper}


def run(run_tag: str = "primary", out_root: Path | None = None) -> dict:
    if out_root is None:
        out_root = OUT
    t0 = time.perf_counter()
    result = build_primary(run_tag, out_root)
    placebo = build_placebo(run_tag, out_root)
    final = save_results(result, placebo, run_tag, out_root)
    final["elapsed_wall_seconds"] = time.perf_counter() - t0
    if out_root == OUT:
        summary = _summary_markdown(final, result, placebo)
        (OUT / "research_summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"run_tag": run_tag, "paper_classification": final["paper"],
                      "by_asset": {a: {"delta_q": final["by_asset"][a]["delta_q_positive_favors_qml"],
                                       "delta_i": final["by_asset"][a]["delta_i_positive_favors_incremental_information"],
                                       "decision": final["decisions"][a]["status"]} for a in ASSETS},
                      "elapsed_wall_seconds": final["elapsed_wall_seconds"]}, indent=2), flush=True)
    return final


def _summary_markdown(final: dict, result: dict, placebo: dict) -> str:
    b = final["by_asset"]
    lines = [
        "# Phase 9A Bybit 24-month external QML replication",
        "",
        f"The preregistered window was {START.date()} through {END.date()} UTC under the pre-result sample amendment. The common calendar split and all model settings were frozen before model fitting.",
        "",
        "The target is the arithmetic mean of the next six 4-hour Amihud observations, with the frozen `1e6 × |4h log return| / USDT quote volume` definition and `log1p` primary transform. Bybit BTC and ETH were evaluated separately on identical complete common dates. The holdout was not used for feature or model selection.",
        "",
        "BTC used the frozen Hybrid QNN with four qubits; ETH used the frozen QNN with six qubits. Both were compared with their frozen five-seed matched MLPs. The economic endpoint is the complete corrected forecast versus SpotOnly; QML versus MLP is a separate algorithmic endpoint.",
        "",
    ]
    for a in ASSETS:
        r = b[a]
        p = placebo["by_asset"][a]
        lines += [f"For {a}, SpotOnly transformed MAE was {r['spotonly_mae']['transformed_mae']:.8g}; SpotOnly plus MLP was {r['spotonly_plus_mlp_mae']['transformed_mae']:.8g}; SpotOnly plus QML was {r['spotonly_plus_qml_mae']['transformed_mae']:.8g}. Delta_Q was {r['delta_q_positive_favors_qml']:.8g}, and Delta_I was {r['delta_i_positive_favors_incremental_information']:.8g}. The 42-bar bootstrap intervals were Delta_Q {r['delta_q_bootstrap']['42']['ci95']} and Delta_I {r['delta_i_bootstrap']['42']['ci95']}; the 84-bar intervals were {r['delta_q_bootstrap']['84']['ci95']} and {r['delta_i_bootstrap']['84']['ci95']}. {r['seeds_favoring_qml']}/5 frozen paired seeds favored QML. Holm-adjusted p-values were {r['delta_q_holm_p_block42']:.5g} for Delta_Q and {r['delta_i_holm_p_block42']:.5g} for Delta_I. Decision: {final['decisions'][a]['status']}.",
                     f"The matched impossible-time placebo Delta was {p['delta_placebo_positive_warns']:.8g}, with 42-bar interval {p['delta_placebo_bootstrap']['42']['ci95']} and Holm p={p['delta_placebo_holm_p_block42']:.5g}; material warning: {p['material_placebo_warning']}.", ""]
    lines += [f"Paper-level classification: {final['paper']}.",
              "",
              "The sample amendment was made before panel construction and before any forecast result was available. The study can support only venue-specific predictive statements under this frozen design. It cannot establish structural causality, general superiority of quantum learning, or computational/hardware quantum advantage. Prior Phase 7/8 generalization failures remain part of the evidence base.",
              ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default="primary")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    out_root = Path(args.output_dir).resolve() if args.output_dir else OUT
    run(args.run_tag, out_root)


if __name__ == "__main__":
    main()
