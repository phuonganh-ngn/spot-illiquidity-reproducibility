from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "phase9_external_qml_replication"
sys.path.insert(0, str(ROOT / "src" / "targeted_24h_amihud"))
from qml_models import fit_qnn, _preprocess
from classical_models import fit_predict, stable_seed

PANEL = OUT / "external_panel.csv.gz"
ASSETS = ("BTC", "ETH")
SPOT = ["spot_return", "spot_realized_volatility", "spot_downside_semivariance",
        "spot_amihud", "spot_taker_imbalance", "spot_quote_volume_growth"]
DER = ["funding_rate_state", "oi_log_change", "futures_taker_imbalance", "basis_mean"]
STATIC_DER = ["funding_change", "basis_change", "oi_shock_30d_robust", "flow_shock_30d_robust",
              "funding_x_oi_shock", "oi_shock_x_flow_shock", "funding_x_basis", "flow_x_basis"]
CAL = [*[f"utc_slot_{i}" for i in range(1, 6)], *[f"weekday_{i}" for i in range(1, 7)], "funding_schedule_window"]
TARGET_RAW = "future_amihud_24h_mean"
TARGET = "future_amihud_24h_log1p"
BOOT_REPS = 1000
BASE_SEED = 20261007
PURGE = 6


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_write(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def json_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def config_for(asset: str, short: bool = False) -> dict:
    name = "frozen_btc_hybrid4q.json" if asset == "BTC" else "frozen_eth_qnn6q.json"
    d = json.loads((OUT / name).read_text())
    if short:
        return {"asset": asset, "family": d["frozen_primary_model"], "qubits": d["qubits"],
                "hybrid": asset == "BTC", "pca_components": d["pca_components"], "epochs": d["epochs"],
                "batch_size": d["batch_size"], "learning_rate": d["learning_rate"], "seeds": d["seeds"]}
    return d


def get_arrays(panel: pd.DataFrame, asset: str):
    g = panel[panel.asset == asset].sort_values("period_start_utc").reset_index(drop=True).copy()
    x0 = pd.DataFrame(index=g.index)
    for lag in range(5, -1, -1):
        for col in SPOT:
            x0[f"{col}__lag{lag}"] = g[col].shift(lag)
    for col in CAL:
        x0[col] = g[col]
    xd = pd.DataFrame(index=g.index)
    for lag in range(5, -1, -1):
        for col in DER:
            xd[f"{col}__lag{lag}"] = g[col].shift(lag)
    for col in STATIC_DER + CAL:
        xd[col] = g[col]
    raw = pd.to_numeric(g[TARGET_RAW], errors="coerce").to_numpy(float)
    y = np.log1p(raw)
    valid = (np.isfinite(raw) & np.isfinite(x0.to_numpy(float)).all(axis=1) &
             np.isfinite(xd.to_numpy(float)).all(axis=1))
    return g, x0.to_numpy(float), xd.to_numpy(float), raw, y, valid


def check_frozen_inputs() -> dict:
    frozen = json.loads((OUT / "frozen_artifact_hashes.json").read_text())
    if frozen.get("status") != "FROZEN_BEFORE_EXTERNAL_FORECASTS":
        raise RuntimeError("Protocol/data freeze is absent")
    errors = []
    for rel, expected in frozen["files"].items():
        p = OUT / rel
        if not p.exists() or sha256(p) != expected:
            errors.append(rel)
    protocol_hash = sha256(OUT / "phase9_protocol.json")
    recorded = (OUT / "phase9_protocol.sha256").read_text().split()[0]
    if protocol_hash != recorded:
        errors.append("phase9_protocol.sha256")
    protocol = json.loads((OUT / "phase9_protocol.json").read_text())
    code_manifest_path = OUT / "execution_code_manifest.json"
    if not code_manifest_path.exists() or sha256(code_manifest_path) != protocol.get("execution_code_manifest_sha256"):
        errors.append("execution_code_manifest.json")
    elif not errors:
        code_manifest = json.loads(code_manifest_path.read_text())
        for rel, expected in code_manifest.get("files", {}).items():
            if not (ROOT / rel).exists() or sha256(ROOT / rel) != expected:
                errors.append("code:" + rel)
    prior = json.loads((OUT / "prior_artifact_hashes_before.json").read_text())
    prior_errors = []
    for rel, expected in prior.items():
        p = ROOT / rel
        if not p.exists() or sha256(p) != expected:
            prior_errors.append(rel)
    if errors or prior_errors:
        raise RuntimeError(f"Frozen-file hash failure={errors}; prior artifact changes={prior_errors[:10]}")
    split = json.loads((OUT / "temporal_split.json").read_text())
    if not split.get("external_holdout_untouched_before_freeze"):
        raise RuntimeError("Holdout freeze gate failed")
    return {"protocol_sha256": protocol_hash, "frozen_files_verified": len(frozen["files"]),
            "prior_files_verified": len(prior), "split": split}


def make_spot_oof(asset: str, g: pd.DataFrame, x0: np.ndarray, y: np.ndarray,
                  train_n: int, split: dict, family: str, cfg: dict, run_tag: str):
    train_dates = pd.DatetimeIndex(g.period_start_utc.iloc[:train_n])
    folds = split["oof_folds"]
    records = []
    for fold in folds:
        a = int(fold["validation_index_start"])
        b = int(fold["validation_index_end_exclusive"])
        fit_end = a - PURGE
        tr = np.arange(0, fit_end, dtype=int)
        va = np.arange(a, b, dtype=int)
        if not len(tr) or not len(va) or tr.max() + PURGE >= va.min():
            raise RuntimeError(f"OOF date purge violation {asset}/{fold['fold']}")
        if train_dates[tr].max() + pd.Timedelta(hours=28) > train_dates[va].min() + pd.Timedelta(hours=4):
            # Label end-exclusive is base timestamp + 28h. Validation begins
            # with the next target at base + 4h; equality is safe.
            raise RuntimeError(f"OOF labels overlap validation target period {asset}/{fold['fold']}")
        seed = stable_seed(BASE_SEED, asset, family, "phase9_oof", fold["fold"])
        with threadpool_limits(limits=1):
            pred, elapsed, warns = fit_predict(family, cfg, x0[tr], y[tr], x0[va], seed=seed)
        if not np.isfinite(pred).all():
            raise RuntimeError(f"Nonfinite OOF prediction {asset}/{fold['fold']}")
        for idx, value in zip(va, pred):
            records.append({"asset": asset, "fold": fold["fold"], "date": train_dates[idx].isoformat(),
                            "panel_index": int(idx), "target_transformed": float(y[idx]),
                            "target_raw": float(np.expm1(y[idx])), "oof_spot_prediction_transformed": float(value),
                            "residual_transformed": float(y[idx] - value)})
        print(f"{run_tag} spot OOF {asset} {fold['fold']}: n={len(va)} fit={elapsed:.1f}s", flush=True)
    if len({r["date"] for r in records}) != len(records):
        raise RuntimeError("Duplicate training OOF dates")
    return records


def fit_mlp_residual(xtrain, ytrain, xtest, cfg: dict, seed: int):
    components = min(int(cfg["pca_components"]), xtrain.shape[1], max(1, len(xtrain) - 1))
    ztr, zte, meta = _preprocess(xtrain, xtest, components, seed)
    y = np.asarray(ytrain, float)
    ym = float(y.mean())
    ys = float(y.std(ddof=1))
    if not np.isfinite(ys) or ys <= 0:
        ys = 1.0
    model_cfg = cfg["matched_classical"]
    model = MLPRegressor(hidden_layer_sizes=(int(model_cfg["hidden_width"]),), activation="relu",
                         solver="adam", alpha=float(model_cfg["alpha"]),
                         learning_rate_init=float(model_cfg["learning_rate_init"]),
                         max_iter=int(model_cfg["max_iter"]), tol=float(model_cfg["tol"]),
                         early_stopping=bool(model_cfg["early_stopping"]), random_state=seed)
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        model.fit(ztr, (y - ym) / ys)
        pred = model.predict(zte) * ys + ym
    return np.asarray(pred, float), meta | {"family": "one-hidden-layer MLP", "hidden_width": int(model_cfg["hidden_width"]),
            "pca_components": components, "max_iter": int(model_cfg["max_iter"]), "n_iter": int(model.n_iter_),
            "warning_count": len(ws), "seed": int(seed)}


def fit_strong_classical(asset: str, xtrain, ytrain, xtest, seed: int):
    registry = json.loads((OUT / "matched_classical_models.json").read_text())
    item = registry["strongest_previously_frozen_classical_residual_learner"][asset]
    family, cfg = item["family"], item["config"]
    if family == "ElasticNet":
        y = np.asarray(ytrain, float)
        ym, ys = float(y.mean()), float(y.std(ddof=1))
        if not np.isfinite(ys) or ys <= 0:
            ys = 1.0
        model = make_pipeline(RobustScaler(), ElasticNet(alpha=float(cfg["alpha"]), l1_ratio=float(cfg["l1_ratio"]),
                                max_iter=int(cfg["max_iter"]), tol=float(cfg["tol"]), random_state=seed,
                                selection="cyclic"))
        with threadpool_limits(limits=1):
            model.fit(xtrain, (y - ym) / ys)
            pred = model.predict(xtest) * ys + ym
        return np.asarray(pred, float), {"family": family, "config": cfg, "seed": seed}
    with threadpool_limits(limits=1):
        pred, elapsed, warns = fit_predict(family, cfg, xtrain, np.asarray(ytrain, float), xtest, seed=seed)
    return np.asarray(pred, float), {"family": family, "config": cfg, "seed": seed,
                                   "fit_seconds": elapsed, "warnings": warns}


def fit_qml_one(asset: str, xtrain, ytrain, xtest, seed: int, cfg: dict):
    return fit_qnn(xtrain, ytrain, xtest, int(cfg["qubits"]), int(seed), epochs=int(cfg["epochs"]),
                   batch_size=int(cfg["batch_size"]), learning_rate=float(cfg["learning_rate"]),
                   hybrid=bool(asset == "BTC"), encoder_components=int(cfg["pca_components"]))


def moving_ids(dates, block_len: int, n: int, rng: np.random.Generator) -> np.ndarray:
    d = pd.to_datetime(pd.Series(dates), utc=True)
    unit = d.dtype.unit
    ticks_per_second = {"s": 1, "ms": 1000, "us": 10**6, "ns": 10**9}[unit]
    step = 4 * 3600 * ticks_per_second
    iv = d.astype("int64").to_numpy() // step
    first, last = int(iv.min()), int(iv.max())
    starts = np.arange(first, max(first + 1, last - block_len + 2))
    if len(starts) == 0:
        starts = np.array([first])
    out = []
    while len(out) < n:
        s = int(rng.choice(starts))
        out.extend(np.flatnonzero((iv >= s) & (iv < s + block_len)).tolist())
    return np.asarray(out[:n], dtype=int)


def bootstrap_delta(dates, delta, asset: str, endpoint: str) -> dict:
    dates = np.asarray(dates)
    delta = np.asarray(delta, float)
    out = {}
    for block in (42, 84):
        seed = stable_seed(BASE_SEED, asset, endpoint, block)
        rng = np.random.default_rng(seed)
        draws = np.empty(BOOT_REPS)
        for b in range(BOOT_REPS):
            ids = moving_ids(dates, block, len(delta), rng)
            draws[b] = delta[ids].mean()
        n_le = int(np.sum(draws <= 0))
        n_ge = int(np.sum(draws >= 0))
        # Test H0: mean delta = 0 with the centered moving-block bootstrap.
        # Centering is essential: the uncentered fraction of bootstrap draws
        # crossing zero is a sign-stability diagnostic, not a null p-value.
        centered = draws - float(delta.mean())
        p = float((1 + np.sum(np.abs(centered) >= abs(float(delta.mean())))) / (BOOT_REPS + 1))
        out[str(block)] = {"replications": BOOT_REPS, "block_length_4h_intervals": block,
                           "seed": seed, "mean_delta": float(delta.mean()),
                           "ci95": [float(x) for x in np.quantile(draws, [.025, .975])],
                           "p_two_sided_centered_block_bootstrap": p,
                           "p_two_sided": p,
                           "probability_delta_positive": float(np.mean(draws > 0)),
                           "bootstrap_draws_crossing_zero": int(min(n_le, n_ge)),
                           "p_value_method": "plus-one corrected two-sided centered moving-block bootstrap under zero-mean null"}
    return out


def hac_loss_diff(delta: np.ndarray, max_lag: int = 6) -> dict:
    x = np.asarray(delta, float)
    n = len(x)
    mu = float(x.mean())
    z = x - mu
    var = float(np.dot(z, z) / n)
    for lag in range(1, min(max_lag, n - 1) + 1):
        gamma = float(np.dot(z[lag:], z[:-lag]) / n)
        var += 2 * (1 - lag / (max_lag + 1)) * gamma
    se = float(np.sqrt(max(0.0, var) / n))
    stat = mu / se if se > 0 else (math.inf if mu > 0 else (-math.inf if mu < 0 else 0.0))
    return {"mean_loss_delta": mu, "hac_se": se, "hac_lags": max_lag,
            "z_statistic": float(stat), "p_two_sided_normal": float(2 * norm.sf(abs(stat))) if np.isfinite(stat) else 0.0}


def holm(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    out = np.empty(len(p))
    last = 0.0
    for rank, j in enumerate(order):
        last = max(last, min(1.0, (len(p) - rank) * float(p[j])))
        out[j] = last
    return out.tolist()


def metrics(y_raw, y_trans, pred_trans) -> dict:
    yr = np.asarray(y_raw, float)
    yt = np.asarray(y_trans, float)
    pt = np.asarray(pred_trans, float)
    e_t = yt - pt
    pr = np.maximum(np.expm1(pt), 0.0)
    e_r = yr - pr
    return {"n": int(len(yr)), "transformed_mae": float(np.mean(np.abs(e_t))),
            "transformed_rmse": float(np.sqrt(np.mean(e_t**2))),
            "transformed_median_absolute_error": float(np.median(np.abs(e_t))),
            "original_scale_mae": float(np.mean(np.abs(e_r))),
            "original_scale_rmse": float(np.sqrt(np.mean(e_r**2))),
            "original_scale_median_absolute_error": float(np.median(np.abs(e_r)))}


def fixture_validation() -> dict:
    results = []
    rng = np.random.default_rng(9092401)
    cfgs = {a: config_for(a) for a in ASSETS}
    for asset in ASSETS:
        cfg = cfgs[asset]
        dim = max(20, int(cfg["pca_components"]) + 2)
        x = rng.normal(size=(128, dim))
        y = .4 * x[:, 0] + .3 * np.tanh(x[:, 1] * x[:, 2]) + rng.normal(0, .05, len(x))
        train_x, train_y = x[:96], y[:96]
        test_x = x[96:112]
        seed = int(cfg["seeds"][0])
        p1, meta1 = fit_qml_one(asset, train_x, train_y, test_x, seed, cfg)
        p2, meta2 = fit_qml_one(asset, train_x, train_y, test_x, seed, cfg)
        altered = np.vstack([test_x, np.full((1, dim), 1e6)])
        p3, _ = fit_qml_one(asset, train_x, train_y, altered, seed, cfg)
        leak_delta = float(np.max(np.abs(np.asarray(p1) - np.asarray(p3[:len(p1)]))))
        finite = bool(np.isfinite(p1).all() and np.isfinite(p2).all() and np.isfinite(p3).all())
        repeat_delta = float(np.max(np.abs(np.asarray(p1) - np.asarray(p2))))
        seed_fixture = []
        for frozen_seed in cfg["seeds"]:
            pred, _ = fit_qml_one(asset, train_x, train_y, test_x, int(frozen_seed), cfg)
            seed_fixture.append({"seed": int(frozen_seed), "finite_predictions": bool(np.isfinite(pred).all()),
                                 "prediction_mean": float(np.mean(pred)), "prediction_std": float(np.std(pred))})
        results.append({"asset_configuration": asset, "qubits": cfg["qubits"], "finite_loss_and_predictions": finite,
                        "repeat_max_abs_prediction_delta": repeat_delta,
                        "test_distribution_change_max_abs_delta": leak_delta,
                        "training_only_preprocessing_pass": leak_delta < 1e-7,
                        "reproducible_fixed_seed": repeat_delta < 1e-7,
                        "all_five_frozen_seeds_finite": all(r["finite_predictions"] for r in seed_fixture),
                        "five_seed_fixture": seed_fixture,
                        "fixture_type": "synthetic nonlinear interaction regression; fixed train/test split",
                        "metadata": meta1})
    status = "VALID" if all(r["finite_loss_and_predictions"] and r["training_only_preprocessing_pass"] and r["reproducible_fixed_seed"] and r["all_five_frozen_seeds_finite"] for r in results) else "FAIL"
    res = {"status": status, "fixtures": results, "seed": 9092401,
           "external_market_data_used": False, "description": "Deterministic synthetic regression; verifies finite QML fitting, repeatability, and training-only PCA/scaling independence from evaluation input distribution."}
    json_write("qml_fixture_validation.json", res)
    if status != "VALID":
        raise RuntimeError("QML synthetic fixture validation failed")
    return res


def evaluate_main(run_tag: str = "primary", save_outputs: bool = True) -> dict:
    freeze = check_frozen_inputs()
    fixture_path = OUT / "qml_fixture_validation.json"
    if not fixture_path.exists() or json.loads(fixture_path.read_text()).get("status") != "VALID":
        raise RuntimeError("QML fixture gate failed or was not run")
    split = freeze["split"]
    panel = pd.read_csv(PANEL, parse_dates=["period_start_utc"], low_memory=False)
    panel["period_start_utc"] = pd.to_datetime(panel.period_start_utc, utc=True)
    dates_common = None
    arrays = {}
    for asset in ASSETS:
        g, x0, xd, raw, y, valid = get_arrays(panel, asset)
        d = pd.DatetimeIndex(g.period_start_utc[valid])
        dates_common = d if dates_common is None else dates_common.intersection(d)
        arrays[asset] = (g, x0, xd, raw, y, valid)
    dates_common = dates_common.sort_values()
    if len(dates_common) != split["common_valid_rows_before_purge"] or dates_common[0].isoformat() != split["common_start_utc"] or dates_common[-1].isoformat() != split["common_end_utc"]:
        raise RuntimeError("Common valid sample no longer matches frozen split")
    n = len(dates_common)
    boundary = int(np.floor(.70 * n))
    train_n = boundary - PURGE
    if train_n != split["training_rows_used_after_purge"]:
        raise RuntimeError("Training split count mismatch")
    date_pos = {d: i for i, d in enumerate(dates_common)}
    cfgs = {a: config_for(a) for a in ASSETS}
    match = json.loads((OUT / "matched_classical_models.json").read_text())
    all_results = {}
    outputs = {"oof": [], "resid": [], "p0": [], "mlp": [], "qml": [], "strong": []}
    for asset in ASSETS:
        g0, x0all, xdall, rawall, yall, validall = arrays[asset]
        g = g0[g0.period_start_utc.isin(dates_common)].sort_values("period_start_utc").reset_index(drop=True)
        # All common dates are complete for both assets, so the sorted panel rows align exactly.
        rowmap = {d: i for i, d in enumerate(pd.DatetimeIndex(g.period_start_utc))}
        panel_ix = np.asarray([rowmap[d] for d in dates_common], dtype=int)
        x0 = x0all[panel_ix]
        xd = xdall[panel_ix]
        yraw = rawall[panel_ix]
        y = yall[panel_ix]
        if not np.isfinite(x0).all() or not np.isfinite(xd).all() or not np.isfinite(y).all():
            raise RuntimeError(f"Frozen common sample contains nonfinite values for {asset}")
        train_dates = dates_common[:train_n]
        hold_dates = dates_common[boundary:]
        oof_records = make_spot_oof(asset, g, x0, y, train_n, split,
                                    match["spot_only"][asset]["family"], match["spot_only"][asset]["config"], run_tag)
        oof_rows = np.asarray([r["panel_index"] for r in oof_records], int)
        y_res = np.asarray([r["residual_transformed"] for r in oof_records], float)
        dx_train = xd[oof_rows]
        x0_train = x0[:train_n]
        y_train = y[:train_n]
        x0_test = x0[boundary:]
        dx_test = xd[boundary:]
        y_test = y[boundary:]
        raw_test = yraw[boundary:]
        if len(oof_rows) < 100 or len(hold_dates) < 100:
            raise RuntimeError(f"Insufficient training OOF/holdout rows for {asset}")

        spot_seed = stable_seed(BASE_SEED, asset, match["spot_only"][asset]["family"], "phase9_final")
        with threadpool_limits(limits=1):
            p0, spot_elapsed, spot_warnings = fit_predict(match["spot_only"][asset]["family"],
                    match["spot_only"][asset]["config"], x0_train, y_train, x0_test, seed=spot_seed)
        if not np.isfinite(p0).all():
            raise RuntimeError(f"Nonfinite SpotOnly holdout prediction for {asset}")

        mcfg = cfgs[asset]["matched_classical"]
        q_preds, m_preds, q_meta, m_meta = [], [], [], []
        for s in cfgs[asset]["seeds"]:
            q_seed = int(s)
            qp, qm = fit_qml_one(asset, dx_train, y_res, dx_test, q_seed, cfgs[asset])
            mp, mm = fit_mlp_residual(dx_train, y_res, dx_test, cfgs[asset], q_seed)
            q_preds.append(np.asarray(qp, float))
            m_preds.append(np.asarray(mp, float))
            q_meta.append(qm)
            m_meta.append(mm)
            checkpoint = {"run": run_tag, "asset": asset, "seed": q_seed,
                          "qml_prediction_sha256": json_hash(np.asarray(qp, float).tolist()),
                          "mlp_prediction_sha256": json_hash(np.asarray(mp, float).tolist()),
                          "qml_prediction_residual": np.asarray(qp, float).tolist(),
                          "mlp_prediction_residual": np.asarray(mp, float).tolist()}
            ck = OUT / f"{run_tag}_prediction_checkpoints.jsonl"
            with ck.open("a", encoding="utf-8") as f:
                f.write(json.dumps(checkpoint, sort_keys=True, allow_nan=False) + "\n")
                f.flush()
            print(f"{run_tag} residual learners {asset} seed={q_seed} complete", flush=True)
        qmat = np.vstack(q_preds)
        mmat = np.vstack(m_preds)
        qcorr = p0 + qmat.mean(axis=0)
        mcorr = p0 + mmat.mean(axis=0)
        loss0 = np.abs(y_test - p0)
        lossq = np.abs(y_test - qcorr)
        lossm = np.abs(y_test - mcorr)
        d_full = loss0 - lossq
        d_qml_mlp = lossm - lossq
        seed_deltas = [float(np.mean(np.abs(y_test - (p0 + mmat[i])) - np.abs(y_test - (p0 + qmat[i])))) for i in range(5)]

        strong_seed = stable_seed(BASE_SEED, asset, match["strongest_previously_frozen_classical_residual_learner"][asset]["family"], "phase9_strong_residual")
        strong_res, strong_meta = fit_strong_classical(asset, dx_train, y_res, dx_test, strong_seed)
        strong_corr = p0 + strong_res
        lossstrong = np.abs(y_test - strong_corr)

        metric_pack = {
            "asset": asset, "venue": "Bybit", "n_holdout_rows": int(len(hold_dates)),
            "n_unique_4h_dates": int(len(hold_dates)), "first_holdout_date": hold_dates[0].isoformat(),
            "last_holdout_date": hold_dates[-1].isoformat(),
            "spot_only": metrics(raw_test, y_test, p0),
            "spot_only_plus_mlp_residual_correction": metrics(raw_test, y_test, mcorr),
            "spot_only_plus_qml_residual_correction": metrics(raw_test, y_test, qcorr),
            "spot_only_plus_strong_classical_residual_correction": metrics(raw_test, y_test, strong_corr),
            "delta_full_positive_favors_qml": float(np.mean(d_full)),
            "delta_qml_mlp_positive_favors_qml": float(np.mean(d_qml_mlp)),
            "delta_qml_strong_classical_positive_favors_qml": float(np.mean(lossstrong - lossq)),
            "seed_delta_qml_vs_mlp_positive_favors_qml": seed_deltas,
            "seeds_favoring_qml_vs_mlp": int(sum(v > 0 for v in seed_deltas)),
            "seed_ensemble_rule": "mean residual prediction across the exact five frozen seeds, then add to the SpotOnly forecast",
            "spot_only_fit": {"family": match["spot_only"][asset]["family"], "config": match["spot_only"][asset]["config"],
                              "seed": spot_seed, "fit_seconds": spot_elapsed, "warnings": spot_warnings},
            "strong_classical_fit": strong_meta,
            "qml_metadata_by_seed": q_meta, "matched_mlp_metadata_by_seed": m_meta,
            "oof_residual_training_rows": int(len(oof_records)),
            "oof_residual_train_first": min(r["date"] for r in oof_records),
            "oof_residual_train_last": max(r["date"] for r in oof_records),
        }
        metric_pack["bootstrap_delta_full"] = bootstrap_delta(hold_dates, d_full, asset, "delta_full")
        metric_pack["bootstrap_delta_qml_mlp"] = bootstrap_delta(hold_dates, d_qml_mlp, asset, "delta_qml_mlp")
        metric_pack["bootstrap_delta_qml_strong_classical"] = bootstrap_delta(hold_dates, lossstrong - lossq, asset, "delta_qml_strong_classical")
        metric_pack["hac_delta_full"] = hac_loss_diff(d_full)
        metric_pack["hac_delta_qml_mlp"] = hac_loss_diff(d_qml_mlp)
        all_results[asset] = metric_pack
        for rec in oof_records:
            outputs["oof"].append(rec)
            outputs["resid"].append({"asset": asset, "date": rec["date"], "residual_transformed": rec["residual_transformed"],
                                     "target_transformed": rec["target_transformed"], "oof_spot_prediction_transformed": rec["oof_spot_prediction_transformed"]})
        for i, d in enumerate(hold_dates):
            outputs["p0"].append({"asset": asset, "date": d.isoformat(), "target_raw": float(raw_test[i]),
                                  "target_transformed": float(y_test[i]), "spotonly_prediction_transformed": float(p0[i])})
            outputs["mlp"].append({"asset": asset, "date": d.isoformat(), "target_raw": float(raw_test[i]),
                                   "target_transformed": float(y_test[i]), "spotonly_prediction_transformed": float(p0[i]),
                                   "mlp_residual_by_seed": mmat[:, i].tolist(), "corrected_ensemble_prediction": float(mcorr[i])})
            outputs["qml"].append({"asset": asset, "date": d.isoformat(), "target_raw": float(raw_test[i]),
                                   "target_transformed": float(y_test[i]), "spotonly_prediction_transformed": float(p0[i]),
                                   "qml_residual_by_seed": qmat[:, i].tolist(), "corrected_ensemble_prediction": float(qcorr[i])})
            outputs["strong"].append({"asset": asset, "date": d.isoformat(), "target_raw": float(raw_test[i]),
                                      "target_transformed": float(y_test[i]), "spotonly_prediction_transformed": float(p0[i]),
                                      "strong_classical_residual_prediction": float(strong_res[i]),
                                      "corrected_prediction": float(strong_corr[i])})
        print(f"{run_tag} full forecast {asset}: delta_full={metric_pack['delta_full_positive_favors_qml']:.8g}; delta_qml_mlp={metric_pack['delta_qml_mlp_positive_favors_qml']:.8g}", flush=True)

    if save_outputs:
        for key, name in (("oof", "spot_oof_predictions.json"), ("resid", "spot_residual_training_targets.json"),
                          ("p0", "external_spotonly_predictions.json"), ("mlp", "external_mlp_corrected_predictions.json"),
                          ("qml", "external_qml_corrected_predictions.json")):
            json_write(name, {"run": run_tag, "rows": outputs[key]})
        json_write("qml_vs_strong_classical.json", {"status": "COMPLETE", "rows": outputs["strong"],
                     "metrics": {a: r["spot_only_plus_strong_classical_residual_correction"] for a, r in all_results.items()},
                     "delta_qml_minus_strong_classical_positive_favors_qml": {a: r["delta_qml_strong_classical_positive_favors_qml"] for a, r in all_results.items()}})
    primary = {"status": "COMPLETE", "metric": "MAE on frozen T2_log1p target", "by_asset": all_results,
               "holdout_dates_identical_for_SpotOnly_MLP_QML": True, "qnn_seed_aggregation": "mean of five seed-specific residual predictions"}
    if save_outputs:
        json_write("primary_full_forecast_results.json", primary)
        json_write("qml_vs_mlp_results.json", {"by_asset": {a: {k: v for k, v in r.items() if "qml_mlp" in k or k in ["spot_only_plus_mlp_residual_correction", "spot_only_plus_qml_residual_correction", "seed_delta_qml_vs_mlp_positive_favors_qml", "seeds_favoring_qml_vs_mlp"]} for a, r in all_results.items()}})
        json_write("paired_bootstrap_42.json", {"block_length": 42, "by_asset": {a: {"delta_full": r["bootstrap_delta_full"]["42"], "delta_qml_mlp": r["bootstrap_delta_qml_mlp"]["42"], "delta_qml_strong_classical": r["bootstrap_delta_qml_strong_classical"]["42"]} for a, r in all_results.items()}})
        json_write("paired_bootstrap_84.json", {"block_length": 84, "by_asset": {a: {"delta_full": r["bootstrap_delta_full"]["84"], "delta_qml_mlp": r["bootstrap_delta_qml_mlp"]["84"], "delta_qml_strong_classical": r["bootstrap_delta_qml_strong_classical"]["84"]} for a, r in all_results.items()}})
        json_write("loss_differential_tests.json", {"method": "paired transformed absolute-loss differential, Bartlett/Newey-West HAC with six 4-hour lags; supporting evidence only", "by_asset": {a: {"delta_full": r["hac_delta_full"], "delta_qml_mlp": r["hac_delta_qml_mlp"]} for a, r in all_results.items()}})
        raw_full = [all_results[a]["bootstrap_delta_full"]["42"]["p_two_sided"] for a in ASSETS]
        raw_qm = [all_results[a]["bootstrap_delta_qml_mlp"]["42"]["p_two_sided"] for a in ASSETS]
        json_write("primary_holm_correction.json", {"families": {
            "delta_full_across_assets": [{"asset": a, "raw_p": p, "holm_p": adj} for a, p, adj in zip(ASSETS, raw_full, holm(raw_full))],
            "delta_qml_mlp_across_assets": [{"asset": a, "raw_p": p, "holm_p": adj} for a, p, adj in zip(ASSETS, raw_qm, holm(raw_qm))]},
            "method": "Holm across exactly the two asset hypotheses separately for each primary endpoint"})
    return {"primary": primary, "predictions": outputs, "bootstrap": {a: all_results[a] for a in ASSETS}}


def evaluate_placebo(run_tag: str = "primary", save_outputs: bool = True) -> dict:
    freeze = check_frozen_inputs()
    split = freeze["split"]
    panel = pd.read_csv(PANEL, parse_dates=["period_start_utc"], low_memory=False)
    panel["period_start_utc"] = pd.to_datetime(panel.period_start_utc, utc=True)
    assets_arrays = {a: get_arrays(panel, a) for a in ASSETS}
    valid_dates = []
    for asset in ASSETS:
        ga, _, _, _, _, valid = assets_arrays[asset]
        valid_dates.append(pd.DatetimeIndex(ga.period_start_utc[valid]))
    common = valid_dates[0].intersection(valid_dates[1]).sort_values()
    n = len(common)
    if n != split["common_valid_rows_before_purge"] or common[0].isoformat() != split["common_start_utc"] or common[-1].isoformat() != split["common_end_utc"]:
        raise RuntimeError("Placebo complete-case sample differs from frozen external split")
    # Use the same complete common panel, then constrain anchors so t+7..t+12
    # displaced derivative state remains inside its own train/holdout segment.
    boundary = int(np.floor(.70 * n))
    out = {}
    for asset in ASSETS:
        g0, x0all, xdall, rawall, yall, validall = assets_arrays[asset]
        original_pos = {d: i for i, d in enumerate(pd.DatetimeIndex(g0.period_start_utc))}
        selected = np.asarray([original_pos[d] for d in common], int)
        g = g0.iloc[selected].reset_index(drop=True)
        x0 = x0all[selected]
        xd = xdall[selected]
        rawspot = pd.to_numeric(g.spot_amihud, errors="coerce").to_numpy(float)
        past_raw = np.full(n, np.nan)
        for i in range(6, n):
            w = rawspot[i-6:i]
            if np.isfinite(w).all():
                past_raw[i] = float(np.mean(w))
        past_y = np.log1p(past_raw)
        dx_future = np.full_like(xd, np.nan)
        dx_future[:-12] = xd[12:]
        # The placebo target is Amihud over t-6..t-1. Give its matched
        # SpotOnly benchmark information only through t-7, so no predictor
        # overlaps or follows the already-realized placebo target.
        x0_past = np.full_like(x0, np.nan, dtype=float)
        x0_past[7:] = x0[:-7]
        valid0 = np.isfinite(x0_past).all(axis=1) & np.isfinite(past_y)
        train_anchor_end = max(0, boundary - 12)
        train_possible = np.flatnonzero(valid0[:train_anchor_end])
        # Keep the entire past target window inside the holdout; derivative
        # features remain strictly later, at t+7..t+12.
        hold_idx = np.arange(boundary + 6, n - 12, dtype=int)
        hold_idx = hold_idx[valid0[hold_idx] & np.isfinite(dx_future[hold_idx]).all(axis=1)]
        if len(train_possible) < 250 or len(hold_idx) < 100:
            raise RuntimeError(f"Temporal placebo has insufficient dates for {asset}")
        # OOF baseline targets on the same fixed chronological schedule, with a
        # warm-up and six-row purge; train only on rows whose future derivatives
        # remain before the frozen holdout boundary.
        train_anchor_end = min(train_anchor_end, boundary - 12)
        warmup = int(np.ceil(.50 * train_anchor_end))
        val_blocks = [b for b in np.array_split(np.arange(warmup, train_anchor_end), 5) if len(b)]
        cfg = config_for(asset)
        base = json.loads((OUT / "matched_classical_models.json").read_text())["spot_only"][asset]
        oof_r, oof_x = [], []
        for j, block in enumerate(val_blocks, 1):
            first = int(block[0]); fit_end = first - PURGE
            tr = np.arange(13, fit_end, dtype=int)
            va = block.astype(int)
            tr = tr[valid0[tr] & np.isfinite(dx_future[tr]).all(axis=1)]
            va = va[valid0[va] & np.isfinite(dx_future[va]).all(axis=1)]
            if len(tr) < 100 or len(va) == 0:
                continue
            seed = stable_seed(BASE_SEED, asset, base["family"], "phase9_placebo_oof", j)
            pred, _, _ = fit_predict(base["family"], base["config"], x0_past[tr], past_y[tr], x0_past[va], seed=seed)
            oof_r.extend((past_y[va] - pred).tolist())
            oof_x.extend(dx_future[va])
        xtr = np.asarray(oof_x, float)
        ytr = np.asarray(oof_r, float)
        cfg_model = config_for(asset)
        seed_spot = stable_seed(BASE_SEED, asset, base["family"], "phase9_placebo_final")
        # SpotOnly uses the largest permitted venue-local training span; only
        # the residual learner's future-shifted derivative features need the
        # additional 12-interval truncation above.
        tr_final = np.arange(13, boundary - PURGE, dtype=int)
        tr_final = tr_final[valid0[tr_final] & np.isfinite(dx_future[tr_final]).all(axis=1)]
        p0, _, _ = fit_predict(base["family"], base["config"], x0_past[tr_final], past_y[tr_final], x0_past[hold_idx], seed=seed_spot)
        q_by_seed, m_by_seed = [], []
        for seed in cfg_model["seeds"]:
            q, _ = fit_qml_one(asset, xtr, ytr, dx_future[hold_idx], int(seed), cfg_model)
            m, _ = fit_mlp_residual(xtr, ytr, dx_future[hold_idx], cfg_model, int(seed))
            q_by_seed.append(np.asarray(q, float)); m_by_seed.append(np.asarray(m, float))
        qmat, mmat = np.vstack(q_by_seed), np.vstack(m_by_seed)
        ytest = past_y[hold_idx]
        qcorr = p0 + qmat.mean(axis=0)
        mcorr = p0 + mmat.mean(axis=0)
        delta = np.abs(ytest - p0) - np.abs(ytest - qcorr)
        dates = common[hold_idx]
        boot = bootstrap_delta(dates, delta, asset, "delta_placebo")
        p42 = boot["42"]["p_two_sided"]
        ci42 = boot["42"]["ci95"]
        warning = bool(float(delta.mean()) > 0 and ci42[0] > 0 and p42 < .05)
        out[asset] = {"n_holdout_rows": int(len(hold_idx)), "first_date": dates[0].isoformat(), "last_date": dates[-1].isoformat(),
                      "label": "past Amihud mean over t−6..t−1", "derivative_information": "future derivative feature history over t+7..t+12", "same_spotonly_family_and_config": base,
                      "spotonly_mae": float(np.mean(np.abs(ytest - p0))), "qml_corrected_mae": float(np.mean(np.abs(ytest - qcorr))),
                      "mlp_corrected_mae": float(np.mean(np.abs(ytest - mcorr))), "delta_placebo_positive_warns": float(delta.mean()),
                      "bootstrap": boot, "matched_mlp_seed_deltas": [float(np.mean(np.abs(ytest-(p0+mmat[i])) - np.abs(ytest-(p0+qmat[i])))) for i in range(5)],
                      "placebo_warning": warning, "interpretation": "A positive supported gain from derivative information dated after the past target flags temporal persistence/confounding; it is not forward predictive evidence."}
        print(f"{run_tag} temporal placebo {asset}: delta={delta.mean():.8g}, warning={warning}", flush=True)
    result = {"status": "COMPLETE", "by_asset": out, "rule": "Warning if positive placebo improvement has 42-block 95% CI entirely above zero and two-sided bootstrap p<0.05.", "outcome_fit_before_freeze": False}
    if save_outputs:
        json_write("negative_controls.json", result)
    return result


def inference_payload(main_result: dict, placebo_result: dict) -> dict:
    payload = {"main": {}, "placebo": {}}
    for asset in ASSETS:
        r = main_result["primary"]["by_asset"][asset]
        payload["main"][asset] = {
            "n_holdout_rows": r["n_holdout_rows"],
            "first_holdout_date": r["first_holdout_date"],
            "last_holdout_date": r["last_holdout_date"],
            "spot_only": r["spot_only"],
            "spot_only_plus_mlp_residual_correction": r["spot_only_plus_mlp_residual_correction"],
            "spot_only_plus_qml_residual_correction": r["spot_only_plus_qml_residual_correction"],
            "spot_only_plus_strong_classical_residual_correction": r["spot_only_plus_strong_classical_residual_correction"],
            "delta_full": r["delta_full_positive_favors_qml"],
            "delta_qml_mlp": r["delta_qml_mlp_positive_favors_qml"],
            "delta_qml_strong_classical": r["delta_qml_strong_classical_positive_favors_qml"],
            "seed_deltas": r["seed_delta_qml_vs_mlp_positive_favors_qml"],
            "bootstrap_delta_full": r["bootstrap_delta_full"],
            "bootstrap_delta_qml_mlp": r["bootstrap_delta_qml_mlp"],
            "bootstrap_delta_qml_strong_classical": r["bootstrap_delta_qml_strong_classical"],
            "hac_delta_full": r["hac_delta_full"],
            "hac_delta_qml_mlp": r["hac_delta_qml_mlp"],
        }
        p = placebo_result["by_asset"][asset]
        payload["placebo"][asset] = {k: p[k] for k in ["n_holdout_rows", "first_date", "last_date", "spotonly_mae",
                                                            "qml_corrected_mae", "mlp_corrected_mae", "delta_placebo_positive_warns",
                                                            "bootstrap", "matched_mlp_seed_deltas", "placebo_warning"]}
    return payload


def run_full(run_tag: str = "primary", save_outputs: bool = True) -> dict:
    if json.loads((OUT / "qml_fixture_validation.json").read_text()).get("status") != "VALID":
        fixture_validation()
    main = evaluate_main(run_tag, save_outputs)
    placebo = evaluate_placebo(run_tag, save_outputs)
    if save_outputs:
        prediction_hashes = json.loads((OUT / "forecast_prediction_hashes.json").read_text())
        prediction_hashes["canonical_inference_payload_sha256"] = json_hash(inference_payload(main, placebo))
        json_write("forecast_prediction_hashes.json", prediction_hashes)
        strong_p = {a: main["primary"]["by_asset"][a]["delta_qml_strong_classical_positive_favors_qml"] for a in ASSETS}
        json_write("random_feature_control.json", {"status": "NOT_RUN_PRE-SPECIFIED_OPTIONAL_DIAGNOSTIC", "reason": "The random-feature diagnostic was optional and was not needed for either primary decision; the frozen required negative temporal control was run instead."})
        json_write("seed_stability.json", {"by_asset": {a: {"seeds": config_for(a)["seeds"],
                    "seed_deltas_positive_favor_qml": main["primary"]["by_asset"][a]["seed_delta_qml_vs_mlp_positive_favors_qml"],
                    "seeds_favoring_qml": main["primary"]["by_asset"][a]["seeds_favoring_qml_vs_mlp"],
                    "required_minimum": 4, "pass": main["primary"]["by_asset"][a]["seeds_favoring_qml_vs_mlp"] >= 4} for a in ASSETS}})
        # Recompute Holm adjusted p values for the primary test families.
        h = json.loads((OUT / "primary_holm_correction.json").read_text())
        adj_full = {r["asset"]: r["holm_p"] for r in h["families"]["delta_full_across_assets"]}
        adj_qm = {r["asset"]: r["holm_p"] for r in h["families"]["delta_qml_mlp_across_assets"]}
        for asset in ASSETS:
            main["primary"]["by_asset"][asset]["holm_p_delta_full"] = adj_full[asset]
            main["primary"]["by_asset"][asset]["holm_p_delta_qml_mlp"] = adj_qm[asset]
        json_write("primary_full_forecast_results.json", main["primary"])
        decisions = {}
        for asset in ASSETS:
            r = main["primary"]["by_asset"][asset]
            p = placebo["by_asset"][asset]
            pass_full = r["delta_full_positive_favors_qml"] > 0 and r["bootstrap_delta_full"]["42"]["ci95"][0] > 0 and adj_full[asset] < .05
            pass_qml_mlp = r["delta_qml_mlp_positive_favors_qml"] > 0 and r["bootstrap_delta_qml_mlp"]["42"]["ci95"][0] > 0 and adj_qm[asset] < .05
            pass_seeds = r["seeds_favoring_qml_vs_mlp"] >= 4
            pass_placebo = not p["placebo_warning"]
            passed = pass_full and pass_qml_mlp and pass_seeds and pass_placebo
            algorithmic_only = pass_qml_mlp and not pass_full
            if passed:
                status = asset + "_PASS"
                classification = "QUANTUM_INCREMENTAL_VALUE"
            elif algorithmic_only:
                status = asset + "_FAIL"
                classification = "ALGORITHMIC_QML_GAIN_WITHOUT_INCREMENTAL_INFORMATION"
            elif len(r["bootstrap_delta_full"]["42"]["ci95"]) and r["n_holdout_rows"] < 100:
                status = asset + "_INCONCLUSIVE"
                classification = "INCONCLUSIVE"
            elif pass_full or pass_qml_mlp or p["placebo_warning"]:
                status = asset + "_FAIL"
                classification = "MODEL_SENSITIVE_SIGNAL"
            else:
                status = asset + "_FAIL"
                classification = "NO_INCREMENTAL_INFORMATION"
            decisions[asset] = {"decision": status, "primary_classification": classification,
                                "conditions": {"delta_full_positive_and_bootstrap_ci_positive_and_holm": bool(pass_full),
                                               "delta_qml_mlp_positive_and_bootstrap_ci_positive_and_holm": bool(pass_qml_mlp),
                                               "at_least_4_of_5_seeds_favor_qml": bool(pass_seeds),
                                               "no_matched_temporal_placebo_failure": bool(pass_placebo)},
                                "delta_full": r["delta_full_positive_favors_qml"], "delta_qml_mlp": r["delta_qml_mlp_positive_favors_qml"],
                                "holm_p_delta_full": adj_full[asset], "holm_p_delta_qml_mlp": adj_qm[asset],
                                "seed_deltas": r["seed_delta_qml_vs_mlp_positive_favors_qml"],
                                "temporal_placebo_warning": p["placebo_warning"]}
        json_write("btc_replication_decision.json", decisions["BTC"])
        json_write("eth_replication_decision.json", decisions["ETH"])
        passes = [decisions[a]["decision"] == a + "_PASS" for a in ASSETS]
        if all(passes):
            paper = "EXTERNALLY_REPLICATED_QML_INCREMENTAL_VALUE"
        elif any(passes):
            paper = "ASSET_SPECIFIC_EXTERNAL_QML_SIGNAL"
        elif any(decisions[a]["primary_classification"] == "ALGORITHMIC_QML_GAIN_WITHOUT_INCREMENTAL_INFORMATION" for a in ASSETS):
            paper = "ALGORITHMIC_QML_GAIN_WITHOUT_INCREMENTAL_INFORMATION"
        elif all(decisions[a]["decision"].endswith("_FAIL") for a in ASSETS):
            paper = "NO_EXTERNAL_QML_REPLICATION"
        else:
            paper = "INCONCLUSIVE"
        json_write("venue_replication_decision.json", {"venue": "Bybit", "status": "ELIGIBLE", "asset_decisions": decisions})
        json_write("paper_level_decision.json", {"classification": paper, "btc": decisions["BTC"]["decision"], "eth": decisions["ETH"]["decision"],
                    "primary_success_requires_both_assets": True, "strong_classical_delta": strong_p,
                    "quantum_advantage_claim": "NOT_SUPPORTED; exact statevector simulation only"})
        summary = {
            "venue": "Bybit (only external venue passing the frozen data/semantic eligibility gate; OKX excluded for insufficient historical OI)",
            "data": {"panel_sha256": sha256(PANEL), "split": freeze["split"], "semantic_parity": "See semantic_parity.json; Bybit baseline fields passed the preregistered gate."},
            "target": "24-hour future spot Amihud, arithmetic mean t+1..t+6, T2_log1p transformed target",
            "models": {"BTC": "frozen Hybrid QNN 4q", "ETH": "frozen variational QNN 6q", "matched_comparator": "frozen one-hidden-layer MLP; exact five paired seeds"},
            "results": {a: {"spotonly_transformed_mae": main["primary"]["by_asset"][a]["spot_only"]["transformed_mae"],
                             "spotonly_plus_qml_transformed_mae": main["primary"]["by_asset"][a]["spot_only_plus_qml_residual_correction"]["transformed_mae"],
                             "spotonly_plus_mlp_transformed_mae": main["primary"]["by_asset"][a]["spot_only_plus_mlp_residual_correction"]["transformed_mae"],
                             "delta_full": main["primary"]["by_asset"][a]["delta_full_positive_favors_qml"],
                             "delta_qml_mlp": main["primary"]["by_asset"][a]["delta_qml_mlp_positive_favors_qml"],
                             "delta_full_42_ci": main["primary"]["by_asset"][a]["bootstrap_delta_full"]["42"]["ci95"],
                             "delta_qml_mlp_42_ci": main["primary"]["by_asset"][a]["bootstrap_delta_qml_mlp"]["42"]["ci95"],
                             "holm_p_full": adj_full[a], "holm_p_qml_mlp": adj_qm[a],
                             "seeds_favoring_qml": main["primary"]["by_asset"][a]["seeds_favoring_qml_vs_mlp"],
                             "placebo_warning": placebo["by_asset"][a]["placebo_warning"], "decision": decisions[a]["decision"]} for a in ASSETS},
            "interpretation": "Predictive information only; no causal interpretation. QML beating MLP is separate from improving SpotOnly. No quantum advantage claim is supported by statevector simulation.",
            "paper_level_classification": paper,
        }
        report = ["# Phase 9 research summary", "", "Bybit is the only eligible external venue. OKX has a contract OI history endpoint, but direct historical 4-hour probes returned no rows for older windows and did not meet the minimum common history. No predictive results were used for venue selection.",
                  "", "The target is the frozen 24-hour future spot Amihud measure, averaged over the next six 4-hour intervals and transformed using the prior-study log1p rule. BTC uses Hybrid QNN 4q and ETH uses QNN 6q. Both use the exact five prior-study seeds and paired one-hidden-layer MLPs. The chronological 70/30 split was frozen from common complete rows before model execution.", ""]
        for a in ASSETS:
            r = summary["results"][a]
            report.append(f"For {a}, SpotOnly transformed MAE was {r['spotonly_transformed_mae']:.8g}; the corrected QML forecast MAE was {r['spotonly_plus_qml_transformed_mae']:.8g}; the matched MLP-corrected MAE was {r['spotonly_plus_mlp_transformed_mae']:.8g}. Delta_FULL was {r['delta_full']:.8g} (42-interval block CI {r['delta_full_42_ci']}); QML-minus-MLP was {r['delta_qml_mlp']:.8g} (CI {r['delta_qml_mlp_42_ci']}). Holm p-values were {r['holm_p_full']:.4g} and {r['holm_p_qml_mlp']:.4g}. {r['seeds_favoring_qml']}/5 seeds favored QML. The temporal placebo warning was {r['placebo_warning']}. Decision: {r['decision']}.")
        report.extend(["", f"The paper-level classification is {paper}. These results can support only a venue-specific predictive-information statement under the frozen forecast design. They cannot establish causal transmission, universal derivatives-to-spot predictability, quantum advantage, or hardware benefit. The previous Phase 7/8 negative generalization evidence remains in force and is not overwritten.", ""])
        (OUT / "research_summary.md").write_text("\n".join(report), encoding="utf-8")
        json_write("forecast_prediction_hashes.json", {
            "run": run_tag,
            "canonical_prediction_payload_sha256": json_hash(main["predictions"]),
            "holdout_prediction_hashes": {
                name: json_hash(main["predictions"][key])
                for key, name in (("oof", "spot_oof"), ("resid", "oof_residuals"),
                                  ("p0", "spotonly_holdout"), ("mlp", "mlp_corrected_holdout"),
                                  ("qml", "qml_corrected_holdout"), ("strong", "strong_classical_holdout"))
            },
            "scope": "predictions/targets/dates only; excludes wall-clock fit times and runtime metadata",
        })
    return {"main": main, "placebo": placebo}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "fixtures-only":
        print(json.dumps(fixture_validation(), indent=2))
        return
    if mode == "run":
        res = run_full("primary", True)
        print(json.dumps({"status": "COMPLETE", "assets": list(res["main"]["primary"]["by_asset"]),
                          "paper_level": json.loads((OUT / "paper_level_decision.json").read_text())["classification"]}, indent=2))
        return
    if mode == "offline-rerun":
        # Must be called in an environment with networking disabled. It rebuilds
        # forecasts from the already-hashed local panel and repeats all fits.
        if os.environ.get("PHASE9_NETWORK_DISABLED") != "1":
            raise RuntimeError("Offline reproduction requires the Phase 9 socket-blocking sitecustomize guard")
        res = run_full("offline", False)
        existing_hash = json.loads((OUT / "forecast_prediction_hashes.json").read_text())
        rerun_hash = json_hash(res["main"]["predictions"])
        rerun_inference_hash = json_hash(inference_payload(res["main"], res["placebo"]))
        summary = {"status": "COMPLETE", "protocol_sha256": sha256(OUT / "phase9_protocol.json"),
                   "panel_sha256": sha256(PANEL),
                   "network_disabled": os.environ.get("PHASE9_NETWORK_DISABLED") == "1",
                   "canonical_prediction_payload_sha256": rerun_hash,
                   "prediction_payload_matches_primary": rerun_hash == existing_hash["canonical_prediction_payload_sha256"],
                   "canonical_inference_payload_sha256": rerun_inference_hash,
                   "inference_payload_matches_primary": rerun_inference_hash == existing_hash.get("canonical_inference_payload_sha256"),
                   "asset_primary": {a: {"delta_full": res["main"]["primary"]["by_asset"][a]["delta_full_positive_favors_qml"],
                                         "delta_qml_mlp": res["main"]["primary"]["by_asset"][a]["delta_qml_mlp_positive_favors_qml"],
                                         "spotonly_mae": res["main"]["primary"]["by_asset"][a]["spot_only"]["transformed_mae"],
                                         "qml_mae": res["main"]["primary"]["by_asset"][a]["spot_only_plus_qml_residual_correction"]["transformed_mae"]} for a in ASSETS},
                   "placebo": {a: {"delta": res["placebo"]["by_asset"][a]["delta_placebo_positive_warns"], "warning": res["placebo"]["by_asset"][a]["placebo_warning"]} for a in ASSETS}}
        json_write("offline_rerun_payload.json", summary)
        if not summary["prediction_payload_matches_primary"] or not summary["inference_payload_matches_primary"]:
            raise RuntimeError("Offline prediction/inference payload differs from primary run")
        print(json.dumps(summary, indent=2))
        return
    raise SystemExit("Usage: phase9_external_qml_estimate.py [fixtures-only|run|offline-rerun]")


if __name__ == "__main__":
    main()
