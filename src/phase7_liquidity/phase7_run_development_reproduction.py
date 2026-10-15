from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "phase7_liquidity_replication" / "offline_reproduction_development"
PANEL = ROOT / "data" / "processed" / "phase7" / "market_liquidity_panel_4h.csv.gz"
DEV_START = pd.Timestamp("2021-12-01T00:00:00Z")
DEV_LAST = pd.Timestamp("2023-06-30T20:00:00Z")
DEV_END_EXCL = pd.Timestamp("2023-07-01T00:00:00Z")
SEED = 724051
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

SPOT = ["spot_return", "spot_realized_volatility", "spot_downside_semivariance", "spot_amihud", "spot_taker_imbalance", "spot_quote_volume_growth"]
DERIV = ["funding_rate_state", "oi_log_change", "futures_taker_imbalance", "basis_mean"]
CALENDAR = [f"utc_slot_{i}" for i in range(6)] + [f"weekday_{i}" for i in range(7)]
DERIV_STATIC = ["funding_change", "basis_change", "oi_shock", "flow_shock", "funding_basis_interaction", "funding_oishock_interaction", "oishock_flow_interaction", "flow_basis_interaction"]
TARGETS = ["future_amihud_24h_mean", "future_kyle_abs_lambda_24h", "future_corwin_schultz_spread_24h"]
INFOSETS = ["legacy_spot_only", "legacy_spot_plus_derivatives", "enhanced_spot_only", "enhanced_spot_plus_derivatives"]
ASSETS = ["BTC", "ETH", "SOL", "BNB", "XRP"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating, float)): return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.bool_, bool)): return bool(x)
    if isinstance(x, pd.Timestamp): return x.isoformat()
    if isinstance(x, np.ndarray): return clean(x.tolist())
    return x


def dump(path: Path, obj):
    path.write_text(json.dumps(clean(obj), indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_dev_only() -> pd.DataFrame:
    """Stream through the panel but numerically parse only rows ending in development."""
    cols = ["period_start_utc", "asset", "future_window_end_utc_exclusive"] + SPOT + DERIV + CALENDAR + DERIV_STATIC + TARGETS + ["spot_close", "corwin_schultz_4h"]
    keep = []
    with gzip.open(PANEL, "rt", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected = set(reader.fieldnames or [])
        missing = sorted(set(cols) - expected)
        if missing: raise RuntimeError(f"Missing frozen panel columns: {missing}")
        for row in reader:
            ts = pd.Timestamp(row["period_start_utc"])
            if ts < pd.Timestamp("2021-10-01T00:00:00Z") or ts > DEV_LAST:
                continue
            a = row["asset"]
            if a not in ASSETS: continue
            out = {"period_start_utc": ts, "asset": a, "future_window_end_utc_exclusive": pd.Timestamp(row["future_window_end_utc_exclusive"])}
            for c in cols:
                if c in out or c in ("period_start_utc", "asset", "future_window_end_utc_exclusive"): continue
                val = row[c]
                out[c] = float(val) if val not in ("", "nan", "NaN", "None") else np.nan
            keep.append(out)
    df = pd.DataFrame(keep).sort_values(["asset", "period_start_utc"]).reset_index(drop=True)
    if len(df) == 0 or df.period_start_utc.max() > DEV_LAST:
        raise RuntimeError("Development-only streaming load failed")
    return df


class TinyGRU(nn.Module):
    def __init__(self, n_seq: int, n_static: int, hidden: int):
        super().__init__()
        self.gru = nn.GRU(input_size=n_seq, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden + n_static, 1)

    def forward(self, seq, static):
        _, h = self.gru(seq)
        z = h[-1]
        if static.shape[1]: z = torch.cat([z, static], dim=1)
        return self.head(z).squeeze(-1)


def transform_y(y, name, scale=None):
    y = np.maximum(np.asarray(y, float), 0.0)
    if name == "raw_nonnegative": return y.copy(), 1.0
    if name == "log1p_y": return np.log1p(y), 1.0
    if name == "log1p_y_over_train_median_positive":
        s = float(scale) if scale is not None else (float(np.median(y[y > 0])) if np.any(y > 0) else 1.0)
        s = max(s, np.finfo(float).tiny)
        return np.log1p(y / s), s
    raise ValueError(name)


def inverse_y(z, name, scale):
    z = np.asarray(z, float)
    if name == "raw_nonnegative": return np.maximum(z, 0.0)
    z = np.maximum(z, 0.0)
    if name == "log1p_y": return np.expm1(z)
    if name == "log1p_y_over_train_median_positive": return scale * np.expm1(z)
    raise ValueError(name)


def make_features(df: pd.DataFrame, asset: str, infoset: str):
    g = df[df.asset == asset].sort_values("period_start_utc").reset_index(drop=True)
    if infoset.startswith("legacy"):
        base = SPOT if infoset == "legacy_spot_only" else SPOT + DERIV
        static_vars = []
    elif infoset == "enhanced_spot_only":
        base, static_vars = SPOT, CALENDAR
    elif infoset == "enhanced_spot_plus_derivatives":
        base, static_vars = SPOT + DERIV, CALENDAR + DERIV_STATIC
    else: raise ValueError(infoset)
    seqraw = g[base].to_numpy(float)
    statraw = g[static_vars].to_numpy(float) if static_vars else np.empty((len(g), 0), float)
    seq = np.full((len(g), 6, len(base)), np.nan, float)
    for i in range(5, len(g)):
        # t-5 through t, in chronological order.
        seq[i] = seqraw[i-5:i+1]
    y = {t: g[t].to_numpy(float) for t in TARGETS}
    target_end = g.future_window_end_utc_exclusive
    date = g.period_start_utc
    # Paired information-set comparison uses the larger feature set's complete rows.
    if infoset in ("legacy_spot_only", "legacy_spot_plus_derivatives"):
        pair_base = SPOT + DERIV
        pair_static = []
    else:
        pair_base = SPOT + DERIV
        pair_static = CALENDAR + DERIV_STATIC
    pair_seq = np.full((len(g), 6, len(pair_base)), np.nan, float)
    pair_raw = g[pair_base].to_numpy(float)
    for i in range(5, len(g)):
        pair_seq[i] = pair_raw[i-5:i+1]
    pair_static_raw = g[pair_static].to_numpy(float) if pair_static else np.empty((len(g), 0), float)
    base_valid = np.isfinite(pair_seq).all(axis=(1,2))
    if pair_static: base_valid &= np.isfinite(pair_static_raw).all(axis=1)
    base_valid &= date.ge(DEV_START).to_numpy() & date.le(DEV_LAST).to_numpy()
    base_valid &= (target_end <= DEV_END_EXCL).to_numpy()
    seq_valid = np.isfinite(seq).all(axis=(1,2))
    if static_vars: seq_valid &= np.isfinite(statraw).all(axis=1)
    return {
        "date": pd.DatetimeIndex(date), "target_end": pd.DatetimeIndex(target_end),
        "seq": seq, "static": statraw, "y": y, "base_valid": base_valid,
        "info_valid": seq_valid, "static_names": static_vars, "base_names": base,
        "spot_current": g.spot_amihud.to_numpy(float), "cs_current": g.corwin_schultz_4h.to_numpy(float),
    }


def fold_masks(cache, fold):
    _, start, end = fold
    d, e = cache["date"], cache["target_end"]
    train = (d >= DEV_START) & (d < start) & (e <= start)
    val = (d >= start) & (d < end) & (e <= end)
    return train, val


def get_cfg_grid(protocol):
    out = {}
    for fam, obj in protocol["families"].items():
        out[fam] = obj["configs"]
    return out


def fit_model(family, cfg, xseq_tr, xstat_tr, ytr, xseq_va, xstat_va, seed):
    if family == "GRU":
        ntr, t, nf = xseq_tr.shape
        scale_seq = StandardScaler().fit(xseq_tr.reshape(-1, nf))
        trseq = scale_seq.transform(xseq_tr.reshape(-1, nf)).reshape(ntr, t, nf)
        nv = len(xseq_va)
        vaseq = scale_seq.transform(xseq_va.reshape(-1, nf)).reshape(nv, t, nf)
        if xstat_tr.shape[1]:
            scale_static = StandardScaler().fit(xstat_tr)
            trstat = scale_static.transform(xstat_tr); vastat = scale_static.transform(xstat_va)
        else:
            trstat=np.empty((ntr,0));vastat=np.empty((nv,0))
        torch.manual_seed(seed); np.random.seed(seed)
        model = TinyGRU(nf, trstat.shape[1], int(cfg["hidden_size"]))
        opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=float(cfg["weight_decay"]))
        xt=torch.as_tensor(trseq,dtype=torch.float32); st=torch.as_tensor(trstat,dtype=torch.float32); yt=torch.as_tensor(ytr,dtype=torch.float32)
        for _ in range(int(cfg["epochs"])):
            model.train();opt.zero_grad(set_to_none=True);loss=torch.mean((model(xt,st)-yt)**2);loss.backward();opt.step()
        model.eval()
        with torch.no_grad():
            pred=model(torch.as_tensor(vaseq,dtype=torch.float32),torch.as_tensor(vastat,dtype=torch.float32)).cpu().numpy()
        return np.maximum(np.asarray(pred, float), 0.0)
    xtr = np.concatenate([xseq_tr.reshape(len(xseq_tr), -1), xstat_tr], axis=1)
    xva = np.concatenate([xseq_va.reshape(len(xseq_va), -1), xstat_va], axis=1)
    if family in ("Ridge", "RBF_SVR", "MLP"):
        scaler=StandardScaler().fit(xtr);xtr=scaler.transform(xtr);xva=scaler.transform(xva)
    if family == "Ridge":
        model=Ridge(alpha=float(cfg["alpha"]))
    elif family == "RBF_SVR":
        model=SVR(C=float(cfg["C"]),gamma=cfg["gamma"],epsilon=float(cfg["epsilon"]))
    elif family == "ExtraTrees":
        model=ExtraTreesRegressor(n_estimators=int(cfg["n_estimators"]),min_samples_leaf=int(cfg["min_samples_leaf"]),max_features=1.0,random_state=seed,n_jobs=1)
    elif family == "HistGradientBoosting":
        model=HistGradientBoostingRegressor(max_iter=int(cfg["max_iter"]),learning_rate=float(cfg["learning_rate"]),max_leaf_nodes=int(cfg["max_leaf_nodes"]),l2_regularization=1.0,early_stopping=False,random_state=seed)
    elif family == "MLP":
        model=MLPRegressor(hidden_layer_sizes=tuple(cfg["hidden_layer_sizes"]),activation="relu",solver="adam",alpha=float(cfg["alpha"]),learning_rate_init=0.001,max_iter=int(cfg["max_iter"]),early_stopping=False,shuffle=True,random_state=seed,tol=1e-5,n_iter_no_change=30)
    else: raise ValueError(family)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(xtr,ytr)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        # Keep deterministic convergence status for the audit; do not tune on it.
        pass
    pred=np.asarray(model.predict(xva),float)
    if not np.isfinite(pred).all(): raise FloatingPointError(f"{family} returned nonfinite development predictions")
    return np.maximum(pred, 0.0)


def main():
    freeze=json.loads((OUT/"phase7_freeze_manifest.json").read_text())
    if not freeze.get("protocol_frozen") or sha(OUT/"phase7_protocol.json")!=freeze["phase7_protocol_sha256"]:
        raise RuntimeError("Phase-7 protocol hash mismatch")
    if sha(PANEL)!=freeze["frozen_inputs_and_rules"]["data/processed/phase7/market_liquidity_panel_4h.csv.gz"]:
        raise RuntimeError("Frozen Phase-7 panel changed after protocol freeze")
    p=json.loads((OUT/"phase7_protocol.json").read_text())
    df=load_dev_only()
    folds=[(x["fold"],pd.Timestamp(x["validation_start"]),pd.Timestamp(x["validation_end_exclusive"])) for x in p["development_folds"]]
    transforms=p["transform_candidates"]
    configs=get_cfg_grid(p)
    families=list(configs)
    transform_results=[]; best_predictions={}
    nfits=0; started=time.time()
    # First select one target transform per asset-target from frozen expanding folds.
    for asset in ASSETS:
      for target in TARGETS:
       by_transform={}
       caches={info:make_features(df,asset,info) for info in INFOSETS}
       for transform in transforms:
        group_rows=[]; best_by_group={}
        for info in INFOSETS:
         c=caches[info]
         valid=c["base_valid"] & c["info_valid"] & np.isfinite(c["y"][target]) & (c["y"][target]>=0)
         # shared complete-case rule for the Spot-only / Plus comparison
         for family in families:
          cfg_scores=[]
          for cfg_i,cfg in enumerate(configs[family]):
           fold_payload=[];pred_t=[];truth_t=[];pred_raw_all=[];true_raw_all=[];abs_raw_norm=[];n_fold=0
           for fold_i,fold in enumerate(folds):
            tr,va=fold_masks(c,fold);tr &= valid;va &= valid
            if tr.sum()<300 or va.sum()<100: raise RuntimeError(f"Insufficient fold observations {asset}/{target}/{info}/{family}/{fold[0]}")
            ytr,scale=transform_y(c["y"][target][tr],transform)
            yva,_=transform_y(c["y"][target][va],transform,scale)
            seed=SEED+ASSETS.index(asset)*10000+TARGETS.index(target)*1000+fold_i*100+families.index(family)*10+cfg_i
            pred=fit_model(family,cfg,c["seq"][tr],c["static"][tr],ytr,c["seq"][va],c["static"][va],seed)
            yt_true=c["y"][target][va]
            pred_raw=inverse_y(pred,transform,scale);true_raw=yt_true
            norm=float(np.median(c["y"][target][tr][c["y"][target][tr]>0]))
            norm=max(norm,np.finfo(float).tiny)
            pred_t.append(pred);truth_t.append(yva)
            pred_raw_all.append(pred_raw);true_raw_all.append(true_raw)
            abs_raw_norm.extend((np.abs(pred_raw-true_raw)/norm).tolist())
            fold_payload.append({"fold":fold[0],"n_train":int(tr.sum()),"n_validation":int(va.sum()),"train_first":c["date"][tr].min(),"train_last":c["date"][tr].max(),"validation_first":c["date"][va].min(),"validation_last":c["date"][va].max(),"target_transform_scale":scale})
            n_fold += int(va.sum()); nfits += 1
           yp=np.concatenate(pred_t); yt=np.concatenate(truth_t)
           mae=float(np.mean(np.abs(yp-yt)))
           raw_norm_mae=float(np.mean(abs_raw_norm))
           cfg_scores.append({"config_index":cfg_i,"config":cfg,"transformed_oof_mae":mae,"normalized_original_scale_oof_mae":raw_norm_mae,"oof_n":n_fold,"folds":fold_payload,"oof_pred":yp,"oof_true_transformed":yt,"oof_true_raw":np.concatenate(true_raw_all),"oof_pred_raw":np.concatenate(pred_raw_all),"oof_dates":np.concatenate([c["date"][fold_masks(c,f)[1]&valid].asi8 for f in folds])})
          chosen=min(cfg_scores,key=lambda z:(z["transformed_oof_mae"],z["config_index"]))
          # Retain the selected configuration's OOF predictions for threshold construction.
          best_by_group[(info,family)]={k:v for k,v in chosen.items() if k.startswith("oof_")}
          group_rows.append({"infoset":info,"family":family,"family_category":p["families"][family]["category"],"selected_config_index":chosen["config_index"],"selected_config":chosen["config"],"selected_transformed_oof_mae":chosen["transformed_oof_mae"],"normalized_original_scale_oof_mae":chosen["normalized_original_scale_oof_mae"],"candidate_scores":[{k:v for k,v in s.items() if not k.startswith("oof_")} for s in cfg_scores],"folds":chosen["folds"]})
        pooled=float(np.mean([x["normalized_original_scale_oof_mae"] for x in group_rows]))
        by_transform[transform]={"pooled_normalized_original_scale_mae":pooled,"groups":group_rows,"selected_oof":best_by_group}
       selected_transform=min(transforms,key=lambda x:(by_transform[x]["pooled_normalized_original_scale_mae"],transforms.index(x)))
       transform_results.append({"asset":asset,"target":target,"selected_transform":selected_transform,"selection_scores":{x:by_transform[x]["pooled_normalized_original_scale_mae"] for x in transforms},"groups_by_selected_transform":by_transform[selected_transform]["groups"]})
       for (info,family),payload in by_transform[selected_transform]["selected_oof"].items():
        best_predictions[(asset,target,info,family)]=(selected_transform,payload)
       print(f"development selected transform {asset}/{target}: {selected_transform}",flush=True)
    # Build chosen configurations and selected-model OOF series for stress thresholds.
    selected_models=[];oof=[]
    for tr in transform_results:
      asset,target,transform=tr["asset"],tr["target"],tr["selected_transform"]
      for group in tr["groups_by_selected_transform"]:
        key=(asset,target,group["infoset"],group["family"])
        _tr,payload=best_predictions[key]
        selected_models.append({"asset":asset,"target":target,"infoset":group["infoset"],"family":group["family"],"family_category":group["family_category"],"target_transform":transform,"selected_config_index":group["selected_config_index"],"selected_config":group["selected_config"],"dev_transformed_oof_mae":group["selected_transformed_oof_mae"],"dev_normalized_original_scale_oof_mae":group["normalized_original_scale_oof_mae"]})
        pred_raw_ids=np.asarray(payload["oof_dates"],dtype=np.int64)
        for i in range(len(payload["oof_pred"])):
          yraw=float(payload["oof_true_raw"][i]); ypred=float(payload["oof_pred_raw"][i])
          oof.append({"asset":asset,"target":target,"infoset":group["infoset"],"family":group["family"],"date":pd.to_datetime(pred_raw_ids[i],utc=True).isoformat(),"y_raw":yraw,"y_transformed":float(payload["oof_true_transformed"][i]),"prediction_transformed":float(payload["oof_pred"][i]),"prediction_raw":ypred})
    # Stress threshold is based on dev target quantiles; alarm threshold is based on selected-model OOF predictions.
    oof_df=pd.DataFrame(oof)
    stress={}
    for asset in ASSETS:
      g=df[df.asset==asset]
      for target in TARGETS:
        y=g.loc[(g.period_start_utc>=DEV_START)&(g.period_start_utc<=DEV_LAST)&(g.future_window_end_utc_exclusive<=DEV_END_EXCL),target].dropna().to_numpy(float)
        if not len(y): continue
        target_q90=float(np.quantile(y,0.9))
        thresholds={}
        for info in INFOSETS:
          for family in families:
            q=oof_df[(oof_df.asset==asset)&(oof_df.target==target)&(oof_df.infoset==info)&(oof_df.family==family)]
            thresholds[f"{info}__{family}"]=float(q.prediction_raw.quantile(0.9)) if len(q) else None
        stress[f"{asset}__{target}"]={"development_target_q90":target_q90,"development_oof_prediction_q90_raw":thresholds}
    oof_path=OUT/"phase7_development_oof_predictions.csv.gz"
    oof_df.to_csv(oof_path,index=False,compression={"method":"gzip","mtime":0})
    result={
      "protocol_sha256":freeze["phase7_protocol_sha256"],"dataset_sha256":sha(PANEL),"development_rows_parsed":int(len(df)),"development_date_min":df.period_start_utc.min(),"development_date_max":df.period_start_utc.max(),
      "external_outcome_values_parsed":False,"estimation_sample": "development folds only; no records after 2023-06-30 20:00 UTC numerically parsed",
      "folds":[{"fold":n,"validation_start":s,"validation_end_exclusive":e} for n,s,e in folds],"families":p["families"],"transforms":transform_results,"selected_models":selected_models,"stress_thresholds":stress,
      "development_oof_path":str(oof_path.relative_to(ROOT)),"development_oof_sha256":sha(oof_path),"fit_count":nfits,"random_seed":SEED,
      "selection_rule":p["development_selection"],"warnings": "MLP convergence warnings were retained by deterministic sklearn fits; no configuration was altered in response."
    }
    dump(OUT/"phase7_development_cv.json",result)
    dump(OUT/"phase7_selected_models.json",{"protocol_sha256":freeze["phase7_protocol_sha256"],"dataset_sha256":sha(PANEL),"selected_models":selected_models,"transform_by_asset_target":[{"asset":x["asset"],"target":x["target"],"transform":x["selected_transform"],"selection_scores":x["selection_scores"]} for x in transform_results],"stress_thresholds":stress})
    print(json.dumps({"development_complete":True,"fit_count":nfits,"elapsed_seconds":round(time.time()-started,1),"selected_transforms":{f"{x['asset']}__{x['target']}":x["selected_transform"] for x in transform_results},"selected_models":len(selected_models)},indent=2),flush=True)


if __name__=="__main__":
    main()
