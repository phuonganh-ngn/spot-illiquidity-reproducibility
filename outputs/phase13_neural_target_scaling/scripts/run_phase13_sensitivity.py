#!/usr/bin/env python3
"""Run the locked Phase 13 target-scaling sensitivity once; never edits Phase 6."""
from __future__ import annotations
import csv, hashlib, json, math, os, platform, sys, time, warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"outputs/phase13_neural_target_scaling"
P6=ROOT/"outputs/phase6_autonomous_benchmark"
sys.path.insert(0,str(ROOT/"src"))
import phase6_benchmark_core as core

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")

def seed_hash(*parts):
    return core.SEED+sum(sum(ord(c) for c in str(p)) for p in parts)

def check_lock():
    lockpath=OUT/"pretraining_lock_manifest.json"
    expected=(OUT/"pretraining_lock_manifest.sha256").read_text().split()[0]
    if sha(lockpath)!=expected: raise RuntimeError("pretraining lock hash mismatch")
    lock=json.loads(lockpath.read_text())
    if not lock.get("locked_before_training"): raise RuntimeError("pretraining lock is not marked frozen")
    for row in lock["files"]:
        if sha(OUT/row["file"])!=row["sha256"]: raise RuntimeError(f"locked file changed: {row['file']}")
    source=json.loads((OUT/"03_source_manifest_before.json").read_text())
    mismatches=[]
    for row in source["source_inventory"]:
        path=ROOT/row["path"]
        if not path.is_file() or sha(path)!=row["sha256"]: mismatches.append(row["path"])
    if mismatches: raise RuntimeError(f"pre-training source mismatch: {mismatches[:10]}")
    return lock,source

def load_history():
    pred=json.loads((P6/"final_test_predictions.json").read_text())
    met=json.loads((P6/"final_test_metrics.json").read_text())
    oof=json.loads((P6/"oof_predictions.json").read_text())
    selected=json.loads((P6/"selected_models.json").read_text())
    by_task={(r["asset"],r["target"],r["horizon_intervals"]):r for r in pred["tasks"]}
    metrics={(r["asset"],r["target"],r["horizon_intervals"]):r for r in met["tasks"]}
    oof_map={}
    for r in oof["rows"]:
        if r["target"]!="future_amihud" or r["horizon_intervals"]!=6: continue
        for family in ["MLP","GRU"]:
            oof_map[(r["asset"],r["information_set"],family)]={
                pd.Timestamp(d).isoformat():float(v)
                for d,v in zip(r["date"],r["predictions_log1p"][family])
            }
    return pred,met,oof,selected,by_task,metrics,oof_map

def historical_prediction_map(task_row,info,family):
    return {pd.Timestamp(d).isoformat():float(v) for d,v in zip(task_row["date"],task_row["predictions_log1p"][info][family])}

def metric(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    e=p-y
    return {"n":int(len(y)),"mae_log1p":float(np.mean(np.abs(e))),"rmse_log1p":float(np.sqrt(np.mean(e**2))),
            "median_absolute_error_log1p":float(np.median(np.abs(e))),
            "mae_original_scale":float(np.mean(np.abs(np.expm1(p)-np.expm1(y))))}

def describe(y,p,train_y):
    y=np.asarray(y,float); p=np.asarray(p,float); ae=np.abs(p-y)
    train_y=np.asarray(train_y,float)
    lim=10.0*float(np.max(train_y)) if len(train_y) else float("nan")
    return {
        "target_min":float(np.min(y)),"target_median":float(np.median(y)),"target_max":float(np.max(y)),
        "prediction_min":float(np.min(p)),"prediction_median":float(np.median(p)),"prediction_max":float(np.max(p)),
        "prediction_mean":float(np.mean(p)),"prediction_sd":float(np.std(p,ddof=0)),
        "absolute_error_median":float(np.median(ae)),"absolute_error_p90":float(np.quantile(ae,.90)),"absolute_error_p99":float(np.quantile(ae,.99)),
        "fraction_below_target_min":float(np.mean(p<float(np.min(y)))),"fraction_above_target_max":float(np.mean(p>float(np.max(y)))),
        "fraction_outside_target_range":float(np.mean((p<float(np.min(y)))|(p>float(np.max(y))))),
        "extreme_error_threshold_10x_train_ylog_max":lim,
        "n_errors_above_extreme_threshold":int(np.sum(ae>lim)) if np.isfinite(lim) else None,
        "mae_excluding_largest_error_diagnostic":float(np.mean(np.sort(ae)[:-1])) if len(ae)>1 else None
    }

def fit_once(family,cfg,branch,xtr_flat,xtr_seq,ytr_log,xev_flat,xev_seq,seed,fit_id):
    start=time.perf_counter(); warnings_seen=[]; history={}; target_mu=0.0; target_sd=1.0
    try:
        if branch=="TARGET_SCALED":
            raw_sd=float(np.std(np.asarray(ytr_log,float),ddof=0))
            if not np.isfinite(raw_sd) or raw_sd<=np.finfo(float).eps:
                raise ValueError(f"degenerate training target scale sigma={raw_sd}")
            target_scaler=StandardScaler().fit(np.asarray(ytr_log,float).reshape(-1,1))
            target_mu=float(target_scaler.mean_[0]); target_sd=float(target_scaler.scale_[0])
            y_fit=target_scaler.transform(np.asarray(ytr_log,float).reshape(-1,1)).ravel()
        elif branch=="UNSCALED_CONTROL":
            y_fit=np.asarray(ytr_log,float)
        else: raise ValueError(branch)

        if family=="MLP":
            xin=StandardScaler().fit(xtr_flat)
            tr=xin.transform(xtr_flat); ev=xin.transform(xev_flat)
            cfg_use=dict(cfg); cfg_use["hidden_layer_sizes"]=tuple(cfg_use["hidden_layer_sizes"])
            torch.manual_seed(seed); np.random.seed(seed)
            model=MLPRegressor(hidden_layer_sizes=cfg_use["hidden_layer_sizes"],activation="relu",solver="adam",
                alpha=cfg_use["alpha"],learning_rate_init=0.001,max_iter=cfg_use["max_iter"],early_stopping=False,
                shuffle=True,random_state=seed,tol=1e-5,n_iter_no_change=30)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(tr,y_fit)
                warnings_seen=[str(w.message) for w in caught]
            raw=np.asarray(model.predict(ev),float)
            history={"training_loss_curve":model.loss_curve_.tolist(),"n_iter":int(model.n_iter_),"final_training_loss":float(model.loss_),
                     "converged_before_max_iter":bool(model.n_iter_<cfg_use["max_iter"]),"coefs_shapes":[list(x.shape) for x in model.coefs_]}
        elif family=="GRU":
            ntr,tlen,nf=xtr_seq.shape
            xin=StandardScaler().fit(xtr_seq.reshape(-1,nf))
            tr=xin.transform(xtr_seq.reshape(-1,nf)).reshape(ntr,tlen,nf)
            nev=xev_seq.shape[0]
            ev=xin.transform(xev_seq.reshape(-1,nf)).reshape(nev,tlen,nf)
            torch.manual_seed(seed); np.random.seed(seed)
            model=core.TinyGRU(nf,int(cfg["hidden_size"]))
            opt=torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=float(cfg["weight_decay"]))
            xt=torch.as_tensor(tr,dtype=torch.float32); yt=torch.as_tensor(y_fit,dtype=torch.float32)
            losses=[]; grad_norms=[]
            for _ in range(int(cfg["epochs"])):
                model.train(); opt.zero_grad(set_to_none=True)
                loss=torch.mean((model(xt)-yt)**2); loss.backward()
                gn=torch.sqrt(sum((p.grad.detach()**2).sum() for p in model.parameters() if p.grad is not None))
                grad_norms.append(float(gn.detach().cpu().item()))
                opt.step()
                model.eval()
                with torch.no_grad(): losses.append(float(torch.mean((model(xt)-yt)**2).cpu().item()))
            model.eval()
            with torch.no_grad(): raw=model(torch.as_tensor(ev,dtype=torch.float32)).cpu().numpy().astype(float)
            history={"epoch_training_loss_post_update":losses,"gradient_global_norm_pre_update":grad_norms,
                     "final_training_loss":losses[-1],"n_epochs":int(cfg["epochs"]),
                     "parameter_count":int(sum(p.numel() for p in model.parameters()))}
        else: raise ValueError(family)

        pred=(raw*target_sd+target_mu) if branch=="TARGET_SCALED" else raw
        if not np.isfinite(pred).all(): raise FloatingPointError("non-finite prediction")
        return {"status":"PASS","prediction":pred,"raw_model_output":raw,"training_target_mean":target_mu,
                "training_target_scale":target_sd,"runtime_seconds":time.perf_counter()-start,
                "warnings":warnings_seen,"history":history,"error":None}
    except Exception as e:
        return {"status":"FIT_FAILED","prediction":None,"raw_model_output":None,"training_target_mean":target_mu,
                "training_target_scale":target_sd,"runtime_seconds":time.perf_counter()-start,
                "warnings":warnings_seen,"history":history,"error":repr(e)}

def append_jsonl(path,obj):
    with Path(path).open("a",encoding="utf-8") as f:
        f.write(json.dumps(obj,ensure_ascii=False,allow_nan=False)+"\n"); f.flush(); os.fsync(f.fileno())

def md_append(path,text):
    with Path(path).open("a",encoding="utf-8") as f: f.write(text+"\n")

def expected_seed(asset,info,family,fold=None,cfgidx=None):
    task=f"{asset}__future_amihud__h6"
    if fold is None: return seed_hash("final",asset,"future_amihud",6,info,family)
    return seed_hash(task,info,family,cfgidx,fold)

def main():
    started=time.perf_counter()
    lock,source=check_lock()
    marker=OUT/"training_started.marker"
    if marker.exists(): raise RuntimeError("training_started.marker exists; refusing to overwrite/retry this experiment")
    marker.write_text(datetime.now(timezone.utc).isoformat()+"\n",encoding="utf-8")
    (OUT/"predictions").mkdir(exist_ok=True); (OUT/"training_logs").mkdir(exist_ok=True)
    for folder in [OUT/"predictions",OUT/"training_logs"]:
        if any(folder.iterdir()): raise RuntimeError(f"non-empty Phase 13 output subdirectory: {folder}")
    exec_log=OUT/"04_training_execution_log.md"
    exec_log.write_text("# Training execution log\n\nStatus: RUNNING; protocol lock verified before first fit.\n\n",encoding="utf-8")
    cfgreg=json.loads((OUT/"01_frozen_configuration_registry.json").read_text())
    _,_,_,selected,phase6_pred,phase6_metrics,oof_map=load_history()
    df=core.load_panel_full_after_freeze()
    if len(df)!=9888: raise RuntimeError(f"frozen source panel row count changed: {len(df)}")
    if df.duplicated(["asset","period_start_utc"]).any(): raise RuntimeError("duplicate asset/date in certified panel")
    all_pred_rows=[]; unit_rows=[]; fit_counter=0; failed=[]

    for asset in ["BTC","ETH"]:
      task=core.make_task(df,asset,6,"future_amihud")
      ylog=core.logtarget(task["y"]); dates=task["date"]; ends=task["target_end"]
      hist_task=phase6_pred[(asset,"future_amihud",6)]
      hist_test_by_date={(pd.Timestamp(d).isoformat()):float(y) for d,y in zip(hist_task["date"],hist_task["target_log1p"])}
      if len(hist_task["date"])!=1417: raise RuntimeError(f"historical test count changed for {asset}")
      for info in ["SPOT_ONLY","SPOT_PLUS_ALL_DERIVATIVES"]:
        Xflat,Xseq=core.x_for_set(task["Xseq"],core.INFOSETS[info])
        for family in ["MLP","GRU"]:
          task_id=f"{asset}__future_amihud__h6"; selection=selected["selected"][f"{task_id}__{info}"]["models"][family]
          cfg=selection["selected_config"]; cfgidx=int(selection["selected_config_index"])
          splits=[]
          for fold in core.DEV_FOLDS:
              foldname,start,end=fold; train_idx,val_idx=core.fold_masks(task,fold)
              if not train_idx.any() or not val_idx.any(): raise RuntimeError(f"empty fold {asset}/{info}/{family}/{foldname}")
              if (ends[train_idx]>=start).any() or (ends[val_idx]>=end).any(): raise RuntimeError(f"future target crosses dev boundary {asset}/{foldname}")
              splits.append((foldname,train_idx,val_idx,start,end))
          train_final=(dates<core.SPLIT)&(ends<core.SPLIT)
          test_final=(dates>=core.SPLIT)&(dates<core.SAMPLE_END_EXCLUSIVE)
          if (ends[train_final]>=core.SPLIT).any() or (ends[test_final]>=core.SAMPLE_END_EXCLUSIVE).any(): raise RuntimeError("future target crosses final boundary")
          if int(test_final.sum())!=1417: raise RuntimeError(f"new test origins differ from historical: {test_final.sum()}")
          splits.append(("FINAL_TEST",train_final,test_final,core.SPLIT,core.SAMPLE_END_EXCLUSIVE))
          for splitname,train_idx,eval_idx,start,end in splits:
            seed=expected_seed(asset,info,family,None,cfgidx) if splitname=="FINAL_TEST" else expected_seed(asset,info,family,splitname,cfgidx)
            ytr=ylog[train_idx]; yev=ylog[eval_idx]
            if (ends[train_idx]>=start).any(): raise RuntimeError(f"training label not purged: {asset}/{splitname}")
            for branch in ["UNSCALED_CONTROL","TARGET_SCALED"]:
              fit_counter+=1
              res=fit_once(family,cfg,branch,Xflat[train_idx],Xseq[train_idx],ytr,Xflat[eval_idx],Xseq[eval_idx],seed,
                           f"{asset}_{family}_{info}_{splitname}_{branch}")
              fit_id=f"{asset}_{family}_{info}_{splitname}_{branch}"
              predfile=OUT/"predictions"/f"{fit_id}.csv"
              logfile=OUT/"training_logs"/f"{fit_id}.json"
              warningtext=" | ".join(res["warnings"])
              log={"fit_id":fit_id,"asset":asset,"model_family":family,"information_set":info,"split":splitname,
                   "branch":branch,"status":res["status"],"error":res["error"],"selected_config":cfg,"selected_config_index":cfgidx,
                   "seed":seed,"train_n":int(train_idx.sum()),"eval_n":int(eval_idx.sum()),
                   "train_first":dates[train_idx].min().isoformat(),"train_last":dates[train_idx].max().isoformat(),
                   "eval_first":dates[eval_idx].min().isoformat(),"eval_last":dates[eval_idx].max().isoformat(),
                   "train_target_log1p_mean":float(np.mean(ytr)),"train_target_log1p_sd_population":float(np.std(ytr,ddof=0)),
                   "train_target_log1p_min":float(np.min(ytr)),"train_target_log1p_max":float(np.max(ytr)),
                   "target_scaler_mean":res["training_target_mean"],"target_scaler_scale":res["training_target_scale"],
                   "target_scaler_fitted_on_training_labels_only":branch=="TARGET_SCALED",
                   "runtime_seconds":res["runtime_seconds"],"warnings":res["warnings"],**res["history"]}
              write_json(logfile,log); append_jsonl(OUT/"training_logs"/"all_fits.jsonl",log)
              if res["status"]!="PASS":
                  failed.append(fit_id); unit_rows.append({**log,"mae_log1p":None})
                  md_append(exec_log,f"- {fit_id}: FIT_FAILED — {res['error']}")
                  continue
              actual_dates=dates[eval_idx]; ends_eval=ends[eval_idx]
              hmap=(oof_map[(asset,info,family)] if splitname!="FINAL_TEST" else historical_prediction_map(hist_task,info,family))
              output=[]
              for i,(d,te,p) in enumerate(zip(actual_dates,ends_eval,res["prediction"])):
                  ds=d.isoformat(); hist=hmap.get(ds)
                  row={"asset":asset,"model_family":family,"information_set":info,"split":splitname,
                       "branch":branch,"origin_utc":ds,"target_end_utc":te.isoformat(),"target_log1p":float(yev[i]),
                       "target_raw":float(task["y"][eval_idx][i]),"prediction_log1p_unclipped":float(p),
                       "network_output_before_target_inverse":float(res["raw_model_output"][i]),
                       "phase6_historical_prediction_log1p":hist,
                       "control_original_phase6_zero_clipped_prediction_log1p":float(max(p,0.0)) if branch=="UNSCALED_CONTROL" else None,
                       "train_target_scaler_mean":res["training_target_mean"],"train_target_scaler_scale":res["training_target_scale"],
                       "seed":seed,"selected_config_index":cfgidx,"fit_status":res["status"]}
                  output.append(row); all_pred_rows.append(row)
              pd.DataFrame(output).to_csv(predfile,index=False)
              m=metric(yev,res["prediction"])
              if splitname=="FINAL_TEST" and branch=="UNSCALED_CONTROL":
                  clipped=np.maximum(res["prediction"],0.0); m["mae_log1p_phase6_output_clipped_replay"]=metric(yev,clipped)["mae_log1p"]
              record={**log,**m,"warnings_text":warningtext}
              unit_rows.append(record)
              md_append(exec_log,f"- {fit_id}: PASS; MAE(log1p)={m['mae_log1p']:.12g}; train n={record['train_n']}; eval n={record['eval_n']}; {res['runtime_seconds']:.3f}s")
    pd.DataFrame(all_pred_rows).to_csv(OUT/"predictions"/"all_origin_predictions.csv.gz",index=False,compression="gzip")
    pd.DataFrame(unit_rows).to_csv(OUT/"08_fold_level_results.csv",index=False)
    write_json(OUT/"training_logs"/"execution_summary.json",{"fit_count":fit_counter,"failed_fit_ids":failed,"status":"PARTIAL" if failed else "PASS","wall_seconds_so_far":time.perf_counter()-started})
    (OUT/"04_training_execution_log.md").write_text(
        "# Training execution log\n\n"+f"Started {Path(marker).read_text().strip()}; fits attempted: {fit_counter}; failures: {len(failed)}.\n\n"+
        "\n".join(f"- {r['fit_id']}: {r['status']} — {r.get('mae_log1p','')}" for r in unit_rows)+
        ("\n\nFailed units: "+", ".join(failed) if failed else "\n\nNo fitting unit failed.")+"\n",encoding="utf-8")
    print(json.dumps({"fit_count":fit_counter,"failed":len(failed),"prediction_rows":len(all_pred_rows),"wall_seconds":time.perf_counter()-started},indent=2))

    # Analysis occurs only after all locked fits have completed.
    pred_df=pd.DataFrame(all_pred_rows)
    hist_metrics=json.loads((P6/"final_test_metrics.json").read_text())
    hist_met={(t["asset"],t["target"],t["horizon_intervals"]):t for t in hist_metrics["tasks"]}
    baseline_rows=[]; diag_rows=[]; mlp_rows=[]; gru_rows=[]
    test_df=pred_df[pred_df.split=="FINAL_TEST"]
    diagnostics_index={}
    for asset in ["BTC","ETH"]:
      hist_task=phase6_pred[(asset,"future_amihud",6)]
      y=np.asarray(hist_task["target_log1p"],float); dates=[pd.Timestamp(x).isoformat() for x in hist_task["date"]]
      for info in ["SPOT_ONLY","SPOT_PLUS_ALL_DERIVATIVES"]:
        basepred=hist_task["predictions_log1p"][info]
        for family in ["MLP","GRU"]:
          historical=np.asarray(basepred[family],float)
          for branch in ["UNSCALED_CONTROL","TARGET_SCALED"]:
            sub=test_df[(test_df.asset==asset)&(test_df.information_set==info)&(test_df.model_family==family)&(test_df.branch==branch)].sort_values("origin_utc")
            if len(sub)!=len(y) or sub.origin_utc.tolist()!=dates: raise RuntimeError(f"test origin mismatch {asset}/{info}/{family}/{branch}")
            if not np.allclose(sub.target_log1p.to_numpy(float),y,rtol=0,atol=1e-15): raise RuntimeError(f"test target mismatch {asset}/{info}/{family}/{branch}")
            p=sub.prediction_log1p_unclipped.to_numpy(float)
            phase6clip=np.maximum(p,0.0)
            legacy_metric=metric(y,phase6clip)
            saved_hist_metric=float(hist_met[(asset,"future_amihud",6)]["metrics"][info][family]["mae_log1p"])
            hist_mae=metric(y,historical)["mae_log1p"]
            row={"asset":asset,"model_family":family,"information_set":info,"historical_phase6_saved_mae_log1p":saved_hist_metric,
                 "historical_saved_prediction_recomputed_mae_log1p":hist_mae,"unscaled_control_replay_raw_mae_log1p":metric(y,p)["mae_log1p"] if branch=="UNSCALED_CONTROL" else None,
                 "unscaled_control_replay_phase6_clipped_mae_log1p":legacy_metric["mae_log1p"] if branch=="UNSCALED_CONTROL" else None,
                 "target_scaled_mae_log1p":metric(y,p)["mae_log1p"] if branch=="TARGET_SCALED" else None,
                 "n_test":len(y),"split_first":dates[0],"split_last":dates[-1],"status":"RETROSPECTIVE_SENSITIVITY"}
            (mlp_rows if family=="MLP" else gru_rows).append(row)
            log=json.loads((OUT/"training_logs"/f"{asset}_{family}_{info}_FINAL_TEST_{branch}.json").read_text())
            control_log=json.loads((OUT/"training_logs"/f"{asset}_{family}_{info}_FINAL_TEST_UNSCALED_CONTROL.json").read_text())
            for tag,pp in [("HISTORICAL_PHASE6",historical),("CONTROL_REPLAY_RAW",p if branch=="UNSCALED_CONTROL" else None),
                           ("CONTROL_REPLAY_PHASE6_CLIPPED",phase6clip if branch=="UNSCALED_CONTROL" else None),
                           ("TARGET_SCALED",p if branch=="TARGET_SCALED" else None)]:
                if pp is None: continue
                dd={"asset":asset,"model_family":family,"information_set":info,"forecast_version":tag,
                    **describe(y,pp,[control_log["train_target_log1p_max"]])}
                diag_rows.append(dd)
        # baseline model rows, all from same frozen test origins
        for family in ["training_mean","persistence","OLS","Ridge","RBF_SVR","ExtraTrees","HistGradientBoosting"]:
            p=np.asarray(hist_task["predictions_log1p"][info][family],float)
            baseline_rows.append({"asset":asset,"information_set":info,"model_family":family,"historical_phase6_mae_log1p":metric(y,p)["mae_log1p"],
                                  "role":"SIMPLE_BASELINE" if family in ["training_mean","persistence"] else "CLASSICAL_REFERENCE",
                                  "selection_status":"frozen baseline" if family in ["training_mean","persistence"] else "test-period oracle ranking is descriptive only","n_test":len(y)})
        for family in ["MLP","GRU"]:
            fitrow=next(r for r in unit_rows if r["asset"]==asset and r["model_family"]==family and r["information_set"]==info and r["split"]=="FINAL_TEST" and r["branch"]=="TARGET_SCALED")
            bestclass=min([r for r in baseline_rows if r["asset"]==asset and r["information_set"]==info and r["role"]=="CLASSICAL_REFERENCE"],key=lambda x:x["historical_phase6_mae_log1p"])
            baseline_rows.append({"asset":asset,"information_set":info,"model_family":family,"target_scaled_mae_log1p":fitrow["mae_log1p"],
                                  "strongest_classical_family_test_oracle":bestclass["model_family"],"strongest_classical_mae_log1p":bestclass["historical_phase6_mae_log1p"],
                                  "delta_classical_minus_scaled_neural":bestclass["historical_phase6_mae_log1p"]-fitrow["mae_log1p"],
                                  "scaled_neural_beats_classical":fitrow["mae_log1p"]<bestclass["historical_phase6_mae_log1p"],
                                  "role":"RETROSPECTIVE_TEST_ORACLE_COMPARISON","n_test":fitrow["eval_n"]})
    pd.DataFrame(mlp_rows).to_csv(OUT/"05_mlp_sensitivity_results.csv",index=False)
    pd.DataFrame(gru_rows).to_csv(OUT/"06_gru_sensitivity_results.csv",index=False)
    pd.DataFrame(diag_rows).to_csv(OUT/"07_neural_prediction_diagnostics.csv",index=False)
    pd.DataFrame(baseline_rows).to_csv(OUT/"09_strong_baseline_comparison.csv",index=False)

    # Add training-range-based descriptive prediction diagnostics and fold performance.
    fold_df=pd.read_csv(OUT/"08_fold_level_results.csv")
    for i,r in fold_df.iterrows():
        if r.get("status")!="PASS": continue
        fit_id=f"{r['asset']}_{r['model_family']}_{r['information_set']}_{r['split']}_{r['branch']}"
        file=OUT/"predictions"/f"{fit_id}.csv"
        if not file.exists(): continue
        z=pd.read_csv(file); p=z.prediction_log1p_unclipped.to_numpy(float); yy=z.target_log1p.to_numpy(float)
        fitlog=json.loads((OUT/"training_logs"/f"{fit_id}.json").read_text())
        ae=np.abs(p-yy); lim=10*fitlog["train_target_log1p_max"]
        fold_df.loc[i,"prediction_min"]=float(p.min()); fold_df.loc[i,"prediction_median"]=float(np.median(p)); fold_df.loc[i,"prediction_max"]=float(p.max())
        fold_df.loc[i,"prediction_outside_eval_target_range_fraction"]=float(np.mean((p<yy.min())|(p>yy.max())))
        fold_df.loc[i,"n_errors_above_10x_training_target_max"]=int(np.sum(ae>lim))
    fold_df.to_csv(OUT/"08_fold_level_results.csv",index=False)

    # Equal-weight ensemble sensitivity and auditable category-average reconstruction.
    ensemble_rows=[]; category_rows=[]; original_categories={}
    categories={"linear":["OLS","Ridge"],"kernel":["RBF_SVR"],"bagged_trees":["ExtraTrees"],"boosting":["HistGradientBoosting"],"neural":["MLP","GRU"],"equal_weight_ensemble":["equal_weight_ensemble"]}
    for asset in ["BTC","ETH"]:
      taskrow=phase6_pred[(asset,"future_amihud",6)]; y=np.asarray(taskrow["target_log1p"],float)
      for info in ["SPOT_ONLY","SPOT_PLUS_ALL_DERIVATIVES"]:
        comp={name:np.asarray(taskrow["predictions_log1p"][info][name],float) for name in ["Ridge","RBF_SVR","ExtraTrees","HistGradientBoosting","MLP","GRU"]}
        historical_ens=np.asarray(taskrow["predictions_log1p"][info]["equal_weight_ensemble"],float)
        rebuilt=np.mean(np.column_stack([comp[x] for x in ["Ridge","RBF_SVR","ExtraTrees","HistGradientBoosting","MLP","GRU"]]),axis=1)
        max_rebuild=float(np.max(np.abs(historical_ens-rebuilt)))
        scaledcomp=dict(comp)
        for family in ["MLP","GRU"]:
            sub=test_df[(test_df.asset==asset)&(test_df.information_set==info)&(test_df.model_family==family)&(test_df.branch=="TARGET_SCALED")].sort_values("origin_utc")
            scaledcomp[family]=sub.prediction_log1p_unclipped.to_numpy(float)
        new_ens=np.mean(np.column_stack([scaledcomp[x] for x in ["Ridge","RBF_SVR","ExtraTrees","HistGradientBoosting","MLP","GRU"]]),axis=1)
        for version,p in [("PHASE6_SAVED",historical_ens),("PHASE6_RECONSTRUCTED",rebuilt),("RETROSPECTIVE_ENSEMBLE_SENSITIVITY",new_ens)]:
            rec={"asset":asset,"information_set":info,"ensemble_version":version,"mae_log1p":metric(y,p)["mae_log1p"],"rmse_log1p":metric(y,p)["rmse_log1p"],"max_abs_difference_vs_saved":max_rebuild if version=="PHASE6_RECONSTRUCTED" else None,"n_test":len(y),"status":"RETROSPECTIVE_ENSEMBLE_SENSITIVITY" if version.endswith("SENSITIVITY") else "FROZEN_HISTORICAL"}
            for family in ["Ridge","RBF_SVR","ExtraTrees","HistGradientBoosting","MLP","GRU"]:
                rec[f"mean_abs_component_{family}"]=float(np.mean(np.abs((scaledcomp if version.endswith("SENSITIVITY") else comp)[family]))/6.0)
            ensemble_rows.append(rec)
      # Registered category-average statistic uses family-relative MAE changes,
      # averaged inside each category, then equal-weight across six categories.
      info_arrays={}
      for info in ["SPOT_ONLY","SPOT_PLUS_ALL_DERIVATIVES"]:
        arrays={k:np.asarray(v,float) for k,v in taskrow["predictions_log1p"][info].items()}
        info_arrays[info]=arrays
      scaled_arrays={info:dict(arrays) for info,arrays in info_arrays.items()}
      for info in scaled_arrays:
        for family in ["MLP","GRU"]:
            sub=test_df[(test_df.asset==asset)&(test_df.information_set==info)&(test_df.model_family==family)&(test_df.branch=="TARGET_SCALED")].sort_values("origin_utc")
            scaled_arrays[info][family]=sub.prediction_log1p_unclipped.to_numpy(float)
        c=np.mean(np.column_stack([scaled_arrays[info][x] for x in ["Ridge","RBF_SVR","ExtraTrees","HistGradientBoosting","MLP","GRU"]]),axis=1)
        scaled_arrays[info]["equal_weight_ensemble"]=c
      for category,members in categories.items():
        rels=[]
        for family in members:
            a=info_arrays["SPOT_ONLY"][family]; b=info_arrays["SPOT_PLUS_ALL_DERIVATIVES"][family]
            rels.append((metric(y,a)["mae_log1p"]-metric(y,b)["mae_log1p"])/max(metric(y,a)["mae_log1p"],1e-15)*100)
        old=float(np.mean(rels))
        newrels=[]
        for family in members:
            a=scaled_arrays["SPOT_ONLY"][family]; b=scaled_arrays["SPOT_PLUS_ALL_DERIVATIVES"][family]
            newrels.append((metric(y,a)["mae_log1p"]-metric(y,b)["mae_log1p"])/max(metric(y,a)["mae_log1p"],1e-15)*100)
        new=float(np.mean(newrels))
        category_rows.append({"asset":asset,"category":category,"historical_relative_reduction_pct_reconstructed":old,
                              "target_scaled_sensitivity_relative_reduction_pct":new,"change_percentage_points":new-old,
                              "member_models":"|".join(members),"n_test":len(y),"status":"RETROSPECTIVE_SENSITIVITY"})
      avg_old=float(np.mean([r["historical_relative_reduction_pct_reconstructed"] for r in category_rows if r["asset"]==asset]))
      avg_new=float(np.mean([r["target_scaled_sensitivity_relative_reduction_pct"] for r in category_rows if r["asset"]==asset]))
      frozen=json.loads((P6/"incremental_value_results.json").read_text())
      histrow=next(x for x in frozen["tasks"] if x["task_id"]==f"{asset}__future_amihud__h6")
      expected=float(histrow["mean_category_relative_reduction_pct"])
      if abs(avg_old-expected)>1e-7: raise RuntimeError(f"historical category reconstruction mismatch {asset}: {avg_old} vs {expected}")
      original_categories[asset]={"reported_pct":expected,"reconstructed_pct":avg_old,"target_scaled_sensitivity_pct":avg_new,"reconstruction_matches":True}
    pd.DataFrame(ensemble_rows).to_csv(OUT/"10_ensemble_sensitivity.csv",index=False)
    catdf=pd.DataFrame(category_rows)
    for asset in ["BTC","ETH"]:
        avg_old=original_categories[asset]["reconstructed_pct"]; avg_new=original_categories[asset]["target_scaled_sensitivity_pct"]
        catdf.loc[len(catdf)]={"asset":asset,"category":"SIX_CATEGORY_MACRO_AVERAGE","historical_relative_reduction_pct_reconstructed":avg_old,
            "target_scaled_sensitivity_relative_reduction_pct":avg_new,"change_percentage_points":avg_new-avg_old,
            "member_models":"equal-weight macro-average of six registered categories","n_test":1417,"status":"RETROSPECTIVE_SENSITIVITY"}
    catdf.to_csv(OUT/"11_category_average_sensitivity.csv",index=False)

    # Paired temporal moving-block intervals; fixed bootstrap seed per asset/block.
    boot_rows=[]; uncertainty_lines=["# Post-hoc exploratory uncertainty intervals","", "Paired moving-block percentile intervals use 1,000 draws, fixed base seed 20261013, blocks 42 and 84 four-hour rows. These are retrospective descriptive intervals, not confirmatory tests; no p-values or new registered PASS/FAIL decisions are produced.",""]
    for asset_i,asset in enumerate(["BTC","ETH"]):
      taskrow=phase6_pred[(asset,"future_amihud",6)]; y=np.asarray(taskrow["target_log1p"],float); n=len(y)
      for block in [42,84]:
        seed=20261013+asset_i*100+block; rng=np.random.default_rng(seed); nb=int(np.ceil(n/block)); max_start=n-block
        if max_start<0: raise RuntimeError("test shorter than fixed bootstrap block")
        idxmat=[]
        for _ in range(1000):
            starts=rng.integers(0,max_start+1,size=nb); idxmat.append(np.concatenate([np.arange(s,s+block) for s in starts])[:n])
        idxmat=np.asarray(idxmat,dtype=int)
        def record(label,base,plus,info=None,family=None):
            diff=np.abs(y-np.asarray(base,float))-np.abs(y-np.asarray(plus,float))
            draws=np.mean(diff[idxmat],axis=1)
            rec={"asset":asset,"block_length":block,"comparison":label,"family":family,"information_set":info,
                 "mean_delta":float(np.mean(diff)),"ci95_lower":float(np.quantile(draws,.025)),"ci95_upper":float(np.quantile(draws,.975)),
                 "probability_delta_positive":float(np.mean(draws>0)),"replications":1000,"seed":seed,"status":"POST_HOC_EXPLORATORY_INTERVAL"}
            boot_rows.append(rec); return rec
        for family in ["MLP","GRU"]:
            for info in ["SPOT_ONLY","SPOT_PLUS_ALL_DERIVATIVES"]:
                hist=np.asarray(taskrow["predictions_log1p"][info][family],float)
                scaled=test_df[(test_df.asset==asset)&(test_df.information_set==info)&(test_df.model_family==family)&(test_df.branch=="TARGET_SCALED")].sort_values("origin_utc").prediction_log1p_unclipped.to_numpy(float)
                record("HISTORICAL_PHASE6_LOSS_MINUS_TARGET_SCALED_LOSS",hist,scaled,info,family)
            a=test_df[(test_df.asset==asset)&(test_df.information_set=="SPOT_ONLY")&(test_df.model_family==family)&(test_df.branch=="TARGET_SCALED")].sort_values("origin_utc").prediction_log1p_unclipped.to_numpy(float)
            b=test_df[(test_df.asset==asset)&(test_df.information_set=="SPOT_PLUS_ALL_DERIVATIVES")&(test_df.model_family==family)&(test_df.branch=="TARGET_SCALED")].sort_values("origin_utc").prediction_log1p_unclipped.to_numpy(float)
            record("TARGET_SCALED_SPOTONLY_LOSS_MINUS_PLUSDERIVATIVES_LOSS",a,b,None,family)
        for info in ["SPOT_ONLY","SPOT_PLUS_ALL_DERIVATIVES"]:
            hist=np.asarray(taskrow["predictions_log1p"][info]["equal_weight_ensemble"],float)
            scaled=pd.read_csv(OUT/"predictions"/f"{asset}_MLP_{info}_FINAL_TEST_TARGET_SCALED.csv").sort_values("origin_utc").prediction_log1p_unclipped.to_numpy(float)
            scaled_gru=pd.read_csv(OUT/"predictions"/f"{asset}_GRU_{info}_FINAL_TEST_TARGET_SCALED.csv").sort_values("origin_utc").prediction_log1p_unclipped.to_numpy(float)
            old_components=[np.asarray(taskrow["predictions_log1p"][info][x],float) for x in ["Ridge","RBF_SVR","ExtraTrees","HistGradientBoosting","MLP","GRU"]]
            new_components=[np.asarray(taskrow["predictions_log1p"][info][x],float) for x in ["Ridge","RBF_SVR","ExtraTrees","HistGradientBoosting"]]+[scaled,scaled_gru]
            record("PHASE6_ENSEMBLE_LOSS_MINUS_SCALED_COMPONENT_ENSEMBLE_LOSS",np.mean(np.column_stack(old_components),axis=1),np.mean(np.column_stack(new_components),axis=1),info,"equal_weight_ensemble")
    pd.DataFrame(boot_rows).to_csv(OUT/"predictions"/"paired_block_bootstrap_intervals.csv",index=False)
    # Convert tabular intervals to a concise auditable Markdown supplement.
    bootdf=pd.DataFrame(boot_rows)
    for r in boot_rows:
        uncertainty_lines.append(f"- {r['asset']}, block={r['block_length']}, {r['comparison']}, {r['family'] or 'all'} {r['information_set'] or ''}: mean Δ={r['mean_delta']:.8g}, nominal 95% CI [{r['ci95_lower']:.8g}, {r['ci95_upper']:.8g}], Pr(Δ>0)={r['probability_delta_positive']:.3f}.")
    (OUT/"12_exploratory_uncertainty.md").write_text("\n".join(uncertainty_lines)+"\n",encoding="utf-8")

    # Build finalized test summaries and plots from actual stored predictions only.
    final_summary={"frozen_phase6_passes":"2/18 unchanged","category_average_reconstruction":original_categories,
                   "failed_fit_ids":failed,"status":"PARTIAL" if failed else "COMPLETE_RETROSPECTIVE_SENSITIVITY"}
    new_performance={}
    for asset in ["BTC","ETH"]:
      taskrow=phase6_pred[(asset,"future_amihud",6)]; y=np.asarray(taskrow["target_log1p"],float)
      new_performance[asset]={}
      for info in ["SPOT_ONLY","SPOT_PLUS_ALL_DERIVATIVES"]:
       mrow={}
       for family in ["MLP","GRU"]:
         hist=np.asarray(taskrow["predictions_log1p"][info][family],float)
         sub=test_df[(test_df.asset==asset)&(test_df.information_set==info)&(test_df.model_family==family)&(test_df.branch=="TARGET_SCALED")].sort_values("origin_utc")
         scaled=sub.prediction_log1p_unclipped.to_numpy(float)
         control=pd.read_csv(OUT/"predictions"/f"{asset}_{family}_{info}_FINAL_TEST_UNSCALED_CONTROL.csv").sort_values("origin_utc").prediction_log1p_unclipped.to_numpy(float)
         base_hist=metric(y,hist)["mae_log1p"]; treatment=metric(y,scaled)["mae_log1p"]; control_raw=metric(y,control)["mae_log1p"]
         clipped_replay=metric(y,np.maximum(control,0)) if family=="MLP" else metric(y,np.maximum(control,0))
         best=min([("TrainingMean",metric(y,taskrow["predictions_log1p"]["SPOT_ONLY"]["training_mean"])["mae_log1p"]),
                   ("Persistence",metric(y,taskrow["predictions_log1p"]["SPOT_ONLY"]["persistence"])["mae_log1p"]),
                   ("OLS",metric(y,taskrow["predictions_log1p"]["SPOT_ONLY"]["OLS"])["mae_log1p"]),
                   ("Ridge",metric(y,taskrow["predictions_log1p"]["SPOT_ONLY"]["Ridge"])["mae_log1p"]),
                   ("RBF_SVR",metric(y,taskrow["predictions_log1p"]["SPOT_ONLY"]["RBF_SVR"])["mae_log1p"]),
                   ("ExtraTrees",metric(y,taskrow["predictions_log1p"]["SPOT_ONLY"]["ExtraTrees"])["mae_log1p"]),
                   ("HistGradientBoosting",metric(y,taskrow["predictions_log1p"]["SPOT_ONLY"]["HistGradientBoosting"])["mae_log1p"])],key=lambda z:z[1])
         mrow[family]={"historical_phase6_saved_mae_log1p":base_hist,"control_replay_raw_mae_log1p":control_raw,
                       "control_replay_clipped_mae_log1p":metric(y,np.maximum(control,0))["mae_log1p"],
                       "scaled_mae_log1p":treatment,"historical_to_scaled_mae_relative_reduction_pct":100*(base_hist-treatment)/base_hist,
                       "scaled_spotonly_or_plus_mae_log1p":treatment,"strongest_classical_test_oracle":best[0],"strongest_classical_test_oracle_mae_log1p":best[1],
                       "scaled_beats_strongest_spotonly_classical_test_oracle":bool(treatment<best[1]),"n_test":len(y),
                       "control_clip_reproduction_abs_diff_vs_saved":abs(metric(y,np.maximum(control,0))["mae_log1p"]-base_hist)}
       new_performance[asset][info]=mrow
      for family in ["MLP","GRU"]:
       a=new_performance[asset]["SPOT_ONLY"][family]["scaled_mae_log1p"]
       b=new_performance[asset]["SPOT_PLUS_ALL_DERIVATIVES"][family]["scaled_mae_log1p"]
       new_performance[asset][family+"_scaled_derivatives_delta_mae"]=a-b
       new_performance[asset][family+"_scaled_derivatives_relative_reduction_pct"]=100*(a-b)/a if a else None
    write_json(OUT/"analysis_summary.json",{"test_metrics":new_performance,"category_average":original_categories,"bootstrap_intervals":boot_rows,"reproduction_check":"see reproducibility report"})
    make_figures(test_df,phase6_pred,catdf,ensemble_rows)
    print(json.dumps({"analysis_summary":str(OUT/"analysis_summary.json"),"figures":sorted(p.name for p in (OUT/"figures").glob("*.png"))},indent=2))

def make_figures(test_df,phase6_pred,catdf,ensemble_rows):
    figdir=OUT/"figures"; figdir.mkdir(exist_ok=True)
    # 1: saved historical versus new scaled neural forecast distributions.
    fig,axs=plt.subplots(2,2,figsize=(11,8))
    for i,asset in enumerate(["BTC","ETH"]):
      t=phase6_pred[(asset,"future_amihud",6)]
      for j,family in enumerate(["MLP","GRU"]):
        ax=axs[i,j]
        for info,ls in [("SPOT_ONLY","-"),("SPOT_PLUS_ALL_DERIVATIVES","--")]:
          hist=np.asarray(t["predictions_log1p"][info][family],float)
          q=test_df[(test_df.asset==asset)&(test_df.information_set==info)&(test_df.model_family==family)&(test_df.branch=="TARGET_SCALED")].prediction_log1p_unclipped.to_numpy(float)
          ax.hist(hist,bins=50,density=True,histtype="step",linestyle=ls,label=f"Historical {info}")
          ax.hist(q,bins=50,density=True,histtype="step",linestyle=ls,linewidth=2,label=f"Scaled {info}")
        ax.set_title(f"{asset} {family}");ax.set_xlabel("Prediction on log1p target scale");ax.set_ylabel("Density");ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(figdir/"figure_1_forecast_distributions.png",dpi=160);plt.close(fig)
    # 2: historical versus scaled test MAE.
    fig,axs=plt.subplots(2,2,figsize=(11,8))
    for i,asset in enumerate(["BTC","ETH"]):
      task=phase6_pred[(asset,"future_amihud",6)];y=np.asarray(task["target_log1p"],float)
      for j,family in enumerate(["MLP","GRU"]):
       ax=axs[i,j]; names=[]; vals=[]
       for info in ["SPOT_ONLY","SPOT_PLUS_ALL_DERIVATIVES"]:
        names.append(info.replace("SPOT_PLUS_ALL_DERIVATIVES","S1").replace("SPOT_ONLY","S0")+" historical");vals.append(metric(y,task["predictions_log1p"][info][family])["mae_log1p"])
        z=test_df[(test_df.asset==asset)&(test_df.information_set==info)&(test_df.model_family==family)&(test_df.branch=="TARGET_SCALED")].prediction_log1p_unclipped.to_numpy(float)
        names.append(info.replace("SPOT_PLUS_ALL_DERIVATIVES","S1").replace("SPOT_ONLY","S0")+" scaled");vals.append(metric(y,z)["mae_log1p"])
       ax.bar(range(4),vals);ax.set_yscale("log");ax.set_xticks(range(4),names,rotation=25,ha="right");ax.set_title(f"{asset} {family}");ax.set_ylabel("MAE, log1p target (log scale)")
    fig.tight_layout();fig.savefig(figdir/"figure_2_neural_mae.png",dpi=160);plt.close(fig)
    # 3: spot only versus derivatives after scaling.
    fig,axs=plt.subplots(1,2,figsize=(11,4.5))
    for i,asset in enumerate(["BTC","ETH"]):
      task=phase6_pred[(asset,"future_amihud",6)];y=np.asarray(task["target_log1p"],float);ax=axs[i]
      for j,family in enumerate(["MLP","GRU"]):
        v=[]
        for info in ["SPOT_ONLY","SPOT_PLUS_ALL_DERIVATIVES"]:
          z=test_df[(test_df.asset==asset)&(test_df.information_set==info)&(test_df.model_family==family)&(test_df.branch=="TARGET_SCALED")].prediction_log1p_unclipped.to_numpy(float);v.append(metric(y,z)["mae_log1p"])
        ax.plot([j-.12,j+.12],v,marker="o",label=family)
      ax.set_xticks([0,1],["MLP","GRU"]);ax.set_yscale("log");ax.set_title(asset);ax.set_ylabel("MAE on original log1p scale (lower is better)");ax.legend()
    fig.tight_layout();fig.savefig(figdir/"figure_3_information_set_comparison.png",dpi=160);plt.close(fig)
    # 4: original and component-replaced equal-weight ensemble.
    fig,axs=plt.subplots(1,2,figsize=(11,4.5))
    for i,asset in enumerate(["BTC","ETH"]):
      z=[x for x in ensemble_rows if x["asset"]==asset and x["ensemble_version"] in ["PHASE6_SAVED","RETROSPECTIVE_ENSEMBLE_SENSITIVITY"]]
      labels=[];vals=[]
      for r in z: labels.append(r["information_set"].replace("SPOT_ONLY","S0").replace("SPOT_PLUS_ALL_DERIVATIVES","S1")+" "+("old" if r["ensemble_version"]=="PHASE6_SAVED" else "scaled"));vals.append(r["mae_log1p"])
      axs[i].bar(range(len(vals)),vals);axs[i].set_yscale("log");axs[i].set_xticks(range(len(vals)),labels,rotation=25,ha="right");axs[i].set_title(asset);axs[i].set_ylabel("Ensemble MAE, log1p scale")
    fig.tight_layout();fig.savefig(figdir/"figure_4_ensemble_sensitivity.png",dpi=160);plt.close(fig)
    # 5: category average decomposition.
    fig,axs=plt.subplots(1,2,figsize=(12,4.8))
    for i,asset in enumerate(["BTC","ETH"]):
      sub=catdf[(catdf.asset==asset)&(catdf.category!="SIX_CATEGORY_MACRO_AVERAGE")]
      x=np.arange(len(sub));w=.38
      axs[i].bar(x-w/2,sub.historical_relative_reduction_pct_reconstructed,w,label="Historical")
      axs[i].bar(x+w/2,sub.target_scaled_sensitivity_relative_reduction_pct,w,label="Target-scaled sensitivity")
      axs[i].axhline(0,color="black",linewidth=.7);axs[i].set_xticks(x,sub.category,rotation=25,ha="right");axs[i].set_title(asset);axs[i].set_ylabel("Relative log-MAE reduction (%)");axs[i].legend(fontsize=8)
    fig.tight_layout();fig.savefig(figdir/"figure_5_category_average.png",dpi=160);plt.close(fig)

if __name__=="__main__": main()
