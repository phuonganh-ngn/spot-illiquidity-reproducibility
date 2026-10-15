#!/usr/bin/env python3
"""Fit the locked Phase 14 candidate registry on development folds only."""
from __future__ import annotations
import hashlib, json, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import torch

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"outputs/phase14_strong_baseline_benchmark"
sys.path.insert(0,str(ROOT/"src"))
import phase6_benchmark_core as core
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def jdump(path,obj):
 Path(path).write_text(json.dumps(obj,sort_keys=True,ensure_ascii=False,default=lambda x: x.tolist() if isinstance(x,np.ndarray) else str(x))+"\n",encoding='utf-8')

def stable_seed(*parts): return core.SEED+sum(sum(ord(c) for c in str(p)) for p in parts)

def config_id(i): return f"cfg{i+1:02d}"

def get_candidate_specs(registry,asset,info):
 specs={}
 for family in ["Ridge","ExtraTrees","HistGradientBoosting","CatBoost","XGBoost"]:
  specs[family]=[(i,c) for i,c in enumerate(registry["models"][family]["configs"])]
 phase13=json.loads((OUT.parent/"phase13_neural_target_scaling/01_frozen_configuration_registry.json").read_text())
 for family in ["MLP","GRU"]:
  key=f"{asset}__{family}__{info}"
  cfg=phase13["configs"][key]["selected_config"]
  idx=int(phase13["configs"][key]["selected_config_index"])
  specs[family]=[(idx,cfg)]
 return specs

def fit_once(family,cfg,config_index,asset,info,fold_name,task_id,xflat_train,xseq_train,ytrain,xflat_eval,xseq_eval):
 seed=stable_seed(task_id,info,family,config_index,fold_name)
 started=time.perf_counter(); caught=[]; target_mu=None; target_scale=None
 feature_scaler=None; fit_loss=[]; n_iter=None; converged=None; param_count=None
 model_params={}
 if family=="GRU":
  ntr,tlen,nf=xseq_train.shape
  feature_scaler=StandardScaler().fit(xseq_train.reshape(-1,nf))
  tr=feature_scaler.transform(xseq_train.reshape(-1,nf)).reshape(ntr,tlen,nf)
  nev=xseq_eval.shape[0]
  ev=feature_scaler.transform(xseq_eval.reshape(-1,nf)).reshape(nev,tlen,nf)
 else:
  feature_scaler=StandardScaler().fit(xflat_train)
  tr=feature_scaler.transform(xflat_train); ev=feature_scaler.transform(xflat_eval)
 if family in ["MLP","GRU"]:
  ts=StandardScaler().fit(np.asarray(ytrain,float).reshape(-1,1))
  target_mu=float(ts.mean_[0]); target_scale=float(ts.scale_[0])
  if not np.isfinite(target_scale) or target_scale<=np.finfo(float).eps:
   raise FloatingPointError(f"degenerate training target scale for {asset}/{info}/{family}/{fold_name}")
  yfit=ts.transform(np.asarray(ytrain,float).reshape(-1,1)).ravel()
 else: yfit=np.asarray(ytrain,float)
 try:
  with warnings.catch_warnings(record=True) as rec:
   warnings.simplefilter("always")
   if family=="Ridge":
    model=Ridge(alpha=float(cfg["alpha"])).fit(tr,yfit); pred_scaled=model.predict(ev)
    model_params={"alpha":float(cfg["alpha"])}
    param_count=int(model.coef_.size+np.size(model.intercept_))
   elif family=="ExtraTrees":
    model=ExtraTreesRegressor(n_estimators=int(cfg["n_estimators"]),min_samples_leaf=int(cfg["min_samples_leaf"]),max_features=1.0,random_state=seed,n_jobs=1).fit(tr,yfit)
    pred_scaled=model.predict(ev); model_params=model.get_params(deep=False)
    param_count=int(sum(est.tree_.node_count for est in model.estimators_))
   elif family=="HistGradientBoosting":
    model=HistGradientBoostingRegressor(max_iter=int(cfg["max_iter"]),learning_rate=float(cfg["learning_rate"]),max_leaf_nodes=int(cfg["max_leaf_nodes"]),l2_regularization=1.0,early_stopping=False,random_state=seed).fit(tr,yfit)
    pred_scaled=model.predict(ev); model_params=model.get_params(deep=False)
    param_count=None
   elif family=="CatBoost":
    params={**cfg,"random_seed":seed,"verbose":False,"allow_writing_files":False}
    model=CatBoostRegressor(**params).fit(tr,yfit,verbose=False)
    pred_scaled=model.predict(ev); model_params=model.get_params()
    fit_loss=model.get_evals_result().get("learn",{}).get("RMSE",[])
    param_count=None
   elif family=="XGBoost":
    params={**cfg,"random_state":seed,"verbosity":0,"n_jobs":1,"device":"cpu"}
    model=XGBRegressor(**params).fit(tr,yfit,verbose=False)
    pred_scaled=model.predict(ev); model_params=model.get_params(deep=False)
    param_count=None
   elif family=="MLP":
    params={"hidden_layer_sizes":tuple(cfg["hidden_layer_sizes"]),"activation":"relu","solver":"adam","alpha":float(cfg["alpha"]),"learning_rate_init":0.001,"max_iter":int(cfg["max_iter"]),"early_stopping":False,"shuffle":True,"random_state":seed,"tol":1e-5,"n_iter_no_change":30}
    model=MLPRegressor(**params).fit(tr,yfit); pred_scaled=model.predict(ev)
    fit_loss=model.loss_curve_; n_iter=int(model.n_iter_); converged=(n_iter<int(cfg["max_iter"]))
    model_params=model.get_params(deep=False); param_count=int(sum(w.size for w in model.coefs_)+sum(b.size for b in model.intercepts_))
   elif family=="GRU":
    from phase6_benchmark_core import TinyGRU
    torch.manual_seed(seed); np.random.seed(seed)
    model=TinyGRU(nf,int(cfg["hidden_size"]))
    opt=torch.optim.Adam(model.parameters(),lr=0.005,weight_decay=float(cfg["weight_decay"]))
    xt=torch.as_tensor(tr,dtype=torch.float32); yt=torch.as_tensor(yfit,dtype=torch.float32)
    for _ in range(int(cfg["epochs"])):
     model.train();opt.zero_grad(set_to_none=True);loss=torch.mean((model(xt)-yt)**2);loss.backward();opt.step();fit_loss.append(float(loss.detach().cpu()))
    model.eval()
    with torch.no_grad(): pred_scaled=model(torch.as_tensor(ev,dtype=torch.float32)).cpu().numpy()
    n_iter=int(cfg["epochs"]);converged=None
    param_count=int(sum(p.numel() for p in model.parameters()))
    model_params={"hidden_size":int(cfg["hidden_size"]),"epochs":int(cfg["epochs"]),"weight_decay":float(cfg["weight_decay"]),"learning_rate":0.005,"sequence_length":6,"layers":1,"output":"linear from last hidden state","batch":"full"}
   else: raise ValueError(f"Unknown family {family}")
   caught=[str(w.message) for w in rec]
  pred=np.asarray(pred_scaled,float)
  if family in ["MLP","GRU"]: pred=pred*target_scale+target_mu
  if not np.isfinite(pred).all(): raise FloatingPointError("non-finite predictions")
  record={"fit_status":"PASS","error":None,"family":family,"asset":asset,"information_set":info,"fold":fold_name,"config_id":config_id(config_index),"config_index":int(config_index),"config":cfg,"seed":int(seed),"train_n":int(len(ytrain)),"eval_n":int(len(pred)),"runtime_seconds":time.perf_counter()-started,"warnings":caught,"n_iter":n_iter,"converged_before_max_iter":converged,"training_loss_curve":fit_loss,"final_training_loss":float(fit_loss[-1]) if len(fit_loss) else None,"target_scaler_mean":target_mu,"target_scaler_scale":target_scale,"target_scaler_training_only":family in ["MLP","GRU"],"feature_scaler_train_only":True,"n_parameters_or_tree_nodes":param_count,"model_params":model_params}
  return pred,record
 except Exception as e:
  return np.full(len(ytrain)*0+len(xflat_eval),np.nan),{"fit_status":"FAIL","error":f"{type(e).__name__}: {e}","family":family,"asset":asset,"information_set":info,"fold":fold_name,"config_id":config_id(config_index),"config_index":int(config_index),"config":cfg,"seed":int(seed),"train_n":int(len(ytrain)),"eval_n":int(len(xflat_eval)),"runtime_seconds":time.perf_counter()-started,"warnings":caught,"target_scaler_mean":target_mu,"target_scaler_scale":target_scale}

def safe_mae(y,p):
 m=np.isfinite(y)&np.isfinite(p)
 if not m.any(): return None
 return float(np.mean(np.abs(np.asarray(y)[m]-np.asarray(p)[m])))

def main():
 lock=json.loads((OUT/"pretraining_lock.json").read_text())
 if lock.get("status")!="LOCKED_BEFORE_FIT": raise SystemExit("Phase 14 pretraining lock missing")
 expected=(OUT/"pretraining_lock.sha256").read_text().split()[0]
 if sha(OUT/"pretraining_lock.json")!=expected: raise SystemExit("Pretraining lock checksum mismatch")
 reg=json.loads((OUT/"03_frozen_model_registry.json").read_text())
 if sha(OUT/"03_frozen_model_registry.json")!=lock["files"]["03_frozen_model_registry.json"]: raise SystemExit("Frozen model registry changed")
 source_manifest=json.loads((OUT/"01_source_manifest_before.json").read_text())
 for rel,meta in source_manifest["source_files"].items():
  p=ROOT/rel
  if not p.is_file() or sha(p)!=meta["sha256"]: raise SystemExit(f"Frozen source changed after audit: {rel}")
 df=core.load_panel_dev_only()
 if (df.period_start_utc>=core.SPLIT).any(): raise SystemExit("Test data entered development loader")
 logpath=OUT/"training_logs/development_fit_log.jsonl"
 logpath.write_text("",encoding="utf-8")
 selection_rows=[]; selection_registry={}; all_results=[]
 family_order=reg["models_to_evaluate"]
 for asset in ["BTC","ETH"]:
  task=core.make_task(df,asset,6,"future_amihud")
  if not len(task["y"]): raise SystemExit(f"No data for {asset}")
  ylog=core.logtarget(task["y"]);task_id=f"{asset}__future_amihud__h6"
  for info in ["SPOT_ONLY","SPOT_PLUS_ALL_DERIVATIVES"]:
   xflat,xseq=core.x_for_set(task["Xseq"],core.INFOSETS[info])
   folds=[];fold_masks=[]
   for fold in core.DEV_FOLDS:
    name,start,end=fold;tr,va=core.fold_masks(task,fold)
    if (task["target_end"][tr]>=start).any() or (task["target_end"][va]>=end).any(): raise SystemExit("Horizon purge failed")
    fold_masks.append((name,tr,va));folds.append(name)
   family_selected={}; family_cv_rows=[]; dev_predictions={}
   specs=get_candidate_specs(reg,asset,info)
   for family,candidates in specs.items():
    candidate_records=[]
    for cfgidx,cfg in candidates:
     pred_all=np.full(len(ylog),np.nan); fold_scores=[];fit_records=[]
     for foldname,tr,va in fold_masks:
      p,record=fit_once(family,cfg,cfgidx,asset,info,foldname,task_id,xflat[tr],xseq[tr],ylog[tr],xflat[va],xseq[va])
      pred_all[va]=p
      record.update({"train_first":task["date"][tr].min().isoformat(),"train_last":task["date"][tr].max().isoformat(),"train_target_end_last":task["target_end"][tr].max().isoformat(),"eval_first":task["date"][va].min().isoformat(),"eval_last":task["date"][va].max().isoformat(),"eval_target_end_last":task["target_end"][va].max().isoformat()})
      fit_records.append(record);fold_scores.append({"fold":foldname,"n":int(va.sum()),"mae_log1p":safe_mae(ylog[va],p)})
      with logpath.open("a",encoding="utf-8") as f:f.write(json.dumps(record,default=str)+"\n")
      if record["fit_status"]=="PASS":
       save=pd.DataFrame({"asset":asset,"information_set":info,"family":family,"config_id":config_id(cfgidx),"fold":foldname,"origin":task["date"][va].astype(str),"target_end":task["target_end"][va].astype(str),"target_raw":task["y"][va],"target_log1p":ylog[va],"prediction_log1p":p})
       save.to_csv(OUT/f"development_folds/{asset}_{info}_{family}_{config_id(cfgidx)}_{foldname}.csv.gz",index=False,compression="gzip")
     valid=np.isfinite(pred_all)
     pooled=safe_mae(ylog[valid],pred_all[valid])
     cr={"family":family,"config_index":int(cfgidx),"config_id":config_id(cfgidx),"config":cfg,"pooled_dev_mae_log1p":pooled,"pooled_n":int(valid.sum()),"fold_scores":fold_scores,"fit_status":"PASS" if valid.sum()==int(sum(x[2].sum() for x in fold_masks)) else "PARTIAL_OR_FAIL","fit_count":len(fit_records),"fit_failure_count":sum(r["fit_status"]!="PASS" for r in fit_records),"warning_count":sum(bool(r.get("warnings")) for r in fit_records),"runtime_seconds":float(sum(r.get("runtime_seconds",0) for r in fit_records)),"parameter_count_last_fold":fit_records[-1].get("n_parameters_or_tree_nodes") if fit_records else None}
     candidate_records.append((cr,pred_all,fit_records))
    valid_candidates=[x for x in candidate_records if x[0]["fit_status"]=="PASS" and x[0]["pooled_dev_mae_log1p"] is not None]
    if not valid_candidates: raise SystemExit(f"No valid dev config for {asset}/{info}/{family}")
    chosen=min(valid_candidates,key=lambda x:(x[0]["pooled_dev_mae_log1p"],x[0]["config_index"]))
    cr,pred_all,fit_records=chosen
    cr["selected_within_family"]=True
    family_selected[family]={"config_index":cr["config_index"],"config_id":cr["config_id"],"config":cr["config"],"pooled_dev_mae_log1p":cr["pooled_dev_mae_log1p"],"pooled_n":cr["pooled_n"],"fold_scores":cr["fold_scores"],"runtime_seconds":cr["runtime_seconds"],"fit_status":cr["fit_status"],"warning_count":cr["warning_count"]}
    dev_predictions[family]=pred_all
    for candidate,_,_ in candidate_records:
     candidate["selected_within_family"]=(candidate["config_id"]==cr["config_id"])
     family_cv_rows.append(candidate)
   # References are generated fold-by-fold and use the exact common fold rows.
   for baseline in ["TrainingMean","Persistence"]:
    pred=np.full(len(ylog),np.nan);scores=[]
    for foldname,tr,va in fold_masks:
     if baseline=="TrainingMean": pred[va]=float(ylog[tr].mean())
     else: pred[va]=core.logtarget(task["persistence"][va])
     scores.append({"fold":foldname,"n":int(va.sum()),"mae_log1p":safe_mae(ylog[va],pred[va])})
    dev_predictions[baseline]=pred
    row={"family":baseline,"config_index":0,"config_id":"fixed","config":{"rule":"training mean" if baseline=="TrainingMean" else "frozen persistence"},"pooled_dev_mae_log1p":safe_mae(ylog[np.isfinite(pred)],pred[np.isfinite(pred)]),"pooled_n":int(np.isfinite(pred).sum()),"fold_scores":scores,"fit_status":"PASS","fit_count":3,"fit_failure_count":0,"warning_count":0,"runtime_seconds":0.0,"parameter_count_last_fold":0,"selected_within_family":True}
    family_cv_rows.append(row)
   for row in family_cv_rows:
    selection_rows.append({"asset":asset,"information_set":info,**row})
   eligible=[r for r in family_cv_rows if r["fit_status"]=="PASS" and np.isfinite(r["pooled_dev_mae_log1p"])]
   chosen_system=min(eligible,key=lambda r:(r["pooled_dev_mae_log1p"],reg["models"][r["family"]]["complexity_rank"],r["family"]))
   selection_registry[f"{asset}__{info}"]={"asset":asset,"information_set":info,"selected_system":chosen_system["family"],"selected_config_id":chosen_system["config_id"],"selected_config_index":chosen_system["config_index"],"selected_config":chosen_system["config"],"pooled_dev_mae_log1p":chosen_system["pooled_dev_mae_log1p"],"pooled_n":chosen_system["pooled_n"],"family_configs_locked":family_selected,"all_family_dev_mae":{r["family"]:r["pooled_dev_mae_log1p"] for r in eligible},"tie_break_complexity_order":reg["system_selection_complexity_order"]}
   print(f"DEV_SELECTED {asset} {info}: {chosen_system['family']} {chosen_system['config_id']} MAE={chosen_system['pooled_dev_mae_log1p']:.12g}",flush=True)
   jdump(OUT/"development_folds"/f"{asset}_{info}_selection.json",{"asset":asset,"information_set":info,"family_config_rows":family_cv_rows,"selected_system":selection_registry[f"{asset}__{info}"]})
   all_results.extend(family_cv_rows)
 # All configurations and systems are now chosen without parsing the test panel.
 pd.DataFrame(selection_rows).to_csv(OUT/"05_development_model_selection.csv",index=False)
 selection_freeze={"status":"FROZEN_AFTER_DEVELOPMENT_BEFORE_TEST_PARSE","pretraining_lock_sha256":expected,"registry_sha256":sha(OUT/"03_frozen_model_registry.json"),"test_values_parsed":False,"selections":selection_registry,"phase6_test_split":"2023-07-01T00:00:00Z to 2024-03-03T20:00:00Z","retrospective_status":"Phase 6/13 historical test interval previously inspected"}
 jdump(OUT/"selection_freeze.json",selection_freeze)
 selection_hash=sha(OUT/"selection_freeze.json")
 (OUT/"selection_freeze.sha256").write_text(selection_hash+"  selection_freeze.json\n",encoding="utf-8")
 print("SELECTION_FREEZE_SHA256",selection_hash)
 print("TEST_VALUES_PARSED",False)
 print("DEV_FIT_LOG",str(logpath.relative_to(ROOT)))

if __name__=="__main__": main()
