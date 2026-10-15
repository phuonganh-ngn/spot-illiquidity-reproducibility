#!/usr/bin/env python3
"""One-shot final test evaluation after the development selection freeze."""
from __future__ import annotations
import hashlib,json,sys,time,warnings
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"outputs/phase14_strong_baseline_benchmark"
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(OUT/"scripts"))
import phase6_benchmark_core as core
from run_development import fit_once,stable_seed

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def jdump(path,obj): Path(path).write_text(json.dumps(obj,sort_keys=True,ensure_ascii=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x))+"\n",encoding="utf-8")

def holm(pvalues):
 p=np.asarray(pvalues,float);m=len(p);order=np.argsort(p);adj=np.empty(m);running=0.
 for rank,idx in enumerate(order):
  running=max(running,min(1.,(m-rank)*p[idx]));adj[idx]=running
 return adj.tolist()

def mbb_indices(n,reps,block,seed):
 if n<block: raise ValueError(f"test length {n} smaller than block {block}")
 rng=np.random.default_rng(seed);nb=int(np.ceil(n/block));out=[]
 for _ in range(reps):
  starts=rng.integers(0,n-block+1,size=nb)
  out.append(np.concatenate([np.arange(s,s+block) for s in starts])[:n])
 return out

def metric(y,p):
 e=np.asarray(y,float)-np.asarray(p,float)
 return {"mae_log1p":float(np.mean(np.abs(e))),"rmse_log1p":float(np.sqrt(np.mean(e*e))),"median_absolute_error_log1p":float(np.median(np.abs(e))),"mae_original_scale":float(np.mean(np.abs(np.expm1(p)-np.expm1(y))))}

def main():
 lock=json.loads((OUT/"pretraining_lock.json").read_text());lock_hash=(OUT/"pretraining_lock.sha256").read_text().split()[0]
 if sha(OUT/"pretraining_lock.json")!=lock_hash or lock.get("status")!="LOCKED_BEFORE_FIT":raise SystemExit("Pretraining lock mismatch")
 if sha(OUT/"02_locked_experiment_protocol.md")!=lock["files"]["02_locked_experiment_protocol.md"]:raise SystemExit("Protocol changed after lock")
 if sha(OUT/"03_frozen_model_registry.json")!=lock["files"]["03_frozen_model_registry.json"]:raise SystemExit("Model registry changed after lock")
 sf=OUT/"selection_freeze.json";sfhash=(OUT/"selection_freeze.sha256").read_text().split()[0]
 if not sf.is_file() or sha(sf)!=sfhash:raise SystemExit("Development selection freeze missing or altered")
 selection=json.loads(sf.read_text())
 if selection.get("status")!="FROZEN_AFTER_DEVELOPMENT_BEFORE_TEST_PARSE" or selection.get("test_values_parsed") is not False:raise SystemExit("Invalid test freeze status")
 if selection.get("pretraining_lock_sha256")!=lock_hash:raise SystemExit("Selection not made under frozen protocol")
 source_manifest=json.loads((OUT/"01_source_manifest_before.json").read_text())
 for rel,meta in source_manifest["source_files"].items():
  p=ROOT/rel
  if not p.is_file() or sha(p)!=meta["sha256"]:raise SystemExit(f"A frozen historical source changed before test scoring: {rel}")
 # This is the first line in Phase 14 allowed to parse the historical test data.
 df=core.load_panel_full_after_freeze()
 if len(df)!=9888 or set(df.asset.unique())!={"BTC","ETH"}:raise SystemExit("Certified panel structure changed")
 if df.duplicated(["asset","period_start_utc"]).any():raise SystemExit("Duplicate asset/timestamp in panel")
 all_predictions=[];test_rows=[];test_fit_logs=[];primary=[];within=[];neural=[];task_context={}
 infos=["SPOT_ONLY","SPOT_PLUS_ALL_DERIVATIVES"]
 for asset in ["BTC","ETH"]:
  task=core.make_task(df,asset,6,"future_amihud")
  dates=task["date"];ends=task["target_end"]; ylog=core.logtarget(task["y"])
  train=(dates<core.SPLIT)&(ends<core.SPLIT)
  test=(dates>=core.SPLIT)&(ends<core.SAMPLE_END_EXCLUSIVE)
  if not train.any() or not test.any():raise SystemExit(f"Empty fixed sample {asset}")
  if (ends[train]>=core.SPLIT).any() or (ends[test]>=core.SAMPLE_END_EXCLUSIVE).any():raise SystemExit("Target label crossed the train/test boundary")
  testdates=dates[test]
  predictions={};metrics={}; selected_info={}
  for info in infos:
   xflat,xseq=core.x_for_set(task["Xseq"],core.INFOSETS[info])
   sel=selection["selections"][f"{asset}__{info}"]
   family_cfgs=sel["family_configs_locked"]
   predmap={}
   for family in ["Ridge","ExtraTrees","HistGradientBoosting","CatBoost","XGBoost","MLP","GRU"]:
    entry=family_cfgs[family];cfg=entry["config"];cfgidx=int(entry["config_index"])
    seed=stable_seed("final",asset,"future_amihud",6,info,family)
    # Preserve final seed separately; fit implementation and all hyperparameters are shared with dev.
    fit_seed=seed
    # fit_once follows the development fold seed formula, so run an explicit model fit below with the locked final seed.
    p,log=fit_final(family,cfg,cfgidx,asset,info,fit_seed,xflat[train],xseq[train],ylog[train],xflat[test],xseq[test])
    log.update({"fold":"FINAL_TEST","selected_by_development":True,"development_selected_config_id":entry["config_id"],"train_first":dates[train].min().isoformat(),"train_last":dates[train].max().isoformat(),"train_target_end_last":ends[train].max().isoformat(),"test_first":dates[test].min().isoformat(),"test_last":dates[test].max().isoformat(),"test_target_end_last":ends[test].max().isoformat()})
    test_fit_logs.append(log);test_rows.append({"asset":asset,"information_set":info,"model":family,"config_id":entry["config_id"],"config":json.dumps(cfg,sort_keys=True),"development_mae_log1p":entry["pooled_dev_mae_log1p"],"test_n":int(test.sum()),**metric(ylog[test],p),"fit_status":log["fit_status"],"warnings":" | ".join(log.get("warnings",[])),"runtime_seconds":log.get("runtime_seconds"),"parameter_count_or_tree_nodes":log.get("n_parameters_or_tree_nodes"),"converged_before_max_iter":log.get("converged_before_max_iter")})
    predmap[family]=p
    if family in ["MLP","GRU"]:
     neural.append({"asset":asset,"information_set":info,"model":family,"development_config":json.dumps(cfg,sort_keys=True),"development_mae_log1p":entry["pooled_dev_mae_log1p"],"test_mae_log1p":metric(ylog[test],p)["mae_log1p"],"prediction_min":float(np.min(p)),"prediction_median":float(np.median(p)),"prediction_max":float(np.max(p)),"prediction_sd":float(np.std(p)),"target_min":float(np.min(ylog[test])),"target_median":float(np.median(ylog[test])),"target_max":float(np.max(ylog[test])),"absolute_error_p90":float(np.quantile(np.abs(ylog[test]-p),.9)),"absolute_error_p99":float(np.quantile(np.abs(ylog[test]-p),.99)),"fraction_outside_target_range":float(np.mean((p<np.min(ylog[test]))|(p>np.max(ylog[test])))),"warnings":" | ".join(log.get("warnings",[])),"iterations_or_epochs":log.get("n_iter"),"final_training_loss":log.get("final_training_loss"),"runtime_seconds":log.get("runtime_seconds"),"parameter_count":log.get("n_parameters_or_tree_nodes")})
   # Fixed references are generated on the exact same origin vector.
   predmap["TrainingMean"]=np.full(int(test.sum()),float(ylog[train].mean()))
   predmap["Persistence"]=core.logtarget(task["persistence"][test])
   for family in ["TrainingMean","Persistence"]:
    entry=sel["all_family_dev_mae"][family]
    test_rows.append({"asset":asset,"information_set":info,"model":family,"config_id":"fixed","config":"{}","development_mae_log1p":entry,"test_n":int(test.sum()),**metric(ylog[test],predmap[family]),"fit_status":"PASS_REFERENCE","warnings":"","runtime_seconds":0.0,"parameter_count_or_tree_nodes":0,"converged_before_max_iter":None})
   predictions[(asset,info)]=predmap;metrics[(asset,info)]={k:metric(ylog[test],v) for k,v in predmap.items()}
   selected_info[info]=sel
   for model,pred in predmap.items():
    for idx,(origin,end,yraw,z,pz) in enumerate(zip(testdates,ends[test],task["y"][test],ylog[test],pred)):
     all_predictions.append({"asset":asset,"information_set":info,"model":model,"origin":origin.isoformat(),"target_end":end.isoformat(),"target_raw":float(yraw),"target_log1p":float(z),"prediction_log1p":float(pz),"absolute_error_log1p":float(abs(z-pz))})
  # Check same calendar rows for the two sets by construction, retain exact sample.
  if len(predictions[(asset,infos[0])]["Ridge"])!=len(predictions[(asset,infos[1])]["Ridge"]):raise SystemExit("S0/S1 test rows not matched")
  task_context[asset]={"ylog":ylog[test],"dates":testdates,"target_end":ends[test],"train_n":int(train.sum()),"test_n":int(test.sum()),"train_first":dates[train].min().isoformat(),"train_last":dates[train].max().isoformat(),"test_first":dates[test].min().isoformat(),"test_last":dates[test].max().isoformat(),"test_target_end_last":ends[test].max().isoformat(),"selected":selected_info}
  s0=selected_info["SPOT_ONLY"]["selected_system"];s1=selected_info["SPOT_PLUS_ALL_DERIVATIVES"]["selected_system"]
  p0=predictions[(asset,"SPOT_ONLY")][s0];p1=predictions[(asset,"SPOT_PLUS_ALL_DERIVATIVES")][s1]
  d=np.abs(ylog[test]-p0)-np.abs(ylog[test]-p1);delta=float(d.mean());mae0=metric(ylog[test],p0)["mae_log1p"];mae1=metric(ylog[test],p1)["mae_log1p"]
  for block,seed in [(42,20261442+(asset=="ETH")),(84,20261484+(asset=="ETH"))]:
   ix=mbb_indices(len(d),1000,block,int(seed));draw=np.array([d[i].mean() for i in ix]);null_draw=np.array([(d-delta)[i].mean() for i in ix]);praw=float((1+np.count_nonzero(np.abs(null_draw)>=abs(delta)))/(len(null_draw)+1))
   primary.append({"asset":asset,"block_length":block,"selected_s0_model":s0,"selected_s0_config_id":selected_info["SPOT_ONLY"]["selected_config_id"],"selected_s1_model":s1,"selected_s1_config_id":selected_info["SPOT_PLUS_ALL_DERIVATIVES"]["selected_config_id"],"mae_s0":mae0,"mae_s1":mae1,"delta_best_mae_s0_minus_s1":delta,"relative_reduction_pct":100*delta/mae0,"ci95_lower_nominal":float(np.quantile(draw,.025)),"ci95_upper_nominal":float(np.quantile(draw,.975)),"raw_p_two_sided_centered":praw,"replications":1000,"block_seed":int(seed),"probability_delta_positive":float(np.mean(draw>0)),"n_test":len(d),"first_test_origin":testdates.min().isoformat(),"last_test_origin":testdates.max().isoformat(),"status":"RETROSPECTIVE_REANALYSIS"})
  # Every same-family effect, using each family's independently selected dev config in S0/S1.
  for family in ["TrainingMean","Persistence","Ridge","ExtraTrees","HistGradientBoosting","CatBoost","XGBoost","MLP","GRU"]:
   a=predictions[(asset,"SPOT_ONLY")][family];b=predictions[(asset,"SPOT_PLUS_ALL_DERIVATIVES")][family]
   ma=a.__len__();m0=metric(ylog[test],a)["mae_log1p"];m1=metric(ylog[test],b)["mae_log1p"]
   within.append({"asset":asset,"model":family,"s0_config_id":selected_info["SPOT_ONLY"]["family_configs_locked"].get(family,{}).get("config_id","fixed"),"s0_development_mae":selected_info["SPOT_ONLY"]["all_family_dev_mae"][family],"s1_config_id":selected_info["SPOT_PLUS_ALL_DERIVATIVES"]["family_configs_locked"].get(family,{}).get("config_id","fixed"),"s1_development_mae":selected_info["SPOT_PLUS_ALL_DERIVATIVES"]["all_family_dev_mae"][family],"test_s0_mae":m0,"test_s1_mae":m1,"delta_mae_s0_minus_s1":m0-m1,"relative_reduction_pct":100*(m0-m1)/m0,"test_n":ma,"direction":"DERIVATIVES_FAVORABLE_POINT" if m0-m1>0 else "SPOT_ONLY_FAVORABLE_POINT" if m0-m1<0 else "TIE","status":"RETROSPECTIVE_WITHIN_MODEL"})
 # Multiplicity: Holm across two asset tests separately within each bootstrap block family.
 for block in [42,84]:
  rows=[r for r in primary if r["block_length"]==block]
  adj=holm([r["raw_p_two_sided_centered"] for r in rows])
  for r,padj in zip(rows,adj):
   r["holm_p_two_asset_family"]=float(padj)
   if block==42:
    r["result"]="FAVORABLE" if r["delta_best_mae_s0_minus_s1"]>0 and r["ci95_lower_nominal"]>0 and padj<.05 else "ADVERSE" if r["delta_best_mae_s0_minus_s1"]<0 and r["ci95_upper_nominal"]<0 and padj<.05 else "INCONCLUSIVE"
 # Ensure primary table order stable, preserve exact predictions.
 predfile=OUT/"predictions/final_test_predictions.csv.gz"
 pd.DataFrame(all_predictions).to_csv(predfile,index=False,compression="gzip")
 pd.DataFrame(test_rows).to_csv(OUT/"06_complete_model_test_results.csv",index=False)
 pd.DataFrame(within).to_csv(OUT/"08_within_model_information_effects.csv",index=False)
 pd.DataFrame(neural).to_csv(OUT/"09_neural_reliability.csv",index=False)
 pd.DataFrame(primary).to_csv(OUT/"10_bootstrap_inference.csv",index=False)
 with (OUT/"training_logs/final_test_fit_log.jsonl").open("w",encoding="utf-8") as f:
  for r in test_fit_logs:f.write(json.dumps(r,default=str)+"\n")
 jdump(OUT/"final_test_metrics_and_selection.json",{"status":"PASS","retrospective_status":"RETROSPECTIVE_REANALYSIS","pretraining_lock_sha256":lock_hash,"selection_freeze_sha256":sfhash,"selection_was_frozen_before_test_parse":True,"test_values_parsed_after_freeze":True,"tasks":[{"asset":a,**{k:v for k,v in task_context[a].items() if k not in ["ylog","dates","target_end","selected"]},"selected_s0":task_context[a]["selected"]["SPOT_ONLY"],"selected_s1":task_context[a]["selected"]["SPOT_PLUS_ALL_DERIVATIVES"]} for a in ["BTC","ETH"]],"primary_comparisons":primary,"predictions_path":"predictions/final_test_predictions.csv.gz","prediction_rows":len(all_predictions),"fit_units":len(test_fit_logs),"fit_failures":sum(r["fit_status"]!="PASS" for r in test_fit_logs)})
 print("PHASE14_TEST_EVALUATION_COMPLETE",flush=True)
 for r in primary:
  if r["block_length"]==42:print(f"PRIMARY {r['asset']} {r['selected_s0_model']}->{r['selected_s1_model']} MAE {r['mae_s0']:.12g}->{r['mae_s1']:.12g}; delta={r['delta_best_mae_s0_minus_s1']:.12g}; ci=[{r['ci95_lower_nominal']:.12g},{r['ci95_upper_nominal']:.12g}]; raw_p={r['raw_p_two_sided_centered']:.6g}; Holm={r['holm_p_two_asset_family']:.6g}; {r['result']}",flush=True)
 print("TEST_PREDICTION_ROWS",len(all_predictions),"TEST_FIT_UNITS",len(test_fit_logs),"FAILURES",sum(x["fit_status"]!="PASS" for x in test_fit_logs),flush=True)

def fit_final(family,cfg,cfgidx,asset,info,seed,xflat_train,xseq_train,ytrain,xflat_eval,xseq_eval):
 import time, warnings
 from sklearn.preprocessing import StandardScaler
 from sklearn.linear_model import Ridge
 from sklearn.ensemble import ExtraTreesRegressor,HistGradientBoostingRegressor
 from sklearn.neural_network import MLPRegressor
 from catboost import CatBoostRegressor
 from xgboost import XGBRegressor
 import torch
 start=time.perf_counter(); caught=[];target_mu=None;target_scale=None;fit_loss=[];n_iter=None;converged=None;param_count=None
 if family=="GRU":
  ntr,tlen,nf=xseq_train.shape;fs=StandardScaler().fit(xseq_train.reshape(-1,nf));tr=fs.transform(xseq_train.reshape(-1,nf)).reshape(ntr,tlen,nf);nev=xseq_eval.shape[0];ev=fs.transform(xseq_eval.reshape(-1,nf)).reshape(nev,tlen,nf)
 else:
  fs=StandardScaler().fit(xflat_train);tr=fs.transform(xflat_train);ev=fs.transform(xflat_eval)
 ts=StandardScaler().fit(np.asarray(ytrain,float).reshape(-1,1)) if family in ["MLP","GRU"] else None
 yfit=ts.transform(np.asarray(ytrain,float).reshape(-1,1)).ravel() if ts is not None else np.asarray(ytrain,float)
 if ts is not None:target_mu=float(ts.mean_[0]);target_scale=float(ts.scale_[0])
 if ts is not None and (not np.isfinite(target_scale) or target_scale<=np.finfo(float).eps):raise FloatingPointError("degenerate final-training target scale")
 with warnings.catch_warnings(record=True) as rec:
  warnings.simplefilter("always")
  if family=="Ridge":model=Ridge(alpha=float(cfg["alpha"])).fit(tr,yfit);ps=model.predict(ev);param_count=int(model.coef_.size+np.size(model.intercept_))
  elif family=="ExtraTrees":model=ExtraTreesRegressor(n_estimators=int(cfg["n_estimators"]),min_samples_leaf=int(cfg["min_samples_leaf"]),max_features=1.0,random_state=seed,n_jobs=1).fit(tr,yfit);ps=model.predict(ev);param_count=int(sum(e.tree_.node_count for e in model.estimators_))
  elif family=="HistGradientBoosting":model=HistGradientBoostingRegressor(max_iter=int(cfg["max_iter"]),learning_rate=float(cfg["learning_rate"]),max_leaf_nodes=int(cfg["max_leaf_nodes"]),l2_regularization=1.0,early_stopping=False,random_state=seed).fit(tr,yfit);ps=model.predict(ev)
  elif family=="CatBoost":model=CatBoostRegressor(**{**cfg,"random_seed":seed,"verbose":False,"allow_writing_files":False}).fit(tr,yfit,verbose=False);ps=model.predict(ev);fit_loss=model.get_evals_result().get("learn",{}).get("RMSE",[])
  elif family=="XGBoost":model=XGBRegressor(**{**cfg,"random_state":seed,"verbosity":0,"n_jobs":1,"device":"cpu"}).fit(tr,yfit,verbose=False);ps=model.predict(ev)
  elif family=="MLP":
   model=MLPRegressor(hidden_layer_sizes=tuple(cfg["hidden_layer_sizes"]),activation="relu",solver="adam",alpha=float(cfg["alpha"]),learning_rate_init=.001,max_iter=int(cfg["max_iter"]),early_stopping=False,shuffle=True,random_state=seed,tol=1e-5,n_iter_no_change=30).fit(tr,yfit);ps=model.predict(ev);fit_loss=model.loss_curve_;n_iter=int(model.n_iter_);converged=(n_iter<int(cfg["max_iter"]));param_count=int(sum(w.size for w in model.coefs_)+sum(b.size for b in model.intercepts_))
  elif family=="GRU":
   from phase6_benchmark_core import TinyGRU
   torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(seed);np.random.seed(seed);model=TinyGRU(nf,int(cfg["hidden_size"]));opt=torch.optim.Adam(model.parameters(),lr=.005,weight_decay=float(cfg["weight_decay"]));xt=torch.as_tensor(tr,dtype=torch.float32);yt=torch.as_tensor(yfit,dtype=torch.float32)
   for _ in range(int(cfg["epochs"])):model.train();opt.zero_grad(set_to_none=True);loss=torch.mean((model(xt)-yt)**2);loss.backward();opt.step();fit_loss.append(float(loss.detach().cpu()))
   model.eval()
   with torch.no_grad():ps=model(torch.as_tensor(ev,dtype=torch.float32)).cpu().numpy()
   n_iter=int(cfg["epochs"]);param_count=int(sum(p.numel() for p in model.parameters()))
  caught=[str(w.message) for w in rec]
 pred=np.asarray(ps,float);pred=pred*target_scale+target_mu if ts is not None else pred
 if not np.isfinite(pred).all():raise FloatingPointError(f"{family} produced non-finite final predictions")
 return pred,{"fit_status":"PASS","error":None,"family":family,"asset":asset,"information_set":info,"seed":int(seed),"config_index":cfgidx,"config":cfg,"train_n":len(ytrain),"eval_n":len(pred),"runtime_seconds":time.perf_counter()-start,"warnings":caught,"n_iter":n_iter,"converged_before_max_iter":converged,"training_loss_curve":fit_loss,"final_training_loss":float(fit_loss[-1]) if len(fit_loss) else None,"target_scaler_mean":target_mu,"target_scaler_scale":target_scale,"target_scaler_training_only":ts is not None,"feature_scaler_training_only":True,"n_parameters_or_tree_nodes":param_count}

if __name__=="__main__":main()
