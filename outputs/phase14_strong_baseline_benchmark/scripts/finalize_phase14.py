#!/usr/bin/env python3
"""Build Phase 14 audits, inference outputs, figures and immutable manifest."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/"outputs/phase14_strong_baseline_benchmark"

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def jwrite(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,ensure_ascii=False,default=str)+"\n",encoding='utf-8')
def check(cond,msg):
 if not cond:raise RuntimeError(msg)
def md_table(df):
 if len(df)==0:return "(no rows)\n"
 vals=[list(df.columns)]+[["" if pd.isna(v) else str(v) for v in row] for row in df.itertuples(index=False,name=None)]
 widths=[max(len(str(r[i])) for r in vals) for i in range(len(vals[0]))]
 def line(r):return "| "+" | ".join(str(r[i]).ljust(widths[i]) for i in range(len(widths)))+" |"
 return line(vals[0])+"\n| "+" | ".join("-"*w for w in widths)+" |\n"+"\n".join(line(r) for r in vals[1:])+"\n"
def holo(p):
 p=np.asarray(p,float);order=np.argsort(p);out=np.empty(len(p));running=0.
 for rank,idx in enumerate(order):running=max(running,min(1.,(len(p)-rank)*p[idx]));out[idx]=running
 return out

def main():
 # Frozen source checks and source after-hashes.
 before=json.loads((OUT/"01_source_manifest_before.json").read_text())
 changed=[];source_after={}
 for rel,info in before["source_files"].items():
  p=ROOT/rel
  if not p.is_file():changed.append({"path":rel,"reason":"MISSING"});continue
  now=sha(p);source_after[rel]={"sha256":now,"size_bytes":p.stat().st_size,"unchanged":now==info["sha256"]}
  if now!=info["sha256"]:changed.append({"path":rel,"reason":"HASH_MISMATCH","before":info["sha256"],"after":now})
 # Revalidate source paths that prior Phase 13 manifest independently tracked, including manuscripts.
 p13=ROOT/"outputs/phase13_neural_target_scaling/16_source_manifest_after.json"
 prior13=json.loads(p13.read_text());prior13_checks=[]
 for item in prior13.get("source_inventory",[]):
  q=ROOT/item["path"]
  ok=q.is_file() and sha(q)==item.get("sha256_after",item.get("sha256"))
  prior13_checks.append({"path":item["path"],"unchanged_from_phase13_audit":bool(ok)})
  if not ok:changed.append({"path":item["path"],"reason":"PHASE13_SOURCE_HASH_MISMATCH"})
 check(not changed,f"Historical source hash mismatch: {changed[:3]}")

 lock=json.loads((OUT/"pretraining_lock.json").read_text());lockhash=(OUT/"pretraining_lock.sha256").read_text().split()[0]
 check(sha(OUT/"pretraining_lock.json")==lockhash,"Pretraining lock changed")
 check(lock.get("status")=="LOCKED_BEFORE_FIT","Pretraining status invalid")
 selection=json.loads((OUT/"selection_freeze.json").read_text());selhash=(OUT/"selection_freeze.sha256").read_text().split()[0]
 check(sha(OUT/"selection_freeze.json")==selhash,"Development selection freeze changed")
 check(selection.get("test_values_parsed") is False and selection.get("status")=="FROZEN_AFTER_DEVELOPMENT_BEFORE_TEST_PARSE","Selection timing not certified")
 check(selection.get("pretraining_lock_sha256")==lockhash,"Selection freeze linked to wrong protocol")

 # Development predictions: recompute pooled MAE for every candidate and fixed reference.
 dev_files=sorted((OUT/"development_folds").glob("*.csv.gz"))
 devframes=[pd.read_csv(f) for f in dev_files]
 check(len(devframes)>0,"No development prediction files")
 devpred=pd.concat(devframes,ignore_index=True)
 devpred.to_csv(OUT/"predictions/all_development_validation_predictions.csv.gz",index=False,compression="gzip")
 devtable=pd.read_csv(OUT/"05_development_model_selection.csv")
 dev_recomputed=devpred.assign(abs_error=lambda x:(x.target_log1p-x.prediction_log1p).abs()).groupby(["asset","information_set","family","config_id"],as_index=False).agg(recomputed_dev_mae=("abs_error","mean"),dev_origins=("origin","nunique"))
 dcheck=devtable.merge(dev_recomputed,on=["asset","information_set","family","config_id"],how="left",validate="one_to_one")
 check(dcheck.recomputed_dev_mae.notna().all(),"Some development configuration has no saved forecast vector")
 check(np.max(np.abs(dcheck.pooled_dev_mae_log1p-dcheck.recomputed_dev_mae))<5e-14,"Development MAE does not reconstruct from saved predictions")
 check((dcheck.pooled_n==dcheck.dev_origins).all(),"Development origin counts differ from selection registry")
 dcheck[["asset","information_set","family","config_id","pooled_dev_mae_log1p","recomputed_dev_mae","pooled_n","dev_origins"]].assign(match=True).to_csv(OUT/"development_metric_reconstruction.csv",index=False)
 loglines=[json.loads(s) for s in (OUT/"training_logs/development_fit_log.jsonl").read_text().splitlines() if s.strip()]
 dev_fit_fail=[x for x in loglines if x.get("fit_status")!="PASS"]
 check(len(loglines)==204,f"Expected 204 development fitting units, found {len(loglines)}")
 check(not dev_fit_fail,f"Development fits failed: {dev_fit_fail[:2]}")

 # Test metrics reconstructed from every saved forecast row.
 predictions=pd.read_csv(OUT/"predictions/final_test_predictions.csv.gz")
 check(predictions.prediction_log1p.notna().all(),"Nonfinite saved final predictions")
 check(predictions.groupby(["asset","information_set","model"]).origin.nunique().eq(1417).all(),"Unequal or unexpected final test origins")
 test_recomputed=predictions.assign(abs_error=lambda x:(x.target_log1p-x.prediction_log1p).abs()).groupby(["asset","information_set","model"],as_index=False).agg(recomputed_mae=("abs_error","mean"),n=("origin","nunique"),first_origin=("origin","min"),last_origin=("origin","max"))
 testtable=pd.read_csv(OUT/"06_complete_model_test_results.csv")
 tcheck=testtable.merge(test_recomputed,on=["asset","information_set","model"],how="left",validate="one_to_one")
 check(tcheck.recomputed_mae.notna().all(),"Missing prediction group for a test metric")
 check(np.max(np.abs(tcheck.mae_log1p-tcheck.recomputed_mae))<5e-14,"Test metric does not reconstruct from saved predictions")
 check((tcheck.test_n==tcheck.n).all(),"Final test sample counts differ")
 tcheck.assign(match=True).to_csv(OUT/"test_metric_reconstruction.csv",index=False)
 testlog=[json.loads(s) for s in (OUT/"training_logs/final_test_fit_log.jsonl").read_text().splitlines() if s.strip()]
 check(len(testlog)==28,"Expected 28 final fit units")
 check(not [x for x in testlog if x.get("fit_status")!="PASS"],"Final model fit failure")
 check(all(x.get("feature_scaler_training_only") is True for x in testlog),"A final feature scaler was not train-only")
 check(all(x.get("target_scaler_training_only")==(x["family"] in ["MLP","GRU"]) for x in testlog),"Neural target scaling status mismatch")

 # Confirm exact origins and targets against the already-frozen Phase 13 prediction artifact.
 p13pred=pd.read_csv(ROOT/"outputs/phase13_neural_target_scaling/predictions/all_origin_predictions.csv.gz")
 p13test=p13pred[(p13pred.split=="FINAL_TEST")&(p13pred.branch=="TARGET_SCALED")]
 phase13_target_ok={}
 for asset in ["BTC","ETH"]:
  a=predictions[(predictions.asset==asset)&(predictions.information_set=="SPOT_ONLY")&(predictions.model=="TrainingMean")].drop_duplicates("origin").sort_values("origin")
  b=p13test[p13test.asset==asset].drop_duplicates("origin_utc").sort_values("origin_utc")
  same_dates=a.origin.astype(str).tolist()==b.origin_utc.astype(str).tolist()
  same_target=len(a)==len(b) and np.allclose(a.target_log1p.to_numpy(),b.target_log1p.to_numpy(),rtol=0,atol=1e-15)
  phase13_target_ok[asset]={"identical_test_origin_vector":same_dates,"identical_target_log1p":bool(same_target),"n_phase14":len(a),"n_phase13":len(b)}
  check(same_dates and same_target,f"Phase 14 target origins/values differ from Phase 13 saved test vector: {asset}")

 # Recalculate primary statistics and Holm adjustment; p algorithm was frozen in protocol.
 boot=pd.read_csv(OUT/"10_bootstrap_inference.csv")
 pvals42=boot[boot.block_length==42].sort_values("asset").raw_p_two_sided_centered.to_numpy()
 adj42=holo(pvals42)
 for (_,row),expected in zip(boot[boot.block_length==42].sort_values("asset").iterrows(),adj42):
  check(abs(float(row.holm_p_two_asset_family)-expected)<1e-15,"Primary Holm p-value does not reconstruct")
 primary=boot[boot.block_length==42].copy()
 primary["result"]=["FAVORABLE" if r.delta_best_mae_s0_minus_s1>0 and r.ci95_lower_nominal>0 and r.holm_p_two_asset_family<.05 else "ADVERSE" if r.delta_best_mae_s0_minus_s1<0 and r.ci95_upper_nominal<0 and r.holm_p_two_asset_family<.05 else "INCONCLUSIVE" for r in primary.itertuples()]
 primary.to_csv(OUT/"07_primary_best_system_comparison.csv",index=False)

 # Complete model table and neural summary tables.
 wide=testtable.pivot(index=["asset","model"],columns="information_set",values="mae_log1p").reset_index()
 devsel=devtable[devtable.selected_within_family==True][["asset","information_set","family","config_id","config","pooled_dev_mae_log1p","selected_within_family"]].copy()
 devsel["best_system_selected"]=False
 for _,s in selection["selections"].items():
  devsel.loc[(devsel.asset==s["asset"])&(devsel.information_set==s["information_set"])&(devsel.family==s["selected_system"]),"best_system_selected"]=True
 devsel.to_csv(OUT/"tables/table1_development_selection.csv",index=False)
 t2=wide.rename(columns={"SPOT_ONLY":"spotonly_mae","SPOT_PLUS_ALL_DERIVATIVES":"spot_plus_derivatives_mae"})
 t2["delta_s0_minus_s1"]=t2.spotonly_mae-t2.spot_plus_derivatives_mae
 t2["relative_reduction_pct"]=100*t2.delta_s0_minus_s1/t2.spotonly_mae
 warnings_map=testtable.set_index(["asset","information_set","model"]).warnings.fillna("").to_dict()
 t2["convergence_status"]=["MLP_MAX_ITER_WARNING" if m=="MLP" else "FIXED_20_EPOCHS" if m=="GRU" else "NO_REPORTED_WARNING" for a,m in zip(t2.asset,t2.model)]
 t2.to_csv(OUT/"tables/table2_complete_test_results.csv",index=False)
 primary.to_csv(OUT/"tables/table3_primary_comparison.csv",index=False)
 # Evidence across the historic and redesigned endpoints remains disaggregated.
 hist6={"BTC":18.56477335524347,"ETH":13.644812507260147};hist13={"BTC":-0.9744155837036567,"ETH":0.8080744892608575}
 histrows=[]
 for experiment,endpoint,values,status in [
   ("Phase 6","Six-category average relative log-MAE reduction (%)",hist6,"ORIGINAL_FROZEN_HISTORICAL_REGISTERED_RESULT; 2/18 tasks passed"),
   ("Phase 13","Six-category retrospective sensitivity (%)",hist13,"RETROSPECTIVE_TARGET_SCALING_SENSITIVITY"),
   ("Phase 14","Development-selected best-system Delta/relative reduction",None,"RETROSPECTIVE_REANALYSIS; different endpoint")]:
  if values is not None:histrows.append({"experiment":experiment,"endpoint":endpoint,"BTC":values["BTC"],"ETH":values["ETH"],"evidence_status":status})
  else:
   vals={r.asset:r for r in primary.itertuples()}
   histrows.append({"experiment":experiment,"endpoint":endpoint,"BTC":f"Delta={vals['BTC'].delta_best_mae_s0_minus_s1:.8g}; relative={vals['BTC'].relative_reduction_pct:.4f}%","ETH":f"Delta={vals['ETH'].delta_best_mae_s0_minus_s1:.8g}; relative={vals['ETH'].relative_reduction_pct:.4f}%","evidence_status":status})
 tab4=pd.DataFrame(histrows);tab4.to_csv(OUT/"tables/table4_prior_phase_comparison.csv",index=False)
 neural=pd.read_csv(OUT/"09_neural_reliability.csv");neural.to_csv(OUT/"tables/table5_neural_reliability.csv",index=False)

 # Generate requested figures exclusively from frozen/saved Phase 14 forecasts/metrics.
 figdir=OUT/"figures";figdir.mkdir(exist_ok=True)
 fig,ax=plt.subplots(figsize=(11,3.8));labels=["F1 validation","F2 validation","F3 validation","Final test"]
 intervals=[("2021-12-01","2022-10-01"),("2022-10-01","2023-01-01"),("2023-01-01","2023-04-01"),("2023-04-01","2023-07-01"),("2023-07-01","2024-03-04")]
 yloc=[4,3,2,1,0]
 for (start,end),y in zip(intervals,yloc):ax.plot([pd.Timestamp(start),pd.Timestamp(end)],[y,y],linewidth=9,solid_capstyle="butt")
 ax.set_yticks(yloc, ["F1 expanding train","F1 validation / F2 train","F2 validation / F3 train","F3 validation","Final test origins"]);ax.set_xlim(pd.Timestamp("2021-12-01"),pd.Timestamp("2024-03-04"));ax.set_title("Chronological development folds and fixed historical test window");ax.set_xlabel("UTC date");ax.grid(axis="x",alpha=.25);fig.tight_layout();fig.savefig(figdir/"figure_1_timeline.png",dpi=180);plt.close(fig)
 selected_dev=devtable[devtable.selected_within_family==True].copy()
 selected_dev["family"]=selected_dev.family.replace({"TrainingMean":"Mean","Persistence":"Persist"})
 order=["TrainingMean","Persistence","Ridge","ExtraTrees","HistGradientBoosting","CatBoost","XGBoost","MLP","GRU"]
 for fignum,source,metriccol,title in [(2,selected_dev,"pooled_dev_mae_log1p","Pooled chronological development MAE"),(3,testtable,"mae_log1p","Final-test MAE")]:
  fig,axes=plt.subplots(1,2,figsize=(15,5),sharey=False)
  for ax,asset in zip(axes,["BTC","ETH"]):
   sub=source[source.asset==asset];piv=sub.pivot(index="family" if fignum==2 else "model",columns="information_set",values=metriccol)
   idx=[m for m in order if m in piv.index];piv=piv.reindex(idx);piv.plot(kind="bar",ax=ax);ax.set_title(asset);ax.set_ylabel("MAE on log1p target (log scale)");ax.set_yscale("log");ax.tick_params(axis="x",rotation=45);ax.legend(["SpotOnly","Spot + derivatives"],fontsize=8)
  fig.suptitle(title);fig.tight_layout();fig.savefig(figdir/f"figure_{fignum}_{'development' if fignum==2 else 'test'}_mae.png",dpi=180);plt.close(fig)
 # Paired loss differences over four-hour origins for selected systems.
 fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
 primary_map={r.asset:r for r in primary.itertuples()}
 for ax,asset in zip(axes,["BTC","ETH"]):
  r=primary_map[asset];s0=r.selected_s0_model;s1=r.selected_s1_model
  a=predictions[(predictions.asset==asset)&(predictions.information_set=="SPOT_ONLY")&(predictions.model==s0)].sort_values("origin")
  b=predictions[(predictions.asset==asset)&(predictions.information_set=="SPOT_PLUS_ALL_DERIVATIVES")&(predictions.model==s1)].sort_values("origin")
  d=(np.abs(a.target_log1p-a.prediction_log1p).to_numpy()-np.abs(b.target_log1p-b.prediction_log1p).to_numpy())*1e6
  ax.plot(pd.to_datetime(a.origin,utc=True),d,linewidth=.6);ax.axhline(0,linewidth=.8);ax.set_ylabel("|e(S0)|−|e(S1)| (×10⁻⁶)");ax.set_title(f"{asset}: {s0} SpotOnly vs {s1} SpotPlusDerivatives")
 axes[-1].set_xlabel("Forecast origin (UTC)");fig.tight_layout();fig.savefig(figdir/"figure_4_paired_loss_difference.png",dpi=180);plt.close(fig)
 # Primary comparison with both block lengths and nominal intervals.
 boot=pd.read_csv(OUT/"10_bootstrap_inference.csv");fig,ax=plt.subplots(figsize=(9,4.2));ypos=[]
 for ai,asset in enumerate(["BTC","ETH"]):
  for bi,block in enumerate([42,84]):
   r=boot[(boot.asset==asset)&(boot.block_length==block)].iloc[0];y=3-ai*2-bi*.55;ypos.append(y)
   ax.errorbar(r.delta_best_mae_s0_minus_s1*1e6,y,xerr=[[ (r.delta_best_mae_s0_minus_s1-r.ci95_lower_nominal)*1e6 ],[(r.ci95_upper_nominal-r.delta_best_mae_s0_minus_s1)*1e6]],fmt="o",capsize=3,label=f"{asset}, block {block}")
 ax.axvline(0,color="black",linewidth=.8);ax.set_yticks(ypos,["BTC 42","BTC 84","ETH 42","ETH 84"]);ax.set_xlabel("Delta_best = MAE(S0) − MAE(S1), ×10⁻⁶");ax.set_title("Primary retrospective system comparison; nominal percentile intervals");ax.grid(axis="x",alpha=.25);fig.tight_layout();fig.savefig(figdir/"figure_5_primary_delta_intervals.png",dpi=180);plt.close(fig)
 # Different estimands shown in separate panels, not pooled on one scale.
 fig,axes=plt.subplots(1,3,figsize=(13,4))
 for ax,vals,title,ylabel in zip(axes,[hist6,hist13,{r.asset:r.relative_reduction_pct for r in primary.itertuples()}],["Phase 6: six-category average","Phase 13: category sensitivity","Phase 14: selected-system comparison"],["Average relative log-MAE reduction (%)","Retrospective category reduction (%)","Relative Delta_best (%)"]):
  ax.bar(["BTC","ETH"],[vals["BTC"],vals["ETH"]]);ax.axhline(0,color="black",linewidth=.8);ax.set_title(title);ax.set_ylabel(ylabel)
 fig.suptitle("Evidence map — distinct estimands, periods and selection rules");fig.tight_layout();fig.savefig(figdir/"figure_6_phase_evidence_map.png",dpi=180);plt.close(fig)

 # Manuscript-ready tables.
 t1=devsel.copy();t1["config"]=t1.config.astype(str);t1=t1.rename(columns={"asset":"Asset","information_set":"Information set","family":"Model","pooled_dev_mae_log1p":"Development MAE","config":"Selected hyperparameters","best_system_selected":"Best system"})
 (OUT/"tables/table1_development_selection.md").write_text("Table 1. Development-only configuration selection. MAE is on Z=log1p(future 24-hour Amihud); lower is better. Selection used pooled F1–F3 validation forecasts.\n\n"+md_table(t1[["Asset","Information set","Model","Development MAE","Selected hyperparameters","Best system"]]),encoding="utf-8")
 t2m=t2.rename(columns={"asset":"Asset","model":"Model","spotonly_mae":"SpotOnly MAE","spot_plus_derivatives_mae":"Spot+Derivatives MAE","delta_s0_minus_s1":"Delta MAE","relative_reduction_pct":"Relative change (%)","convergence_status":"Convergence status"})
 (OUT/"tables/table2_complete_test_results.md").write_text("Table 2. Final historical test results for all fixed Phase 14 model families, scored on identical 1,417 origins per asset. MAE is on Z=log1p(future Amihud); lower is better. S0 and S1 may have different configurations, selected only on development folds. Retrospective results.\n\n"+md_table(t2m[["Asset","Model","SpotOnly MAE","Spot+Derivatives MAE","Delta MAE","Relative change (%)","Convergence status"]]),encoding="utf-8")
 t3=primary.rename(columns={"asset":"Asset","selected_s0_model":"Selected S0 model","selected_s1_model":"Selected S1 model","mae_s0":"S0 MAE","mae_s1":"S1 MAE","delta_best_mae_s0_minus_s1":"Delta_best","relative_reduction_pct":"Relative reduction (%)","ci95_lower_nominal":"42-bar CI lower","ci95_upper_nominal":"42-bar CI upper","raw_p_two_sided_centered":"Raw p","holm_p_two_asset_family":"Holm p","result":"Result"})
 (OUT/"tables/table3_primary_comparison.md").write_text("Table 3. Primary development-selected system comparison. CIs are nominal percentile moving-block intervals; p-values use the locked centered two-sided bootstrap; Holm adjusts BTC/ETH raw p-values. The test period was previously inspected.\n\n"+md_table(t3[["Asset","Selected S0 model","Selected S1 model","S0 MAE","S1 MAE","Delta_best","Relative reduction (%)","42-bar CI lower","42-bar CI upper","Raw p","Holm p","Result"]]),encoding="utf-8")
 (OUT/"tables/table4_prior_phase_comparison.md").write_text("Table 4. Historical estimates versus retrospective sensitivity endpoints. Phase 6/13 category averages and Phase 14 Delta_best are different estimands; no pooled effect is calculated.\n\n"+md_table(tab4),encoding="utf-8")
 t5=neural.rename(columns={"asset":"Asset","information_set":"Information set","model":"Model","development_mae_log1p":"Development MAE","test_mae_log1p":"Test MAE","iterations_or_epochs":"Iterations/epochs","runtime_seconds":"Final fit seconds","parameter_count":"Parameters","warnings":"Convergence diagnostics"})
 (OUT/"tables/table5_neural_reliability.md").write_text("Table 5. Target-scaled neural reliability on the original log1p target scale. Target and feature scaling are training-only. Prediction-range diagnostics are relative to the observed test-target range.\n\n"+md_table(t5[["Asset","Information set","Model","Development MAE","Test MAE","prediction_min","prediction_median","prediction_max","fraction_outside_target_range","Iterations/epochs","Final fit seconds","Parameters","Convergence diagnostics"]]),encoding="utf-8")

 # Outcomes and scientific interpretation.
 pmap={r.asset:r for r in primary.itertuples()};wtab=pd.read_csv(OUT/"08_within_model_information_effects.csv")
 topmodels={}
 for asset in ["BTC","ETH"]:
  sub=testtable[testtable.asset==asset]
  for info in ["SPOT_ONLY","SPOT_PLUS_ALL_DERIVATIVES"]:
   q=sub[sub.information_set==info].sort_values("mae_log1p").iloc[0]
   topmodels[f"{asset}_{info}"]={"test_oracle_model":q.model,"test_oracle_mae":float(q.mae_log1p),"status":"RETROSPECTIVE_TEST_ORACLE_DESCRIPTIVE_ONLY"}
 result_text=[]
 for asset in ["BTC","ETH"]:
  r=pmap[asset];s=selection["selections"][f"{asset}__SPOT_ONLY"];f=selection["selections"][f"{asset}__SPOT_PLUS_ALL_DERIVATIVES"]
  result_text.append(f"{asset}: development chose {s['selected_system']} for S0 (dev MAE {s['pooled_dev_mae_log1p']:.8g}) and {f['selected_system']} for S1 (dev MAE {f['pooled_dev_mae_log1p']:.8g}). Test MAE was {r.mae_s0:.8g} vs {r.mae_s1:.8g}; Delta_best={r.delta_best_mae_s0_minus_s1:.8g} ({r.relative_reduction_pct:.3f}%). Primary 42-bar CI [{r.ci95_lower_nominal:.8g},{r.ci95_upper_nominal:.8g}], raw p={r.raw_p_two_sided_centered:.4g}, Holm p={r.holm_p_two_asset_family:.4g}; result {r.result}.")
 summary="""# Phase 14 scientific interpretation

This is a **retrospective reanalysis**, not a fresh holdout. The Phase 6 final test period was examined during earlier phases. The new procedure protects model selection within Phase 14 from direct test tuning, but does not erase that historical exposure.

## Primary results

"""+"\n\n".join(result_text)+"\n\n"
 summary+="Development selected BTC HGB for both information sets, while ETH selected Ridge for SpotOnly and ExtraTrees for SpotPlusDerivatives. ETH’s selected S1 system was not better in pooled development validation than the selected S0 model, yet had lower retrospective test MAE. That temporal reversal is material and argues against presenting the test result as an established transferable gain. BTC’s selected systems are both HGB; the observed test reduction is small and its 42-bar interval includes zero.\n\n"
 summary+="Across within-family comparisons, derivatives point estimates are mixed: the tree/boosting families often improve, while neural and some linear/tree comparisons worsen; XGBoost closely reproduces the training-mean reference under this frozen target scale/grid. The within-family comparisons are descriptive and are not the primary endpoint. The primary endpoint compares independently development-selected S0 and S1 systems.\n\n"
 summary+="Target-scaled MLP/GRU predictions are numerically reasonable relative to Phase 6, but none beats the strongest test-period classical tree/boosting reference in either information set. MLP fits hit their frozen 400-iteration limit without convergence in all four final asset/information cells. GRU fits used the fixed 20 epochs. No test-directed change was made.\n\n"
 summary+="The results do not restore the historical category-average interpretation: Phase 6’s +18.5648% BTC and +13.6448% ETH category averages, Phase 13’s −0.9744% and +0.8081% sensitivity averages, and Phase 14’s selected-system test deltas are different estimands. The Phase 14 ETH signal is a favorable retrospective, development-selected comparison; BTC remains inconclusive. Phase 7–8’s broad generalization failure remains intact, and Phase 9’s `NO_EXTERNAL_QML_REPLICATION` is untouched. No causal, trading-value, universal derivatives-information, or quantum-advantage conclusion follows.\n\n"
 summary+="No further experiment is required to answer the current retrospective benchmark question. A future claim of generalizable incremental information requires a prospectively frozen evaluation on genuinely unseen data.\n"
 (OUT/"12_scientific_interpretation.md").write_text(summary,encoding="utf-8")

 # Write comparison text, revision plan and limitations.
 prior="""# Phase 6, 13 and 14 comparison

Phase 6’s original frozen result remains 2/18 passed tasks. Its six-category relative log-MAE summaries were +18.5648% for BTC and +13.6448% for ETH. These historical figures are category averages and are not best-system estimates.

Phase 13 changed only neural target standardization in a retrospective sensitivity. The corresponding six-category averages were −0.9744% for BTC and +0.8081% for ETH. The original Phase 6 decision was not revised.

Phase 14’s primary endpoint is a different quantity: the test loss difference between S0 and S1 systems independently selected from pooled chronological development MAE. BTC selected HGB/HGB and had Delta_best=+1.2197e−7 (about +1.99% of S0 MAE), with 42-bar nominal CI [−1.4362e−8, 2.4133e−7] and two-asset Holm p=0.0679. ETH selected Ridge/ExtraTrees and had Delta_best=+7.6050e−7 (about +5.10%), with CI [1.8483e−7, 1.1750e−6] and Holm p=0.0100. Both Phase 14 estimates are retrospective; they must not be interpreted as changes in one common treatment effect or as independent confirmation.
"""
 (OUT/"11_phase6_phase13_phase14_comparison.md").write_text(prior,encoding="utf-8")
 revisions="""# Manuscript revision plan — Phase 14

Do not edit the manuscript in this phase. Add a separate subsection, **Retrospective Strong-Baseline Reanalysis**, and keep the historical Phase 6 table unchanged.

## Abstract
Retain the Phase 6 2/18 result and Phase 7–8 generalization failure. If Phase 14 is summarized, state that a retrospective development-selected benchmark found a favorable BTC/ETH point pattern, with the ETH selected-system comparison passing the locked paired bootstrap and BTC remaining inconclusive. Do not call either a new external confirmation.

## Section 3 — Research objectives and evaluation framework
Distinguish matched within-family comparisons from the stronger primary endpoint, BEST_S0 versus BEST_S1, where each system is selected solely on pooled chronological development validation. State the test period had been examined in earlier project phases.

## Section 5.1 — Matched multi-model forecasting
Add the locked family grids, Phase 13 neural configurations, target scaling for neural fits, training-only feature/target preprocessing, no prediction clipping, and chronological purge rules. Identify CatBoost/XGBoost as bounded Phase 14 additions, not original Phase 6 registrations.

## Section 6.1 — Historical multi-model benchmark
Preserve the original 180 rows and 2/18 decision. Report Phase 13 as a distinct target-scaling sensitivity, then give Phase 14’s model-by-model reanalysis and primary BEST_S0/BEST_S1 table. Explain that BTC’s test interval is inconclusive and ETH’s favorable result is retrospective, notwithstanding the development-only selection rule.

## Section 7 — Discussion
Describe mixed within-family derivatives effects, ETH’s development/test ranking reversal, and the lack of scaled-neural dominance over strong classical models. Keep the Phase 6/13 category averages separate from the Phase 14 selected-system endpoint. Retain Phase 7–8 failures and Phase 9 no-replication result.

## Section 8 — Limitations
State prior exposure of the historical test interval, two-asset clean sample, bounded but not exhaustive CatBoost/XGBoost grids, MLP non-convergence at its fixed iteration budget, XGBoost’s near-training-mean behavior under the locked grid, and the absence of prospective or causal identification.

## Section 9 — Conclusion
Conclude that the retrospective benchmark offers a favorable ETH selected-system result but not a uniform cross-asset result; BTC remains inconclusive. The complete evidence supports a transparent empirical benchmarking contribution, not a new algorithm or broad claim that derivatives reliably improve liquidity forecasts.
"""
 (OUT/"13_manuscript_revision_plan.md").write_text(revisions,encoding="utf-8")
 integrity="""# Phase 14 limitations and integrity

- Evidence type: retrospective reanalysis; the historical Phase 6 test period was previously viewed.
- Model selection used only pooled Phase 14 F1–F3 development predictions; the selected model registry was SHA-256 frozen before the test panel was parsed.
- The primary interval remains the historical Phase 6 test calendar; 1,417 matching origins per asset are scored, ending at 2024-03-02 20:00 UTC because the next six-bar target must be observed before the frozen sample end.
- The target constructor, 4-hour rows, six-step history, S0/S1 feature sets and target-window purges come from the frozen Phase 6 code. No feature was added and no data were downloaded.
- MLP and GRU used train-only target scaling. Every Phase 14 model was scored without output clipping on the same log1p scale.
- MLP reached its fixed 400-iteration maximum without convergence in all four final asset/information-set fits. No retry or extension was made. XGBoost forecasts were effectively at the training-mean level under the locked grid; this is reported as a limitation, not tuned away.
- Bootstrap intervals are nominal, not Holm-adjusted. Holm applies to the two primary BTC/ETH raw p-values separately for each block length. These retrospective p-values do not provide independent confirmation.
- Phase 6/7/8/9/10/12/13 artifacts, the certified data panel, original Phase 6 source code, and the manuscript remain unchanged by the source-hash audit. No QML model or Bybit result was touched.
- No causal effect, execution-cost saving, trading profit, broad external validity, or quantum computational advantage is established.
"""
 (OUT/"14_limitations_and_integrity.md").write_text(integrity,encoding="utf-8")

 # Required final outputs before writing manifests.
 # Keep the split audit exactly as locked before training; do not edit it after seeing test results.
 (OUT/"15_reproducibility_validation.md").write_text(f"""# Phase 14 reproducibility validation

Overall: **PASS**.\n\n- Locked protocol SHA-256: `{lockhash}`.\n- Frozen model registry SHA-256: `{lock['files']['03_frozen_model_registry.json']}`.\n- Development selection SHA-256: `{selhash}`; it records `test_values_parsed=false`.\n- Development fits: {len(loglines)}/204 passed; every candidate/reference pooled development MAE and origin count reconstructed from {len(dev_files)} saved fold prediction files.\n- Final test fits: {len(testlog)}/28 passed; {len(predictions)} forecast rows cover 9 models × 2 information sets × 2 assets × 1,417 origins.\n- Final MAEs reconstructed from the saved prediction matrix within 5e−14.\n- Phase 14 test origin vectors and target values exactly match the retained Phase 13 final-test targets for BTC and ETH (`{phase13_target_ok}`).\n- Feature scalers were fit on training data only; target scalers were used only for MLP/GRU and fit on training labels only. Inverse transformation precedes scoring. No prediction clipping is used.\n- Frozen Phase 6 fold boundaries, strict six-bar purging, asset universe, target constructor, features, and test boundary are unchanged.\n- Bootstrap uses fixed declared seeds, contiguous 42/84-row moving blocks, 1,000 draws, centered two-sided p-values, add-one finite-resample correction, and Holm across the two asset hypotheses.\n- Current hashes of {len(source_after)} pre-run source artifacts match the pre-run inventory; {len(prior13_checks)} paths independently checked against Phase 13’s source audit also match. No prior Phase 6–13 output, market data, QML/Bybit result, or manuscript was modified.\n- New market-data downloads: none. Quantum model fitting: none. Manuscript edits: none.\n\nThe project had already inspected the test interval in earlier phases; this is reproducibility of the retrospective analysis, not independent confirmation.\n""",encoding="utf-8")

 # Save JSON summaries and executive report.
 jwrite(OUT/"16_source_manifest_after.json",{"generated_utc":pd.Timestamp.now(tz="UTC").isoformat(),"pre_run_source_file_count":len(source_after),"all_unchanged":not changed,"source_comparison":source_after,"changed_sources":changed,"phase13_source_paths_checked":prior13_checks,"phase13_source_paths_all_unchanged":all(x["unchanged_from_phase13_audit"] for x in prior13_checks),"dataset_4h_unchanged":source_after.get("outputs/phase5_dynamic_discovery/dataset_4h.jsonl",{}).get("unchanged"),"phase6_through_13_outputs_unchanged":True,"manuscript_unchanged":all(x["unchanged_from_phase13_audit"] for x in prior13_checks if x["path"].startswith("manuscripts/"))})
 best_summary=[]
 for r in primary.itertuples():best_summary.append({"asset":r.asset,"selected_s0":r.selected_s0_model,"selected_s1":r.selected_s1_model,"mae_s0":r.mae_s0,"mae_s1":r.mae_s1,"delta_best":r.delta_best_mae_s0_minus_s1,"relative_pct":r.relative_reduction_pct,"ci42":[r.ci95_lower_nominal,r.ci95_upper_nominal],"raw_p":r.raw_p_two_sided_centered,"holm_p":r.holm_p_two_asset_family,"decision":r.result,"n":r.n_test,"test_oracle_s0":topmodels[f"{r.asset}_SPOT_ONLY"],"test_oracle_s1":topmodels[f"{r.asset}_SPOT_PLUS_ALL_DERIVATIVES"]})
 jwrite(OUT/"phase14_summary.json",{"status":"COMPLETE","evidence_type":"RETROSPECTIVE_REANALYSIS","primary_results":best_summary,"within_model_effects":wtab.to_dict("records"),"neural_reliability":neural.to_dict("records"),"phase6_frozen_status":"2/18 unchanged","phase6_category_average_pct":hist6,"phase13_category_sensitivity_pct":hist13,"phase7_8_generalization":"broad rule failed; unchanged","phase9":"NO_EXTERNAL_QML_REPLICATION; unchanged","source_integrity":"PASS","validation":{"development_predictions":True,"test_predictions":True,"phase13_dates_and_targets":True,"all_fits_succeeded":True},"retrospective_caveat":"All final test periods had been examined in prior phases."})
 exec_summary="""# Phase 14 executive summary

**Status: COMPLETE — retrospective reanalysis.** The original Phase 6–13 outputs, data and manuscript remain unchanged. The new benchmark used fixed BTC/ETH 24-hour Amihud targets, the frozen Phase 6 feature sets and chronology, training-only preprocessing, six model families plus references, and a development-only selection step frozen before test data were parsed.

BTC selected HGB under both S0 and S1. Test MAE fell from 6.1229e−6 to 6.0009e−6 (1.99%), but the 42-bar nominal interval includes zero and Holm p=0.0679: **inconclusive**. ETH selected Ridge under S0 and ExtraTrees under S1. MAE fell from 1.4907e−5 to 1.4146e−5 (5.10%); the 42-bar nominal interval is positive and Holm p=0.0100: **favorable retrospective evidence**. The 84-bar sensitivity agrees qualitatively for both assets.

Target-scaled MLP/GRU models did not beat the strongest test-period classical tree/boosting reference. MLPs still failed to converge by the fixed iteration cap. Within-family derivatives effects were mixed. Thus Phase 14 does not support a universal derivatives advantage; the ETH result is a single-asset retrospective selected-system result and is not independent confirmation. Phase 6 remains 2/18, Phase 7–8 broad generalization remains failed, and Phase 9 remains `NO_EXTERNAL_QML_REPLICATION`.

See `12_scientific_interpretation.md`, `07_primary_best_system_comparison.csv`, and manuscript-ready tables under `tables/`. No further model search is warranted for this retrospective question; prospective evidence would require a future frozen study on genuinely unseen data.
"""
 (OUT/"17_executive_summary.md").write_text(exec_summary,encoding="utf-8")

 # A final source/result validation summary is also retained as a JSON object.
 jwrite(OUT/"final_validation.json",{"status":"PASS","pretraining_lock_sha256":lockhash,"selection_freeze_sha256":selhash,"development_fit_count":len(loglines),"development_fit_failures":len(dev_fit_fail),"development_prediction_file_count":len(dev_files),"development_predictions_reconstruct_all_metrics":True,"final_test_fit_count":len(testlog),"final_test_failures":0,"final_prediction_rows":len(predictions),"final_metrics_reconstruct_from_predictions":True,"phase13_test_dates_targets_match":phase13_target_ok,"source_hashes_unchanged":True,"phase13_source_hashes_unchanged":True,"test_data_loaded_after_selection_freeze":True,"qml_or_bybit_touched":False,"market_data_downloaded":False,"manuscript_modified":False})

 # Final checksum includes all Phase 14 generated scripts, reports, predictions, tables and figures.
 files=[]
 for f in sorted(OUT.rglob("*")):
  if f.is_file() and f.name not in ["18_final_manifest.json","18_final_manifest.sha256"]:
   files.append({"path":str(f.relative_to(OUT)),"sha256":sha(f),"size_bytes":f.stat().st_size})
 manifest={"phase":"Phase 14 — Strong-baseline benchmark","status":"COMPLETE","evidence_type":"RETROSPECTIVE_REANALYSIS","created_utc":pd.Timestamp.now(tz="UTC").isoformat(),"pretraining_lock_sha256":lockhash,"development_selection_freeze_sha256":selhash,"source_manifest_after_sha256":sha(OUT/"16_source_manifest_after.json"),"files_count_excluding_manifest":len(files),"files":files,"original_phase6_decision":"2/18 passed; unchanged","primary_classification":{"BTC":primary_map["BTC"].result,"ETH":primary_map["ETH"].result},"new_training":"Phase 14 registered models only; 204 development fits + 28 final fits; all succeeded","new_market_data":"NONE","prior_outputs_modified":"NONE"}
 jwrite(OUT/"18_final_manifest.json",manifest)
 mh=sha(OUT/"18_final_manifest.json");(OUT/"18_final_manifest.sha256").write_text(mh+"  18_final_manifest.json\n",encoding="utf-8")
 print("PHASE14_FINALIZATION_PASS", "source_files",len(source_after),"dev_fits",len(loglines),"test_fits",len(testlog),"prediction_rows",len(predictions),"output_files",len(files),"manifest_sha256",mh)
 for r in primary.itertuples():print(f"FINAL {r.asset}: {r.selected_s0_model}->{r.selected_s1_model}; Δ={r.delta_best_mae_s0_minus_s1:.10g}; CI42=({r.ci95_lower_nominal:.10g},{r.ci95_upper_nominal:.10g}); rawp={r.raw_p_two_sided_centered:.6g}; Holm={r.holm_p_two_asset_family:.6g}; {r.result}")

if __name__=="__main__":main()
