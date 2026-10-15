from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from phase7_run_development import (
    ASSETS, TARGETS, INFOSETS, SPOT, DERIV, CALENDAR, DERIV_STATIC,
    DEV_START, DEV_LAST, DEV_END_EXCL, SEED, clean, dump, sha,
    transform_y, inverse_y, fit_model, make_features,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "phase7_liquidity_replication"
PANEL = ROOT / "data" / "processed" / "phase7" / "market_liquidity_panel_4h.csv.gz"
EXT_START = pd.Timestamp("2024-03-04T00:00:00Z")
SAMPLE_END_EXCL = pd.Timestamp("2026-10-01T00:00:00Z")
EXT_LAST_FEATURE = pd.Timestamp("2026-09-29T20:00:00Z")
N_BOOT = 2000
BLOCK = 42


def load_dev_result():
    f = json.loads((OUT / "phase7_freeze_manifest.json").read_text())
    p = json.loads((OUT / "phase7_protocol.json").read_text())
    dev_freeze_path = OUT / "phase7_development_freeze.json"
    dev_freeze = json.loads(dev_freeze_path.read_text())
    audit_path = OUT / "phase7_data_audit.json"
    audit = json.loads(audit_path.read_text())
    if not audit.get("data_audit_pass", False):
        raise RuntimeError("Phase-7 data audit did not pass; external evaluation blocked")
    if sha(audit_path) != f.get("data_audit_sha256"):
        raise RuntimeError("Frozen Phase-7 data audit hash mismatch")
    if not f.get("protocol_frozen") or sha(OUT / "phase7_protocol.json") != f["phase7_protocol_sha256"]:
        raise RuntimeError("Frozen protocol hash mismatch")
    amendments = {
        "phase7_protocol_amendment_001.md": f.get("phase7_protocol_amendment_001_sha256"),
        "phase7_protocol_amendment_002.md": f.get("phase7_protocol_amendment_002_sha256"),
    }
    for filename, expected in amendments.items():
        if not expected or sha(OUT / filename) != expected:
            raise RuntimeError(f"Frozen protocol amendment hash mismatch: {filename}")
    if dev_freeze.get("protocol_sha256") != f["phase7_protocol_sha256"]:
        raise RuntimeError("Development freeze uses a different protocol")
    for relpath, expected in dev_freeze.get("files", {}).items():
        if sha(ROOT / relpath) != expected:
            raise RuntimeError(f"Frozen development artifact hash mismatch: {relpath}")
    if sha(PANEL) != f["frozen_inputs_and_rules"]["data/processed/phase7/market_liquidity_panel_4h.csv.gz"]:
        raise RuntimeError("Frozen input panel hash mismatch")
    dev = json.loads((OUT / "phase7_development_cv.json").read_text())
    sel = json.loads((OUT / "phase7_selected_models.json").read_text())
    if dev["protocol_sha256"] != f["phase7_protocol_sha256"] or sel["protocol_sha256"] != f["phase7_protocol_sha256"]:
        raise RuntimeError("Development results were produced under a different protocol")
    return f, p, dev, sel


def target_group(dev, asset, target):
    return next(x for x in dev["transforms"] if x["asset"] == asset and x["target"] == target)


def build_oof_alarm_thresholds(dev, sel):
    oof = pd.read_csv(OUT / "phase7_development_oof_predictions.csv.gz", parse_dates=["date"])
    thresholds = {}
    for sm in sel["selected_models"]:
        a,t,info,fam = sm["asset"],sm["target"],sm["infoset"],sm["family"]
        q=oof[(oof.asset==a)&(oof.target==t)&(oof.infoset==info)&(oof.family==fam)].copy()
        if len(q)==0: raise RuntimeError(f"Missing selected OOF predictions {a}/{t}/{info}/{fam}")
        if "prediction_raw" not in q: raise RuntimeError("Development OOF predictions lack fold-safe inverse-transformed values")
        thresholds[(a,t,info,fam)]=float(q.prediction_raw.quantile(0.9))
    return thresholds


def holm(pvals):
    p=np.asarray(pvals,float);n=len(p);o=np.argsort(p);adj=np.empty(n);running=0.0
    for rank,idx in enumerate(o):
        running=max(running,(n-rank)*p[idx]);adj[idx]=min(1.0,running)
    return adj.tolist()


def block_bootstrap_delta(dates, delta, seed):
    """Paired circular moving-block bootstrap on the full regular date grid."""
    x=np.asarray(delta,float);n=len(x)
    if n < BLOCK*3: raise RuntimeError("External period too short for frozen block bootstrap")
    valid=np.isfinite(x);v=np.where(valid,x,0.0);cnt=valid.astype(float)
    # Block sums are defined on the circular 4-hour grid.
    pad_v=np.r_[v,v[:BLOCK]];pad_c=np.r_[cnt,cnt[:BLOCK]]
    cum_v=np.r_[0.0,np.cumsum(pad_v)];cum_c=np.r_[0.0,np.cumsum(pad_c)]
    starts=np.arange(n)
    sum42=cum_v[starts+BLOCK]-cum_v[starts];cnt42=cum_c[starts+BLOCK]-cum_c[starts]
    full=n//BLOCK;rem=n-full*BLOCK
    rng=np.random.default_rng(seed)
    draw=rng.integers(0,n,size=(N_BOOT,full))
    sums=sum42[draw].sum(axis=1);counts=cnt42[draw].sum(axis=1)
    if rem:
        sv=cum_v[starts+rem]-cum_v[starts];sc=cum_c[starts+rem]-cum_c[starts]
        ds=rng.integers(0,n,size=N_BOOT);sums+=sv[ds];counts+=sc[ds]
    boot=np.divide(sums,counts,out=np.full(N_BOOT,np.nan),where=counts>0)
    boot=boot[np.isfinite(boot)]
    observed=float(np.nanmean(x))
    ci=np.quantile(boot,[.025,.975])
    p=2*min(float(np.mean(boot<=0)),float(np.mean(boot>=0)))
    return {"mean_delta":observed,"ci95":ci.tolist(),"bootstrap_probability_delta_below_zero":float(np.mean(boot<0)),"two_sided_sign_probability":min(1.0,p),"replications":int(len(boot)),"block_length_4h":BLOCK}


def calc_metrics(y, pred, target_transform, scale, target_q90=None, alarm_q90=None):
    y=np.asarray(y,float);pred=np.asarray(pred,float);keep=np.isfinite(y)&np.isfinite(pred);y=y[keep];pred=pred[keep]
    yt,_=transform_y(y,target_transform,scale);pt,_=transform_y(pred,target_transform,scale)
    pred0=np.maximum(pred,0)
    out={"n":int(len(y)),"mae_transformed":float(np.mean(np.abs(pt-yt))),"rmse_transformed":float(np.sqrt(np.mean((pt-yt)**2))),"mae_original":float(np.mean(np.abs(pred0-y))),"rmse_original":float(np.sqrt(np.mean((pred0-y)**2))),"r2_original":float(1-np.sum((pred0-y)**2)/np.sum((y-y.mean())**2)) if np.sum((y-y.mean())**2)>0 else None}
    if target_q90 is not None and alarm_q90 is not None:
        event=y>=target_q90;alarm=pred0>=alarm_q90;tp=int(np.sum(event&alarm));fp=int(np.sum(~event&alarm));fn=int(np.sum(event&~alarm))
        precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.;f1=2*precision*recall/(precision+recall) if precision+recall else 0.;prev=float(np.mean(event))
        out["stress_warning"]={"actual_threshold_dev_q90":float(target_q90),"alarm_threshold_dev_oof_q90":float(alarm_q90),"event_prevalence":prev,"precision":precision,"recall":recall,"f1":f1,"lift":precision/prev if prev>0 else None,"true_positive":tp,"false_positive":fp,"false_negative":fn}
    return out


def full_panel():
    d=pd.read_csv(PANEL,low_memory=False)
    d["period_start_utc"]=pd.to_datetime(d.period_start_utc,utc=True)
    d["future_window_end_utc_exclusive"]=pd.to_datetime(d.future_window_end_utc_exclusive,utc=True)
    return d


def main():
    freeze,p,dev,selected=load_dev_result()
    result_out = Path(os.environ.get("PHASE7_RESULT_DIR", str(OUT))).resolve()
    result_out.mkdir(parents=True, exist_ok=True)
    if result_out == OUT.resolve() and (result_out / "phase7_external_evaluation.json").exists():
        raise RuntimeError("Primary external evaluation already exists; refusing a second pass")
    start=time.time()
    panel=full_panel()
    alarm_thresholds=build_oof_alarm_thresholds(dev,selected)
    selected_map={(x["asset"],x["target"],x["infoset"],x["family"]):x for x in selected["selected_models"]}
    transform_map={(x["asset"],x["target"]):x["transform"] for x in selected["transform_by_asset_target"]}
    family_names=list(p["families"])
    category={k:v["category"] for k,v in p["families"].items()}
    info_pairs=[("legacy_spot_only","legacy_spot_plus_derivatives","legacy"),("enhanced_spot_only","enhanced_spot_plus_derivatives","enhanced")]
    records=[];pair_boot=[];stability=[];stress=[];baselines=[]
    test_grids={}
    for asset in ASSETS:
      caches={info:make_features(panel,asset,info) for info in INFOSETS}
      for target in TARGETS:
       yfull=caches["legacy_spot_only"]["y"][target]
       end=caches["legacy_spot_only"]["target_end"];date=caches["legacy_spot_only"]["date"]
       dev_base=(date>=DEV_START)&(date<=DEV_LAST)&(end<=DEV_END_EXCL)&np.isfinite(yfull)&(yfull>=0)
       transform=transform_map[(asset,target)]
       ytrain=yfull[dev_base]
       _,scale=transform_y(ytrain,transform)
       target_q90=float(np.quantile(ytrain,0.9))
       for spot_info,plus_info,pair_name in info_pairs:
        cs,cp=caches[spot_info],caches[plus_info]
        complete=cs["info_valid"]&cp["info_valid"]&np.isfinite(yfull)&(yfull>=0)
        tr=complete&dev_base
        te=complete&(date>=EXT_START)&(date<=EXT_LAST_FEATURE)&(end<=SAMPLE_END_EXCL)
        if tr.sum()<1000 or te.sum()<1000: raise RuntimeError(f"Insufficient matched rows {asset}/{target}/{pair_name}: train={tr.sum()} test={te.sum()}")
        ytr,scale=transform_y(yfull[tr],transform)
        test_dates=pd.DatetimeIndex(date[te]);grid=pd.date_range(EXT_START,EXT_LAST_FEATURE,freq="4h")
        if len(test_dates.unique())!=len(test_dates): raise RuntimeError("duplicate external dates")
        test_grids[(asset,target,pair_name)]=grid
        spot_oof_thresholds={fam:alarm_thresholds[(asset,target,spot_info,fam)] for fam in family_names}
        plus_oof_thresholds={fam:alarm_thresholds[(asset,target,plus_info,fam)] for fam in family_names}
        pred_by_family={}
        for fam in family_names:
          out_pair={}
          for info,c in [(spot_info,cs),(plus_info,cp)]:
            sm=selected_map[(asset,target,info,fam)];cfg=sm["selected_config"]
            seed=SEED+ASSETS.index(asset)*10000+TARGETS.index(target)*1000+family_names.index(fam)*10+int(sm["selected_config_index"])
            predt=fit_model(fam,cfg,c["seq"][tr],c["static"][tr],ytr,c["seq"][te],c["static"][te],seed)
            predt=np.maximum(predt,0.0);predraw=inverse_y(predt,transform,scale)
            yraw=yfull[te]
            # Compare same observations by construction; assert key alignment for both information sets.
            if len(predraw)!=len(test_dates): raise RuntimeError("prediction/date count mismatch")
            metrics=calc_metrics(yraw,predraw,transform,scale,target_q90,alarm_thresholds[(asset,target,info,fam)])
            out_pair[info]={"prediction_transformed":predt,"prediction_raw":predraw,"metrics":metrics,"selected_config":cfg}
            for i,dt in enumerate(test_dates):
              records.append({"asset":asset,"target":target,"feature_pair":pair_name,"information_set":"spot_only" if info==spot_info else "spot_plus_derivatives","family":fam,"family_category":category[fam],"date":dt.isoformat(),"actual_raw":float(yraw[i]),"actual_transformed":float(transform_y([yraw[i]],transform,scale)[0][0]),"prediction_transformed":float(predt[i]),"prediction_raw":float(predraw[i]),"target_transform":transform,"train_transform_scale":float(scale),"alarm_threshold_raw_dev_oof_q90":float(alarm_thresholds[(asset,target,info,fam)])})
          pred_by_family[fam]=out_pair
        # Non-feature baselines are reported separately; they are not part of derivative increment families.
        tr_target,_=transform_y(yfull[tr],transform);mean_pred=float(np.mean(tr_target));meanraw=float(inverse_y([mean_pred],transform,scale)[0])
        for dt,yraw in zip(test_dates,yfull[te]):
          records.append({"asset":asset,"target":target,"feature_pair":pair_name,"information_set":"baseline","family":"training_mean","family_category":"baseline","date":dt.isoformat(),"actual_raw":float(yraw),"actual_transformed":float(transform_y([yraw],transform,scale)[0][0]),"prediction_transformed":mean_pred,"prediction_raw":meanraw,"target_transform":transform,"train_transform_scale":float(scale),"alarm_threshold_raw_dev_oof_q90":None})
        if target=="future_amihud_24h_mean":
          basevar="spot_amihud"
        elif target=="future_corwin_schultz_spread_24h":
          basevar="corwin_schultz_4h"
        else: basevar=None
        if basevar:
          g=panel[panel.asset==asset].set_index("period_start_utc").reindex(test_dates)
          pval=np.maximum(pd.to_numeric(g[basevar],errors="coerce").to_numpy(float),0.0)
          validp=np.isfinite(pval)&np.isfinite(yfull[te])
          if validp.any():
            bm=calc_metrics(yfull[te][validp],pval[validp],transform,scale)
            baselines.append({"asset":asset,"target":target,"feature_pair":pair_name,"baseline":"persistence_current_proxy","source_variable":basevar,**bm})
        # Paired model-family incremental loss, bootstrap and subperiod stability.
        target_actual=pd.Series(yfull[te],index=test_dates)
        for fam in family_names:
          s=pred_by_family[fam][spot_info]["prediction_transformed"]
          q=pred_by_family[fam][plus_info]["prediction_transformed"]
          yt=transform_y(yfull[te],transform,scale)[0]
          loss_diff=np.abs(q-yt)-np.abs(s-yt)
          diff_full=pd.Series(np.nan,index=grid)
          diff_full.loc[test_dates]=loss_diff
          bt=block_bootstrap_delta(grid,diff_full.to_numpy(float),SEED+ASSETS.index(asset)*100000+TARGETS.index(target)*10000+family_names.index(fam)*100+INFOSETS.index(spot_info))
          pair_boot.append({"asset":asset,"target":target,"feature_pair":pair_name,"family":fam,"family_category":category[fam],"n_dates":int(np.isfinite(loss_diff).sum()),"spot_mae_transformed":float(np.mean(np.abs(s-yt))),"plus_mae_transformed":float(np.mean(np.abs(q-yt))),"delta_plus_minus_spot":bt["mean_delta"],"bootstrap_ci95":bt["ci95"],"bootstrap_probability_improvement":bt["bootstrap_probability_delta_below_zero"],"bootstrap_p_two_sided_sign":bt["two_sided_sign_probability"],"replications":bt["replications"],"block_length_4h":BLOCK})
          for name,startdate,enddate in [("2024-03_to_2024-12",EXT_START,pd.Timestamp("2025-01-01T00:00:00Z")),("2025",pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),("2026_partial",pd.Timestamp("2026-01-01T00:00:00Z"),SAMPLE_END_EXCL)]:
            mask=(test_dates>=startdate)&(test_dates<enddate)
            if mask.any():
              stability.append({"asset":asset,"target":target,"feature_pair":pair_name,"family":fam,"family_category":category[fam],"segment":name,"n_dates":int(mask.sum()),"delta_plus_minus_spot_transformed_mae":float(np.mean(np.abs(q[mask]-yt[mask])-np.abs(s[mask]-yt[mask])))})
          for info,vals,threshold in [(spot_info,s,spot_oof_thresholds[fam]),(plus_info,q,plus_oof_thresholds[fam])]:
            met=calc_metrics(yfull[te],inverse_y(vals,transform,scale),transform,scale,target_q90,threshold)
            if "stress_warning" in met:
              stress.append({"asset":asset,"target":target,"feature_pair":pair_name,"family":fam,"family_category":category[fam],"information_set":"spot_only" if info==spot_info else "spot_plus_derivatives",**met["stress_warning"]})
        print(f"external scored {asset}/{target}/{pair_name}",flush=True)
    scores=pd.DataFrame(records)
    predictions_path=result_out/"phase7_external_predictions.csv.gz"
    scores.to_csv(predictions_path,index=False,compression={"method":"gzip","mtime":0})
    boot_df=pd.DataFrame(pair_boot)
    # Holm within asset-target-feature-pair, across the six learned algorithms.
    boot_df["holm_p_within_asset_target_feature_pair"]=np.nan
    for _,idx in boot_df.groupby(["asset","target","feature_pair"]).groups.items():
        boot_df.loc[idx,"holm_p_within_asset_target_feature_pair"]=holm(boot_df.loc[idx,"bootstrap_p_two_sided_sign"].to_numpy(float))
    category_summary=[]
    for keys,g in boot_df.groupby(["asset","target","feature_pair"]):
        for cat,gc in g.groupby("family_category"):
            category_summary.append({"asset":keys[0],"target":keys[1],"feature_pair":keys[2],"category":cat,"model_count":int(len(gc)),"mean_delta":float(gc.delta_plus_minus_spot.mean()),"improving_models":int((gc.delta_plus_minus_spot<0).sum())})
    aggregate=[]
    catdf=pd.DataFrame(category_summary)
    for keys,g in catdf.groupby(["asset","target","feature_pair"]):
        # Equal weight across independent categories; neural MLP/GRU is averaged within its category.
        delta_by_date=[]
        # Average paired loss differences across models within category and then across categories by date.
        a= scores[(scores.asset==keys[0])&(scores.target==keys[1])&(scores.feature_pair==keys[2])&(scores.information_set=="spot_only")]
        b= scores[(scores.asset==keys[0])&(scores.target==keys[1])&(scores.feature_pair==keys[2])&(scores.information_set=="spot_plus_derivatives")]
        # Recover one delta series per model, then form the equal-category average.
        wide=[]
        for fam in family_names:
            sa=a[a.family==fam].set_index("date");sb=b[b.family==fam].set_index("date")
            if "SUMMARY_ONLY" in sa.index: continue
            yv=sa.actual_transformed.to_numpy(float);ps=sa.prediction_transformed.to_numpy(float);pp=sb.prediction_transformed.to_numpy(float)
            wide.append(pd.Series(np.abs(pp-yv)-np.abs(ps-yv),index=pd.to_datetime(sa.index,utc=True),name=fam))
        if not wide: continue
        mat=pd.concat(wide,axis=1);catseries=[]
        for cat in sorted(set(category.values())):
            cols=[fam for fam in family_names if category[fam]==cat]
            cols=[c for c in cols if c in mat.columns]
            if cols: catseries.append(mat[cols].mean(axis=1).rename(cat))
        bycat=pd.concat(catseries,axis=1);avg=bycat.mean(axis=1,skipna=False)
        grid=test_grids[keys];deltagrid=avg.reindex(grid).to_numpy(float)
        bt=block_bootstrap_delta(grid,deltagrid,SEED+500000+ASSETS.index(keys[0])*1000+TARGETS.index(keys[1])*100+(["legacy","enhanced"].index(keys[2])*10))
        cats={c:float(bycat[c].mean()) for c in bycat.columns}
        aggregate.append({"asset":keys[0],"target":keys[1],"feature_pair":keys[2],"equal_category_mean_delta":bt["mean_delta"],"bootstrap_ci95":bt["ci95"],"bootstrap_probability_improvement":bt["bootstrap_probability_delta_below_zero"],"category_mean_deltas":cats,"categories_improving":int(sum(v<0 for v in cats.values()))})
    # Stress measures and fixed external temporal partitions are kept in separate JSON objects.
    # Negative control: use future derivative sequence t+7..t+12 (after target t+1..t+6), never as a production forecast.
    placebo=[]
    for asset in ASSETS:
      for target in TARGETS:
        cspot=make_features(panel,asset,"legacy_spot_only")
        cplus=make_features(panel,asset,"legacy_spot_plus_derivatives")
        date=cspot["date"];end=cspot["target_end"];y=cspot["y"][target]
        g=panel[panel.asset==asset].sort_values("period_start_utc").reset_index(drop=True)
        # This negative-control comparison is reported descriptively on outcome-window rows.
        rows=(date>=EXT_START)&(date<=EXT_LAST_FEATURE)&(end<=SAMPLE_END_EXCL)&np.isfinite(y)
        ids=np.where(rows)[0]
        # t+7..t+12 derivative sequence: source rows i+7 through i+12.
        good=ids[(ids+12)<len(g)]
        if len(good)<1000: raise RuntimeError("Future-derivative placebo has insufficient support")
        # Fixed 42-step circular displacement of derivatives is a second, non-primary placebos.
        for label,offset in [("after_target_t_plus_7_to_t_plus_12",12),("fixed_42_interval_displacement",42)]:
            # The model specification is deliberately simple/fixed: Ridge alpha=1; it is never tuned on placebo results.
            ydev=(date>=DEV_START)&(date<=DEV_LAST)&(end<=DEV_END_EXCL)&np.isfinite(y)
            train_ids=np.where(ydev)[0]
            dev_valid=train_ids[(train_ids+offset)<len(g)]
            dev_valid=dev_valid[date[dev_valid+offset]<=DEV_LAST]
            test_valid=good if label=="after_target_t_plus_7_to_t_plus_12" else ids[(ids+offset)<len(g)]
            if len(dev_valid)<1000 or len(test_valid)<1000: continue
            ytr_raw=y[dev_valid];transform=transform_map[(asset,target)];ytr,scale=transform_y(ytr_raw,transform)
            # Current spot sequence only and the shifted future derivative values; all observations after t+h remain a placebo.
            spotseq=cspot["seq"][dev_valid]
            plusseq=cplus["seq"][dev_valid+offset,:,len(SPOT):]
            xtr=np.concatenate([spotseq.reshape(len(spotseq),-1),plusseq.reshape(len(plusseq),-1)],axis=1)
            spotseqte=cspot["seq"][test_valid]
            plusseqte=cplus["seq"][test_valid+offset,:,len(SPOT):]
            xte=np.concatenate([spotseqte.reshape(len(spotseqte),-1),plusseqte.reshape(len(plusseqte),-1)],axis=1)
            train_valid=np.isfinite(xtr).all(axis=1)&np.isfinite(ytr)
            test_valid_mask=np.isfinite(xte).all(axis=1)&np.isfinite(y[test_valid])
            xtr=xtr[train_valid];ytr=ytr[train_valid]
            test_valid=test_valid[test_valid_mask];xte=xte[test_valid_mask]
            if len(xtr)<1000 or len(xte)<1000: continue
            scaler=StandardScaler().fit(xtr);model=__import__("sklearn.linear_model",fromlist=["Ridge"]).Ridge(alpha=1.0).fit(scaler.transform(xtr),ytr)
            pred=model.predict(scaler.transform(xte));yt=transform_y(y[test_valid],transform,scale)[0]
            placebo.append({"asset":asset,"target":target,"placebo":label,"derivative_offset_intervals":offset,"n_external_dates":int(len(test_valid)),"fixed_model":"Ridge(alpha=1.0); training-only scaling; development fit","transformed_mae_with_future_derivatives":float(np.mean(np.abs(np.maximum(pred,0)-yt))),"note":"Impossible-time information is used only as a falsification diagnostic; predictive association is not causal evidence."})
    external={
      "protocol_sha256":freeze["phase7_protocol_sha256"],"protocol_amendment_001_sha256":freeze.get("phase7_protocol_amendment_001_sha256"),"protocol_amendment_002_sha256":freeze.get("phase7_protocol_amendment_002_sha256"),"dataset_sha256":sha(PANEL),"external_sample":{"start":EXT_START,"last_feature_interval_start":EXT_LAST_FEATURE,"sample_end_exclusive":SAMPLE_END_EXCL},
      "one_pass_external_evaluation":True,"raw_market_data_downloaded":False,"external_outcome_read_after_protocol_freeze":True,"training_rows_end":DEV_LAST,"preexisting_stage6_test_used_for_training":False,
      "selected_transforms":selected["transform_by_asset_target"],"selected_models":selected["selected_models"],"paired_comparison_results":boot_df.to_dict(orient="records"),"equal_category_aggregate":aggregate,"category_summary":category_summary,"temporal_stability":stability,"stress_warning":stress,"simple_baselines":baselines,
      "prediction_path":str(predictions_path.relative_to(ROOT)),"prediction_sha256":sha(predictions_path),"bootstrap":{"replications":N_BOOT,"block_length_4h":BLOCK,"seed":SEED},"negative_controls":placebo
    }
    dump(result_out/"phase7_external_evaluation.json",external)
    dump(result_out/"phase7_negative_controls.json",{"protocol_sha256":freeze["phase7_protocol_sha256"],"tests":placebo})
    dump(result_out/"phase7_temporal_stability.json",{"protocol_sha256":freeze["phase7_protocol_sha256"],"segments":stability})
    print(json.dumps({"external_evaluation_complete":True,"prediction_rows":len(scores),"evaluation_seconds":round(time.time()-start,1),"pairwise_comparisons":len(boot_df),"aggregate_results":aggregate},indent=2),flush=True)


if __name__=="__main__": main()
