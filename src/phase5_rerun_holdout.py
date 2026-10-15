from __future__ import annotations
import json, hashlib, datetime
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from statsmodels.stats.multitest import multipletests

from phase5_rerun_common import OUT, OLD, PANEL, VARS, ASSETS, sha, write_json, log, load_asset_discovery

STEP_NS=int(pd.Timedelta(hours=4).value)
DISC_START=pd.Timestamp("2021-12-01T00:00:00Z")
DISC_END=pd.Timestamp("2023-06-30T20:00:00Z")
HOLD_START=pd.Timestamp("2023-07-01T00:00:00Z")
HOLD_END=pd.Timestamp("2024-03-03T20:00:00Z")
HAC_LAGS=42
B_REPS=499
BOOT_SEED=20261007

def verify_freeze():
    fr=json.loads((OUT/"holdout_freeze_manifest.json").read_text())
    cand=OUT/fr["candidate_file"]; protocol=OUT/fr["protocol_file"]
    if sha(cand)!=fr["candidate_sha256"] or sha(protocol)!=fr["protocol_sha256"]:
        raise RuntimeError("Holdout candidate/protocol hash mismatch")
    c=json.loads(cand.read_text())
    if c.get("confirmation_values_read") is not False or fr.get("confirmation_read_before_freeze") is not False:
        raise RuntimeError("Confirmation was not frozen before holdout read")
    return c,fr

def load_full_standardized():
    # The caller invokes this only after verify_freeze has passed.
    discovery=load_asset_discovery()
    rows={a:[] for a in ASSETS}
    with PANEL.open(encoding="utf-8") as f:
        for line in f:
            z=json.loads(line)
            if z.get("asset") not in ASSETS: continue
            ts=pd.Timestamp(z["period_start_utc"])
            if ts < DISC_START or ts > HOLD_END: continue
            rows[z["asset"]].append(z)
    out={}
    for a in ASSETS:
        g=pd.DataFrame(rows[a]).sort_values("period_start_utc").reset_index(drop=True)
        g["ts"]=pd.to_datetime(g["period_start_utc"],utc=True)
        mu=discovery[a]["mean"]; sd=discovery[a]["sd"]
        raw=g[VARS].to_numpy(float)
        z=(raw-mu)/sd
        out[a]={"times":g.ts,"time_ns":g.ts.astype("int64").to_numpy(),
                "values":z,"split":g.sample_split.astype(str).to_numpy(),
                "map":{int(t):i for i,t in enumerate(g.ts.astype("int64").to_numpy())},
                "row_count":len(g),"complete_rows":int(np.isfinite(z).all(axis=1).sum()),
                "first":g.ts.iloc[0].isoformat(),"last":g.ts.iloc[-1].isoformat()}
    return out

def candidate_arrays(series, candidate, direction="forward", split="DISCOVERY"):
    src=VARS.index(candidate["source"]); dst=VARS.index(candidate["target"])
    lag=int(candidate["lag"])
    ts=series["time_ns"]; vals=series["values"]; splits=series["split"]; ix=series["map"]
    Xs=[]; Ds=[]; Ys=[]; times=[]
    for target_i, target_ns in enumerate(ts):
        if splits[target_i] != split: continue
        target_ns=int(target_ns)
        if direction=="forward":
            d_ns=target_ns-lag*STEP_NS
            control_end=d_ns
            y_ns=target_ns
        elif direction=="reverse_time":
            # Impossible/backward analogue: future source state predicts an
            # already-realized target state. Controls end before that target.
            d_ns=target_ns+lag*STEP_NS
            control_end=target_ns
            y_ns=target_ns
        else: raise ValueError(direction)
        di=ix.get(d_ns); yi=ix.get(y_ns)
        if di is None or yi is None: continue
        if split=="DISCOVERY" and series["split"][di] != "DISCOVERY": continue
        if split=="DISCOVERY" and series["split"][target_i] != "DISCOVERY": continue
        ctrl=[]; valid=True
        for j in range(1,7):
            ci=ix.get(control_end-j*STEP_NS)
            if ci is None or not np.isfinite(vals[ci]).all(): valid=False;break
            ctrl.extend(vals[ci].tolist())
        if not valid or not np.isfinite(vals[di]).all() or not np.isfinite(vals[yi]).all(): continue
        Xs.append(ctrl); Ds.append(vals[di,src]); Ys.append(vals[yi,dst]); times.append(y_ns)
    if not Xs: return {"X":np.empty((0,6*len(VARS))),"D":np.array([]),"Y":np.array([]),"time_ns":np.array([],dtype=np.int64)}
    order=np.argsort(times)
    return {"X":np.asarray(Xs,float)[order],"D":np.asarray(Ds,float)[order],"Y":np.asarray(Ys,float)[order],"time_ns":np.asarray(times,dtype=np.int64)[order]}

def nuisance_models():
    hgb=HistGradientBoostingRegressor(max_iter=100,learning_rate=.05,max_leaf_nodes=15,
          min_samples_leaf=30,l2_regularization=1.0,random_state=314159)
    ridge=make_pipeline(StandardScaler(),Ridge(alpha=1.0))
    return {"HGB":hgb,"Ridge":ridge}

def fit_orthogonalized(train,test,learner):
    if len(train["D"])<100 or len(test["D"])<50: raise ValueError("insufficient train or holdout rows")
    m=nuisance_models()[learner]
    m.fit(train["X"],train["D"]); dtilde=test["D"]-m.predict(test["X"])
    g=nuisance_models()[learner]
    g.fit(train["X"],train["Y"]); ytilde=test["Y"]-g.predict(test["X"])
    reg=np.column_stack([np.ones(len(dtilde)),dtilde])
    coef=np.linalg.lstsq(reg,ytilde,rcond=None)[0]
    resid=ytilde-reg@coef
    return {"theta":float(coef[1]),"dtilde":dtilde,"ytilde":ytilde,"residual":resid,
            "mse_treatment_train":float(np.mean((train["D"]-m.predict(train["X"]))**2)),
            "n_train":len(train["D"]),"n_test":len(test["D"]),"time_ns":test["time_ns"]}

def hac_slope(result,maxlags=HAC_LAGS):
    d=result["dtilde"];y=result["ytilde"];r=result["residual"];times=result["time_ns"]
    x=np.column_stack([np.ones(len(d)),d]);bread=np.linalg.inv(x.T@x)
    scores=x*r[:,None]; meat=scores.T@scores
    index={int(t):i for i,t in enumerate(times)}
    for lag in range(1,maxlags+1):
        weight=1-lag/(maxlags+1)
        cross=np.zeros((2,2))
        offset=lag*STEP_NS
        for i,ti in enumerate(times):
            j=index.get(int(ti)+offset)
            if j is not None: cross+=np.outer(scores[i],scores[j])
        meat+=weight*(cross+cross.T)
    cov=bread@meat@bread
    se=float(np.sqrt(max(cov[1,1],0)))
    if se==0 or not np.isfinite(se):return {"se":None,"p":None,"ci":[None,None]}
    stat=result["theta"]/se; dof=max(1,len(d)-2)
    p=float(2*student_t.sf(abs(stat),dof));crit=float(student_t.ppf(.975,dof))
    return {"se":se,"statistic":float(stat),"p":p,"ci":[result["theta"]-crit*se,result["theta"]+crit*se],"maxlag_4h":maxlags}

def block_bootstrap(result,seed,reps=B_REPS,block=42):
    d=result["dtilde"];y=result["ytilde"];times=result["time_ns"]
    runs=[];start=0
    for i in range(1,len(times)):
        if times[i]-times[i-1]!=STEP_NS:
            if i-start>=block:runs.append((start,i))
            start=i
    if len(times)-start>=block:runs.append((start,len(times)))
    starts=[i for s,e in runs for i in range(s,e-block+1)]
    if not starts: return {"status":"NO_FULL_BLOCK","valid_replications":0,"ci":[None,None]}
    rng=np.random.default_rng(seed); betas=[]; n=len(d)
    nblocks=int(np.ceil(n/block))
    for _ in range(reps):
        ids=[]
        for b in rng.choice(starts,size=nblocks,replace=True): ids.extend(range(int(b),int(b)+block))
        ids=np.asarray(ids[:n],int)
        x=np.column_stack([np.ones(n),d[ids]])
        beta=np.linalg.lstsq(x,y[ids],rcond=None)[0][1]
        if np.isfinite(beta):betas.append(float(beta))
    if len(betas)<.9*reps:return {"status":"BOOTSTRAP_FAIL","valid_replications":len(betas),"ci":[None,None]}
    ci=np.quantile(betas,[.025,.975])
    return {"status":"COMPLETE","replications":reps,"valid_replications":len(betas),"block_length_4h":block,
            "seed":seed,"ci":[float(ci[0]),float(ci[1])],"median":float(np.median(betas)),
            "positive_fraction":float(np.mean(np.asarray(betas)>0)),"negative_fraction":float(np.mean(np.asarray(betas)<0))}

def main():
    candidates,freeze=verify_freeze()
    log("holdout freeze hashes verified; confirmation rows may now be decoded")
    series=load_full_standardized()
    results=[]
    for k,c in enumerate(candidates["candidates"]):
        a=c["asset"]; s=series[a]
        tr=candidate_arrays(s,c,"forward","DISCOVERY")
        te=candidate_arrays(s,c,"forward","CONFIRMATION")
        rec={"candidate_id":c["candidate_id"],"asset":a,"source":c["source"],"target":c["target"],"lag":c["lag"],
             "discovery_sign":c["discovery_sign"],"discovery_supporting_methods":c["supporting_methods"],
             "n_train":len(tr["D"]),"n_holdout":len(te["D"]),
             "first_holdout_target":pd.to_datetime(te["time_ns"],utc=True).min().isoformat() if len(te["D"]) else None,
             "last_holdout_target":pd.to_datetime(te["time_ns"],utc=True).max().isoformat() if len(te["D"]) else None,
             "learners":{}}
        if len(tr["D"])<100 or len(te["D"])<50:
            rec["status"]="INSUFFICIENT_SAMPLE";results.append(rec);continue
        for learner in ("HGB","Ridge"):
            try:
                fit=fit_orthogonalized(tr,te,learner);hac=hac_slope(fit)
                out={"theta":fit["theta"],"hac":hac,"n_train":fit["n_train"],"n_holdout":fit["n_test"],
                     "block_bootstrap":block_bootstrap(fit,BOOT_SEED+k*31+(0 if learner=="HGB" else 1)) if learner=="HGB" else None,
                     "controls":"all ten frozen variables at exposure time minus 1..6 intervals; no controls at/after exposure",
                     "nuisance_training":"discovery only"}
                rec["learners"][learner]=out
            except Exception as exc: rec["learners"][learner]={"status":"FIT_ERROR","error":f"{type(exc).__name__}: {exc}"}
        # Future-source -> past-target negative control, frozen before opening holdout.
        ptr=candidate_arrays(s,c,"reverse_time","DISCOVERY")
        pte=candidate_arrays(s,c,"reverse_time","CONFIRMATION")
        try:
            pfit=fit_orthogonalized(ptr,pte,"HGB");phac=hac_slope(pfit)
            rec["reverse_time_control"]={"theta":pfit["theta"],"hac":phac,
                "n_train":pfit["n_train"],"n_holdout":pfit["n_test"],
                "flag":"TEMPORAL_DIRECTION_AMBIGUOUS" if phac["p"] is not None and phac["p"]<.05 else "NO_NOMINAL_REVERSE_TIME_SIGNAL"}
        except Exception as exc:
            rec["reverse_time_control"]={"status":"CONTROL_FIT_ERROR","error":f"{type(exc).__name__}: {exc}","flag":"INCONCLUSIVE"}
        rec["status"]="COMPLETE";results.append(rec)
        if (k+1)%5==0 or k==len(candidates["candidates"])-1:
            write_json(OUT/"holdout_confirmation.json",{"candidate_sha256":freeze["candidate_sha256"],
                "protocol_sha256":freeze["protocol_sha256"],"confirmation_data_sha256":json.loads((OUT/"rerun_protocol.json").read_text())["data"]["sha256"],
                "theta_units":"outcome standard deviations per one exposure standard deviation; both use discovery-sample mean/SD scaling",
                "confirmation_sample":"2023-07-01 through 2024-03-03 UTC; exact candidates/lags frozen before decode",
                "results":results})
            log(f"holdout completed candidates={k+1}/{len(candidates['candidates'])}")
    # Holm correction for primary HGB HAC p-values within asset.
    for a in ASSETS:
        inds=[i for i,r in enumerate(results) if r.get("asset")==a and r.get("learners",{}).get("HGB",{}).get("hac",{}).get("p") is not None]
        if inds:
            adj=multipletests([results[i]["learners"]["HGB"]["hac"]["p"] for i in inds],method="holm")[1]
            for i,p in zip(inds,adj):results[i]["learners"]["HGB"]["hac"]["holm_p_within_asset"]=float(p)
    for r in results:
        h=r.get("learners",{}).get("HGB",{}); rr=r.get("learners",{}).get("Ridge",{})
        bb=h.get("block_bootstrap") or {};placebo=r.get("reverse_time_control",{}).get("hac",{})
        theta=h.get("theta");ridge=rr.get("theta")
        sign_match=(theta is not None and r.get("discovery_sign") in (-1,1) and int(np.sign(theta))==r["discovery_sign"])
        ridge_match=(theta is not None and ridge is not None and int(np.sign(theta))==int(np.sign(ridge)))
        boot=bb.get("ci",[None,None]); boot_same=(len(boot)==2 and ((theta is not None and theta>0 and boot[0] is not None and boot[0]>0) or (theta is not None and theta<0 and boot[1] is not None and boot[1]<0)))
        hp=h.get("hac",{}).get("holm_p_within_asset")
        pval=placebo.get("p")
        robust=bool(sign_match and ridge_match and hp is not None and hp<.05 and boot_same and pval is not None and pval>=.05)
        r["classification"]="HOLDOUT_ROBUST_CONFIRMED" if robust else "HOLDOUT_NOMINAL" if hp is not None and hp<.05 else "DISCOVERY_ONLY"
        r["checks"]={"discovery_sign_agreement":sign_match,"ridge_sign_agreement":ridge_match,
            "holm_hgb_below_0_05":bool(hp is not None and hp<.05),"block_bootstrap_95ci_same_sign_excludes_zero":boot_same,
            "reverse_time_control_acceptable":bool(pval is not None and pval>=.05)}
    write_json(OUT/"holdout_confirmation.json",{"candidate_sha256":freeze["candidate_sha256"],
        "protocol_sha256":freeze["protocol_sha256"],"confirmation_data_sha256":json.loads((OUT/"rerun_protocol.json").read_text())["data"]["sha256"],
        "theta_units":"outcome standard deviations per one exposure standard deviation; both use discovery-sample mean/SD scaling",
        "confirmation_sample":"2023-07-01 through 2024-03-03 UTC; exact candidates/lags frozen before decode",
        "primary_holm":"within asset across HGB HAC p-values of candidate edges",
        "results":results,"confirmation_decoded_after_candidate_freeze":True,
        "completed_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")})
    log("HOLDOUT CONFIRMATION COMPLETE")

if __name__=="__main__":main()
