from __future__ import annotations
import json, socket, hashlib, os, datetime
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

from phase5_rerun_common import ROOT, OUT, OLD, PANEL, sha, load_asset_discovery, write_json
from run_phase5_rerun_discovery import fit_pcmci, fit_dynotears, fit_lingam, fit_neural
from run_phase5_rerun_bootstrap import sample_blocks, selected_edges, SEED, BLOCKS, REPS
from phase5_rerun_holdout import main as replay_holdout

def equal_tree(a,b,path="root",atol=1e-8,rtol=1e-7):
    if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):
        aa=np.asarray(a);bb=np.asarray(b)
        if aa.shape!=bb.shape:return False,f"{path}: array shapes {aa.shape} != {bb.shape}"
        try:
            if not np.allclose(aa.astype(float),bb.astype(float),atol=atol,rtol=rtol,equal_nan=True):
                return False,f"{path}: array values differ beyond tolerance"
            return True,""
        except Exception:return False,f"{path}: array comparison failed"
    if isinstance(a,dict) and isinstance(b,dict):
        if set(a)!=set(b): return False,f"{path}: keys differ"
        for k in a:
            ok,msg=equal_tree(a[k],b[k],f"{path}.{k}",atol,rtol)
            if not ok:return ok,msg
        return True,""
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
        if len(a)!=len(b):return False,f"{path}: lengths {len(a)} != {len(b)}"
        for i,(x,y) in enumerate(zip(a,b)):
            ok,msg=equal_tree(x,y,f"{path}[{i}]",atol,rtol)
            if not ok:return ok,msg
        return True,""
    if isinstance(a,(int,float,np.number)) and isinstance(b,(int,float,np.number)):
        if isinstance(a,(bool,np.bool_)) or isinstance(b,(bool,np.bool_)):
            return (bool(a)==bool(b),f"{path}: boolean mismatch")
        try:
            if not np.isclose(float(a),float(b),atol=atol,rtol=rtol,equal_nan=True):
                return False,f"{path}: {a} != {b}"
            return True,""
        except Exception:return False,f"{path}: numeric compare error"
    if a!=b:return False,f"{path}: {a!r} != {b!r}"
    return True,""

def remove_volatile(x):
    if isinstance(x,dict):return {k:remove_volatile(v) for k,v in x.items() if k not in ("elapsed_seconds",)}
    if isinstance(x,list):return [remove_volatile(v) for v in x]
    return x

def main():
    out={"offline":True,"network_guard":"socket connect/create_connection/getaddrinfo blocked during reproduction",
         "network_attempts":0,"checks":{},"tolerance":{"atol":1e-8,"rtol":1e-7}}
    # Disable network at Python socket APIs before any reproduction work.
    def deny(*args,**kwargs):
        out["network_attempts"]+=1
        raise RuntimeError("network disabled for offline reproduction")
    socket.create_connection=deny;socket.getaddrinfo=deny;socket.socket.connect=deny
    input_hash=json.loads((OUT/"data_validation.json").read_text())["panel_sha256"]
    out["checks"]["four_hour_panel_hash"]={"pass":sha(PANEL)==input_hash,"expected":input_hash,"actual":sha(PANEL)}
    inventory=json.loads((OUT/"prior_artifact_hashes.json").read_text())["files"]
    mismatches=[]
    for f in inventory:
        p=ROOT/f["path"]
        if not p.exists() or sha(p)!=f["sha256"]:mismatches.append(f["path"])
    out["checks"]["all_prior_artifacts_unchanged"]={"pass":not mismatches,"inventory_count":len(inventory),"mismatches":mismatches}
    source_checks=json.loads((OUT/"data_validation.json").read_text())["source_hash_checks"]
    out["checks"]["market_raw_hashes_unchanged"]={"pass":all(x["pass"] for x in source_checks),"count":len(source_checks)}
    assets=load_asset_discovery()
    saved={
      "PCMCI+":json.loads((OUT/"pcmci_results.json").read_text()),
      "DYNOTEARS":json.loads((OUT/"dynotears_results.json").read_text()),
      "VAR-LiNGAM":json.loads((OUT/"var_lingam_results.json").read_text()),
      "NOTEARS-MLP":json.loads((OUT/"notears_mlp_results.json").read_text()),
      "Neural Granger/cMLP":json.loads((OUT/"neural_granger_results.json").read_text())}
    methods={"PCMCI+":fit_pcmci,"DYNOTEARS":fit_dynotears,"VAR-LiNGAM":fit_lingam,
             "NOTEARS-MLP":lambda s:fit_neural(s,"notears_mlp"),
             "Neural Granger/cMLP":lambda s:fit_neural(s,"cmlp")}
    point_checks={}
    with threadpool_limits(limits=1):
        for method,fn in methods.items():
            point_checks[method]={}
            for asset,info in assets.items():
                fresh=fn(info["segments"])
                old=remove_volatile(saved[method]["assets"][asset])
                fresh=remove_volatile(fresh)
                ok,msg=equal_tree(old,fresh)
                point_checks[method][asset]={"pass":ok,"detail":msg}
    out["checks"]["deterministic_empirical_point_fits"]={"pass":all(v["pass"] for m in point_checks.values() for v in m.values()),"assets":point_checks}
    # Refit replication zero for every method x asset x block-size configuration.
    bootstrap_checks=[]
    method_order=list(REPS)
    for mi,method in enumerate(method_order):
        for ai,asset in enumerate(("BTC","ETH")):
            ckpath=OUT/"bootstrap_checkpoints"/f"{method.lower().replace('+','plus').replace('/','_').replace(' ','_')}_{asset}_block"
            for bi,block in enumerate(BLOCKS):
                p=Path(str(ckpath)+f"{block}.json")
                ck=json.loads(p.read_text())
                saved_rep=next((x for x in ck["valid_replications"] if int(x["replication"])==0),None)
                if saved_rep is None:
                    bootstrap_checks.append({"method":method,"asset":asset,"block":block,"pass":False,"detail":"replication 0 did not complete"});continue
                rng=np.random.default_rng(SEED+100000*ai+10000*bi+100*mi)
                seg=sample_blocks(assets[asset]["segments"],block,rng)
                with threadpool_limits(limits=1):
                    edges=selected_edges(method,seg,SEED)
                old={(x["source"],x["target"],int(x["lag"])):float(x["statistic"]) for x in saved_rep["edges"]}
                new={(s,t,int(l)):float(v) for s,t,l,v in edges}
                same=set(old)==set(new) and all(np.isclose(old[k],new[k],atol=1e-8,rtol=1e-7) for k in old)
                bootstrap_checks.append({"method":method,"asset":asset,"block":block,"pass":bool(same),
                    "saved_edge_count":len(old),"replayed_edge_count":len(new)})
    out["checks"]["fixed_seed_bootstrap_replication_zero"]={"pass":all(x["pass"] for x in bootstrap_checks),"configurations":bootstrap_checks}
    # Re-run the complete frozen holdout pipeline offline and compare the result rows.
    hp=OUT/"holdout_confirmation.json";old_bytes=hp.read_bytes();old_hold=json.loads(old_bytes)
    try:
        replay_holdout()
        new_hold=json.loads(hp.read_text())
        ok,msg=equal_tree(old_hold["results"],new_hold["results"])
        if not ok: hp.write_bytes(old_bytes)
        out["checks"]["offline_holdout_reproduction"]={"pass":ok,"detail":msg,
            "candidate_sha256_unchanged":old_hold.get("candidate_sha256")==new_hold.get("candidate_sha256"),
            "protocol_sha256_unchanged":old_hold.get("protocol_sha256")==new_hold.get("protocol_sha256")}
    except Exception as exc:
        hp.write_bytes(old_bytes)
        out["checks"]["offline_holdout_reproduction"]={"pass":False,"detail":f"{type(exc).__name__}: {exc}"}
    out["reproduction_scope"]="all method/asset empirical discovery point fits; replication 0 of each method x asset x block-size bootstrap configuration; complete holdout pipeline replay; full bootstrap counts are recorded in bootstrap_stability.json"
    out["overall_pass"]=bool(all(x["pass"] for x in out["checks"].values()))
    out["completed_at_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    write_json(OUT/"reproducibility.json",out)
    print(json.dumps({"overall_pass":out["overall_pass"],"checks":out["checks"]},indent=2))

if __name__=="__main__":main()
