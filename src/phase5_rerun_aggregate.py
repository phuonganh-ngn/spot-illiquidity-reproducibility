from __future__ import annotations
import json
from collections import defaultdict
import numpy as np
from phase5_rerun_common import OUT, VARS, ASSETS, write_json

METHOD_FILES = {
    "PCMCI+":"pcmci_results.json",
    "DYNOTEARS":"dynotears_results.json",
    "NOTEARS-MLP":"notears_mlp_results.json",
    "Neural Granger/cMLP":"neural_granger_results.json",
}
AVAILABLE = list(METHOD_FILES)
DERIVATIVES={"funding_rate_state","oi_log_change","futures_taker_imbalance","basis_mean"}
SPOT=set(VARS)-DERIVATIVES

def load_json(name): return json.loads((OUT/name).read_text())

def method_tuning_support(method, asset):
    obj=load_json(METHOD_FILES[method])["assets"][asset]
    if obj.get("status")!="FIT_COMPLETE": return {}, set()
    out={}
    for e in obj.get("candidate_edges",[]):
        key=(e["source"],e["target"],int(e["lag"]))
        if method=="PCMCI+":
            passed=bool(e.get("stable_across_alpha",False))
            val=float(e.get("median_statistic",0.0))
        elif method=="DYNOTEARS":
            passed=bool(e.get("regularization_stable",False))
            val=float(e.get("median_weight",0.0))
        else:
            passed=bool(e.get("regularization_stable",False))
            val=float(e.get("median_local_derivative",0.0))
        out[key]={"tuning_stable":passed,"discovery_statistic":val,
                  "tuning_support_count":int(e.get("support_count",0))}
    return out, set(out)

def bootstrap_maps():
    b=load_json("bootstrap_stability.json")["bootstrap_status"]
    out={}
    for m in AVAILABLE:
        out[m]={}
        for a in ASSETS:
            out[m][a]={}
            for bl in (42,84):
                item=b[m][a][str(bl)]
                out[m][a][bl]={}
                if item.get("status") not in ("COMPLETE","PARTIAL"): continue
                for e in item["edge_stability"]:
                    key=(e["source"],e["target"],int(e["lag"]))
                    out[m][a][bl][key]=e
    return out

def build_consensus():
    boots=bootstrap_maps()
    tune={m:{a:method_tuning_support(m,a)[0] for a in ASSETS} for m in AVAILABLE}
    per_asset={}; all_pair_keys=set()
    for a in ASSETS:
        exact=[]; pair_acc=defaultdict(lambda:{"methods":{},"lags":defaultdict(list)})
        keys=set()
        for m in AVAILABLE:
            keys |= set(boots[m][a].get(42,{}))
            keys |= set(tune[m][a])
        for src,dst,lag in sorted(keys):
            method_rows=[]
            for m in AVAILABLE:
                b=boots[m][a].get(42,{}).get((src,dst,lag))
                t=tune[m][a].get((src,dst,lag),{})
                if not b: continue
                freq=b.get("selection_frequency")
                stable=bool(freq is not None and freq>=.70 and t.get("tuning_stable",False))
                very=bool(freq is not None and freq>=.85 and t.get("tuning_stable",False))
                row={"method":m,"bootstrap_frequency_42":freq,
                     "bootstrap_frequency_84":boots[m][a].get(84,{}).get((src,dst,lag),{}).get("selection_frequency"),
                     "bootstrap_sign_consistency":b.get("sign_consistency"),
                     "tuning_stable":bool(t.get("tuning_stable",False)),
                     "discovery_statistic":t.get("discovery_statistic"),
                     "method_specific_stable":stable,"very_stable":very}
                method_rows.append(row)
                if stable:
                    pk=(src,dst); pair_acc[pk]["methods"][m]=True
                    pair_acc[pk]["lags"][m].append(lag)
                    all_pair_keys.add((a,src,dst))
            supporters=[x["method"] for x in method_rows if x["method_specific_stable"]]
            n=len(supporters); avail=len(AVAILABLE)
            exact.append({"source":src,"target":dst,"lag":lag,"method_support_count":n,
                "valid_empirical_methods":avail,"support_fraction":n/avail,
                "consensus_class":"STRONG_CONSENSUS" if n>=4 else "MODERATE_CONSENSUS" if n>=3 else "WEAK_CONSENSUS" if n>=2 else "METHOD_SPECIFIC" if n==1 else "NO_SUPPORT",
                "method_evidence":method_rows})
        lag_collapsed=[]
        for (src,dst),record in sorted(pair_acc.items()):
            mlags={m:sorted(set(lags)) for m,lags in record["lags"].items()}
            supports=sorted(mlags)
            lag_collapsed.append({"source":src,"target":dst,"supported_lags_by_method":mlags,
                "supported_lags":sorted({l for ls in mlags.values() for l in ls}),
                "method_support_count":len(supports),"method_count_available":avail,
                "fraction_of_valid_methods_supporting_edge":len(supports)/avail,
                "consensus_class":"STRONG_CONSENSUS" if len(supports)>=4 else "MODERATE_CONSENSUS" if len(supports)>=3 else "WEAK_CONSENSUS" if len(supports)>=2 else "METHOD_SPECIFIC"})
        per_asset[a]={"exact_lag_edges":exact,"lag_collapsed_edges":lag_collapsed}
    write_json(OUT/"lag_collapsed_edges.json",{"method_specific_stability_rule":"42-interval bootstrap frequency >=0.70 and fixed-grid tuning stability; 84-interval frequency is sensitivity only","very_stable_threshold":.85,"assets":per_asset})
    # Cross-method consensus on a lag-collapsed source-target identity.
    consensus=[]
    for a,src,dst in sorted(all_pair_keys):
        row=next(x for x in per_asset[a]["lag_collapsed_edges"] if x["source"]==src and x["target"]==dst)
        consensus.append({"asset":a,**row})
    write_json(OUT/"cross_method_consensus.json",{"valid_methods":AVAILABLE,"valid_method_count":len(AVAILABLE),"thresholds":{"weak":2,"moderate":3,"strong":4},"edges":consensus})
    # Cross-asset replication requires moderate method support in each asset.
    links={}
    for a in ASSETS:
        for e in per_asset[a]["lag_collapsed_edges"]:
            key=(e["source"],e["target"]);links.setdefault(key,{})[a]=e
    replicated=[]
    for (src,dst),by_asset in sorted(links.items()):
        if all(a in by_asset and by_asset[a]["method_support_count"]>=3 for a in ASSETS):
            replicated.append({"source":src,"target":dst,"assets":{
                a:{"method_support_count":by_asset[a]["method_support_count"],"supported_lags":by_asset[a]["supported_lags"],"supported_lags_by_method":by_asset[a]["supported_lags_by_method"]} for a in ASSETS},
                "label":"CROSS_ASSET_REPLICATED"})
    write_json(OUT/"cross_asset_replication.json",{"criterion":"same direction in BTC and ETH; at least three of four valid empirical methods stable for each asset; lags may differ within 1..6","replicated_edges":replicated})
    # Candidate paths use stable, method-specific lagged edges only.
    paths=[]
    for a in ASSETS:
        for m in AVAILABLE:
            stable=[]
            for x in per_asset[a]["exact_lag_edges"]:
                ev=next((q for q in x["method_evidence"] if q["method"]==m),None)
                if ev and ev["method_specific_stable"]:
                    stable.append({"source":x["source"],"target":x["target"],"lag":x["lag"],"frequency":ev["bootstrap_frequency_42"]})
            for e1 in stable:
                for e2 in stable:
                    if e1["target"]!=e2["source"] or e1["source"]==e2["target"]:continue
                    if ((e1["source"] in DERIVATIVES and e2["target"] in SPOT) or
                        (e1["source"] in SPOT and e2["target"] in DERIVATIVES)):
                        paths.append({"asset":a,"method":m,"source":e1["source"],"intermediate":e1["target"],"target":e2["target"],
                            "source_to_intermediate_lag":e1["lag"],"intermediate_to_target_lag":e2["lag"],
                            "implied_horizon_lags":e1["lag"]+e2["lag"],"first_edge_stability":e1["frequency"],"second_edge_stability":e2["frequency"],
                            "interpretation":"candidate temporal path only; not mediation"})
    write_json(OUT/"network_paths.json",{"criterion":"both component edges independently method-specific stable at block length 42","paths":paths})
    return per_asset, replicated

def main():
    build_consensus()

if __name__=="__main__":main()
