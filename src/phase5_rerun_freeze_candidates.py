from __future__ import annotations
import hashlib, json, datetime
from pathlib import Path
from phase5_rerun_common import OUT, write_json, sha

def main():
    # Read discovery-only graph/stability outputs; this script has no panel reader.
    consensus=json.loads((OUT/"cross_method_consensus.json").read_text())
    collapsed=json.loads((OUT/"lag_collapsed_edges.json").read_text())
    replicated=json.loads((OUT/"cross_asset_replication.json").read_text())["replicated_edges"]
    reverse=json.loads((OUT/"reverse_time_checks.json").read_text())
    replic_links={(e["source"],e["target"]):e for e in replicated}
    per_asset={a:{(e["source"],e["target"]):e for e in collapsed["assets"][a]["lag_collapsed_edges"]}
               for a in ("BTC","ETH")}
    rev_maps={a:{(e["source"],e["target"],int(e["lag"])):e for e in reverse["assets"][a]["checked_edge_lags"]}
              for a in ("BTC","ETH")}
    selected=[]
    for asset in ("BTC","ETH"):
        for pair,e in sorted(per_asset[asset].items()):
            src,dst=pair
            exact=next(x for x in consensus["edges"] if x["asset"]==asset and x["source"]==src and x["target"]==dst)
            stable_methods=[m for m,ls in e["supported_lags_by_method"].items() if ls]
            exacts=[x for x in collapsed["assets"][asset]["exact_lag_edges"] if x["source"]==src and x["target"]==dst]
            very_by_method={}
            for x in exacts:
                for ev in x["method_evidence"]:
                    if ev["very_stable"]:
                        very_by_method.setdefault(ev["method"],[]).append(x)
            reasons=[]
            if len(stable_methods)>=3: reasons.append("MODERATE_LAG_COLLAPSED_CONSENSUS")
            if len(very_by_method)>=2: reasons.append("VERY_STABLE_IN_TWO_METHODS")
            cross=replic_links.get((src,dst))
            if cross:
                replicated_methods=set(very_by_method)
                other_asset="ETH" if asset=="BTC" else "BTC"
                other_e=per_asset[other_asset].get((src,dst),{})
                other_exacts=[x for x in collapsed["assets"][other_asset]["exact_lag_edges"] if x["source"]==src and x["target"]==dst]
                other_very={ev["method"] for x in other_exacts for ev in x["method_evidence"] if ev["very_stable"]}
                if replicated_methods & other_very:
                    reasons.append("ONE_METHOD_VERY_STABLE_CROSS_ASSET_REPLICATION")
            if not reasons: continue
            # Freeze one exact lag by support count, then median stability, then smallest lag.
            def score(x):
                es=[v for v in x["method_evidence"] if v["method_specific_stable"]]
                freqs=sorted(v["bootstrap_frequency_42"] for v in es)
                median=float(__import__('numpy').median(freqs)) if freqs else 0.0
                return (len(es),median,-int(x["lag"]))
            best=max(exacts,key=score)
            supporting=[v for v in best["method_evidence"] if v["method_specific_stable"]]
            signs=[1 if (v["discovery_statistic"] or 0)>0 else -1 if (v["discovery_statistic"] or 0)<0 else 0 for v in supporting]
            nonzero=[s for s in signs if s]
            sign=(nonzero[0] if nonzero and len(set(nonzero))==1 else "MIXED" if nonzero else "UNDEFINED")
            rcheck=rev_maps[asset].get((src,dst,int(best["lag"])),{})
            selected.append({"candidate_id":f"{asset}:{src}->{dst}:lag{best['lag']}","asset":asset,
                "source":src,"target":dst,"lag":int(best["lag"]),"lag_hours":4*int(best["lag"]),
                "eligibility_reasons":reasons,"lag_collapsed_method_support_count":len(stable_methods),
                "exact_lag_method_support_count":sum(v["method_specific_stable"] for v in best["method_evidence"]),
                "supporting_methods":[v["method"] for v in supporting],"discovery_sign":sign,
                "discovery_statistics_by_method":{v["method"]:v["discovery_statistic"] for v in supporting},
                "bootstrap_frequency_42_by_method":{v["method"]:v["bootstrap_frequency_42"] for v in supporting},
                "reverse_time_diagnostic":rcheck.get("flag","NOT_CHECKED")})
    # Freeze this modeling rule before any confirmation rows are decoded.
    protocol={"stage":"Phase 5 rerun holdout confirmation","frozen_before_confirmation_read":True,
      "candidate_source":"discovery-only cross-method and bootstrap outputs; no holdout values read",
      "eligibility":"lag-collapsed >=3 method support OR >=2 methods very-stable OR same method very-stable and cross-asset replicated",
      "exact_lag_selection":"within an eligible source-target pair choose lag by largest number of method-specific stable fits, then largest median 42-block selection frequency, then smallest lag; no p-value selection",
      "target_split":"confirmation rows only, target timestamp 2023-07-01T00:00:00Z through 2024-03-03T20:00:00Z",
      "forward_alignment":"source at target_time minus discovered lag; controls are all ten frozen variables at exposure_time minus 1..6 intervals; outcome at target_time",
      "nuisance_training":"discovery target rows only; nuisance predictions for confirmation use training-only fitted models",
      "primary_nuisance":"HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=1.0, random_state=314159)",
      "robustness_nuisance":"Ridge(alpha=1.0), StandardScaler fit on discovery controls only",
      "inference":"OLS slope of orthogonalized residuals with intercept; Bartlett HAC maxlag=42 4-hour observations using exact timestamp separation; moving date-block bootstrap 499 replications, block length 42, seed 20261007",
      "holm":"within asset over candidate forward HGB HAC p-values",
      "reverse_time_control":"source at target_time plus the discovered lag predicts the earlier target; controls use six lags ending before the earlier target; same nuisance learners; any HAC p<0.05 flags temporal ambiguity",
      "robust_confirmation":"Holm p<0.05, unique discovery sign agrees, Ridge sign agrees, 42-block bootstrap 95% interval excludes zero on that sign, and reverse-time HGB p>=0.05",
      "limitations":"holdout candidates and exact lags are selected solely on discovery; holdout inference is predictive/directional, not structural causality"}
    protocol_path=OUT/"holdout_confirmation_protocol.json"
    write_json(protocol_path,protocol)
    protocol_hash=sha(protocol_path)
    obj={"selection_frozen_before_confirmation_read":True,"candidate_count":len(selected),
         "candidates":selected,"eligible_rule":protocol["eligibility"],"lag_selection_rule":protocol["exact_lag_selection"],
         "discovery_consensus_sha256":sha(OUT/"cross_method_consensus.json"),
         "bootstrap_sha256":sha(OUT/"bootstrap_stability.json"),
         "protocol_sha256":protocol_hash,
         "confirmation_values_read":False,
         "frozen_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    out=OUT/"holdout_candidates.json";write_json(out,obj)
    write_json(OUT/"holdout_freeze_manifest.json",{"candidate_file":"holdout_candidates.json","candidate_sha256":sha(out),
        "protocol_file":"holdout_confirmation_protocol.json","protocol_sha256":protocol_hash,
        "confirmation_read_before_freeze":False,"frozen_at_utc":obj["frozen_at_utc"]})
    print(f"Frozen {len(selected)} exact-lag holdout candidates; candidate_sha256={sha(out)}")

if __name__=="__main__":main()
