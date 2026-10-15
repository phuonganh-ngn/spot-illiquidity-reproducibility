from __future__ import annotations
import json
from collections import defaultdict
from phase5_rerun_common import OUT, ASSETS, load_asset_discovery, write_json, log
from phase5_rerun_algorithms import pcmci_plus

def main():
    collapsed=json.loads((OUT/"lag_collapsed_edges.json").read_text())
    byasset={a:[] for a in ASSETS}
    for a in ASSETS:
        for e in collapsed["assets"][a]["exact_lag_edges"]:
            very=sum(bool(m["very_stable"]) for m in e["method_evidence"])
            if e["method_support_count"]>=3 or very>=1:
                byasset[a].append(e)
    assets=load_asset_discovery();out={"method":"PCMCI+ ParCorr reverse-time discovery diagnostic",
        "sample":"discovery only; each complete segment reversed separately",
        "threshold":"same source->target and lag has BH q<.05 in reversed-time PCMCI+ at alpha .05; candidates are moderate-consensus or at least one method-specific very-stable edge",
        "assets":{}}
    for a in ASSETS:
        original=assets[a]["segments"]
        reversed_segments=[s[::-1].copy() for s in original]
        log(f"reverse-time PCMCI diagnostic start asset={a} candidate_pairs={len(byasset[a])}")
        r=pcmci_plus(reversed_segments, __import__('phase5_rerun_common').VARS, pc_alpha=.05, ci_kind="ParCorr")
        found={(x["source"],x["target"],int(x["lag"])):x for x in r["edges"]}
        checks=[]
        candidates=sorted({(e["source"],e["target"],int(e["lag"])) for e in byasset[a]})
        for src,dst,lag in candidates:
            x=found.get((src,dst,lag))
            checks.append({"source":src,"target":dst,"lag":lag,
                "reverse_time_detected":x is not None,
                "reverse_time_p":x.get("p") if x else None,
                "reverse_time_q_bh":x.get("q_bh") if x else None,
                "reverse_time_statistic":x.get("statistic") if x else None,
                "flag":"TEMPORAL_DIRECTION_AMBIGUOUS" if x else "NO_REVERSE_TIME_SIGNAL"})
        out["assets"][a]={"reversed_segment_count":len(reversed_segments),
            "observations":r["n_observations"],"checked_edge_lags":checks}
    write_json(OUT/"reverse_time_checks.json",out)
    log("REVERSE-TIME CHECKS COMPLETE")

if __name__=="__main__":main()
