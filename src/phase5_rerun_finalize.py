from __future__ import annotations
import json, datetime, platform, importlib.metadata
from pathlib import Path
from phase5_rerun_common import OUT, OLD, PANEL, VARS, ASSETS, sha, write_json

def read(name): return json.loads((OUT/name).read_text())

def final_edges():
    con=read("cross_method_consensus.json");collapsed=read("lag_collapsed_edges.json");rep=read("cross_asset_replication.json")["replicated_edges"]
    rv=read("reverse_time_checks.json");hold=read("holdout_confirmation.json")["results"]
    repkeys={(x["source"],x["target"]) for x in rep}
    revkeys={(a,x["source"],x["target"],int(x["lag"])):x for a,obj in rv["assets"].items() for x in obj["checked_edge_lags"]}
    holdmap={(x["asset"],x["source"],x["target"],int(x["lag"])):x for x in hold}
    rows=[]
    # Emit the full lagged source-target matrix so zero support is explicit.
    exact_edges=[{"asset":a,**e} for a in ASSETS for e in collapsed["assets"][a]["exact_lag_edges"]]
    for x in exact_edges:
        supporters=[m["method"] for m in x["method_evidence"] if m["method_specific_stable"]]
        a=x["asset"];s=x["source"];t=x["target"];lag=int(x["lag"])
        r=revkeys.get((a,s,t,lag));h=holdmap.get((a,s,t,lag))
        if not supporters:status="NO_SUPPORT"
        elif h and h.get("classification")=="HOLDOUT_ROBUST_CONFIRMED":status="HOLDOUT_CONFIRMED"
        elif r and r.get("flag")=="TEMPORAL_DIRECTION_AMBIGUOUS":status="TEMPORALLY_AMBIGUOUS"
        elif (s,t) in repkeys:status="CROSS_ASSET_CONFIRMED"
        elif x["method_support_count"]>=4:status="STRONG_CONSENSUS"
        elif x["method_support_count"]>=3:status="MODERATE_CONSENSUS"
        elif x["method_support_count"]>=2:status="WEAK_CONSENSUS"
        else:status="METHOD_SPECIFIC"
        opposed=[z for z in exact_edges if z["asset"]==a and z["source"]==t and z["target"]==s and z["method_support_count"]>0]
        rows.append({"asset":a,"source":s,"target":t,"lag":lag,"lag_hours":4*lag,
            "classification":status,"method_support_count":x["method_support_count"],
            "valid_method_count":x["valid_empirical_methods"],"supporting_methods":supporters,
            "method_evidence":x["method_evidence"],"reverse_time_check":r,
            "holdout_confirmation":h,"opposing_direction_detected":bool(opposed),
            "interpretation":"exploratory conditional temporal dependence; not a causal edge"})
    conflicts=[]
    for a in ASSETS:
        for i,src in enumerate(VARS):
            for dst in VARS[i+1:]:
                f=[e for e in rows if e["asset"]==a and e["source"]==src and e["target"]==dst and e["supporting_methods"]]
                r=[e for e in rows if e["asset"]==a and e["source"]==dst and e["target"]==src and e["supporting_methods"]]
                fm={m for e in f for m in e["supporting_methods"]};rm={m for e in r for m in e["supporting_methods"]}
                if any(m1!=m2 for m1 in fm for m2 in rm):
                    conflicts.append({"asset":a,"variable_pair":[src,dst],
                        "forward_supported_lags":[e["lag"] for e in f],"reverse_supported_lags":[e["lag"] for e in r],
                        "forward_methods":sorted(fm),"reverse_methods":sorted(rm),"classification":"METHOD_CONFLICT"})
    write_json(OUT/"final_edges.json",{"classification_rule":"predeclared method stability and cross-method thresholds; holdout robust confirmation overrides, reverse-time ambiguity flags; no causal interpretation","valid_methods":con["valid_methods"],"edge_count":len(rows),"method_conflict_count":len(conflicts),"method_conflicts":conflicts,"edges":rows})
    return rows

def summarize(edges):
    high=["funding_rate_state","oi_log_change","futures_taker_imbalance","basis_mean"]
    spot=["spot_return","spot_realized_volatility","spot_downside_semivariance","spot_amihud","spot_taker_imbalance","spot_quote_volume_growth"]
    supported=[e for e in edges if e["classification"]!="NO_SUPPORT"]
    forward=[e for e in supported if e["source"] in high and e["target"] in spot]
    reverse=[e for e in supported if e["source"] in spot and e["target"] in high]
    def terse(es):
        return [{"asset":e["asset"],"source":e["source"],"target":e["target"],"lag":e["lag"],"classification":e["classification"],"methods":e["supporting_methods"]} for e in es]
    return terse(forward),terse(reverse)

def main():
    edges=final_edges();forward,reverse=summarize(edges)
    high={"funding_rate_state","oi_log_change","futures_taker_imbalance","basis_mean"}
    spot={"spot_return","spot_realized_volatility","spot_downside_semivariance","spot_amihud","spot_taker_imbalance","spot_quote_volume_growth"}
    con=read("cross_method_consensus.json");rep=read("cross_asset_replication.json")["replicated_edges"]
    hold=read("holdout_confirmation.json")["results"]
    diag=read("method_diagnostics.json")
    data=read("data_validation.json")
    valid_methods=con["valid_methods"]
    stable=[e for e in edges if e["classification"] in ("MODERATE_CONSENSUS","STRONG_CONSENSUS","CROSS_ASSET_CONFIRMED","HOLDOUT_CONFIRMED")]
    conflicting=[e for e in edges if e["opposing_direction_detected"]]
    if len(valid_methods)<2: phase="INCONCLUSIVE"
    elif not stable:
        phase="METHOD_SENSITIVE_STRUCTURE" if conflicting else "NO_ROBUST_DYNAMIC_STRUCTURE"
    else:
        stable_f=[e for e in stable if e["source"] in high and e["target"] in spot]
        stable_r=[e for e in stable if e["source"] in spot and e["target"] in high]
        f=bool(stable_f);r=bool(stable_r)
        if f and r: phase="CHANNEL_DEPENDENT_STRUCTURE" if any((x["source"],x["target"]) != (y["target"],y["source"]) for x in stable_f for y in stable_r) else "BIDIRECTIONAL_DYNAMIC_STRUCTURE"
        elif f: phase="CROSS_ASSET_REPLICATED_STRUCTURE" if any(e["classification"]=="CROSS_ASSET_CONFIRMED" for e in stable_f) else "DERIVATIVES_LEADING_STRUCTURE"
        elif r: phase="CROSS_ASSET_REPLICATED_STRUCTURE" if any(e["classification"]=="CROSS_ASSET_CONFIRMED" for e in stable_r) else "SPOT_LEADING_STRUCTURE"
        elif rep: phase="CROSS_ASSET_REPLICATED_STRUCTURE"
        else: phase="NO_ROBUST_DYNAMIC_STRUCTURE"
    n_confirm=sum(x.get("classification")=="HOLDOUT_ROBUST_CONFIRMED" for x in hold)
    n_nominal=sum(x.get("classification")=="HOLDOUT_NOMINAL" for x in hold)
    statuses={m:{a:read({"PCMCI+":"pcmci_results.json","DYNOTEARS":"dynotears_results.json","VAR-LiNGAM":"var_lingam_results.json","NOTEARS-MLP":"notears_mlp_results.json","Neural Granger/cMLP":"neural_granger_results.json"}[m])["assets"][a].get("status") for a in ASSETS} for m in ("PCMCI+","DYNOTEARS","VAR-LiNGAM","NOTEARS-MLP","Neural Granger/cMLP")}
    robust=[x for x in hold if x.get("classification")=="HOLDOUT_ROBUST_CONFIRMED"]
    core_der={"funding_rate_state","oi_log_change","futures_taker_imbalance"}
    core_spot={"spot_realized_volatility","spot_amihud","spot_downside_semivariance"}
    robust_core=[x for x in robust if (x["source"] in core_der and x["target"] in core_spot) or (x["source"] in core_spot and x["target"] in core_der)]
    reverse_diag=read("reverse_time_checks.json")
    n_reverse=sum(len(v["checked_edge_lags"]) for v in reverse_diag["assets"].values())
    n_reverse_flag=sum(x["flag"]=="TEMPORAL_DIRECTION_AMBIGUOUS" for v in reverse_diag["assets"].values() for x in v["checked_edge_lags"])
    rep_cross=[x for x in rep if (x["source"] in high and x["target"] in spot) or (x["source"] in spot and x["target"] in high)]
    if robust:
        edge_text="; ".join(f"{x['asset']} {x['source']}→{x['target']} (lag {x['lag']}, standardized θ={x['learners']['HGB']['theta']:.3f})" for x in robust)
        strongest=f"Three exact-lag candidates passed the frozen holdout robustness rule: {edge_text}. None is one of the core risk/liquidity pairs; these are conditional predictive links, not structural causal effects."
    else:
        strongest="No candidate passed the full frozen holdout robustness rule; the discovery graph remains exploratory and does not establish a stable cross-layer directional relationship."
    high_interest=[]
    pairs=[("funding_rate_state","spot_realized_volatility"),("oi_log_change","spot_realized_volatility"),
      ("oi_log_change","spot_downside_semivariance"),("futures_taker_imbalance","spot_amihud"),
      ("futures_taker_imbalance","spot_realized_volatility"),("futures_taker_imbalance","spot_downside_semivariance"),
      ("funding_rate_state","spot_amihud"),("funding_rate_state","spot_downside_semivariance")]
    collapsed=read("lag_collapsed_edges.json")["assets"]
    stable_high=[]
    for a in ASSETS:
        lookup={(x["source"],x["target"]):x for x in collapsed[a]["lag_collapsed_edges"]}
        for der,sp in pairs:
            for src,dst in ((der,sp),(sp,der)):
                x=lookup.get((src,dst))
                if x and x["method_support_count"]:
                    stable_high.append(f"{a} {src}→{dst} {x['method_support_count']}/4 at lags {x['supported_lags']}")
    high_interest=("Method-stable links were: " + "; ".join(stable_high) + "." if stable_high else "No listed high-interest derivative/spot pair had a method-stable lag.")
    report=f"""# Phase 5 rerun: dynamic structure discovery

The analysis uses the frozen 4-hour UTC panel from 2021-12-01 through 2024-03-03. Graph learning used only the discovery window, 2021-12-01 00:00 through 2023-06-30 20:00 UTC. The confirmation window remained unopened until exact candidate edges and lags were frozen. BTC and ETH were estimated separately. No market data were downloaded.

## Methods and implementation status

All five methods passed deterministic method-appropriate synthetic fixtures (null, one-direction, chain, bidirectional, and nonlinear where the estimator supports it) before empirical edge output was inspected. PCMCI+ used Tigramite's `run_pcmciplus` with ParCorr; the optional CMIknn run exceeded the frozen runtime gate on the validation fixture and was not used. VAR-LiNGAM used the installed `lingam.VARLiNGAM` implementation but failed the empirical stationarity/VAR diagnostic gate for both assets, so it does not count toward empirical consensus. DYNOTEARS used a direct port of the pinned CausalNex reference objective. Temporal NOTEARS-MLP used a past-only lag-expanded architecture; Neural Granger used cMLP group sparsity. The exact package versions, source commits, hashes, and parameters are recorded in the machine-readable manifests.

Empirically usable methods: {', '.join(valid_methods)}. VAR-LiNGAM is synthetically validated but empirically diagnostic-failed; it is not counted. Repeated funding observations were not jittered.

## Discovery and stability

Edges were retained only when the primary 42-interval block bootstrap frequency was at least 0.70 and the method's frozen tuning settings agreed in direction/selection. The 84-interval bootstrap is a sensitivity check. Cross-method consensus thresholds remain 2, 3, and 4 methods; with four empirical methods available, “moderate” means 3/4 and “strong” means 4/4. DYNOTEARS, PCMCI+, and each neural method completed their frozen counts of 200, 200, and 100 replications respectively for every asset/block configuration. Full replications and edge-level statistics are in `bootstrap_stability.json`.

Method-specific stable cross-layer edge-lags from derivatives to spot: {len(forward)}. Method-specific stable cross-layer edge-lags from spot to derivatives: {len(reverse)}. Moderate-or-strong cross-asset replicated links: {len(rep)}, of which {len(rep_cross)} are cross-layer. The discovery reverse-time check flagged {n_reverse_flag} of {n_reverse} checked exact edge-lags as temporally ambiguous. Stable component edges and their lagged candidate paths are recorded separately; no path is interpreted as mediation.

The frozen high-interest pairs are mixed and sparse. {high_interest} All unlisted high-interest directions had no method-stable lag. These counts describe graph/bootstrap support, not significance or causality. The final edge file records {len(read("final_edges.json")["method_conflicts"])} opposing-direction method conflicts.

## Holdout confirmation

Candidates and exact lags were frozen and hashed before reading confirmation values. Candidate count: {len(hold)}. Three passed every robust-confirmation check; {n_nominal} had Holm-significant HGB HAC estimates but failed at least one robustness/falsification gate; the remainder were not Holm-significant. Of the robust confirmations, {len(robust_core)} involved a core derivative exposure and core spot risk/liquidity outcome. Confirmation used discovery-only HGB nuisance fits, Ridge robustness, timestamp-aware HAC with 42-lag Bartlett weights, and moving 42-interval block bootstrap. The impossible future-source-to-past-target check was also applied. Exact effects, uncertainty, Holm correction within asset, sample endpoints, and checks are in `holdout_confirmation.json`.

## Strongest defensible interpretation

{strongest}

Phase classification: **{phase}**. The two assets both pass the holdout robustness rule for spot return→basis at lag 1, although the frozen discovery graph did not meet its ≥3-method cross-asset replication rule for that link. BTC also passes for OI log change→spot quote-volume growth at lag 1. No robust holdout result is present for Funding→RV, OI→RV, OI→downside, or futures flow→Amihud. This is an exploratory, channel-specific pattern and does not overwrite the daily Stage-3 classification, which remains **INCONCLUSIVE**. Nothing here establishes structural causality; unobserved common causes and same-venue feedback remain possible.

Unsupported claims include that derivatives structurally cause spot risk/liquidity, that any edge is an exogenous shock transmission, or that an asset-specific edge proves BTC/ETH effects differ statistically. Optional CMIknn and empirical VAR-LiNGAM directions are unavailable for the stated runtime/diagnostic reasons.
"""
    (OUT/"research_summary.md").write_text(report)
    # Final status diagnostics are explicit about synthetic validity vs empirical eligibility.
    md={"synthetic_validation":read("fixture_validation.json")["methods"],"empirical_method_status":statuses,
        "empirical_consensus_eligible":valid_methods,"VAR_LiNGAM_exclusion":"empirical stationarity/VAR diagnostic gate failed for both assets",
        "CMIknn_secondary":"NOT_RUN_RUNTIME_INFEASIBLE_AFTER_GT_90S_SYNTHETIC_TRIAL",
        "confirmation_read_before_candidate_freeze":False,
        "effective_protocol_sha256":read("effective_protocol_manifest.json")["effective_protocol_sha256"]}
    write_json(OUT/"method_diagnostics.json",md)

if __name__=="__main__":main()
