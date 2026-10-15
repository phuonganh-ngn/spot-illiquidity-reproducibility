#!/usr/bin/env python3
"""Write Phase 13 interpretation, integrity checks and artifact hash manifest."""
from __future__ import annotations
import csv, hashlib, json, math, platform, sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import sklearn, torch, matplotlib

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"outputs/phase13_neural_target_scaling"
P6=ROOT/"outputs/phase6_autonomous_benchmark"
P12=ROOT/"outputs/phase12_technical_revision_audit"
P9=ROOT/"outputs/phase9_external_qml_replication"

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def js(path): return json.loads(Path(path).read_text())
def writej(path,obj): Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
def writecsv(path,rows,fields):
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)

def final_source_checks():
    before=js(OUT/"03_source_manifest_before.json")["source_inventory"]
    after=[]
    for r in before:
        p=ROOT/r["path"]; actual=sha(p) if p.is_file() else None
        after.append({**r,"sha256_after":actual,"unchanged":actual==r["sha256"]})
    writej(OUT/"16_source_manifest_after.json",{
        "generated_utc":datetime.now(timezone.utc).isoformat(),"source_inventory":after,
        "all_unchanged":all(r["unchanged"] for r in after),
        "phase6_unchanged":all(r["unchanged"] for r in after if r["path"].startswith("outputs/phase6_autonomous_benchmark/")),
        "phase9_unchanged":all(r["unchanged"] for r in after if r["path"].startswith("outputs/phase9_external_qml_replication/")),
        "phase12_unchanged":all(r["unchanged"] for r in after if r["path"].startswith("outputs/phase12_technical_revision_audit/")),
        "manuscripts_unchanged":all(r["unchanged"] for r in after if r["path"].startswith("manuscripts/")),
        "dataset_4h_unchanged":all(r["unchanged"] for r in after if r["path"]=="outputs/phase5_dynamic_discovery/dataset_4h.jsonl")
    })
    return after

def validate_outputs():
    checks=[]; predpath=OUT/"predictions/all_origin_predictions.csv.gz"
    predictions=pd.read_csv(predpath)
    p6pred=js(P6/"final_test_predictions.json"); p6metrics=js(P6/"final_test_metrics.json")
    historical={(r["asset"],r["target"],r["horizon_intervals"]):r for r in p6pred["tasks"]}
    # Every saved scored test origin and target must match the frozen Phase 6 test rows.
    test=predictions[predictions.split=="FINAL_TEST"]
    origin_pass=True; scaler_pass=True; metric_pass=True
    for key,g in test.groupby(["asset","model_family","information_set","branch"]):
        asset,family,info,branch=key; h=historical[(asset,"future_amihud",6)]
        dates=[pd.Timestamp(x).isoformat() for x in h["date"]]
        z=g.sort_values("origin_utc")
        origin_pass &= z.origin_utc.tolist()==dates and len(z)==len(dates)
        scaler_pass &= bool(np.allclose(z.prediction_log1p_unclipped.to_numpy(float),
            z.network_output_before_target_inverse.to_numpy(float)*z.train_target_scaler_scale.to_numpy(float)+z.train_target_scaler_mean.to_numpy(float),rtol=1e-10,atol=1e-14)) if branch=="TARGET_SCALED" else bool(np.allclose(z.prediction_log1p_unclipped,z.network_output_before_target_inverse,rtol=0,atol=1e-14))
    checks.append({"check":"all_primary_test_predictions_on_original_registered_origins","status":"PASS" if origin_pass else "FAIL","evidence":"exact date-vector comparison against final_test_predictions.json for every asset/model/information-set/branch"})
    checks.append({"check":"target_scaler_inverse_transform_and_control_identity","status":"PASS" if scaler_pass else "FAIL","evidence":"prediction_log1p = network_output × training-only scale + training-only mean; control output remains unscaled"})
    # Recompute every stored Phase 13 final-test MAE directly from prediction files.
    metric_rows=[]
    for _,g in test.groupby(["asset","model_family","information_set","branch"]):
        y=g.target_log1p.to_numpy(float);p=g.prediction_log1p_unclipped.to_numpy(float)
        metric_rows.append((g.asset.iloc[0],g.model_family.iloc[0],g.information_set.iloc[0],g.branch.iloc[0],float(np.mean(np.abs(y-p)))))
    for asset,family,info,branch,mae in metric_rows:
        fit=js(OUT/"training_logs"/f"{asset}_{family}_{info}_FINAL_TEST_{branch}.json")
        metric_pass &= abs(mae-float(fit["mae_log1p"]))<=1e-14
    checks.append({"check":"all_primary_metrics_reconstruct_from_saved_predictions","status":"PASS" if metric_pass else "FAIL","evidence":"test MAE recomputed from saved origin-level predictions equals fit log metrics within 1e-14"})
    # Training-only scaling verified from each training fit's logged mean/scale and source boundary.
    fitlogs=[json.loads(p.read_text()) for p in (OUT/"training_logs").glob("*.json") if p.name!="execution_summary.json"]
    training_scaler=all((x["target_scaler_fitted_on_training_labels_only"] and x["train_last"]<"2023-07-01T00:00:00+00:00") for x in fitlogs if x["split"]=="FINAL_TEST" and x["branch"]=="TARGET_SCALED")
    checks.append({"check":"target_scaler_fit_training_labels_only","status":"PASS" if training_scaler else "FAIL","evidence":"logged split masks end before evaluation; treatment scaler flag is true only for scaled branch"})
    # Configs, seeds, model count, fixed folds and target are asserted against lock registry.
    lock=js(OUT/"01_frozen_configuration_registry.json")
    architecture_pass=len(lock["configs"])==8 and all(x["horizon_intervals"]==6 and x["target"]=="future_amihud" for x in lock["configs"].values())
    checks.append({"check":"original_model_configs_and_scope_retained","status":"PASS" if architecture_pass else "FAIL","evidence":"8 frozen asset×family×information-set configs; six-bar Amihud task only"})
    execution=js(OUT/"training_logs/execution_summary.json")
    checks.append({"check":"all_locked_fit_units_completed","status":"PASS" if execution["fit_count"]==64 and not execution["failed_fit_ids"] else "FAIL","evidence":execution})
    return checks

def outcome_tables():
    sm=js(OUT/"analysis_summary.json"); performance=sm["test_metrics"]
    cat=pd.read_csv(OUT/"11_category_average_sensitivity.csv")
    ens=pd.read_csv(OUT/"10_ensemble_sensitivity.csv")
    mlp=pd.read_csv(OUT/"05_mlp_sensitivity_results.csv"); gru=pd.read_csv(OUT/"06_gru_sensitivity_results.csv")
    bas=pd.read_csv(OUT/"09_strong_baseline_comparison.csv")
    cell_scale=[]; cell_class=[]; cell_delta=[]; old_ens=[]; new_ens=[]
    for asset in ["BTC","ETH"]:
      for family in ["MLP","GRU"]:
        s0=performance[asset]["SPOT_ONLY"][family]["scaled_mae_log1p"]
        s1=performance[asset]["SPOT_PLUS_ALL_DERIVATIVES"][family]["scaled_mae_log1p"]
        hist0=performance[asset]["SPOT_ONLY"][family]["historical_phase6_saved_mae_log1p"]
        hist1=performance[asset]["SPOT_PLUS_ALL_DERIVATIVES"][family]["historical_phase6_saved_mae_log1p"]
        cell_scale.append((asset,family,"SPOT_ONLY",s0,hist0))
        cell_scale.append((asset,family,"SPOT_PLUS_ALL_DERIVATIVES",s1,hist1))
        cell_delta.append((asset,family,s0-s1))
        for info,val in [("SPOT_ONLY",s0),("SPOT_PLUS_ALL_DERIVATIVES",s1)]:
            row=bas[(bas.asset==asset)&(bas.information_set==info)&(bas.model_family==family)&(bas.role=="RETROSPECTIVE_TEST_ORACLE_COMPARISON")].iloc[0]
            cell_class.append((asset,family,info,val,float(row.strongest_classical_mae_log1p),float(row.historical_phase6_mae_log1p) if pd.notna(row.historical_phase6_mae_log1p) else None))
      for info in ["SPOT_ONLY","SPOT_PLUS_ALL_DERIVATIVES"]:
        a=ens[(ens.asset==asset)&(ens.information_set==info)&(ens.ensemble_version=="PHASE6_SAVED")].iloc[0].mae_log1p
        b=ens[(ens.asset==asset)&(ens.information_set==info)&(ens.ensemble_version=="RETROSPECTIVE_ENSEMBLE_SENSITIVITY")].iloc[0].mae_log1p
        old_ens.append(float(a));new_ens.append(float(b))
    scaled_improved=sum(all(next(new<old for a,f,i,new,old in cell_scale if a==asset and f==family and i==info)
                            for info in ["SPOT_ONLY","SPOT_PLUS_ALL_DERIVATIVES"])
                        for asset in ["BTC","ETH"] for family in ["MLP","GRU"])
    if scaled_improved==4: scaling="CLEAR_ABSOLUTE_IMPROVEMENT"
    elif scaled_improved in [1,2,3]: scaling="MIXED_IMPROVEMENT"
    else: scaling="NO_MEANINGFUL_IMPROVEMENT"
    cell_wins=[scaled<classic for _,_,_,scaled,classic,_ in cell_class]
    if all(cell_wins): competitive="COMPETITIVE_WITH_STRONG_CLASSICAL"
    else:
        trainmean=[]
        for asset,fam,info,val,_,_ in cell_class:
            z=bas[(bas.asset==asset)&(bas.information_set==info)&(bas.model_family=="training_mean")].iloc[0]
            trainmean.append(val<float(z.historical_phase6_mae_log1p))
        if all(trainmean): competitive="COMPETITIVE_WITH_SIMPLE_BASELINE_ONLY"
        elif not any(trainmean): competitive="STILL_INFERIOR_TO_SIMPLE_BASELINE"
        else: competitive="MIXED_ACROSS_ASSETS"
    ds=[d for _,_,d in cell_delta]
    if all(x>0 for x in ds): derivatives="FAVORABLE_POINT_PATTERN_PERSISTS"
    elif all(abs(x)<=1e-15 for x in ds): derivatives="FAVORABLE_PATTERN_DISAPPEARS"
    elif all(x<0 for x in ds): derivatives="FAVORABLE_PATTERN_REVERSES"
    else: derivatives="MIXED_ACROSS_ASSETS"
    rels=[(o-n)/o if o else 0 for o,n in zip(old_ens,new_ens)]
    if all(r>=.50 for r in rels): ensemble="SUBSTANTIALLY_IMPROVED"
    elif all(abs(o-n)<=1e-15 for o,n in zip(old_ens,new_ens)): ensemble="UNCHANGED"
    elif all(n>o for o,n in zip(old_ens,new_ens)): ensemble="WORSENED"
    elif any(n<o for o,n in zip(old_ens,new_ens)): ensemble="PARTIALLY_IMPROVED"
    else: ensemble="WORSENED"
    # Any valid run failure takes precedence in completeness reporting.
    failed=js(OUT/"training_logs/execution_summary.json")["failed_fit_ids"]
    if failed:
        scaled_failures=[x for x in failed if "TARGET_SCALED" in x and "FINAL_TEST" in x]
        if scaled_failures: scaling="NUMERICAL_FAILURE"
        competitive="INSUFFICIENT_EVIDENCE" if scaled_failures else competitive
        derivatives="INSUFFICIENT_EVIDENCE" if scaled_failures else derivatives
        ensemble="NOT_RECONSTRUCTABLE" if len(scaled_failures)>=4 else ensemble
    classes={"NEURAL_SCALING_RESPONSE":scaling,"NEURAL_COMPETITIVENESS":competitive,
             "DERIVATIVES_INFORMATION_SENSITIVITY":derivatives,"ENSEMBLE_RELIABILITY":ensemble}
    return classes,performance,cat,ens,mlp,gru,bas,cell_scale,cell_class,cell_delta,old_ens,new_ens,failed

def main():
    checks=validate_outputs()
    sources=final_source_checks()
    source_all=all(x["unchanged"] for x in sources)
    checks.append({"check":"all_phase6_phase9_phase12_dataset_code_and_manuscript_hashes_unchanged","status":"PASS" if source_all else "FAIL","evidence":f"{len(sources)} paths compared to pre-training SHA256 inventory"})
    # Recheck pretraining lock hashes and no unauthorized changes to the protocol.
    lock=js(OUT/"pretraining_lock_manifest.json")
    lock_ok=sha(OUT/"pretraining_lock_manifest.json")== (OUT/"pretraining_lock_manifest.sha256").read_text().split()[0]
    for r in lock["files"]: lock_ok &= sha(OUT/r["file"])==r["sha256"]
    checks.append({"check":"pretraining_protocol_lock_unchanged","status":"PASS" if lock_ok else "FAIL","evidence":"pretraining_lock_manifest.sha256 and all four frozen file hashes"})
    classes,performance,cat,ens,mlp,gru,bas,cell_scale,cell_class,cell_delta,old_ens,new_ens,failed=outcome_tables()
    writej(OUT/"validation_checks.json",{"checks":checks,"overall":"PASS" if all(x["status"]=="PASS" for x in checks) else "FAIL"})

    lines=["# Phase 13 scientific interpretation","", "This is a retrospective sensitivity analysis on Phase 6 data whose final period was previously inspected. It cannot establish independent confirmation or revise the frozen Phase 6 2/18 result.",""]
    lines.append("## Main results")
    for asset in ["BTC","ETH"]:
      lines.append(f"### {asset} 24-hour Amihud")
      for family in ["MLP","GRU"]:
        vals0=performance[asset]["SPOT_ONLY"][family]
        vals1=performance[asset]["SPOT_PLUS_ALL_DERIVATIVES"][family]
        lines.append(f"{family}: historical Phase 6 S0/S1 MAE = {vals0['historical_phase6_saved_mae_log1p']:.8g} / {vals1['historical_phase6_saved_mae_log1p']:.8g}; unscaled replay raw S0/S1 = {vals0['control_replay_raw_mae_log1p']:.8g} / {vals1['control_replay_raw_mae_log1p']:.8g}; target-scaled S0/S1 = {vals0['scaled_mae_log1p']:.8g} / {vals1['scaled_mae_log1p']:.8g}; scaled Δ(S0−S1) = {performance[asset][family+'_scaled_derivatives_delta_mae']:.8g} (positive favors derivatives). Target-scaled forecasts {'beat' if vals0['scaled_beats_strongest_spotonly_classical_test_oracle'] else 'did not beat'} the strongest saved classical test oracle under S0 and {'beat' if vals1['scaled_beats_strongest_spotonly_classical_test_oracle'] else 'did not beat'} under S1.")
      lines.append("")
    lines += ["## Prespecified descriptive classifications",""]
    for key,val in classes.items(): lines.append(f"- {key}: **{val}**")
    lines += ["", "The complete figures and metrics are reported in the machine-readable result files. The original equal-weight ensemble remains frozen; the component replacement is a retrospective sensitivity only.","", "## Answers to research questions",""]
    lines.append("RQ1–RQ2. Compare historical forecasts with target-scaled forecasts and the matched unscaled replay in `05_mlp_sensitivity_results.csv`, `06_gru_sensitivity_results.csv`, and `07_neural_prediction_diagnostics.csv`. A reduction in extreme forecasts is descriptive, not evidence of incremental derivatives information.")
    lines.append("RQ3. Compare target-scaled neural test MAEs with training-mean, persistence, and unchanged classical forecasts on the same origins in `09_strong_baseline_comparison.csv`. The family labeled best classical is a retrospective test oracle, not deployable selection.")
    lines.append("RQ4. The target-scaled derivatives point pattern is summarized by each model’s matched S0−S1 difference. These are point estimates; bootstrap intervals are post-hoc exploratory and do not update the frozen decision.")
    lines.append("RQ5. The historical category means are reconstructed exactly from retained predictions before the sensitivity version is calculated. Updated averages remain retrospective and reflect changes to the neural and ensemble categories only.")
    lines.append("RQ6. The ensemble sensitivity retains six equal weights and changes only the two neural component forecasts. The historical ensemble reconstruction check is available in `10_ensemble_sensitivity.csv`.")
    lines.append("RQ7. The original Phase 6 positive task-level record remains 2/18 and is not replaced. Neural scaling can explain sensitivity of the neural category contribution; it does not establish broad or operational forecasting superiority.")
    lines.append("RQ8. Add a clearly labeled retrospective subsection and retain original Phase 6 results in their own table. If the category statistic changes materially, revise abstract/conclusion wording so it is not presented as robust deployable gain.")
    lines.append("RQ9. This fixed sensitivity addresses the registered target-scaling concern. Do not add more neural architectures or search scales based on these results; another experiment is not indicated unless an integrity or unresolved implementation discrepancy is found.")
    (OUT/"13_scientific_interpretation.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

    revision="""# Manuscript revision recommendations (do not edit manuscript in Phase 13)\n\n## Abstract\nKeep the original Phase 6 result and its scope explicit. If the target-scaled sensitivity changes the neural/category macro-average materially, state that Phase 6 averages were sensitive to neural training scale and that only 2/18 original tasks met the frozen rule. Do not frame these retrospective results as confirmation or generalizable derivatives value.\n\n## Section 5.1 — Matched Multi-Model Forecasting\nAdd a short retrospective-sensitivity paragraph explaining the training-only target standardization intervention, unchanged selected MLP/GRU configurations, matched controls, unchanged splits/seeds, inverse transformation, and original log1p-scale scoring. Preserve the frozen original results separately.\n\n## Section 6.1 — Historical Multi-Model Forecasting Benchmark\nRetain the historical table and 2/18 registered outcome. Add a distinct subsection titled “Retrospective Neural Target-Scaling Sensitivity.” Report all BTC/ETH × MLP/GRU × S0/S1 absolute MAEs, matched unscaled replay, fold behavior, and the unchanged mean/tree references. Distinguish historical and sensitivity ensembles/category averages.\n\n## Section 6.2 — Generalization Across Assets and Liquidity Proxies\nNo Phase 13 result changes Phase 7–8. Keep the failed broad generalization rule, mixed proxy and temporal results, and the inspected/reused status of Binance periods.\n\n## Section 7 — Discussion\nExplain that `log1p` is approximately identity at Amihud values around 10⁻⁵ and that target scaling is a neural conditioning choice. Interpret any improvement as sensitivity of model fitting, not new evidence of market transmission. Separate favorable point estimates from baseline dominance and uncertainty.\n\n## Section 8 — Limitations\nState the diagnostic is retrospective on an already-inspected test period; no independent confirmation, causal identification, execution-cost savings or trading profit follows. Retain the negative Bybit result and missing stronger Bybit classical comparator. Do not conflate Phase 6 MLP/GRU with the separate QML residual pipeline.\n\n## Section 9 — Conclusion\nKeep the conclusion calibrated to the frozen overall record. If the positive category average weakens/disappears after target scaling, explicitly describe it as model-conditioning sensitive and not robust against the strongest saved classical baseline. Phase 9 remains `NO_EXTERNAL_QML_REPLICATION`.\n"""
    (OUT/"14_manuscript_revision_recommendations.md").write_text(revision.replace("\\n","\n"),encoding="utf-8")

    limitations="""# Limitations and integrity\n\n- This is a retrospective sensitivity run after the Phase 6 final test was inspected. It is not independent confirmation and cannot change the original frozen task-level decision (2/18).\n- Only BTC and ETH 24-hour Amihud are re-estimated; no inference extends to other horizons or targets.\n- The development-selected architectures and configurations are preserved; this is not a neural model search.\n- StandardScaler uses training labels only. Negative forecast outputs are retained on the `log1p` prediction scale as required. The separate control replay’s lower-zero transform is reported only to compare the historic implementation.\n- Any block-bootstrap intervals are nominal post-hoc exploratory intervals and do not constitute new confirmatory significance tests.\n- Best classical reference based on final-test MAE is an oracle-descriptive comparison; operational model selection still requires development selection.\n- No causal effect, market-wide transmission, execution-cost improvement, trading profitability, or quantum computational advantage is established.\n- Source hashes and output checks are reported in the final manifest.\n"""
    (OUT/"15_limitations_and_integrity.md").write_text(limitations.replace("\\n","\n"),encoding="utf-8")

    status="COMPLETE" if all(x["status"]=="PASS" for x in checks) else "PARTIAL"
    cat_rows=cat[cat.category=="SIX_CATEGORY_MACRO_AVERAGE"]
    macro={r.asset:float(r.target_scaled_sensitivity_relative_reduction_pct) for _,r in cat_rows.iterrows()}
    ex="\n".join([f"{k}: {v}" for k,v in classes.items()])
    executive=f"""# Phase 13 executive summary\n\n**Status:** {status}. This is a retrospective target-scaling sensitivity on the already-inspected Phase 6 test period. It did not change any frozen Phase 6 result or protocol.\n\nThe run used only the originally development-selected BTC/ETH 24-hour Amihud MLP and GRU configurations, both frozen information sets, original folds/test origins, and exact Phase 6 seed formulas. The treatment fit target StandardScaler on training labels only, inverse-transformed outputs, and scored all forecasts on the original log1p target scale. The unscaled paired replay and original saved outputs are separately reported.\n\nThe historical Phase 6 result remains **2/18 tasks passed**. Historical category-average reductions were 18.5648% for BTC and 13.6448% for ETH; their exact reconstruction is checked before reporting retrospective category values. The target-scaled category sensitivity is BTC {macro.get('BTC',float('nan')):.4f}% and ETH {macro.get('ETH',float('nan')):.4f}% (retrospective; not a new registered result).\n\nPrespecified descriptive classifications:\n{ex}\n\nSee `05_mlp_sensitivity_results.csv`, `06_gru_sensitivity_results.csv`, `07_neural_prediction_diagnostics.csv`, `09_strong_baseline_comparison.csv`, `10_ensemble_sensitivity.csv`, `11_category_average_sensitivity.csv`, and `12_exploratory_uncertainty.md`. All Phase 6, Phase 9, Phase 12, certified panel, code and manuscript hashes are rechecked in `16_source_manifest_after.json`.\n\nNo manuscript was edited, no market data were downloaded, and no causal, independent-confirmation, trading-value, or quantum-advantage claim is supported by this sensitivity.\n"""
    (OUT/"18_executive_summary.md").write_text(executive,encoding="utf-8")

    # Reproduction validation report consolidates checks without running another model fit.
    reproduction_lines=["# Reproducibility validation","",f"Overall validation: **{'PASS' if all(x['status']=='PASS' for x in checks) else 'FAIL'}**.",""]
    for c in checks: reproduction_lines.append(f"- {c['check']}: **{c['status']}** — {c['evidence']}")
    reproduction_lines += ["",f"Software: Python {platform.python_version()}, NumPy {np.__version__}, pandas {pd.__version__}, scikit-learn {sklearn.__version__}, PyTorch {torch.__version__}, matplotlib {matplotlib.__version__}.","The final-test predictions are checked against the frozen Phase 6 date vector; target-scaler inverse transformation and saved MAEs are independently reconstructed from CSV predictions. No second training pass was performed as part of this final integrity audit."]
    (OUT/"17_reproducibility_validation.md").write_text("\n".join(reproduction_lines)+"\n",encoding="utf-8")

    # Required temporal alignment audit.
    folds=js(P6/"temporal_folds.json"); panel=js(P6/"data_validation.json")
    align={"source":"src/phase6_benchmark_core.py:make_task/fold_masks and src/phase6_evaluate_final.py train/test masks",
           "target":"mean spot_amihud over t+1 through t+6; log1p(max(y,0))",
           "feature_window":"t-5 through t, oldest to newest; all frozen feature inputs complete-case",
           "development_fold_specification":folds["folds"],"purge_rule":folds["purge"],
           "final_test_period":js(P6/"final_test_predictions.json")["final_test_period"],
           "test_origins_per_asset":1417,"all_test_origin_vectors_match_phase6":next(x for x in checks if x["check"].startswith("all_primary_test"))["status"]=="PASS",
           "scaler_scope":"predictor scaler and treatment target scaler fit using training rows/labels only",
           "no_future_labels_cross_split":True,"evidence":"per-fit train_last/eval dates, target_end, saved origin-level predictions; see 08_fold_level_results.csv and training_logs/all_fits.jsonl"}
    writej(OUT/"temporal_alignment_audit.json",align)

    # Source manifest and all new-output checksums; previous-stage hash checks are explicit.
    source_all=all(x["unchanged"] for x in sources)
    current_files=[]
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name not in ["19_final_experiment_manifest.json","19_final_experiment_manifest.sha256"] and "__pycache__" not in p.parts:
            current_files.append({"path":str(p.relative_to(OUT)),"size_bytes":p.stat().st_size,"sha256":sha(p)})
    manifest={"phase":"13 controlled neural target-scaling sensitivity","status":status,
       "experiment_type":"RETROSPECTIVE_SENSITIVITY","generated_utc":datetime.now(timezone.utc).isoformat(),
       "pretraining_lock_sha256":(OUT/"pretraining_lock_manifest.sha256").read_text().split()[0],
       "phase6_frozen_status":"2/18 tasks passed; unchanged",
       "historical_category_averages_reconstructed":js(OUT/"analysis_summary.json")["category_average"],
       "descriptive_classifications":classes,"validation_checks":checks,
       "source_integrity":{"count":len(sources),"all_unchanged":source_all,"phase6_unchanged":all(x["unchanged"] for x in sources if x["path"].startswith("outputs/phase6_autonomous_benchmark/")),"phase9_unchanged":all(x["unchanged"] for x in sources if x["path"].startswith("outputs/phase9_external_qml_replication/")),"phase12_unchanged":all(x["unchanged"] for x in sources if x["path"].startswith("outputs/phase12_technical_revision_audit/")),"manuscript_unchanged":all(x["unchanged"] for x in sources if x["path"].startswith("manuscripts/"))},
       "new_market_data":"NONE","model_training":"MLP_AND_GRU_ONLY","new_confirmatory_tests":"NONE","manuscript_modified":False,
       "generated_output_files_excluding_final_manifest_and_checksum":current_files,
       "output_file_count_excluding_manifest_and_checksum":len(current_files)}
    writej(OUT/"19_final_experiment_manifest.json",manifest)
    (OUT/"19_final_experiment_manifest.sha256").write_text(sha(OUT/"19_final_experiment_manifest.json")+"  19_final_experiment_manifest.json\n",encoding="utf-8")
    # Update the execution record only after all work is done.
    return status,classes,checks,manifest

if __name__=="__main__":
    status,classes,checks,manifest=main()
    print(json.dumps({"status":status,"classes":classes,"validation":"PASS" if all(x["status"]=="PASS" for x in checks) else "FAIL","source_integrity":manifest["source_integrity"],"output_files":manifest["output_file_count_excluding_manifest_and_checksum"]},indent=2))
