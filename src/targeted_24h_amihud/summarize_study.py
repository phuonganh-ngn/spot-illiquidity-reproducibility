from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from common import OUT

def readj(name): return json.loads((OUT/name).read_text())
def dump(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
def ci_positive(b): return bool(b and b['ci95'][0]>0)
def pct(x): return f'{100*x:.2f}%'
def holm(p):
    p=np.asarray(p,float); out=np.empty(len(p)); running=0.
    for rank,j in enumerate(np.argsort(p)):
        running=max(running,min(1.,(len(p)-rank)*p[j])); out[j]=running
    return out.tolist()

def main():
    direct=readj('direct_incremental_results.json')['results']
    best=readj('best_model_dominance.json')['development_nested_best_model']
    resid=readj('residual_learning_results.json')['summary']
    ab=readj('component_ablation.json')['summary']
    drop=readj('drop_one_ablation.json')['summary']
    joint_details=readj('joint_state_results.json')
    neg=readj('negative_controls.json')['results']
    qkernel=readj('quantum_kernel_results.json')['results']
    qnn=readj('qnn_results.json')['results']
    hybrid=readj('hybrid_qnn_results.json')['results']
    dev=readj('classical_development_results.json')
    phase6_path=OUT.parent/'phase6_autonomous_benchmark'/'phase6_classification.json'
    phase7_path=OUT.parent/'phase7_liquidity_replication'/'phase7_replication_decision.json'
    phase6=json.loads(phase6_path.read_text()) if phase6_path.exists() else {}
    phase7=json.loads(phase7_path.read_text()) if phase7_path.exists() else {}

    qml_global=[]
    for asset in ['BTC','ETH']:
        candidates=[]
        for family,rows in [('quantum_kernel',qkernel),('qnn',qnn),('hybrid_qnn',hybrid)]:
            for r in rows:
                if r['asset']==asset:
                    candidates.append({'asset':asset,'family':family,'qubits':r['qubits'],
                      'raw_p':r['paired_bootstrap']['42']['p_two_sided']})
        adjusted=holm([r['raw_p'] for r in candidates])
        qml_global.extend([r|{'holm_p_across_all_nine_qml_comparisons':p} for r,p in zip(candidates,adjusted)])
    dump('quantum_global_holm_sensitivity.json',{'definition':'Sensitivity correction across all three QML families and all three qubit counts within each asset. The frozen primary QML correction remains Holm within each family across qubit counts.','results':qml_global})

    assets={}
    for asset in ['BTC','ETH']:
        ds=[r for r in direct if r['asset']==asset]
        med=float(np.median([r['delta_s0_minus_s1_positive_favors_derivatives'] for r in ds]))
        improve=sum(r['delta_s0_minus_s1_positive_favors_derivatives']>0 for r in ds)
        sig=[r for r in ds if r.get('holm_p_within_asset_9_families',1)<.05 and r['delta_s0_minus_s1_positive_favors_derivatives']>0]
        groups={
          'regularized_linear':['ElasticNet'],
          'kernel':['RBF_SVR'],
          'bagged_trees':['ExtraTrees'],
          'boosting':['HistGradientBoosting','XGBoost','LightGBM','CatBoost'],
          'feedforward_neural':['MLP'],
          'recurrent_neural':['GRU'],
        }
        group_delta={}
        for name,fams in groups.items():
            rr=[r['delta_s0_minus_s1_positive_favors_derivatives'] for r in ds if r['family'] in fams]
            group_delta[name]=float(np.median(rr)) if rr else None
        positive_groups=sum(v is not None and v>0 for v in group_delta.values())
        nonneural_sig=any(r['family'] not in ['MLP','GRU'] for r in sig)
        sig_groups={name for name,fams in groups.items() if any(r['family'] in fams for r in sig)}
        b=[r for r in best if r['asset']==asset][0]
        b_delta=b['delta_positive_favors_derivatives']; b_mae=b['spot_only_transformed_mae']
        b_ci=b['paired_block_bootstrap_42']['ci95']
        competitive=(b_delta>=0) or (b_ci[0]<=0<=b_ci[1])
        rr=[r for r in resid if r['asset']==asset]
        r_sig=[r for r in rr if r.get('holm_p_within_asset_residual_families',1)<.05 and ci_positive(r['paired_block_bootstrap']['42'])]
        r_any=any(r['delta_positive_favors_derivatives']>0 for r in rr)
        residual_group={'ElasticNet':'linear','RBF_SVR':'kernel','HistGradientBoosting':'boosting','XGBoost':'boosting','MLP':'neural'}
        residual_sig_groups={residual_group[r['family']] for r in r_sig}
        r_multi=len(residual_sig_groups)>=2
        ncc=[r for r in neg if r['asset']==asset]
        nc_warn=[r for r in ncc if ci_positive(r['bootstrap']['42'])]
        ab_asset={r['specification']:r['transformed_mae'] for r in ab if r['asset']==asset}
        drop_asset={r['specification']:r['transformed_mae'] for r in drop if r['asset']==asset}
        full=ab_asset.get('full')
        single_names=['funding_only','oi_only','flow_only','basis_only']
        drop_names=['drop_funding','drop_oi','drop_flow','drop_basis']
        joint=(full is not None and all(ab_asset.get(x,float('inf'))>full for x in single_names) and sum(drop_asset.get(x,float('inf'))>full for x in drop_names)>=3)
        season=ab_asset.get('no_calendar')
        seasonality_needed=(full is not None and season is not None and season>full)
        qrows=[('quantum_kernel',r) for r in qkernel if r['asset']==asset]
        qrows += [('qnn',r) for r in qnn if r['asset']==asset]
        qrows += [('hybrid_qnn',r) for r in hybrid if r['asset']==asset]
        q_pass=[]
        for fam,r in qrows:
            boot=r['paired_bootstrap']
            delta=(r['delta_positive_favors_quantum'] if fam=='quantum_kernel' else r['mean_seed_loss_delta_positive_favors_quantum'])
            seeds_ok=(fam=='quantum_kernel' or (r.get('seed_fraction_positive',0)>=.8 and r.get('seed_delta_median',-1)>0))
            matched_mlp=r.get('matched_mlp_metadata',[])
            matched_classical_valid=(fam=='quantum_kernel' or (len(matched_mlp)==5 and all(m.get('converged',False) for m in matched_mlp)))
            if delta>0 and r.get('holm_p_within_asset_family',1)<.05 and ci_positive(boot['42']) and seeds_ok and matched_classical_valid:
                q_pass.append({'family':fam,'qubits':r['qubits'],'delta':delta,'holm_p':r['holm_p_within_asset_family'],'global_holm_p':next(x['holm_p_across_all_nine_qml_comparisons'] for x in qml_global if x['asset']==asset and x['family']==fam and x['qubits']==r['qubits']),'seed_stability':r.get('seed_fraction_positive'),'matched_classical_converged':matched_classical_valid})
        q_value=bool(q_pass)
        # Conservative classification hierarchy; classify model-sensitive apparent gains explicitly.
        robust=(med>0 and positive_groups>=3 and len(sig_groups)>=2 and nonneural_sig and r_multi and competitive and not nc_warn)
        nonlinear=(group_delta['regularized_linear']<=0 and group_delta['kernel']>0 and group_delta['boosting']>0 and (group_delta['feedforward_neural']>0 or group_delta['recurrent_neural']>0))
        if q_value: primary='QUANTUM_INCREMENTAL_VALUE'
        elif robust: primary='ROBUST_INCREMENTAL_DERIVATIVES_INFORMATION'
        elif joint: primary='JOINT_DERIVATIVES_STATE_INFORMATION'
        elif nonlinear: primary='NONLINEAR_INCREMENTAL_INFORMATION'
        elif r_multi: primary='RESIDUAL_INCREMENTAL_INFORMATION'
        elif improve>0 or r_any: primary='MODEL_SENSITIVE_SIGNAL'
        else: primary='NO_INCREMENTAL_INFORMATION'
        assets[asset]={
          'primary_classification':primary,
          'secondary_labels':[x for x,v in [('ROBUST_INCREMENTAL_DERIVATIVES_INFORMATION',robust),('NONLINEAR_INCREMENTAL_INFORMATION',nonlinear),('RESIDUAL_INCREMENTAL_INFORMATION',r_multi),('JOINT_DERIVATIVES_STATE_INFORMATION',joint),('QUANTUM_INCREMENTAL_VALUE',q_value),('MODEL_SENSITIVE_SIGNAL',(improve>0 and not robust))] if v],
          'direct_information_set':{'median_family_delta_s0_minus_s1':med,'families_improving':improve,'families_total':len(ds),'holm_significant_improvements':len(sig),'independent_holm_supported_groups':sorted(sig_groups),'holm_survivors':[{'family':r['family'],'delta':r['delta_s0_minus_s1_positive_favors_derivatives'],'holm_p':r['holm_p_within_asset_9_families']} for r in sig],'group_median_deltas':group_delta,'positive_broad_groups':positive_groups},
          'best_model_dominance':{'spot_only_mae':b_mae,'spot_plus_derivatives_mae':b['spot_plus_derivatives_transformed_mae'],'delta':b_delta,'relative_change_pct':100*(b['spot_plus_derivatives_transformed_mae']-b_mae)/max(b_mae,1e-15),'paired_bootstrap_42':b['paired_block_bootstrap_42'],'competitive_under_frozen_operational_rule':competitive,'families_by_outer_fold':{'spot_only':b['spot_only_families_by_fold'],'spot_plus_derivatives':b['spot_plus_families_by_fold']}},
          'residual_learning':{'models':rr,'holm_supported_families':[r['family'] for r in r_sig],'independent_holm_supported_groups':sorted(residual_sig_groups),'residual_incremental_information_supported':r_multi},
          'component_state':{'full_mae':full,'single_component_mae':{k:v for k,v in ab_asset.items() if k.endswith('_only')},'drop_one_mae':drop_asset,'joint_state_rule_passed':joint,'paired_full_vs_component_comparisons':[r for r in joint_details.get('results',[]) if r['asset']==asset]},
          'seasonality':{'full_mae':full,'without_calendar_mae':season,'calendar_controls_help_under_fixed_model':seasonality_needed},
          'quantum':{'supported_comparisons':q_pass,'quantum_incremental_value_supported_for_asset':q_value,'results_by_family':{'kernel':[r for r in qkernel if r['asset']==asset],'qnn':[r for r in qnn if r['asset']==asset],'hybrid_qnn':[r for r in hybrid if r['asset']==asset]}},
          'negative_control':{'folds':ncc,'positive_future_derivative_placebo_with_bootstrap_support':bool(nc_warn),'temporal_warning':bool(nc_warn)}
        }
    btc=assets['BTC']; eth=assets['ETH']
    qrep=btc['quantum']['quantum_incremental_value_supported_for_asset'] and eth['quantum']['quantum_incremental_value_supported_for_asset']
    if qrep: paper='QUANTUM_ENHANCED_24H_AMIHUD_INFORMATION'
    elif btc['primary_classification']=='ROBUST_INCREMENTAL_DERIVATIVES_INFORMATION' and eth['primary_classification']=='ROBUST_INCREMENTAL_DERIVATIVES_INFORMATION': paper='ROBUST_24H_AMIHUD_INCREMENTAL_INFORMATION'
    elif btc['primary_classification']=='JOINT_DERIVATIVES_STATE_INFORMATION' and eth['primary_classification']=='JOINT_DERIVATIVES_STATE_INFORMATION': paper='JOINT_DERIVATIVES_STATE_INFORMATION'
    elif btc['primary_classification']=='NONLINEAR_INCREMENTAL_INFORMATION' and eth['primary_classification']=='NONLINEAR_INCREMENTAL_INFORMATION': paper='NONLINEAR_24H_AMIHUD_INFORMATION'
    elif any(a['primary_classification'] in ['MODEL_SENSITIVE_SIGNAL','RESIDUAL_INCREMENTAL_INFORMATION','QUANTUM_INCREMENTAL_VALUE'] for a in assets.values()): paper='MODEL_SENSITIVE_24H_AMIHUD_SIGNAL'
    elif all(a['primary_classification']=='NO_INCREMENTAL_INFORMATION' for a in assets.values()): paper='NO_ROBUST_24H_AMIHUD_INFORMATION'
    else: paper='INCONCLUSIVE'
    dump('btc_classification.json',assets['BTC']); dump('eth_classification.json',assets['ETH'])
    dump('paper_level_classification.json',{'classification':paper,'btc':assets['BTC']['primary_classification'],'eth':assets['ETH']['primary_classification'],'quantum_replicated_in_both_assets':qrep,'rule':'Both assets are required for paper-level robust, nonlinear, joint-state or replicated quantum labels; otherwise an apparent asset/model-dependent signal is classified MODEL_SENSITIVE.'})
    dump('joint_state_classification.json',{'rule':'Joint state requires full S1 OOF MAE below every single-component variant and at least three of four drop-one variants, using the same inner-CV-selected model/config per fold with no retuning.','assets':{a:assets[a]['component_state'] for a in assets}})

    p6=phase6.get('classification','not found'); p7=phase7.get('classification','not found')
    bdom=dev['best_model_historical_reuse'] if 'best_model_historical_reuse' in dev else []
    ext={a:[r for r in bdom if r['asset']==a] for a in ['BTC','ETH']}
    btcd=[r for r in readj('direct_incremental_results.json')['results'] if r['asset']=='BTC']
    ethd=[r for r in readj('direct_incremental_results.json')['results'] if r['asset']=='ETH']
    res_sig={a:[r['family'] for r in resid if r['asset']==a and r.get('holm_p_within_asset_residual_families',1)<.05 and ci_positive(r['paired_block_bootstrap']['42'])] for a in ['BTC','ETH']}
    mcs_rows=dev.get('mcs',[])
    s1_mcs=[(r['asset'],r['outer_fold'],name) for r in mcs_rows for name in r['results'].get('combined',{}).get('included',[]) if name.startswith('SPOT_PLUS_DERIVATIVES__')]
    regime=readj('regime_exploration.json')['results']
    global_q_valid=[{'asset':a,'family':q['family'],'qubits':q['qubits'],'global_holm_p':q['global_holm_p']} for a in ['BTC','ETH'] for q in assets[a]['quantum']['supported_comparisons'] if q['global_holm_p']<.05]
    mlp_warnings=sum(bool(r.get('fit_warnings')) for r in dev['group_results'] if r['family']=='MLP')
    summary=f'''# Targeted 24-hour Amihud study

This is a predictive-information study, not a causal-identification design. It uses certified Binance 4-hour BTC/ETH data only and does not download new market data. The sole target is the arithmetic mean of 4-hour spot Amihud over t+1 through t+6, with `Amihud = 1e6 × |4-hour log return| / 4-hour spot quote volume in USDT`. The frozen T2 `log1p(Amihud)` transform is numerically close to raw Amihud at the observed scale. Development evaluation uses nested expanding folds from 2022-10-01 through 2023-06-30 20:00 UTC, with horizon purges. Phase-6/Phase-7 periods are reused exploratory history, not fresh confirmation.

1. **Does Spot+Derivatives improve BTC?** No in the matched direct comparison: {btc['direct_information_set']['families_improving']}/9 families improve; the median S0-minus-S1 MAE difference is {btc['direct_information_set']['median_family_delta_s0_minus_s1']:.5g}, favoring SpotOnly.
2. **Does it improve ETH?** No robustly: {eth['direct_information_set']['families_improving']}/9 families improve, none survives within-asset Holm, and the median S0-minus-S1 difference is {eth['direct_information_set']['median_family_delta_s0_minus_s1']:.5g}.
3. **Does the gain hold across classical families?** No. BTC has no positive family delta; ETH's two small positive deltas are not bootstrap/Holm supported. The other families tie or worsen.
4. **Does the result depend on neural models?** The direct model comparison supplies no neural gain: GRU worsens in BTC and is nearly tied in ETH; the unscaled-target development MLP performs poorly and has convergence warnings in {mlp_warnings} fitted groups. Residual corrections likewise have no Holm-supported classical family.
5. **Do kernel and boosting models confirm it?** No. RBF-SVR ties across the matched information sets. Boosting/tree comparisons mostly worsen with derivatives; ETH ExtraTrees is the only positive (S0-minus-S1) result in that broad tree/boosting group, and its paired bootstrap includes zero.
6. **Does the best S1 beat the best S0?** No. Inner-CV-selected BTC transformed MAE is {btc['best_model_dominance']['spot_only_mae']:.5g} for S0 versus {btc['best_model_dominance']['spot_plus_derivatives_mae']:.5g} for S1; ETH is {eth['best_model_dominance']['spot_only_mae']:.5g} versus {eth['best_model_dominance']['spot_plus_derivatives_mae']:.5g}. S1 is worse by {btc['best_model_dominance']['relative_change_pct']:.1f}% in BTC and {eth['best_model_dominance']['relative_change_pct']:.1f}% in ETH on this metric.
7. **Do derivatives predict strict SpotOnly OOF residuals?** No supported evidence: BTC supported families {res_sig['BTC'] or 'none'}, ETH {res_sig['ETH'] or 'none'}. BTC has small positive point deltas in individual learners, but their paired intervals include zero; ETH residual errors worsen across all five learners.
8. **Are residual relationships linear or nonlinear?** The classical residual learner set does not establish either: no independent family survives both within-asset Holm and positive block-bootstrap CI. The later QML residual result is method/configuration specific.
9. **Does the full joint derivatives state beat individual components?** No. In BTC the full S1 configuration loses to all four single-component fits; ETH comparisons are mixed. The frozen joint-state rule fails in both assets.
10. **Which transformations contribute?** The frozen FundingChange, 30-day OIShock/FlowShock, BasisChange and four interactions are evaluated as part of the registered feature set, but no individual-transform ablation was specified. The study therefore cannot attribute value to one transformation. Full-versus-component and drop-one results are in `joint_state_results.json`.
11. **Are seasonality controls important?** Removing the frozen calendar controls worsens the fixed S1 specification in both assets, much more in BTC than ETH. This is an ablation result, not evidence that derivatives improve prediction.
12. **Does QML beat matched classical nonlinear models?** For selected residual corrections, yes under the frozen family-wise Holm rule: BTC has supported QNN/hybrid comparisons at 4 qubits; ETH has supported QNN at 6 and hybrid at 8 qubits. Quantum-kernel comparisons do not show a general advantage; BTC's 4-qubit kernel is significantly worse than matched RBF under the frozen correction.
13. **Is QML stable across seeds?** Selected comparisons have positive median paired deltas and 4/5 seeds favor QML. This is stability within one F3 holdout, not repeated-sample stability. The matched MLP solver ceiling was amended after convergence warnings and before viewing QML/classical loss comparisons; only comparisons with all five MLP fits converged count.
14. **Is QML replicated in both assets?** The frozen per-family criteria identify at least one qualifying QNN/hybrid comparison in each asset, but the supported qubit dimensions differ and both assets share the same holdout dates. This is not independent external replication.
15. **Do multiplicity corrections retain the advantage?** Selected QNN/hybrid tests survive the frozen Holm correction within each QML family. Under the stricter all-nine-comparison sensitivity, BTC hybrid-4q and ETH QNN-6q remain below 0.05; the BTC QNN-4q within-family result does not. Exact values are in `quantum_global_holm_sensitivity.json`. The classical MCS completes in all six asset/fold combined analyses after exact duplicate forecast series are collapsed.
16. **Do negative controls remain null?** No future-derivative predictor has a positive block-bootstrap-supported gain for already-realized Amihud. Most placebo deltas are negative; the rest include zero. This diagnostic does not prove the forward design is unconfounded.
17. **Is the signal regime concentrated?** The frozen descriptive regime splits show no consistent derivative improvement. BTC S0-minus-S1 deltas are negative in every reported regime; ETH regime deltas are small and mixed. Regime counts and means are exploratory, without subgroup significance claims.
18. **How do Phases 6 and 7 fit?** Phase 6 reported target-specific predictive value (`{p6}`); Phase 7 failed its frozen broad replication rule (`{p7}`). This study again finds no classical S1 dominance or residual-learning support, while selected QNN/hybrid corrections perform better on one development holdout. The combined record is a narrow, model-sensitive 24-hour Amihud signal, not generalizable derivatives information.
19. **What can be claimed?** Under this frozen historical development evaluation, specific shallow QNN/hybrid residual corrections contain incremental predictive information relative to their matched MLPs for selected asset/configuration pairs. The broader Spot+Derivatives information set does not beat the strongest SpotOnly model.
20. **What cannot be claimed?** No causal effect, universal derivatives predictability, generalization to new venues/periods or liquidity measures, or quantum advantage is established. The Binance 2024–2026 data were already inspected, and the external replication failure remains in force.

BTC classification: **{btc['primary_classification']}**. ETH classification: **{eth['primary_classification']}**. Paper-level classification under the frozen rules: **{paper}**. The most defensible next step is to reserve a genuinely unseen venue or future period for a preregistered replication; no further modeling is run in this study.
'''
    (OUT/'research_summary.md').write_text(summary)
    print(json.dumps({'paper_level_classification':paper,'btc':btc['primary_classification'],'eth':eth['primary_classification']}))

if __name__=='__main__': main()
