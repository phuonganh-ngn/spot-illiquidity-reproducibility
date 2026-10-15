#!/usr/bin/env python3
"""One-shot final test evaluation for the frozen Phase-6 benchmark.

The --reproduce mode is an offline deterministic reproduction only; it cannot
change model selection or the frozen protocol and writes a comparison audit.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from phase6_benchmark_core import *

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stable_seed(*parts): return SEED + sum(sum(ord(c) for c in str(p)) for p in parts)

def holm(pvals):
    p=np.asarray(pvals,float); m=len(p); order=np.argsort(p); adj=np.empty(m); running=0.
    for rank,idx in enumerate(order):
        val=min(1.,(m-rank)*p[idx]); running=max(running,val); adj[idx]=running
    return adj.tolist()

def centered_bootstrap_p(draws, observed):
    """Two-sided paired-bootstrap p-value using a null-centered distribution.

    The percentile draws are retained unchanged for confidence intervals and
    sign probability.  Centering at the observed statistic imposes the null
    delta=0 before measuring tail mass, avoiding the incorrect practice of
    treating the fraction of uncentered draws on either side of zero as a
    p-value.
    """
    draws=np.asarray(draws,float)
    centered=draws-float(observed)
    p=(1+np.count_nonzero(np.abs(centered)>=abs(float(observed))))/(len(draws)+1)
    return min(1.,float(p))

def moving_block_indices(n,reps,block,seed):
    rng=np.random.default_rng(seed); blocks=int(np.ceil(n/block)); out=[]
    max_start=n-block
    if max_start<0: raise ValueError('test series shorter than bootstrap block')
    for _ in range(reps):
        starts=rng.integers(0,max_start+1,size=blocks)
        idx=np.concatenate([np.arange(s,s+block) for s in starts])[:n]
        out.append(idx)
    return out

def all_families(): return ['OLS','Ridge','RBF_SVR','ExtraTrees','HistGradientBoosting','MLP','GRU','equal_weight_ensemble']

def model_categories():
    return {'linear':['OLS','Ridge'],'kernel':['RBF_SVR'],'bagged_trees':['ExtraTrees'],
            'boosting':['HistGradientBoosting'],'neural':['MLP','GRU'],
            'equal_weight_ensemble':['equal_weight_ensemble']}

def category_delta(y, pred_spot, pred_plus, idx, members):
    vals=[]
    for m in members:
        a=float(np.mean(np.abs(y[idx]-pred_spot[m][idx])))
        b=float(np.mean(np.abs(y[idx]-pred_plus[m][idx])))
        vals.append((a-b)/max(a,1e-15))
    return float(np.mean(vals))

def fit_predictions(task,asset,target,horizon,info_name,model_params,train_idx,test_idx):
    Xflat,Xseq=x_for_set(task['Xseq'],INFOSETS[info_name]); ylog=logtarget(task['y'])
    pred={}; fitted_params={}
    families=['OLS']+list(MODEL_CONFIGS.keys())
    for family in families:
        if family=='OLS': cfg={}
        else:
            key=f'{asset}__{target}__h{horizon}__{info_name}'
            cfg=model_params['selected'][key]['models'][family]['selected_config']
        p,_=fit_predict(family,cfg,Xflat[train_idx],Xseq[train_idx],ylog[train_idx],
                        Xflat[test_idx],Xseq[test_idx],stable_seed('final',asset,target,horizon,info_name,family))
        pred[family]=p; fitted_params[family]=cfg
    base_mean=np.full(int(test_idx.sum()),float(ylog[train_idx].mean()))
    base_persist=logtarget(task['persistence'][test_idx])
    pred['training_mean']=base_mean; pred['persistence']=base_persist
    pred['equal_weight_ensemble']=np.mean(np.column_stack([pred[x] for x in ['Ridge','RBF_SVR','ExtraTrees','HistGradientBoosting','MLP','GRU']]),axis=1)
    return pred,fitted_params

def evaluate(reproduce=False):
    protocol_path=OUT/'benchmark_protocol.json'; selected_path=OUT/'selected_models.json'
    protocol_hash=(OUT/'benchmark_protocol.sha256').read_text().split()[0]
    if sha(protocol_path)!=protocol_hash: raise RuntimeError('Protocol hash changed before final test')
    if not selected_path.exists(): raise RuntimeError('Development selection missing; test cannot be opened')
    selected_hash_file=(OUT/'selected_models.sha256').read_text().split()[0]
    if sha(selected_path)!=selected_hash_file: raise RuntimeError('Selected-model freeze hash mismatch')
    params=json.loads(selected_path.read_text())
    if params.get('status')!='FROZEN_AFTER_DEVELOPMENT_CV' or params.get('protocol_sha256')!=protocol_hash: raise RuntimeError('Selected model registry not frozen under active protocol')
    df=load_panel_full_after_freeze()  # first numerical parse of final-test rows
    if len(df)!=9888 or set(df.asset.unique())!={'BTC','ETH'}: raise RuntimeError('Frozen panel count/universe changed')
    if df.duplicated(['asset','period_start_utc']).any(): raise RuntimeError('Duplicate date-asset key')
    pred_output={'protocol_sha256':protocol_hash,'selected_models_sha256':selected_hash_file,
      'prediction_scale':'log1p target','final_test_period':['2023-07-01T00:00:00Z','2024-03-03T20:00:00Z'],'tasks':[]}
    metric_output={'protocol_sha256':protocol_hash,'metric_primary':'MAE on log1p target; lower is better','tasks':[]}
    ablation_output={'protocol_sha256':protocol_hash,'status':'FIXED_HYPERPARAMETER_SECONDARY_ABLATION','retuned':False,'tasks':[]}
    uncertainty_output={'protocol_sha256':protocol_hash,'block_length_intervals':42,'replications':2000,'seed':SEED,'tasks':[]}
    incremental_output={'protocol_sha256':protocol_hash,'delta_definition':'MAE_log(Spot Only) - MAE_log(Spot + All Derivatives); positive favors derivatives','tasks':[]}
    task_store={}
    for asset in ['BTC','ETH']:
      for target in TARGETS:
       for horizon in HORIZONS:
        task=make_task(df,asset,horizon,target)
        dates=task['date']; ends=task['target_end']
        train=(dates<SPLIT)&(ends<SPLIT)
        test=(dates>=SPLIT)&(ends<SAMPLE_END_EXCLUSIVE)
        if not test.any() or train.sum()<1000: raise RuntimeError(f'Insufficient fixed train/test sample {asset}/{target}/h{horizon}')
        if (ends[train]>=SPLIT).any() or (ends[test]>=SAMPLE_END_EXCLUSIVE).any(): raise RuntimeError('Future target window leaked across sample boundary')
        y=task['y']; ylog=logtarget(y)
        test_dates=dates[test]; test_end=ends[test]
        pred_sets={}; config_sets={}
        for info_name in ['SPOT_ONLY','SPOT_PLUS_ALL_DERIVATIVES']:
            pred_sets[info_name],config_sets[info_name]=fit_predictions(task,asset,target,horizon,info_name,params,train,test)
        # Secondary single-component add-on sets: use the selected full-information
        # configuration for the same family, with no retuning.
        Xflat_all,Xseq_all=x_for_set(task['Xseq'],INFOSETS['SPOT_PLUS_ALL_DERIVATIVES'])
        for info_name in ['SPOT_PLUS_FUNDING','SPOT_PLUS_OI','SPOT_PLUS_FUTURES_FLOW','SPOT_PLUS_BASIS']:
            names=INFOSETS[info_name]; xflat,xseq=x_for_set(task['Xseq'],names)
            pred={}; key=f'{asset}__{target}__h{horizon}__SPOT_PLUS_ALL_DERIVATIVES'
            for family in ['OLS']+list(MODEL_CONFIGS.keys()):
                cfg={} if family=='OLS' else params['selected'][key]['models'][family]['selected_config']
                pred[family],_=fit_predict(family,cfg,xflat[train],xseq[train],ylog[train],xflat[test],xseq[test],
                    stable_seed('ablation',asset,target,horizon,info_name,family))
            pred['training_mean']=np.full(int(test.sum()),float(ylog[train].mean()))
            pred['persistence']=logtarget(task['persistence'][test])
            pred['equal_weight_ensemble']=np.mean(np.column_stack([pred[x] for x in ['Ridge','RBF_SVR','ExtraTrees','HistGradientBoosting','MLP','GRU']]),axis=1)
            pred_sets[info_name]=pred
        task_id=f'{asset}__{target}__h{horizon}'
        testrows={'asset':asset,'target':target,'horizon_intervals':horizon,'horizon_hours':4*horizon,
          'n_train':int(train.sum()),'n_test':int(test.sum()),'train_first':dates[train].min().isoformat(),'train_last':dates[train].max().isoformat(),
          'test_first':test_dates.min().isoformat(),'test_last':test_dates.max().isoformat(),
          'target_end_last':test_end.max().isoformat(),'target_raw':y[test].tolist(),'target_log1p':ylog[test].tolist(),
          'date':[x.isoformat() for x in test_dates],'target_end':[x.isoformat() for x in test_end],
          'predictions_log1p':{k:{m:np.asarray(v).tolist() for m,v in pp.items()} for k,pp in pred_sets.items()}}
        pred_output['tasks'].append(testrows)
        metrics={}
        for info_name,pp in pred_sets.items():
            metrics[info_name]={}
            for family,p in pp.items(): metrics[info_name][family]=metric_pack(ylog[test],p,y[test],target)
        metric_output['tasks'].append({**{k:v for k,v in testrows.items() if k not in ['target_raw','target_log1p','date','target_end','predictions_log1p']},'metrics':metrics})
        task_store[task_id]={'task':task,'train':train,'test':test,'y':y[test],'ylog':ylog[test],
                            'dates':test_dates,'pred':pred_sets,'metrics':metrics,'testrows':testrows}
        # Paired moving-block uncertainty, with shared resampled indices across all models and sets.
        n=int(test.sum()); boot_indices=moving_block_indices(n,2000,42,stable_seed('block',task_id))
        ytest=ylog[test]
        model_deltas={}; model_boot={}
        base_pred=pred_sets['SPOT_ONLY']; full_pred=pred_sets['SPOT_PLUS_ALL_DERIVATIVES']
        for fam in all_families():
            a=base_pred[fam]; b=full_pred[fam]
            obs=float(np.mean(np.abs(ytest-a))-np.mean(np.abs(ytest-b)))
            draws=np.array([np.mean(np.abs(ytest[ix]-a[ix]))-np.mean(np.abs(ytest[ix]-b[ix])) for ix in boot_indices])
            model_deltas[fam]={'delta_mae_log1p':obs,'relative_reduction_pct':100*obs/max(float(np.mean(np.abs(ytest-a))),1e-15),
              'ci95':[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))],
              'bootstrap_sign_probability_favors_derivatives':float(np.mean(draws>0)),
              'p_two_sided':centered_bootstrap_p(draws,obs)}
            model_boot[fam]=draws
        categories={}; cat_draws=[]
        for cat,members in model_categories().items():
            draws=np.mean(np.column_stack([model_boot[m] for m in members]),axis=1)
            obs=float(np.mean([model_deltas[m]['delta_mae_log1p'] for m in members]))
            base_maes=[metrics['SPOT_ONLY'][m]['mae_log1p'] for m in members]
            rel=100*obs/max(float(np.mean(base_maes)),1e-15)
            categories[cat]={'delta_mae_log1p':obs,'relative_reduction_pct':rel,
                'ci95':[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))],
                'p_two_sided':centered_bootstrap_p(draws,obs)}
            cat_draws.append(draws)
        adj=holm([categories[c]['p_two_sided'] for c in categories])
        for (cat,row),p_adj in zip(categories.items(),adj): row['holm_p']=float(p_adj)
        mean_rel_draw=np.mean(np.column_stack([
          np.array([100*d/max(float(np.mean([metrics['SPOT_ONLY'][m]['mae_log1p'] for m in model_categories()[c]])),1e-15) for d in draws])
          for c,draws in zip(categories.keys(),cat_draws)]),axis=1)
        avg_rel=float(np.mean([v['relative_reduction_pct'] for v in categories.values()]))
        average_ci=[float(np.quantile(mean_rel_draw,.025)),float(np.quantile(mean_rel_draw,.975))]
        poscats=sum(v['relative_reduction_pct']>0 for v in categories.values())
        sigpos=sum(v['holm_p']<.05 and v['relative_reduction_pct']>0 for v in categories.values())
        signeg=sum(v['holm_p']<.05 and v['relative_reduction_pct']<0 for v in categories.values())
        task_robust=bool(poscats>=4 and sigpos>=2 and avg_rel>=1.0 and signeg<2)
        uncertainty_output['tasks'].append({'task_id':task_id,'n_test':n,'model_families':model_deltas,'independent_categories':categories,
          'mean_category_relative_reduction_pct':avg_rel,'mean_category_relative_reduction_ci95_pct':average_ci,
          'positive_category_count':poscats,'holm_supported_positive_category_count':sigpos,
          'holm_supported_deterioration_category_count':signeg,'task_meets_robust_rule':task_robust})
        incremental_output['tasks'].append({'task_id':task_id,'n_test':n,'model_family_deltas':model_deltas,
          'category_deltas':categories,'mean_category_relative_reduction_pct':avg_rel,
          'robust_incremental_value':task_robust})
        abl_metrics={}
        for ab in ['SPOT_PLUS_FUNDING','SPOT_PLUS_OI','SPOT_PLUS_FUTURES_FLOW','SPOT_PLUS_BASIS']:
            abl_metrics[ab]={}
            for family,p in pred_sets[ab].items(): abl_metrics[ab][family]=metric_pack(ylog[test],p,y[test],target)
        ablation_output['tasks'].append({'task_id':task_id,'n_test':n,'metrics':abl_metrics,
          'compare_to_spot_only':{ab:{fam:{'delta_mae_log1p':metrics['SPOT_ONLY'][fam]['mae_log1p']-abl_metrics[ab][fam]['mae_log1p'],
            'relative_reduction_pct':100*(metrics['SPOT_ONLY'][fam]['mae_log1p']-abl_metrics[ab][fam]['mae_log1p'])/max(metrics['SPOT_ONLY'][fam]['mae_log1p'],1e-15)} for fam in all_families()} for ab in abl_metrics}})
    # Aggregate summaries after all targets/assets have been evaluated once.
    family_agg={}
    for fam in all_families():
        vals=[]
        for item in incremental_output['tasks']:
            vals.append(item['model_family_deltas'][fam]['relative_reduction_pct'])
        family_agg[fam]={'mean_relative_reduction_pct':float(np.mean(vals)),
                         'positive_task_count':int(np.sum(np.asarray(vals)>0)),
                         'task_count':len(vals)}
    category_task_rows=[]
    robust_keys=set()
    for item in uncertainty_output['tasks']:
        if item['task_meets_robust_rule']: robust_keys.add(item['task_id'])
        category_task_rows.append(item)
    family_summary={'protocol_sha256':protocol_hash,'model_families':family_agg,
      'task_count':len(incremental_output['tasks']),'robust_task_ids':sorted(robust_keys),
      'nonlinear_categories':['kernel','bagged_trees','boosting','neural'],
      'category_results_by_task':{x['task_id']:x['independent_categories'] for x in category_task_rows}}
    # Cross-asset replication requires both separate assets to meet the same frozen task rule.
    cross=[]
    for target in TARGETS:
      for horizon in HORIZONS:
        b=f'BTC__{target}__h{horizon}'; e=f'ETH__{target}__h{horizon}'
        cross.append({'target':target,'horizon_intervals':horizon,'horizon_hours':4*horizon,
          'BTC_robust':b in robust_keys,'ETH_robust':e in robust_keys,
          'replicated':bool(b in robust_keys and e in robust_keys)})
    cross_summary={'protocol_sha256':protocol_hash,'assets_estimated_separately':True,'tasks':cross,
      'replicated_task_count':sum(x['replicated'] for x in cross)}
    # Predeclared fixed-hyperparameter component ablations are the interpretability audit.
    interpretation={'protocol_sha256':protocol_hash,'method':'single-component information-set ablations with full-set-selected configs; not causal feature importance',
      'interpretation_limit':'Ablation changes predictive information supplied to the model but does not identify causal contribution.',
      'component_summary':{}}
    for comp in ['SPOT_PLUS_FUNDING','SPOT_PLUS_OI','SPOT_PLUS_FUTURES_FLOW','SPOT_PLUS_BASIS']:
        changes=[]
        for row in ablation_output['tasks']:
            for fam in ['Ridge','RBF_SVR','ExtraTrees','HistGradientBoosting','MLP','GRU','equal_weight_ensemble']:
                changes.append(row['compare_to_spot_only'][comp][fam]['relative_reduction_pct'])
        interpretation['component_summary'][comp]={'mean_relative_reduction_pct':float(np.mean(changes)),
          'positive_comparisons':int(np.sum(np.asarray(changes)>0)),'comparisons':len(changes)}
    # One overall paper-level classification based on prespecified criteria.
    robust_by_asset={a:[k for k in robust_keys if k.startswith(a+'__')] for a in ['BTC','ETH']}
    replicated=[x for x in cross if x['replicated']]
    if len(replicated)>=2 and len({x['target'] for x in replicated})>=2 and len({x['horizon_intervals'] for x in replicated})>=2:
        classification='A_DERIVATIVES_PROVIDE_BROAD_INCREMENTAL_PREDICTIVE_VALUE'
    elif replicated:
        nonlinear_rep=False
        for x in replicated:
            for asset in ['BTC','ETH']:
                tid=f"{asset}__{x['target']}__h{x['horizon_intervals']}"
                cats=next(q['independent_categories'] for q in uncertainty_output['tasks'] if q['task_id']==tid)
                nl=sum(cats[c]['holm_p']<.05 and cats[c]['relative_reduction_pct']>0 for c in ['kernel','bagged_trees','boosting','neural'])
                lin=cats['linear']['holm_p']<.05 and cats['linear']['relative_reduction_pct']>0
                nonlinear_rep |= bool(nl>=2 and not lin)
        if nonlinear_rep: classification='B_DERIVATIVES_PROVIDE_PRIMARILY_NONLINEAR_PREDICTIVE_VALUE'
        else:
            targets={x['target'] for x in replicated}; horizons={x['horizon_intervals'] for x in replicated}
            if len(targets)==1: classification='C_DERIVATIVES_PROVIDE_TARGET_SPECIFIC_PREDICTIVE_VALUE'
            elif 6 not in horizons: classification='D_DERIVATIVES_PROVIDE_SHORT_HORIZON_PREDICTIVE_VALUE'
            else: classification='E_DERIVATIVES_PROVIDE_ASSET_SPECIFIC_PREDICTIVE_VALUE'
    elif robust_keys:
        classification='E_DERIVATIVES_PROVIDE_ASSET_SPECIFIC_PREDICTIVE_VALUE'
    else:
        supported=any(v['holm_p']<.05 for x in uncertainty_output['tasks'] for v in x['independent_categories'].values())
        if supported: classification='F_DERIVATIVES_VALUE_IS_MODEL_SENSITIVE'
        else:
            # A valid, sufficiently large test with no family passing the frozen
            # rule is reported as no robust value; intervals remain visible.
            classification='G_DERIVATIVES_DO_NOT_PROVIDE_ROBUST_OUT_OF_SAMPLE_INCREMENTAL_VALUE'
    phase_class={'classification':classification,'protocol_sha256':protocol_hash,'robust_task_ids':sorted(robust_keys),
      'cross_asset_replicated_task_ids':[f"{x['target']}__h{x['horizon_intervals']}" for x in replicated],
      'claim_scope':'Predictive information only; no causal interpretation.'}
    # Save final one-shot evaluation artifacts. All rows are ordered by asset/task/time.
    json_write(OUT/'final_test_predictions.json',pred_output)
    json_write(OUT/'final_test_metrics.json',metric_output)
    json_write(OUT/'incremental_value_results.json',incremental_output)
    json_write(OUT/'paired_uncertainty.json',uncertainty_output)
    json_write(OUT/'ablation_results.json',ablation_output)
    json_write(OUT/'model_family_summary.json',family_summary)
    json_write(OUT/'cross_asset_summary.json',cross_summary)
    json_write(OUT/'interpretability_summary.json',interpretation)
    json_write(OUT/'phase6_classification.json',phase_class)
    if reproduce:
        return {p.name:sha(p) for p in [OUT/'final_test_predictions.json',OUT/'final_test_metrics.json',OUT/'incremental_value_results.json',OUT/'paired_uncertainty.json',OUT/'ablation_results.json',OUT/'model_family_summary.json',OUT/'cross_asset_summary.json',OUT/'interpretability_summary.json',OUT/'phase6_classification.json']}
    return None

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--reproduce',action='store_true'); args=parser.parse_args()
    existing=OUT/'final_test_predictions.json'
    if args.reproduce:
        # Preserve the saved outputs; evaluate into a temporary result set is not
        # used because the normal writer targets canonical paths. Capture hashes,
        # compare, then rewrite the identical outputs from the frozen pipeline.
        before={p.name:sha(p) for p in [OUT/'final_test_predictions.json',OUT/'final_test_metrics.json',OUT/'incremental_value_results.json',OUT/'paired_uncertainty.json',OUT/'ablation_results.json',OUT/'model_family_summary.json',OUT/'cross_asset_summary.json',OUT/'interpretability_summary.json',OUT/'phase6_classification.json']}
        result=evaluate(reproduce=True)
        after=result
        passed=before==after
        repro={'status':'PASS' if passed else 'FAIL','offline':True,'recomputed_with_network_disabled':True,
          'deterministic_artifacts_byte_identical':passed,'before_sha256':before,'after_sha256':after,
          'source_sha256':sha(SOURCE),'protocol_sha256':sha(OUT/'benchmark_protocol.json'),
          'selected_models_sha256':sha(OUT/'selected_models.json'),'network_requests':0}
        json_write(OUT/'reproducibility.json',repro)
        print(f"PHASE6_OFFLINE_REPRODUCTION={repro['status']}")
        if not passed: raise SystemExit(1)
        return
    if existing.exists(): raise SystemExit('Final test already evaluated; use --reproduce only')
    result=evaluate(reproduce=False)
    report={'status':'FINAL_TEST_EVALUATED_ONCE','protocol_sha256':sha(OUT/'benchmark_protocol.json'),
      'selected_models_sha256':sha(OUT/'selected_models.json'),'source_sha256':sha(SOURCE),
      'network_requests':0,'note':'Model selection and preprocessing were frozen before test values were parsed.'}
    json_write(OUT/'final_test_evaluation_manifest.json',report)
    print('PHASE6_FINAL_TEST_EVALUATED_ONCE=PASS')

if __name__=='__main__': main()
