#!/usr/bin/env python3
"""Temporal-CV tuning using development values only; never parses test values."""
from __future__ import annotations
import hashlib,json,time,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from phase6_benchmark_core import *

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stable_seed(*parts): return SEED + sum(sum(ord(c) for c in str(p)) for p in parts)

def main():
    protocol_path=OUT/'benchmark_protocol.json'
    expected=(OUT/'benchmark_protocol.sha256').read_text().split()[0]
    if sha(protocol_path)!=expected: raise SystemExit('Frozen protocol hash mismatch')
    protocol=json.loads(protocol_path.read_text())
    if protocol.get('protocol_status')!='FROZEN_BEFORE_FINAL_TEST_EVALUATION': raise SystemExit('Protocol not frozen')
    df=load_panel_dev_only()
    if (df.period_start_utc>=SPLIT).any(): raise SystemExit('Final-test date entered dev frame')
    data_validation={'status':'PASS','input_path':str(SOURCE.relative_to(ROOT)),
        'input_sha256':sha(SOURCE),'protocol_sha256':expected,
        'full_panel_expected_rows':9888,'development_rows_by_asset':{a:int(n) for a,n in df.groupby('asset').size().items()},
        'final_test_rows_counted_without_value_parsing':2964,
        'development_values_parsed_only':True,'no_final_test_feature_or_target_values_parsed':True,
        'development_sample_bounds':{'first':df.period_start_utc.min().isoformat(),'last':df.period_start_utc.max().isoformat()},
        'horizon_task_counts':{},'same_information_set_rows':'intersection requiring every frozen feature at t-5...t and valid full target window'}
    development={ 'status':'RUNNING','protocol_sha256':expected,'selection_metric':'pooled MAE on log1p target',
        'tuning_fits':[],'selected':{},'task_sample_sizes':{},'model_family_cv_summary':{} }
    oof={ 'protocol_sha256':expected,'prediction_scale':'log1p target','rows':[] }
    fold_realized={ 'protocol_sha256':expected,'folds':[] }
    families=['OLS']+list(MODEL_CONFIGS.keys())
    config_map={'OLS':[{}],**MODEL_CONFIGS}
    task_counter=0
    for asset in ['BTC','ETH']:
      for target in TARGETS:
       for horizon in HORIZONS:
        task=make_task(df,asset,horizon,target)
        n=len(task['y']); task_id=f'{asset}__{target}__h{horizon}'
        data_validation['horizon_task_counts'][task_id]={'n':n,'first':task['date'].min().isoformat() if n else None,'last':task['date'].max().isoformat() if n else None,
          'last_target_end':task['target_end'].max().isoformat() if n else None}
        development['task_sample_sizes'][task_id]={'n':n,'first':task['date'].min().isoformat() if n else None,'last':task['date'].max().isoformat() if n else None}
        if n<1000: raise RuntimeError(f'Insufficient dev examples for {task_id}: {n}')
        task_counter+=1
        ylog=logtarget(task['y'])
        for info_name in ['SPOT_ONLY','SPOT_PLUS_ALL_DERIVATIVES']:
          names=INFOSETS[info_name]
          Xflat,Xseq=x_for_set(task['Xseq'],names)
          family_config_preds={}; family_results={}
          for family in families:
            configs=config_map[family]
            scores=[]; fold_preds_by_config=[]; runtimes=[]
            for cfg_idx,config in enumerate(configs):
              preds=np.full(n,np.nan); folds_used=[]; fit_seconds=0.0
              for fold in DEV_FOLDS:
                fold_name,start,end=fold
                train_idx,val_idx=fold_masks(task,fold)
                if train_idx.sum()<500 or val_idx.sum()<100:
                    raise RuntimeError(f'Invalid fold size {task_id}/{fold_name}: train={train_idx.sum()} val={val_idx.sum()}')
                if (task['target_end'][train_idx]>=start).any() or (task['target_end'][val_idx]>=end).any():
                    raise RuntimeError('Target window crossed temporal fold boundary')
                # Fold-specific preprocessing is inside fit_predict.
                ytr=ylog[train_idx]
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    p,elapsed=fit_predict(family,config,Xflat[train_idx],Xseq[train_idx],ytr,
                                          Xflat[val_idx],Xseq[val_idx],stable_seed(task_id,info_name,family,cfg_idx,fold_name))
                fit_seconds+=elapsed
                preds[val_idx]=p
                folds_used.append({'fold':fold_name,'train_n':int(train_idx.sum()),'validation_n':int(val_idx.sum()),
                  'train_first':task['date'][train_idx].min().isoformat(),'train_last':task['date'][train_idx].max().isoformat(),
                  'validation_first':task['date'][val_idx].min().isoformat(),'validation_last':task['date'][val_idx].max().isoformat(),
                  'fit_seconds':elapsed,'warnings':[str(w.message) for w in caught]})
                fold_realized['folds'].append({'asset':asset,'target':target,'horizon':horizon,'information_set':info_name,
                  'model_family':family,'config_index':cfg_idx,**folds_used[-1]})
              valid=np.isfinite(preds)
              if valid.sum()==0: raise RuntimeError(f'No CV predictions for {task_id}/{family}')
              metrics=metric_pack(ylog[valid],preds[valid],task['y'][valid],target)
              scores.append(metrics['mae_log1p']); runtimes.append(fit_seconds)
              fold_preds_by_config.append(preds)
              development['tuning_fits'].append({'task_id':task_id,'information_set':info_name,'model_family':family,
                'config_index':cfg_idx,'config':config,'cv_metrics':metrics,'fit_seconds_total':fit_seconds,
                'cv_n':int(valid.sum()),'folds':folds_used})
            best_idx=int(np.argmin(scores))
            best_preds=fold_preds_by_config[best_idx]
            family_config_preds[family]=best_preds
            family_results[family]={'selected_config_index':best_idx,'selected_config':configs[best_idx],
                'config_mae_log1p':scores,'selected_cv_mae_log1p':float(scores[best_idx]),
                'selected_fit_seconds':runtimes[best_idx]}
          # Fixed no-tuning baselines, generated within each expanding fold.
          baseline_preds={'training_mean':np.full(n,np.nan),'persistence':np.full(n,np.nan)}
          for fold in DEV_FOLDS:
            train_idx,val_idx=fold_masks(task,fold)
            bb=simple_baseline_predictions(task,train_idx,val_idx,target)
            for bname,p in bb.items(): baseline_preds[bname][val_idx]=p
          allpred={**family_config_preds,**baseline_preds}
          ensemble_sources=['Ridge','RBF_SVR','ExtraTrees','HistGradientBoosting','MLP','GRU']
          mat=np.column_stack([allpred[x] for x in ensemble_sources])
          ens=np.nanmean(mat,axis=1); ens[np.isnan(mat).any(axis=1)]=np.nan
          allpred['equal_weight_ensemble']=ens
          chosen_metrics={}
          for family,pred in allpred.items():
            valid=np.isfinite(pred)
            chosen_metrics[family]=metric_pack(ylog[valid],pred[valid],task['y'][valid],target)
          development['selected'][f'{task_id}__{info_name}']={'models':family_results,
            'baseline_cv_metrics':{k:chosen_metrics[k] for k in ['training_mean','persistence']},
            'selected_model_cv_metrics':{k:chosen_metrics[k] for k in family_results.keys()},
            'ensemble_cv_metrics':chosen_metrics['equal_weight_ensemble'],
            'n_oof':int(np.isfinite(ens).sum())}
          development['model_family_cv_summary'][f'{task_id}__{info_name}']=chosen_metrics
          # Store selected OOF predictions only; no in-sample development predictions.
          valid=np.isfinite(ens)
          oof['rows'].append({'asset':asset,'target':target,'horizon_intervals':horizon,'horizon_hours':4*horizon,
            'information_set':info_name,'date':[x.isoformat() for x in task['date'][valid]],
            'target_end':[x.isoformat() for x in task['target_end'][valid]],
            'target_raw':task['y'][valid].tolist(),'target_log1p':ylog[valid].tolist(),
            'predictions_log1p':{k:np.asarray(v)[valid].tolist() for k,v in allpred.items()}})
    development['status']='PASS'
    json_write(OUT/'data_validation.json',data_validation)
    json_write(OUT/'development_results.json',development)
    json_write(OUT/'selected_models.json',{'status':'FROZEN_AFTER_DEVELOPMENT_CV','protocol_sha256':expected,'selection_metric':'pooled log1p MAE','selected':development['selected']})
    json_write(OUT/'oof_predictions.json',oof)
    fold_spec=json.loads((OUT/'temporal_folds.json').read_text())
    fold_spec['realized']=fold_realized['folds']
    json_write(OUT/'temporal_folds.json',fold_spec)
    (OUT/'selected_models.sha256').write_text(sha(OUT/'selected_models.json')+'  selected_models.json\n')
    print(f"PHASE6_DEVELOPMENT_PASS tasks={task_counter} tuning_fits={len(development['tuning_fits'])} protocol={expected}")

if __name__=='__main__': main()
