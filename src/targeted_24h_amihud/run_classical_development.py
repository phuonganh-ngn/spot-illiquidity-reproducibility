from __future__ import annotations
import json, hashlib, time, warnings, math, os
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd
from scipy.stats import spearmanr, norm
from arch.bootstrap import MCS
from joblib import Parallel, delayed
from common import OUT,ASSETS,TARGET,SPOT,DER,STATIC_DER,CAL,load_panel,feature_frames,complete_mask,transform_target
from classical_models import fit_predict,stable_seed

SEED=20261007
FAMILIES=['ElasticNet','RBF_SVR','ExtraTrees','HistGradientBoosting','XGBoost','LightGBM','CatBoost','MLP','GRU']
OUTER=[
 {'id':'F1','start':'2022-10-01T00:00:00Z','end':'2023-01-01T00:00:00Z'},
 {'id':'F2','start':'2023-01-01T00:00:00Z','end':'2023-04-01T00:00:00Z'},
 {'id':'F3','start':'2023-04-01T00:00:00Z','end':'2023-07-01T00:00:00Z'},
]

def readj(name):return json.loads((OUT/name).read_text())
def dump(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
def datetimes(s):return pd.to_datetime(s,utc=True)
def metrics(yraw,yt,pred,transform):
 yraw=np.asarray(yraw,float); yt=np.asarray(yt,float); pred=np.asarray(pred,float)
 if transform=='T1_raw': rawp=pred
 elif transform=='T2_log1p': rawp=np.expm1(pred)
 elif transform=='T3_log_eps': rawp=np.exp(pred)-1e-8
 else: rawp=(np.exp(pred)-1e-8)/10000.
 rawp=np.maximum(rawp,0.)
 rho=spearmanr(yraw,rawp).statistic if len(yraw)>2 else np.nan
 return {'n':int(len(yraw)),'transformed_mae':float(np.mean(np.abs(yt-pred))),'transformed_rmse':float(np.sqrt(np.mean((yt-pred)**2))),'transformed_median_absolute_error':float(np.median(np.abs(yt-pred))),'original_scale_mae':float(np.mean(np.abs(yraw-rawp))),'original_scale_rmse':float(np.sqrt(np.mean((yraw-rawp)**2))),'original_scale_median_absolute_error':float(np.median(np.abs(yraw-rawp))),'original_scale_spearman':float(rho) if np.isfinite(rho) else None}

def moving_indices(dates,block_len,n,rng):
 d=pd.to_datetime(pd.Series(dates),utc=True)
 # Pandas 3 may retain UTC timestamps at microsecond resolution. Convert the
 # four-hour divisor into the integer unit used by this datetime array.
 ticks_per_second={'s':1,'ms':10**3,'us':10**6,'ns':10**9}[d.dtype.unit]
 tick=4*3600*ticks_per_second
 iv=np.asarray(d.astype('int64').to_numpy()//tick,dtype=np.int64)
 first=int(iv.min()); last=int(iv.max()); starts=np.arange(first,max(first+1,last-block_len+2))
 if len(starts)==0: starts=np.array([first])
 ids=[]
 while len(ids)<n:
  s=int(rng.choice(starts)); loc=np.flatnonzero((iv>=s)&(iv<s+block_len))
  if len(loc):ids.extend(loc.tolist())
 return np.asarray(ids[:n],dtype=int)

def paired_boot(df,delta_col,seed,blocks=(42,84),reps=1000):
 vals=df[delta_col].to_numpy(float); rng=np.random.default_rng(seed); out={}
 for L in blocks:
  b=np.empty(reps)
  for r in range(reps):
   chunks=[]
   for _,g in df.groupby('fold',sort=False):
    ids=moving_indices(g.date,L,len(g),rng); chunks.append(g[delta_col].to_numpy(float)[ids])
   b[r]=np.concatenate(chunks).mean()
  # Finite-bootstrap plus-one correction prevents reporting a zero p-value
  # from 1,000 draws; ties remain conservative and yield p=1.
  n_le=int(np.sum(b<=0)); n_ge=int(np.sum(b>=0))
  p=float(min(1.,2*(1+min(n_le,n_ge))/(reps+1)))
  out[str(L)]={'replications':reps,'block_length_intervals':L,'mean_delta':float(vals.mean()),'ci95':[float(x) for x in np.quantile(b,[.025,.975])],'p_two_sided_bootstrap':p,'probability_improvement':float(np.mean(b>0)),'bootstrap_mean_sd':float(b.std(ddof=1))}
 return out

def holm(pvals):
 a=np.asarray(pvals,float); n=len(a); order=np.argsort(a); adj=np.empty(n); prev=0.
 for rank,j in enumerate(order):
  v=min(1.,(n-rank)*a[j]); prev=max(prev,v); adj[j]=prev
 return adj.tolist()

def hac_dm(diff,maxlag=6):
 x=np.asarray(diff,float); n=len(x); mu=float(x.mean()); z=x-mu
 gamma0=float(np.dot(z,z)/n); var=gamma0
 for lag in range(1,min(maxlag,n-1)+1):
  g=float(np.dot(z[lag:],z[:-lag])/n); var+=2*(1-lag/(maxlag+1))*g
 se=float(np.sqrt(max(var,0)/n))
 t=mu/se if se>0 else (math.inf if mu>0 else (-math.inf if mu<0 else 0.))
 p=float(2*norm.sf(abs(t))) if np.isfinite(t) else 0.
 return {'mean_loss_advantage_s0_minus_s1':mu,'hac_se':se,'hac_lags':maxlag,'z_statistic':float(t),'p_two_sided_normal':p,'n':n}

def arrays_for_asset(panel,asset):
 g,f0,f1,y=feature_frames(panel[panel.asset==asset])
 mask=complete_mask(f0,f1,y)
 dates=g.period_start_utc.reset_index(drop=True)
 yraw=y.to_numpy(float); transform=readj('target_transform_selection.json')['selected_transform']; yt=transform_target(yraw,transform)
 X={'SPOT_ONLY':f0.drop(columns='date').to_numpy(float),'SPOT_PLUS_DERIVATIVES':f1.drop(columns='date').to_numpy(float)}
 seq={}
 for info,vars_ in [('SPOT_ONLY',SPOT),('SPOT_PLUS_DERIVATIVES',SPOT+DER)]:
  seq[info]=np.stack([np.column_stack([g[v].shift(l).to_numpy(float) for v in vars_]) for l in range(5,-1,-1)],axis=1)
 static={'SPOT_ONLY':g[CAL].to_numpy(float),'SPOT_PLUS_DERIVATIVES':g[CAL+STATIC_DER].to_numpy(float)}
 return g,dates,mask,yraw,yt,X,seq,static

def score_candidate(asset,outer_id,family,config_index,cfg,info,x,yt,sq,st,inner_masks):
 total_abs=0.; total_n=0; per_fold=[]; fit_seconds=0.; warnlist=[]
 for inner,itr,iva in inner_masks:
  seed=stable_seed(SEED,asset,outer_id,family,config_index)
  pred,elapsed,warn=fit_predict(family,cfg,x[itr],yt[itr],x[iva],sq[itr],sq[iva],st[itr],st[iva],seed)
  if not np.isfinite(pred).all(): raise RuntimeError(f'Nonfinite prediction {asset}/{outer_id}/{family}/{info}/{config_index}')
  mae=float(np.mean(np.abs(yt[iva]-pred))); n=int(iva.sum()); total_abs+=mae*n; total_n+=n; fit_seconds+=elapsed
  per_fold.append({'fold':inner['id'],'n_validation':n,'transformed_mae':mae})
  warnlist.extend(warn)
 return {'config_index':config_index,'config':cfg,'inner_pooled_transformed_mae':float(total_abs/total_n),'inner_n_validation':total_n,'inner_fold_scores':per_fold,'fit_seconds_total':fit_seconds,'warnings':warnlist}

def verify_fold(train,val,dates,fold):
 if not train.any() or not val.any(): raise RuntimeError(f'Empty fold {fold}')
 max_train=dates[train].max()+pd.Timedelta(hours=28)
 if max_train>datetimes(fold['start']): raise RuntimeError(f'Train target crosses validation start in {fold}: {max_train}')
 max_val=dates[val].max()+pd.Timedelta(hours=28)
 if max_val>datetimes(fold['end']): raise RuntimeError(f'Validation target crosses block end in {fold}: {max_val}')
 if (dates[train].max()>=dates[val].min()): raise RuntimeError(f'Train/test dates overlap {fold}')

def main():
 t0=time.time(); panel=load_panel(); transform_data=readj('target_transform_selection.json'); transform=transform_data['selected_transform']
 registry=readj('model_registry.json')['families']; foldsdata=readj('temporal_folds.json')['inner_by_outer']
 data={a:arrays_for_asset(panel,a) for a in ASSETS}
 outer_predictions=[]; group_results=[]; crossfit_records=[]
 checkpoint_path=OUT/'classical_group_checkpoints.jsonl'
 completed_keys=set()
 if checkpoint_path.exists():
  for line in checkpoint_path.read_text().splitlines():
   try: rec=json.loads(line)
   except json.JSONDecodeError: continue  # tolerate an interrupted final append only
   key=(rec['asset'],rec['outer_fold'],rec['family'],rec['info_set'])
   if key in completed_keys: raise RuntimeError(f'Duplicate checkpoint key: {key}')
   completed_keys.add(key); group_results.append(rec['group_result']); outer_predictions.extend(rec['predictions'])
 # Independent candidate fits use separate deterministic seeds. Process-based
 # parallelism preserves the frozen search grid and ordered results while
 # avoiding the single-core bottleneck of the original sequential runner.
 # GRU remains sequential because it uses the GPU and global torch RNG state.
 with Parallel(n_jobs=8,backend='loky',max_nbytes='1M',verbose=0) as parallel:
  for asset in ASSETS:
   g,dates,ok,yraw,yt,X,seq,static=data[asset]
   for outer in OUTER:
    start=datetimes(outer['start']); end=datetimes(outer['end']); train_cut=start-pd.Timedelta(hours=24); val_cut=end-pd.Timedelta(hours=24)
    train=ok&(dates>=datetimes('2021-12-01T00:00:00Z'))&(dates<train_cut)
    val=ok&(dates>=start)&(dates<val_cut)
    verify_fold(train,val,dates,outer)
    inner_outer=next(x for x in foldsdata if x['outer_fold']==outer['id'])['inner_folds']
    inner_masks=[]
    for inner in inner_outer:
     is_=datetimes(inner['validation_start']); ie=datetimes(inner['validation_feature_end_exclusive']); it=datetimes(inner['train_end_exclusive'])
     itr=ok&(dates>=datetimes('2021-12-01T00:00:00Z'))&(dates<it)
     iva=ok&(dates>=is_)&(dates<ie)
     if not itr.any() or not iva.any(): raise RuntimeError(f'Empty inner fold {asset}/{outer}/{inner}')
     if dates[itr].max()+pd.Timedelta(hours=28)>is_: raise RuntimeError(f'Inner train target crosses validation start {asset}/{inner}')
     if dates[iva].max()+pd.Timedelta(hours=28)>datetimes(inner['validation_end_exclusive']): raise RuntimeError(f'Inner validation target crosses block end {asset}/{inner}')
     inner_masks.append((inner,itr,iva))
    crossfit_records.append({'asset':asset,'outer_fold':outer['id'],'train_first':dates[train].min().isoformat(),'train_last':dates[train].max().isoformat(),'validation_first':dates[val].min().isoformat(),'validation_last':dates[val].max().isoformat(),'n_train':int(train.sum()),'n_validation':int(val.sum()),'train_max_label_end_exclusive':(dates[train].max()+pd.Timedelta(hours=28)).isoformat(),'validation_max_label_end_exclusive':(dates[val].max()+pd.Timedelta(hours=28)).isoformat()})
    for family in FAMILIES:
     for info in ['SPOT_ONLY','SPOT_PLUS_DERIVATIVES']:
      checkpoint_key=(asset,outer['id'],family,info)
      if checkpoint_key in completed_keys:
       print(f"resume-skip {asset} {outer['id']} {family} {info}",flush=True)
       continue
      x=X[info]; sq=seq[info]; st=static[info]
      task_list=[(asset,outer['id'],family,ci,cfg,info,x,yt,sq,st,inner_masks) for ci,cfg in enumerate(registry[family])]
      if family=='GRU':
       candidates=[score_candidate(*task) for task in task_list]
      else:
       candidates=parallel(delayed(score_candidate)(*task) for task in task_list)
      best=min(candidates,key=lambda z:(z['inner_pooled_transformed_mae'],z['config_index']))
      seed=stable_seed(SEED,asset,outer['id'],family,best['config_index'])
      pred,fit_s,warn=fit_predict(family,best['config'],X[info][train],yt[train],X[info][val],seq[info][train],seq[info][val],static[info][train],static[info][val],seed)
      idx=np.flatnonzero(val)
      fold_rows=[]
      for j,ridx in enumerate(idx):
       item={'asset':asset,'fold':outer['id'],'date':dates.iloc[ridx].isoformat(),'target_raw':float(yraw[ridx]),'target_transformed':float(yt[ridx]),'info_set':info,'family':family,'prediction_transformed':float(pred[j]),'selected_config_index':best['config_index']}
       outer_predictions.append(item); fold_rows.append(item)
      group_results.append({'asset':asset,'outer_fold':outer['id'],'info_set':info,'family':family,'n_train':int(train.sum()),'n_validation':int(val.sum()),'train_first':dates[train].min().isoformat(),'train_last':dates[train].max().isoformat(),'validation_first':dates[val].min().isoformat(),'validation_last':dates[val].max().isoformat(),'selected_config_index':best['config_index'],'selected_config':best['config'],'inner_selected_score':best['inner_pooled_transformed_mae'],'outer_metrics':metrics(yraw[val],yt[val],pred,transform),'outer_fit_seconds':fit_s,'fit_warnings':warn,'all_inner_config_scores':candidates})
      checkpoint={'asset':asset,'outer_fold':outer['id'],'family':family,'info_set':info,'group_result':group_results[-1],'predictions':fold_rows}
      with checkpoint_path.open('a') as f:
       f.write(json.dumps(checkpoint,sort_keys=True,allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
      completed_keys.add(checkpoint_key)
      print(f"completed {asset} {outer['id']} {family} {info} best={best['config_index']} MAE={best['inner_pooled_transformed_mae']:.8g}",flush=True)
 # OOF metrics by asset, information set, family.
 pred_df=pd.DataFrame(outer_predictions)
 summary=[]
 for (asset,info,family),z in pred_df.groupby(['asset','info_set','family'],sort=False):
  summary.append({'asset':asset,'info_set':info,'family':family,**metrics(z.target_raw,z.target_transformed,z.prediction_transformed,transform),'n_dates':int(z.date.nunique()),'first_date':z.date.min(),'last_date':z.date.max()})
 # Matched S0/S1 family loss deltas and bootstrap inference.
 deltas=[]; bootstrap_outputs=[]; dm_outputs=[]
 for asset in ASSETS:
  for family in FAMILIES:
   a=pred_df[(pred_df.asset==asset)&(pred_df.family==family)&(pred_df.info_set=='SPOT_ONLY')]
   b=pred_df[(pred_df.asset==asset)&(pred_df.family==family)&(pred_df.info_set=='SPOT_PLUS_DERIVATIVES')]
   m=a.merge(b,on=['asset','fold','date','target_raw','target_transformed'],suffixes=('_s0','_s1'),validate='one_to_one')
   if len(m)!=len(a) or len(m)!=len(b): raise RuntimeError('S0/S1 row mismatch')
   m['loss_s0']=np.abs(m.target_transformed-m.prediction_transformed_s0);m['loss_s1']=np.abs(m.target_transformed-m.prediction_transformed_s1);m['delta']=m.loss_s0-m.loss_s1
   boot=paired_boot(m,'delta',stable_seed(SEED,asset,family,'direct'))
   dm=hac_dm(m.delta,6)
   row={'asset':asset,'family':family,'n_asset_day_rows':len(m),'n_unique_dates':m.date.nunique(),'first_date':m.date.min(),'last_date':m.date.max(),'spot_only_transformed_mae':float(m.loss_s0.mean()),'spot_plus_derivatives_transformed_mae':float(m.loss_s1.mean()),'delta_s0_minus_s1_positive_favors_derivatives':float(m.delta.mean()),'relative_reduction_vs_spot_only_pct':float(100*m.delta.mean()/m.loss_s0.mean()),'paired_block_bootstrap':boot,'hac_loss_differential':dm}
   deltas.append(row); bootstrap_outputs.append({'asset':asset,'family':family,'bootstrap':boot}); dm_outputs.append({'asset':asset,'family':family,**dm})
 # Holm for direct families within each asset, using 42-step paired bootstrap p-values.
 for asset in ASSETS:
  rows=[r for r in deltas if r['asset']==asset]
  p=[r['paired_block_bootstrap']['42']['p_two_sided_bootstrap'] for r in rows]
  adj=holm(p)
  for r,pa in zip(rows,adj): r['holm_p_within_asset_9_families']=float(pa)
 # best family per information set is selected only from inner-fold scores for each outer fold.
 best_predictions=[]
 for asset in ASSETS:
  for outer in OUTER:
   for info in ['SPOT_ONLY','SPOT_PLUS_DERIVATIVES']:
    gr=[x for x in group_results if x['asset']==asset and x['outer_fold']==outer['id'] and x['info_set']==info]
    best=min(gr,key=lambda x:(x['inner_selected_score'],FAMILIES.index(x['family'])))
    pred_best=pred_df[(pred_df.asset==asset)&(pred_df.fold==outer['id'])&(pred_df.info_set==info)&(pred_df.family==best['family'])]
    best_predictions.extend({'asset':asset,'fold':outer['id'],'date':r.date,'target_raw':float(r.target_raw),'target_transformed':float(r.target_transformed),'info_set':info,'selected_family_in_inner_training':best['family'],'prediction_transformed':float(r.prediction_transformed)} for r in pred_best.itertuples())
 bestdf=pd.DataFrame(best_predictions)
 bestdom=[]
 for asset in ASSETS:
  s0=bestdf[(bestdf.asset==asset)&(bestdf.info_set=='SPOT_ONLY')].rename(columns={'selected_family_in_inner_training':'family_s0','prediction_transformed':'pred_s0'})
  s1=bestdf[(bestdf.asset==asset)&(bestdf.info_set=='SPOT_PLUS_DERIVATIVES')].rename(columns={'selected_family_in_inner_training':'family_s1','prediction_transformed':'pred_s1'})
  m=s0.merge(s1,on=['asset','fold','date','target_raw','target_transformed'],validate='one_to_one')
  m['loss_s0']=np.abs(m.target_transformed-m.pred_s0);m['loss_s1']=np.abs(m.target_transformed-m.pred_s1);m['delta']=m.loss_s0-m.loss_s1
  bestdom.append({'asset':asset,'selection_rule':'best family selected solely by inner CV within each outer training fold','spot_only_families_by_fold':{f:next(x.family_s0 for x in m.itertuples() if x.fold==f) for f in sorted(m.fold.unique())},'spot_plus_families_by_fold':{f:next(x.family_s1 for x in m.itertuples() if x.fold==f) for f in sorted(m.fold.unique())},'n_asset_day_rows':len(m),'n_unique_dates':m.date.nunique(),'spot_only_transformed_mae':float(m.loss_s0.mean()),'spot_plus_derivatives_transformed_mae':float(m.loss_s1.mean()),'delta_positive_favors_derivatives':float(m.delta.mean()),'paired_block_bootstrap_42':paired_boot(m,'delta',stable_seed(SEED,asset,'best-dom'))['42']})
 # Aggregate development tuning scores to select final refit configs.
 final_cfg={}
 for asset in ASSETS:
  for info in ['SPOT_ONLY','SPOT_PLUS_DERIVATIVES']:
   for family in FAMILIES:
    recs=[r for r in group_results if r['asset']==asset and r['info_set']==info and r['family']==family]
    by=defaultdict(lambda:[0.,0.])
    for r in recs:
     for c in r['all_inner_config_scores']:
      by[c['config_index']][0]+=c['inner_pooled_transformed_mae']*c['inner_n_validation'];by[c['config_index']][1]+=c['inner_n_validation']
    scores={i:v[0]/v[1] for i,v in by.items()}
    ci=min(scores,key=lambda i:(scores[i],i));final_cfg[(asset,info,family)]={'config_index':ci,'config':registry[family][ci],'dev_pooled_inner_score':scores[ci],'all_config_scores':{str(i):float(v) for i,v in scores.items()}}
 # Fit final configs on dev only and evaluate historical reused periods exploratorily.
 historical=[]; final_models=[]
 for asset in ASSETS:
  g,dates,ok,yraw,yt,X,seq,static=data[asset]
  # Last training feature interval whose target ends by July 1, 2023.
  dev_train=ok&(dates>=datetimes('2021-12-01T00:00:00Z'))&(dates<datetimes('2023-06-30T00:00:00Z'))
  # historical labels at t start on/after 2023-07-01; target must be complete
  hist=ok&(dates>=datetimes('2023-07-01T00:00:00Z'))
  if not dev_train.any() or not hist.any(): raise RuntimeError('Missing final development or reused historical period rows')
  period_masks={'phase6_reuse':hist&(dates<datetimes('2024-03-04T00:00:00Z')),'phase7_reuse':hist&(dates>=datetimes('2024-03-04T00:00:00Z'))}
  for info in ['SPOT_ONLY','SPOT_PLUS_DERIVATIVES']:
   for family in FAMILIES:
    sel=final_cfg[(asset,info,family)]; seed=stable_seed(SEED,asset,family,'final')
    pred,elapsed,warn=fit_predict(family,sel['config'],X[info][dev_train],yt[dev_train],X[info][hist],seq[info][dev_train],seq[info][hist],static[info][dev_train],static[info][hist],seed)
    final_models.append({'asset':asset,'info_set':info,'family':family,'config_index':sel['config_index'],'config':sel['config'],'train_n':int(dev_train.sum()),'train_first':dates[dev_train].min().isoformat(),'train_last':dates[dev_train].max().isoformat(),'fit_seconds':elapsed,'warnings':warn})
    for period,pmask in period_masks.items():
     loc=np.flatnonzero(pmask); local_pred=pred[np.flatnonzero(hist)[np.isin(np.flatnonzero(hist),loc)-np.flatnonzero(hist)[0]]] if False else None
     # Pred array order is all hist rows; map through original row positions.
     histidx=np.flatnonzero(hist); pos={idx:j for j,idx in enumerate(histidx)}
     for i in loc:
      historical.append({'asset':asset,'date':dates.iloc[i].isoformat(),'period':period,'target_raw':float(yraw[i]),'target_transformed':float(yt[i]),'info_set':info,'family':family,'prediction_transformed':float(pred[pos[i]])})
    print(f"refit/evaluated historical reuse {asset} {family} {info}",flush=True)
 histdf=pd.DataFrame(historical)
 historical_summary=[]
 for (asset,period,family),z in histdf.groupby(['asset','period','family'],sort=False):
  z0=z[z.info_set=='SPOT_ONLY'].set_index('date');z1=z[z.info_set=='SPOT_PLUS_DERIVATIVES'].set_index('date')
  m=z0.join(z1,lsuffix='_s0',rsuffix='_s1',how='inner',validate='one_to_one')
  ytv=m.target_transformed_s0.to_numpy(float); rawv=m.target_raw_s0.to_numpy(float)
  l0=np.abs(ytv-m.prediction_transformed_s0.to_numpy(float));l1=np.abs(ytv-m.prediction_transformed_s1.to_numpy(float));d=l0-l1
  historical_summary.append({'asset':asset,'period':period,'family':family,'n':len(m),'first_date':m.index.min(),'last_date':m.index.max(),'spot_only_mae':float(l0.mean()),'spot_plus_derivatives_mae':float(l1.mean()),'delta_positive_favors_derivatives':float(d.mean()),'relative_reduction_pct':float(100*d.mean()/l0.mean())})
 # Development best-family select using pooled outer OOF loss, then compare reused periods.
 best_family_dev={}
 for asset in ASSETS:
  for info in ['SPOT_ONLY','SPOT_PLUS_DERIVATIVES']:
   sm=[x for x in summary if x['asset']==asset and x['info_set']==info]
   best=min(sm,key=lambda x:(x['transformed_mae'],FAMILIES.index(x['family'])))
   best_family_dev[(asset,info)]=best['family']
 best_external=[]
 for asset in ASSETS:
  f0=best_family_dev[(asset,'SPOT_ONLY')];f1=best_family_dev[(asset,'SPOT_PLUS_DERIVATIVES')]
  for period in ['phase6_reuse','phase7_reuse']:
   a=histdf[(histdf.asset==asset)&(histdf.period==period)&(histdf.info_set=='SPOT_ONLY')&(histdf.family==f0)].set_index('date')
   b=histdf[(histdf.asset==asset)&(histdf.period==period)&(histdf.info_set=='SPOT_PLUS_DERIVATIVES')&(histdf.family==f1)].set_index('date')
   m=a.join(b,lsuffix='_s0',rsuffix='_s1',how='inner',validate='one_to_one')
   l0=np.abs(m.target_transformed_s0-m.prediction_transformed_s0);l1=np.abs(m.target_transformed_s0-m.prediction_transformed_s1);d=l0-l1
   best_external.append({'asset':asset,'period':period,'spot_only_family_selected_on_development':f0,'spot_plus_family_selected_on_development':f1,'n':len(m),'spot_only_mae':float(l0.mean()),'spot_plus_derivatives_mae':float(l1.mean()),'delta_positive_favors_derivatives':float(d.mean()),'status':'EXPLORATORY_REUSE_NOT_FRESH_CONFIRMATION'})
 # MCS by outer block: separate S0, S1 and 18-model combined candidates.
 mcs_results=[]
 for asset in ASSETS:
  for fold in [x['id'] for x in OUTER]:
   z=pred_df[(pred_df.asset==asset)&(pred_df.fold==fold)]
   by={}
   for info in ['SPOT_ONLY','SPOT_PLUS_DERIVATIVES']:
    mat=z[z.info_set==info].pivot(index='date',columns='family',values='prediction_transformed').sort_index()
    ymat=z[z.info_set==info].drop_duplicates('date').set_index('date').loc[mat.index].target_transformed.to_numpy(float)
    losses=np.abs(ymat[:,None]-mat.to_numpy(float))
    try:
     mcs=MCS(pd.DataFrame(losses,columns=list(mat.columns)),size=.10,reps=1000,block_size=42,method='R',bootstrap='moving block',seed=stable_seed(SEED,asset,fold,info,'mcs'))
     mcs.compute(); by[info]={'status':'PASS','size':.10,'reps':1000,'block_size':42,'included':list(mcs.included),'excluded':list(mcs.excluded),'pvalues':{str(k):float(v) for k,v in mcs.pvalues.iloc[:,0].items()}}
    except Exception as e: by[info]={'status':'FAIL','error':f'{type(e).__name__}: {e}'}
   # Combined 18 family/information-set candidates on exactly paired rows.
   wide=z.pivot(index='date',columns=['info_set','family'],values='prediction_transformed').sort_index(); ywide=z.drop_duplicates('date').set_index('date').loc[wide.index].target_transformed.to_numpy(float)
   names=[f'{a}__{b}' for a,b in wide.columns]
   # Some frozen families produce exactly identical forecasts (for example,
   # regularized linear variants on this feature geometry). MCS implementations
   # can become singular when identical loss columns enter together. Collapse
   # exact duplicates only; retain the alias map so every registered model is
   # still traceable and no distinct forecast is discarded.
   loss_matrix=np.abs(ywide[:,None]-wide.to_numpy(float))
   unique_idx=[]; aliases={}
   for j,name in enumerate(names):
    match=next((i for i in unique_idx if np.array_equal(loss_matrix[:,j],loss_matrix[:,i])),None)
    if match is None: unique_idx.append(j)
    else: aliases[name]=names[match]
   try:
    unique_names=[names[i] for i in unique_idx]
    cm=MCS(pd.DataFrame(loss_matrix[:,unique_idx],columns=unique_names),size=.10,reps=1000,block_size=42,method='R',bootstrap='moving block',seed=stable_seed(SEED,asset,fold,'combined','mcs'))
    cm.compute();by['combined']={'status':'PASS','size':.10,'reps':1000,'block_size':42,'original_candidate_count':len(names),'unique_loss_series_count':len(unique_names),'exact_duplicate_aliases':aliases,'included':list(cm.included),'excluded':list(cm.excluded),'pvalues':{str(k):float(v) for k,v in cm.pvalues.iloc[:,0].items()}}
   except Exception as e: by['combined']={'status':'FAIL','error':f'{type(e).__name__}: {e}'}
   mcs_results.append({'asset':asset,'outer_fold':fold,'n_dates':int(z.date.nunique()),'results':by})
   print(f'completed MCS {asset} {fold}',flush=True)
 # Write all machine-readable results.
 result={'status':'COMPLETE','primary_development_sample':'nested expanding OOF 2022-10-01 through 2023-06-29 20:00 UTC','target_transform':transform,'development_fit_count':len(group_results),'candidate_tuning_fit_count':sum(len(x['all_inner_config_scores'])*3 for x in group_results),'selected_outer_fit_count':len(group_results),'families':FAMILIES,'candidate_count_per_family':20,'outer_folds':crossfit_records,'group_results':group_results,'family_oof_summary':summary,'direct_incremental_results':deltas,'inner_cv_best_models_by_fold':bestdom,'development_selected_best_family':{f'{a}|{i}':v for (a,i),v in best_family_dev.items()},'final_refit_configs':[{ 'asset':a,'info_set':i,'family':f,**v} for (a,i,f),v in final_cfg.items()],'historical_reuse_summary':historical_summary,'best_model_historical_reuse':best_external,'historical_reuse_model_fits':final_models,'mcs':mcs_results,'seeds':{'base':SEED,'derived':'SHA256-derived by asset/fold/family/config; same across S0/S1'},'software':'See qml_environment.json; classical fits use same environment.','elapsed_seconds':time.time()-t0}
 dump('classical_development_results.json',result)
 dump('spot_only_models.json',{'models':[x for x in summary if x['info_set']=='SPOT_ONLY'],'best_family_by_asset':{a:best_family_dev[(a,'SPOT_ONLY')] for a in ASSETS},'outer_fold_selected':[x for x in bestdom]})
 dump('spot_plus_derivatives_models.json',{'models':[x for x in summary if x['info_set']=='SPOT_PLUS_DERIVATIVES'],'best_family_by_asset':{a:best_family_dev[(a,'SPOT_PLUS_DERIVATIVES')] for a in ASSETS},'outer_fold_selected':[x for x in bestdom]})
 dump('direct_incremental_results.json',{'target':TARGET,'positive_delta_favors_derivatives':True,'results':deltas})
 dump('best_model_dominance.json',{'development_nested_best_model':bestdom,'best_families_selected_on_development_cv':{f'{a}|{i}':v for (a,i),v in best_family_dev.items()},'historical_reuse_only':best_external})
 # Strict OOF predictions for selected spot-only model, plus all-family scores to support audit.
 spot_oof=[]
 for asset in ASSETS:
  for outer in OUTER:
   rows=bestdf[(bestdf.asset==asset)&(bestdf.fold==outer['id'])&(bestdf.info_set=='SPOT_ONLY')]
   for r in rows.itertuples(): spot_oof.append({'asset':asset,'fold':outer['id'],'date':r.date,'target_raw':float(r.target_raw),'target_transformed':float(r.target_transformed),'oof_spot_prediction_transformed':float(r.prediction_transformed),'selected_spot_family_in_inner_training':r.selected_family_in_inner_training})
 dump('spot_oof_predictions.json',{'definition':'Nested forward OOF; each spot-only family/config and the selected family for each outer fold were fit/tuned only on earlier dates.','target_transform':transform,'predictions':spot_oof})
 dump('spot_oof_residuals.json',{'definition':'target_transformed minus strictly forward OOF prediction of inner-CV-selected SPOT_ONLY family. Residuals remain in frozen log1p(target) units.','residuals':[r|{'residual_transformed':r['target_transformed']-r['oof_spot_prediction_transformed']} for r in spot_oof]})
 dump('paired_bootstrap.json',{'results':bootstrap_outputs,'replications':1000,'block_lengths_intervals':[42,84],'seed':SEED,'sampling':'paired moving calendar blocks within each outer validation fold; same dates for S0/S1'})
 dump('loss_differential_tests.json',{'method':'paired forecast absolute-loss differential, Bartlett/Newey-West HAC with six 4-hour lags','results':dm_outputs,'direction':'positive mean S0 loss minus S1 loss favors derivatives'})
 dump('multiple_model_correction.json',{'primary_family':'per asset across nine frozen classical family comparisons','method':'Holm on two-sided paired moving-block-bootstrap p-values (42 intervals)','results':[{'asset':r['asset'],'family':r['family'],'raw_p':r['paired_block_bootstrap']['42']['p_two_sided_bootstrap'],'holm_p':r['holm_p_within_asset_9_families']} for r in deltas]})
 dump('model_confidence_set.json',{'implementation':'arch.bootstrap.MCS; 10% size, 1,000 moving-block bootstrap replications, block length 42 four-hour intervals; computed separately within each outer fold and asset for SpotOnly, Spot+Derivatives, and combined 18 candidates.','results':mcs_results})
 print(json.dumps({'status':'COMPLETE','development_fit_groups':len(group_results),'candidate_fits':result['candidate_tuning_fit_count'],'outer_predictions':len(outer_predictions),'selected_transform':transform,'elapsed_seconds':result['elapsed_seconds']},indent=2))

if __name__=='__main__': main()
