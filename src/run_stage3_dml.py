#!/usr/bin/env python3
"""Stage 3: confirmatory blocked TS-DML, robustness, and past-timing tests."""
from __future__ import annotations
import csv, hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy, sklearn
from stage3_common import *

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'; PROC=ROOT/'data/processed'; CACHE=ROOT/'data/interim/stage3'
CACHE.mkdir(parents=True,exist_ok=True)
CHANNELS={
 'C1':('POSITIONING / RISK','FUNDING','RV','funding_mean','spot_realized_volatility_fwd_{}d','spot_realized_volatility','funding_mean_fwd_{}d'),
 'C2':('LEVERAGE / RISK','OI','RV','oi_log_change_1d','spot_realized_volatility_fwd_{}d','spot_realized_volatility','oi_log_change_fwd_{}d'),
 'C3':('LEVERAGE / TAIL RISK','OI','DOWNSIDE','oi_log_change_1d','spot_downside_semivariance_fwd_{}d','spot_downside_semivariance','oi_log_change_fwd_{}d'),
 'C4':('ORDER FLOW / LIQUIDITY','FLOW','AMIHUD','futures_taker_imbalance','spot_amihud_mean_fwd_{}d','spot_amihud','futures_taker_imbalance_mean_fwd_{}d'),
}
DERV={'FUNDING':'funding_mean','OI':'oi_log_change_1d','FLOW':'futures_taker_imbalance'}
SPOT={'RV':'spot_realized_volatility','AMIHUD':'spot_amihud','DOWNSIDE':'spot_downside_semivariance'}


def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
 return h.hexdigest()

def verify_manifests():
 for name in ['protocol_manifest.csv','protocol_manifest_v2.csv']:
  p=OUT/name
  with p.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
  for r in rows:
   path=ROOT/r['file']
   if digest(path)!=r['SHA256']: raise RuntimeError(f'Protocol hash mismatch: {r["file"]}')

def make_panel():
 p=pd.read_csv(PROC/'derivatives_spot_daily_panel.csv',parse_dates=['date'])
 p['date']=pd.to_datetime(p.date,utc=True)
 p=p[(p.date>=START)&(p.date<=END)].copy().sort_values(['asset','date'])
 if len(p)!=1648 or p.groupby('asset').date.nunique().to_dict()!={'BTC':824,'ETH':824}: raise RuntimeError('Certified clean panel/sample mismatch')
 if p.duplicated(['date','asset']).any(): raise RuntimeError('Duplicate date-asset rows')
 for a,g in p.groupby('asset'):
  dates=pd.DatetimeIndex(g.date)
  if not dates.equals(pd.date_range(START,END,freq='D')): raise RuntimeError(f'Calendar mismatch {a}')
  if set(g.asset.unique())!={'BTC' if a=='BTC' else 'ETH'}: raise RuntimeError('Asset key error')
  # Row-level timing certification is not a universal eligibility mask: the
  # audited uncertified rows are specifically unknown-layout OI observations,
  # which are already missing. Verify that no uncertified OI value is consumed.
  uncert=~g.metrics_timing_certified.fillna(False).astype(bool)
  if (uncert & g.oi_close.notna()).any() or (uncert & g.oi_log_change_1d.notna()).any():
   raise RuntimeError(f'uncertified OI value appears populated for {a}')
  if not g.eligibility_h1.any(): raise RuntimeError(f'missing future labels {a}')
 return p

def write_csv(name, rows, fields=None):
 path=OUT/name
 if isinstance(rows,pd.DataFrame): rows.to_csv(path,index=False); return
 if fields is None:
  fields=list(rows[0].keys()) if rows else []
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def exposure_outcome(row):
 direction=row['direction']; asset=row['asset'];h=int(float(row['horizon']))
 deriv=row['derivative'];spot=row['spot']
 if direction=='FWD': d=DERV[deriv]; y={'RV':f'spot_realized_volatility_fwd_{h}d','AMIHUD':f'spot_amihud_mean_fwd_{h}d','DOWNSIDE':f'spot_downside_semivariance_fwd_{h}d'}[spot]
 else: d=SPOT[spot];y={'FUNDING':f'funding_mean_fwd_{h}d','OI':f'oi_log_change_fwd_{h}d','FLOW':f'futures_taker_imbalance_mean_fwd_{h}d'}[deriv]
 return d,y,h

def raw_registry():
 with (OUT/'hypothesis_registry.csv').open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def describe(panel):
 rows=[]
 series=['funding_mean','oi_log_change_1d','futures_taker_imbalance','spot_realized_volatility','spot_amihud','spot_downside_semivariance','spot_log_return_1d','spot_quote_volume_growth_1d','spot_taker_imbalance']
 for a,g in panel.groupby('asset'):
  for v in series:
   x=pd.to_numeric(g[v],errors='coerce').to_numpy(float);x=x[np.isfinite(x)]
   if not len(x):continue
   r={'asset':a,'variable':v,'n':len(x),'mean':float(np.mean(x)),'sd':float(np.std(x,ddof=1)),'min':float(np.min(x)),'max':float(np.max(x))}
   for q in [0.01,.05,.25,.5,.75,.95,.99]:r[f'p{int(100*q):02d}']=float(np.quantile(x,q))
   r['lag1_autocorrelation']=float(pd.Series(x).autocorr(1)) if len(x)>2 else np.nan
   rows.append(r)
 write_csv('table_stage3_descriptive.csv',rows)
 (OUT/'table_descriptive.csv').write_bytes((OUT/'table_stage3_descriptive.csv').read_bytes())
 corr=[]
 for a,g in panel.groupby('asset'):
  cols=['funding_mean','oi_log_change_1d','futures_taker_imbalance','spot_realized_volatility','spot_amihud','spot_downside_semivariance','spot_log_return_1d']
  for i,x in enumerate(cols):
   for y in cols[i+1:]:corr.append({'asset':a,'variable_x':x,'variable_y':y,'raw_correlation':g[[x,y]].corr().iloc[0,1]})
 write_csv('table_stage3_raw_correlations.csv',corr)

def run_primary_and_robustness(panel):
 verify_manifests()
 v2=pd.read_csv(OUT/'primary_hypothesis_registry_v2.csv',dtype=str)
 prim=v2[v2.confirmatory=='True'].copy()
 if len(prim)!=48:raise RuntimeError('V2 primary test count did not remain 48')
 prim['h_int']=prim.horizon.astype(int)
 base_results=[];base_scores={};fold_rows=[]
 # The confirmatory universe is estimated first, in frozen registry order.
 for n,row in enumerate(prim.to_dict('records'),1):
  test=row['test_id']; asset=row['asset'];h=int(row['horizon']);D=row['exposure'];Y=row['outcome']
  fit=fit_dml(panel,asset,D,Y,h,learner='HGB',scheme='BLOCKED_4',embargo=h+8)
  st=dict(fit.stats);st.update(test_id=test,channel_id=row['channel_id'],channel=row['channel_name'],direction=row['direction'],exposure=D,outcome=Y,holm_family=row['holm_family'],confirmatory=True)
  base_results.append(st);base_scores[test]=fit.scores
  for fr in fit.folds:fold_rows.append(dict(test_id=test,asset=asset,horizon=h,variant='PRIMARY_HGB_BLOCKED',**fr))
  if n%8==0:print(f'PRIMARY_HGB {n}/48',flush=True)
 primary=pd.DataFrame(base_results)
 primary['holm_p']=np.nan
 for fam,idx in primary.groupby('holm_family').groups.items():primary.loc[idx,'holm_p']=holm_adjust(primary.loc[idx,'raw_p'].to_numpy())
 # Primary date-block bootstrap: baseline block length and frozen 2B sensitivity.
 boot_rows=[];spine=pd.date_range(START,END,freq='D')
 for row in primary.to_dict('records'):
  s=base_scores[row['test_id']];h=int(row['horizon']);base_b=max(7,h+1)
  for label,b in [('BASE',base_b),('LONGER_2B',2*base_b)]:
   br=moving_block_bootstrap(s,spine,h,f"{row['test_id']}|{label}",999,block_length=b)
   boot_rows.append({'test_id':row['test_id'],'channel_id':row['channel_id'],'asset':row['asset'],'direction':row['direction'],'horizon':h,'bootstrap_type':label,**br})
  if len(boot_rows)%24==0:print(f'BOOTSTRAP {len(boot_rows)//2}/48',flush=True)
 bootstrap=pd.DataFrame(boot_rows)
 primary=primary.merge(bootstrap[bootstrap.bootstrap_type=='BASE'][['test_id','ci_lower','ci_upper','status','valid_reps']].rename(columns={'ci_lower':'bootstrap_ci_lower','ci_upper':'bootstrap_ci_upper','status':'bootstrap_status','valid_reps':'bootstrap_valid_reps'}),on='test_id',how='left')
 write_csv('table_stage3_primary_ts_dml.csv',primary)
 write_csv('table_stage3_forward.csv',primary[primary.direction=='FORWARD'])
 write_csv('table_stage3_reverse.csv',primary[primary.direction=='REVERSE'])
 write_csv('table_stage3_bootstrap.csv',bootstrap)
 # Fit matched-date robustness variants, then refit the primary HGB on each variant's support intersection.
 ridge_rows=[];forward_rows=[];timing_rows=[];robust_rows=[]
 for row in prim.to_dict('records'):
  test=row['test_id'];a=row['asset'];h=int(row['horizon']);D=row['exposure'];Y=row['outcome']
  rr=fit_dml(panel,a,D,Y,h,learner='RIDGE',scheme='BLOCKED_4',embargo=h+8)
  ff=fit_dml(panel,a,D,Y,h,learner='HGB',scheme='FORWARD_4',embargo=h+8,require_primary=True)
  tt=fit_dml(panel,a,D,Y,h,learner='HGB',scheme='BLOCKED_4',control_shift=1,exposure_shift=1,embargo=h+9)
  for variant,res,dest in [('RIDGE_BLOCKED',rr,ridge_rows),('HGB_FORWARD',ff,forward_rows),('HGB_TIMING_GAP',tt,timing_rows)]:
   z=dict(res.stats);z.update(test_id=test,channel_id=row['channel_id'],direction=row['direction'],holm_family=row['holm_family'],variant=variant)
   dest.append(z)
   for fr in res.folds:fold_rows.append(dict(test_id=test,asset=a,horizon=h,variant=variant,**fr))
  bdates=set(base_scores[test].date); vdates=set(rr.scores.date)&bdates; fdates=set(ff.scores.date)&bdates; tdates=set(tt.scores.date)&bdates
  # Freeze matched support and refit baseline nuisance functions/final score on that exact intersection.
  matched={}
  for name,common in [('RIDGE',vdates),('FORWARD',fdates),('TIMING',tdates)]:
   if common:
    mr=fit_dml(panel,a,D,Y,h,learner='HGB',scheme='BLOCKED_4',embargo=h+8,allowed_dates=common)
    matched[name]=mr.stats
   else:matched[name]={'theta':np.nan,'raw_p':np.nan,'status':'NO_COMMON_DATES'}
  br=primary.loc[primary.test_id==test].iloc[0]
  robust_rows.append({'test_id':test,'channel_id':row['channel_id'],'asset':a,'direction':row['direction'],'horizon':h,
    'primary_sign':int(np.sign(br.theta)) if np.isfinite(br.theta) else 0,
    'ridge_sign':int(np.sign(rr.stats['theta'])) if np.isfinite(rr.stats['theta']) else 0,
    'ridge_sign_match':bool(np.sign(matched['RIDGE']['theta'])==np.sign(rr.stats['theta'])) if np.isfinite(matched['RIDGE']['theta']) and np.isfinite(rr.stats['theta']) else False,
    'ridge_theta':rr.stats['theta'],'ridge_hac_p':rr.stats['raw_p'],'ridge_matched_hgb_theta':matched['RIDGE']['theta'],
    'forward_sign':int(np.sign(ff.stats['theta'])) if np.isfinite(ff.stats['theta']) else 0,
    'forward_hac_p':ff.stats['raw_p'],'forward_matched_hgb_theta':matched['FORWARD']['theta'],
    'timing_gap_sign':int(np.sign(tt.stats['theta'])) if np.isfinite(tt.stats['theta']) else 0,
    'timing_gap_hac_p':tt.stats['raw_p'],'timing_matched_hgb_theta':matched['TIMING']['theta'],
    'ridge_sign_match_primary':bool(np.sign(rr.stats['theta'])==np.sign(br.theta)) if np.isfinite(rr.stats['theta']) and np.isfinite(br.theta) else False,
    'forward_sign_match_primary':bool(np.sign(ff.stats['theta'])==np.sign(br.theta)) if np.isfinite(ff.stats['theta']) and np.isfinite(br.theta) else False,
    'timing_sign_match_primary':bool(np.sign(tt.stats['theta'])==np.sign(br.theta)) if np.isfinite(tt.stats['theta']) and np.isfinite(br.theta) else False,
    'ridge_n_dates':rr.stats['n_dates'],'forward_n_dates':ff.stats['n_dates'],'timing_n_dates':tt.stats['n_dates']})
  if len(robust_rows)%8==0:print(f'ROBUSTNESS {len(robust_rows)}/48',flush=True)
 ridge=pd.DataFrame(ridge_rows);forward=pd.DataFrame(forward_rows);timing=pd.DataFrame(timing_rows);robust=pd.DataFrame(robust_rows)
 write_csv('table_stage3_learner_robustness.csv',ridge)
 write_csv('table_stage3_timing_robustness.csv',timing)
 write_csv('table_stage3_forward_crossfit_robustness.csv',forward)
 write_csv('stage3_crossfit_realized.csv',pd.DataFrame(fold_rows))
 # Past-window placebo set: all 108 originally registered directional falsifications.
 old=raw_registry();placebo_specs=[r for r in old if r['kind']=='PAST_TIMING_FALSIFICATION']
 if len(placebo_specs)!=108:raise RuntimeError(f'Expected 108 frozen placebo rows; found {len(placebo_specs)}')
 placebo_rows=[]
 for n,r in enumerate(placebo_specs,1):
  k=int(float(r['horizon']));placebo_panel,d,y,k=make_placebo(panel,r,k)
  fit=fit_dml(placebo_panel,r['asset'],d,y,k,learner='HGB',scheme='BLOCKED_4',control_shift=k,embargo=k+8,require_primary=False)
  s=dict(fit.stats);s.update(test_id=r['test_id'],channel_pair=r['pair'],direction='FORWARD' if r['direction']=='FWD' else 'REVERSE',family_id=r['family_id'],placebo_k=k,registered_status='PRIMARY_CHANNEL' if is_primary_pair(r) else 'SECONDARY_PAIR')
  placebo_rows.append(s)
  for fr in fit.folds:fold_rows.append(dict(test_id=r['test_id'],asset=r['asset'],horizon=k,variant='PAST_PLACEBO',**fr))
  if n%18==0:print(f'PLACEBO {n}/108',flush=True)
 placebo=pd.DataFrame(placebo_rows);placebo['holm_p']=np.nan
 for _,idx in placebo.groupby('family_id').groups.items():placebo.loc[idx,'holm_p']=holm_adjust(placebo.loc[idx,'raw_p'].to_numpy())
 write_csv('table_stage3_falsification.csv',placebo)
 write_csv('table_stage3_primary_falsification.csv',placebo[placebo.registered_status=='PRIMARY_CHANNEL'])
 write_csv('stage3_crossfit_realized.csv',pd.DataFrame(fold_rows))
 # Merge robustness/falsification columns without collapsing their diagnostics.
 pmain=primary.merge(robust,on=['test_id','channel_id','asset','direction','horizon'],suffixes=('','_rob'),how='left')
 p_placebo=placebo[placebo.registered_status=='PRIMARY_CHANNEL'].groupby(['channel_pair','asset','direction']).agg(placebo_any_holm_reject=('holm_p',lambda x:bool((x<.05).any())),placebo_estimable=('status',lambda x:bool((x=='PASS').all()))).reset_index()
 pairmap={'C1':'FUNDING<->RV','C2':'OI<->RV','C3':'OI<->DOWNSIDE','C4':'FLOW<->AMIHUD'}
 pmain['pair']=pmain.channel_id.map(pairmap)
 # Save exact OOF scores for inference reproduction/paired BTC-ETH contrasts.
 pd.to_pickle(base_scores,CACHE/'primary_oof_scores.pkl')
 write_csv('table_stage3_multiple_testing.csv',primary[['test_id','channel_id','asset','direction','horizon','raw_p','holm_family','holm_p','status']])
 write_csv('table_stage3_robustness_matrix.csv',robust)
 # No classifications until validated Granger/TE methods have run.
 metadata={
  'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__,
  'protocol_v2':'PASS','clean_sample_start':START.date().isoformat(),'clean_sample_end':END.date().isoformat(),'unique_dates':824,
  'assets':['BTC','ETH'],'confirmatory_test_count':48,'secondary_original_dml_tests':60,
  'primary_learner':'HistGradientBoostingRegressor','primary_hyperparameters':{'loss':'squared_error','learning_rate':0.05,'max_iter':200,'max_leaf_nodes':7,'min_samples_leaf':25,'l2_regularization':1.0,'max_bins':255,'early_stopping':False,'random_state':314159},
  'robust_learner':'Ridge(alpha=1.0,fit_intercept=True), training-fold StandardScaler','controls':CONTROLS,
  'crossfit':'4 fixed contiguous blocks; blocked and forward-only; calendar embargo h+8; timing-gap h+9; no row shuffling',
  'hac':'Bartlett Newey-West; L=max(7,2h-1); calendar-spine scores, gaps remain gaps',
  'bootstrap':{'replications':999,'seed_base':314159,'block_length':'max(7,h+1)','longer_block':'2B','fixed_oof_nuisance_predictions':True},
  'interpretation':'conditional directional associations; no structural causal identification',
  'network_access':'none; cached certified inputs only',
 }
 (OUT/'stage3_model_manifest.json').write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print('PRIMARY_DML_COMPLETE=48; PLACEBOS=108; ESTIMATION_SCOPE=clean sample only',flush=True)


def is_primary_pair(r):
 return (r['derivative'],r['spot']) in {('FUNDING','RV'),('OI','RV'),('OI','DOWNSIDE'),('FLOW','AMIHUD')}

def make_placebo(panel,row,k):
 asset=row['asset'];direction=row['direction'];deriv=row['derivative'];spot=row['spot']
 g=panel[panel.asset==asset].sort_values('date').copy().set_index('date')
 if direction=='FWD':
  d=DERV[deriv]; base=SPOT[spot]
  vals=pd.concat([g[base].shift(j) for j in range(1,k+1)],axis=1)
  if spot=='RV': y=np.sqrt(np.square(vals).sum(axis=1,min_count=k))
  elif spot=='DOWNSIDE': y=vals.sum(axis=1,min_count=k)
  elif spot=='AMIHUD': y=vals.mean(axis=1).where(vals.notna().all(axis=1))
  else:raise ValueError(spot)
 else:
  d=SPOT[spot]
  if deriv=='FUNDING':
   vals=pd.concat([g['funding_mean'].shift(j) for j in range(1,k+1)],axis=1);y=vals.mean(axis=1).where(vals.notna().all(axis=1))
  elif deriv=='FLOW':
   vals=pd.concat([g['futures_taker_imbalance'].shift(j) for j in range(1,k+1)],axis=1);y=vals.mean(axis=1).where(vals.notna().all(axis=1))
  elif deriv=='OI':
   y=np.log(g['oi_close'].shift(1)/g['oi_close'].shift(k+1))
  else:raise ValueError(deriv)
 # Add a unique target column in panel by date-asset; no future fields are used.
 temp=panel.copy();temp['_stage3_past_y']=np.nan
 idx=temp.asset.eq(asset);ymap=y.to_dict();temp.loc[idx,'_stage3_past_y']=temp.loc[idx,'date'].map(ymap).to_numpy()
 return temp,d,'_stage3_past_y',k

if __name__=='__main__':
 verify_manifests();panel=make_panel();describe(panel);run_primary_and_robustness(panel)
