#!/usr/bin/env python3
"""Paired BTC/ETH contrasts and frozen confirmatory channel classification."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
from stage3_common import *

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';CACHE=ROOT/'data/interim/stage3'
PAIR={'C1':'FUNDING<->RV','C2':'OI<->RV','C3':'OI<->DOWNSIDE','C4':'FLOW<->AMIHUD'}


def paired_asset_tests():
 primary=pd.read_csv(OUT/'table_stage3_primary_ts_dml.csv')
 scores=pd.read_pickle(CACHE/'primary_oof_scores.pkl');spine=pd.date_range(START,END,freq='D')
 rows=[]
 for (cid,direction,h),g in primary.groupby(['channel_id','direction','horizon']):
  b=g[g.asset=='BTC'].iloc[0];e=g[g.asset=='ETH'].iloc[0]
  sb=scores[b.test_id].set_index('date');se=scores[e.test_id].set_index('date');common=sb.index.intersection(se.index)
  if len(common)<400:
   rows.append({'channel_id':cid,'direction':direction,'horizon':int(h),'n_common_dates':len(common),'status':'NOT_ESTIMABLE'});continue
  sb=sb.loc[common];se=se.loc[common]
  def theta(z):
   den=float(z.v@z.v);return float(z.v@z.u/den) if den>0 else np.nan
  tb=theta(sb);te=theta(se);diff=tb-te
  vb=sb.v.to_numpy();ub=sb.u.to_numpy();ve=se.v.to_numpy();ue=se.u.to_numpy()
  jb=np.mean(vb*vb);je=np.mean(ve*ve)
  psi_b=vb*(ub-tb*vb)/jb;psi_e=ve*(ue-te*ve)/je;psi=psi_b-psi_e
  arr=np.full(len(spine),np.nan);loc={d:i for i,d in enumerate(spine)}
  for d,x in zip(common,psi):arr[loc[d]]=x
  n=len(common);L=max(7,2*int(h)-1);lrv=np.nansum(arr*arr)/n
  for ell in range(1,L+1):
   aa=arr[ell:];bb=arr[:-ell];ok=np.isfinite(aa)&np.isfinite(bb);gam=np.sum(aa[ok]*bb[ok])/n if ok.any() else 0
   lrv+=2*(1-ell/(L+1))*gam
  se_diff=float(np.sqrt(max(lrv,0)/n));p=float(2*norm.sf(abs(diff/se_diff))) if se_diff>0 else np.nan
  # Paired moving calendar-block bootstrap, same sampled dates for both assets.
  B=max(7,int(h)+1);rng=np.random.default_rng(deterministic_seed(f'BTCETH|{cid}|{direction}|{h}'));ncal=len(spine);draws=[]
  at_b=np.full(ncal,-1,dtype=int);at_e=np.full(ncal,-1,dtype=int)
  for ix,d in enumerate(common):at_b[loc[d]]=ix;at_e[loc[d]]=ix
  startsmax=ncal-B+1
  for _ in range(999):
   starts=rng.integers(0,startsmax,size=int(np.ceil(ncal/B)));ind=np.concatenate([np.arange(s,s+B) for s in starts])[:ncal]
   ix=at_b[ind];ix=ix[ix>=0]
   if len(ix)==0:continue
   db=float(np.dot(vb[ix],vb[ix]));de=float(np.dot(ve[ix],ve[ix]))
   if db<=0 or de<=0:continue
   draws.append(float(np.dot(vb[ix],ub[ix])/db-np.dot(ve[ix],ue[ix])/de))
  draws=np.asarray(draws);status='PASS' if len(draws)>=950 else 'BOOTSTRAP_ADEQUACY_FAIL'
  rows.append({'channel_id':cid,'channel_pair':PAIR[cid],'direction':direction,'horizon':int(h),'theta_btc_common':tb,'theta_eth_common':te,'difference_btc_minus_eth':diff,'hac_se_difference':se_diff,'raw_p':p,'ci_lower':diff-1.959963984540054*se_diff,'ci_upper':diff+1.959963984540054*se_diff,'bootstrap_ci_lower':float(np.quantile(draws,.025)) if len(draws)>=950 else np.nan,'bootstrap_ci_upper':float(np.quantile(draws,.975)) if len(draws)>=950 else np.nan,'bootstrap_valid_reps':len(draws),'bootstrap_block_length':B,'n_common_dates':n,'first_date':common.min().date().isoformat(),'last_date':common.max().date().isoformat(),'status':status})
 out=pd.DataFrame(rows);out['holm_family']=out.channel_id+'_'+out.direction+'_ASSET_DIFFERENCE';out['holm_p']=np.nan
 for _,idx in out.groupby('holm_family').groups.items():out.loc[idx,'holm_p']=holm_adjust(out.loc[idx,'raw_p'].to_numpy())
 out.to_csv(OUT/'table_stage3_btc_eth.csv',index=False)
 return out


def method_noncontradiction(cid,asset,direction):
 tea=pd.read_csv(OUT/'table_stage3_transfer_entropy.csv');te=tea[(tea.channel_id==cid)&(tea.asset==asset)&(tea.analysis_variant=='PRIMARY_L1_B14')]
 gra=pd.read_csv(OUT/'table_stage3_granger.csv');gr=gra[(gra.channel_id==cid)&(gra.asset==asset)]
 flags=[];conflicts=[];available=[]
 for name,df,pcol,statuscol in [('TE',te,'holm_p','status'),('Granger',gr,'holm_p','diagnostic_status')]:
  f=df[df.direction=='FORWARD'];r=df[df.direction=='REVERSE']
  if len(f)!=1 or len(r)!=1:continue
  f=f.iloc[0];r=r.iloc[0]
  valid=(f[statuscol] in ('PASS','LINEAR_DIAGNOSTIC_PASS')) and (r[statuscol] in ('PASS','LINEAR_DIAGNOSTIC_PASS'))
  if not valid:continue
  available.append(name);fs=bool(float(f[pcol])<.05);rs=bool(float(r[pcol])<.05)
  own=fs if direction=='FORWARD' else rs;opp=rs if direction=='FORWARD' else fs
  if opp and not own:conflicts.append(name)
  if own or (not fs and not rs):flags.append(name)
 return {'noncontradictory':bool(flags),'available':available,'conflicts':conflicts,'supporting_methods':flags}


def directional_check(primary,robust,placebo,bootstrap,granger,te,cid,asset,direction):
 q=primary[(primary.channel_id==cid)&(primary.asset==asset)&(primary.direction==direction)].sort_values('horizon')
 adequate=len(q)==3 and bool((q.status=='PASS').all())
 signs=np.sign(q.theta.to_numpy())
 # Meaningful numerical sign threshold from frozen clean-sample descriptive SD ratio.
 sig=q[(q.holm_p<.05)&(q.theta.abs()>1e-12*(q.outcome_sd/q.exposure_sd_clean))] if {'outcome_sd','exposure_sd_clean'}<=set(q.columns) else q[q.holm_p<.05]
 if len(sig):
  svals=np.sign(sig.theta.to_numpy());candidate=int(svals[0]) if np.all(svals==svals[0]) else 0
 else:candidate=0
 if candidate:
  selected=sig.horizon.astype(int).tolist()
  if len(selected)<2:candidate=0
 else:selected=[]
 reason=[]
 if not adequate:reason.append('PRIMARY_DML_DIAGNOSTIC_FAILURE')
 if candidate==0:reason.append('FEWER_THAN_TWO_SAME_SIGN_HOLM_REJECTIONS')
 ridge=robust[(robust.channel_id==cid)&(robust.asset==asset)&(robust.direction==direction)]
 forward=pd.read_csv(OUT/'table_stage3_forward_crossfit_robustness.csv');forward=forward[(forward.channel_id==cid)&(forward.asset==asset)&(forward.direction==direction)]
 timing=pd.read_csv(OUT/'table_stage3_timing_robustness.csv');timing=timing[(timing.channel_id==cid)&(timing.asset==asset)&(timing.direction==direction)]
 byh={int(r['horizon']):r for r in ridge.to_dict('records')}
 ridgeok=True;forwardok=True;timingok=True;bootok=True
 for h in selected:
  r=byh.get(h)
  f=forward[forward.horizon==h];t=timing[timing.horizon==h]
  rm=(r.get('ridge_matched_hgb_theta',np.nan) if r is not None else np.nan)
  fm=(float(f.iloc[0].get('forward_matched_hgb_theta',np.nan)) if len(f)==1 else np.nan)
  tm=(float(t.iloc[0].get('timing_matched_hgb_theta',np.nan)) if len(t)==1 else np.nan)
  if r is None or not np.isfinite(r.get('ridge_theta',np.nan)) or not np.isfinite(rm) or int(np.sign(r['ridge_theta']))!=candidate or int(np.sign(rm))!=candidate:ridgeok=False
  if len(f)!=1 or f.iloc[0].status!='PASS' or f.iloc[0].n_dates<400 or int(np.sign(f.iloc[0].theta))!=candidate or not np.isfinite(fm) or int(np.sign(fm))!=candidate:forwardok=False
  if len(t)!=1 or t.iloc[0].status!='PASS' or int(np.sign(t.iloc[0].theta))!=candidate or not np.isfinite(tm) or int(np.sign(tm))!=candidate:timingok=False
  p=q[q.horizon==h].iloc[0]
  b=bootstrap[(bootstrap.test_id==p.test_id)&(bootstrap.bootstrap_type=='BASE')]
  l=bootstrap[(bootstrap.test_id==p.test_id)&(bootstrap.bootstrap_type=='LONGER_2B')]
  if len(b)!=1 or len(l)!=1 or b.iloc[0].status!='PASS' or l.iloc[0].status!='PASS':bootok=False
  elif candidate>0:
   if not (p.ci_lower>0 and b.iloc[0].ci_lower>0 and l.iloc[0].ci_lower>0):bootok=False
  elif not (p.ci_upper<0 and b.iloc[0].ci_upper<0 and l.iloc[0].ci_upper<0):bootok=False
 if selected and not ridgeok:reason.append('RIDGE_SIGN_OR_MATCHED_SUPPORT_FAILURE')
 if selected and not forwardok:reason.append('FORWARD_ONLY_FOLD_ROBUSTNESS_FAILURE')
 if selected and not timingok:reason.append('CONSERVATIVE_TIMING_SIGN_FAILURE')
 if selected and not bootok:reason.append('HAC_OR_BOOTSTRAP_INTERVAL_DISAGREEMENT')
 pair=PAIR[cid]
 pl=placebo[(placebo.asset==asset)&(placebo.direction==direction)&(placebo.channel_pair==pair)]
 placebo_pass=len(pl)==3 and bool((pl.status=='PASS').all()) and bool((pl.holm_p>=.05).all())
 if not placebo_pass:reason.append('PAST_TIMING_PLACEBO_REJECTS_OR_UNESTIMABLE')
 methods=method_noncontradiction(cid,asset,direction)
 if methods['conflicts']:reason.append('SECONDARY_METHOD_CONFLICT:'+','.join(methods['conflicts']))
 if not methods['available']:reason.append('NO_ADMISSIBLE_GRANGER_OR_VALIDATED_TE_BENCHMARK')
 method_ok=methods['noncontradictory'] and not methods['conflicts']
 robust_direction=bool(adequate and candidate and len(selected)>=2 and ridgeok and forwardok and timingok and bootok and placebo_pass and method_ok)
 return {'robust':robust_direction,'sign':candidate,'supported_horizons':selected,'primary_holm_horizons':sig.horizon.astype(int).tolist(),'ridge_stable':ridgeok,'forward_only_stable':forwardok,'timing_stable':timingok,'hac_bootstrap_agree':bootok,'placebo_pass':placebo_pass,'method_available':methods['available'],'method_noncontradictory':method_ok,'method_conflict':methods['conflicts'],'reasons':reason}


def create_classification():
 primary=pd.read_csv(OUT/'table_stage3_primary_ts_dml.csv');robust=pd.read_csv(OUT/'table_stage3_robustness_matrix.csv');pl=pd.read_csv(OUT/'table_stage3_falsification.csv');bt=pd.read_csv(OUT/'table_stage3_bootstrap.csv');gr=pd.read_csv(OUT/'table_stage3_granger.csv');te=pd.read_csv(OUT/'table_stage3_transfer_entropy.csv')
 # Provide full-sample clean SD only for the frozen numerical sign guard; the SD is descriptive, not a model feature.
 panel=pd.read_csv(ROOT/'data/processed/derivatives_spot_daily_panel.csv',parse_dates=['date']);panel.date=pd.to_datetime(panel.date,utc=True);panel=panel[(panel.date>=START)&(panel.date<=END)]
 d_sd={}; y_sd={}
 for asset,g in panel.groupby('asset'):
  for variable in ['funding_mean','oi_log_change_1d','futures_taker_imbalance','spot_realized_volatility','spot_amihud','spot_downside_semivariance']:
   d_sd[(asset,variable)]=float(g[variable].std(ddof=1))
 # Outcome SD depends on future aggregation, use clean-sample daily underlying scale consistently as sign guard.
 rows=[]
 for cid in PAIR:
  for asset in ['BTC','ETH']:
   result={}
   for direction in ['FORWARD','REVERSE']:
    q=primary[(primary.channel_id==cid)&(primary.asset==asset)&(primary.direction==direction)]
    variable=(q.exposure.iloc[0] if direction=='FORWARD' else q.exposure.iloc[0]);d_sd_val=d_sd.get((asset,variable),1.)
    rawy=(q.outcome.iloc[0].replace('_fwd_1d','').replace('_mean','') if direction=='FORWARD' else q.outcome.iloc[0].replace('_fwd_1d','').replace('_mean',''))
    # Resolve raw daily series names for the sign tolerance.
    if rawy=='spot_realized_volatility':rawy='spot_realized_volatility'
    elif rawy.startswith('spot_amihud'):rawy='spot_amihud'
    elif rawy.startswith('spot_downside_semivariance'):rawy='spot_downside_semivariance'
    elif rawy=='futures_taker_imbalance':rawy='futures_taker_imbalance'
    elif rawy=='oi_log_change':rawy='oi_log_change_1d'
    elif rawy=='funding_mean':rawy='funding_mean'
    ysd=d_sd.get((asset,rawy),1.)
    qq=q.copy();qq['outcome_sd']=ysd;qq['exposure_sd_clean']=d_sd_val
    result[direction]=directional_check(qq,robust,pl,bt,gr,te,cid,asset,direction)
   f=result['FORWARD'];r=result['REVERSE']
   # Required method/diagnostic failures produce INCONCLUSIVE, not a forced null.
   if f['method_conflict'] or r['method_conflict']:
    final='INCONCLUSIVE'
   elif not f['method_available'] or not r['method_available']:
    final='INCONCLUSIVE'
   elif f['robust'] and r['robust']:final='BIDIRECTIONAL'
   elif f['robust']:final='DERIVATIVES_LEADING'
   elif r['robust']:final='SPOT_LEADING'
   else:final='NO_ROBUST_DIRECTION'
   def primary_sig(direction):
    x=primary[(primary.channel_id==cid)&(primary.asset==asset)&(primary.direction==direction)]
    return '|'.join(f'h{int(z.horizon)}' for z in x.itertuples() if z.holm_p<.05)
   fpl=pl[(pl.channel_pair==PAIR[cid])&(pl.asset==asset)&(pl.direction=='FORWARD')]
   rpl=pl[(pl.channel_pair==PAIR[cid])&(pl.asset==asset)&(pl.direction=='REVERSE')]
   grpair=gr[(gr.channel_id==cid)&(gr.asset==asset)];tepair=te[(te.channel_id==cid)&(te.asset==asset)&(te.analysis_variant=='PRIMARY_L1_B14')]
   rows.append({'channel_id':cid,'channel':{'C1':'Funding/RV','C2':'OI/RV','C3':'OI/Downside','C4':'FuturesFlow/Amihud'}[cid],'asset':asset,
    'forward_holm_signal':primary_sig('FORWARD'),'reverse_holm_signal':primary_sig('REVERSE'),
    'forward_placebo':'PASS' if len(fpl)==3 and (fpl.holm_p>=.05).all() else 'FAIL',
    'reverse_placebo':'PASS' if len(rpl)==3 and (rpl.holm_p>=.05).all() else 'FAIL',
    'learner_stability':'PASS' if f['ridge_stable'] and r['ridge_stable'] else 'FAIL',
    'timing_stability':'PASS' if f['timing_stable'] and r['timing_stable'] else 'FAIL',
    'bootstrap_agreement':'PASS' if f['hac_bootstrap_agree'] and r['hac_bootstrap_agree'] else 'FAIL',
    'granger_status':'PASS' if len(grpair) and (grpair.diagnostic_status=='LINEAR_DIAGNOSTIC_PASS').any() else 'LINEAR_DIAGNOSTIC_FAIL',
    'te_status':'PASS' if len(tepair) and (tepair.status=='PASS').all() else 'PARTIAL_OR_UNSUITABLE',
    'time_variation':'PENDING_TEMPORAL_OUTPUTS','final_classification':final,
    'forward_reasons':'|'.join(f['reasons']),'reverse_reasons':'|'.join(r['reasons']),
    'forward_supported_horizons':'|'.join(map(str,f['supported_horizons'])),'reverse_supported_horizons':'|'.join(map(str,r['supported_horizons']))})
  print(f'CLASS {cid} {asset}: {final}',flush=True)
 out=pd.DataFrame(rows);out.to_csv(OUT/'channel_classification_summary.csv',index=False)
 # Directional evidence matrix row per asset/channel, with transparent multi-method status.
 out.rename(columns={'channel':'channel_pair'}).to_csv(OUT/'directional_evidence_matrix.csv',index=False)
 return out

def finalize_temporal_classification():
 path=OUT/'stage3_temporal_endpoint_tests.csv'
 if not path.exists():return
 ep=pd.read_csv(path);core={'C1','C2','C4'}
 ep['first_ci_lower']=ep.first_theta-1.959963984540054*ep.first_se
 ep['first_ci_upper']=ep.first_theta+1.959963984540054*ep.first_se
 ep['last_ci_lower']=ep.last_theta-1.959963984540054*ep.last_se
 ep['last_ci_upper']=ep.last_theta+1.959963984540054*ep.last_se
 ep['first_holm_p']=np.nan;ep['last_holm_p']=np.nan;ep['difference_holm_p']=np.nan
 for (asset,direction),idx in ep[ep.channel_id.isin(core)].groupby(['asset','direction']).groups.items():
  ep.loc[idx,'first_holm_p']=holm_adjust(ep.loc[idx,'first_p'].to_numpy())
  ep.loc[idx,'last_holm_p']=holm_adjust(ep.loc[idx,'last_p'].to_numpy())
  ep.loc[idx,'difference_holm_p']=holm_adjust(ep.loc[idx,'difference_p'].to_numpy())
 def classify(r):
  if r.channel_id not in core:return 'DESCRIPTIVE_ONLY_NOT_IN_ENDPOINT_FAMILY'
  if r.status!='ESTIMABLE' or not np.isfinite(r.difference_holm_p):return 'INCONCLUSIVE'
  reversal=(np.sign(r.first_theta)!=np.sign(r.last_theta) and r.first_holm_p<.05 and r.last_holm_p<.05 and r.difference_holm_p<.05)
  nonoverlap=(r.first_ci_upper<r.last_ci_lower) or (r.last_ci_upper<r.first_ci_lower)
  magnitude=(np.sign(r.first_theta)==np.sign(r.last_theta) and nonoverlap and r.difference_holm_p<.05)
  if reversal:return 'SUPPORTED_SIGN_REVERSAL'
  if magnitude:return 'SUPPORTED_MAGNITUDE_CHANGE'
  return 'NO_DEMONSTRATED_TIME_VARIATION'
 ep['temporal_status']=ep.apply(classify,axis=1)
 ep.to_csv(path,index=False)
 summary=pd.read_csv(OUT/'channel_classification_summary.csv')
 for i,r in summary.iterrows():
  z=ep[(ep.channel_id==r.channel_id)&(ep.asset==r.asset)]
  states=set(z.temporal_status)
  summary.loc[i,'time_variation']='SUPPORTED_SIGN_REVERSAL' if 'SUPPORTED_SIGN_REVERSAL' in states else 'SUPPORTED_MAGNITUDE_CHANGE' if 'SUPPORTED_MAGNITUDE_CHANGE' in states else 'DESCRIPTIVE_ONLY_NOT_IN_ENDPOINT_FAMILY' if 'DESCRIPTIVE_ONLY_NOT_IN_ENDPOINT_FAMILY' in states else 'NO_DEMONSTRATED_TIME_VARIATION'
 summary.to_csv(OUT/'channel_classification_summary.csv',index=False)
 summary.rename(columns={'channel':'channel_pair'}).to_csv(OUT/'directional_evidence_matrix.csv',index=False)
 return ep

if __name__=='__main__':
 paired_asset_tests();create_classification();finalize_temporal_classification()
