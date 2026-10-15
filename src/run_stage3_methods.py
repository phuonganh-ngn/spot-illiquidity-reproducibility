#!/usr/bin/env python3
"""Stage 3 linear Granger benchmark and validated continuous TE audit."""
from __future__ import annotations
import csv, hashlib, json, math, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import digamma
from scipy.stats import chi2
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.api as sm
from stage3_common import holm_adjust, deterministic_seed

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'; PROC=ROOT/'data/processed'
CHANNELS={
 'C1':('FUNDING','RV','funding_mean','spot_realized_volatility'),
 'C2':('OI','RV','oi_log_change_1d','spot_realized_volatility'),
 'C3':('OI','DOWNSIDE','oi_log_change_1d','spot_downside_semivariance'),
 'C4':('FLOW','AMIHUD','futures_taker_imbalance','spot_amihud'),
}

def transform(name,x):
 x=np.asarray(x,dtype=float)
 if name=='spot_realized_volatility':return np.log1p(x/0.01)
 if name=='spot_amihud':return np.log1p(x)
 if name=='spot_downside_semivariance':return np.log1p(x/0.0001)
 return x.copy()

def longest_consecutive(df):
 if df.empty:return df
 d=df.sort_values('date').copy(); gap=d.date.diff().dt.days.fillna(1).ne(1)
 grp=gap.cumsum(); sizes=d.groupby(grp).size(); best=sizes.max(); gbest=sizes[sizes==best].index[0]
 return d.loc[grp==gbest].copy()

def run_granger(panel):
 station=[]; rows=[]
 for cid,(deriv,spot,dcol,scol) in CHANNELS.items():
  for asset in ['BTC','ETH']:
   g=panel[panel.asset==asset].sort_values('date').copy()
   z=pd.DataFrame({'date':g.date,'x':transform(dcol,g[dcol]),'y':transform(scol,g[scol])}).dropna()
   z=longest_consecutive(z)
   for nm in ['x','y']:
    vals=z[nm].to_numpy(dtype=float)
    rec={'channel_id':cid,'asset':asset,'variable':'source' if nm=='x' else 'target','source_field':dcol if nm=='x' else scol,'n_dates':len(vals),'first_date':str(z.date.min().date()) if len(z) else '', 'last_date':str(z.date.max().date()) if len(z) else ''}
    try:
     adf=adfuller(vals,maxlag=7,regression='c',autolag='BIC',result_object=False);rec.update(adf_stat=float(adf[0]),adf_p=float(adf[1]),adf_lags=int(adf[2]))
     with warnings.catch_warnings():
      warnings.simplefilter('ignore'); kp=kpss(vals,regression='c',nlags=7)
     rec.update(kpss_stat=float(kp[0]),kpss_p=float(kp[1]),stationary=(adf[1]<.05 and kp[1]>.05))
    except Exception as e:rec.update(adf_stat=np.nan,adf_p=np.nan,adf_lags=np.nan,kpss_stat=np.nan,kpss_p=np.nan,stationary=False,stationarity_error=str(e))
    station.append(rec)
   base={'channel_id':cid,'asset':asset,'pair':f'{deriv}<->{spot}','n_consecutive_dates':len(z),'first_date':str(z.date.min().date()) if len(z) else '', 'last_date':str(z.date.max().date()) if len(z) else ''}
   if len(z)<400 or not all(x['stationary'] for x in station[-2:]):
    for direction in ['FORWARD','REVERSE']:
     rows.append({**base,'direction':direction,'lag_order':np.nan,'wald_stat':np.nan,'hac_p':np.nan,'standard_granger_p':np.nan,'spectral_radius':np.nan,'whiteness_p':np.nan,'diagnostic_status':'LINEAR_DIAGNOSTIC_FAIL','reason':'fewer than 400 consecutive dates or ADF/KPSS admissibility failed'})
    continue
   a=z[['x','y']].to_numpy(dtype=float)
   try:
    model=VAR(a)
    sel=model.select_order(7)
    p=int(sel.selected_orders.get('bic') or 1);p=max(1,min(7,p))
    fitted=model.fit(p,trend='c')
    comp=np.zeros((2*p,2*p));comp[:2,:]=np.concatenate(fitted.coefs,axis=1)
    if p>1:comp[2:,:-2]=np.eye(2*(p-1))
    rho=float(np.max(np.abs(np.linalg.eigvals(comp))))
    white=fitted.test_whiteness(nlags=max(14,p+7),adjusted=True); wp=float(white.pvalue)
    stable=rho<0.999999; whiteok=wp>.05; diagnostic=stable and whiteok
    for source,target,di in [(0,1,'FORWARD'),(1,0,'REVERSE')]:
     # In equation target: intercept, target lags, then source lags.
     Y=a[p:,target]; X=[np.ones(len(Y))]
     for lag in range(1,p+1):X.append(a[p-lag:len(a)-lag,target])
     first_source_col=1+p
     for lag in range(1,p+1):X.append(a[p-lag:len(a)-lag,source])
     Xmat=np.column_stack(X); fit=sm.OLS(Y,Xmat).fit()
     cov=sm.stats.sandwich_covariance.cov_hac(fit,nlags=7)
     ids=np.arange(first_source_col,first_source_col+p)
     b=fit.params[ids];v=cov[np.ix_(ids,ids)]
     w=float(b@np.linalg.pinv(v)@b);hp=float(chi2.sf(w,p))
     caused='y' if target==1 else 'x';causing='x' if source==0 else 'y'
     gp=float(fitted.test_causality(caused=caused,causing=causing,kind='wald').pvalue)
     rows.append({**base,'direction':di,'lag_order':p,'bic':float(sel.bic),'wald_stat':w,'hac_p':hp,'standard_granger_p':gp,'spectral_radius':rho,'stable_root_pass':stable,'whiteness_p':wp,'whiteness_pass':whiteok,'diagnostic_status':'LINEAR_DIAGNOSTIC_PASS' if diagnostic else 'LINEAR_DIAGNOSTIC_FAIL','corroboration_status':'CORROBORATING' if diagnostic else 'LINEAR_DIAGNOSTIC_FAIL','reason':'' if diagnostic else 'unstable roots or residual Portmanteau whiteness failed'})
   except Exception as e:
    for direction in ['FORWARD','REVERSE']:
     rows.append({**base,'direction':direction,'lag_order':np.nan,'wald_stat':np.nan,'hac_p':np.nan,'standard_granger_p':np.nan,'spectral_radius':np.nan,'whiteness_p':np.nan,'diagnostic_status':'LINEAR_DIAGNOSTIC_FAIL','corroboration_status':'LINEAR_DIAGNOSTIC_FAIL','reason':str(e)})
 out=pd.DataFrame(rows);out['holm_p']=np.nan
 for asset,idx in out.groupby('asset').groups.items():out.loc[idx,'holm_p']=holm_adjust(out.loc[idx,'hac_p'].to_numpy())
 stationdf=pd.DataFrame(station)
 stationdf.to_csv(OUT/'table_stage3_stationarity.csv',index=False)
 out.to_csv(OUT/'table_stage3_granger.csv',index=False)
 return out,stationdf


def _cmi_fp(x, yf, yh, dates, k=5, theiler=7, workers=1):
 """Frenzel-Pompe/KSG CMI, max norm, strict-radius counts, Theiler exclusion."""
 x=np.asarray(x,dtype=float);yf=np.asarray(yf,dtype=float).reshape(-1,1);yh=np.asarray(yh,dtype=float)
 if yh.ndim==1:yh=yh.reshape(-1,1)
 dates=np.asarray(dates,dtype=int);n=len(yf)
 if x.ndim==1:x=x.reshape(-1,1)
 if n<=k+2*theiler+5 or len(x)!=n or len(yh)!=n:return np.nan,'SAMPLE_TOO_SMALL'
 xyz=np.column_stack([x,yf,yh])
 if not np.isfinite(xyz).all():return np.nan,'NONFINITE'
 # Use enough nearest neighbours to replace every point excluded by the Theiler window.
 tree=cKDTree(xyz)
 kk=min(n,k+2*theiler+3)
 distances,indices=tree.query(xyz,k=kk,p=np.inf,workers=workers)
 if kk==1:distances=distances[:,None];indices=indices[:,None]
 eps=np.empty(n)
 for i in range(n):
  ids=np.atleast_1d(indices[i]);ds=np.atleast_1d(distances[i])
  valid=(ids!=i)&(np.abs(dates[ids]-dates[i])>theiler)
  dv=ds[valid]
  if len(dv)<k:return np.nan,'INSUFFICIENT_THEILER_NEIGHBORS'
  eps[i]=np.partition(dv,k-1)[k-1]
 if np.any(~np.isfinite(eps)) or np.any(eps<=0):return np.nan,'ZERO_OR_NONFINITE_KTH_RADIUS'
 strict=np.nextafter(eps,0.0)
 projections=[np.column_stack([x,yh]),np.column_stack([yf,yh]),yh]
 counts=[]
 # cKDTree vector-radius counts; remove self and all within-Theiler neighbours.
 for proj in projections:
  tr=cKDTree(proj)
  allc=tr.query_ball_point(proj,r=strict,p=np.inf,return_length=True,workers=workers).astype(int)
  excluded=np.zeros(n,dtype=int)
  for i in range(n):
   lo=np.searchsorted(dates,dates[i]-theiler,'left');hi=np.searchsorted(dates,dates[i]+theiler,'right')
   ids=np.r_[np.arange(lo,i),np.arange(i+1,hi)]
   if len(ids):excluded[i]=int(np.sum(np.max(np.abs(proj[ids]-proj[i]),axis=1)<eps[i]))
  c=allc-1-excluded
  counts.append(np.maximum(c,0))
 nxz,nyz,nz=counts
 val=digamma(k)+digamma(nz+1)-digamma(nxz+1)-digamma(nyz+1)
 return float(np.mean(val)),'PASS'


def _embedding(g,source,target,history):
 x=transform(source,g[source]); y=transform(target,g[target]);date_idx=np.arange(len(g),dtype=int)
 xhist=np.column_stack([pd.Series(x).shift(l).to_numpy() for l in range(history)])
 yhist=np.column_stack([pd.Series(y).shift(l).to_numpy() for l in range(history)])
 yf=pd.Series(y).shift(-1).to_numpy()
 ok=np.isfinite(xhist).all(axis=1)&np.isfinite(yhist).all(axis=1)&np.isfinite(yf)
 # Keep only full 14-calendar-day embedding blocks anchored at clean-sample start.
 idx=[];blocks=[];block_ids=[];nblocks=len(g)//14
 for bi in range(nblocks):
  block=np.arange(bi*14,(bi+1)*14)
  if ok[block].all():
   blocks.append(block);block_ids.append(bi);idx.extend(block.tolist())
 if not blocks:return None
 idx=np.asarray(idx,dtype=int)
 X=xhist[idx].copy();YF=yf[idx].copy();YH=yhist[idx].copy();dates=date_idx[idx]
 scalar=np.column_stack([X,YF.reshape(-1,1),YH])
 max_atom=max(float(np.unique(scalar[:,j],return_counts=True)[1].max()/len(scalar)) for j in range(scalar.shape[1]))
 # Scalar standard-deviation normalization is only for kNN distance geometry.
 for mat in [X,YF.reshape(-1,1),YH]:
  mu=np.mean(mat,axis=0);sd=np.std(mat,axis=0,ddof=1);sd=np.where(sd>0,sd,1.0);mat-=mu;mat/=sd
 return {'X':X,'YF':YF,'YH':YH,'dates':dates,
         'blocks':[np.arange(j*14,(j+1)*14) for j in range(len(blocks))],
         'block_ids':block_ids,'n_blocks':len(blocks),'n_dates':len(idx),
         'source':source,'target':target,'history':history,'max_exact_atom':max_atom}


def _te_one(e,k=5,theiler=7,seed=314159,reps=999,block_size=14,workers=1):
 obs,status=_cmi_fp(e['X'],e['YF'],e['YH'],e['dates'],k,theiler,workers)
 if status!='PASS':return {'observed_te':obs,'status':status,'valid_surrogates':0,'surrogate_mean':np.nan,'effective_te':np.nan,'empirical_p':np.nan}
 nb=e['n_blocks'];
 if nb<20 or e['n_dates']<400:return {'observed_te':obs,'status':'INSUFFICIENT_FULL_BLOCKS','valid_surrogates':0,'surrogate_mean':np.nan,'effective_te':np.nan,'empirical_p':np.nan,'n_dates':e['n_dates'],'n_blocks':nb}
 rng=np.random.default_rng(seed);vals=[];x=e['X'];
 for b in range(reps):
  perm=rng.permutation(nb)
  xperm=np.concatenate([x[e['blocks'][j]] for j in perm],axis=0)
  val,st=_cmi_fp(xperm,e['YF'],e['YH'],e['dates'],k,theiler,workers)
  if st=='PASS' and np.isfinite(val):vals.append(val)
 vals=np.asarray(vals,dtype=float)
 if len(vals)<reps:return {'observed_te':obs,'status':'SURROGATE_ESTIMATOR_FAILURE','valid_surrogates':len(vals),'surrogate_mean':float(np.mean(vals)) if len(vals) else np.nan,'effective_te':obs-float(np.mean(vals)) if len(vals) else np.nan,'empirical_p':np.nan,'n_dates':e['n_dates'],'n_blocks':nb}
 p=(1+int(np.sum(vals>=obs)))/(reps+1)
 return {'observed_te':obs,'surrogate_mean':float(np.mean(vals)),'effective_te':float(obs-np.mean(vals)),'surrogate_ci_lower':float(np.quantile(vals,.025)),'surrogate_ci_upper':float(np.quantile(vals,.975)),'empirical_p':p,'valid_surrogates':len(vals),'status':'PASS','n_dates':e['n_dates'],'n_blocks':nb,'seed':seed,'block_size':block_size,'k':k,'theiler_days':theiler,'units':'nats'}


def validate_te():
 rng=np.random.default_rng(88421);n=1000
 z=rng.normal(size=n);x=z+rng.normal(size=n);ynull=z+rng.normal(size=n)
 ydep=np.zeros(n)
 for t in range(n-1):ydep[t+1]=0.35*ydep[t]+x[t]+rng.normal(scale=.35)
 dates=np.arange(n)
 null,s0=_cmi_fp(x[:-1],ynull[1:],ynull[:-1],dates[:-1],k=5,theiler=0)
 dep,s1=_cmi_fp(x[:-1],ydep[1:],ydep[:-1],dates[:-1],k=5,theiler=0)
 ok=s0=='PASS' and s1=='PASS' and dep>null
 if not ok:raise RuntimeError(f'kNN CMI validation failed: null={null}, dependent={dep}, {s0}/{s1}')
 return {'fixture_seed':88421,'n':n,'null_fixture_te':null,'known_dependence_fixture_te':dep,'status':'PASS'}


def run_te(panel):
 validation=validate_te();(OUT/'te_estimator_validation.json').write_text(json.dumps(validation,indent=2,sort_keys=True)+'\n')
 specs=[]
 for cid,(deriv,spot,dcol,scol) in CHANNELS.items():
  for asset in ['BTC','ETH']:
   g=panel[panel.asset==asset].sort_values('date').reset_index(drop=True)
   for direction in ['FORWARD','REVERSE']:
    source,target=(dcol,scol) if direction=='FORWARD' else (scol,dcol)
    for history,label,reps,bsize in [(1,'PRIMARY_L1_B14',999,14),(1,'BLOCK28_SENSITIVITY',999,28),(3,'HISTORY3_ROBUSTNESS',999,14)]:
     # The strict primary null uses 14-day blocks; 28-day and l=3 are robustness only.
     e=_embedding(g,source,target,history)
     if e is None:
      result={'status':'INSUFFICIENT_FULL_BLOCKS','n_dates':0,'n_blocks':0,'valid_surrogates':0}
     elif e['max_exact_atom']>.05:
      result={'status':'TE_ESTIMATOR_UNSUITABLE_ATOM_GT_5PCT','n_dates':e['n_dates'],'n_blocks':e['n_blocks'],'valid_surrogates':0,'max_exact_atom':e['max_exact_atom']}
     else:
      # The 28-day sensitivity keeps only adjacent, complete 14-day calendar blocks.
      if bsize==28:
       id_to_pos={bi:j for j,bi in enumerate(e['block_ids'])}
       pairs=[(id_to_pos[bi],id_to_pos[bi+1]) for bi in e['block_ids'] if bi%2==0 and bi+1 in id_to_pos]
       selected=[]
       for left,right in pairs:selected.extend(e['blocks'][left].tolist());selected.extend(e['blocks'][right].tolist())
       selected=np.asarray(selected,dtype=int)
       nfull=len(pairs)
       if nfull<20 or nfull*28<400:result={'status':'INSUFFICIENT_FULL_BLOCKS','n_dates':nfull*28,'n_blocks':nfull,'valid_surrogates':0}
       else:
        e={**e,'X':e['X'][selected],'YF':e['YF'][selected],
           'YH':e['YH'][selected],'dates':e['dates'][selected],
           'blocks':[np.arange(j*28,(j+1)*28) for j in range(nfull)],
           'block_ids':list(range(nfull)),'n_blocks':nfull,'n_dates':nfull*28}
      if 'result' not in locals() or result.get('status') not in {'INSUFFICIENT_FULL_BLOCKS'}:
       seed=deterministic_seed(f'TE|{cid}|{asset}|{direction}|{label}')
       result=_te_one(e,k=5,theiler=7,seed=seed,reps=reps,block_size=bsize,workers=1)
     rec={'channel_id':cid,'channel':{'C1':'FUNDING_RV','C2':'OI_RV','C3':'OI_DOWNSIDE','C4':'FLOW_AMIHUD'}[cid], 'asset':asset,'direction':direction,'source':source,'target':target,'history_days':history,'analysis_variant':label,**result}
     if e is not None:rec['max_exact_atom']=e['max_exact_atom']
     specs.append(rec)
     print(f'TE {cid} {asset} {direction} {label}: {rec.get("status")}',flush=True)
     if 'result' in locals():del result
     # Check elapsed progress without adaptive changes; fixed replication counts stay 999.
 out=pd.DataFrame(specs)
 # Primary TE multiplicity: eight direction/pair tests per asset; unestimable p=1.
 out['holm_p']=np.nan
 mask=out.analysis_variant=='PRIMARY_L1_B14'
 for asset,idx in out[mask].groupby('asset').groups.items():out.loc[idx,'holm_p']=holm_adjust(out.loc[idx,'empirical_p'].to_numpy())
 out.to_csv(OUT/'table_stage3_transfer_entropy.csv',index=False)
 net=[]
 for (cid,asset),g in out[mask].groupby(['channel_id','asset']):
  f=g[g.direction=='FORWARD'].iloc[0];r=g[g.direction=='REVERSE'].iloc[0]
  net.append({'channel_id':cid,'asset':asset,'forward_effective_te':f.get('effective_te',np.nan),'reverse_effective_te':r.get('effective_te',np.nan),'normalized_forward_te':np.nan,'normalized_reverse_te':np.nan,'net_information_flow':np.nan,'status':'NOT_COMPUTED_PER_FROZEN_PROTOCOL','note':'Frozen Stage-2 protocol prohibits target-conditional differential-entropy normalization and raw cross-target TE comparisons; no defensible common normalization was preregistered.'})
 pd.DataFrame(net).to_csv(OUT/'table_stage3_net_information_flow.csv',index=False)
 return out

if __name__=='__main__':
 panel=pd.read_csv(PROC/'derivatives_spot_daily_panel.csv',parse_dates=['date']);panel['date']=pd.to_datetime(panel.date,utc=True)
 panel=panel[(panel.date>='2021-12-01')&(panel.date<='2024-03-03')].copy()
 print('GRANGER starting',flush=True);g,s=run_granger(panel);print('GRANGER complete',len(g),flush=True)
 print('TE starting',flush=True);te=run_te(panel);print('TE complete',len(te),flush=True)
