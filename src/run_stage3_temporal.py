#!/usr/bin/env python3
"""Predefined rolling and filtered state-space diagnostics for Stage 3."""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from stage3_common import *
from run_stage3_methods import CHANNELS as METHOD_CHANNELS, transform

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';PROC=ROOT/'data/processed'
CHANNELS={
 'C1':('FUNDING','RV','funding_mean','spot_realized_volatility'),
 'C2':('OI','RV','oi_log_change_1d','spot_realized_volatility'),
 'C3':('OI','DOWNSIDE','oi_log_change_1d','spot_downside_semivariance'),
 'C4':('FLOW','AMIHUD','futures_taker_imbalance','spot_amihud'),
}

def run_rolling(panel):
 rows=[];spine=pd.date_range(START,END,freq='D')
 for win in [365,540]:
  last_start=len(spine)-win
  starts=list(range(0,last_start+1,30))
  for start_i in starts:
   a=spine[start_i];b=a+pd.Timedelta(days=win-1)
   window=(a,b)
   for cid,(deriv,spot,dcol,scol) in CHANNELS.items():
    for asset in ['BTC','ETH']:
     g=panel[panel.asset==asset]
     for direction in ['FORWARD','REVERSE']:
      D,Y=(dcol,scol) if direction=='FORWARD' else (scol,dcol)
      # Rolling DML uses the existing future h1 label, not a newly constructed outcome.
      if direction=='FORWARD':Y=f'{scol}_fwd_1d' if scol!='spot_amihud' else 'spot_amihud_mean_fwd_1d'
      else:
       if deriv=='FUNDING':Y='funding_mean_fwd_1d'
       elif deriv=='OI':Y='oi_log_change_fwd_1d'
       else:Y='futures_taker_imbalance_mean_fwd_1d'
      try:
       fit=fit_dml(g,asset,D,Y,1,learner='HGB',scheme='BLOCKED_4',embargo=9,window=window,require_primary=False)
       rec={'window_days':win,'window_start':a.date().isoformat(),'window_end':b.date().isoformat(),'channel_id':cid,'asset':asset,'direction':direction,'source':D,'outcome':Y,**fit.stats}
       rows.append(rec)
      except Exception as e:
       rows.append({'window_days':win,'window_start':a.date().isoformat(),'window_end':b.date().isoformat(),'channel_id':cid,'asset':asset,'direction':direction,'source':D,'outcome':Y,'status':'DML_ERROR','reason':str(e),'n_dates':0})
  print(f'ROLLING_{win}D_COMPLETE={len([r for r in rows if r["window_days"]==win])}',flush=True)
 out=pd.DataFrame(rows)
 out.to_csv(OUT/'table_stage3_time_variation.csv',index=False)
 out[out.window_days==365].to_csv(OUT/'stage3_rolling_directionality.csv',index=False)
 # Frozen endpoint-only difference tests for h=1; intermediate windows remain descriptive.
 ends=[]
 for cid in CHANNELS:
  for asset in ['BTC','ETH']:
   for direction in ['FORWARD','REVERSE']:
    q=out[(out.window_days==365)&(out.channel_id==cid)&(out.asset==asset)&(out.direction==direction)].sort_values('window_start')
    if len(q)<2:continue
    first=q.iloc[0];last=q.iloc[-1]
    se1=float(first.hac_se) if np.isfinite(first.hac_se) else np.nan
    se2=float(last.hac_se) if np.isfinite(last.hac_se) else np.nan
    diff=float(last.theta-first.theta) if np.isfinite(last.theta) and np.isfinite(first.theta) else np.nan
    sed=float(np.sqrt(se1*se1+se2*se2)) if np.isfinite(se1) and np.isfinite(se2) else np.nan
    p=float(2*norm.sf(abs(diff/sed))) if np.isfinite(diff) and np.isfinite(sed) and sed>0 else np.nan
    ends.append({'channel_id':cid,'asset':asset,'direction':direction,'first_window_start':first.window_start,'first_window_end':first.window_end,'first_theta':first.theta,'first_se':se1,'first_p':first.raw_p,'first_n':first.n_dates,'last_window_start':last.window_start,'last_window_end':last.window_end,'last_theta':last.theta,'last_se':se2,'last_p':last.raw_p,'last_n':last.n_dates,'difference_last_minus_first':diff,'difference_se_disjoint_hac':sed,'difference_p':p,'sign_reversal':bool(np.sign(first.theta)!=np.sign(last.theta)) if np.isfinite(first.theta) and np.isfinite(last.theta) else False,'status':'ESTIMABLE' if np.isfinite(p) else 'INCONCLUSIVE'})
 ed=pd.DataFrame(ends);ed['family']='ENDPOINT_DIFFERENCE_BY_ASSET_DIRECTION';ed['holm_p']=np.nan
 if len(ed):
  for _,idx in ed.groupby(['asset','direction']).groups.items():ed.loc[idx,'holm_p']=holm_adjust(ed.loc[idx,'difference_p'].to_numpy())
 ed.to_csv(OUT/'stage3_temporal_endpoint_tests.csv',index=False)
 return out,ed

def kalman_filter(y,ylag,xlag,alpha,rho,obsvar,q,beta0=0.,P0=1.,start=1,stop=None):
 n=len(y);stop=n if stop is None else min(n,stop);beta=np.full(n,np.nan);pout=np.full(n,np.nan);ll=0.;b=float(beta0);P=float(P0)
 for t in range(start,stop):
  if not (np.isfinite(y[t]) and np.isfinite(ylag[t]) and np.isfinite(xlag[t])):
   P=P+q;beta[t]=b;pout[t]=P
   continue
  F=P*xlag[t]**2+obsvar
  if not np.isfinite(F) or F<=0:return None
  innov=y[t]-alpha-rho*ylag[t]-b*xlag[t]
  ll+=-.5*(np.log(2*np.pi*F)+innov*innov/F)
  K=P*xlag[t]/F
  b=b+K*innov;P=max((1-K*xlag[t])*P+q,0.)
  beta[t]=b;pout[t]=P
 return ll,b,P,beta,pout

def fit_tvp(g,asset,cid,direction,source,target):
 g=g.sort_values('date').copy().reset_index(drop=True)
 x=transform(source,g[source]);y=transform(target,g[target]);xl=np.r_[np.nan,x[:-1]];yl=np.r_[np.nan,y[:-1]]
 train_end=min(206,len(g));train=np.arange(1,train_end)
 valid=np.isfinite(y[train])&np.isfinite(yl[train])&np.isfinite(xl[train]);train=train[valid]
 if len(train)<150:return None,{'status':'STATE_SPACE_UNSTABLE','reason':'fewer than 150 training rows'}
 mx=float(np.mean(x[train-1]));sx=float(np.std(x[train-1],ddof=1));my=float(np.mean(y[train]));sy=float(np.std(y[train],ddof=1))
 if sx<=0 or sy<=0:return None,{'status':'STATE_SPACE_UNSTABLE','reason':'zero training scale'}
 xs=(x-mx)/sx;ys=(y-my)/sy;xsl=np.r_[np.nan,xs[:-1]];ysl=np.r_[np.nan,ys[:-1]]
 def objective(par):
  alpha=par[0];rho=.99*np.tanh(par[1]);ov=np.exp(par[2]);q=np.exp(par[3])
  z=kalman_filter(ys,ysl,xsl,alpha,rho,ov,q,P0=1.,stop=train_end)
  if z is None:return 1e20
  # Likelihood only from observations among first 206 sample dates.
  return -z[0]
 fits=[]
 for q0 in [1e-6,1e-4,1e-2]:
  init=np.array([0.,np.arctanh(.1/.99),np.log(max(np.nanvar(ys[train]),1e-4)),np.log(q0)])
  with warnings.catch_warnings():
   warnings.simplefilter('ignore')
   fit=minimize(objective,init,method='L-BFGS-B',bounds=[(-5,5),(-2.65,2.65),(-20,5),(-20,2)],options={'maxiter':1000,'ftol':1e-10})
  fits.append(fit)
 good=[f for f in fits if f.success and np.isfinite(f.fun)]
 if not good:return None,{'status':'STATE_SPACE_UNSTABLE','reason':'all fixed-start optimizations failed','optimizer_messages':' | '.join(str(f.message) for f in fits)}
 best=min(good,key=lambda f:f.fun);alpha=best.x[0];rho=.99*np.tanh(best.x[1]);ov=np.exp(best.x[2]);q=np.exp(best.x[3])
 # Filter first 206 dates to obtain an initialization, then continue one-sided only.
 train_filter=kalman_filter(ys,ysl,xsl,alpha,rho,ov,q,P0=1.,stop=train_end)
 _,b,P,_,_=train_filter
 full=kalman_filter(ys,ysl,xsl,alpha,rho,ov,q,beta0=b,P0=P,start=train_end,stop=len(g))
 if full is None:return None,{'status':'STATE_SPACE_UNSTABLE','reason':'non-finite filtered covariance'}
 _,_,_,betas,vars_=full
 betaraw=betas*sy/sx
 df=pd.DataFrame({'date':g.date,'asset':asset,'channel_id':cid,'direction':direction,'source':source,'target':target,'filtered_beta_raw':betaraw,'filtered_state_variance_scaled':vars_*sy*sy/(sx*sx),'training_end':g.date.iloc[train_end-1].date().isoformat(),'status':'FILTERED_AFTER_TRAINING_ONLY'})
 diag={'status':'PASS','training_observations':len(train),'training_first_date':g.date.iloc[train[0]].date().isoformat(),'training_end_date':g.date.iloc[train_end-1].date().isoformat(),'rho_static':float(rho),'observation_variance_scaled':float(ov),'state_variance_scaled':float(q),'training_loglikelihood':float(-best.fun),'optimizer_starts':3,'optimizer_successes':len(good),'optimizer_success':bool(best.success),'start_likelihoods':[float(-f.fun) if np.isfinite(f.fun) else None for f in fits],'filtered_only_after_training':True,'P_min':float(np.nanmin(vars_[train_end:]))}
 return df,diag

def run_tvp(panel):
 records=[];diag=[]
 for cid,(deriv,spot,dcol,scol) in CHANNELS.items():
  for asset in ['BTC','ETH']:
   g=panel[panel.asset==asset]
   for direction in ['FORWARD','REVERSE']:
    source,target=(dcol,scol) if direction=='FORWARD' else (scol,dcol)
    try:
     df,d=fit_tvp(g,asset,cid,direction,source,target);d.update(channel_id=cid,asset=asset,direction=direction,source=source,target=target)
     diag.append(d)
     if df is not None:records.extend(df.to_dict('records'))
    except Exception as e:diag.append({'status':'STATE_SPACE_UNSTABLE','reason':str(e),'channel_id':cid,'asset':asset,'direction':direction,'source':source,'target':target})
 out=pd.DataFrame(records);out.to_csv(OUT/'stage3_tvp_coefficients.csv',index=False)
 pd.DataFrame(diag).to_csv(OUT/'stage3_tvp_diagnostics.csv',index=False)
 return out,pd.DataFrame(diag)

if __name__=='__main__':
 p=pd.read_csv(PROC/'derivatives_spot_daily_panel.csv',parse_dates=['date']);p.date=pd.to_datetime(p.date,utc=True);p=p[(p.date>='2021-12-01')&(p.date<='2024-03-03')].copy()
 r,e=run_rolling(p);print('ROLLING_DONE',len(r),flush=True)
 t,d=run_tvp(p);print('TVP_DONE',len(t),d.status.value_counts().to_dict(),flush=True)
