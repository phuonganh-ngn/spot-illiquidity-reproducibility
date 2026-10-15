from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs'/'targeted_24h_amihud'
P7=ROOT/'data'/'processed'/'phase7'/'market_liquidity_panel_4h.csv.gz'
P5=ROOT/'outputs'/'phase5_dynamic_discovery'/'dataset_4h.jsonl'
EXPECTED={'phase7':'5a339c5c4a8778ceb74344a10c67806c2323ffac80556be18bf7a9c99de8dcf6','phase5':'e50637340052f505c585bbce70c86ac7f4d3f607b6ef127a24c1317d1d99402d'}
START=pd.Timestamp('2021-12-01T00:00:00Z')
END=pd.Timestamp('2026-09-29T20:00:00Z')
ASSETS=['BTC','ETH']
SPOT=['spot_return','spot_realized_volatility','spot_downside_semivariance','spot_amihud','spot_taker_imbalance','spot_quote_volume_growth']
DER=['funding_rate_state','oi_log_change','futures_taker_imbalance','basis_mean']

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def writej(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
def finite_stats(x):
 a=np.asarray(x,dtype=float); a=a[np.isfinite(a)]
 if not len(a): return {'n':0}
 q=[.001,.01,.05,.25,.5,.75,.95,.99,.999]
 return {'n':int(len(a)),'mean':float(a.mean()),'std':float(a.std(ddof=1)) if len(a)>1 else None,'min':float(a.min()),'max':float(a.max()),'quantiles':{str(p):float(v) for p,v in zip(q,np.quantile(a,q))}}
def max_streak(mask):
 best=cur=0
 for x in mask:
  cur=cur+1 if x else 0; best=max(best,cur)
 return int(best)

OUT.mkdir(parents=True,exist_ok=True)
source_hashes={'phase7':sha(P7),'phase5':sha(P5)}
if source_hashes!=EXPECTED: raise RuntimeError(f'Certified input hash changed: {source_hashes}')
raw=pd.read_csv(P7,parse_dates=['period_start_utc','future_window_start_utc','future_window_end_utc_exclusive'],low_memory=False)
raw['period_start_utc']=pd.to_datetime(raw.period_start_utc,utc=True)
raw=raw[raw.asset.isin(ASSETS)].copy().sort_values(['asset','period_start_utc']).reset_index(drop=True)
raw=raw[(raw.period_start_utc>=START)&(raw.period_start_utc<=END)].copy()
keydup=int(raw.duplicated(['asset','period_start_utc']).sum())
# Independently reproduce target and Amihud from certified ingredients.
formula_amihud=1e6*raw.spot_return.abs()/raw.spot_quote_volume.where(raw.spot_quote_volume>0)
amihud_delta=(formula_amihud-raw.spot_amihud).abs()
rebuilt=[]
for asset,g in raw.groupby('asset',sort=False):
 idx=g.index
 fut=pd.concat([g.spot_amihud.shift(-k).rename(str(k)) for k in range(1,7)],axis=1)
 y=fut.mean(axis=1).where(fut.notna().all(axis=1))
 rebuilt.append(pd.Series(y.to_numpy(),index=idx))
raw['future_amihud_rebuilt']=pd.concat(rebuilt).sort_index()
target_delta=(raw.future_amihud_rebuilt-raw.future_amihud_24h_mean).abs()
window_start=raw.future_window_start_utc-raw.period_start_utc
window_end=raw.future_window_end_utc_exclusive-raw.period_start_utc
# Derived derivative features: strict past-only median/MAD.
raw['funding_change']=raw.groupby('asset',sort=False).funding_rate_state.diff()
raw['basis_change']=raw.groupby('asset',sort=False).basis_mean.diff()
for source,name in [('oi_log_change','oi_shock_30d_robust'),('futures_taker_imbalance','flow_shock_30d_robust')]:
 values=[]
 for _,g in raw.groupby('asset',sort=False):
  past=g[source].shift(1)
  med=past.rolling(180,min_periods=180).median()
  mad=past.rolling(180,min_periods=180).apply(lambda z:float(np.median(np.abs(z-np.median(z)))),raw=True)
  values.append(((g[source]-med)/(1.4826*mad.where(mad>0))).rename(name))
 raw[name]=pd.concat(values).sort_index()
raw['funding_x_oi_shock']=raw.funding_rate_state*raw.oi_shock_30d_robust
raw['oi_shock_x_flow_shock']=raw.oi_shock_30d_robust*raw.flow_shock_30d_robust
raw['funding_x_basis']=raw.funding_rate_state*raw.basis_mean
raw['flow_x_basis']=raw.futures_taker_imbalance*raw.basis_mean
slot=raw.period_start_utc.dt.hour//4; weekday=raw.period_start_utc.dt.dayofweek
for i in range(1,6): raw[f'utc_slot_{i}']=(slot==i).astype('int8')
for i in range(1,7): raw[f'weekday_{i}']=(weekday==i).astype('int8')
raw['funding_schedule_window']=raw.period_start_utc.dt.hour.isin([0,8,16]).astype('int8')

# Persist the exact, deterministic analysis panel in the new namespace.
raw.to_csv(OUT/'analysis_panel.csv.gz',index=False,compression={'method':'gzip','mtime':0},date_format='%Y-%m-%dT%H:%M:%SZ')

per_asset={}; coverage={}
for asset,g in raw.groupby('asset',sort=False):
 g=g.sort_values('period_start_utc'); dates=g.period_start_utc
 expected=len(pd.date_range(dates.min(),dates.max(),freq='4h'))
 miss=~g[['spot_return','spot_realized_volatility','spot_downside_semivariance','spot_amihud','spot_taker_imbalance','spot_quote_volume_growth',*DER,'future_amihud_24h_mean']].notna().all(axis=1).to_numpy()
 coverage[asset]={'first_interval':dates.min().isoformat(),'last_interval':dates.max().isoformat(),'expected_grid_intervals':expected,'observed_intervals':int(len(g)),'coverage_grid_pct':100*len(g)/expected if expected else None,'duplicate_asset_date_rows':int(g.duplicated(['asset','period_start_utc']).sum()),'missing_required_rows':int(miss.sum()),'complete_case_rows':int((~miss).sum()),'longest_streak_with_any_required_missing':max_streak(miss)}
 y=g.future_amihud_24h_mean.dropna().to_numpy(float)
 x=g.spot_amihud.to_numpy(float); qv=g.spot_quote_volume.to_numpy(float)
 qpositive=qv[np.isfinite(qv)&(qv>0)]
 log1p=np.log1p(y)
 per_asset[asset]={
  'spot_amihud_formula':'1e6 * abs(spot_return) / spot_quote_volume, where spot_quote_volume > 0; quote-volume units are USDT per 4h Binance spot interval.',
  'spot_amihud':finite_stats(x),'future_target':finite_stats(y),
  'spot_return':finite_stats(g.spot_return),'spot_quote_volume_positive':finite_stats(qpositive),
  'missing_spot_amihud':int(np.isnan(x).sum()),'zero_spot_amihud':int(np.sum(x==0)),'negative_spot_amihud':int(np.sum(x<0)),
  'zero_quote_volume':int(np.sum(np.isfinite(qv)&(qv==0))),'negative_quote_volume':int(np.sum(np.isfinite(qv)&(qv<0))),
  'quote_volume_le_1_usdt':int(np.sum(np.isfinite(qv)&(qv>0)&(qv<=1))),'quote_volume_le_1000_usdt':int(np.sum(np.isfinite(qv)&(qv>0)&(qv<=1000))),
  'near_zero_denominator_min_positive':float(qpositive.min()) if len(qpositive) else None,
  'target_zero_count':int(np.sum(y==0)),'target_negative_count':int(np.sum(y<0)),
  'log1p_scale_diagnostic':{'max_abs_difference_log1p_vs_raw':float(np.max(np.abs(log1p-y))),'max_relative_difference':float(np.max(np.abs(log1p-y)/np.maximum(np.abs(y),1e-30))),'median_relative_difference':float(np.median(np.abs(log1p-y)/np.maximum(np.abs(y),1e-30))),'interpretation':'log1p is numerically almost identity on this target scale; it does not materially compress the upper tail.'},
  'largest_10_target_observations':[{'date':d.isoformat(),'value':float(v)} for d,v in zip(g.loc[g.future_amihud_24h_mean.nlargest(10).index,'period_start_utc'],g.future_amihud_24h_mean.nlargest(10).to_numpy())],
 }

# Cross-source parity on overlapping BTC/ETH dates.
p5=pd.read_json(P5,lines=True)
p5['period_start_utc']=pd.to_datetime(p5.period_start_utc,utc=True)
common=raw[['asset','period_start_utc','spot_amihud','spot_quote_volume','spot_return']].merge(p5[['asset','period_start_utc','spot_amihud','spot_raw_quote_volume','spot_return']],on=['asset','period_start_utc'],suffixes=('_p7','_p5'),how='inner')
parity={}
for v,left,right in [('spot_amihud','spot_amihud_p7','spot_amihud_p5'),('spot_return','spot_return_p7','spot_return_p5'),('spot_quote_volume','spot_quote_volume','spot_raw_quote_volume')]:
 z=(common[left]-common[right]).abs().to_numpy(float)
 parity[v]={'overlap_rows':int(len(common)),'max_abs_difference':float(np.nanmax(z)),'median_abs_difference':float(np.nanmedian(z))}

# Data period roles keep previously inspected dates explicit.
roles={
 'development':{'start':'2021-12-01T00:00:00Z','end':'2023-06-30T20:00:00Z','status':'previously used in Stage 6 development; nested reanalysis, not pristine'},
 'phase6_reuse':{'start':'2023-07-01T00:00:00Z','end':'2024-03-03T20:00:00Z','status':'previously inspected; exploratory reuse'},
 'phase7_reuse':{'start':'2024-03-04T00:00:00Z','end':'2026-09-29T20:00:00Z','status':'previously inspected; exploratory reuse'},
}
validation={
 'status':'PASS' if keydup==0 and bool((amihud_delta.dropna()<1e-12).all()) and bool((target_delta.dropna()<1e-12).all()) and raw.future_window_start_utc.sub(raw.period_start_utc).eq(pd.Timedelta(hours=4)).all() and raw.future_window_end_utc_exclusive.sub(raw.period_start_utc).eq(pd.Timedelta(hours=28)).all() else 'FAIL',
 'source_hashes':source_hashes,'expected_source_hashes':EXPECTED,'source_hashes_pass':source_hashes==EXPECTED,
 'panel_rows':int(len(raw)),'asset_rows':coverage,'asset_date_duplicates':keydup,'utc_step_violations':int((raw.groupby('asset').period_start_utc.diff().dropna()!=pd.Timedelta(hours=4)).sum()),
 'amihud_recalculation_max_abs_error':float(amihud_delta.max()),'future_target_recalculation_max_abs_error':float(target_delta.max()),
 'future_window_start_offset_unique_hours':sorted(set(window_start.dropna().dt.total_seconds().div(3600).tolist())),
 'future_window_end_exclusive_offset_unique_hours':sorted(set(window_end.dropna().dt.total_seconds().div(3600).tolist())),
 'negative_target_count':int((raw.future_amihud_24h_mean<0).sum()),'nonfinite_target_count':int((~np.isfinite(raw.future_amihud_24h_mean)).sum()),
 'sample_roles':roles,'matched_information_set_fields':True,'fresh_external_confirmation_available':False,
 'no_market_data_downloaded':True,'analysis_panel_sha256':sha(OUT/'analysis_panel.csv.gz')
}
writej('data_validation.json',validation)
audit={
 'formula':{'numerator':'absolute four-hour log return computed from spot close-to-close price ratio for interval t','denominator':'four-hour spot quote volume in USDT for the same interval','scale_factor':1000000,'unit':'1e6 * absolute log return / USDT quote notional','zero_denominator':'returns missing when quote volume <= 0; no replacement or epsilon in raw Amihud','near_zero_denominator':'small positive quote volume can create large Amihud; preserved, no clipping or winsorization'},
 'target':{'name':'future_amihud_24h_mean','aggregation':'arithmetic mean of six four-hour Amihud observations at t+1...t+6','future_window_start':'t+4 hours','future_window_end_exclusive':'t+28 hours','requires_six_valid_future_intervals':True},
 'timestamp_alignment':'period_start_utc identifies interval start; interval-t aggregates become available at interval end t+4h; label starts at the next interval t+4h. This is the certified Phase-6/7 operational alignment; no same-interval label overlap.',
 'cross_source_parity_phase5':parity,
 'assets':per_asset,
 'pooled_target_max':float(raw.future_amihud_24h_mean.max()),
 'pooled_target_log1p_max_relative_difference':float(np.max(np.abs(np.log1p(raw.future_amihud_24h_mean.dropna())-raw.future_amihud_24h_mean.dropna())/raw.future_amihud_24h_mean.dropna())),
 'structural_notes':['Negative Amihud is impossible; none should be present.','Zero Amihud is valid when the interval absolute return is zero and quote volume is positive.','Log1p is nearly identity at the observed small numerical scale, so transform candidates include explicit log scaling based only on the fixed Amihud units.','No tails are removed; high values remain part of the dataset.']
}
writej('amihud_audit.json',audit)
print(json.dumps({'data_status':validation['status'],'assets':coverage,'amihud_formula_max_error':validation['amihud_recalculation_max_abs_error'],'target_rebuild_max_error':validation['future_target_recalculation_max_abs_error'],'log1p_max_relative_diff':audit['pooled_target_log1p_max_relative_difference'],'analysis_panel_sha256':validation['analysis_panel_sha256']},indent=2))
