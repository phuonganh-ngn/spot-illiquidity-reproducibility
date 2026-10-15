"""Descriptive timestamp/valuation checks; no economic direction models."""
import json
import pandas as pd
import numpy as np
from common import *
def read(sym,market,dataset,freq='5m'):return pd.read_pickle(ROOT/'data/interim'/f'{sym}_{market}_{dataset}_{freq}.pkl')
def main():
 alignment=[];cross=[];manifest=pd.read_csv(ROOT/'outputs/api_validation_manifest.csv')
 for sym in SYMBOLS:
  metrics=read(sym,'futures','metrics');k=read(sym,'futures','klines');mark=read(sym,'futures','markPriceKlines');index=read(sym,'futures','indexPriceKlines')
  buy=k.taker_buy_base_volume;sell=k.base_volume-buy;ratio=buy/sell.where(sell>0)
  implied=metrics.sum_open_interest_value/metrics.sum_open_interest
  for period,d in metrics.groupby(metrics.index.strftime('%Y-%m')):
   r=dict(symbol=sym,period=period,observations=len(d))
   for offset in [-300,0,300]:
    # offset identifies actual candle OPEN time relative to archive create_time.
    idx=d.index+pd.Timedelta(seconds=offset);observed=d.sum_taker_long_short_vol_ratio.to_numpy();reference=ratio.reindex(idx).to_numpy();valid=np.isfinite(observed)&np.isfinite(reference)
    r[f'flow_mae_offset_{offset}']=float(np.mean(np.abs(observed[valid]-reference[valid]))) if valid.any() else np.nan
    r[f'flow_median_relative_error_offset_{offset}']=float(np.median(np.abs(observed[valid]-reference[valid])/(np.abs(reference[valid])+1e-12))) if valid.any() else np.nan
    for dataset,frame in [('mark',mark),('index',index)]:
     ref=frame.open.reindex(idx).to_numpy();v=np.isfinite(ref)&np.isfinite(implied.reindex(d.index).to_numpy())
     r[f'oi_price_relative_error_{dataset}_offset_{offset}']=float(np.median(np.abs(implied.reindex(d.index).to_numpy()[v]/ref[v]-1))) if v.any() else np.nan
   r['best_flow_offset_seconds']=min([-300,0,300],key=lambda x:r[f'flow_median_relative_error_offset_{x}'] if np.isfinite(r[f'flow_median_relative_error_offset_{x}']) else np.inf)
   alignment.append(r)
  for rec in manifest[manifest.symbol==sym].itertuples():
   result=json.loads((ROOT/rec.raw_file_path).read_text()) if rec.http_status==200 else None
   if not isinstance(result,list):
    cross.append(dict(symbol=sym,dataset=rec.dataset,date=rec.file_period,status='UNVERIFIED',n_api=0,matched=0,notes=str(result)[:200]));continue
   a=pd.DataFrame(result)
   if a.empty:
    cross.append(dict(symbol=sym,dataset=rec.dataset,date=rec.file_period,status='UNAVAILABLE',n_api=0,matched=0));continue
   if rec.dataset=='funding_api_validation':
    f=read(sym,'futures','fundingRate','FUNDING_SETTLEMENT');a.index=pd.to_datetime(a.fundingTime,unit='ms',utc=True);v=pd.to_numeric(a.fundingRate);v.index=a.index;z=f.last_funding_rate.reindex(a.index)
    matched=int(np.isclose(v,z,atol=1e-12,rtol=1e-10).sum());cross.append(dict(symbol=sym,dataset=rec.dataset,date=rec.file_period,status='VERIFIED' if matched==len(a) else 'FAIL',n_api=len(a),matched=matched,notes='Exact fundingTime and calc_time mapping; rate equality.'))
   else:
    a.index=pd.to_datetime(a.timestamp,unit='ms',utc=True)
    for offset in [-300,0,300]:
     z=metrics.sum_open_interest.reindex(a.index+pd.Timedelta(seconds=offset)).to_numpy();v=pd.to_numeric(a.sumOpenInterest).to_numpy();good=np.isclose(z,v,atol=1e-8,rtol=1e-10);cross.append(dict(symbol=sym,dataset=rec.dataset,date=rec.file_period,offset_seconds=offset,status='VERIFIED' if good.sum()==len(a) else 'DISAGREEMENT',n_api=len(a),matched=int(good.sum()),notes='API timestamp documented period-end; archive create_time compared without assumed shift.'))
 pd.DataFrame(alignment).to_csv(ROOT/'outputs/metrics_alignment_audit.csv',index=False)
 pd.DataFrame(cross).to_csv(ROOT/'outputs/api_archive_crosscheck.csv',index=False)
if __name__=='__main__':main()
