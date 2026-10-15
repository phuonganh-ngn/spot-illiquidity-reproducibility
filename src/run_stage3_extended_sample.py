#!/usr/bin/env python3
"""Restricted robustness only on Stage-1 timestamp-harmonized extended data."""
from pathlib import Path
import numpy as np,pandas as pd
from stage3_extended_common import *
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';PROC=ROOT/'data/processed'
S=pd.Timestamp('2021-12-01',tz='UTC');E=pd.Timestamp('2026-09-30',tz='UTC')
CHANNELS={
 'C1':('funding_mean','spot_realized_volatility_fwd_{}d','spot_realized_volatility','funding_mean_fwd_{}d'),
 'C2':('oi_log_change_1d','spot_realized_volatility_fwd_{}d','spot_realized_volatility','oi_log_change_fwd_{}d'),
 'C3':('oi_log_change_1d','spot_downside_semivariance_fwd_{}d','spot_downside_semivariance','oi_log_change_fwd_{}d'),
 'C4':('futures_taker_imbalance','spot_amihud_mean_fwd_{}d','spot_amihud','futures_taker_imbalance_mean_fwd_{}d'),
}

def main():
 lock=pd.read_csv(OUT/'stage3_clean_analysis_lock_manifest.csv')
 if len(lock)<30:raise RuntimeError('Clean sample must be locked before extended sample')
 p=pd.read_csv(PROC/'derivatives_spot_extended_panel.csv',parse_dates=['date']);p.date=pd.to_datetime(p.date,utc=True)
 p=p[(p.date>=S)&(p.date<=E)].sort_values(['asset','date']).copy()
 expected=pd.date_range(S,E,freq='D')
 for a,g in p.groupby('asset'):
  if len(g)!=len(expected) or not pd.DatetimeIndex(g.date).equals(expected):raise RuntimeError(f'Extended sample calendar mismatch {a}')
 # Four date-contiguous blocks, constructed deterministically from the extended calendar.
 parts=np.array_split(expected,4);blocks=[(pd.Timestamp(z[0]),pd.Timestamp(z[-1])) for z in parts]
 rows=[]
 for cid,(dfwd,yfwd,drev,yrev) in CHANNELS.items():
  for asset in ['BTC','ETH']:
   for direction in ['FORWARD','REVERSE']:
    for h in [1,3,7]:
     D=dfwd;Y=yfwd.format(h) if direction=='FORWARD' else yrev.format(h)
     if direction=='REVERSE':D=drev
     fit=fit_dml(p,asset,D,Y,h,learner='HGB',scheme='BLOCKED_4',embargo=h+8,require_primary=True,sample_start=S,sample_end=E,predefined_blocks=blocks)
     rows.append({'test_id':f'EXT_{cid}_{asset}_{direction}_H{h}','sample_label':'EXTENDED_TIMESTAMP_HARMONIZED_SAMPLE','channel_id':cid,'asset':asset,'direction':direction,'exposure':D,'outcome':Y,'horizon':h,**fit.stats,'folds':json_folds(fit.folds)})
    if len(rows)%12==0:print(f'EXTENDED_CORE_TESTS {len(rows)}/48',flush=True)
 out=pd.DataFrame(rows)
 out.to_csv(OUT/'table_stage3_extended_sample.csv',index=False)
 print('EXTENDED_SAMPLE_COMPLETE',len(out),flush=True)

def json_folds(folds):
 import json
 return json.dumps([{k:(v.isoformat() if hasattr(v,'isoformat') else v) for k,v in r.items()} for r in folds],sort_keys=True)
if __name__=='__main__':main()
