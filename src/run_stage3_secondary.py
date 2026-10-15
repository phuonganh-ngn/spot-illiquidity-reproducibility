#!/usr/bin/env python3
"""Run only after the clean-sample confirmatory estimates/classification are locked."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np,pandas as pd
from stage3_common import *
from run_stage3_dml import make_panel
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'


def main():
 panel=make_panel()
 cls=pd.read_csv(OUT/'channel_classification_summary.csv')
 if len(cls)!=8:raise RuntimeError('Primary classifications must be locked before secondary analyses')
 with (OUT/'hypothesis_registry.csv').open(newline='',encoding='utf-8') as f:old=list(csv.DictReader(f))
 v2=pd.read_csv(OUT/'primary_hypothesis_registry_v2.csv',dtype=str)
 secondary_ids=set(v2.loc[v2.status=='SECONDARY','test_id'])
 specs=[]
 for r in old:
  if r['test_id'] not in secondary_ids:continue
  h=int(float(r['horizon']));a=r['asset'];direction=r['direction'];der=r['derivative'];spot=r['spot']
  D=({'FUNDING':'funding_mean','OI':'oi_log_change_1d','FLOW':'futures_taker_imbalance'}[der] if direction=='FWD' else {'RV':'spot_realized_volatility','AMIHUD':'spot_amihud','DOWNSIDE':'spot_downside_semivariance'}[spot])
  Y=({'RV':f'spot_realized_volatility_fwd_{h}d','AMIHUD':f'spot_amihud_mean_fwd_{h}d','DOWNSIDE':f'spot_downside_semivariance_fwd_{h}d'}[spot] if direction=='FWD' else {'FUNDING':f'funding_mean_fwd_{h}d','OI':f'oi_log_change_fwd_{h}d','FLOW':f'futures_taker_imbalance_mean_fwd_{h}d'}[der])
  specs.append({'test_id':r['test_id'],'asset':a,'direction':'FORWARD' if direction=='FWD' else 'REVERSE','channel_pair':r['pair'],'exposure':D,'outcome':Y,'horizon':h,'analysis_type':'ORIGINAL_REGISTERED_SECONDARY','family_id':r['family_id']})
 # All originally registered extended horizons remain persistence/exploratory only.
 extended=[r for r in old if r['kind']=='DIRECTIONAL_DML' and r['priority']=='EXTENDED_HORIZON']
 ext_specs=[]
 for r in extended:
  h=int(float(r['horizon']));a=r['asset'];direction=r['direction'];der=r['derivative'];spot=r['spot']
  D=({'FUNDING':'funding_mean','OI':'oi_log_change_1d','FLOW':'futures_taker_imbalance'}[der] if direction=='FWD' else {'RV':'spot_realized_volatility','AMIHUD':'spot_amihud','DOWNSIDE':'spot_downside_semivariance'}[spot])
  Y=({'RV':f'spot_realized_volatility_fwd_{h}d','AMIHUD':f'spot_amihud_mean_fwd_{h}d','DOWNSIDE':f'spot_downside_semivariance_fwd_{h}d'}[spot] if direction=='FWD' else {'FUNDING':f'funding_mean_fwd_{h}d','OI':f'oi_log_change_fwd_{h}d','FLOW':f'futures_taker_imbalance_mean_fwd_{h}d'}[der])
  ext_specs.append({'test_id':r['test_id'],'asset':a,'direction':'FORWARD' if direction=='FWD' else 'REVERSE','channel_pair':r['pair'],'exposure':D,'outcome':Y,'horizon':h,'analysis_type':'EXTENDED_PERSISTENCE_ONLY','family_id':r['family_id']})
 # Frozen secondary variable extensions: funding sum and basis, each matched to spot RV.
 for asset in ['BTC','ETH']:
  for d,out,code in [('funding_sum','spot_realized_volatility','FUNDING_SUM'),('basis_mean','spot_realized_volatility','BASIS')]:
   for direction in ['FORWARD','REVERSE']:
    for h in [1,3,7,14,30]:
     D=d if direction=='FORWARD' else 'spot_realized_volatility'
     Y=(f'spot_realized_volatility_fwd_{h}d' if direction=='FORWARD' else (f'funding_cumulative_fwd_{h}d' if code=='FUNDING_SUM' else f'basis_mean_fwd_{h}d'))
     specs.append({'test_id':f'SEC_{asset}_{code}_{direction}_RV_H{h}','asset':asset,'direction':direction,'channel_pair':f'{code}<->RV','exposure':D,'outcome':Y,'horizon':h,'analysis_type':'SECONDARY_VARIABLE_EXPLORATORY','family_id':f'SEC_{asset}_{code}_{direction}'})

 def estimate(items,tag):
  rows=[]
  for i,s in enumerate(items,1):
   fit=fit_dml(panel,s['asset'],s['exposure'],s['outcome'],s['horizon'],learner='HGB',scheme='BLOCKED_4',embargo=int(s['horizon'])+8,require_primary=True)
   z=dict(fit.stats);z.update(s);z['analysis_label']='SECONDARY_NOT_PRIMARY_CLASSIFICATION'
   rows.append(z)
   if i%20==0:print(f'{tag} {i}/{len(items)}',flush=True)
  return pd.DataFrame(rows)
 sec=estimate(specs,'SECONDARY');ex=estimate(ext_specs,'EXTENDED_HORIZON')
 sec.to_csv(OUT/'table_stage3_secondary.csv',index=False)
 ex.to_csv(OUT/'table_stage3_extended_horizons.csv',index=False)
 print('SECONDARY_COMPLETE',len(sec),'EXTENDED_HORIZON_COMPLETE',len(ex),flush=True)

if __name__=='__main__':main()
