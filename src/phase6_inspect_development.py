#!/usr/bin/env python3
"""Read schema and development-period values only for Phase-6 design."""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/phase5_dynamic_discovery/dataset_4h.jsonl'
OUT=ROOT/'outputs/phase6_autonomous_benchmark'
SPLIT=pd.Timestamp('2023-07-01T00:00:00Z')
FEATURES={
 'spot':['spot_return','spot_realized_volatility','spot_downside_semivariance','spot_amihud','spot_taker_imbalance','spot_quote_volume_growth'],
 'derivatives':['funding_rate_state','oi_log_change','futures_taker_imbalance','basis_mean'],
}
ALL=sum((FEATURES['spot'],FEATURES['derivatives']),[])
TS_RE=re.compile(r'"period_start_utc":"([^"]+)"')

def make_samples(g, horizons=(1,2,6)):
    g=g.sort_values('period_start_utc').reset_index(drop=True)
    values=g[ALL].to_numpy(float)
    targets={}
    for h in horizons:
        rows=[]
        for t in range(5,len(g)-h):
            x=values[t-5:t+1]
            future=g.iloc[t+1:t+h+1]
            if not np.isfinite(x).all(): continue
            if not np.isfinite(future[['spot_realized_volatility','spot_downside_semivariance','spot_amihud']].to_numpy(float)).all(): continue
            target_rv=float(np.sqrt(np.square(future.spot_realized_volatility.to_numpy(float)).sum()))
            target_down=float(future.spot_downside_semivariance.to_numpy(float).sum())
            target_amihud=float(future.spot_amihud.to_numpy(float).mean())
            rows.append({'t':t,'date':g.period_start_utc.iloc[t],
                         'future_end':g.period_start_utc.iloc[t+h],
                         'future_rv':target_rv,'future_downside':target_down,'future_amihud':target_amihud})
        targets[h]=pd.DataFrame(rows)
    return targets

def main():
    raw_hash=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    dev=[]; total_lines=0; test_date_rows=0; keys=set(); schema=None
    with SOURCE.open(encoding='utf-8') as f:
        for line in f:
            total_lines+=1
            m=TS_RE.search(line)
            if not m: raise RuntimeError(f'missing timestamp at JSONL line {total_lines}')
            ts=pd.Timestamp(m.group(1))
            if ts < SPLIT:
                row=json.loads(line)
                schema=sorted(row) if schema is None else schema
                key=(row['asset'],row['period_start_utc'])
                if key in keys: raise RuntimeError(f'duplicate key {key}')
                keys.add(key)
                if not row.get('discovery_sample'): raise RuntimeError('development row mislabeled')
                dev.append(row)
            else:
                # Deliberately do not parse any final-test feature/outcome values.
                test_date_rows+=1
    d=pd.DataFrame(dev)
    d['period_start_utc']=pd.to_datetime(d.period_start_utc,utc=True)
    expected={'BTC':3462,'ETH':3462}
    counts=d.groupby('asset').size().to_dict()
    if counts!=expected: raise RuntimeError(f'unexpected development rows: {counts}')
    if total_lines!=9888 or test_date_rows!=2964: raise RuntimeError(f'unexpected full panel line count {total_lines}, holdout rows {test_date_rows}')
    summary={
      'status':'PASS','input_path':str(SOURCE.relative_to(ROOT)),'input_sha256':raw_hash,
      'only_development_values_parsed':True,'development_period':{'start':'2021-12-01T00:00:00Z','end_inclusive':'2023-06-30T20:00:00Z'},
      'holdout_period':{'start':'2023-07-01T00:00:00Z','end_inclusive':'2024-03-03T20:00:00Z'},
      'schema':schema,'frozen_feature_groups':FEATURES,'rows_read':int(total_lines),
      'development_rows':{k:int(v) for k,v in counts.items()},'holdout_rows_counted_without_value_parsing':int(test_date_rows),
      'usable_samples':{},'development_feature_statistics':{},'development_target_statistics':{},
      'target_characteristics':{},
    }
    for asset,g in d.groupby('asset'):
        fg=g[ALL].apply(pd.to_numeric,errors='coerce')
        summary['development_feature_statistics'][asset]={}
        for c in ALL:
            x=fg[c].to_numpy(float); x=x[np.isfinite(x)]
            summary['development_feature_statistics'][asset][c]={
              'valid':int(len(x)),'missing':int(len(g)-len(x)),
              'mean':float(x.mean()) if len(x) else None,'sd':float(x.std(ddof=1)) if len(x)>1 else None,
              'p01':float(np.quantile(x,.01)) if len(x) else None,'median':float(np.quantile(x,.5)) if len(x) else None,
              'p99':float(np.quantile(x,.99)) if len(x) else None,
            }
        samples=make_samples(g)
        for h,task in samples.items():
            summary['usable_samples'][f'{asset}_h{h}']={'n':int(len(task)),
              'first_feature_time':task.date.min().isoformat() if len(task) else None,
              'last_feature_time':task.date.max().isoformat() if len(task) else None,
              'last_target_end':task.future_end.max().isoformat() if len(task) else None}
            summary['development_target_statistics'][f'{asset}_h{h}']={}
            for col in ['future_rv','future_downside','future_amihud']:
                x=task[col].to_numpy(float)
                summary['development_target_statistics'][f'{asset}_h{h}'][col]={
                  'mean':float(x.mean()),'sd':float(x.std(ddof=1)),'p50':float(np.quantile(x,.5)),
                  'p90':float(np.quantile(x,.9)),'p99':float(np.quantile(x,.99)),
                  'positive_fraction':float((x>0).mean()),'max':float(x.max())}
                summary['target_characteristics'][col]=summary['target_characteristics'].get(col,{'all_horizons_nonnegative':True})
                summary['target_characteristics'][col]['all_horizons_nonnegative'] &= bool((x>=0).all())
    (OUT/'development_inspection.json').write_text(json.dumps(summary,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n')
    print(f"PHASE6_DEV_INSPECTION=PASS rows={len(d)} test_values_parsed=0 input_sha256={raw_hash}")

if __name__=='__main__': main()
