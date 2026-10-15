"""Audit immutable archive bytes and persist only validated numerical intermediates."""
import zipfile,io,json
import pandas as pd
import numpy as np
from common import *
ANOMALIES=[]
KCOL=['open_time','open','high','low','close','base_volume','close_time','quote_volume','number_of_trades','taker_buy_base_volume','taker_buy_quote_volume','ignore']
MCOL=['sum_open_interest','sum_open_interest_value','count_toptrader_long_short_ratio','sum_toptrader_long_short_ratio','count_long_short_ratio','sum_taker_long_short_vol_ratio']
def longest(mask):
 a=np.asarray(mask,dtype=bool);b=np.r_[False,a,False];edges=np.flatnonzero(b[1:]!=b[:-1]);return int((edges[1::2]-edges[::2]).max()) if len(edges) else 0

def load(row):
 p=ROOT/row.raw_file_path
 assert sha(p)==row.sha256,'RAW IMMUTABILITY FAIL'
 with zipfile.ZipFile(p) as z:
  assert z.testzip() is None,'ZIP CRC FAIL'; names=[n for n in z.namelist() if n.endswith('.csv')];assert len(names)==1
  data=z.read(names[0]);first=data.splitlines()[0].decode()
  header=not first.split(',')[0].replace('.','',1).isdigit()
  if row.dataset=='metrics':
   d=pd.read_csv(io.BytesIO(data));assert list(d.columns)==['create_time','symbol']+MCOL,'Unknown metrics schema'
   d['timestamp']=pd.to_datetime(d.create_time,utc=True,errors='coerce');unit='UTC string';cols=MCOL
  elif row.dataset=='fundingRate':
   d=pd.read_csv(io.BytesIO(data));assert list(d.columns)==['calc_time','funding_interval_hours','last_funding_rate'],'Unknown funding schema'
   d['timestamp']=pd.to_datetime(d.calc_time,unit='ms',utc=True,errors='coerce');unit='ms';cols=['last_funding_rate','funding_interval_hours']
  else:
   d=pd.read_csv(io.BytesIO(data),header=0 if header else None);assert d.shape[1]==12,'Unknown candle schema';d.columns=KCOL
   unit='us' if pd.to_numeric(d.open_time,errors='coerce').median()>1e14 else 'ms'
   d['timestamp']=pd.to_datetime(d.open_time,unit=unit,utc=True,errors='coerce')
   d['interval_end']=pd.to_datetime(d.close_time,unit=unit,utc=True,errors='coerce')
   if row.frequency=='5m':
    invalid=~(d.interval_end-d.timestamp).dt.total_seconds().between(299.99,300)
    for x in d.loc[invalid].itertuples(): ANOMALIES.append(dict(key=row.key,timestamp=str(x.timestamp),interval_end=str(x.interval_end),reason='Malformed candle interval; all analytical values invalidated, raw bytes preserved'))
    d.loc[invalid,[c for c in KCOL if c not in ['open_time','close_time']]]=np.nan
   if row.market=='spot':assert unit==('us' if row.file_period>='2025-01' else 'ms'),'Spot documented time-unit change mismatch'
   cols=KCOL[1:6]+KCOL[7:11] if row.dataset=='klines' else ['open','high','low','close']
  for c in cols:d[c]=pd.to_numeric(d[c],errors='coerce')
  d['_file_period']=row.file_period
  return d,cols,f'{"headered" if header else "unheadered"}:{8 if row.dataset=='metrics' else (3 if row.dataset=='fundingRate' else 12)}:{unit}'

def main():
 manifest=pd.read_csv(ROOT/'outputs/source_manifest.csv');manifest=manifest[manifest.dataset.isin(['klines','markPriceKlines','indexPriceKlines','premiumIndexKlines','fundingRate','metrics'])];rawaudit=[];tsaudit=[];fileaudit=[]
 for (symbol,market,dataset,freq),rows in manifest.groupby(['symbol','market','dataset','frequency'],sort=True):
  parts=[];schemas=[];numeric=None
  for r in rows.sort_values('file_period').itertuples():
   assert r.http_status==200,'Archive unavailable';d,cols,schema=load(r);numeric=cols
   exact=d.drop(columns=['_file_period']).duplicated().sum();dups=d.timestamp.duplicated().sum();delta=d.timestamp.sort_values().drop_duplicates().diff().dt.total_seconds()
   fileaudit.append(dict(symbol=symbol,market=market,dataset=dataset,frequency=freq,file_period=r.file_period,schema_version=schema,rows=len(d),unique_timestamps=d.timestamp.nunique(),duplicate_timestamps=int(dups),exact_duplicate_rows=int(exact),unordered=not d.timestamp.is_monotonic_increasing,first_timestamp=str(d.timestamp.min()),last_timestamp=str(d.timestamp.max()),median_interval=delta.median(),modulo_300_seconds=float((d.timestamp.astype('int64')/1e9%300).median()),catalogue_last_modified=r.catalogue_last_modified))
   schemas.append(schema);parts.append(d)
  frame=pd.concat(parts,ignore_index=True)
  # Identical repetitions only are collapsed. Conflicting timestamp groups are entirely invalidated.
  bare=[c for c in frame.columns if c!='_file_period'];dedup=frame.drop_duplicates(bare)
  conflict=dedup.timestamp.duplicated(keep=False);bad=dedup.loc[conflict,'timestamp'].drop_duplicates()
  clean=dedup.loc[~conflict].drop(columns=['_file_period']).set_index('timestamp').sort_index()
  clean=clean.loc[(clean.index>=pd.Timestamp(PRE,tz='UTC'))&(clean.index<pd.Timestamp(END,tz='UTC')+pd.Timedelta(days=1))]
  assert clean.index.is_unique
  tag=f'{symbol}_{market}_{dataset}_{freq}';clean.to_pickle(ROOT/'data/interim'/f'{tag}.pkl')
  for year in range(2020,2027):
   lo=max(pd.Timestamp(PRE,tz='UTC'),pd.Timestamp(f'{year}-01-01',tz='UTC'));hi=min(pd.Timestamp(END,tz='UTC')+pd.Timedelta(days=1),pd.Timestamp(f'{year+1}-01-01',tz='UTC'))
   if hi<=lo:continue
   base=clean[(clean.index>=lo)&(clean.index<hi)];original=frame[(frame.timestamp>=lo)&(frame.timestamp<hi)]
   step='8h' if dataset=='fundingRate' else ('1d' if freq=='1d' else '5min')
   grid=pd.date_range(lo,hi,freq=step,inclusive='left')
   if dataset=='fundingRate':
    assert (base.funding_interval_hours==8).all(),'Funding interval change requires new schedule audit'
    mapped=base.copy();mapped.index=mapped.index.floor('8h');assert mapped.index.is_unique,'Multiple settlements in one schedule slot'
   else:mapped=base
   for c in numeric:
    values=mapped[c].reindex(grid);valid=np.isfinite(values)
    impossible=(values<=0) if dataset!='premiumIndexKlines' and c in ['open','high','low','close','sum_open_interest','sum_open_interest_value','funding_interval_hours'] else ((values<0) if c!='last_funding_rate' and dataset!='premiumIndexKlines' else pd.Series(False,index=grid))
    good=valid&~impossible;actual=values[good]
    rawaudit.append(dict(asset=symbol[:-4],symbol=symbol,market=market,dataset=dataset,frequency=freq,year=year,metric=c,expected_observations=len(grid),actual_observations=len(original),valid_observations=int(good.sum()),coverage_pct=float(good.mean()*100),duplicate_timestamps=int(original.timestamp.duplicated().sum()),conflicting_timestamps=int(((bad>=lo)&(bad<hi)).sum()),missing_timestamps=int((~good).sum()),longest_missing_streak=longest(~good),nonfinite_values=int((~np.isfinite(base[c])).sum()),zero_values=int((base[c]==0).sum()),negative_values=int((base[c]<0).sum()),negative_impossible=int(impossible.sum()),first_observation=str(actual.index.min()) if len(actual) else '',last_observation=str(actual.index.max()) if len(actual) else ''))
   filepart=[x for x in fileaudit if x['symbol']==symbol and x['market']==market and x['dataset']==dataset and x['frequency']==freq and str(x['file_period']).startswith(str(year))]
   tsaudit.append(dict(source_dataset=f'{market}/{dataset}',symbol=symbol,field='timestamp',period=str(year),timestamp_column='create_time' if dataset=='metrics' else ('calc_time' if dataset=='fundingRate' else 'open_time / close_time'),semantic_definition='UNVERIFIED archive create_time to field measurement mapping; API OI says interval end' if dataset=='metrics' else ('archive calculation timestamp; checked against funding settlement API' if dataset=='fundingRate' else 'UTC candle start; close_time is inclusive interval end'),expected_interval=step,observed_interval_seconds=base.index.to_series().diff().dt.total_seconds().median(),offset_seconds=float(np.median(base.index.astype('int64')/1e9%300)) if len(base) else '',duplicate_count=int(original.timestamp.duplicated().sum()),missing_count=int(len(grid)-len(mapped.index.intersection(grid))),schema_version=';'.join(sorted(set(x['schema_version'] for x in filepart))),status='UNVERIFIED' if dataset=='metrics' else 'VERIFIED',notes='Spot microseconds from 2025; no unit inference alone. Exact repeated rows collapsed; conflicting duplicates removed. Sparse 1d files are validation-only.' if dataset!='metrics' else 'Inspect metrics_alignment_audit.csv before certification; no assumed June break.'))
  print('Audited',tag,'valid unique rows',len(clean),'conflicts',len(bad),flush=True)
 pd.DataFrame(ANOMALIES,columns=['key','timestamp','interval_end','reason']).to_csv(ROOT/'outputs/malformed_record_audit.csv',index=False)
 pd.DataFrame(rawaudit).to_csv(ROOT/'outputs/raw_coverage_audit.csv',index=False)
 pd.DataFrame(tsaudit).to_csv(ROOT/'outputs/timestamp_semantics_audit.csv',index=False)
 pd.DataFrame(fileaudit).to_csv(ROOT/'outputs/archive_file_schema_audit.csv',index=False)
if __name__=='__main__':main()
