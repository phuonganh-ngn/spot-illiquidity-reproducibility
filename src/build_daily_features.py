"""UTC features. Missing intervals stay missing; no imputation or estimation."""
import json
import numpy as np
import pandas as pd
from common import *
from audit_raw_data import longest
from validate_semantics import read
DAYS=pd.date_range(PRE,END,freq='D',tz='UTC')
DICTIONARY=[]
def entry(variable,source,aggregation,formula,unit,layer='derivatives',role='state_candidate',notes=''):
 DICTIONARY.append(dict(variable=variable,asset='BTC;ETH',market_layer=layer,raw_or_derived='derived',source_dataset=source,frequency_raw='5m unless funding settlement',aggregation=aggregation,formula=formula,unit=unit,timing='UTC day; no record after day end',available_at='next UTC midnight, subject to live-feed latency; archive publication is later',role_candidate=role,notes=notes))
def complete(frame,cols,expected=288):
 good=np.isfinite(frame[cols]).all(axis=1);counts=good.groupby(frame.index.floor('D')).sum().reindex(DAYS,fill_value=0)
 return counts==expected,counts

def candles(sym,market):
 k=read(sym,market,'klines');cols=['open','high','low','close','base_volume','quote_volume','number_of_trades','taker_buy_base_volume','taker_buy_quote_volume']
 v=k[cols].copy();bad=(v[['open','high','low','close']]<=0).any(axis=1)|(v[['base_volume','quote_volume','number_of_trades','taker_buy_base_volume','taker_buy_quote_volume']]<0).any(axis=1)|(v.taker_buy_base_volume>v.base_volume+1e-7)|(v.taker_buy_quote_volume>v.quote_volume+1e-6)
 v.loc[bad]=np.nan
 valid,n=complete(v,cols);g=v.groupby(v.index.floor('D'))
 daily=g.agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),base_volume=('base_volume','sum'),quote_volume=('quote_volume','sum'),number_of_trades=('number_of_trades','sum'),taker_buy_base_volume=('taker_buy_base_volume','sum'),taker_buy_quote_volume=('taker_buy_quote_volume','sum')).reindex(DAYS)
 daily.loc[~valid]=np.nan
 daily['intraday_valid_count']=n
 # Returns include previous 5m close: full grid prevents bridging outages.
 grid=pd.date_range(PRE,pd.Timestamp(END)+pd.Timedelta(days=1),freq='5min',inclusive='left',tz='UTC');close=v.close.reindex(grid)
 r=np.log(close/close.shift(1));rg=r.groupby(r.index.floor('D'));fullr=rg.count().reindex(DAYS)==288
 daily['realized_volatility']=np.sqrt(r.pow(2).groupby(r.index.floor('D')).sum(min_count=288)).reindex(DAYS).where(fullr&valid)
 daily['downside_semivariance']=r.clip(upper=0).pow(2).groupby(r.index.floor('D')).sum(min_count=288).reindex(DAYS).where(fullr&valid)
 daily['log_return_1d']=np.log(daily.close/daily.close.shift(1))
 daily['taker_imbalance']=(2*daily.taker_buy_base_volume-daily.base_volume)/daily.base_volume.where(daily.base_volume>0)
 daily['log_quote_volume']=np.log(daily.quote_volume.where(daily.quote_volume>0));daily['quote_volume_growth_1d']=daily.log_quote_volume.diff()
 daily['amihud']=1e6*daily.log_return_1d.abs()/daily.quote_volume.where(daily.quote_volume>0)
 daily['parkinson_volatility']=np.sqrt(np.log(daily.high/daily.low).pow(2)/(4*np.log(2)))
 daily['garman_klass_volatility']=np.sqrt((.5*np.log(daily.high/daily.low).pow(2)-(2*np.log(2)-1)*np.log(daily.close/daily.open).pow(2)).clip(lower=0))
 return daily.add_prefix(market+'_')

def main():
 out=[];checks=[];breaks=[]
 for sym in SYMBOLS:
  daily=candles(sym,'spot').join(candles(sym,'futures'));fund=read(sym,'futures','fundingRate','FUNDING_SETTLEMENT');rate=fund.last_funding_rate.where(np.isfinite(fund.last_funding_rate));g=rate.groupby(rate.index.floor('D'))
  f=g.agg(['mean','sum','last','max','min','count']).reindex(DAYS);f.loc[f['count']!=3,['mean','sum','last','max','min']]=np.nan
  for stat in ['mean','sum','last','max','min']:daily['funding_'+stat]=f[stat]
  daily['funding_abs_mean']=rate.abs().groupby(rate.index.floor('D')).mean().reindex(DAYS).where(f['count']==3)
  daily['funding_positive_fraction']=(rate>0).where(rate.notna()).groupby(rate.index.floor('D')).mean().reindex(DAYS).where(f['count']==3)
  daily['funding_count']=f['count']
  m=read(sym,'futures','metrics');m=m[~m.index.isna()].copy()
  layout=pd.read_csv(ROOT/'outputs/metrics_daily_layout_audit.csv');layout=layout[layout.symbol==sym].set_index('date')
  offsets=layout.stock_effective_offset_seconds.reindex(m.index.strftime('%Y-%m-%d')).to_numpy()
  verified=np.isfinite(offsets);m=m.loc[verified].copy();m.index=m.index+pd.to_timedelta(offsets[verified],unit='s')
  # Conflicting observations after mapping are never chosen arbitrarily.
  m=m.loc[~m.index.duplicated(keep=False)].sort_index()
  # OI snapshots use empirically verified field times; no unknown-layout day is retained.
  # This is historical reconstruction; exact historical provider publication times are unavailable.
  safe=m[(m.index>=pd.Timestamp(PRE,tz='UTC'))&(m.index<pd.Timestamp(END,tz='UTC')+pd.Timedelta(days=1))];good=(safe.sum_open_interest>0)&(safe.sum_open_interest_value>0)
  oi=safe[['sum_open_interest','sum_open_interest_value']].where(good);gc=oi.groupby(oi.index.floor('D'));n=gc.count().reindex(DAYS)
  for raw,prefix in [('sum_open_interest','oi'),('sum_open_interest_value','oi_value')]:
   a=gc[raw].agg(['last','mean','max','min']).reindex(DAYS).where(n[raw]>=274) # >=95% of287 safe snapshots
   for stat in ['last','mean','max','min']:daily[prefix+('_close' if stat=='last' else '_'+stat)]=a[stat]
  daily['oi_valid_snapshots']=n.sum_open_interest;daily['oi_log_change_1d']=np.log(daily.oi_close/daily.oi_close.shift(1));daily['log_oi']=np.log(daily.oi_close);daily['oi_value_log_change_1d']=np.log(daily.oi_value_close/daily.oi_value_close.shift(1))
  for raw,var in [('count_long_short_ratio','global_account_long_short_ratio'),('count_toptrader_long_short_ratio','top_trader_account_long_short_ratio'),('sum_toptrader_long_short_ratio','top_trader_position_long_short_ratio'),('sum_taker_long_short_vol_ratio','metrics_taker_buy_sell_ratio')]:
   x=safe[raw].where(np.isfinite(safe[raw])&(safe[raw]>=0));mg=x.groupby(x.index.floor('D'));daily[var]=mg.last().reindex(DAYS).where(mg.count().reindex(DAYS)>=274)
  mark=read(sym,'futures','markPriceKlines').close;index=read(sym,'futures','indexPriceKlines').close
  grid=pd.date_range(PRE,pd.Timestamp(END)+pd.Timedelta(days=1),freq='5min',inclusive='left',tz='UTC');basis=(mark.reindex(grid)/index.reindex(grid).where(index.reindex(grid)>0)-1);basis=basis.where(np.isfinite(basis));bg=basis.groupby(basis.index.floor('D'));valid=bg.count().reindex(DAYS)==288
  for stat in ['mean','last','std']:daily['basis_'+('close' if stat=='last' else stat)]=bg.agg(stat).reindex(DAYS).where(valid)
  daily['basis_abs_mean']=basis.abs().groupby(basis.index.floor('D')).mean().reindex(DAYS).where(valid)
  premium=read(sym,'futures','premiumIndexKlines').close;pg=premium.groupby(premium.index.floor('D'));daily['premium_index_mean']=pg.mean().reindex(DAYS).where(pg.count().reindex(DAYS)==288)
  daily['metrics_timing_certified']=daily.oi_close.notna() # Independently verified layout and >=95% safe observations required.
  daily['available_at_utc']=DAYS+pd.Timedelta(days=1)
  daily['timing_reference']='outputs/timestamp_semantics_audit.csv'
  cross=read(sym,'spot','klines','1d')
  main_dates=pd.date_range(START,END,tz='UTC');sample={main_dates[int((len(main_dates)-1)*q)] for q in [0,.25,.5,.75,1]}|{pd.Timestamp(x,tz='UTC') for x in ['2024-12-31','2025-01-01','2026-06-24','2026-06-25']}
  for day in sorted(sample):
   if day not in cross.index:continue
   for col in ['open','high','low','close','base_volume','quote_volume','number_of_trades']:
    a=daily.loc[day,'spot_'+col];b=cross.loc[day,col];checks.append(dict(asset=sym[:-4],date=str(day.date()),field=col,reconstructed=a,provider_daily=b,absolute_difference=abs(a-b),status='PASS' if np.isclose(a,b,rtol=1e-9,atol=1e-7) else 'FAIL'))
  change=daily.oi_log_change_1d
  for day in change[change.abs()>.15].index:
   breaks.append(dict(asset=sym[:-4],date=str(day.date()),oi_log_change_1d=change.loc[day],status='UNVERIFIED',classification='FLAGGED_NOT_DELETED',reason='Absolute log quantity change >0.15; does not establish economic event or mechanical unit break',evidence='metrics_alignment_audit.csv; archive_file_schema_audit.csv'))
  daily['asset']=sym[:-4];daily.index.name='date';out.append(daily.reset_index())
 pd.concat(out).sort_values(['date','asset']).to_pickle(ROOT/'data/interim/daily_features.pkl')
 pd.DataFrame(checks).to_csv(ROOT/'outputs/daily_price_crosscheck.csv',index=False)
 pd.DataFrame(breaks,columns=['asset','date','oi_log_change_1d','status','classification','reason','evidence']).to_csv(ROOT/'outputs/oi_structural_break_audit.csv',index=False)
 # Explicit base-feature provenance for every analytical numerical feature.
 for col in out[0].columns:
  if col in ['date','asset','available_at_utc','timing_reference','metrics_timing_certified']:continue
  layer='spot' if col.startswith('spot_') else 'derivatives';source='spot/klines' if layer=='spot' else ('futures/fundingRate' if col.startswith('funding_') else ('futures/metrics' if col.startswith(('oi','log_oi','global_','top_','metrics_')) else ('futures/markPriceKlines + indexPriceKlines' if col.startswith('basis_') else ('futures/premiumIndexKlines' if col.startswith('premium_') else 'futures/klines'))))
  formulas={
   'intraday_valid_count':'count of five-minute rows with all required finite, economically valid candle fields in UTC day (out of288)', 'log_return_1d':'log(close[t]/close[t-1])','realized_volatility':'sqrt(sum_{288 5m close returns}(log(close[j]/close[j-1])^2))','downside_semivariance':'sum_{288}(min(log(close[j]/close[j-1]),0)^2)','taker_imbalance':'(2*sum(taker_buy_base_volume)-sum(base_volume))/sum(base_volume); denominator>0','amihud':'1e6*abs(log(close[t]/close[t-1]))/sum(quote_volume)','log_quote_volume':'log(sum(quote_volume)); quote_volume>0','quote_volume_growth_1d':'log_quote_volume[t]-log_quote_volume[t-1]','parkinson_volatility':'sqrt(log(high/low)^2/(4*log(2)))','garman_klass_volatility':'sqrt(max(0,0.5*log(high/low)^2-(2*log(2)-1)*log(close/open)^2))',
   'oi_log_change_1d':'log(oi_close[t]/oi_close[t-1])','oi_value_log_change_1d':'log(oi_value_close[t]/oi_value_close[t-1])','log_oi':'log(oi_close)','oi_valid_snapshots':'count of positive finite quantity and notional OI snapshots after field-time alignment in UTC day (out of288)',
   'funding_abs_mean':'mean(abs(last_funding_rate))','funding_positive_fraction':'mean(last_funding_rate>0)',
  }
  stem=col.removeprefix('spot_').removeprefix('futures_');formula=formulas.get(col,formulas.get(stem,''))
  if not formula:
   if col.startswith('basis_'):formula={'basis_mean':'mean(mark.close/index.close-1)','basis_close':'last(mark.close/index.close-1)','basis_std':'sample_sd(mark.close/index.close-1)','basis_abs_mean':'mean(abs(mark.close/index.close-1))'}[col]
   elif col.startswith('funding_'):formula=f'{col.removeprefix("funding_")}(last_funding_rate by actual UTC calc_time); exactly 3 settlements'
   elif col.startswith('oi_'):formula=col.removeprefix('oi_value_').removeprefix('oi_')+'(positive OI safe snapshots; field-specific observed layout; effective snapshot before day end; >=274/288 snapshots)'
   elif col=='premium_index_mean':formula='mean(premiumIndexKlines.close); full288 bars'
   elif col.endswith('ratio'):formula='last nonnegative '+col+' archive field among safe snapshots; >=274/288'
   elif stem=='open':formula='first intraday open'
   elif stem=='close':formula='last intraday close'
   elif stem=='high':formula='max intraday high'
   elif stem=='low':formula='min intraday low'
   else:formula='sum valid intraday '+stem+'; all288 required (count variables retain observed counts)'
  unit='USDT' if ('quote_volume' in col or 'oi_value' in col) and 'growth' not in col and 'log' not in col else ('base-asset units' if col in ['oi_close','oi_mean','oi_max','oi_min'] or 'base_volume' in col else ('1e6/USDT' if col.endswith('amihud') else 'dimensionless unless count/trades'))
  entry(col,source,'UTC daily',formula,unit,layer,notes='OI field time is empirically aligned via mark-price valuation and API quantity validation; ratios remain SECONDARY and their archive-field mapping is unverified.' if source=='futures/metrics' else 'Complete-day rule; no imputation. Quote USDT is not assumed USD.')
 pd.DataFrame(DICTIONARY).to_csv(ROOT/'outputs/variable_dictionary.csv',index=False)
if __name__=='__main__':main()
