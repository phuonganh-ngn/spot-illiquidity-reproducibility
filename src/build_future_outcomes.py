"""Calendar-exact lags and strictly future windows; no missing-value filling."""
import pandas as pd,numpy as np,json
from common import *
HORIZONS=[1,3,7,14,30]
CORE=['spot_log_return_1d','spot_realized_volatility','spot_quote_volume','funding_mean','oi_close','futures_taker_imbalance']
def future(s,h,how):
 # Date index is reindexed before use. Every target must have all h observations.
 a=pd.concat([s.shift(-j) for j in range(1,h+1)],axis=1);n=a.notna().sum(axis=1);full=n==h
 if how=='sum':v=a.sum(axis=1,min_count=h)
 elif how=='rms':v=np.sqrt(a.pow(2).sum(axis=1,min_count=h))
 else:v=a.mean(axis=1).where(full)
 return v.where(full)
def main():
 frame=pd.read_pickle(ROOT/'data/interim/daily_features.pkl');dictionary=pd.read_csv(ROOT/'outputs/variable_dictionary.csv');records=dictionary.to_dict('records');parts=[]
 for asset,g in frame.groupby('asset',sort=True):
  g=g.set_index('date').sort_index();assert g.index.to_series().diff().dropna().eq(pd.Timedelta(days=1)).all()
  lagset={'spot_log_return_1d':[1,2,7],'spot_realized_volatility':[1,7],'spot_amihud':[1],'spot_downside_semivariance':[1],'spot_quote_volume_growth_1d':[1],'spot_taker_imbalance':[1],'funding_mean':[1,2],'oi_log_change_1d':[1],'basis_mean':[1],'futures_taker_imbalance':[1]}
  for col,lags in lagset.items():
   for lag in lags:
    name=f'{col}_lag{lag}';g[name]=g[col].shift(lag)
    if asset=='BTC':records.append(dict(variable=name,asset='BTC;ETH',market_layer='controls',raw_or_derived='derived',source_dataset=col,frequency_raw='daily',aggregation='calendar shift',formula=f'{col}[t-{lag}]',unit='same as source',timing=f'ends t-{lag}',available_at='by end t subject to source timing caveat',role_candidate='predetermined_control',notes='No contemporaneous or future outcomes'))
  for window in [7,30]:
   for col in ['funding_mean','spot_log_return_1d']:
    name=f'{col}_rolling_{window}d_lag1';g[name]=(g[col].rolling(window,min_periods=window).mean() if col=='funding_mean' else g[col].rolling(window,min_periods=window).std(ddof=1)).shift(1)
    if asset=='BTC':records.append(dict(variable=name,asset='BTC;ETH',market_layer='controls',raw_or_derived='derived',source_dataset=col,frequency_raw='daily',aggregation='backward rolling then lag1',formula=f'{"mean" if col=="funding_mean" else "sample_sd"}({col}[t-{window}:t-1])',unit='same as source',timing='through t-1',available_at='by end t subject to source timing caveat',role_candidate='predetermined_control',notes='No centered windows; no full-sample normalization'))
   name=f'oi_growth_{window}d';g[name]=np.log(g.oi_close/g.oi_close.shift(window))
   if asset=='BTC':records.append(dict(variable=name,asset='BTC;ETH',market_layer='derivatives',raw_or_derived='derived',source_dataset='oi_close',frequency_raw='daily',aggregation='endpoint log growth',formula=f'log(oi_close[t]/oi_close[t-{window}])',unit='log difference',timing='end t',available_at='after end t; provisional metrics timing',role_candidate='state_candidate',notes='Empirically aligned OI; historical publication vintage unavailable'))
  outcome_map={'spot_return_cumulative':('spot_log_return_1d','sum'),'spot_realized_volatility':('spot_realized_volatility','rms'),'spot_downside_semivariance':('spot_downside_semivariance','sum'),'spot_amihud_mean':('spot_amihud','mean'),'spot_quote_volume_mean':('spot_quote_volume','mean'),'spot_taker_imbalance_mean':('spot_taker_imbalance','mean'),'funding_mean':('funding_mean','mean'),'funding_cumulative':('funding_sum','sum'),'basis_mean':('basis_mean','mean'),'futures_taker_imbalance_mean':('futures_taker_imbalance','mean'),'global_account_long_short_ratio_mean':('global_account_long_short_ratio','mean')}
  for h in HORIZONS:
   for stem,(col,method) in outcome_map.items():
    name=f'{stem}_fwd_{h}d';g[name]=future(g[col],h,method)
    if asset=='BTC':records.append(dict(variable=name,asset='BTC;ETH',market_layer='future_outcome',raw_or_derived='derived',source_dataset=col,frequency_raw='daily',aggregation=f'{method} future window',formula=f'{method}({col}[t+1:t+{h}]); requires all {h} daily values',unit='same as source; volatility root sum of squares',timing=f'strictly t+1 through t+{h}',available_at=f'after end t+{h}; NEVER a control',role_candidate='future_outcome',notes=f'overlapping_window={h>1}'))
   name=f'oi_log_change_fwd_{h}d';g[name]=np.log(g.oi_close.shift(-h)/g.oi_close).where(future(g.oi_close,h,'mean').notna())
   if asset=='BTC':records.append(dict(variable=name,asset='BTC;ETH',market_layer='future_outcome',raw_or_derived='derived',source_dataset='oi_close',frequency_raw='daily',aggregation='endpoint change',formula=f'log(oi_close[t+{h}]/oi_close[t]); all future OI days valid',unit='log difference',timing=f'end t to end t+{h}',available_at=f'after end t+{h}',role_candidate='future_outcome',notes=f'overlapping_window={h>1}; provisional source semantics'))
   g[f'horizon_end_{h}d']=g.index+pd.Timedelta(days=h)
   targets=[f'spot_return_cumulative_fwd_{h}d',f'spot_realized_volatility_fwd_{h}d',f'spot_downside_semivariance_fwd_{h}d',f'spot_amihud_mean_fwd_{h}d',f'funding_mean_fwd_{h}d',f'oi_log_change_fwd_{h}d']
   g[f'numeric_eligibility_h{h}']=g[CORE+targets+['funding_mean_lag1','oi_log_change_1d_lag1','spot_log_return_1d_lag1','spot_realized_volatility_lag1','spot_amihud_lag1','futures_taker_imbalance_lag1']].notna().all(axis=1)&(g[f'horizon_end_{h}d']<=pd.Timestamp(END,tz='UTC'))
   g[f'eligibility_h{h}']=g[f'numeric_eligibility_h{h}']&g.metrics_timing_certified
  g['asset']=asset;parts.append(g.reset_index())
 pd.concat(parts).sort_values(['date','asset']).to_pickle(ROOT/'data/interim/full_analytical_panel.pkl')
 panel=pd.concat(parts).sort_values(['date','asset']);panel=panel[(panel.date>=pd.Timestamp(START,tz='UTC'))&(panel.date<=pd.Timestamp(END,tz='UTC'))]
 panel['source_manifest_reference']='outputs/source_manifest.csv';panel['baseline_semantically_eligible']=panel.metrics_timing_certified
 config=json.loads((ROOT/'outputs/sample_definition.json').read_text())
 panel=panel[panel.date>=pd.Timestamp(config['extended_start'],tz='UTC')]
 panel.to_csv(ROOT/'data/processed/derivatives_spot_extended_panel.csv',index=False,float_format='%.12g',date_format='%Y-%m-%dT%H:%M:%S%z')
 panel=panel[panel.date<=pd.Timestamp(config['primary_clean_end'],tz='UTC')].copy()
 for h in HORIZONS:
  beyond=panel[f'horizon_end_{h}d']>pd.Timestamp(config['primary_clean_end'],tz='UTC')
  panel.loc[beyond,[c for c in panel if c.endswith(f'_fwd_{h}d')]]=np.nan
  panel.loc[beyond,[f'numeric_eligibility_h{h}',f'eligibility_h{h}']]=False
 panel.to_csv(ROOT/'data/processed/derivatives_spot_daily_panel.csv',index=False,float_format='%.12g',date_format='%Y-%m-%dT%H:%M:%S%z')
 wide=panel.drop(columns=['timing_reference','source_manifest_reference']).set_index(['date','asset']).unstack('asset');wide.columns=[f'{asset.lower()}_{var}' for var,asset in wide.columns];wide.reset_index().to_csv(ROOT/'data/processed/derivatives_spot_system_daily.csv',index=False,float_format='%.12g',date_format='%Y-%m-%dT%H:%M:%S%z')
 for h in HORIZONS:
  for name,formula,role in [(f'horizon_end_{h}d',f'date+{h} calendar days','timing_metadata'),(f'numeric_eligibility_h{h}',f'complete current core fields + selected lag1 controls + mandatory future labels and horizon_end_{h}d <= selected sample end','eligibility_flag'),(f'eligibility_h{h}',f'numeric_eligibility_h{h} AND metrics_timing_certified','eligibility_flag')]:
   records.append(dict(variable=name,asset='BTC;ETH',market_layer='metadata',raw_or_derived='derived',source_dataset='daily features and future_window_metadata.csv',frequency_raw='daily',aggregation='calendar rule',formula=formula,unit='UTC date or boolean',timing='sample-rule metadata only',available_at='horizon endpoints refer to future labels, never control inputs',role_candidate=role,notes='Eligibility is a retrospective sample flag, NEVER a predictor'))
 for name,formula in [('available_at_utc','date+1 day: earliest measurement-completion boundary; NOT an observed publication receipt'),('metrics_timing_certified','known OI field layout and >=274/288 valid daily snapshots'),('baseline_semantically_eligible','metrics_timing_certified')]:
  records.append(dict(variable=name,asset='BTC;ETH',market_layer='metadata',raw_or_derived='derived',source_dataset='timestamp and coverage audits',frequency_raw='daily',aggregation='timing or coverage rule',formula=formula,unit='UTC timestamp or boolean',timing='metadata',available_at='publication latency not reconstructed',role_candidate='metadata_only',notes='No metadata/eligibility flag belongs in model predictors'))
 pd.DataFrame(records).to_csv(ROOT/'outputs/variable_dictionary.csv',index=False)
 pd.DataFrame([dict(horizon=h,overlapping_window=h>1,future_start='t+1',future_end=f't+{h}',last_possible_exposure_date=str((pd.Timestamp(END)-pd.Timedelta(days=h)).date())) for h in HORIZONS]).to_csv(ROOT/'outputs/future_window_metadata.csv',index=False)
if __name__=='__main__':main()
