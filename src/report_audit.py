"""Deterministic availability reports, semantics and QA; no inferential statistics."""
import json,sys,platform
import pandas as pd,numpy as np
from common import *
from audit_raw_data import longest
CORE=['spot_log_return_1d','spot_realized_volatility','spot_quote_volume','funding_mean','oi_close','futures_taker_imbalance']
SEMANTICS={
'spot/klines':('UTC five-minute spot trades OHLC; total base and quote traded volume, trade count, taker-buy base/quote volume','BTC or ETH base; USDT quote; counts','Open time interval start, close_time inclusive interval end. ms before2025, us since2025','Binance spot venue only; sum(base)-sum(taker_buy_base) represents taker-sell volume because every trade has one taker side'),
'futures/klines':('USDⓈ-M perpetual trades OHLC; total base and quote traded volume, trade count and taker-buy volume','BTC or ETH base; USDT quote; counts','UTC five-minute interval start and inclusive interval end, milliseconds','Linear USDT-margined perpetual; quote volume is USDT not independently verified USD'),
'futures/fundingRate':('Archive last_funding_rate and funding_interval_hours; calc_time verified against REST fundingTime on deterministic dates','dimensionless rate per settlement; hours','Actual millisecond settlement timestamp, including small observed offsets','Sum of settlements is an arithmetic sum, not a compounded realized strategy return; no markPrice field in archive'),
'futures/metrics':('sum_open_interest quantity; sum_open_interest_value notional; three distinct account/position ratios; taker buy/sell ratio','base quantity; USDT notional; ratios','Historical field-time reconstruction: END_LABEL stock at create_time; START_LABEL stock at create_time+300s. Unmatched days excluded','OI timing inferred from notional/quantity mark-price matching and validated independently against September2026 REST. Provider historical publication/vintage not reconstructed. Ratios field mappings remain secondary UNVERIFIED'),
'futures/markPriceKlines':('Mark-price OHLC, not transaction-price OHLC','USDT per base unit','UTC five-minute candle start and end','Placeholder volume/trade fields NOT economic volume/trades; excluded'),
'futures/indexPriceKlines':('Index-price OHLC, not Binance spot transaction price','USDT per base unit','UTC five-minute candle start and end','Index constituent changes not audited historically; basis ratio uses contemporaneous five-minute closes, not an instantaneous executable arbitrage spread'),
'futures/premiumIndexKlines':('Provider premium-index OHLC, separate from mark/index ratio','signed dimensionless premium','UTC five-minute candle start and end','Negative and zero premium are valid; no percent/annualization conversion; not interchangeable with basis ratio'),
}
def coverage(frame,start,end,label):
 rows=[];grid=pd.date_range(start,end,tz='UTC')
 dictionary=pd.read_csv(ROOT/'outputs/variable_dictionary.csv').set_index('variable')
 for asset,g in frame.groupby('asset'):
  g=g.set_index('date').reindex(grid)
  for col in g.select_dtypes(include=np.number):
   if col.startswith(('eligibility','numeric_eligibility')):continue
   s=g[col];ok=np.isfinite(s);first=s[ok].index.min();last=s[ok].index.max()
   source=str(dictionary.loc[col,'source_dataset']) if col in dictionary.index else ''
   uncertain=col.endswith('ratio') or col.startswith(('top_','global_','metrics_taker'))
   role='EXCLUDE' if uncertain else ('CORE_BASELINE' if col in CORE else 'SECONDARY_EXTENSION')
   rows.append(dict(asset=asset,variable=col,sample=label,first_date=str(first.date()) if pd.notna(first) else '',last_date=str(last.date()) if pd.notna(last) else '',expected_days=len(grid),valid_days=int(ok.sum()),coverage_pct=float(ok.mean()*100),missing_days=int((~ok).sum()),longest_missing_streak=longest(~ok),nonfinite_count=int((~ok).sum()),zero_count=int((s==0).sum()),negative_count=int((s<0).sum()),semantic_status='UNVERIFIED' if uncertain else 'EMPIRICALLY_ALIGNED' if 'metrics' in source or col.startswith(('oi_','log_oi')) else 'VERIFIED',baseline_eligible=bool(ok.mean()>=.95 and not uncertain and col not in ['spot_intraday_valid_count','futures_intraday_valid_count','oi_valid_snapshots','funding_count']),role=role,reason='Archive ratio population/field-time mapping not independently certified' if uncertain else ('Signed/zero premiums are economically valid; retain as separate secondary provider-defined premium index' if col=='premium_index_mean' else 'No imputation; calendar-exact variables; current archive vintage')))
 return rows

def main():
 manifest=pd.read_csv(ROOT/'outputs/source_manifest.csv');manifest=manifest[manifest['dataset'].isin(['klines','markPriceKlines','indexPriceKlines','premiumIndexKlines','fundingRate','metrics'])].copy();manifest['kind']='archive';manifest['provider']='Binance';manifest['timezone']='UTC'
 filemeta=pd.read_csv(ROOT/'outputs/archive_file_schema_audit.csv');manifest=manifest.drop(columns=['raw_start_date','raw_end_date','schema_version','definition_reference'],errors='ignore').merge(filemeta[['symbol','market','dataset','frequency','file_period','first_timestamp','last_timestamp','schema_version']].rename(columns={'first_timestamp':'raw_start_date','last_timestamp':'raw_end_date'}),on=['symbol','market','dataset','frequency','file_period'],how='left');manifest['definition_reference']='outputs/metric_semantics_manifest.csv; outputs/timestamp_semantics_audit.csv'
 meta=[]
 for envelope in sorted((ROOT/'data/raw/metadata').glob('*.json')):
  if envelope.name=='web_documentation_extract.json':continue
  info=json.loads(envelope.read_text())
  if not isinstance(info,dict) or 'url' not in info:continue
  meta.append({**info,'kind':'metadata','provider':'Binance or GitHub official repository (issue content third-party)','dataset':'metadata/API/documentation','market':'metadata','symbol':'','frequency':'','file_period':'','timezone':'UTC'})
 web=ROOT/'data/raw/metadata/web_documentation_extract.json'
 if web.exists():meta.append(dict(kind='metadata',provider='Binance developer docs via web extraction',dataset='documentation_extract',market='metadata',url='https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data',raw_file_path=str(web.relative_to(ROOT)),sha256=sha(web),file_size=web.stat().st_size,http_status=200,retrieval_timestamp='2026-10-07 (date only; exact extractor timestamp not supplied)',timezone='UTC'))
 manifest=pd.concat([manifest,pd.DataFrame(meta)],ignore_index=True);manifest['access_status']=manifest.http_status.map(lambda x:'ACCESSIBLE' if x==200 else ('ACCESS_RESTRICTED' if x in [401,403,451] else ('UNAVAILABLE' if x==404 else 'UNVERIFIED')));manifest.sort_values(['kind','url']).to_csv(ROOT/'outputs/source_manifest.csv',index=False)
 config=json.loads((ROOT/'outputs/sample_definition.json').read_text());p=pd.read_csv(ROOT/'data/processed/derivatives_spot_daily_panel.csv',parse_dates=['date']);ext=pd.read_csv(ROOT/'data/processed/derivatives_spot_extended_panel.csv',parse_dates=['date']);full=pd.read_pickle(ROOT/'data/interim/full_analytical_panel.pkl')
 cov=pd.DataFrame(coverage(full,START,END,'REQUESTED')+coverage(p,config['primary_clean_start'],config['primary_clean_end'],'PRIMARY_CLEAN')+coverage(ext,config['extended_start'],config['extended_end'],'EXTENDED_HARMONIZED'));cov.to_csv(ROOT/'outputs/model_variable_coverage.csv',index=False)
 # Numerical equivalences only, descriptive correlations never interpreted as direction.
 p.select_dtypes(include=np.number).drop(columns=[c for c in p if '_fwd_' in c or 'lag' in c or 'count' in c],errors='ignore').corr().to_csv(ROOT/'outputs/qa_correlation_matrix.csv')
 quant=[]
 for asset,g in p.groupby('asset'):
  for col in [c for c in g.select_dtypes(include=np.number) if '_fwd_' not in c and 'lag' not in c]:
   for q,val in g[col].quantile([.001,.005,.01,.99,.995,.999]).items():quant.append(dict(asset=asset,variable=col,quantile=q,value=val,action='FLAG_ONLY_NO_WINSORIZATION'))
 pd.DataFrame(quant).to_csv(ROOT/'outputs/outlier_diagnostics.csv',index=False)
 pd.concat([pd.concat([g.nlargest(5,'funding_mean'),g.nsmallest(5,'funding_mean')])[['date','asset','funding_mean','funding_sum']] for _,g in full.groupby('asset')]).sort_values(['asset','funding_mean']).to_csv(ROOT/'outputs/funding_extremes.csv',index=False)
 dictionary=pd.read_csv(ROOT/'outputs/variable_dictionary.csv');sem=[]
 for name,(meaning,unit,timing,caveat) in SEMANTICS.items():
  fields=['open','high','low','close'] if 'PriceKlines' in name or 'premium' in name else (['last_funding_rate','funding_interval_hours','calc_time'] if 'funding' in name else (['sum_open_interest','sum_open_interest_value','count_toptrader_long_short_ratio','sum_toptrader_long_short_ratio','count_long_short_ratio','sum_taker_long_short_vol_ratio'] if 'metrics' in name else ['open','high','low','close','base_volume','quote_volume','number_of_trades','taker_buy_base_volume','taker_buy_quote_volume']))
  for field in fields:sem.append(dict(dataset=name,field=field,economic_meaning=meaning,provider_definition='Official archive README / Binance API schema; archive snake_case mappings audited in code',unit=unit,time_aggregation=timing,known_caveats=caveat,schema_changes='Spot timestamp units Jan2025; futures header presence changes; metrics layout boundary detected Mar2024 in CURRENT archive',cross_asset_comparable='YES after transformations; raw base OI quantity not directly comparable across assets',documentation_source='https://github.com/binance/binance-public-data/; https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data'))
 field_defs={
 'open':('First recorded interval price','USDT per base asset'), 'high':('Maximum recorded interval price','USDT per base asset'), 'low':('Minimum recorded interval price','USDT per base asset'), 'close':('Last recorded interval price','USDT per base asset'),
 'base_volume':('Total base asset quantity traded during the interval','BTC or ETH quantity'), 'quote_volume':('Total quote asset quantity traded during the interval','USDT'), 'number_of_trades':('Transaction candle trade count','count'), 'taker_buy_base_volume':('Base quantity from buyer-aggressor trades','BTC or ETH quantity'), 'taker_buy_quote_volume':('Quote quantity from buyer-aggressor trades','USDT'),
 'last_funding_rate':('Observed rate for the funding settlement; verified against REST fundingRate','dimensionless per settlement'), 'funding_interval_hours':('Funding interval supplied in the settlement archive','hours'), 'calc_time':('Funding archive timestamp matching REST fundingTime on fixed dates','UTC epoch milliseconds'),
 'sum_open_interest':('Outstanding open interest quantity for linear perpetual','BTC or ETH quantity; consistent with USDT notional/mark valuation'), 'sum_open_interest_value':('Open interest marked notional for linear perpetual','USDT'),
 'count_long_short_ratio':('Candidate global long-account/short-account count ratio; archive-to-API mapping UNVERIFIED','ratio'), 'count_toptrader_long_short_ratio':('Candidate top20% margin-balance long-account/short-account ratio; archive mapping UNVERIFIED','ratio'), 'sum_toptrader_long_short_ratio':('Candidate top20% margin-balance long-position/short-position ratio; archive mapping UNVERIFIED','ratio'), 'sum_taker_long_short_vol_ratio':('Candidate taker buyer/seller base-volume ratio; empirical interval diagnostic only','ratio')}
 for record in sem:
  definition,unit=field_defs[record['field']];record['provider_definition']=definition;record['unit']=unit
  if 'premiumIndex' in record['dataset']:record['unit']='signed dimensionless premium';record['provider_definition']='Provider premium-index '+record['field']+' over the five-minute interval; NOT a market price'
  record['verified_for_baseline']=not (record['dataset']=='futures/metrics' and 'ratio' in record['field'])
 pd.DataFrame(sem).to_csv(ROOT/'outputs/metric_semantics_manifest.csv',index=False)
 ts=pd.read_csv(ROOT/'outputs/timestamp_semantics_audit.csv');sel=ts.source_dataset.eq('futures/metrics');ts.loc[sel,'status']='EMPIRICALLY_ALIGNED_OI_ONLY';ts.loc[sel,'semantic_definition']='Field-specific: OI at label in END_LABEL layout, label+300sec in START_LABEL; inferred by mark valuation, independently REST-validated in Sep2026; unknown days unusable';ts.loc[sel,'notes']='Boundary 2024-03-04 in retrieved vintage. No additional 2026 boundary detected. Stock and flow fields must not share one blind shift; ratios remain UNVERIFIED.';ts.to_csv(ROOT/'outputs/timestamp_semantics_audit.csv',index=False)
 base=cov[cov['sample']=='PRIMARY_CLEAN'];core=base[base.variable.isin(CORE)];numeric_ok=bool(len(core)==12 and core.coverage_pct.min()>=95)
 result=dict(ready_for_modeling='PENDING_VALIDATION' if numeric_ok else 'NO',numeric_core_pass=numeric_ok,primary_clean_start=config['primary_clean_start'],primary_clean_end=config['primary_clean_end'],extended_start=config['extended_start'],extended_end=config['extended_end'],unique_dates=int(p.date.nunique()),panel_rows=len(p),horizon_rows={str(h):int(p[f'eligibility_h{h}'].sum()) for h in [1,3,7,14,30]},extended_horizon_rows={str(h):int(ext[f'eligibility_h{h}'].sum()) for h in [1,3,7,14,30]},fatal_blockers=[] if numeric_ok else ['Required core daily coverage below95%'],limitations=['Historical OI field-time mapping is an empirical reconstruction, not publisher-certified vintage history','Current archive files can be revised; raw hashes protect this research snapshot only','Same Binance venue and USDT quote environment; no global-spot or unconditional USD interpretation','Ratios excluded from baseline pending archive-field population/timing confirmation','No independently audited complete liquidation history; optional only','SOL not audited'],core_coverage=core[['asset','variable','coverage_pct']].to_dict('records'))
 (ROOT/'outputs/readiness_summary.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
 versions=dict(python=platform.python_version(),pandas=pd.__version__,numpy=np.__version__,requests=__import__('requests').__version__,randomness='NONE',models='NONE',raw_frequency='5-minute',funding_cadence='8-hour verified',lags=[1,2,7],future_horizons=[1,3,7,14,30]);(ROOT/'outputs/software_configuration.json').write_text(json.dumps(versions,sort_keys=True,indent=2)+'\n')
 inventory=[]
 for f in sorted((ROOT/'data/raw').rglob('*')):
  if f.is_file():inventory.append(dict(path=str(f.relative_to(ROOT)),sha256=sha(f),file_size=f.stat().st_size))
 invpath=ROOT/'outputs/raw_hash_inventory.csv'
 if invpath.exists():
  old=pd.read_csv(invpath);assert old.equals(pd.DataFrame(inventory)),'Raw inventory changed unexpectedly'
 else:pd.DataFrame(inventory).to_csv(invpath,index=False)
 (ROOT/'outputs/transformation_plan.md').write_text('''# Transformation plan — data preparation only

No predictive model or stationarity hypothesis test was run. Price and OI levels are retained for reconstruction, not proposed as automatically stationary regressors. Candidate exposures are funding mean/sum (distinct), base-quantity OI log change, nonannualized mark/index close ratio, and futures taker imbalance. Candidate spot states are daily return, five-minute realized volatility, intraday downside semivariance, USDT-denominated Amihud, and spot taker imbalance. Volume uses log growth. Levels of positive skewed risk/liquidity variables can later receive pre-specified log1p transformations; none were selected from significance. No full-sample scaling, clipping or composites were applied. OI notional includes a mechanical price component; quantity growth and notional growth stay separate. Null economic interpretations are not decided here.
''')
 (ROOT/'outputs/stage2_modeling_plan.md').write_text('''# Stage 2 plan — DOCUMENT ONLY

Run only after a modeling protocol is frozen and readiness is YES. The recommended original-layout primary sample is specified in sample_definition.json; the harmonized extension must be reported separately. Start with individual funding, OI log change, mark/index basis and futures taker imbalance exposures, and spot return, RV, downside, Amihud and imbalance states. No pressure index is created. Later methods: bidirectional temporal TS-DML conditional relationships; a small linear VAR/Granger predictive benchmark; validated continuous-variable Transfer Entropy; rolling/TVP dynamics; timing and impossible-timing falsifications; pre-specified multiple-testing families. Group all splits and bootstrap blocks by date. Same-date BTC/ETH rows cannot supply independent common-factor treatment draws. Use next-day outcomes and lag1 exposure timing robustness. Overlapping future windows need horizon-aware dependence inference. No causal conclusion follows from these methods without external identification. Provider vintage, same-venue linkage and USDT denomination are limitations.
''')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
