"""Empirical field-time audit from independent archive candles, not outcome models."""
import numpy as np,pandas as pd,json
from common import *
from validate_semantics import read

def main():
 records=[]
 for symbol in SYMBOLS:
  m=read(symbol,'futures','metrics');k=read(symbol,'futures','klines');mark=read(symbol,'futures','markPriceKlines')
  ratio=k.taker_buy_base_volume/(k.base_volume-k.taker_buy_base_volume).where(k.base_volume>k.taker_buy_base_volume)
  implied=m.sum_open_interest_value/m.sum_open_interest
  frame=pd.DataFrame(index=m.index)
  for offset in [-300,0,300]:
   ix=m.index+pd.Timedelta(seconds=offset)
   ref=ratio.reindex(ix).to_numpy();obs=m.sum_taker_long_short_vol_ratio.to_numpy()
   frame[f'flow_{offset}']=np.abs(obs-ref)/(np.abs(ref)+1e-12)
   price=mark.open.reindex(ix).to_numpy();frame[f'valuation_{offset}']=np.abs(implied.to_numpy()/price-1)
  for day,g in frame.groupby(frame.index.floor('D')):
   median=g.median();flow=min([-300,0,300],key=lambda v:median[f'flow_{v}'] if pd.notna(median[f'flow_{v}']) else np.inf);stock=min([-300,0,300],key=lambda v:median[f'valuation_{v}'] if pd.notna(median[f'valuation_{v}']) else np.inf)
   # No threshold is selected from market responses. Five-fold separation anchors matching quality.
   flow_confirm=median[f'flow_{flow}']<.05 and median[f'flow_{flow}']*5<min([median[f'flow_{v}'] for v in [-300,0,300] if v!=flow])
   price_confirm=median[f'valuation_{stock}']<.0001 and median[f'valuation_{stock}']*5<min([median[f'valuation_{v}'] for v in [-300,0,300] if v!=stock])
   version='END_LABEL' if stock==0 and price_confirm else ('START_LABEL' if stock==300 and price_confirm else 'UNVERIFIED')
   records.append(dict(symbol=symbol,date=str(day.date()),layout=version,flow_best_candle_open_offset_seconds=flow,oi_best_mark_open_offset_seconds=stock,flow_interval_verified=bool(flow_confirm),flow_matching_relative_error=median[f'flow_{flow}'],valuation_matching_relative_error=median[f'valuation_{stock}'],stock_effective_offset_seconds=0 if version=='END_LABEL' else (300 if version=='START_LABEL' else np.nan),notes='Historical stock mapping is empirical inference from notional/quantity valuation and cross-field flow interval; September2026 REST quantity checks independently verify +300 seconds.'))
 out=pd.DataFrame(records);out.to_csv(ROOT/'outputs/metrics_daily_layout_audit.csv',index=False)
 for symbol,g in out.groupby('symbol'):
  print(symbol,g.groupby('layout').date.agg(['min','max','count']).to_string(),flush=True)
 common=out[out.layout=='START_LABEL'].groupby('symbol').date.min();clean_end=(pd.Timestamp(common.min())-pd.Timedelta(days=1)).strftime('%Y-%m-%d')
 config=dict(requested_start=START,requested_end=END,numeric_common_start=str(max(out.groupby('symbol').date.min()).__str__()),primary_clean_start=str(max(out.groupby('symbol').date.min()).__str__()),primary_clean_end=clean_end,extended_start='2021-12-01',extended_end=END,break_detected=common.to_dict(),method='Fixed field-specific layout rule; stocks shifted only where independently matched; unknown-layout days not used')
 (ROOT/'outputs/sample_definition.json').write_text(json.dumps(config,sort_keys=True,indent=2)+'\n')
if __name__=='__main__':main()
