import pandas as pd,json
from common import *
def main(offline=False):
 jobs=[]
 for symbol in SYMBOLS:
  for day in ['2021-01-01','2022-04-10','2023-11-16','2025-04-23','2026-09-30']:
   start=int(pd.Timestamp(day,tz='UTC').timestamp()*1000);end=start+86400000-1
   jobs.append(dict(symbol=symbol,dataset='funding_api_validation',file_period=day,url=f'https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&startTime={start}&endTime={end}&limit=1000'))
  for day in ['2026-09-15','2026-09-30']:
   start=int(pd.Timestamp(day,tz='UTC').timestamp()*1000);end=start+86400000-1
   jobs.append(dict(symbol=symbol,dataset='oi_api_validation',file_period=day,url=f'https://fapi.binance.com/futures/data/openInterestHist?symbol={symbol}&period=5m&startTime={start}&endTime={end}&limit=500'))
 records=[]
 for j in jobs:
  _,r=fetch(j['url'],'metadata',offline);records.append({**j,**r})
 pd.DataFrame(records).to_csv(ROOT/'outputs/api_validation_manifest.csv',index=False)
if __name__=='__main__':
 import sys
 main('--offline' in sys.argv)
