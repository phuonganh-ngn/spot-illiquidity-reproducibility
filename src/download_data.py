"""Discover actual S3 keys, then fetch only verified catalogue entries."""
import argparse,concurrent.futures,json,logging,xml.etree.ElementTree as ET
from urllib.parse import urlencode
from datetime import datetime
import pandas as pd
from common import *
DOCS={
'archive_readme':'https://raw.githubusercontent.com/binance/binance-public-data/master/README.md',
'archive_updates':'https://raw.githubusercontent.com/binance/binance-public-data/master/updates/README.md',
'spot_klines':'https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints',
'futures_market':'https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data',
'issue_491':'https://api.github.com/repos/binance/binance-public-data/issues/491',
'issue_491_comments':'https://api.github.com/repos/binance/binance-public-data/issues/491/comments',
'legacy_docs':'https://raw.githubusercontent.com/binance/binance-futures-connector-python/main/binance/um_futures/market.py',
'futures_exchange_info':'https://fapi.binance.com/fapi/v1/exchangeInfo',
'funding_info':'https://fapi.binance.com/fapi/v1/fundingInfo',
'liquidation_archive':'https://data.binance.vision/?prefix=data/futures/um/daily/liquidationSnapshot/',
'current_oi_probe':'https://fapi.binance.com/futures/data/openInterestHist?symbol=BTCUSDT&period=5m&limit=500',
}
NAMESPACE={'s':'http://s3.amazonaws.com/doc/2006-03-01/'}
def catalogue(prefix,offline):
 marker='';records=[]
 while True:
  params={'prefix':prefix,'max-keys':1000}
  if marker:params['marker']=marker
  path,meta=fetch(BUCKET+'?'+urlencode(params),offline=offline)
  if meta['http_status']!=200:raise RuntimeError('Catalogue inaccessible '+prefix)
  xml=ET.fromstring(path.read_bytes());items=xml.findall('s:Contents',NAMESPACE)
  for item in items:
   key=item.findtext('s:Key',namespaces=NAMESPACE)
   records.append(dict(key=key,last_modified=item.findtext('s:LastModified',namespaces=NAMESPACE),size=int(item.findtext('s:Size',namespaces=NAMESPACE)),catalogue_raw=meta['raw_file_path']))
  if xml.findtext('s:IsTruncated',namespaces=NAMESPACE)!='true':break
  marker=xml.findtext('s:NextMarker',namespaces=NAMESPACE) or records[-1]['key']
 return records

def inventory(offline=False):
 docs=[]
 for label,url in DOCS.items():
  path,meta=fetch(url,offline=offline);docs.append(dict(name=label,**meta))
 pd.DataFrame(docs).to_csv(ROOT/'outputs/documentation_manifest.csv',index=False)
 allkeys=[];datasets=[('spot','klines','monthly','5m'),('futures','klines','monthly','5m'),('futures','markPriceKlines','monthly','5m'),('futures','indexPriceKlines','monthly','5m'),('futures','premiumIndexKlines','monthly','5m'),('futures','fundingRate','monthly',''),('futures','metrics','daily',''),('spot','klines','monthly','1d')]
 for symbol in SYMBOLS:
  for market,dataset,period,interval in datasets:
   prefix=f'data/{"spot" if market=="spot" else "futures/um"}/{period}/{dataset}/{symbol}/'+(interval+'/' if interval else '')
   records=catalogue(prefix,offline)
   for r in records:r.update(symbol=symbol,market=market,dataset=dataset,period=period,interval=interval)
   allkeys+=records
   print('Catalogue',symbol,market,dataset,interval,'objects',len(records),flush=True)
 # Optional liquidation archive audit only, not large liquidation downloads.
 records=catalogue('data/futures/um/daily/liquidationSnapshot/BTCUSDT/',offline)
 for r in records:r.update(symbol='BTCUSDT',market='futures',dataset='liquidationSnapshot',period='daily',interval='')
 allkeys+=records
 df=pd.DataFrame(allkeys);df.to_csv(ROOT/'outputs/archive_catalogue.csv',index=False)
 return df

def jobs(df):
 selected=[];crossdates=pd.date_range(START,END,freq='D')
 selectedmonths={crossdates[int((len(crossdates)-1)*q)].strftime('%Y-%m') for q in [0,.25,.5,.75,1]}|{'2024-12','2025-01','2026-06'}
 for row in df.itertuples():
  if not row.key.endswith('.zip') or row.dataset=='liquidationSnapshot':continue
  stamp=row.key.rsplit('-',2)[-2]+'-'+row.key.rsplit('-',1)[-1].replace('.zip','') if row.period=='monthly' else row.key[-14:-4]
  if row.period=='monthly':
   stamp=row.key[-11:-4]
   if not('2020-10'<=stamp<='2026-09'):continue
   if row.interval=='1d' and stamp not in selectedmonths:continue
  else:
   if not(PRE<=stamp<=END):continue
  selected.append(dict(url=BASE+row.key,market=row.market,symbol=row.symbol,dataset=row.dataset,frequency=row.interval or ('5m' if row.dataset=='metrics' else 'FUNDING_SETTLEMENT'),file_period=stamp,key=row.key,catalogue_last_modified=row.last_modified))
 return selected

def download(job,offline):
 path,record=fetch(job['url'],job['market'],offline)
 # Archive-side checksums are discovered from the same listing, never assumed authoritative when absent.
 return {**job,**record}

def main():
 p=argparse.ArgumentParser();p.add_argument('--offline',action='store_true');p.add_argument('--metadata-only',action='store_true');args=p.parse_args()
 logging.basicConfig(filename=ROOT/'logs/download.log',level=logging.INFO,format='%(asctime)s %(message)s')
 cat=inventory(args.offline);work=jobs(cat);(ROOT/'data/interim/download_plan.json').write_text(json.dumps(work,indent=2)+'\n')
 if args.metadata_only:return
 records=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  futures=[pool.submit(download,j,args.offline) for j in work]
  for n,f in enumerate(futures,1):
   records.append(f.result())
   if n%100==0:print('Raw archives',n,'/',len(work),flush=True)
 pd.DataFrame(records).to_csv(ROOT/'outputs/source_manifest.csv',index=False)
 print('Downloaded/cached',len(records),'files; failures',sum(r['http_status']!=200 for r in records),flush=True)
if __name__=='__main__':main()
