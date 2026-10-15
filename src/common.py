"""New-project constants and immutable cached retrieval. No modeling imports."""
from pathlib import Path
import hashlib,json,time,logging
from datetime import datetime,timezone
import requests
ROOT=Path(__file__).resolve().parents[1]
START='2021-01-01';END='2026-09-30';PRE='2020-10-01';SYMBOLS=['BTCUSDT','ETHUSDT']
BASE='https://data.binance.vision/';BUCKET='https://s3-ap-northeast-1.amazonaws.com/data.binance.vision'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fetch(url,folder='metadata',offline=False):
 identity=hashlib.sha256(url.encode()).hexdigest();target=ROOT/'data/raw'/folder/(identity+'.raw');meta=target.with_suffix('.json')
 if meta.exists():
  record=json.loads(meta.read_text());assert target.exists() and sha(target)==record['sha256'],'Raw cache modified '+str(target)
  return target,record
 if offline:raise RuntimeError('Offline cache absent: '+url)
 response=None
 for attempt in range(5):
  try:
   response=requests.get(url,timeout=(15,90),headers={'User-Agent':'DerivativesSpotResearchAudit/1.0'})
   if response.status_code not in [429,500,502,503,504]:break
   delay=min(float(response.headers.get('Retry-After','2')) if response.headers.get('Retry-After','2').isdigit() else 2**attempt,30)
  except requests.RequestException:delay=min(2**attempt,30)
  if attempt<4:time.sleep(delay)
 if response is None:status=0;body=b''
 else:status=response.status_code;body=response.content
 target.parent.mkdir(parents=True,exist_ok=True)
 # Exclusive creation is required: never overwrite a raw path.
 with target.open('xb') as f:f.write(body)
 record=dict(url=url,http_status=status,retrieval_timestamp=datetime.now(timezone.utc).isoformat(),raw_file_path=str(target.relative_to(ROOT)),sha256=sha(target),file_size=target.stat().st_size)
 with meta.open('x') as f:json.dump(record,f,indent=2,sort_keys=True)
 logging.info('HTTP %s %s',status,url)
 return target,record
