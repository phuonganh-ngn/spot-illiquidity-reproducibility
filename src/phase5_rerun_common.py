from __future__ import annotations
import json, re, hashlib, math, datetime
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'/'phase5_dynamic_discovery_rerun'
OLD=ROOT/'outputs'/'phase5_dynamic_discovery'
PANEL=OLD/'dataset_4h.jsonl'
SPLIT=pd.Timestamp('2023-07-01T00:00:00Z')
VARS=json.loads((OLD/'phase5_protocol.json').read_text())['variables']
ASSETS=('BTC','ETH')

def sha(path:Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def clean(x:Any):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(np.floating,float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x,(np.bool_,)):return bool(x)
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,pd.Timestamp):return x.isoformat()
    return x

def write_json(path:Path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(obj),sort_keys=True,indent=2,allow_nan=False)+'\n')

def log(message):
    now=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    with (OUT/'rerun.log').open('a') as f:f.write(f'{now} {message}\n')
    print(message,flush=True)

def load_discovery_only():
    """Stream only rows whose timestamp precedes the frozen split."""
    rows=[]
    with PANEL.open(encoding='utf-8') as f:
        for line in f:
            m=re.search(r'"period_start_utc":"([^"]+)"',line)
            if not m:raise RuntimeError('Missing period_start_utc in frozen JSONL')
            if m.group(1)>='2023-07-01T00:00:00':
                continue
            z=json.loads(line)
            if z['period_start_utc']<'2021-12-01T00:00:00':continue
            if z['asset'] in ASSETS and z['sample_split']=='DISCOVERY':rows.append(z)
    d=pd.DataFrame(rows)
    d['period_start_utc']=pd.to_datetime(d.period_start_utc,utc=True)
    d=d.sort_values(['asset','period_start_utc']).reset_index(drop=True)
    if len(d)!=6924:raise RuntimeError(f'Discovery row count changed: {len(d)}')
    return d

def standardize_discovery(g:pd.DataFrame):
    arr=g[VARS].to_numpy(float)
    complete=np.isfinite(arr).all(axis=1)
    mean=np.nanmean(arr[complete],axis=0); sd=np.nanstd(arr[complete],axis=0,ddof=1)
    if not np.isfinite(sd).all() or (sd<=0).any():raise RuntimeError('Invalid discovery-only scaling')
    z=(arr-mean)/sd
    times=g.period_start_utc.to_numpy()
    segments=[]; seg_times=[]; current=[]; ctimes=[]; previous=None
    for i,t in enumerate(times):
        good=bool(complete[i]); ts=pd.Timestamp(t)
        contiguous=previous is not None and ts-pd.Timestamp(previous)==pd.Timedelta(hours=4)
        if good and (not current or contiguous):
            current.append(z[i]);ctimes.append(ts)
        else:
            if current:segments.append(np.asarray(current,float));seg_times.append(ctimes)
            current=[];ctimes=[]
            if good:current.append(z[i]);ctimes.append(ts)
        previous=ts
    if current:segments.append(np.asarray(current,float));seg_times.append(ctimes)
    return segments,seg_times,mean,sd,complete

def load_asset_discovery():
    d=load_discovery_only();out={}
    for a,g in d.groupby('asset',sort=True):
        seg,seg_times,mean,sd,mask=standardize_discovery(g)
        out[a]={'segments':seg,'segment_times':seg_times,'mean':mean,'sd':sd,
                'complete_n':int(mask.sum()),'rows':g,
                'segment_lengths':[len(x) for x in seg],
                'first_complete':min(t[0] for t in seg_times).isoformat(),
                'last_complete':max(t[-1] for t in seg_times).isoformat()}
    return out

def edge_list_from_matrices(matrix,names,threshold,orientation='lag_source_target'):
    a=np.asarray(matrix,float);out=[]
    for lag in range(a.shape[0]):
        for i,src in enumerate(names):
            for j,dst in enumerate(names):
                v=float(a[lag,i,j])
                if abs(v)>=threshold:
                    out.append({'source':src,'target':dst,'lag':lag+1,'weight':v})
    return out
