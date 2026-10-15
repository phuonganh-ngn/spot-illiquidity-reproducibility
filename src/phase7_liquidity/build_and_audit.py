from __future__ import annotations
import hashlib, json, zipfile, datetime, math, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs' / 'phase7_liquidity_replication'
RAW_CAND = ROOT / 'data' / 'raw' / 'phase7_assets'
PROC = ROOT / 'data' / 'processed' / 'phase7'
START = pd.Timestamp('2021-10-01T00:00:00Z')
SAMPLE_START = pd.Timestamp('2021-12-01T00:00:00Z')
END_EXCL = pd.Timestamp('2026-10-01T00:00:00Z')
EXT_START = pd.Timestamp('2024-03-04T00:00:00Z')
CANDIDATES = ['SOL','BNB','XRP']
ALL_ASSETS = ['BTC','ETH'] + CANDIDATES
KLINE_COLS = ['open_time','open','high','low','close','base_volume','close_time','quote_volume','number_of_trades','taker_buy_base_volume','taker_buy_quote_volume','ignore']
METRIC_COLS = ['create_time','symbol','sum_open_interest','sum_open_interest_value','count_toptrader_long_short_ratio','sum_toptrader_long_short_ratio','count_long_short_ratio','sum_taker_long_short_vol_ratio']


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def jclean(x):
    if isinstance(x,dict): return {str(k):jclean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [jclean(v) for v in x]
    if isinstance(x,(np.integer,)): return int(x)
    if isinstance(x,(np.floating,float)): return float(x) if np.isfinite(x) else None
    if isinstance(x,(np.bool_,)): return bool(x)
    if isinstance(x,(pd.Timestamp,)): return x.isoformat()
    if isinstance(x,np.ndarray): return jclean(x.tolist())
    return x
def write_json(path,obj): path.write_text(json.dumps(jclean(obj),indent=2,sort_keys=True,allow_nan=False)+'\n')

def to_utc_mixed(values):
    x=pd.to_numeric(pd.Series(values),errors='coerce').to_numpy(dtype='float64')
    out=np.full(len(x),np.datetime64('NaT'),dtype='datetime64[ns]')
    us=np.isfinite(x)&(x>=1e14)
    ms=np.isfinite(x)&(x<1e14)
    if us.any(): out[us]=pd.to_datetime(x[us].astype('int64'),unit='us',utc=True).tz_localize(None).to_numpy(dtype='datetime64[ns]')
    if ms.any(): out[ms]=pd.to_datetime(x[ms].astype('int64'),unit='ms',utc=True).tz_localize(None).to_numpy(dtype='datetime64[ns]')
    return pd.DatetimeIndex(out).tz_localize('UTC')

def canonicalize_kline(df, source):
    d=df.copy()
    # Archive header variants are normalized only when names map exactly to the
    # Binance kline meanings; raw CSV/ZIP files remain byte-for-byte unchanged.
    low={str(c).strip().lower():c for c in d.columns}
    if 'open_time' in low:
        def take(*names, default=None):
            for n in names:
                if n in low: return d[low[n]]
            return default
        out=pd.DataFrame({
          'open_time':take('open_time'), 'open':take('open'), 'high':take('high'),
          'low':take('low'), 'close':take('close'),
          'base_volume':take('base_volume','volume'), 'close_time':take('close_time'),
          'quote_volume':take('quote_volume'), 'number_of_trades':take('number_of_trades','count'),
          'taker_buy_base_volume':take('taker_buy_base_volume','taker_buy_volume'),
          'taker_buy_quote_volume':take('taker_buy_quote_volume'), 'ignore':take('ignore',default=0)})
    else:
        if d.shape[1] != 12: raise RuntimeError(f'{source}: unheadered kline has {d.shape[1]} fields')
        out=d.copy(); out.columns=KLINE_COLS
    out['ts']=to_utc_mixed(out.open_time)
    for c in KLINE_COLS:
        if c not in ('open_time',): out[c]=pd.to_numeric(out[c],errors='coerce')
    out=out.set_index('ts').sort_index()
    return out

def read_candidate_kline(manifest_rows, symbol, dataset):
    pieces=[]; schemas=[]; month_counts=[]; hashes=[]
    rows=sorted([r for r in manifest_rows if r['symbol']==symbol and r['dataset']==dataset],key=lambda x:x['period'])
    for r in rows:
        p=ROOT/r['raw_file_path']
        with zipfile.ZipFile(p) as z:
            member=z.namelist()[0]
            with z.open(member) as f:
                first=f.readline().decode('utf-8-sig').strip()
                f.seek(0)
                if first.lower().startswith('open_time,'):
                    d=pd.read_csv(f)
                    schema='headered:'+first
                else:
                    d=pd.read_csv(f,header=None,names=KLINE_COLS)
                    schema='headerless_12'
        c=canonicalize_kline(d,member)
        pieces.append(c)
        schemas.append({'period':r['period'],'schema':schema,'rows':len(c),'sha256':r['sha256']})
        month_counts.append({'period':r['period'],'rows':len(c)})
    if not pieces: raise RuntimeError(f'missing archives {symbol}/{dataset}')
    z=pd.concat(pieces).sort_index()
    return z, {'symbol':symbol,'dataset':dataset,'archive_count':len(rows),'schemas':schemas,'period_rows':month_counts}

def read_stage1_kline(asset, dataset):
    names={'spot':'spot_klines_5m','futures':'futures_klines_5m','mark':'futures_markPriceKlines_5m','index':'futures_indexPriceKlines_5m'}
    p=ROOT/'data'/'interim'/f'{asset}USDT_{names[dataset]}.pkl'
    d=pd.read_pickle(p)
    c=canonicalize_kline(d,p.name)
    return c, {'path':str(p.relative_to(ROOT)),'sha256':sha(p),'rows':len(c)}

def dedupe_frame(df, value_cols):
    dup=int(df.index.duplicated(keep=False).sum()); conflict=0
    if dup:
        q=df[df.index.duplicated(keep=False)]
        for _,g in q.groupby(level=0,sort=False):
            if len(g[value_cols].drop_duplicates())>1: conflict+=len(g)
        bad=[]
        for t,g in q.groupby(level=0,sort=False):
            if len(g[value_cols].drop_duplicates())>1: bad.append(t)
        if bad: df=df[~df.index.isin(bad)]
        df=df[~df.index.duplicated(keep='first')]
    return df.sort_index(), {'duplicate_rows':dup,'conflicting_duplicate_rows_removed':conflict}

def data_quality_5m(df, dataset, full_grid, sample_grid):
    d,dd=dedupe_frame(df,['open','high','low','close','base_volume','quote_volume','number_of_trades','taker_buy_base_volume','taker_buy_quote_volume'])
    # Only the study interval is scored; Oct/Nov archives provide warm-up.
    q=d.reindex(full_grid)
    qs=q.reindex(sample_grid)
    valid=np.isfinite(qs[['open','high','low','close']].to_numpy(float)).all(axis=1)
    valid &= (qs[['open','high','low','close']].to_numpy(float)>0).all(axis=1)
    for c in ['base_volume','quote_volume','number_of_trades','taker_buy_base_volume','taker_buy_quote_volume']:
        valid &= np.isfinite(qs[c].to_numpy(float)) & (qs[c].to_numpy(float)>=0)
    valid &= qs.taker_buy_base_volume.to_numpy(float) <= qs.base_volume.to_numpy(float)+1e-7
    # Timestamp alignment and inclusive candle-close duration audit.
    opens=d.index.asi8
    aligned=int(np.sum(opens % pd.Timedelta(minutes=5).value==0))
    raw_open=pd.to_numeric(d.open_time,errors='coerce').to_numpy(float)
    raw_close=pd.to_numeric(d.close_time,errors='coerce').to_numpy(float)
    us=raw_open>=1e14
    expected=np.where(us,299999999,299999)
    close_offsets=raw_close-raw_open
    close_ok=np.isfinite(close_offsets)&(close_offsets==expected)
    # Longest consecutive invalid streak on expected grid.
    longest=cur=0
    for ok in valid:
        if ok: cur=0
        else: cur+=1;longest=max(longest,cur)
    return {
      'dataset':dataset,'expected_5m_rows':len(sample_grid),'actual_unique_rows_in_grid':int(qs.index.isin(d.index).sum()),
      'valid_rows':int(valid.sum()),'coverage_pct':float(valid.mean()*100),'duplicate_rows':dd['duplicate_rows'],
      'conflicting_duplicate_rows_removed':dd['conflicting_duplicate_rows_removed'],
      'timestamp_grid_alignment_pct':float(aligned/max(1,len(d))*100),
      'inclusive_close_duration_valid_pct':float(close_ok.mean()*100),
      'nonfinite_field_counts':{c:int((~np.isfinite(qs[c].to_numpy(float))).sum()) for c in ['open','high','low','close','base_volume','quote_volume','taker_buy_base_volume','taker_buy_quote_volume']},
      'zero_quote_volume_count':int((qs.quote_volume==0).sum()),'zero_base_volume_count':int((qs.base_volume==0).sum()),
      'negative_impossible_count':int(((qs[['open','high','low','close']]<=0).any(axis=1)|(qs[['base_volume','quote_volume','number_of_trades','taker_buy_base_volume','taker_buy_quote_volume']]<0).any(axis=1)).sum()),
      'longest_missing_or_invalid_streak_5m':longest,'first_valid_utc':str(sample_grid[valid].min()) if valid.any() else None,
      'last_valid_utc':str(sample_grid[valid].max()) if valid.any() else None,'dedupe':dd,
      'valid_mask':valid,'aligned_frame':q
    }

def read_candidate_funding(rows,symbol):
    pieces=[]; schema=[]
    files=sorted([r for r in rows if r['symbol']==symbol and r['dataset']=='funding_monthly'],key=lambda x:x['period'])
    for r in files:
        with zipfile.ZipFile(ROOT/r['raw_file_path']) as z:
            member=z.namelist()[0]
            with z.open(member) as f: d=pd.read_csv(f)
        expected=['calc_time','funding_interval_hours','last_funding_rate']
        if list(d.columns)!=expected: raise RuntimeError(f'{member}: unexpected funding schema {d.columns}')
        d['calc_time']=pd.to_numeric(d.calc_time,errors='coerce')
        d['ts']=to_utc_mixed(d.calc_time)
        d['funding_interval_hours']=pd.to_numeric(d.funding_interval_hours,errors='coerce')
        d['last_funding_rate']=pd.to_numeric(d.last_funding_rate,errors='coerce')
        pieces.append(d.set_index('ts').sort_index()[['funding_interval_hours','last_funding_rate']])
        schema.append({'period':r['period'],'rows':len(d),'header':expected})
    x=pd.concat(pieces).sort_index()
    x,dd=dedupe_frame(x,['funding_interval_hours','last_funding_rate'])
    return x, {'archive_count':len(files),'rows':len(x),'duplicate_rows':dd['duplicate_rows'],'conflicting_duplicate_rows_removed':dd['conflicting_duplicate_rows_removed'],'interval_hours_distribution':{str(k):int(v) for k,v in x.funding_interval_hours.value_counts(dropna=False).items()},'first_timestamp':str(x.index.min()),'last_timestamp':str(x.index.max()),'schema_by_month':schema}

def read_candidate_oi(rows,symbol):
    pieces=[]; schemas=[]; day_coverage=[]
    files=sorted([r for r in rows if r['symbol']==symbol and r['dataset']=='metrics_daily'],key=lambda x:x['period'])
    for r in files:
        with zipfile.ZipFile(ROOT/r['raw_file_path']) as z:
            member=z.namelist()[0]
            with z.open(member) as f: d=pd.read_csv(f)
        if list(d.columns)!=METRIC_COLS: raise RuntimeError(f'{member}: metric schema changed {list(d.columns)}')
        ts=pd.to_datetime(d.create_time,utc=True,errors='coerce')
        d['ts']=ts; d['sum_open_interest']=pd.to_numeric(d.sum_open_interest,errors='coerce');d['sum_open_interest_value']=pd.to_numeric(d.sum_open_interest_value,errors='coerce')
        d=d.set_index('ts').sort_index()
        day=r['period']
        day_coverage.append({'date':day,'rows':len(d),'valid_oi':int((np.isfinite(d.sum_open_interest)&(d.sum_open_interest>0)&np.isfinite(d.sum_open_interest_value)&(d.sum_open_interest_value>0)).sum()),'duplicate_timestamps':int(d.index.duplicated(keep=False).sum())})
        pieces.append(d[['sum_open_interest','sum_open_interest_value']])
        schemas.append({'date':day,'columns':METRIC_COLS,'rows':len(d),'sha256':r['sha256']})
    x=pd.concat(pieces).sort_index();x,dd=dedupe_frame(x,['sum_open_interest','sum_open_interest_value'])
    return x, {'archive_days':len(files),'rows':len(x),'duplicate_rows':dd['duplicate_rows'],'conflicting_duplicate_rows_removed':dd['conflicting_duplicate_rows_removed'],'daily_coverage':day_coverage,'schema_by_day':schemas}

def oi_offset_audit(oi,mark):
    implied=(oi.sum_open_interest_value/oi.sum_open_interest).replace([np.inf,-np.inf],np.nan)
    mark_open=mark.open.copy(); mark_open.index=mark.index
    offsets=[-300,0,300]
    all_rows=[]
    dates=pd.date_range(SAMPLE_START.normalize(),END_EXCL.normalize()-pd.Timedelta(days=1),freq='D',tz='UTC')
    for day in dates:
        day0=day;day1=day+pd.Timedelta(days=1)
        oi_day=oi[(oi.index>=day0)&(oi.index<day1)]
        metrics=[]
        for off in offsets:
            query=oi_day.index+pd.Timedelta(seconds=off)
            m=mark_open.reindex(query).to_numpy(float); imp=implied.reindex(oi_day.index).to_numpy(float)
            er=np.abs(imp/m-1);ok=np.isfinite(er)&(m>0)&(imp>0)
            metrics.append({'offset_seconds':off,'match_count':int(ok.sum()),'median_relative_error':float(np.median(er[ok])) if ok.any() else None,'mean_relative_error':float(np.mean(er[ok])) if ok.any() else None})
        eligible=[x for x in metrics if x['match_count']>=int(.95*max(1,len(oi_day)))]
        ranked=sorted([x for x in eligible if x['median_relative_error'] is not None],key=lambda x:x['median_relative_error'])
        best=ranked[0] if ranked else None;second=ranked[1] if len(ranked)>1 else None
        valid=bool(best and best['median_relative_error']<1e-4 and second and second['median_relative_error']>=5*max(best['median_relative_error'],1e-12))
        all_rows.append({'date':day.strftime('%Y-%m-%d'),'snapshot_rows':len(oi_day),'offset_diagnostics':metrics,'best_offset_seconds':best['offset_seconds'] if best else None,'best_median_relative_error':best['median_relative_error'] if best else None,'second_best_median_relative_error':second['median_relative_error'] if second else None,'mapping_valid':valid})
    return all_rows

def read_stage1_oi(asset):
    p=ROOT/'data/interim'/f'{asset}USDT_futures_metrics_5m.pkl';d=pd.read_pickle(p)
    ts=pd.to_datetime(d.create_time,utc=True,errors='coerce')
    audit=pd.read_csv(ROOT/'outputs/metrics_daily_layout_audit.csv')
    layout=audit[audit.symbol==f'{asset}USDT'].copy();layout['date']=pd.to_datetime(layout.date).dt.strftime('%Y-%m-%d')
    mp=dict(zip(layout.date,pd.to_numeric(layout.stock_effective_offset_seconds,errors='coerce')))
    dates=ts.dt.strftime('%Y-%m-%d'); offs=pd.to_numeric(dates.map(mp),errors='coerce').to_numpy(float)
    mapped=ts+pd.to_timedelta(np.where(np.isfinite(offs),offs,0),unit='s')
    out=pd.DataFrame({'sum_open_interest':pd.to_numeric(d.sum_open_interest,errors='coerce').to_numpy(),'sum_open_interest_value':pd.to_numeric(d.sum_open_interest_value,errors='coerce').to_numpy()},index=mapped)
    out=out[np.isfinite(offs)];out.index=pd.DatetimeIndex(out.index).tz_convert('UTC');out,dd=dedupe_frame(out,['sum_open_interest','sum_open_interest_value'])
    extent=[]
    for date,offset in sorted(mp.items()):
        if date>='2021-12-01' and date<='2026-09-30': extent.append({'date':date,'offset_seconds':offset})
    return out,{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'rows':len(d),'mapped_rows':len(out),'unmapped_rows':int((~np.isfinite(offs)).sum()),'effective_offset_by_date':extent,'duplicate_rows':dd['duplicate_rows']}

def aggregate_4h(spot,futures,mark,index,oi,funding,asset):
    grid5=pd.date_range(START,END_EXCL,freq='5min',inclusive='left')
    spot=spot.reindex(grid5); futures=futures.reindex(grid5); mark=mark.reindex(grid5); index=index.reindex(grid5);oi=oi.reindex(grid5)
    group=grid5.floor('4h')
    def agg_market(z, prefix):
        valid=np.isfinite(z[['open','high','low','close','base_volume','quote_volume','number_of_trades','taker_buy_base_volume','taker_buy_quote_volume']].to_numpy(float)).all(axis=1)
        valid &= (z[['open','high','low','close']].to_numpy(float)>0).all(axis=1)
        valid &= (z[['base_volume','quote_volume','number_of_trades','taker_buy_base_volume','taker_buy_quote_volume']].to_numpy(float)>=0).all(axis=1)
        valid &= z.taker_buy_base_volume.to_numpy(float)<=z.base_volume.to_numpy(float)+1e-7
        q=z.copy();q.loc[~valid,:]=np.nan;g=q.groupby(group)
        count=pd.Series(valid.astype(int),index=grid5).groupby(group).sum()
        out=pd.DataFrame(index=count.index)
        out[prefix+'_open']=g.open.first();out[prefix+'_high']=g.high.max();out[prefix+'_low']=g.low.min();out[prefix+'_close']=g.close.last()
        for c in ['base_volume','quote_volume','number_of_trades','taker_buy_base_volume','taker_buy_quote_volume']:
            out[prefix+'_'+c]=g[c].sum(min_count=48)
        out[prefix+'_valid_5m_bars']=count
        out.loc[count.ne(48),out.columns.drop(prefix+'_valid_5m_bars')]=np.nan
        r5=np.log(q.close/q.close.shift(1)); rg=r5.groupby(group); n=rg.count()
        out[prefix+'_rv']=np.sqrt(rg.apply(lambda s:np.square(s).sum(min_count=48))).where(n.eq(48)&count.eq(48))
        out[prefix+'_downside']=rg.apply(lambda s:np.square(s.clip(upper=0)).sum(min_count=48)).where(n.eq(48)&count.eq(48))
        imb=((2*out[prefix+'_taker_buy_base_volume']-out[prefix+'_base_volume'])/out[prefix+'_base_volume'].where(out[prefix+'_base_volume']>0)).where(count.eq(48))
        out[prefix+'_imbalance']=imb
        out[prefix+'_volume_growth']=np.log(out[prefix+'_quote_volume'].where(out[prefix+'_quote_volume']>0)).diff()
        if prefix=='spot':
            out['spot_return']=np.log(out.spot_close/out.spot_close.shift(1))
            out['spot_realized_volatility']=out.spot_rv
            out['spot_downside_semivariance']=out.spot_downside
            out['spot_amihud']=1e6*out.spot_return.abs()/out.spot_quote_volume.where(out.spot_quote_volume>0)
            out['spot_taker_imbalance']=out.spot_imbalance
            out['spot_quote_volume_growth']=out.spot_volume_growth
        else:
            out['futures_taker_imbalance']=out.futures_imbalance
        return out
    s=agg_market(spot,'spot');f=agg_market(futures,'futures')
    # Basis from 5m mark/index closes; require a complete matched 4h interval.
    b=(mark.close/(index.close.where(index.close>0))-1).replace([np.inf,-np.inf],np.nan)
    basis_count=b.groupby(group).count();basis=b.groupby(group).mean().where(basis_count.eq(48))
    oi_good=np.isfinite(oi.sum_open_interest.to_numpy(float))&(oi.sum_open_interest.to_numpy(float)>0)&np.isfinite(oi.sum_open_interest_value.to_numpy(float))&(oi.sum_open_interest_value.to_numpy(float)>0)
    oi_close=oi.sum_open_interest.where(oi_good).groupby(group).last()
    oi_count=pd.Series(oi_good.astype(int),index=grid5).groupby(group).sum();oi_close=oi_close.where(oi_count.eq(48))
    oi_change=np.log(oi_close/oi_close.shift(1))
    starts=pd.date_range(START,END_EXCL,freq='4h',inclusive='left')
    out=pd.DataFrame(index=starts).join(s,how='left').join(f,how='left')
    out['oi_close']=oi_close.reindex(starts);out['oi_valid_5m_snapshots']=oi_count.reindex(starts);out['oi_log_change']=oi_change.reindex(starts)
    out['basis_mean']=basis.reindex(starts);out['basis_valid_5m_bars']=basis_count.reindex(starts)
    end_times=starts+pd.Timedelta(hours=4)
    if len(funding):
        rates=funding['last_funding_rate'].to_numpy(float);ts=funding.index.asi8;ends=end_times.asi8
        pos=np.searchsorted(ts,ends,side='right')-1; vals=np.full(len(starts),np.nan);ages=np.full(len(starts),np.nan);ft=np.full(len(starts),np.datetime64('NaT'),dtype='datetime64[ns]')
        ok=pos>=0;ix=pos[ok];vals[ok]=rates[ix];ft[ok]=funding.index.to_numpy(dtype='datetime64[ns]')[ix];ages[ok]=(ends[ok]-ts[ix])/3.6e12;vals[(ages>12)|(ages<0)]=np.nan
        out['funding_rate_state']=vals;out['funding_age_hours']=ages;out['funding_last_event_utc']=pd.to_datetime(ft,utc=True)
    else:
        out['funding_rate_state']=np.nan;out['funding_age_hours']=np.nan;out['funding_last_event_utc']=pd.NaT
    out=out.reset_index(names='period_start_utc');out['asset']=asset
    # 4h CS spread proxy based on successive future-window 4h OHLC bars.
    cs=corwin_schultz(out[['spot_high','spot_low']])
    out['corwin_schultz_4h']=cs
    # Future h=6 labels use only t+1..t+6; all labels end by 2026-09-30.
    for i in range(1,7):
        out[f'_amihud_f{i}']=out.spot_amihud.shift(-i)
        out[f'_rv_sq_f{i}']=np.square(out.spot_rv.shift(-i))
        out[f'_downside_f{i}']=out.spot_downside.shift(-i)
        out[f'_volume_f{i}']=out.spot_quote_volume.shift(-i)
    out['future_amihud_24h_mean']=out[[f'_amihud_f{i}' for i in range(1,7)]].mean(axis=1).where(out[[f'_amihud_f{i}' for i in range(1,7)]].notna().all(axis=1))
    out['future_rv_24h']=np.sqrt(out[[f'_rv_sq_f{i}' for i in range(1,7)]].sum(axis=1,min_count=6))
    out['future_downside_semivariance_24h']=out[[f'_downside_f{i}' for i in range(1,7)]].sum(axis=1,min_count=6)
    out['future_quote_volume_24h_mean']=out[[f'_volume_f{i}' for i in range(1,7)]].mean(axis=1).where(out[[f'_volume_f{i}' for i in range(1,7)]].notna().all(axis=1))
    # CS estimates use five adjacent-bar pairs fully inside t+1..t+6.
    cs_future=[]
    for j in range(1,6):
        # pair is (future bar t+1+j-1, future bar t+1+j)
        a=out.corwin_schultz_4h.shift(-(j+1))
        cs_future.append(a)
    out['future_corwin_schultz_spread_24h']=pd.concat(cs_future,axis=1).mean(axis=1).where(pd.concat(cs_future,axis=1).notna().all(axis=1))
    # 5m Kyle-type slope from future 288 5m returns on signed USDT quote flow.
    # This is a 24h target with an intercept, using only the 288 following bars.
    px=spot.reindex(grid5).close.to_numpy(float)
    qv=spot.reindex(grid5).quote_volume.to_numpy(float)
    buyq=spot.reindex(grid5).taker_buy_quote_volume.to_numpy(float)
    ret=np.full(len(px),np.nan);ret[1:]=np.log(px[1:]/px[:-1])
    flow=2*buyq-qv
    n=288
    good=np.isfinite(ret)&np.isfinite(flow)&(qv>=0)
    x=np.where(good,flow,0.);y=np.where(good,ret,0.)
    # Prefix sums allow exact complete-window regressions in O(N).
    def rollsum(v):
        csu=np.r_[0.,np.cumsum(v,dtype=float)];return csu[n:]-csu[:-n]
    cnt=rollsum(good.astype(float));sx=rollsum(x);sy=rollsum(y);sxx=rollsum(x*x);sxy=rollsum(x*y)
    varx=sxx-sx*sx/n;cov=sxy-sx*sy/n
    slope=np.divide(cov,varx,out=np.full_like(cov,np.nan),where=(cnt==n)&(varx>0))
    # Rolling window ending at 5m row k maps to feature t where k is six
    # 4h intervals after t, i.e. k=(t+7)*48-1.
    end_idx=np.arange(n-1,len(px));delta=end_idx-7*48+1
    valid_map=(delta>=0)&(delta%48==0)
    feature_idx=np.full(len(end_idx),-1,dtype=int);feature_idx[valid_map]=delta[valid_map]//48
    valid_map &= feature_idx<len(out)
    kyle=np.full(len(out),np.nan);kyle_signed=np.full(len(out),np.nan)
    kyle[feature_idx[valid_map]]=np.abs(slope[valid_map]);kyle_signed[feature_idx[valid_map]]=slope[valid_map]
    # Require target feature dates to be after the pre-sample support and before the boundary.
    out['future_kyle_abs_lambda_24h']=kyle;out['future_kyle_signed_lambda_24h']=kyle_signed
    out['future_kyle_negative_slope']=(out.future_kyle_signed_lambda_24h<0).astype('boolean')
    out['future_window_start_utc']=out.period_start_utc+pd.Timedelta(hours=4)
    out['future_window_end_utc_exclusive']=out.period_start_utc+pd.Timedelta(hours=28)
    # 24h future labels outside source sample are omitted, not backfilled.
    bad_end=out.future_window_end_utc_exclusive>END_EXCL
    labelcols=[c for c in out if c.startswith('future_') and c!='future_kyle_negative_slope']
    out.loc[bad_end,labelcols]=np.nan
    out.loc[bad_end,'future_kyle_negative_slope']=pd.NA
    # Remove helper columns.
    out=out.drop(columns=[c for c in out if c.startswith('_')])
    return out

def corwin_schultz(ohlc):
    hi=pd.to_numeric(ohlc.spot_high,errors='coerce').to_numpy(float);lo=pd.to_numeric(ohlc.spot_low,errors='coerce').to_numpy(float)
    beta=np.full(len(hi),np.nan);gamma=np.full(len(hi),np.nan)
    lhi=np.log(hi);llo=np.log(lo)
    for t in range(1,len(hi)):
        if np.isfinite(lhi[t]) and np.isfinite(llo[t]) and np.isfinite(lhi[t-1]) and np.isfinite(llo[t-1]):
            beta[t]=(lhi[t]-llo[t])**2+(lhi[t-1]-llo[t-1])**2
            gamma[t]=(max(lhi[t],lhi[t-1])-min(llo[t],llo[t-1]))**2
    den=3-2*np.sqrt(2)
    alpha=(np.sqrt(2*beta)-np.sqrt(beta))/den-np.sqrt(gamma/den)
    alpha=np.maximum(alpha,0)
    return 2*(np.exp(alpha)-1)/(1+np.exp(alpha))

def load_asset_data(asset, manifest_rows, full_grid, sample_grid):
    inputs={};raw={}
    if asset in ('BTC','ETH'):
        for ds in ['spot','futures','mark','index']:
            raw[ds],inputs[ds]=read_stage1_kline(asset,ds)
    else:
        for ds in ['spot_5m','futures_5m','mark_5m','index_5m']:
            rawds,diag=read_candidate_kline(manifest_rows,f'{asset}USDT',ds)
            key={'spot_5m':'spot','futures_5m':'futures','mark_5m':'mark','index_5m':'index'}[ds]
            raw[key]=rawds;inputs[key]=diag
    for ds in ['spot','futures','mark','index']:
        diag=data_quality_5m(raw[ds],ds,full_grid,sample_grid);raw[ds]=diag['aligned_frame'];inputs[ds]['coverage_audit']={k:v for k,v in diag.items() if k not in ('valid_mask','aligned_frame')}
        inputs[ds]['valid_mask_coverage']=diag['coverage_pct']
    if asset in ('BTC','ETH'):
        oi,odiag=read_stage1_oi(asset)
        # Stage-1 field-specific effective timestamp mappings are reused verbatim.
        oi=oi[(oi.index>=START)&(oi.index<END_EXCL)]
        inputs['oi']=odiag
        fpath=ROOT/'data/interim'/f'{asset}USDT_futures_fundingRate_FUNDING_SETTLEMENT.pkl';fd=pd.read_pickle(fpath)
        fts=to_utc_mixed(fd.calc_time);fund=pd.DataFrame({'funding_interval_hours':pd.to_numeric(fd.funding_interval_hours,errors='coerce').to_numpy(),'last_funding_rate':pd.to_numeric(fd.last_funding_rate,errors='coerce').to_numpy()},index=fts).sort_index()
        fund,_=dedupe_frame(fund,['funding_interval_hours','last_funding_rate']);inputs['funding']={'path':str(fpath.relative_to(ROOT)),'sha256':sha(fpath),'rows':len(fund),'interval_hours_distribution':{str(k):int(v) for k,v in fund.funding_interval_hours.value_counts().items()}}
    else:
        oi,odiag=read_candidate_oi(manifest_rows,f'{asset}USDT')
        layout=oi_offset_audit(oi,raw['mark']);
        break_date='2024-03-04'
        mapping={x['date']:(0 if x['date']<break_date else 300) if x['mapping_valid'] and x['best_offset_seconds']==(0 if x['date']<break_date else 300) else None for x in layout}
        ts=oi.index;date=ts.strftime('%Y-%m-%d');offs=pd.to_numeric(pd.Series(date).map(mapping),errors='coerce').to_numpy(float)
        aligned=oi.copy();aligned.index=ts+pd.to_timedelta(np.where(np.isfinite(offs),offs,0),unit='s');aligned=aligned[np.isfinite(offs)]
        aligned,_=dedupe_frame(aligned,['sum_open_interest','sum_open_interest_value']);oi=aligned
        expected_mapped=sum(v is not None for v in mapping.values())
        observed_runs=[]
        for r in layout:
            v=r['best_offset_seconds'] if r['mapping_valid'] else None
            if v is not None and (not observed_runs or observed_runs[-1]['best_offset_seconds']!=v):
                observed_runs.append({'start_date':r['date'],'best_offset_seconds':v})
        inputs['oi']={**odiag,'offset_audit':layout,'mapped_rows':len(oi),'unmapped_layout_rows':int((~np.isfinite(offs)).sum()),
                      'fixed_piecewise_mapping':{'break_date':break_date,'before_seconds':0,'from_break_seconds':300},
                      'expected_mapping_valid_days':expected_mapped,'expected_mapping_valid_days_pct':100*expected_mapped/max(1,len(layout)),
                      'observed_daily_best_offset_runs':observed_runs,
                      'isolated_or_unverified_days':[x['date'] for x in layout if mapping.get(x['date']) is None]}
        fund,fdiag=read_candidate_funding(manifest_rows,f'{asset}USDT');inputs['funding']=fdiag
    panel=aggregate_4h(raw['spot'],raw['futures'],raw['mark'],raw['index'],oi,fund,asset)
    return panel,inputs,raw

def main():
    OUT.mkdir(parents=True,exist_ok=True);PROC.mkdir(parents=True,exist_ok=True)
    man_path=OUT/'candidate_source_manifest.json';man=json.loads(man_path.read_text());rows=man['files']
    if man['retrieved_job_count']!=man['expected_job_count'] or man['phase1_manifest_modified']:
        raise RuntimeError('Candidate archive catalogue/download manifest is incomplete or Stage-1 manifest was modified')
    # Immutable archive check includes SHA-256 and ZIP CRC; candidate content remains cached in place.
    integrity=[]
    for i,r in enumerate(rows,1):
        p=ROOT/r['raw_file_path']
        digest=sha(p) if p.exists() else None
        crc_error=None
        try:
            with zipfile.ZipFile(p) as z: crc_error=z.testzip()
        except Exception as e: crc_error=f'{type(e).__name__}:{e}'
        ok=(digest==r['sha256'] and crc_error is None and p.stat().st_size==r['file_size'])
        integrity.append({'path':r['raw_file_path'],'sha256_expected':r['sha256'],'sha256_actual':digest,'size_expected':r['file_size'],'size_actual':p.stat().st_size if p.exists() else None,'crc_error':crc_error,'pass':bool(ok)})
        if i%1000==0: print(f'archive integrity {i}/{len(rows)}',flush=True)
    if not all(x['pass'] for x in integrity):
        write_json(OUT/'candidate_archive_integrity.json',{'pass':False,'files':integrity});raise RuntimeError('Candidate archive integrity failure; see audit')
    (OUT/'candidate_archive_integrity.json').write_text(json.dumps({'pass':True,'archive_count':len(rows),'bytes':sum(x['size_actual'] for x in integrity),'sha256_verified':len(integrity),'zip_crc_verified':len(integrity),'files':integrity},indent=2)+'\n')
    full_grid=pd.date_range(START,END_EXCL,freq='5min',inclusive='left')
    sample_grid=pd.date_range(SAMPLE_START,END_EXCL,freq='5min',inclusive='left')
    full=[];audit={'study_period':{'start':SAMPLE_START,'end_exclusive':END_EXCL},'external_period_start':EXT_START,'candidate_assets':{},'core_assets':{},'archive_layout_breaks':{'2024-03': 'candidate monthly USD-M kline files switch from headerless 12-field CSV to headered CSV; field labels map to the same Binance kline concepts and timestamps remain alignable','2025-01': 'spot open_time/close_time values switch from milliseconds to microseconds; decoded by raw magnitude, not fixed unit'},'archive_integrity_pass':True}
    all_inputs={}
    for asset in ALL_ASSETS:
        print(f'loading/auditing/building {asset}',flush=True)
        panel,inputs,raw=load_asset_data(asset,rows,full_grid,sample_grid)
        full.append(panel);all_inputs[asset]=inputs
        # Coverage of 4h key features across fixed calendar grid.
        g=panel[(panel.period_start_utc>=SAMPLE_START)&(panel.period_start_utc<END_EXCL)]
        vars=['spot_return','spot_rv','spot_quote_volume','funding_rate_state','oi_log_change','futures_taker_imbalance','basis_mean','spot_amihud']
        cov={v:{'valid_4h':int(g[v].notna().sum()),'expected_4h':len(g),'coverage_pct':float(g[v].notna().mean()*100),'first_valid':str(g.loc[g[v].notna(),'period_start_utc'].min()) if g[v].notna().any() else None,'last_valid':str(g.loc[g[v].notna(),'period_start_utc'].max()) if g[v].notna().any() else None} for v in vars}
        audit['candidate_assets' if asset in CANDIDATES else 'core_assets'][asset]={'input_audit':inputs,'feature_coverage':cov,'rows':len(panel),'first_4h':str(panel.period_start_utc.min()),'last_4h':str(panel.period_start_utc.max())}
        if asset in CANDIDATES:
            policy=json.loads((OUT/'asset_eligibility_protocol.json').read_text())['pass_rules']
            required=['spot_return','spot_rv','spot_quote_volume','futures_taker_imbalance','funding_rate_state','oi_log_change','basis_mean']
            checks={v:cov[v]['coverage_pct']>=95 for v in required}
            layout=inputs['oi'].get('offset_audit',[])
            mapping_rate=inputs['oi']['expected_mapping_valid_days_pct']
            offset_seq=[]
            for x in layout:
                expected=0 if x['date']<'2024-03-04' else 300
                if x['mapping_valid'] and x['best_offset_seconds']==expected and (not offset_seq or offset_seq[-1]['offset']!=expected):offset_seq.append({'start':x['date'],'offset':expected})
            map_pass=mapping_rate>=95 and len(offset_seq)==2 and offset_seq[0]['offset']==0 and offset_seq[1]['offset']==300 and offset_seq[1]['start']=='2024-03-04'
            checks['oi_offset_mapping']=map_pass
            eligible=all(checks.values())
            audit['candidate_assets'][asset]['eligibility']={'checks':checks,'oi_mapping_valid_days_pct':mapping_rate,'oi_offset_breaks':offset_seq,'eligible':eligible,'rule_sha256':sha(OUT/'asset_eligibility_protocol.json')}
    panel=pd.concat(full,ignore_index=True).sort_values(['asset','period_start_utc']).reset_index(drop=True)
    # Independent temporal features and products; strict backward rolling denominators.
    panel['period_start_utc']=pd.to_datetime(panel.period_start_utc,utc=True)
    derived=[]
    for a,g in panel.groupby('asset',sort=False):
        g=g.sort_values('period_start_utc').copy()
        for c,new in [('funding_rate_state','funding_change'),('basis_mean','basis_change')]:g[new]=g[c].diff()
        for c,new in [('oi_log_change','oi_shock'),('futures_taker_imbalance','flow_shock')]:
            past=g[c].shift(1);mu=past.rolling(180,min_periods=120).mean();sd=past.rolling(180,min_periods=120).std(ddof=1)
            g[new]=(g[c]-mu)/sd.where(sd>0)
        g['funding_basis_interaction']=g.funding_rate_state*g.basis_mean
        g['funding_oishock_interaction']=g.funding_rate_state*g.oi_shock
        g['oishock_flow_interaction']=g.oi_shock*g.futures_taker_imbalance
        g['flow_basis_interaction']=g.futures_taker_imbalance*g.basis_mean
        ts=g.period_start_utc.dt.hour//4;dow=g.period_start_utc.dt.dayofweek
        for i in range(6):g[f'utc_slot_{i}']=(ts==i).astype('int8')
        for i in range(7):g[f'weekday_{i}']=(dow==i).astype('int8')
        g['funding_schedule_window']=(g.period_start_utc.dt.hour.isin([0,8,16])).astype('int8')
        derived.append(g)
    panel=pd.concat(derived,ignore_index=True).sort_values(['asset','period_start_utc']).reset_index(drop=True)
    panel_path=PROC/'market_liquidity_panel_4h.csv.gz';panel.to_csv(panel_path,index=False,compression={'method':'gzip','mtime':0},date_format='%Y-%m-%dT%H:%M:%SZ')
    # Input hashes: prior Phase-5/6 data and every normalized Stage-1 pkl used.
    used=[]
    for a in ['BTC','ETH']:
        for suffix in ['spot_klines_5m','futures_klines_5m','futures_markPriceKlines_5m','futures_indexPriceKlines_5m','futures_metrics_5m','futures_fundingRate_FUNDING_SETTLEMENT']:
            p=ROOT/'data/interim'/f'{a}USDT_{suffix}.pkl';used.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'size':p.stat().st_size})
    audit['candidate_archive_layout_change_evidence']='The monthly archive files were inspected for each month; deterministic header parsing begins in 2024-03 for futures/mark/index, while raw field names and 5-minute grid remain the same. Spot timestamps decode as ms before 2025-01 and us beginning 2025-01-01.'
    audit['common_4h_rows']=int(len(panel));audit['assets']=ALL_ASSETS;audit['panel_path']=str(panel_path.relative_to(ROOT));audit['panel_sha256']=sha(panel_path);audit['panel_bytes']=panel_path.stat().st_size
    audit['phase6_and_stage1_input_hashes']=used;audit['candidate_manifest_sha256']=sha(man_path)
    audit['selection_was_performance_blind']=True
    write_json(OUT/'external_data_audit.json',audit)
    write_json(OUT/'phase7_data_build_manifest.json',{'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script_sha256':sha(Path(__file__)),'panel_path':str(panel_path.relative_to(ROOT)),'panel_sha256':sha(panel_path),'input_hashes':used,'candidate_manifest_sha256':sha(man_path),'asset_eligibility_protocol_sha256':sha(OUT/'asset_eligibility_protocol.json'),'network_requests':0})
    print('BUILD_COMPLETE',json.dumps({'rows':len(panel),'panel_sha256':sha(panel_path),'candidate_eligibility':{a:audit['candidate_assets'][a]['eligibility'] for a in CANDIDATES}},indent=2),flush=True)

if __name__=='__main__': main()
