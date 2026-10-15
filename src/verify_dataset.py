"""Boundary, mathematical identity and immutable-cache tests. No modeling."""
import json,hashlib,inspect
from pathlib import Path
import pandas as pd,numpy as np
from common import *
from validate_semantics import read
from build_future_outcomes import future,HORIZONS,CORE

def assert_equal(a,b,label):
 assert np.allclose(np.asarray(a,dtype=float),np.asarray(b,dtype=float),equal_nan=True,rtol=1e-9,atol=1e-11),label

def main():
 tests=[]
 def check(name,fn):
  fn();tests.append(dict(test=name,status='PASS'))
 p=pd.read_csv(ROOT/'data/processed/derivatives_spot_daily_panel.csv');p['date']=pd.to_datetime(p.date,utc=True);wide=pd.read_csv(ROOT/'data/processed/derivatives_spot_system_daily.csv');full=pd.read_pickle(ROOT/'data/interim/full_analytical_panel.pkl');config=json.loads((ROOT/'outputs/sample_definition.json').read_text())
 def unique():
  assert not p.duplicated(['date','asset']).any();assert not wide.date.duplicated().any();assert set(p.asset)=={'BTC','ETH'};assert p.groupby('date').asset.nunique().eq(2).all()
 check('Panel and wide key uniqueness; BTC ETH date grouping',unique)
 def bounds():
  assert p.date.min()==pd.Timestamp(config['primary_clean_start'],tz='UTC');assert p.date.max()==pd.Timestamp(config['primary_clean_end'],tz='UTC');assert len(p)==2*len(pd.date_range(config['primary_clean_start'],config['primary_clean_end']));assert p.date.dt.hour.eq(0).all()
 check('UTC alignment and explicit sample range',bounds)
 def immutable():
  inv=pd.read_csv(ROOT/'outputs/raw_hash_inventory.csv')
  for r in inv.itertuples():assert sha(ROOT/r.path)==r.sha256,'Raw hash changed '+r.path
  prior=json.loads((ROOT/'outputs/preexisting_project_hashes.json').read_text());parent=ROOT.parent
  for path,digest in prior.items():assert sha(parent/path)==digest,'Previous research file modified '+path
 check('Raw immutability and preservation of prior project',immutable)
 def archive():
  for sym in SYMBOLS:
   for market,dataset,freq in [('spot','klines','5m'),('futures','klines','5m'),('futures','metrics','5m'),('futures','fundingRate','FUNDING_SETTLEMENT')]:
    d=read(sym,market,dataset,freq);assert d.index.is_unique;assert d.index.tz is not None
 check('Validated raw timestamp uniqueness and UTC',archive)
 def identity():
  for sym in SYMBOLS:
   g=full[full.asset==sym[:-4]].set_index('date');f=read(sym,'futures','fundingRate','FUNDING_SETTLEMENT');group=f.last_funding_rate.groupby(f.index.floor('D'));r=group.sum().reindex(g.index).where(group.count().reindex(g.index)==3)
   assert_equal(g.funding_sum,r,'Funding actual timestamp sum')
   assert_equal(g.funding_mean,r/3,'Funding rate mean')
   assert_equal(g.oi_log_change_1d,np.log(g.oi_close/g.oi_close.shift(1)),'OI log change')
   assert_equal(g.spot_log_return_1d,np.log(g.spot_close/g.spot_close.shift(1)),'Spot log return')
   assert_equal(g.spot_amihud,1e6*g.spot_log_return_1d.abs()/g.spot_quote_volume.where(g.spot_quote_volume>0),'USDT Amihud')
   for layer in ['spot','futures']:
    imb=g[layer+'_taker_imbalance'];assert imb.dropna().between(-1-1e-12,1+1e-12).all();assert_equal(imb,(2*g[layer+'_taker_buy_base_volume']-g[layer+'_base_volume'])/g[layer+'_base_volume'].where(g[layer+'_base_volume']>0),'Taker volume partition')
   # Independent spot RV reconstruction over an actual complete day, including previous close.
   day=pd.Timestamp('2022-01-15',tz='UTC');raw=read(sym,'spot','klines');idx=pd.date_range(day-pd.Timedelta(minutes=5),day+pd.Timedelta(days=1),freq='5min',inclusive='left');r=np.log(raw.close.reindex(idx)).diff().iloc[1:]
   assert len(r)==288 and r.notna().all();assert_equal([g.loc[day,'spot_realized_volatility']],[np.sqrt(np.square(r).sum())],'Intraday RV')
   assert_equal([g.loc[day,'spot_downside_semivariance']],[np.square(r.clip(upper=0)).sum()],'Intraday downside')
   mark=read(sym,'futures','markPriceKlines');index=read(sym,'futures','indexPriceKlines');intraday=mark.close/index.close-1;values=intraday.loc[day:day+pd.Timedelta(days=1)-pd.Timedelta(microseconds=1)];assert len(values)==288
   assert_equal([g.loc[day,'basis_mean']],[values.mean()],'Basis closes ratio')
 check('Funding OI spot return RV downside basis and imbalance identities',identity)
 def windows():
  for asset,g in full.groupby('asset'):
   g=g.set_index('date')
   for h in HORIZONS:
    assert_equal(g[f'spot_return_cumulative_fwd_{h}d'],future(g.spot_log_return_1d,h,'sum'),'Forward return direction')
    assert_equal(g[f'spot_realized_volatility_fwd_{h}d'],future(g.spot_realized_volatility,h,'rms'),'Future RV root sum')
    assert_equal(g[f'funding_cumulative_fwd_{h}d'],future(g.funding_sum,h,'sum'),'Future funding sum')
    end=pd.Timestamp(END,tz='UTC');assert g.loc[g.index>end-pd.Timedelta(days=h),f'spot_return_cumulative_fwd_{h}d'].isna().all()
   # A tagged sequence checks direction without mirroring financial features.
  tagged=pd.Series(np.arange(40,dtype=float));assert future(tagged,3,'sum').iloc[10]==11+12+13;assert np.isnan(future(tagged,3,'sum').iloc[-3]);tagged.iloc[12]=np.nan;assert np.isnan(future(tagged,3,'sum').iloc[10])
  for h in HORIZONS:
   bad=p.date+pd.Timedelta(days=h)>pd.Timestamp(config['primary_clean_end'],tz='UTC');assert not p.loc[bad,f'eligibility_h{h}'].any();assert p.loc[bad,[c for c in p if c.endswith(f'_fwd_{h}d')]].isna().all().all()
 check('Strictly future windows missingness and primary/extended end truncation',windows)
 def controls():
  dictionary=pd.read_csv(ROOT/'outputs/variable_dictionary.csv');controls=dictionary[dictionary.role_candidate.eq('predetermined_control')];assert not controls.source_dataset.str.contains('fwd').any()
  for asset,g in full.groupby('asset'):
   g=g.set_index('date')
   for c in g:
    if '_lag' in c and '_rolling_' not in c:
     stem,lag=c.rsplit('_lag',1)
     if stem in g and lag.isdigit():assert_equal(g[c],g[stem].shift(int(lag)),'Lag direction '+c)
   # Perturb a future value: earlier controls MUST be invariant.
   s=g.spot_log_return_1d.copy();cut=100;before=s.rolling(30,min_periods=30).std().shift(1).iloc[:cut];s.iloc[cut:]=999;after=s.rolling(30,min_periods=30).std().shift(1).iloc[:cut];assert_equal(before,after,'Future perturbation must not enter past controls')
  for filename in ['build_daily_features.py','build_future_outcomes.py']:
   source=(ROOT/'src'/filename).read_text();assert 'center=True' not in source;assert 'StandardScaler' not in source
 check('Predetermined controls no centered windows no future/full-sample scaling',controls)
 def timing():
  for sym in SYMBOLS:
   k=read(sym,'spot','klines');valid=k.close.notna();assert (k.loc[valid,'interval_end']<k.loc[valid].index.floor('D')+pd.Timedelta(days=1)).all()
   api=pd.read_csv(ROOT/'outputs/api_archive_crosscheck.csv');selected=api[api.dataset.eq('oi_api_validation')&api.offset_seconds.eq(-300)];assert len(selected)==4 and selected.status.eq('VERIFIED').all()
   funds=api[api.dataset.eq('funding_api_validation')];assert len(funds)==10 and funds.status.eq('VERIFIED').all()
   layout=pd.read_csv(ROOT/'outputs/metrics_daily_layout_audit.csv');assert not layout[(layout.date<=config['primary_clean_end'])&(layout.layout=='START_LABEL')].shape[0]
 check('Daily boundaries settlement API mapping and empirically aligned OI',timing)
 def prices():
  c=pd.read_csv(ROOT/'outputs/daily_price_crosscheck.csv');assert len(c)>=2*5*4 and c.status.eq('PASS').all(),c[c.status!='PASS'].to_string()
 check('Deterministic reconstructed vs official daily candles',prices)
 def coverage():
  c=pd.read_csv(ROOT/'outputs/model_variable_coverage.csv');core=c[(c['sample']=='PRIMARY_CLEAN')&c.variable.isin(CORE)];assert len(core)==12 and core.coverage_pct.ge(95).all()
 check('Both assets required core >=95% with independent futures imbalance',coverage)
 pd.DataFrame(tests).to_csv(ROOT/'outputs/automated_validation_tests.csv',index=False)
 print('PASS',len(tests),'required validation groups')
if __name__=='__main__':main()
