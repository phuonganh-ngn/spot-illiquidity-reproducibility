from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs'/'targeted_24h_amihud'
PANEL=OUT/'analysis_panel.csv.gz'
ASSETS=['BTC','ETH']
SPOT=['spot_return','spot_realized_volatility','spot_downside_semivariance','spot_amihud','spot_taker_imbalance','spot_quote_volume_growth']
DER=['funding_rate_state','oi_log_change','futures_taker_imbalance','basis_mean']
STATIC_DER=['funding_change','basis_change','oi_shock_30d_robust','flow_shock_30d_robust','funding_x_oi_shock','oi_shock_x_flow_shock','funding_x_basis','flow_x_basis']
CAL=[*[f'utc_slot_{i}' for i in range(1,6)],*[f'weekday_{i}' for i in range(1,7)],'funding_schedule_window']
TARGET='future_amihud_24h_mean'

def load_panel():
 d=pd.read_csv(PANEL,parse_dates=['period_start_utc'],low_memory=False)
 d['period_start_utc']=pd.to_datetime(d.period_start_utc,utc=True)
 return d.sort_values(['asset','period_start_utc']).reset_index(drop=True)

def feature_frames(g):
 g=g.sort_values('period_start_utc').reset_index(drop=True).copy()
 date=g.period_start_utc
 f0=pd.DataFrame(index=g.index)
 f0['date']=date
 # Chronological history from t-5 through t.
 for lag in range(5,-1,-1):
  for c in SPOT:
   f0[f'{c}__lag{lag}']=g[c].shift(lag)
 for c in CAL: f0[c]=g[c]
 f1=f0.copy()
 for lag in range(5,-1,-1):
  for c in DER:
   f1[f'{c}__lag{lag}']=g[c].shift(lag)
 for c in STATIC_DER: f1[c]=g[c]
 y=g[TARGET].rename(TARGET)
 return g,f0,f1,y

def complete_mask(f0,f1,y):
 return np.isfinite(f0.drop(columns='date').to_numpy(float)).all(1) & np.isfinite(f1.drop(columns='date').to_numpy(float)).all(1) & np.isfinite(y.to_numpy(float))

def transform_target(y, name):
 y=np.asarray(y,dtype=float)
 if name=='T1_raw': return y.copy()
 if name=='T2_log1p': return np.log1p(y)
 if name=='T3_log_eps': return np.log(y+1e-8)
 if name=='T4_log_scaled': return np.log(10000.0*y+1e-8)
 raise KeyError(name)

def transform_by_name(y,name): return transform_target(y,name)

def get_target_transform():
 p=json.loads((OUT/'target_transform_selection.json').read_text())
 return p['selected_transform']

def loss_mae(y,p): return float(np.mean(np.abs(np.asarray(y)-np.asarray(p))))
