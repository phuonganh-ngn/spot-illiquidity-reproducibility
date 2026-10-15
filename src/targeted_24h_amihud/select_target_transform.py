from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from common import OUT,ASSETS,TARGET,load_panel,feature_frames,transform_target,loss_mae

TRANSFORMS=['T1_raw','T2_log1p','T3_log_eps','T4_log_scaled']
FOLDS=[
 {'fold':'T1','start':'2022-04-01T00:00:00Z','end':'2022-06-01T00:00:00Z'},
 {'fold':'T2','start':'2022-06-01T00:00:00Z','end':'2022-08-01T00:00:00Z'},
 {'fold':'T3','start':'2022-08-01T00:00:00Z','end':'2022-10-01T00:00:00Z'},
]
panel=load_panel(); results=[]
for asset in ASSETS:
 g,f0,f1,y=feature_frames(panel[panel.asset==asset])
 ok=np.isfinite(f0.drop(columns='date').to_numpy(float)).all(1) & np.isfinite(y.to_numpy(float))
 for fd in FOLDS:
  start=pd.Timestamp(fd['start']); end=pd.Timestamp(fd['end']); cut=start-pd.Timedelta(hours=24); val_end=end-pd.Timedelta(hours=24)
  train=ok & (g.period_start_utc>=pd.Timestamp('2021-12-01T00:00:00Z')) & (g.period_start_utc<cut)
  val=ok & (g.period_start_utc>=start) & (g.period_start_utc<val_end)
  if train.sum()<150 or val.sum()<100: raise RuntimeError(f'Insufficient transform-selection rows {asset}/{fd}: {train.sum()} {val.sum()}')
  xtr=f0.loc[train].drop(columns='date').to_numpy(float); xv=f0.loc[val].drop(columns='date').to_numpy(float)
  ytr=y.loc[train].to_numpy(float); yv=y.loc[val].to_numpy(float)
  for tr in TRANSFORMS:
   yt=transform_target(ytr,tr); yv_t=transform_target(yv,tr)
   scale=float(np.subtract(*np.quantile(yt,[.75,.25])))
   if not np.isfinite(scale) or scale<=0: raise RuntimeError(f'Invalid training IQR {asset}/{fd}/{tr}')
   model=make_pipeline(StandardScaler(),Ridge(alpha=1.0))
   model.fit(xtr,yt); pred=model.predict(xv)
   score=loss_mae(yv_t,pred)/scale
   results.append({'asset':asset,'fold':fd['fold'],'transform':tr,'n_train':int(train.sum()),'n_validation':int(val.sum()),'train_first':g.loc[train,'period_start_utc'].min().isoformat(),'train_last':g.loc[train,'period_start_utc'].max().isoformat(),'validation_first':g.loc[val,'period_start_utc'].min().isoformat(),'validation_last':g.loc[val,'period_start_utc'].max().isoformat(),'train_transformed_target_iqr':scale,'normalized_transformed_mae':score})
means={tr:float(np.mean([r['normalized_transformed_mae'] for r in results if r['transform']==tr])) for tr in TRANSFORMS}
selected=min(TRANSFORMS,key=lambda t:(means[t],TRANSFORMS.index(t)))
out={'status':'FROZEN_BEFORE_MAIN_MODEL_COMPARISON','selected_transform':selected,'selection_rule':'minimum equal-weight mean normalized transformed-target MAE over 2 assets x 3 expanding calibration folds; fixed Ridge(alpha=1), StandardScaler fit in fold only; tie order T1,T2,T3,T4','candidate_scores':means,'fold_results':results,'constants':{'T3_epsilon':1e-8,'T4_c':10000.0,'T4_epsilon':1e-8},'calibration_period':'2021-12-01 through 2022-09-30; validation labels end within each calibration block','outer_evaluation_data_not_used':True,'primary_model_comparison_not_run':True}
(OUT/'target_transform_selection.json').write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'selected_transform':selected,'candidate_scores':means,'primary_comparison_not_run':True},indent=2))
