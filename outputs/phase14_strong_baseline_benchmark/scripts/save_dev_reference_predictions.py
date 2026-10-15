#!/usr/bin/env python3
"""Reconstruct fixed reference forecasts on already evaluated dev folds."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/"outputs/phase14_strong_baseline_benchmark"
sys.path.insert(0,str(ROOT/"src"));import phase6_benchmark_core as core
df=core.load_panel_dev_only()
rows=[]
for asset in ["BTC","ETH"]:
 task=core.make_task(df,asset,6,"future_amihud");y=core.logtarget(task["y"])
 for info in ["SPOT_ONLY","SPOT_PLUS_ALL_DERIVATIVES"]:
  for fold in core.DEV_FOLDS:
   name,start,end=fold;tr,va=core.fold_masks(task,fold)
   if (task["target_end"][tr]>=start).any() or (task["target_end"][va]>=end).any():raise SystemExit("fold purge failed")
   preds={"TrainingMean":np.full(int(va.sum()),float(y[tr].mean())),"Persistence":core.logtarget(task["persistence"][va])}
   for model,p in preds.items():
    dfout=pd.DataFrame({"asset":asset,"information_set":info,"family":model,"config_id":"fixed","fold":name,"origin":task["date"][va].astype(str),"target_end":task["target_end"][va].astype(str),"target_raw":task["y"][va],"target_log1p":y[va],"prediction_log1p":p})
    path=OUT/f"development_folds/{asset}_{info}_{model}_fixed_{name}.csv.gz";dfout.to_csv(path,index=False,compression="gzip")
    rows.extend(dfout.to_dict("records"))
pd.DataFrame(rows).to_csv(OUT/"predictions/development_reference_predictions.csv.gz",index=False,compression="gzip")
print("REFERENCE_FORECASTS_RECONSTRUCTED",len(rows),"rows; no models fitted; development-only loader")
