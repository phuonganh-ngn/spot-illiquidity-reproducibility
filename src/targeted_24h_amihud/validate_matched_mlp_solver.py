from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_qml_residual import fit_mlp
from common import OUT

SEED=81723
rng=np.random.default_rng(SEED)
X=rng.normal(size=(160,12))
fixtures={
 'linear':X[:,0]+.35*X[:,1]+rng.normal(0,.08,160),
 'nonlinear_interaction':X[:,0]*X[:,1]+.3*np.sin(np.pi*X[:,2])+rng.normal(0,.08,160),
 'null_noise':rng.normal(0,1,160),
}
rows=[]
for name,y in fixtures.items():
 a,ma=fit_mlp(X[:120],y[:120],X[120:],4,SEED)
 b,mb=fit_mlp(X[:120],y[:120],X[120:],4,SEED)
 rows.append({'fixture':name,'finite':bool(np.isfinite(a).all() and np.isfinite(b).all()),
   'reproducible_fixed_seed':bool(np.max(np.abs(a-b))<1e-10),
   'max_abs_prediction_delta':float(np.max(np.abs(a-b))),
   'test_mae':float(np.mean(np.abs(y[120:]-a))),
   'training_mean_baseline_mae':float(np.mean(np.abs(y[120:]-np.mean(y[:120])))),
   'metadata':ma})
out={'status':'VALID' if all(r['finite'] and r['reproducible_fixed_seed'] for r in rows) else 'VALIDATION_FAIL',
 'seed':SEED,'max_iter':2000,'tolerance':1e-5,'n_train':120,'n_test':40,'fixtures':rows,'no_empirical_data_loaded':True}
(OUT/'matched_mlp_solver_validation.json').write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'status':out['status'],'runs':len(rows),'converged':{r['fixture']:r['metadata']['converged'] for r in rows}},indent=2))
