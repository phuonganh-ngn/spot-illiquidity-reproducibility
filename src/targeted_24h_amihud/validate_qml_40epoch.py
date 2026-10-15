from __future__ import annotations
import json
import numpy as np
from qml_models import fit_qnn
from common import OUT

SEED=81723
rng=np.random.default_rng(SEED)
n=160
X=rng.uniform(-1,1,size=(n,4))
fixtures={
 'linear':X[:,0]+.35*X[:,1]+rng.normal(0,.08,n),
 'nonlinear_interaction':X[:,0]*X[:,1]+.3*np.sin(np.pi*X[:,2])+rng.normal(0,.08,n),
 'null_noise':rng.normal(0,1,n),
}
results=[]
for name,y in fixtures.items():
 tr=np.arange(0,120); te=np.arange(120,n)
 for method,hybrid in [('qnn',False),('hybrid_qnn',True)]:
  p1,m=fit_qnn(X[tr],y[tr],X[te],4,SEED,epochs=40,batch_size=64,learning_rate=.01,hybrid=hybrid,encoder_components=16)
  p2,_=fit_qnn(X[tr],y[tr],X[te],4,SEED,epochs=40,batch_size=64,learning_rate=.01,hybrid=hybrid,encoder_components=16)
  finite=bool(np.isfinite(p1).all() and np.isfinite(p2).all())
  reproducible=bool(np.max(np.abs(p1-p2))<1e-6)
  results.append({'fixture':name,'method':method,'finite':finite,'reproducible_fixed_seed':reproducible,
   'max_abs_prediction_delta':float(np.max(np.abs(p1-p2))),
   'test_mae':float(np.mean(np.abs(y[te]-p1))),
   'training_mean_baseline_mae':float(np.mean(np.abs(y[te]-np.mean(y[tr])))),
   'metadata':m})
status='VALID' if all(x['finite'] and x['reproducible_fixed_seed'] for x in results) else 'VALIDATION_FAIL'
out={'status':status,'seed':SEED,'n_train':120,'n_test':40,'epochs':40,'qubits':4,
 'fixture_definitions':{'linear':'y=x0+0.35*x1+N(0,0.08^2)','nonlinear_interaction':'y=x0*x1+0.3*sin(pi*x2)+N(0,0.08^2)','null_noise':'y independent N(0,1)'},
 'results':results,'no_empirical_data_loaded':True}
(OUT/'qml_fixture_validation_40epoch.json').write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'status':status,'runs':len(results),'all_finite':all(r['finite'] for r in results),'all_reproducible':all(r['reproducible_fixed_seed'] for r in results)},indent=2))
