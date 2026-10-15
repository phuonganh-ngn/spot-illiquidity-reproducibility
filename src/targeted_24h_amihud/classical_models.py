from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import time, warnings, hashlib
import numpy as np
import torch
from torch import nn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor,HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from threadpoolctl import threadpool_limits

class GRUNet(nn.Module):
 def __init__(self,input_size,static_size,hidden_size):
  super().__init__(); self.gru=nn.GRU(input_size,hidden_size,batch_first=True)
  self.head=nn.Linear(hidden_size+static_size,1)
 def forward(self,x,static):
  _,h=self.gru(x); z=torch.cat([h[-1],static],dim=1); return self.head(z).squeeze(-1)

def stable_seed(*parts):
 b='|'.join(map(str,parts)).encode(); return int(hashlib.sha256(b).hexdigest()[:8],16)&0x7fffffff

def _sk_model(family,cfg,seed):
 c=cfg.copy(); scaler=c.pop('scaler',None)
 if family=='ElasticNet': model=ElasticNet(**c,random_state=seed,selection='cyclic')
 elif family=='RBF_SVR': model=SVR(**c)
 elif family=='ExtraTrees': model=ExtraTreesRegressor(**c,random_state=seed)
 elif family=='HistGradientBoosting': model=HistGradientBoostingRegressor(**c,random_state=seed)
 elif family=='XGBoost': model=XGBRegressor(**c,random_state=seed,verbosity=0)
 elif family=='LightGBM': model=LGBMRegressor(**c,random_state=seed,deterministic=True,force_col_wise=True)
 elif family=='CatBoost': model=CatBoostRegressor(**c,random_seed=seed)
 elif family=='MLP': model=MLPRegressor(**c,random_state=seed,solver='adam',activation='relu',learning_rate='adaptive')
 else: raise KeyError(family)
 if family in ['ElasticNet','RBF_SVR','MLP']:
  z=StandardScaler() if scaler=='standard' else RobustScaler()
  model=make_pipeline(z,model)
 return model

def fit_predict(family,cfg,xtrain,ytrain,xtest,seq_train=None,seq_test=None,static_train=None,static_test=None,seed=20261007):
 t0=time.perf_counter(); caught=[]
 with threadpool_limits(limits=1):
  if family!='GRU':
   model=_sk_model(family,cfg,seed)
   with warnings.catch_warnings(record=True) as ws:
    warnings.simplefilter('always'); model.fit(xtrain,ytrain); pred=model.predict(xtest)
   caught=[f'{type(w.message).__name__}:{str(w.message)[:180]}' for w in ws]
  else:
   torch.set_num_threads(1); torch.manual_seed(seed); np.random.seed(seed)
   torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
   torch.use_deterministic_algorithms(True)
   device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   st=np.asarray(seq_train,dtype=np.float64); sv=np.asarray(seq_test,dtype=np.float64)
   # Training-only channel scaler, shared over the six sequence positions.
   mean=st.reshape(-1,st.shape[-1]).mean(0); sd=st.reshape(-1,st.shape[-1]).std(0,ddof=1); sd=np.where(sd>0,sd,1.)
   st=((st-mean)/sd).astype(np.float32); sv=((sv-mean)/sd).astype(np.float32)
   qtr=np.asarray(static_train,dtype=np.float64); qte=np.asarray(static_test,dtype=np.float64)
   qm=qtr.mean(0); qs=qtr.std(0,ddof=1); qs=np.where(qs>0,qs,1.)
   qtr=((qtr-qm)/qs).astype(np.float32); qte=((qte-qm)/qs).astype(np.float32)
   y=np.asarray(ytrain,dtype=np.float32); ym=float(y.mean()); ys=float(y.std(ddof=1)) or 1.; y=(y-ym)/ys
   model=GRUNet(st.shape[-1],qtr.shape[-1],cfg['hidden_size']).to(device)
   opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
   loss=nn.MSELoss(); rng=np.random.default_rng(seed)
   tx=torch.as_tensor(st,device=device); ts=torch.as_tensor(qtr,device=device); ty=torch.as_tensor(y,device=device)
   model.train()
   for _ in range(cfg['epochs']):
    order=rng.permutation(len(ty))
    for s in range(0,len(order),cfg['batch_size']):
     ids=torch.as_tensor(order[s:s+cfg['batch_size']],device=device)
     opt.zero_grad(set_to_none=True); yp=model(tx[ids],ts[ids]); l=loss(yp,ty[ids]); l.backward(); opt.step()
   model.eval()
   with torch.no_grad():
    yp=model(torch.as_tensor(sv,device=device),torch.as_tensor(qte,device=device)).cpu().numpy()
   pred=yp*ys+ym; caught=[]
 return np.asarray(pred,dtype=float),time.perf_counter()-t0,caught


def fixture_smoke():
 # Small deterministic synthetic smoke test for every non-QML estimator family.
 rng=np.random.default_rng(4401); x=rng.normal(size=(100,8)); y=x[:,0]-.4*x[:,1]+rng.normal(0,.1,100)
 from pathlib import Path
 import json
 from common import OUT
 reg=json.loads((OUT/'model_registry.json').read_text())['families']; results=[]
 for fam,configs in reg.items():
  cfg=configs[0]; a=x[:80]; b=x[80:]; ya=y[:80]
  kw={}
  if fam=='GRU':
   kw={'seq_train':a.reshape(80,1,8).repeat(6,axis=1),'seq_test':b.reshape(20,1,8).repeat(6,axis=1),'static_train':np.zeros((80,2)),'static_test':np.zeros((20,2))}
  try:
   p1,_,_=fit_predict(fam,cfg,a,ya,b,seed=8271,**kw)
   if fam=='GRU': kw2=kw
   else: kw2={}
   p2,_,_=fit_predict(fam,cfg,a,ya,b,seed=8271,**kw2)
   results.append({'family':fam,'finite':bool(np.isfinite(p1).all()),'repeat_same_seed_max_delta':float(np.max(np.abs(p1-p2)))})
  except Exception as e: results.append({'family':fam,'finite':False,'error':f'{type(e).__name__}: {e}'})
 return results
