"""Core utilities for the frozen Phase-6 predictive benchmark."""
from __future__ import annotations
import json, math, re, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/phase5_dynamic_discovery/dataset_4h.jsonl'
OUT=ROOT/'outputs/phase6_autonomous_benchmark'
SPLIT=pd.Timestamp('2023-07-01T00:00:00Z')
SAMPLE_END_EXCLUSIVE=pd.Timestamp('2024-03-04T00:00:00Z')
SPOT=['spot_return','spot_realized_volatility','spot_downside_semivariance','spot_amihud','spot_taker_imbalance','spot_quote_volume_growth']
DERIV=['funding_rate_state','oi_log_change','futures_taker_imbalance','basis_mean']
ALL=SPOT+DERIV
HORIZONS=(1,2,6)
TARGETS=('future_realized_volatility','future_downside_semivariance','future_amihud')
INFOSETS={
 'SPOT_ONLY':SPOT,
 'SPOT_PLUS_ALL_DERIVATIVES':ALL,
 'SPOT_PLUS_FUNDING':SPOT+['funding_rate_state'],
 'SPOT_PLUS_OI':SPOT+['oi_log_change'],
 'SPOT_PLUS_FUTURES_FLOW':SPOT+['futures_taker_imbalance'],
 'SPOT_PLUS_BASIS':SPOT+['basis_mean'],
}
HISTORY=6
DEV_FOLDS=[
 ('F1',pd.Timestamp('2022-10-01T00:00:00Z'),pd.Timestamp('2023-01-01T00:00:00Z')),
 ('F2',pd.Timestamp('2023-01-01T00:00:00Z'),pd.Timestamp('2023-04-01T00:00:00Z')),
 ('F3',pd.Timestamp('2023-04-01T00:00:00Z'),pd.Timestamp('2023-07-01T00:00:00Z')),
]
MODEL_CONFIGS={
 'Ridge':[{'alpha':0.1},{'alpha':1.0},{'alpha':10.0}],
 'RBF_SVR':[{'C':0.1,'gamma':'scale','epsilon':0.05},{'C':1.0,'gamma':'scale','epsilon':0.05},{'C':10.0,'gamma':'scale','epsilon':0.05}],
 'ExtraTrees':[{'n_estimators':200,'min_samples_leaf':2},{'n_estimators':200,'min_samples_leaf':5},{'n_estimators':200,'min_samples_leaf':10}],
 'HistGradientBoosting':[{'max_iter':100,'learning_rate':0.05,'max_leaf_nodes':7},{'max_iter':100,'learning_rate':0.05,'max_leaf_nodes':15},{'max_iter':100,'learning_rate':0.05,'max_leaf_nodes':31}],
 'MLP':[{'hidden_layer_sizes':(8,),'alpha':0.001,'max_iter':400},{'hidden_layer_sizes':(16,),'alpha':0.001,'max_iter':400},{'hidden_layer_sizes':(32,),'alpha':0.001,'max_iter':400}],
 'GRU':[{'hidden_size':4,'epochs':20,'weight_decay':0.001},{'hidden_size':8,'epochs':20,'weight_decay':0.001},{'hidden_size':16,'epochs':20,'weight_decay':0.001}],
}
SEED=20261007
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

def load_panel_dev_only():
    """Parse only development values; final-test payloads are skipped at line level."""
    import re
    pattern=re.compile(r'"period_start_utc":"([^"]+)"')
    rows=[]
    with SOURCE.open(encoding='utf-8') as f:
        for line in f:
            m=pattern.search(line)
            if not m: raise ValueError('missing period timestamp')
            if pd.Timestamp(m.group(1)) < SPLIT:
                rows.append(json.loads(line))
    df=pd.DataFrame(rows)
    df['period_start_utc']=pd.to_datetime(df.period_start_utc,utc=True)
    df['period_end_utc']=pd.to_datetime(df.period_end_utc,utc=True)
    return df

def load_panel_full_after_freeze():
    df=pd.read_json(SOURCE,lines=True)
    df['period_start_utc']=pd.to_datetime(df.period_start_utc,utc=True)
    df['period_end_utc']=pd.to_datetime(df.period_end_utc,utc=True)
    return df

def make_task(df,asset,horizon,target):
    g=df[df.asset==asset].sort_values('period_start_utc').reset_index(drop=True)
    xraw=g[ALL].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    spot=g[SPOT].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    rows=[]; Xseq=[]; ys=[]; dates=[]; ends=[]; persist=[]
    source_index={name:i for i,name in enumerate(SPOT)}
    for t in range(HISTORY-1,len(g)-horizon):
        seq=xraw[t-HISTORY+1:t+1]
        if not np.isfinite(seq).all(): continue
        fut_spot=spot[t+1:t+horizon+1]
        if not np.isfinite(fut_spot[:,[1,2,3]]).all(): continue
        rv=float(np.sqrt(np.square(fut_spot[:,1]).sum()))
        downside=float(fut_spot[:,2].sum())
        amihud=float(fut_spot[:,3].mean())
        y={'future_realized_volatility':rv,'future_downside_semivariance':downside,'future_amihud':amihud}[target]
        if not np.isfinite(y) or y<0: continue
        proxy={'future_realized_volatility':spot[t,1]*np.sqrt(horizon),
               'future_downside_semivariance':spot[t,2]*horizon,
               'future_amihud':spot[t,3]}[target]
        Xseq.append(seq); ys.append(y); dates.append(g.period_start_utc.iloc[t]); ends.append(g.period_start_utc.iloc[t+horizon]); persist.append(max(float(proxy),0.0))
    if not ys:
        return {'Xseq':np.empty((0,HISTORY,len(ALL))), 'y':np.array([]), 'date':pd.DatetimeIndex([]), 'target_end':pd.DatetimeIndex([]), 'persistence':np.array([])}
    return {'Xseq':np.asarray(Xseq,float),'y':np.asarray(ys,float),'date':pd.DatetimeIndex(dates),'target_end':pd.DatetimeIndex(ends),'persistence':np.asarray(persist,float)}

def feature_indices(names):
    idx={name:i for i,name in enumerate(ALL)}
    return [idx[v] for v in names]

def x_for_set(Xseq,names):
    inds=feature_indices(names)
    seq=Xseq[:,:,inds]
    flat=seq.reshape(len(seq),-1)
    return flat,seq

def fold_masks(task,fold):
    name,start,end=fold
    date=task['date']; target_end=task['target_end']
    train=(date<SPLIT)&(target_end<start)
    val=(date>=start)&(date<end)&(target_end<end)
    return np.asarray(train),np.asarray(val)

def safe_json(x):
    if isinstance(x,dict): return {str(k):safe_json(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [safe_json(v) for v in x]
    if isinstance(x,(np.integer,)): return int(x)
    if isinstance(x,(np.floating,float)): return float(x) if np.isfinite(x) else None
    if isinstance(x,(np.bool_,bool)): return bool(x)
    if isinstance(x,(pd.Timestamp,)): return x.isoformat()
    if isinstance(x,np.ndarray): return safe_json(x.tolist())
    return x

def json_write(path,obj):
    path.write_text(json.dumps(safe_json(obj),sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n',encoding='utf-8')

def logtarget(y): return np.log1p(np.maximum(np.asarray(y,float),0))
def pred_to_log(pred): return np.maximum(np.asarray(pred,float),0)

class TinyGRU(nn.Module):
    def __init__(self,n_features,hidden):
        super().__init__(); self.gru=nn.GRU(input_size=n_features,hidden_size=hidden,batch_first=True); self.head=nn.Linear(hidden,1)
    def forward(self,x):
        _,h=self.gru(x); return self.head(h[-1]).squeeze(-1)

def fit_predict(family,config,xtrain_flat,xtrain_seq,ytrain_log,xtest_flat,xtest_seq,seed):
    """Fold-local scaling and deterministic fit; returns log1p predictions and timing."""
    start=time.perf_counter()
    if family=='GRU':
        ntr,tlen,nf=xtrain_seq.shape
        seq_scaler=StandardScaler().fit(xtrain_seq.reshape(-1,nf))
        trs=seq_scaler.transform(xtrain_seq.reshape(-1,nf)).reshape(ntr,tlen,nf)
        nte=xtest_seq.shape[0]
        tes=seq_scaler.transform(xtest_seq.reshape(-1,nf)).reshape(nte,tlen,nf)
        torch.manual_seed(seed); np.random.seed(seed)
        net=TinyGRU(nf,int(config['hidden_size']))
        opt=torch.optim.Adam(net.parameters(),lr=0.005,weight_decay=float(config['weight_decay']))
        xt=torch.as_tensor(trs,dtype=torch.float32); yt=torch.as_tensor(ytrain_log,dtype=torch.float32)
        # Full-batch fixed-epoch training avoids stochastic loader order and keeps tuning bounded.
        for _ in range(int(config['epochs'])):
            net.train(); opt.zero_grad(set_to_none=True); loss=torch.mean((net(xt)-yt)**2); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): pred=net(torch.as_tensor(tes,dtype=torch.float32)).cpu().numpy()
    else:
        scaler=StandardScaler().fit(xtrain_flat)
        tr=scaler.transform(xtrain_flat); te=scaler.transform(xtest_flat)
    if family=='OLS':
        model=LinearRegression().fit(tr,ytrain_log); pred=model.predict(te)
    elif family=='Ridge':
        model=Ridge(alpha=config['alpha']).fit(tr,ytrain_log); pred=model.predict(te)
    elif family=='RBF_SVR':
        model=SVR(C=config['C'],gamma=config['gamma'],epsilon=config['epsilon']).fit(tr,ytrain_log); pred=model.predict(te)
    elif family=='ExtraTrees':
        model=ExtraTreesRegressor(n_estimators=config['n_estimators'],min_samples_leaf=config['min_samples_leaf'],
             max_features=1.0,random_state=seed,n_jobs=1).fit(tr,ytrain_log); pred=model.predict(te)
    elif family=='HistGradientBoosting':
        model=HistGradientBoostingRegressor(max_iter=config['max_iter'],learning_rate=config['learning_rate'],
             max_leaf_nodes=config['max_leaf_nodes'],l2_regularization=1.0,early_stopping=False,random_state=seed).fit(tr,ytrain_log); pred=model.predict(te)
    elif family=='MLP':
        model=MLPRegressor(hidden_layer_sizes=tuple(config['hidden_layer_sizes']),activation='relu',solver='adam',
             alpha=config['alpha'],learning_rate_init=0.001,max_iter=config['max_iter'],early_stopping=False,
             shuffle=True,random_state=seed,tol=1e-5,n_iter_no_change=30).fit(tr,ytrain_log); pred=model.predict(te)
    elif family=='GRU':
        pass
    else:
        raise ValueError(family)
    pred=pred_to_log(pred)
    if not np.isfinite(pred).all(): raise FloatingPointError(f'{family} generated non-finite predictions')
    return pred,float(time.perf_counter()-start)

def simple_baseline_predictions(task,train_idx,val_idx,target):
    ylog=logtarget(task['y'])
    mean=float(ylog[train_idx].mean())
    return {'training_mean':np.full(val_idx.sum(),mean),'persistence':logtarget(task['persistence'][val_idx])}

def metric_pack(y_true_log,pred_log,y_true_raw,target):
    err=np.asarray(pred_log)-np.asarray(y_true_log)
    rmse=float(np.sqrt(np.mean(err**2))); mae=float(np.mean(np.abs(err)))
    pred_raw=np.expm1(np.asarray(pred_log)); true_raw=np.asarray(y_true_raw)
    raw_mae=float(np.mean(np.abs(pred_raw-true_raw))); raw_rmse=float(np.sqrt(np.mean((pred_raw-true_raw)**2)))
    out={'mae_log1p':mae,'rmse_log1p':rmse,'mae_original':raw_mae,'rmse_original':raw_rmse}
    if target=='future_realized_volatility':
        variance=np.square(true_raw); predvar=np.maximum(np.square(pred_raw),1e-15)
        ratio=np.maximum(variance,1e-15)/predvar
        out['qlike_variance']=float(np.mean(ratio-np.log(ratio)-1))
    return out
