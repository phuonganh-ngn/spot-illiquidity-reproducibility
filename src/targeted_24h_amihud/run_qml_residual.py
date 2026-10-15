from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from threadpoolctl import threadpool_limits

from common import OUT, ASSETS, DER, STATIC_DER, CAL, load_panel, feature_frames
from qml_models import fit_quantum_kernel, fit_rbf_nystrom, fit_qnn, _preprocess
from classical_models import stable_seed

SEED=20261007
QUBITS=[4,6,8]
SEEDS=[101,211,307,401,503]
BOOT=1000

def dump(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
def moving_ids(dates,L,n,rng):
    d=pd.to_datetime(pd.Series(dates),utc=True)
    # Match the divisor to pandas' stored datetime resolution (us in pandas 3).
    tick=4*3600*{'s':1,'ms':10**3,'us':10**6,'ns':10**9}[d.dtype.unit]
    ix=d.astype('int64').to_numpy()//tick; first=int(ix.min()); last=int(ix.max())
    starts=np.arange(first,max(first+1,last-L+2)); starts=starts if len(starts) else np.array([first]); out=[]
    while len(out)<n:
        s=int(rng.choice(starts)); out.extend(np.flatnonzero((ix>=s)&(ix<s+L)).tolist())
    return np.asarray(out[:n],int)
def bootstrap(dates,delta,seed):
    d=np.asarray(dates); delta=np.asarray(delta,float); rng=np.random.default_rng(seed); out={}
    for L in [42,84]:
        vals=np.empty(BOOT)
        for b in range(BOOT): vals[b]=delta[moving_ids(d,L,len(d),rng)].mean()
        n_le=int(np.sum(vals<=0)); n_ge=int(np.sum(vals>=0))
        p_two=float(min(1.,2*(1+min(n_le,n_ge))/(BOOT+1)))
        out[str(L)]={'replications':BOOT,'block_length':L,'mean_delta':float(delta.mean()),'ci95':[float(x) for x in np.quantile(vals,[.025,.975])],'p_two_sided':p_two,'probability_positive':float((vals>0).mean())}
    return out
def holm(ps):
    ps=np.asarray(ps,float); out=np.empty(len(ps)); last=0.
    for r,j in enumerate(np.argsort(ps)):
        last=max(last,min(1.,(len(ps)-r)*ps[j]));out[j]=last
    return out.tolist()
def loss_metrics(y,p):
    e=np.abs(np.asarray(y,float)-np.asarray(p,float)); return {'mae':float(e.mean()),'rmse':float(np.sqrt(np.mean((np.asarray(y,float)-np.asarray(p,float))**2)))}

def feature_matrix(panel,asset):
    g=panel[panel.asset==asset].sort_values('period_start_utc').reset_index(drop=True)
    g,f0,f1,y=feature_frames(g)
    dcols=[c for c in f1.columns if any(c.startswith(v+'__lag') for v in DER)]
    cols=dcols+STATIC_DER+CAL
    x=f1[cols].to_numpy(float)
    return g,cols,x

def fit_mlp(train_x,train_y,test_x,dim,seed,hybrid=False):
    comp=min(16 if hybrid else dim,train_x.shape[1],max(1,len(train_x)-1))
    ztr,zte,meta=_preprocess(train_x,test_x,comp,seed)
    y=np.asarray(train_y,float); ym=float(y.mean()); ys=float(y.std(ddof=1)) or 1.; ys=ys if ys>0 else 1.
    with threadpool_limits(limits=1):
        model=MLPRegressor(hidden_layer_sizes=(dim,),activation='relu',solver='adam',alpha=.01,
                           learning_rate_init=.001,max_iter=2000,tol=1e-5,early_stopping=False,
                           random_state=seed)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            model.fit(ztr,(y-ym)/ys)
        pred=model.predict(zte)*ys+ym
    return pred,meta|{'matched_classical':'one-hidden-layer MLP','input_components':comp,'hidden_width':dim,
                      'alpha':.01,'max_iter':2000,'tolerance':1e-5,'n_iter':int(model.n_iter_),
                      'converged':bool(model.n_iter_<2000),'warnings':[f'{type(w.message).__name__}:{str(w.message)[:180]}' for w in ws],
                      'seed':seed}

def main():
    plan=json.loads((OUT/'qml_residual_execution_plan.json').read_text())
    if plan.get('status')!='FROZEN_BEFORE_QML_EVALUATION': raise RuntimeError('QML execution plan is not frozen')
    fixture=json.loads((OUT/'qml_fixture_validation_40epoch.json').read_text())
    if fixture.get('status')!='VALID': raise RuntimeError('QML synthetic fixtures failed')
    # Train/test QML on the actual strict OOF spot-only residual target, not on
    # residual-model predictions produced in the separate classical benchmark.
    pred=pd.DataFrame(json.loads((OUT/'spot_oof_predictions.json').read_text())['predictions'])
    pred['date']=pd.to_datetime(pred.date,utc=True)
    pred['residual']=pred.target_transformed-pred.oof_spot_prediction_transformed
    panel=load_panel(); ck=OUT/'qml_residual_checkpoints.jsonl'; done={};
    if ck.exists():
        for line in ck.read_text().splitlines():
            try:
                r=json.loads(line); key=(r['asset'],r['method'],r['qubits'])
                if key in done: raise RuntimeError(f'duplicate QML checkpoint {key}')
                done[key]=r
            except json.JSONDecodeError: continue
    for asset in ASSETS:
        g,cols,X=feature_matrix(panel,asset); date_to_row={pd.Timestamp(d):i for i,d in enumerate(pd.to_datetime(g.period_start_utc,utc=True))}
        z=pred[pred.asset==asset].copy(); z['row']=[date_to_row[d] for d in z.date]
        train=z[z.fold.isin(['F1','F2'])].sort_values('date'); test=z[z.fold=='F3'].sort_values('date')
        if len(train)<100 or len(test)<100: raise RuntimeError(f'insufficient OOF residual rows for QML: {asset} train={len(train)} test={len(test)}')
        tri=train.row.to_numpy(int); tei=test.row.to_numpy(int); xtr=X[tri]; xte=X[tei]; ytr=train.residual.to_numpy(float); yte=test.residual.to_numpy(float)
        base=test.oof_spot_prediction_transformed.to_numpy(float); true=test.target_transformed.to_numpy(float); dates=test.date.astype(str).to_numpy()
        for q in QUBITS:
            # Quantum and RBF kernels share the exact split, PCA dimension and landmark rule.
            for method in ['quantum_kernel','matched_rbf']:
                key=(asset,method,q)
                if key in done: continue
                t0=time.time()
                if method=='quantum_kernel': p,meta=fit_quantum_kernel(xtr,ytr,xte,q,landmarks=128,alpha=.001,seed=stable_seed(SEED,asset,q,'kernel'))
                else: p,meta=fit_rbf_nystrom(xtr,ytr,xte,q,landmarks=128,alpha=.001,seed=stable_seed(SEED,asset,q,'kernel'))
                item={'asset':asset,'method':method,'qubits':q,'prediction_residual':p.tolist(),'meta':meta,'fit_seconds':time.time()-t0}
                with ck.open('a') as f: f.write(json.dumps(item,sort_keys=True,allow_nan=False)+'\n'); f.flush()
                done[key]=item
                print(f'completed {asset} {method} q={q}',flush=True)
            for method in ['qnn','hybrid_qnn']:
                key=(asset,method,q)
                if key in done: continue
                seed_preds=[]; seed_metas=[]; mlp_preds=[]; mlp_metas=[]; fit_seconds=0.
                for s in SEEDS:
                    qseed=stable_seed(SEED,asset,method,q,s)
                    t0=time.time(); qp,meta=fit_qnn(xtr,ytr,xte,q,qseed,epochs=40,batch_size=64,learning_rate=.01,hybrid=(method=='hybrid_qnn'),encoder_components=16); fit_seconds+=time.time()-t0
                    mp,mmeta=fit_mlp(xtr,ytr,xte,q,qseed,hybrid=(method=='hybrid_qnn'))
                    seed_preds.append(qp); seed_metas.append(meta); mlp_preds.append(mp); mlp_metas.append(mmeta)
                    print(f'completed {asset} {method} q={q} seed={s}',flush=True)
                item={'asset':asset,'method':method,'qubits':q,'seeds':SEEDS,'prediction_residual_by_seed':[p.tolist() for p in seed_preds],
                      'matched_mlp_prediction_by_seed':[p.tolist() for p in mlp_preds],'seed_metadata':seed_metas,'matched_mlp_metadata':mlp_metas,'fit_seconds':fit_seconds}
                with ck.open('a') as f: f.write(json.dumps(item,sort_keys=True,allow_nan=False)+'\n'); f.flush()
                done[key]=item
    # Convert checkpoint predictions into paired comparisons on identical F3 dates.
    qkernel=[]; qnn=[]; hybrid=[]; alltests=[]
    for asset in ASSETS:
        z=pred[(pred.asset==asset)&(pred.fold=='F3')].sort_values('date'); true=z.target_transformed.to_numpy(float); base=z.oof_spot_prediction_transformed.to_numpy(float); dates=z.date.astype(str).to_numpy()
        for q in QUBITS:
            qk=done[(asset,'quantum_kernel',q)]['prediction_residual']; rb=done[(asset,'matched_rbf',q)]['prediction_residual']
            dk=np.abs(true-(base+np.asarray(rb)))-np.abs(true-(base+np.asarray(qk)))
            qkernel.append({'asset':asset,'qubits':q,'n':len(true),'quantum_kernel_corrected_metrics':loss_metrics(true,base+np.asarray(qk)),'matched_rbf_corrected_metrics':loss_metrics(true,base+np.asarray(rb)),'delta_positive_favors_quantum':float(dk.mean()),'paired_bootstrap':bootstrap(dates,dk,stable_seed(SEED,asset,'qkernel',q))})
            for method,target in [('qnn',qnn),('hybrid_qnn',hybrid)]:
                item=done[(asset,method,q)]; qp=np.asarray(item['prediction_residual_by_seed'],float); mp=np.asarray(item['matched_mlp_prediction_by_seed'],float)
                qensemble=qp.mean(axis=0); mensemble=mp.mean(axis=0)
                seed_delta=[]
                for a,b in zip(qp,mp): seed_delta.append(float(np.mean(np.abs(true-(base+b))-np.abs(true-(base+a)))))
                # Mean seed-level loss per date, so the comparison does not reward ensembling alone.
                qloss=np.mean(np.abs(true[None,:]-(base[None,:]+qp)),axis=0)
                mloss=np.mean(np.abs(true[None,:]-(base[None,:]+mp)),axis=0)
                delta=mloss-qloss
                target.append({'asset':asset,'qubits':q,'n':len(true),'seed_deltas_positive_favor_quantum':seed_delta,'seed_delta_median':float(np.median(seed_delta)),'seed_delta_mean':float(np.mean(seed_delta)),'seed_fraction_positive':float(np.mean(np.asarray(seed_delta)>0)),'quantum_seed_ensemble_metrics':loss_metrics(true,base+qensemble),'matched_mlp_seed_ensemble_metrics':loss_metrics(true,base+mensemble),'mean_seed_loss_delta_positive_favors_quantum':float(delta.mean()),'paired_bootstrap':bootstrap(dates,delta,stable_seed(SEED,asset,method,q,'paired')),'qnn_metadata':item['seed_metadata'],'matched_mlp_metadata':item['matched_mlp_metadata']})
    for family,rows in [('quantum_kernel',qkernel),('qnn',qnn),('hybrid_qnn',hybrid)]:
        for asset in ASSETS:
            rr=[r for r in rows if r['asset']==asset]; adj=holm([r['paired_bootstrap']['42']['p_two_sided'] for r in rr])
            for r,p in zip(rr,adj): r['holm_p_within_asset_family']=p
    dump('quantum_kernel_results.json',{'status':'COMPLETE','evaluation':'F3 only; residual training F1+F2, no F3 tuning','matched_design':'same rows and dimensions; quantum fidelity kernel vs RBF Nyström with identical 128 training landmarks','results':qkernel})
    dump('qnn_results.json',{'status':'COMPLETE','evaluation':'F3 only; residual training F1+F2; five-seed ensemble and per-seed metrics; no best-seed selection','results':qnn})
    dump('hybrid_qnn_results.json',{'status':'COMPLETE','evaluation':'F3 only; residual training F1+F2; five-seed ensemble and per-seed metrics; no best-seed selection','results':hybrid})
    dump('quantum_vs_classical.json',{'status':'COMPLETE','rule':'QML versus matched kernel/MLP at identical split and PCA dimensions. Positive delta favors QML. Holm corrected across q=4/6/8 separately by asset and family. No hardware quantum-advantage claim.','quantum_kernel':qkernel,'qnn':qnn,'hybrid_qnn':hybrid})
    dump('qml_paired_bootstrap.json',{'replications':BOOT,'block_lengths':[42,84],'seed':SEED,'results':{'quantum_kernel':qkernel,'qnn':qnn,'hybrid_qnn':hybrid}})
    print(json.dumps({'status':'COMPLETE','qkernel_comparisons':len(qkernel),'qnn_comparisons':len(qnn),'hybrid_comparisons':len(hybrid)}))

if __name__=='__main__': main()
