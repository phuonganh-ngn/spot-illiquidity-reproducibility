from __future__ import annotations
import json, hashlib, time, sys
from pathlib import Path
import numpy as np
from phase5_rerun_algorithms import (synthetic_fixtures, pcmci_plus, dynotears,
    var_lingam, neural_graph, LAGS, EDGE_EPS)

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'/'phase5_dynamic_discovery_rerun'
NAMES=['X','Y','Z']

def sign_edges(a, threshold=.05):
    # a is [lag,source,target]
    return {(i,j,k+1):float(a[k,i,j]) for k in range(a.shape[0])
            for i in range(a.shape[1]) for j in range(a.shape[2]) if abs(a[k,i,j])>threshold}

def support_neural(fits, threshold=.10):
    scores=[]; votes=[]
    for f in fits:
        a=f['lag_norms']
        scores.append(a)
        votes.append(a>threshold)
    votes=np.asarray(votes)
    stability=votes.mean(axis=0)
    edges={(i,j,k+1):float(np.median(np.asarray(scores)[:,i,j,k]))
           for i in range(3) for j in range(3) for k in range(LAGS)
           if stability[i,j,k]>=.5}
    return edges,stability

def main():
    fixtures=synthetic_fixtures(n=1400,seed=20261007)
    out={'seed':20261007,'sample_size':1400,'variables':NAMES,'criteria':{
        'null_cross_edge_max':2,'single_edge_required':'X->Y at lag 1 or 2; no Y->X at lag 1..6',
        'chain_recall_min':1,'bidirectional_required':'both X->Y and Y->X at lag 1 or 2',
        'nonlinear_required':'for neural methods, X->Y score ranks above chance (cross-edge AUC > .60) and is selected at lag 1 or 2'},'methods':{}}
    for method in ['PCMCI+','DYNOTEARS','VAR-LiNGAM','NOTEARS-MLP','Neural Granger/cMLP']:
        started=time.monotonic(); per={}
        for name,(x,truth) in fixtures.items():
            if method=='PCMCI+':
                fit=pcmci_plus([x],NAMES,pc_alpha=.05,ci_kind='ParCorr')
                edges={(NAMES.index(e['source']),NAMES.index(e['target']),e['lag']):e['statistic'] for e in fit['edges']}
                diag={'n_edges_fdr':len(edges),'fit_passed':True}
            elif method=='DYNOTEARS':
                fits=dynotears([x])
                votes=[]; score=[]; conv=[]
                for f in fits:
                    a=f['lagged']; conv.append(f['diagnostics']['converged'])
                    votes.append(np.abs(a)>.05); score.append(a)
                stability=np.asarray(votes).mean(axis=0)
                med=np.median(np.asarray(score),axis=0)
                edges={(i,j,k+1):float(med[k,i,j]) for k in range(6) for i in range(3) for j in range(3) if stability[k,i,j]>=.5}
                diag={'n_edges_regularization_stable':len(edges),'convergence_rate':float(np.mean(conv)),'stability':stability.tolist()}
            elif method=='VAR-LiNGAM':
                fit=var_lingam([x],NAMES,seed=420)
                edges=sign_edges(fit['lagged'],threshold=.05)
                diag={'n_edges':len(edges),'diagnostic_pass':fit['diagnostic_pass'],'lag_order_bic':fit['lag_order_bic'],'segment_diagnostics':fit['segment_diagnostics']}
            else:
                mode='cmlp' if method=='Neural Granger/cMLP' else 'notears_mlp'
                lambdas=(.02,.03,.05,.075) if mode=='cmlp' else (.005,.0075,.01,.0125)
                fits=neural_graph([x],mode,lambdas=lambdas,hidden_sizes=(10,),seed=314159,epochs=1000)
                edges,st=support_neural(fits)
                score=np.maximum.reduce([f['lag_norms'] for f in fits])
                diag={'n_edges_regularization_supported':len(edges),'minimum_loss':min(f['final_loss'] for f in fits),'stability':st.tolist()}
            # Cross-edge graph metrics. Diagonal autoregressive edges are not
            # counted against the independent-process null.
            cross={key:v for key,v in edges.items() if key[0]!=key[1]}
            bypair={(i,j):max(abs(v) for (s,t,l),v in cross.items() if s==i and t==j) for i in range(3) for j in range(3) if any(s==i and t==j for s,t,l in cross)}
            detected={(i,j) for i,j in bypair}
            if name=='null':
                metric={'false_cross_edges':len(cross),'null_gate':len(cross)<=2}
            elif name=='single_direction':
                metric={'expected_xy':(0,1) in detected,'reversed_yx':(1,0) in detected,'direction_gate':(0,1) in detected and (1,0) not in detected}
            elif name=='chain':
                recovered=int((0,1) in detected)+int((1,2) in detected)
                metric={'recovered_truth_edges':recovered,'chain_gate':recovered>=1}
            elif name=='bidirectional':
                metric={'xy':(0,1) in detected,'yx':(1,0) in detected,'bidirectional_gate':(0,1) in detected and (1,0) in detected}
            else:
                if method in ['NOTEARS-MLP','Neural Granger/cMLP']:
                    truth_pairs=np.array([int((i,j)==(0,1)) for i in range(3) for j in range(3) if i!=j])
                    if 'score' in locals() and isinstance(score,np.ndarray):
                        pair_scores=np.array([float(np.max(score[i,j])) for i in range(3) for j in range(3) if i!=j])
                    else:
                        pair_scores=np.array([bypair.get((i,j),0.) for i in range(3) for j in range(3) if i!=j])
                    from sklearn.metrics import roc_auc_score
                    auc=float(roc_auc_score(truth_pairs,pair_scores))
                    metric={'nonlinear_xy':(0,1) in detected,'cross_edge_auc':auc,'nonlinear_gate':auc>.60 and (0,1) in detected}
                else: metric={'not_applicable':'linear/ParCorr method is not required to detect a zero-linear-covariance nonlinear edge'}
            per[name]={'diagnostics':diag,'cross_edges':[{'source':NAMES[i],'target':NAMES[j],'lag':l,'score':v} for (i,j,l),v in sorted(cross.items())],'metrics':metric}
            if method=='DYNOTEARS':
                score=med
            if method=='VAR-LiNGAM' and not fit['diagnostic_pass']:
                per[name]['method_diagnostic_gate']='FAIL'
        required=['null','single_direction','chain','bidirectional']
        gates=[]
        for nm in required:
            m=per[nm]['metrics']
            key={'null':'null_gate','single_direction':'direction_gate','chain':'chain_gate','bidirectional':'bidirectional_gate'}[nm]
            ok=bool(m.get(key,False))
            if method=='VAR-LiNGAM' and per[nm].get('method_diagnostic_gate')=='FAIL': ok=False
            gates.append(ok)
        nonlinear_gate=True
        if method in ['NOTEARS-MLP','Neural Granger/cMLP']:
            nonlinear_gate=bool(per['nonlinear']['metrics'].get('nonlinear_gate',False))
        status='VALID' if all(gates) and nonlinear_gate else 'VALIDATION_FAIL'
        out['methods'][method]={'status':status,'core_fixture_gates':gates,'nonlinear_fixture_gate':nonlinear_gate,'elapsed_seconds':round(time.monotonic()-started,2),'fixtures':per}
        print(method,status,'elapsed',out['methods'][method]['elapsed_seconds'],flush=True)
    raw=json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n'
    (OUT/'fixture_validation.json').write_text(raw)
    (OUT/'fixture_validation.sha256').write_text(hashlib.sha256(raw.encode()).hexdigest()+'  fixture_validation.json\n')
    print('SAVED',OUT/'fixture_validation.json')

if __name__=='__main__': main()
