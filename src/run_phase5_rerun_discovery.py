from __future__ import annotations
import json, time, traceback, hashlib
from pathlib import Path
import numpy as np
from phase5_rerun_common import OUT, VARS, ASSETS, sha, log, write_json, load_asset_discovery
from phase5_rerun_algorithms import pcmci_plus, dynotears, var_lingam, neural_graph, LAMBDAS

def matrix_edge_records(arr,names,threshold,lag_offset=1):
    a=np.asarray(arr,float);out=[]
    if a.ndim==2:a=a[None,:,:]
    for k in range(a.shape[0]):
        for i,src in enumerate(names):
            for j,dst in enumerate(names):
                w=float(a[k,i,j])
                if abs(w)>=threshold:
                    out.append({'source':src,'target':dst,'lag':k+lag_offset,'weight':w})
    return out

def fit_pcmci(segments):
    fits=[]
    for alpha in (.01,.05,.10):
        r=pcmci_plus(segments,VARS,pc_alpha=alpha,ci_kind='ParCorr')
        fits.append(r)
        log(f"PCMCI+ alpha={alpha} observations={r['n_observations']} q_edges={len(r['edges'])}")
    counts={}
    for r in fits:
        for e in r['edges']:
            k=(e['source'],e['target'],e['lag'])
            counts.setdefault(k,[]).append({'pc_alpha':r['pc_alpha'],'p':e['p'],'q_bh':e['q_bh'],'statistic':e['statistic']})
    edges=[]
    for (src,dst,lag),v in sorted(counts.items()):
        signs=[int(np.sign(x['statistic'])) for x in v if x['statistic']!=0]
        edges.append({'source':src,'target':dst,'lag':lag,'support_count':len(v),
                      'supporting_pc_alpha':[x['pc_alpha'] for x in v],
                      'median_statistic':float(np.median([x['statistic'] for x in v])),
                      'sign_consistent':len(set(signs))<=1,
                      'stable_across_alpha':len(v)>=2 and len(set(signs))<=1,
                      'runs':v})
    return {'status':'FIT_COMPLETE','ci_test':'ParCorr','alpha_levels':[.01,.05,.10],
            'fdr':'Benjamini-Hochberg across all lagged source-target-lag hypotheses within each run',
            'run_summaries':[{'pc_alpha':r['pc_alpha'],'n_segments':r['n_segments'],'n_observations':r['n_observations'],
                              'p_matrix':r['p_matrix'],'q_matrix':r['q_matrix'],'val_matrix':r['val_matrix'],
                              'edge_count':len(r['edges'])} for r in fits],
            'candidate_edges':edges}

def fit_dynotears(segments):
    fits=dynotears(segments,LAMBDAS)
    edges_by={}
    for f in fits:
        current=matrix_edge_records(f['lagged'],VARS,.05)
        for e in current:
            k=(e['source'],e['target'],e['lag'])
            edges_by.setdefault(k,[]).append({'lambda':f['lambda'],'weight':e['weight']})
    edges=[]
    for (src,dst,lag),v in sorted(edges_by.items()):
        signs={int(np.sign(x['weight'])) for x in v if x['weight']!=0}
        edges.append({'source':src,'target':dst,'lag':lag,'support_count':len(v),
                      'supporting_lambdas':[x['lambda'] for x in v],
                      'median_weight':float(np.median([x['weight'] for x in v])),
                      'sign_consistent':len(signs)<=1,'regularization_stable':len(v)>=2 and len(signs)<=1})
    return {'status':'FIT_COMPLETE','implementation':'CausalNex reference DYNOTEARS objective, local port',
            'lambda_grid':list(LAMBDAS),'fits':[{'lambda':f['lambda'],'contemporaneous':f['contemporaneous'],
            'lagged':f['lagged'],'diagnostics':f['diagnostics']} for f in fits],
            'candidate_edges':edges,
            'convergence_rate':float(np.mean([f['diagnostics']['converged'] for f in fits]))}

def fit_neural(segments,mode):
    if mode=='notears_mlp':
        lambdas=(.005,.0075,.01,.0125);hidden=(10,20);seed=314159;label='temporal NOTEARS-MLP'
    else:
        lambdas=(.02,.03,.05,.075);hidden=(10,);seed=271828;label='Neural Granger/cMLP'
    fits=neural_graph(segments,mode,lambdas=lambdas,hidden_sizes=hidden,seed=seed,epochs=1000)
    records=[]
    config_edges={}
    for f in fits:
        norms=np.asarray(f['lag_norms']); deriv=np.asarray(f['local_derivative'])
        record={'hidden':f['hidden'],'lambda':f['lambda'],'n':f['n'],'final_loss':f['final_loss'],
                'lag_norms':norms,'local_derivative':deriv}
        records.append(record)
        for i,src in enumerate(VARS):
            for j,dst in enumerate(VARS):
                for k in range(norms.shape[2]):
                    if norms[i,j,k]>=.10:
                        config_edges.setdefault((src,dst,k+1),[]).append({'hidden':f['hidden'],'lambda':f['lambda'],
                            'norm':float(norms[i,j,k]),'derivative':float(deriv[i,j,k])})
    edges=[]
    for (src,dst,lag),v in sorted(config_edges.items()):
        signs={int(np.sign(x['derivative'])) for x in v if abs(x['derivative'])>1e-6}
        edges.append({'source':src,'target':dst,'lag':lag,'support_count':len(v),
                      'supporting_settings':[{'hidden':x['hidden'],'lambda':x['lambda']} for x in v],
                      'median_norm':float(np.median([x['norm'] for x in v])),
                      'median_local_derivative':float(np.median([x['derivative'] for x in v])),
                      'sign_consistent':len(signs)<=1,'regularization_stable':len(v)>=2 and len(signs)<=1})
    # Keep the full setting-specific norms and diagnostics; no graph is selected
    # from a single preferred neural setting.
    return {'status':'FIT_COMPLETE','implementation':label,'lambda_grid':list(lambdas),
            'hidden_sizes':list(hidden),'epochs':1000,'learning_rate':.02,
            'edge_norm_threshold':.10,'fits':records,'candidate_edges':edges}

def fit_lingam(segments):
    r=var_lingam(segments,VARS,seed=314159)
    if r['diagnostic_pass']:
        edges=[]
        lagged=np.asarray(r['lagged']);freq=np.asarray(r['lagged_segment_frequency'])
        for k in range(lagged.shape[0]):
            for i,src in enumerate(VARS):
                for j,dst in enumerate(VARS):
                    w=float(lagged[k,i,j]);fr=float(freq[k,i,j])
                    if abs(w)>=.05 and fr>=.5:
                        edges.append({'source':src,'target':dst,'lag':k+1,'median_weight':w,
                                      'segment_selection_frequency':fr,'sign_consistent':True})
        r['candidate_edges']=edges;r['status']='FIT_COMPLETE'
    else:
        r['candidate_edges']=[];r['status']='DIAGNOSTIC_FAILURE'
    return r

def main():
    fixture=json.loads((OUT/'fixture_validation.json').read_text())
    if any(x['status']!='VALID' for x in fixture['methods'].values()):
        raise RuntimeError('Synthetic validation hard stop: every method must be VALID before empirical fitting')
    protocol=json.loads((OUT/'effective_protocol_manifest.json').read_text())
    if not protocol['frozen_before_empirical_graphs']:raise RuntimeError('Protocol not frozen')
    assets=load_asset_discovery()
    write_json(OUT/'discovery_sample_manifest.json',{
       'read_mode':'streamed discovery rows only; holdout rows skipped before JSON decoding',
       'discovery_interval':'2021-12-01 00:00 UTC through 2023-06-30 20:00 UTC',
       'variables':VARS,'assets':{a:{'complete_observations':x['complete_n'],'segment_lengths':x['segment_lengths'],
           'first_complete_utc':x['first_complete'],'last_complete_utc':x['last_complete'],
           'discovery_mean':x['mean'],'discovery_sd':x['sd']} for a,x in assets.items()},
       'confirmation_values_decoded':False,'effective_protocol_sha256':protocol['effective_protocol_sha256']})
    method_fns={'pcmci_results.json':fit_pcmci,'dynotears_results.json':fit_dynotears,
                'var_lingam_results.json':fit_lingam}
    diagnostics={}
    for filename,fn in method_fns.items():
        method=filename.replace('_results.json','')
        result={'method':method,'validation_status':fixture['methods'][{'pcmci':'PCMCI+','dynotears':'DYNOTEARS','var_lingam':'VAR-LiNGAM'}[method]]['status'],
                'assets':{},'effective_protocol_sha256':protocol['effective_protocol_sha256'],
                'discovery_only':True}
        for a,v in assets.items():
            log(f'{method} empirical fit start asset={a}')
            t=time.monotonic()
            try:
                r=fn(v['segments']);r['elapsed_seconds']=round(time.monotonic()-t,3)
                result['assets'][a]=r
                log(f"{method} empirical fit complete asset={a} status={r['status']} edges={len(r.get('candidate_edges',[]))} seconds={r['elapsed_seconds']}")
            except Exception as e:
                result['assets'][a]={'status':'FIT_ERROR','error':f'{type(e).__name__}: {e}','traceback':traceback.format_exc()}
                log(f'{method} FIT_ERROR asset={a}: {type(e).__name__}: {e}')
            write_json(OUT/filename,result)
        diagnostics[method]={a:{'status':r.get('status'),'elapsed_seconds':r.get('elapsed_seconds'),
                       'candidate_edge_count':len(r.get('candidate_edges',[])),
                       'diagnostic_pass':r.get('diagnostic_pass')} for a,r in result['assets'].items()}
        write_json(OUT/'method_diagnostics.json',{'synthetic_validation':{k:v['status'] for k,v in fixture['methods'].items()},
                     'empirical_fit_status':diagnostics,'cmi_knn_secondary':'NOT_RUN_RUNTIME_INFEASIBLE_AFTER_GT_90S_SYNTHETIC_TRIAL',
                     'funding_values':'unchanged repeated settlement state; no jitter',
                     'confirmation_values_decoded':False,'effective_protocol_sha256':protocol['effective_protocol_sha256']})
    # Neural models use fixed synthetic-validated proximal implementation and
    # the frozen small architecture / penalty grids.
    for mode,filename,key in [('notears_mlp','notears_mlp_results.json','NOTEARS-MLP'),
                              ('cmlp','neural_granger_results.json','Neural Granger/cMLP')]:
        result={'method':key,'validation_status':fixture['methods'][key]['status'],'assets':{},
                'effective_protocol_sha256':protocol['effective_protocol_sha256'],'discovery_only':True}
        for a,v in assets.items():
            log(f'{key} empirical fit start asset={a}')
            t=time.monotonic()
            try:
                r=fit_neural(v['segments'],mode);r['elapsed_seconds']=round(time.monotonic()-t,3)
                result['assets'][a]=r
                log(f"{key} empirical fit complete asset={a} status={r['status']} edges={len(r.get('candidate_edges',[]))} seconds={r['elapsed_seconds']}")
            except Exception as e:
                result['assets'][a]={'status':'FIT_ERROR','error':f'{type(e).__name__}: {e}','traceback':traceback.format_exc()}
                log(f'{key} FIT_ERROR asset={a}: {type(e).__name__}: {e}')
            write_json(OUT/filename,result)
        diagnostics[key]={a:{'status':r.get('status'),'elapsed_seconds':r.get('elapsed_seconds'),
                         'candidate_edge_count':len(r.get('candidate_edges',[]))} for a,r in result['assets'].items()}
        write_json(OUT/'method_diagnostics.json',{'synthetic_validation':{k:v['status'] for k,v in fixture['methods'].items()},
                     'empirical_fit_status':diagnostics,'cmi_knn_secondary':'NOT_RUN_RUNTIME_INFEASIBLE_AFTER_GT_90S_SYNTHETIC_TRIAL',
                     'funding_values':'unchanged repeated settlement state; no jitter',
                     'confirmation_values_decoded':False,'effective_protocol_sha256':protocol['effective_protocol_sha256']})
    log('ALL_DISCOVERY_FITS_COMPLETE; confirmation remains unopened')

if __name__=='__main__':main()
