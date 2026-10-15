from __future__ import annotations

import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from common import (OUT, ASSETS, TARGET, SPOT, DER, STATIC_DER, CAL,
                    load_panel, feature_frames, complete_mask, transform_target)
from classical_models import fit_predict, stable_seed

SEED = 20261007
FAMILY_ORDER=['ElasticNet','RBF_SVR','ExtraTrees','HistGradientBoosting','XGBoost','LightGBM','CatBoost','MLP','GRU']
OUTER = [
    {'id':'F1','start':'2022-10-01T00:00:00Z','end':'2023-01-01T00:00:00Z'},
    {'id':'F2','start':'2023-01-01T00:00:00Z','end':'2023-04-01T00:00:00Z'},
    {'id':'F3','start':'2023-04-01T00:00:00Z','end':'2023-07-01T00:00:00Z'},
]

def readj(name): return json.loads((OUT/name).read_text())
def dump(name, x): (OUT/name).write_text(json.dumps(x, indent=2, sort_keys=True, allow_nan=False)+'\n')
def dt(x): return pd.to_datetime(x, utc=True)

def moving_ids(dates, length, n, rng):
    d = pd.to_datetime(pd.Series(dates), utc=True)
    # Match the divisor to pandas' stored datetime resolution (us in pandas 3).
    tick = 4*3600*{'s':1,'ms':10**3,'us':10**6,'ns':10**9}[d.dtype.unit]
    ticks = d.astype('int64').to_numpy() // tick
    first, last = int(ticks.min()), int(ticks.max())
    starts = np.arange(first, max(first+1, last-length+2))
    if not len(starts): starts = np.array([first])
    ids=[]
    while len(ids)<n:
        s=int(rng.choice(starts)); ids.extend(np.flatnonzero((ticks>=s)&(ticks<s+length)).tolist())
    return np.asarray(ids[:n], int)

def paired_boot(dates, delta, seed, reps=1000, lengths=(42,84)):
    dates=np.asarray(dates); delta=np.asarray(delta,float); rng=np.random.default_rng(seed); out={}
    # Keep each chronological validation block separate when a fold vector is supplied.
    for L in lengths:
        vals=np.empty(reps)
        for b in range(reps):
            ix=moving_ids(dates,L,len(dates),rng); vals[b]=delta[ix].mean()
        n_le=int(np.sum(vals<=0)); n_ge=int(np.sum(vals>=0))
        p=min(1.,2*(1+min(n_le,n_ge))/(reps+1))
        out[str(L)]={'replications':reps,'block_length_intervals':L,'delta_mean':float(delta.mean()),
                     'ci95':[float(v) for v in np.quantile(vals,[.025,.975])],
                     'p_two_sided':float(p),'probability_positive':float((vals>0).mean())}
    return out

def holm(p):
    p=np.asarray(p,float); n=len(p); order=np.argsort(p); adj=np.empty(n); running=0.
    for r,j in enumerate(order): running=max(running,min(1.,(n-r)*p[j])); adj[j]=running
    return adj.tolist()

def inverse(y, name):
    y=np.asarray(y,float)
    if name=='T1_raw': return np.maximum(y,0.)
    if name=='T2_log1p': return np.maximum(np.expm1(y),0.)
    if name=='T3_log_eps': return np.maximum(np.exp(y)-1e-8,0.)
    return np.maximum((np.exp(y)-1e-8)/10000.,0.)

def feature_sets(g):
    _, f0, f1, _ = feature_frames(g)
    dcols=[c for c in f1.columns if any(c.startswith(v+'__lag') for v in DER)]
    xfull=pd.concat([f0.drop(columns='date'),f1[dcols+STATIC_DER]],axis=1)
    calcols=CAL
    components={
      'funding':['funding_rate_state__lag'+str(i) for i in range(6)]+['funding_change'],
      'oi':['oi_log_change__lag'+str(i) for i in range(6)]+['oi_shock_30d_robust'],
      'flow':['futures_taker_imbalance__lag'+str(i) for i in range(6)]+['flow_shock_30d_robust'],
      'basis':['basis_mean__lag'+str(i) for i in range(6)]+['basis_change'],
    }
    inter={
      'funding':['funding_x_oi_shock','funding_x_basis'],
      'oi':['funding_x_oi_shock','oi_shock_x_flow_shock'],
      'flow':['oi_shock_x_flow_shock','flow_x_basis'],
      'basis':['funding_x_basis','flow_x_basis'],
    }
    sets={'full':xfull.columns.tolist(),'no_calendar':[c for c in xfull.columns if c not in CAL]}
    for name in components:
        sets[f'{name}_only']=f0.drop(columns='date').columns.tolist()+components[name]+CAL
        removed=set(components[name]+[c for c in inter[name] if c in STATIC_DER])
        sets[f'drop_{name}']=[c for c in xfull.columns if c not in removed]
    # De-duplicate any repeated names while preserving order.
    return g, f0, xfull, {k:list(dict.fromkeys(v)) for k,v in sets.items()}

def create_ablation_outputs(panel, result):
    records=[]; predrows=[]
    for asset in ASSETS:
        g=panel[panel.asset==asset].sort_values('period_start_utc').reset_index(drop=True)
        g, f0, xfull, sets0=feature_sets(g)
        dates=pd.to_datetime(g.period_start_utc,utc=True)
        yraw=g[TARGET].to_numpy(float); trans=readj('target_transform_selection.json')['selected_transform']; yt=transform_target(yraw,trans)
        ok=np.isfinite(yraw)&np.isfinite(xfull.to_numpy(float)).all(1)
        # The same family and selected config as the nested inner-CV winner in each outer fold.
        gr=result['group_results']
        family_by_fold={}
        for fold in OUTER:
            cands=[r for r in gr if r['asset']==asset and r['outer_fold']==fold['id'] and r['info_set']=='SPOT_PLUS_DERIVATIVES']
            win=min(cands,key=lambda r:(r['inner_selected_score'],FAMILY_ORDER.index(r['family'])))
            family_by_fold[fold['id']]=win
        for fold in OUTER:
            start,end=dt(fold['start']),dt(fold['end']); train_cut=start-pd.Timedelta(hours=24); val_cut=end-pd.Timedelta(hours=24)
            tr=ok&(dates>=dt('2021-12-01T00:00:00Z'))&(dates<train_cut)
            va=ok&(dates>=start)&(dates<val_cut)
            win=family_by_fold[fold['id']]; fam=win['family']; cfg=win['selected_config'];
            # GRU ablations remove channels before model fitting, preserving six ordered steps.
            for spec, cols in sets0.items():
                cols=[c for c in cols if c in xfull.columns]
                if fam=='GRU':
                    chosen=[v for v in SPOT+DER if any(c.startswith(v+'__lag') for c in cols)]
                    seq=np.stack([np.column_stack([g[v].shift(l).to_numpy(float) for v in chosen]) for l in range(5,-1,-1)],axis=1)
                    scol=[c for c in cols if c in CAL+STATIC_DER]
                    st=g[scol].to_numpy(float)
                    pred,_,warn=fit_predict(fam,cfg,xfull.loc[:,cols].to_numpy(float)[tr],yt[tr],xfull.loc[:,cols].to_numpy(float)[va],seq[tr],seq[va],st[tr],st[va],stable_seed(SEED,asset,fold['id'],spec,'ablation'))
                else:
                    mat=xfull.loc[:,cols].to_numpy(float)
                    pred,_,warn=fit_predict(fam,cfg,mat[tr],yt[tr],mat[va],seed=stable_seed(SEED,asset,fold['id'],spec,'ablation'))
                for j,idx in enumerate(np.flatnonzero(va)):
                    predrows.append({'asset':asset,'fold':fold['id'],'date':dates.iloc[idx].isoformat(),'specification':spec,'target_transformed':float(yt[idx]),'prediction_transformed':float(pred[j]),'target_raw':float(yraw[idx])})
                records.append({'asset':asset,'fold':fold['id'],'specification':spec,'family':fam,'selected_config_index':win['selected_config_index'],'n_train':int(tr.sum()),'n_test':int(va.sum()),'first_test_date':dates[va].min().isoformat(),'last_test_date':dates[va].max().isoformat(),'fit_warnings':warn})
    pdf=pd.DataFrame(predrows); summaries=[]
    for (asset,spec),z in pdf.groupby(['asset','specification'],sort=False):
        err=np.abs(z.target_transformed-z.prediction_transformed)
        summaries.append({'asset':asset,'specification':spec,'n':len(z),'n_dates':z.date.nunique(),'transformed_mae':float(err.mean()),'original_scale_mae':float(np.abs(z.target_raw-inverse(z.prediction_transformed,readj('target_transform_selection.json')['selected_transform'])).mean())})
    dump('component_ablation.json',{'definition':'Single-component uses only that derivative family (six lags plus its frozen univariate state transform), common spot history and calendar; interactions are excluded from single-component models. Same inner-CV selected model family/config per outer fold; no retuning.','fit_records':records,'summary':summaries})
    dump('drop_one_ablation.json',{'definition':'Each drop-one variant removes the component, its univariate transformation, and every frozen interaction involving that component. Same inner-CV selected model family/config per outer fold; no retuning.','fit_records':[r for r in records if r['specification'].startswith('drop_')],'summary':[r for r in summaries if r['specification'].startswith('drop_')]})
    dump('seasonality_ablation.json',{'definition':'Remove all frozen UTC slot, weekday and funding-window indicators from the full S1 model; all else fixed.','fit_records':[r for r in records if r['specification']=='no_calendar'],'summary':[r for r in summaries if r['specification']=='no_calendar']})
    joint=[]
    for asset in ASSETS:
        a=pdf[pdf.asset==asset]
        full=a[a.specification=='full'].set_index(['fold','date'])
        for spec in ['funding_only','oi_only','flow_only','basis_only','drop_funding','drop_oi','drop_flow','drop_basis']:
            ref=a[a.specification==spec].set_index(['fold','date'])
            pair=full[['target_transformed','prediction_transformed']].join(
                ref[['prediction_transformed']],lsuffix='_full',rsuffix='_reference',how='inner',validate='one_to_one')
            d=np.abs(pair.target_transformed-pair.prediction_transformed_reference)-np.abs(pair.target_transformed-pair.prediction_transformed_full)
            dates=pair.index.get_level_values('date').astype(str).to_numpy()
            boot=paired_boot(dates,d.to_numpy(float),stable_seed(SEED,asset,spec,'joint_state'))
            joint.append({'asset':asset,'comparison':f'full_vs_{spec}','n_asset_day_rows':len(pair),'n_unique_dates':len(set(dates)),
                'delta_reference_loss_minus_full_loss_positive_favors_joint_state':float(d.mean()),
                'reference_transformed_mae':float(np.abs(pair.target_transformed-pair.prediction_transformed_reference).mean()),
                'full_transformed_mae':float(np.abs(pair.target_transformed-pair.prediction_transformed_full).mean()),
                'paired_block_bootstrap':boot})
    dump('joint_state_results.json',{'definition':'Same selected S1 family/config per outer fold and identical evaluation dates. Positive loss_reference minus loss_full favors the joint derivatives state. Single-component comparisons omit interactions; drop-one comparisons remove each component and every frozen interaction involving it. Descriptive/secondary only; this does not change the primary classification.','results':joint})

def inner_splits(dates):
    n=len(dates); out=[]
    if n<80: return out
    # Two expanding validation windows; 6 intervals are purged before validation.
    for lo,hi in [(int(.60*n),int(.80*n)),(int(.80*n),n)]:
        lo=max(lo,1); hi=max(hi,lo+1)
        train=np.arange(0,max(0,lo-6)); val=np.arange(lo,min(hi,n))
        if len(train)>=50 and len(val)>=20: out.append((train,val))
    return out

def create_residual_outputs(panel):
    oo=readj('spot_oof_predictions.json')['predictions']; odf=pd.DataFrame(oo)
    odf['date']=pd.to_datetime(odf.date,utc=True)
    trans=readj('target_transform_selection.json')['selected_transform']; reg=readj('model_registry.json')['families']
    families=['ElasticNet','RBF_SVR','HistGradientBoosting','XGBoost','MLP']
    all_pred=[]; fitlog=[]
    for asset in ASSETS:
        g=panel[panel.asset==asset].sort_values('period_start_utc').reset_index(drop=True)
        g,f0,f1,y=feature_frames(g); dates=pd.to_datetime(g.period_start_utc,utc=True)
        dcols=[c for c in f1.columns if any(c.startswith(v+'__lag') for v in DER)]
        x=pd.concat([f1[dcols+STATIC_DER+CAL]],axis=1).to_numpy(float)
        local=odf[odf.asset==asset].copy().sort_values('date')
        local['residual']=local.target_transformed-local.oof_spot_prediction_transformed
        pos={v:i for i,v in enumerate(dates)}
        local['row']=[pos[d] for d in local.date]
        for fold in ['F2','F3']:
            val=local[local.fold==fold].copy()
            train=local[(local.date<val.date.min())].copy()
            # At least one earlier OOF block is required to train residual learners.
            if len(train)<100 or len(val)<100: continue
            tri=train.row.to_numpy(int); vai=val.row.to_numpy(int); yt=train.residual.to_numpy(float); yv=val.residual.to_numpy(float)
            split=inner_splits(train.date.reset_index(drop=True))
            if not split: raise RuntimeError(f'No chronological inner folds for residual {asset}/{fold}')
            for family in families:
                scores=[]
                for ci,cfg in enumerate(reg[family]):
                    abs_sum=0.; count=0
                    for trloc,valoc in split:
                        pred,_,_=fit_predict(family,cfg,x[tri[trloc]],yt[trloc],x[tri[valoc]],seed=stable_seed(SEED,asset,fold,family,ci,'residual_inner'))
                        abs_sum+=float(np.abs(yt[valoc]-pred).sum());count+=len(valoc)
                    scores.append({'config_index':ci,'config':cfg,'mae':abs_sum/count})
                best=min(scores,key=lambda z:(z['mae'],z['config_index']))
                pred,elapsed,warn=fit_predict(family,best['config'],x[tri],yt,x[vai],seed=stable_seed(SEED,asset,fold,family,best['config_index'],'residual_outer'))
                # Forecast correction is additive in the frozen transformed-target scale.
                corrected=val.oof_spot_prediction_transformed.to_numpy(float)+pred
                base=val.oof_spot_prediction_transformed.to_numpy(float)
                raw=val.target_raw.to_numpy(float); truth=val.target_transformed.to_numpy(float)
                for k,(_,r) in enumerate(val.iterrows()):
                    all_pred.append({'asset':asset,'fold':fold,'date':r.date.isoformat(),'family':family,'target_raw':float(raw[k]),'target_transformed':float(truth[k]),'spot_oof_prediction_transformed':float(base[k]),'residual_prediction':float(pred[k]),'corrected_prediction_transformed':float(corrected[k]),'selected_config_index':best['config_index']})
                fitlog.append({'asset':asset,'fold':fold,'family':family,'n_train_oof_residuals':len(train),'n_test':len(val),'train_first':train.date.min().isoformat(),'train_last':train.date.max().isoformat(),'test_first':val.date.min().isoformat(),'test_last':val.date.max().isoformat(),'inner_folds':[{'train_first':train.date.iloc[a[0]].isoformat(),'train_last':train.date.iloc[a[-1]].isoformat(),'validation_first':train.date.iloc[b[0]].isoformat(),'validation_last':train.date.iloc[b[-1]].isoformat(),'n_train':len(a),'n_validation':len(b)} for a,b in split],'selected_config_index':best['config_index'],'selected_config':best['config'],'inner_scores':scores,'fit_seconds':elapsed,'warnings':warn})
    df=pd.DataFrame(all_pred); summary=[]
    for (asset,family),z in df.groupby(['asset','family'],sort=False):
        dates=z.date.to_numpy(); base=np.abs(z.target_transformed-z.spot_oof_prediction_transformed).to_numpy(); corr=np.abs(z.target_transformed-z.corrected_prediction_transformed).to_numpy(); delta=base-corr
        boot=paired_boot(dates,delta,stable_seed(SEED,asset,family,'residual_boot'))
        summary.append({'asset':asset,'family':family,'n_asset_day_rows':len(z),'n_unique_dates':z.date.nunique(),'folds':sorted(z.fold.unique()),'spot_only_residual_mae':float(base.mean()),'residual_model_mae':float(corr.mean()),'delta_positive_favors_derivatives':float(delta.mean()),'paired_block_bootstrap':boot})
    for asset in ASSETS:
        rows=[r for r in summary if r['asset']==asset]
        adj=holm([r['paired_block_bootstrap']['42']['p_two_sided'] for r in rows])
        for r,p in zip(rows,adj): r['holm_p_within_asset_residual_families']=p
    dump('residual_learning_results.json',{'status':'COMPLETE','definition':'Strict OOF residuals are target minus inner-CV-selected SPOT_ONLY outer-fold predictions. Residual correction is evaluated only on F2/F3; training labels are earlier outer-fold OOF residuals (F1 for F2; F1+F2 for F3). This intentionally sacrifices F1 evaluation rather than train on in-sample residuals.','features':'derivative lags t-5..t, frozen derivative innovations/interactions, and frozen calendar controls; no spot features.','target_transform':trans,'fit_records':fitlog,'predictions':all_pred,'summary':summary})

def create_negative_control(panel):
    trans=readj('target_transform_selection.json')['selected_transform']; outputs=[]
    for asset in ASSETS:
        g=panel[panel.asset==asset].sort_values('period_start_utc').reset_index(drop=True)
        g,f0,f1,y=feature_frames(g); dates=pd.to_datetime(g.period_start_utc,utc=True)
        x0=f0.drop(columns='date').to_numpy(float)
        future=np.column_stack([g[v].shift(-lead).to_numpy(float) for lead in range(7,13) for v in DER])
        x1=np.column_stack([x0,future])
        past=np.column_stack([g.spot_amihud.shift(k).to_numpy(float) for k in range(1,7)]).mean(axis=1)
        yt=transform_target(past,trans)
        ok=np.isfinite(x0).all(1)&np.isfinite(x1).all(1)&np.isfinite(yt)
        for fold in OUTER:
            start,end=dt(fold['start']),dt(fold['end']); tr=ok&(dates>=dt('2021-12-01T00:00:00Z'))&(dates<start-pd.Timedelta(hours=48)); va=ok&(dates>=start)&(dates<end-pd.Timedelta(hours=48))
            if tr.sum()<100 or va.sum()<100: continue
            ytr,yva=yt[tr],yt[va]
            p0=make_pipeline(StandardScaler(),Ridge(alpha=1.)).fit(x0[tr],ytr).predict(x0[va])
            p1=make_pipeline(StandardScaler(),Ridge(alpha=1.)).fit(x1[tr],ytr).predict(x1[va])
            d=np.abs(yva-p0)-np.abs(yva-p1)
            outputs.append({'asset':asset,'fold':fold['id'],'n':int(va.sum()),'n_dates':int(dates[va].nunique()),'first_date':dates[va].min().isoformat(),'last_date':dates[va].max().isoformat(),'target':'mean Amihud at t-6..t-1','spot_features':'certified spot history t-5..t, same Ridge(StandardScaler), alpha=1 in S0/S1','derivative_features':'four certified raw derivative variables at t+7..t+12 (strictly after target interval t-6..t-1 and primary target t+1..t+6)','delta_s0_minus_s1':float(d.mean()),'bootstrap':paired_boot(dates[va].astype(str).to_numpy(),d,stable_seed(SEED,asset,fold['id'],'negative_control'))})
    dump('negative_controls.json',{'status':'EXPLORATORY_NEGATIVE_CONTROL','definition':'Matched impossible-time control: derivative information from t+7..t+12 predicts Amihud already realized over t-6..t-1; comparison is fixed Ridge alpha=1 with identical rows and spot histories. This is a timing/confounding diagnostic, not a causal test. Train folds purge 12 intervals.','results':outputs,'interpretation_rule':'Any positive delta with block-bootstrap support is a temporal leakage/confounding warning; it cannot support forward transmission.'})

def create_regime_summary(panel):
    oo=pd.DataFrame(readj('spot_oof_predictions.json')['predictions']); oo['date']=pd.to_datetime(oo.date,utc=True)
    pdf=[]
    for asset in ASSETS:
        g=panel[panel.asset==asset].sort_values('period_start_utc').copy();g['period_start_utc']=pd.to_datetime(g.period_start_utc,utc=True)
        pre=g[g.period_start_utc<dt('2022-10-01T00:00:00Z')]
        thresholds={'high_rv_q75':float(pre.spot_realized_volatility.quantile(.75)),'high_amihud_q75':float(pre.spot_amihud.quantile(.75)),'abs_oishock_q90':float(pre.oi_shock_30d_robust.abs().quantile(.90)),'abs_funding_q90':float(pre.funding_rate_state.abs().quantile(.90)),'basis_median':float(pre.basis_mean.median())}
        # Reconstruct fold-specific predictions using only each fold's inner-CV winner.
        cp=OUT/'classical_group_checkpoints.jsonl'; rows=[]
        if cp.exists():
            for line in cp.read_text().splitlines():
                rec=json.loads(line)
                if rec['asset']==asset:
                    for q in rec['predictions']:
                        rows.append({'date':pd.Timestamp(q['date']),'fold':q['fold'],'family':q['family'],'info_set':rec['info_set'],'target_transformed':q['target_transformed'],'prediction_transformed':q['prediction_transformed']})
        ap=pd.DataFrame(rows)
        if ap.empty: continue
        devres=readj('classical_development_results.json')
        merged=[]; order=devres['families']
        for fold in sorted(ap.fold.unique()):
            chosen={}
            for info in ['SPOT_ONLY','SPOT_PLUS_DERIVATIVES']:
                candidates=[r for r in devres['group_results'] if r['asset']==asset and r['outer_fold']==fold and r['info_set']==info]
                chosen[info]=min(candidates,key=lambda r:(r['inner_selected_score'],order.index(r['family'])))['family']
            a=ap[(ap.fold==fold)&(ap.info_set=='SPOT_ONLY')&(ap.family==chosen['SPOT_ONLY'])].rename(columns={'prediction_transformed':'p0'})
            b=ap[(ap.fold==fold)&(ap.info_set=='SPOT_PLUS_DERIVATIVES')&(ap.family==chosen['SPOT_PLUS_DERIVATIVES'])].rename(columns={'prediction_transformed':'p1'})
            merged.append(a[['date','target_transformed','p0']].merge(b[['date','p1']],on='date',validate='one_to_one'))
        m=pd.concat(merged,ignore_index=True)
        m=m.merge(g[['period_start_utc','spot_realized_volatility','spot_amihud','oi_shock_30d_robust','funding_rate_state','basis_mean']],left_on='date',right_on='period_start_utc',validate='one_to_one')
        m['delta']=np.abs(m.target_transformed-m.p0)-np.abs(m.target_transformed-m.p1)
        masks={'high_volatility':m.spot_realized_volatility>thresholds['high_rv_q75'],'low_volatility':m.spot_realized_volatility<=thresholds['high_rv_q75'],'high_illiquidity':m.spot_amihud>thresholds['high_amihud_q75'],'low_illiquidity':m.spot_amihud<=thresholds['high_amihud_q75'],'extreme_oi_shock':m.oi_shock_30d_robust.abs()>thresholds['abs_oishock_q90'],'extreme_funding':m.funding_rate_state.abs()>thresholds['abs_funding_q90'],'positive_basis':m.basis_mean>thresholds['basis_median'],'negative_basis':m.basis_mean<=thresholds['basis_median']}
        for label,mask in masks.items():
            if mask.any(): pdf.append({'asset':asset,'regime':label,'n':int(mask.sum()),'mean_delta_s0_minus_s1':float(m.loc[mask,'delta'].mean()),'median_delta':float(m.loc[mask,'delta'].median()),'thresholds_learned_pre_outer_evaluation':thresholds,'status':'EXPLORATORY_DESCRIPTIVE'})
    dump('regime_exploration.json',{'definition':'Thresholds are asset-specific empirical quantiles from the pre-outer-evaluation period ending 2022-09-30; summaries use only nested OOF errors and selected families. Descriptive only.','results':pdf})

def main():
    panel=load_panel(); result=readj('classical_development_results.json')
    if result.get('status')!='COMPLETE': raise RuntimeError('Classical main run is not complete')
    create_ablation_outputs(panel,result)
    create_residual_outputs(panel)
    create_negative_control(panel)
    create_regime_summary(panel)
    print(json.dumps({'status':'COMPLETE','outputs':['component_ablation.json','drop_one_ablation.json','seasonality_ablation.json','residual_learning_results.json','negative_controls.json','regime_exploration.json']}))

if __name__=='__main__': main()
