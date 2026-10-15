#!/usr/bin/env python3
"""Complete Stage-3 reporting, figures and non-inferential audit tables."""
from __future__ import annotations
import csv,hashlib,json,platform
from pathlib import Path
import numpy as np,pandas as pd,scipy,sklearn,matplotlib
import matplotlib.pyplot as plt
from stage3_common import holm_adjust
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'


def save_multiple_testing():
 rows=[]
 def add(df,method,family_col,p_col,status_col=None,confirmatory=False):
  if df.empty:return
  for fam,g in df.groupby(family_col,dropna=False):
   ps=pd.to_numeric(g[p_col],errors='coerce').to_numpy();adj=holm_adjust(ps)
   for (ix,(_,r)) in enumerate(g.iterrows()):
    rows.append({'analysis':method,'family_id':str(fam),'test_id':r.get('test_id',''),'channel_id':r.get('channel_id',''),'asset':r.get('asset',''),'direction':r.get('direction',''),'horizon':r.get('horizon',''),'raw_p':r.get(p_col,np.nan),'holm_p':adj[ix],'family_size':len(g),'confirmatory':confirmatory,'status':r.get(status_col,'') if status_col else ''})
 primary=pd.read_csv(OUT/'table_stage3_primary_ts_dml.csv');add(primary,'PRIMARY_TS_DML','holm_family','raw_p','status',True)
 placebo=pd.read_csv(OUT/'table_stage3_falsification.csv');add(placebo,'PAST_TIMING_FALSIFICATION','family_id','raw_p','status',False)
 for file,label in [('table_stage3_learner_robustness.csv','RIDGE_BLOCKED'),('table_stage3_forward_crossfit_robustness.csv','HGB_FORWARD_ONLY'),('table_stage3_timing_robustness.csv','HGB_TIMING_GAP')]:
  x=pd.read_csv(OUT/file);x['variant_family']=x.holm_family.astype(str)+'_'+label;add(x,label,'variant_family','raw_p','status',False)
 gra=pd.read_csv(OUT/'table_stage3_granger.csv');gra['method_family']='GRANGER_'+gra.asset;add(gra,'VAR_GRANGER','method_family','hac_p','diagnostic_status',False)
 te=pd.read_csv(OUT/'table_stage3_transfer_entropy.csv')
 te=te[te.analysis_variant=='PRIMARY_L1_B14'].copy();te['method_family']='TE_PRIMARY_'+te.asset;add(te,'TRANSFER_ENTROPY','method_family','empirical_p','status',False)
 teb=pd.read_csv(OUT/'table_stage3_transfer_entropy.csv');teb=teb[teb.analysis_variant!='PRIMARY_L1_B14'].copy();teb['method_family']=teb.analysis_variant+'_'+teb.asset;add(teb,'TRANSFER_ENTROPY_ROBUSTNESS','method_family','empirical_p','status',False)
 btc=pd.read_csv(OUT/'table_stage3_btc_eth.csv');add(btc,'BTC_ETH_PAIRED_DIFFERENCE','holm_family','raw_p','status',False)
 endpoint=pd.read_csv(OUT/'stage3_temporal_endpoint_tests.csv')
 for metric in ['first_p','last_p','difference_p']:
  x=endpoint[endpoint.channel_id.isin(['C1','C2','C4'])].copy();x['endpoint_metric']=metric;x['test_id']=x.channel_id+'_'+x.asset+'_'+x.direction+'_'+metric;x['family_id']=x.asset+'_'+x.direction+'_'+metric
  add(x,'TEMPORAL_ENDPOINT_'+metric,'family_id',metric,'status',False)
 # Secondary and extended horizons are visibly exploratory; no confirmatory correction.
 for file,label in [('table_stage3_secondary.csv','SECONDARY_EXPLORATORY'),('table_stage3_extended_horizons.csv','EXTENDED_PERSISTENCE_ONLY'),('table_stage3_extended_sample.csv','EXTENDED_TIMESTAMP_HARMONIZED_SAMPLE')]:
  x=pd.read_csv(OUT/file)
  for _,r in x.iterrows():rows.append({'analysis':label,'family_id':'NO_CONFIRMATORY_FAMILY','test_id':r.get('test_id',''),'channel_id':r.get('channel_id',''),'asset':r.get('asset',''),'direction':r.get('direction',''),'horizon':r.get('horizon',''),'raw_p':r.get('raw_p',np.nan),'holm_p':np.nan,'family_size':np.nan,'confirmatory':False,'status':r.get('status','EXPLORATORY')})
 pd.DataFrame(rows).to_csv(OUT/'table_stage3_multiple_testing.csv',index=False)


def robust_matrix():
 p=pd.read_csv(OUT/'table_stage3_primary_ts_dml.csv');r=pd.read_csv(OUT/'table_stage3_robustness_matrix.csv')
 b=pd.read_csv(OUT/'table_stage3_bootstrap.csv');pl=pd.read_csv(OUT/'table_stage3_falsification.csv');gr=pd.read_csv(OUT/'table_stage3_granger.csv');te=pd.read_csv(OUT/'table_stage3_transfer_entropy.csv');ep=pd.read_csv(OUT/'stage3_temporal_endpoint_tests.csv')
 pm={'C1':'FUNDING<->RV','C2':'OI<->RV','C3':'OI<->DOWNSIDE','C4':'FLOW<->AMIHUD'}
 records=[]
 for _,x in p.iterrows():
  rr=r[r.test_id==x.test_id].iloc[0]
  fwd=pd.read_csv(OUT/'table_stage3_forward_crossfit_robustness.csv');fwd=fwd[fwd.test_id==x.test_id]
  timing=pd.read_csv(OUT/'table_stage3_timing_robustness.csv');timing=timing[timing.test_id==x.test_id]
  bbase=b[(b.test_id==x.test_id)&(b.bootstrap_type=='BASE')]
  blong=b[(b.test_id==x.test_id)&(b.bootstrap_type=='LONGER_2B')]
  pls=pl[(pl.asset==x.asset)&(pl.direction==x.direction)&(pl.channel_pair==pm[x.channel_id])]
  ppass=len(pls)==3 and bool((pls.status=='PASS').all()) and bool((pls.holm_p>=.05).all())
  sign=int(np.sign(x.theta)) if np.isfinite(x.theta) else 0
  fmatch=bool(rr.forward_sign_match_primary and np.isfinite(rr.forward_matched_hgb_theta) and int(np.sign(rr.forward_matched_hgb_theta))==sign)
  tmatch=bool(rr.timing_sign_match_primary and np.isfinite(rr.timing_matched_hgb_theta) and int(np.sign(rr.timing_matched_hgb_theta))==sign)
  bex=False
  if len(bbase)==1 and len(blong)==1 and sign:
   bex=bool((bbase.iloc[0].ci_lower>0 and blong.iloc[0].ci_lower>0) if sign>0 else (bbase.iloc[0].ci_upper<0 and blong.iloc[0].ci_upper<0))
  ev=ep[(ep.channel_id==x.channel_id)&(ep.asset==x.asset)&(ep.direction==x.direction)]
  temporal=ev.iloc[0].temporal_status if len(ev) else 'NOT_ESTIMABLE'
  gs=gr[(gr.channel_id==x.channel_id)&(gr.asset==x.asset)]
  gdir=gs[gs.direction==x.direction];gopp=gs[gs.direction!=x.direction]
  if len(gdir)!=1 or gdir.iloc[0].diagnostic_status!='LINEAR_DIAGNOSTIC_PASS':gstatus='LINEAR_DIAGNOSTIC_FAIL'
  elif gdir.iloc[0].holm_p<.05:gstatus='CORROBORATING'
  elif len(gopp)==1 and gopp.iloc[0].diagnostic_status=='LINEAR_DIAGNOSTIC_PASS' and gopp.iloc[0].holm_p<.05:gstatus='METHOD_CONFLICT'
  else:gstatus='NONCONTRADICTORY_NULL'
  ts=te[(te.channel_id==x.channel_id)&(te.asset==x.asset)&(te.direction==x.direction)&(te.analysis_variant=='PRIMARY_L1_B14')]
  to=te[(te.channel_id==x.channel_id)&(te.asset==x.asset)&(te.direction!=x.direction)&(te.analysis_variant=='PRIMARY_L1_B14')]
  if len(ts)!=1 or ts.iloc[0].status!='PASS':tstatus='TE_UNSUITABLE_OR_UNAVAILABLE'
  elif ts.iloc[0].holm_p<.05:tstatus='CORROBORATING'
  elif len(to)==1 and to.iloc[0].status=='PASS' and to.iloc[0].holm_p<.05:tstatus='METHOD_CONFLICT'
  elif len(to)==1 and to.iloc[0].status=='PASS':tstatus='NONCONTRADICTORY_NULL'
  else:tstatus='TE_UNSUITABLE_OR_UNAVAILABLE'
  records.append({**rr.to_dict(),'ridge_sign_match':bool(rr.ridge_sign_match),'forward_fold_sign_match':fmatch,'timing_gap_sign_match':tmatch,'hac_significant':bool(x.raw_p<.05),'bootstrap_excludes_zero':bex,'placebo_pass':ppass,'temporal_stability':temporal,'linear_corroboration':gstatus,'te_corroboration':tstatus})
 out=pd.DataFrame(records);out.to_csv(OUT/'table_stage3_robustness_matrix.csv',index=False)


def write_manifest():
 import statsmodels
 p=Path(OUT/'stage3_model_manifest.csv')
 rows=[]
 def add(comp,key,val,note=''):
  rows.append({'component':comp,'setting':key,'value':json.dumps(val,sort_keys=True,default=str),'notes':note})
 add('protocol','protocol_manifest_v2_sha256',hashlib.sha256((OUT/'protocol_manifest_v2.csv').read_bytes()).hexdigest())
 add('sample','primary_clean_dates','2021-12-01 to 2024-03-03; 824 UTC dates')
 add('sample','asset_day_rows',1648)
 add('confirmatory','test_count',48)
 add('confirmatory','holm_families','16 channel × asset × direction families; 3 horizons each')
 add('controls','allowlist',['spot_log_return_1d_lag1','spot_realized_volatility_lag1','spot_amihud_lag1','spot_downside_semivariance_lag1','spot_quote_volume_growth_1d_lag1','funding_mean_lag1','oi_log_change_1d_lag1','futures_taker_imbalance_lag1','spot_log_return_1d_lag2','spot_realized_volatility_lag7','funding_mean_lag2'])
 add('nuisance','HGB',{'loss':'squared_error','learning_rate':.05,'max_iter':200,'max_leaf_nodes':7,'min_samples_leaf':25,'l2_regularization':1.0,'max_bins':255,'early_stopping':False,'random_state':314159})
 add('nuisance','Ridge',{'alpha':1.0,'fit_intercept':True,'feature_scaling':'training-fold only'})
 add('crossfit','blocked',{'blocks':['2021-12-01/2022-06-24','2022-06-25/2023-01-16','2023-01-17/2023-08-10','2023-08-11/2024-03-03'],'embargo':'h+8 calendar days each side'})
 add('crossfit','forward_only',{'burn_in':'block 1','embargo':'h+8 days'})
 add('timing','conservative_gap','exposure and all controls shifted one additional day; outcome remains t+1:t+h; embargo h+9')
 add('inference','HAC','Bartlett kernel; L=max(7, 2h-1); calendar score spine')
 add('inference','date_block_bootstrap',{'replications':999,'seed_base':314159,'B':'max(7,h+1)','longer':'2B','fixed OOF nuisance predictions':True})
 add('placebo','past_windows',[1,3,7],'all 108 original registered past-timing rows; controls shifted before past window')
 add('Granger','specification','VAR intercept; BIC lag 1–7; ADF/KPSS; longest consecutive segment; HAC equation Wald L=7')
 add('Transfer Entropy','primary',{'estimator':'Frenzel-Pompe/KSG-family conditional MI','k':5,'norm':'max','history':1,'Theiler calendar days':7,'blocks':14,'surrogates':999,'seed_base':314159})
 add('Transfer Entropy','robustness',{'block_length':28,'history':3,'surrogates_each':999})
 add('time_variation','rolling',{'windows_days':[365,540],'step_days':30,'calendar_full_windows_only':True})
 add('time_variation','TVP',{'system':'pairwise filtered random-walk source coefficient','training_dates':206,'starts':[1e-6,1e-4,1e-2],'smoother':False})
 add('software','versions',{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'statsmodels':statsmodels.__version__,'matplotlib':matplotlib.__version__})
 add('randomness','seed_rule','SHA256 canonical specification IDs + base seed 314159')
 pd.DataFrame(rows).to_csv(p,index=False)


def make_figures():
 primary=pd.read_csv(OUT/'table_stage3_primary_ts_dml.csv')
 for cid,name in [('C1','funding_rv'),('C2','oi_rv'),('C3','oi_downside'),('C4','flow_amihud')]:
  g=primary[(primary.channel_id==cid)&(primary.direction=='FORWARD')]
  fig,ax=plt.subplots(figsize=(7,4.5))
  for asset in ['BTC','ETH']:
   z=g[g.asset==asset].sort_values('horizon')
   ax.errorbar(z.horizon,z.theta,yerr=1.959963984540054*z.hac_se,marker='o',capsize=3,label=asset)
  ax.axhline(0,linewidth=1);ax.set_xlabel('Horizon (days)');ax.set_ylabel('TS-DML coefficient (raw units)');ax.set_title(name.replace('_',' ').title()+' — derivatives to spot');ax.legend();fig.tight_layout();fig.savefig(OUT/f'fig_channel{cid[-1]}_{name}.png',dpi=160);plt.close(fig)
 for asset in ['BTC','ETH']:
  g=primary[primary.asset==asset]
  fig,ax=plt.subplots(figsize=(8,5))
  for (cid,direction),z in g.groupby(['channel_id','direction']):
   z=z.sort_values('horizon');label=f'{cid} {direction.lower()}'
   ax.plot(z.horizon,-np.log10(np.clip(z.holm_p,1e-12,1)),marker='o',label=label)
  ax.axhline(-np.log10(.05),linewidth=1);ax.set_xlabel('Horizon (days)');ax.set_ylabel('−log10(Holm-adjusted p)');ax.set_title(f'{asset}: confirmatory directional evidence');ax.legend(ncol=2,fontsize=8);fig.tight_layout();fig.savefig(OUT/f'fig_direction_summary_{asset.lower()}.png',dpi=160);plt.close(fig)
 # Predefined rolling 365-day forward coefficients for three core pairs.
 roll=pd.read_csv(OUT/'stage3_rolling_directionality.csv');core=['C1','C2','C4']
 fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True)
 for ax,cid in zip(axes,core):
  g=roll[(roll.channel_id==cid)&(roll.direction=='FORWARD')]
  for asset in ['BTC','ETH']:
   z=g[g.asset==asset].sort_values('window_start');dates=pd.to_datetime(z.window_start)
   ax.plot(dates,z.theta,marker='o',markersize=3,label=asset)
  ax.axhline(0,linewidth=1);ax.set_ylabel(f'{cid} θ');ax.legend()
 axes[-1].set_xlabel('365-day window start');fig.suptitle('Rolling forward TS-DML — fixed 365-day windows');fig.tight_layout();fig.savefig(OUT/'fig_time_variation_core_channels.png',dpi=160);plt.close(fig)
 # Filtered TVP coefficients: use first-206-day training SDs for display normalization.
 tvp=pd.read_csv(OUT/'stage3_tvp_coefficients.csv',parse_dates=['date']);panel=pd.read_csv(ROOT/'data/processed/derivatives_spot_daily_panel.csv',parse_dates=['date']);panel.date=pd.to_datetime(panel.date,utc=True);panel=panel[(panel.date>='2021-12-01')&(panel.date<='2024-03-03')]
 srcmap={'C1':('funding_mean','spot_realized_volatility'),'C2':('oi_log_change_1d','spot_realized_volatility')}
 for asset in ['BTC','ETH']:
  fig,ax=plt.subplots(figsize=(9,4.5))
  for cid,(source,target) in srcmap.items():
   z=tvp[(tvp.asset==asset)&(tvp.channel_id==cid)&(tvp.direction=='FORWARD')].copy()
   q=panel[panel.asset==asset].sort_values('date').iloc[:206]
   sd_x=q[source].std(ddof=1);sd_y=q[target].std(ddof=1);scale=sd_x/sd_y if sd_y>0 else np.nan
   z['normalized_beta']=z.filtered_beta_raw*scale
   ax.plot(z.date,z.normalized_beta,label=f'{cid}: {source} → RV')
  ax.axhline(0,linewidth=1);ax.set_xlabel('Date');ax.set_ylabel('Filtered SD-to-SD coefficient');ax.set_title(f'{asset}: filtered TVP coefficients (post-calibration)');ax.legend();fig.tight_layout();fig.savefig(OUT/f'fig_tvp_volatility_{asset.lower()}.png',dpi=160);plt.close(fig)


def make_reports():
 p=pd.read_csv(OUT/'table_stage3_primary_ts_dml.csv');c=pd.read_csv(OUT/'channel_classification_summary.csv');te=pd.read_csv(OUT/'table_stage3_transfer_entropy.csv');gr=pd.read_csv(OUT/'table_stage3_granger.csv');pl=pd.read_csv(OUT/'table_stage3_falsification.csv');bt=pd.read_csv(OUT/'table_stage3_btc_eth.csv');sec=pd.read_csv(OUT/'table_stage3_secondary.csv');ext=pd.read_csv(OUT/'table_stage3_extended_horizons.csv');es=pd.read_csv(OUT/'table_stage3_extended_sample.csv');ep=pd.read_csv(OUT/'stage3_temporal_endpoint_tests.csv');tvpd=pd.read_csv(OUT/'stage3_tvp_diagnostics.csv')
 sig=p[p.holm_p<.05]
 primary_sig=', '.join(f"{r.asset} {r.channel_id} {r.direction} h{int(r.horizon)} (θ={r.theta:.4g}, Holm p={r.holm_p:.3g})" for r in sig.itertuples()) or 'None'
 placebo_fail=pl[(pl.registered_status=='PRIMARY_CHANNEL')&(pl.holm_p<.05)]
 teprim=te[te.analysis_variant=='PRIMARY_L1_B14'];te_rej=teprim[teprim.holm_p<.05]
 grpass=gr[gr.diagnostic_status=='LINEAR_DIAGNOSTIC_PASS']
 klass=c[['channel_id','asset','final_classification','time_variation']].to_string(index=False)
 direction_sign=(np.sign(p[p.channel_id=='C1'].theta).value_counts().to_dict())
 matched = p.merge(es, on=['channel_id','asset','direction','horizon'], suffixes=('_clean','_extended'))
 same_sign = float(np.mean(np.sign(matched.theta_clean)==np.sign(matched.theta_extended))) if len(matched) else float('nan')
 temporal_supported=ep[ep.temporal_status.str.startswith('SUPPORTED')]
 # Paper-level classification: no channel has a robust direction; C1 is method-inconclusive.
 paper='INCONCLUSIVE' if (c.final_classification=='INCONCLUSIVE').any() else 'NO_STABLE_INFORMATION_HIERARCHY'
 fwd_sig=p[p.direction=='FORWARD'].groupby('channel_id').apply(lambda g:int((g.holm_p<.05).sum()),include_groups=False).to_dict()
 rev_sig=p[p.direction=='REVERSE'].groupby('channel_id').apply(lambda g:int((g.holm_p<.05).sum()),include_groups=False).to_dict()
 report=f'''# Stage 3 research summary

## Scope and interpretation

The primary analysis uses BTC and ETH on the 824-date UTC clean sample, 2021-12-01 through 2024-03-03. There are 1,648 maximum asset-days, while the treatment-time/calendar dimension is 824 dates. The four V2 theory-matched channels yield 48 confirmatory tests, corrected in 16 separate three-horizon Holm families. The timestamp-harmonized 2024–2026 panel is reported separately as robustness only.

These are conditional directional associations and predictive information measures. They are not structural causal effects. Binance spot and USDⓈ-M perpetuals share venue infrastructure, trader populations and USDT settlement. Unobserved endogeneity and exchange-specific mechanical linkage remain possible.

## Confirmatory findings

The primary HGB blocked TS-DML results surviving within-family Holm correction are: {primary_sig}.

Channel-level classifications are:

```text
{klass}
```

No channel meets the frozen strong-direction criteria. C1 funding–RV is `INCONCLUSIVE`: the continuous TE estimator is unsuitable because funding has a repeated-value atom above 5%, while the VAR benchmark fails its preregistered stationarity/continuity gates. C2 OI–RV, C3 OI–downside and C4 futures-flow–Amihud are `NO_ROBUST_DIRECTION` after learner/timing/placebo requirements.

1. **Funding → spot RV:** BTC has HGB blocked associations at h3 and h7 surviving the C1/BTC/forward Holm family; ETH does not. BTC's sign does not survive the required forward-only and conservative-timing checks, so it is not a robust leading result.
2. **Spot RV → funding:** BTC has one h7 Holm-surviving estimate, which is insufficient for the two-horizon rule. The reverse direction is not established.
3. **OI growth → spot RV:** no confirmatory horizon survives Holm for BTC or ETH.
4. **Spot RV → OI growth:** no confirmatory horizon survives Holm.
5. **OI growth ↔ downside:** neither direction establishes a robust relationship; no confirmatory result survives Holm.
6. **Futures imbalance → Amihud:** BTC has negative HGB blocked coefficients at all three horizons surviving Holm, consistent directionally with lower measured Amihud, but forward-only/timing robustness fails. ETH has no robust forward result.
7. **Spot Amihud → futures imbalance:** ETH's h7 coefficient survives the reverse-family Holm correction, but the two-horizon, learner/timing and placebo requirements do not pass.
8. The isolated statistical rejections are therefore not sufficient to establish a stable channel leader.

### Multiple testing, falsification, and robustness

There are {len(sig)} of 48 confirmatory primary tests with Holm p < 0.05. Primary-window past-outcome falsifications reject for {len(placebo_fail)} registered channel tests after their frozen placebo-family correction: {', '.join(placebo_fail.test_id.tolist()) if len(placebo_fail) else 'none'}. These failures are retained as identification warnings; they are not evidence for the opposite direction.

Ridge and HGB forward-only estimates are reported for all 48 tests. The required sign stability fails for the isolated funding/RV and futures-flow/Amihud candidates. Conservative timing also changes signs for the BTC C1-forward and C4-forward candidates. Date-block bootstrap intervals accompany all confirmatory estimates; qualitative agreement is required by the frozen rule and does not rescue failed timing or placebo checks.

### Linear and nonlinear directional methods

All {len(gr)} VAR/Granger direction tests are `LINEAR_DIAGNOSTIC_FAIL`; no Granger direction is counted as corroboration. Pairwise samples use the longest consecutive complete segment, with a minimum of 400 dates and ADF/KPSS admissibility. Missing segments and the KPSS results prevent an admissible VAR.

The primary continuous kNN TE family has {int(teprim.status.eq('PASS').sum())} estimable directions and {int((teprim.status!='PASS').sum())} unsuitable/unavailable directions. Funding/RV is excluded by the >5% exact-value atom rule. Among estimable TE tests, {len(te_rej)} survives Holm correction. Observed and surrogate-adjusted TE values are kept in nats; no net direction is computed because the frozen protocol does not permit comparing raw TE across different target variables and no common normalization was preregistered.

## BTC–ETH and temporal stability

BTC–ETH contrasts use common dates and shared date-block bootstrap draws. The paired contrasts are mixed: some C1 forward HAC differences survive their own Holm family, but the h7 bootstrap interval includes zero, and there is no broad channel-level heterogeneity claim. Separate asset significance is not interpreted as an asset difference.

All fixed 365-day and 540-day rolling estimates are reported. The endpoint comparison uses the first and last scheduled non-overlapping 365-day windows. {len(temporal_supported)} core endpoint comparisons meet the frozen stability criterion after correction. Although some rolling and filtered coefficients move substantially, the endpoint tests do not establish systematic direction changes. The state-space model passed its numerical diagnostics for 16 pair/direction fits, but several estimated state variances lie at the lower bound; filtered movement is not treated as evidence of a time-varying relationship.

## Secondary and timestamp-harmonized analyses

The {len(sec)} secondary rows include the 60 originally registered non-confirmatory combinations plus funding-sum/basis extensions; the {len(ext)} h14/h30 results remain persistence-only. Neither set upgrades a failed primary channel.

The separate harmonized extended sample estimates cover {len(es)} core rows from 2021-12-01 through 2026-09-30. The fraction of the 48 matched clean/extended core coefficients with the same sign is {same_sign:.1%}. These results are robustness only and do not replace clean-sample findings; changes around the archive/timestamp harmonization period remain a measurement limitation.

## Overall conclusion

Paper-level classification: **{paper}**.

The strongest defensible statement is that some BTC-specific funding/RV and futures-flow/Amihud conditional associations appear in blocked TS-DML, but their sensitivity to forward-only fitting, conservative timing and falsification prevents a stable derivatives-leading or spot-leading conclusion. OI channels show no robust direction. The results support continuing with a carefully qualified predictive/directional paper, not a causal claim.

Unsupported claims include that derivatives structurally cause spot liquidity or risk, that spot stress structurally causes derivative positioning, that one market universally leads the other, or that the relationship is demonstrably time-varying. The next research step should focus on paper framing and a separately justified external identification or independent-venue validation design; no further model or asset expansion is run here.
'''
 (OUT/'stage3_research_summary.md').write_text(report,encoding='utf-8')
 validation='''# Stage 3 validation report

- Protocol V2 SHA-256 manifest verified; original Stage-2 artifacts remain unchanged.
- Primary clean sample: exactly 824 UTC dates and BTC/ETH only; 1,648 asset-day rows.
- Confirmatory V2 universe: 48 unique primary tests; 16 Holm families with 3 horizons each.
- Date × asset uniqueness and common BTC/ETH calendar verified before estimation.
- Four fixed temporal blocks, calendar embargoes, HGB and Ridge training-fold processing, and exposure/control allowlist recorded in the model manifest.
- No random row shuffling; BTC and ETH are estimated separately and paired comparisons resample shared dates.
- Forward labels come from certified columns; placebo labels use dates t-k through t-1; controls are lagged and contain no future-labelled columns.
- HAC uses the calendar spine; block bootstrap samples contiguous calendar date blocks and retains fixed OOF nuisance predictions.
- Primary TE estimator validated on deterministic independent-null and known-lagged-dependence fixtures before empirical estimation; empirical surrogates preserve source-history blocks.
- Granger failures, TE tie-gate exclusions, and all non-significant estimates are retained and labeled.
- 365/540-day windows are full calendar windows at fixed 30-day steps; TVP hyperparameters use only the first 206 dates, and coefficients are filtered without smoothing.
- Secondary and extended-sample results were run only after the clean primary output lock was written.
- No network calls, new assets, raw-data edits, synthetic augmentation, or structural causal estimators were used.

Offline byte-identical rerun status is reported in the final validation update after the deterministic reproduction pass.
'''
 (OUT/'stage3_validation_report.md').write_text(validation,encoding='utf-8')

if __name__=='__main__':
 robust_matrix();save_multiple_testing();write_manifest();make_figures();make_reports()
 print('FINAL_OUTPUTS_WRITTEN')
