#!/usr/bin/env python3
"""Freeze the Phase-6 protocol using development-only inspection metadata."""
from __future__ import annotations
import hashlib,json,importlib.metadata as md,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/phase6_autonomous_benchmark'

def canonical(obj): return json.dumps(obj,sort_keys=True,ensure_ascii=False,allow_nan=False,indent=2)+'\n'

def main():
    inspect=json.loads((OUT/'development_inspection.json').read_text())
    if inspect.get('only_development_values_parsed') is not True or inspect.get('status')!='PASS':
        raise SystemExit('Development-only gate failed')
    features={
      'schema_version':'1.0','information_timing':'features through interval t only; targets begin at t+1',
      'history_window_intervals':6,'history_hours':24,'feature_indices':'t-5,t-4,t-3,t-2,t-1,t, ordered oldest to newest',
      'spot_only':['spot_return','spot_realized_volatility','spot_downside_semivariance','spot_amihud','spot_taker_imbalance','spot_quote_volume_growth'],
      'derivatives_all':['funding_rate_state','oi_log_change','futures_taker_imbalance','basis_mean'],
      'information_sets':{
        'SPOT_ONLY':'six lags of the six frozen spot features',
        'SPOT_PLUS_ALL_DERIVATIVES':'SPOT_ONLY plus six lags of all four frozen derivatives features',
        'SPOT_PLUS_FUNDING':'SPOT_ONLY plus six funding lags',
        'SPOT_PLUS_OI':'SPOT_ONLY plus six OI-log-change lags',
        'SPOT_PLUS_FUTURES_FLOW':'SPOT_ONLY plus six futures-taker-imbalance lags',
        'SPOT_PLUS_BASIS':'SPOT_ONLY plus six basis-mean lags'},
      'complete_case_rule':'Use identical rows for every information set: all 10 features finite at each of t-5 through t, and the full future target window finite. No imputation.'}
    targets={
      'task_type':'continuous regression',
      'horizons_intervals':[1,2,6], 'horizon_hours':[4,8,24],
      'targets':[
        {'id':'future_realized_volatility','formula':'sqrt(sum(spot_realized_volatility[t+k]^2 for k=1..h))','unit':'unannualized cumulative realized volatility','training_transform':'log1p'},
        {'id':'future_downside_semivariance','formula':'sum(spot_downside_semivariance[t+k] for k=1..h)','unit':'squared log-return over future window','training_transform':'log1p'},
        {'id':'future_amihud','formula':'mean(spot_amihud[t+k] for k=1..h)','unit':'scaled Amihud proxy averaged over future window','training_transform':'log1p'}],
      'classification':'Not selected. The primary question is incremental continuous predictive accuracy; event classification would require a training-fold threshold and adds a second estimand. No class imbalance applies to these regression tasks.',
      'assets':['BTC','ETH'],'fit_separately':True,'sample':'4-hour UTC panel; development dates through 2023-06-30 20:00 UTC; final test feature dates from 2023-07-01 00:00 UTC.'}
    folds={
      'design':'3 expanding-origin walk-forward development folds; no random CV',
      'purge':'Training labels must end strictly before validation start; validation labels must end before validation block end. Thus no overlapping future target crosses a split boundary.',
      'folds':[
        {'fold':'F1','train_feature_start':'2021-12-01T00:00:00Z','validation_start':'2022-10-01T00:00:00Z','validation_end_exclusive':'2023-01-01T00:00:00Z'},
        {'fold':'F2','train_feature_start':'2021-12-01T00:00:00Z','validation_start':'2023-01-01T00:00:00Z','validation_end_exclusive':'2023-04-01T00:00:00Z'},
        {'fold':'F3','train_feature_start':'2021-12-01T00:00:00Z','validation_start':'2023-04-01T00:00:00Z','validation_end_exclusive':'2023-07-01T00:00:00Z'}],
      'test':'Untouched feature timestamps 2023-07-01 through 2024-03-03 20:00 UTC; future windows truncated at the certified sample boundary.',
      'tuning':'Each tunable model gets exactly 3 fixed configurations on the same 3 folds for SPOT_ONLY and SPOT_PLUS_ALL_DERIVATIVES; choose minimum pooled validation log-MAE. No final-test use.'}
    models={
      'primary_metric':'MAE on log1p(target); lower is better. All model/information-set comparisons use the same target rows.',
      'secondary_metrics':['RMSE on log1p(target)','MAE and RMSE on original scale','QLIKE on future realized variance (RV target squared)'],
      'models':[
        {'id':'persistence','family':'simple baseline','included':True,'configs':[{'rule':'future RV = current RV*sqrt(h); future downside = current downside*h; future Amihud = current Amihud'}]},
        {'id':'training_mean','family':'simple baseline','included':True,'configs':[{'rule':'mean log1p target in that training fold'}]},
        {'id':'OLS','family':'linear','included':True,'configs':[{'fit_intercept':True}]},
        {'id':'Ridge','family':'regularized linear','included':True,'configs':[{'alpha':0.1},{'alpha':1.0},{'alpha':10.0}]},
        {'id':'RBF_SVR','family':'kernel','included':True,'configs':[{'C':0.1,'gamma':'scale','epsilon':0.05},{'C':1.0,'gamma':'scale','epsilon':0.05},{'C':10.0,'gamma':'scale','epsilon':0.05}]},
        {'id':'ExtraTrees','family':'bagged trees','included':True,'configs':[{'n_estimators':200,'min_samples_leaf':2},{'n_estimators':200,'min_samples_leaf':5},{'n_estimators':200,'min_samples_leaf':10}]},
        {'id':'HistGradientBoosting','family':'gradient boosting','included':True,'configs':[{'max_iter':100,'learning_rate':0.05,'max_leaf_nodes':7},{'max_iter':100,'learning_rate':0.05,'max_leaf_nodes':15},{'max_iter':100,'learning_rate':0.05,'max_leaf_nodes':31}]},
        {'id':'MLP','family':'neural feed-forward','included':True,'configs':[{'hidden_layer_sizes':[8],'alpha':0.001,'max_iter':400},{'hidden_layer_sizes':[16],'alpha':0.001,'max_iter':400},{'hidden_layer_sizes':[32],'alpha':0.001,'max_iter':400}]},
        {'id':'GRU','family':'compact temporal sequence','included':True,'configs':[{'hidden_size':4,'epochs':20,'weight_decay':0.001},{'hidden_size':8,'epochs':20,'weight_decay':0.001},{'hidden_size':16,'epochs':20,'weight_decay':0.001}]},
        {'id':'equal_weight_ensemble','family':'ensemble','included':True,'configs':[{'rule':'unweighted mean of selected OOF predictions from Ridge, RBF_SVR, ExtraTrees, HistGradientBoosting, MLP, GRU; no fitted meta learner'}]}],
      'excluded_families':[
        {'family':'Transformer/LSTM/large sequence architectures','reason':'The effective sample is about 3.2k development rows per asset/task; larger sequence models have an unfavorable parameter-to-sample ratio and are not needed beyond a small GRU.'},
        {'family':'XGBoost/LightGBM/CatBoost','reason':'HistGradientBoosting provides a maintained tree-boosting benchmark without another dependency or materially expanding the bounded model comparison.'},
        {'family':'Stacking/meta-learning','reason':'The sample and number of temporal folds do not justify a fitted meta learner; an equal-weight ensemble uses only development OOF predictions and has no extra tuning.'},
        {'family':'AutoML / deep hyperparameter search','reason':'Unbounded search would invalidate comparable tuning budgets and encourage selection on validation noise.'}],
      'fairness':'Same rows, folds, transformations, scoring, test period, and three-configuration budget for each tunable family and each primary information set.',
      'preprocessing':'StandardScaler fitted within each training fold for linear, kernel, MLP and GRU models. Tree models receive training-fold-scaled features too for a uniform pipeline. No imputation, feature selection, or global normalization.'}
    protocol={
      'phase':'PHASE_6_AUTONOMOUS_PREDICTIVE_BENCHMARK','protocol_status':'FROZEN_BEFORE_FINAL_TEST_EVALUATION',
      'question':'Does adding certified derivatives history improve out-of-sample prediction of future spot risk/liquidity beyond spot history?',
      'claim_scope':'Predictive information only; no causal interpretation.',
      'source_dataset':'outputs/phase5_dynamic_discovery/dataset_4h.jsonl',
      'source_sha256':inspect['input_sha256'],'frequency':'4-hour UTC','assets':['BTC','ETH'],
      'development_period':['2021-12-01T00:00:00Z','2023-06-30T20:00:00Z'],
      'final_test_period':['2023-07-01T00:00:00Z','2024-03-03T20:00:00Z'],
      'prediction_targets':targets,'features':features,'temporal_cv':folds,'model_design':models,
      'leakage_controls':[
        'Inputs at t include only certified features from intervals t-5 through t; each target begins at t+1.',
        'No target window crosses a train/validation split; train labels end strictly before validation starts.',
        'All preprocessing is fitted within each training fold; missing values are not imputed.',
        'The final test values are not parsed during protocol design or model tuning.',
        'No feature selection, target threshold, or scaling statistic uses final-test observations.',
        'The equal-weight ensemble is formed only from development OOF predictions; no in-sample stacking.',
        'All information-set comparisons use the exact same complete-case rows.'],
      'uncertainty':{'method':'paired moving-block bootstrap on final-test prediction rows','block_length_intervals':42,'replications':2000,'seed':20261007,'paired':'same sampled temporal indices for Spot Only and Spot+Derivatives','multiplicity':'Holm-adjust paired bootstrap p-values within each asset x target x horizon family across model categories.'},
      'primary_robustness_rule':{
        'family_categories':['linear (OLS/Ridge)','kernel (RBF_SVR)','bagged trees (ExtraTrees)','boosting (HistGradientBoosting)','neural (mean of MLP and GRU)','equal-weight ensemble'],
        'task_level':'positive average relative log-MAE reduction in at least 4 of 6 family categories; Holm-adjusted paired bootstrap p<0.05 in at least 2 independent categories; mean relative reduction at least 1%; and no two categories show statistically supported deterioration.',
        'asset_replication':'Call a target/horizon replicated only if the task-level rule passes separately for both BTC and ETH.',
        'broad_incremental':'Replicated task-level evidence in at least two distinct targets and at least two horizons.',
        'nonlinear_value':'At least two nonlinear categories (kernel, bagging, boosting, neural) support positive improvements while linear does not meet task-level support.',
        'model_sensitive':'Only one model category supports an improvement, or evidence is isolated and fails replication.',
        'no_robust_value':'Adequate test coverage and uncertainty, but no task meets task-level rule.',
        'inconclusive':'Material pipeline/coverage failure or block uncertainty too wide to distinguish practical improvement from deterioration.'},
      'practical_magnitude':'At least 1% mean relative reduction in log-MAE at the task level; report absolute and relative deltas regardless.',
      'ablation_policy':'After primary configurations are frozen, refit the selected SPOT_PLUS_ALL configuration for each family on SPOT_PLUS_FUNDING/OI/FUTURES_FLOW/BASIS without retuning. These component ablations are secondary and cannot change primary model selection.',
      'stopping_rule':'After one final test evaluation, paired uncertainty, prespecified ablations and development-based interpretability, stop. No new model/feature/horizon or protocol changes.',
      'interpretation':'A predictive improvement indicates incremental out-of-sample information under this sample/split; it does not establish causality or market-wide generalization.'}

    # JSON protocol is immutable once written.
    p=OUT/'benchmark_protocol.json'
    if p.exists(): raise SystemExit('Protocol already exists; refusing to overwrite')
    p.write_text(canonical(protocol))
    h=hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT/'benchmark_protocol.sha256').write_text(h+'  benchmark_protocol.json\n')
    (OUT/'feature_registry.json').write_text(canonical(features))
    (OUT/'target_registry.json').write_text(canonical(targets))
    (OUT/'temporal_folds.json').write_text(canonical(folds))
    (OUT/'model_registry.json').write_text(canonical(models))
    packages={}
    for name in ['numpy','pandas','scipy','scikit-learn','statsmodels','torch']:
        try: packages[name]=md.version(name)
        except md.PackageNotFoundError: packages[name]='NOT_INSTALLED'
    (OUT/'package_versions.json').write_text(canonical({'python':platform.python_version(),'packages':packages,'environment':'.venv_phase5_rerun'}))
    prose=f'''# Phase 6 benchmark protocol\n\nThe confirmatory comparison is Spot Only versus Spot Plus All Derivatives for BTC and ETH, estimated separately. The data are 4-hour UTC observations; development ends 2023-06-30 20:00 UTC and the final test begins 2023-07-01 00:00 UTC. The source panel hash is `{inspect['input_sha256']}`.\n\nThe tasks are continuous forecasts of future realized volatility, downside semivariance, and Amihud illiquidity over 4-hour, 8-hour, and 24-hour windows. Inputs are six intervals of history through t, and every outcome begins at t+1. Positive skewed targets use a fixed log1p training transform. Primary scoring is MAE on the log1p target; original-scale MAE/RMSE and volatility QLIKE are secondary.\n\nThe primary information sets use the same complete rows. Spot Only includes six spot features at lags t-5 through t; Spot Plus All Derivatives adds funding, OI log change, futures taker imbalance, and basis over the same history. Single-component derivative additions are fixed-hyperparameter secondary ablations. Missing values are not imputed. Scaling is trained inside each temporal fold.\n\nModel families are OLS/Ridge, RBF-SVR, ExtraTrees, HistGradientBoosting, a compact MLP, a compact GRU, and a fixed equal-weight OOF ensemble. Each tunable family receives three prespecified configurations on the same three expanding-origin folds. Larger sequence models and additional boosting libraries are excluded because the per-asset development sample is about 3.2k usable rows and the chosen families already cover the relevant model classes.\n\nFinal-test uncertainty uses paired 42-interval moving blocks and 2,000 seeded replications. Holm correction is within each asset × target × horizon model-family comparison. Robust incremental value requires at least four of six family categories to improve, two independent categories to pass Holm-adjusted paired inference, at least 1% average relative log-MAE reduction, and no two categories with supported deterioration. Replication and broader classifications follow the frozen JSON criteria.\n\nThe final test is reserved for one evaluation after development tuning and selected configurations are saved. The protocol JSON is frozen and hashed before any final-test feature or target value is parsed. Predictive improvement will not be described as causal evidence.\n'''
    (OUT/'benchmark_protocol.md').write_text(prose)
    print(f'PHASE6_PROTOCOL_FROZEN sha256={h}')

if __name__=='__main__': main()
