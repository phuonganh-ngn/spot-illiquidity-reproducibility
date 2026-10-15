#!/usr/bin/env python3
"""Audit sources and freeze Phase 14 before any new model fitting."""
from __future__ import annotations
import hashlib, importlib.metadata, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/phase14_strong_baseline_benchmark"
sys.path.insert(0, str(ROOT / "src"))
import phase6_benchmark_core as core

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def dump(path: Path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False,default=str)+"\n",encoding="utf-8")

def main():
    for d in ["predictions","development_folds","training_logs","figures","scripts","tables"]:
        (OUT/d).mkdir(parents=True,exist_ok=True)
    phase6=ROOT/"outputs/phase6_autonomous_benchmark"
    phase13=ROOT/"outputs/phase13_neural_target_scaling"
    dataset=ROOT/"outputs/phase5_dynamic_discovery/dataset_4h.jsonl"
    required=[dataset,ROOT/"src/phase6_benchmark_core.py",ROOT/"src/phase6_run_development.py",ROOT/"src/phase6_evaluate_final.py",
      phase6/"benchmark_protocol.json",phase6/"model_registry.json",phase6/"feature_registry.json",phase6/"target_registry.json",
      phase6/"temporal_folds.json",phase6/"selected_models.json",phase6/"development_results.json",phase6/"data_validation.json",
      phase13/"01_frozen_configuration_registry.json",phase13/"16_source_manifest_after.json",phase13/"19_final_experiment_manifest.json",
      phase13/"19_final_experiment_manifest.sha256",ROOT/"outputs/phase12_technical_revision_audit/11_audit_manifest.json"]
    missing=[str(p.relative_to(ROOT)) for p in required if not p.is_file()]
    if missing: raise SystemExit("Required frozen source missing: "+", ".join(missing))

    # Validate the already-frozen Phase 6/13 checksums and the Phase 13 source audit.
    p6hash=(phase6/"benchmark_protocol.sha256").read_text().split()[0]
    if sha(phase6/"benchmark_protocol.json")!=p6hash: raise SystemExit("Phase 6 protocol hash mismatch")
    p6selected=(phase6/"selected_models.sha256").read_text().split()[0]
    if sha(phase6/"selected_models.json")!=p6selected: raise SystemExit("Phase 6 selected-model hash mismatch")
    p13checksum=(phase13/"19_final_experiment_manifest.sha256").read_text().split()[0]
    if sha(phase13/"19_final_experiment_manifest.json")!=p13checksum: raise SystemExit("Phase 13 final manifest hash mismatch")
    p13manifest=json.loads((phase13/"19_final_experiment_manifest.json").read_text())
    p13bad=[]
    for item in p13manifest.get("generated_output_files_excluding_final_manifest_and_checksum",[]):
        path=phase13/item["path"]
        if not path.is_file() or path.stat().st_size!=item["size_bytes"] or sha(path)!=item["sha256"]: p13bad.append(item["path"])
    if p13bad: raise SystemExit(f"Phase 13 generated output checksum mismatch: {p13bad[:5]}")
    p13source=json.loads((phase13/"16_source_manifest_after.json").read_text())
    if not all(p13source.get(k) is True for k in ["all_unchanged","phase6_unchanged","phase9_unchanged","phase12_unchanged","manuscripts_unchanged","dataset_4h_unchanged"]):
        raise SystemExit("Phase 13 source manifest did not certify unchanged source artifacts")

    protocol=json.loads((phase6/"benchmark_protocol.json").read_text())
    registry=json.loads((phase6/"model_registry.json").read_text())
    phase13cfg=json.loads((phase13/"01_frozen_configuration_registry.json").read_text())
    # Reuse source artifact hashes, but hash the actual files now before Phase 14.
    source_roots=[ROOT/f"outputs/{x}" for x in ["phase6_autonomous_benchmark","phase7_liquidity_replication","phase8_liquidity_generalization","phase9_external_qml_replication","phase10_manuscript_evidence_audit","phase12_technical_revision_audit","phase13_neural_target_scaling"]]
    source_files={}
    for directory in source_roots:
        if not directory.is_dir(): raise SystemExit(f"Required historical output directory missing: {directory}")
        for f in sorted(directory.rglob("*")):
            if f.is_file():
                rel=str(f.relative_to(ROOT))
                source_files[rel]={"sha256":sha(f),"size_bytes":f.stat().st_size}
    extra_sources=[dataset,ROOT/"src/phase6_benchmark_core.py",ROOT/"src/phase6_run_development.py",ROOT/"src/phase6_evaluate_final.py"]
    for f in extra_sources:
        rel=str(f.relative_to(ROOT)); source_files[rel]={"sha256":sha(f),"size_bytes":f.stat().st_size}

    # Read only the certified development prefix. No final-test value is parsed here.
    df=core.load_panel_dev_only()
    if (df.period_start_utc>=core.SPLIT).any(): raise SystemExit("Development-only loader included a final-test timestamp")
    task_counts={}; fold_rows=[]
    for asset in ["BTC","ETH"]:
        task=core.make_task(df,asset,6,"future_amihud")
        if not len(task["y"]): raise SystemExit(f"No development rows for {asset}")
        # Phase 6's complete-case rule is based on all ten frozen variables.
        if not __import__('numpy').isfinite(task["y"]).all(): raise SystemExit(f"Nonfinite development target for {asset}")
        task_counts[asset]={"n_complete_development_origins":int(len(task["y"])),"first_origin":task["date"].min().isoformat(),"last_origin":task["date"].max().isoformat()}
        for fold in core.DEV_FOLDS:
            name,start,end=fold
            tr,va=core.fold_masks(task,fold)
            if tr.sum()<500 or va.sum()<100: raise SystemExit(f"Unexpected fold coverage: {asset}/{name}")
            if (task["target_end"][tr]>=start).any() or (task["target_end"][va]>=end).any(): raise SystemExit(f"Target window crosses fold boundary: {asset}/{name}")
            fold_rows.append({"asset":asset,"fold":name,"train_n":int(tr.sum()),"validation_n":int(va.sum()),"train_first":task["date"][tr].min().isoformat(),"train_last":task["date"][tr].max().isoformat(),"validation_first":task["date"][va].min().isoformat(),"validation_last":task["date"][va].max().isoformat(),"train_target_end_last":task["target_end"][tr].max().isoformat(),"validation_target_end_last":task["target_end"][va].max().isoformat()})

    # Model candidates: archived Phase 6 grids where compatible; Phase 13-selected neural configs fixed.
    archived={m["id"]:m for m in registry["models"]}
    def configs(model): return archived[model]["configs"]
    fixed_neural={}
    for asset in ["BTC","ETH"]:
      for info in ["SPOT_ONLY","SPOT_PLUS_ALL_DERIVATIVES"]:
        for family in ["MLP","GRU"]:
            k=f"{asset}__{family}__{info}"
            fixed_neural[k]=phase13cfg["configs"][k]["selected_config"]
    # Fixed compact configurations are preregistered here because Phase 6 explicitly excluded these libraries.
    cat_grid=[{"depth":d,"iterations":200,"learning_rate":0.05,"loss_function":"RMSE","l2_leaf_reg":3.0,"random_strength":1.0,"thread_count":1,"allow_writing_files":False,"verbose":False} for d in [4,6,8]]
    xgb_grid=[{"max_depth":d,"n_estimators":200,"learning_rate":0.05,"min_child_weight":1.0,"subsample":1.0,"colsample_bytree":1.0,"reg_lambda":1.0,"objective":"reg:squarederror","tree_method":"hist","n_jobs":1} for d in [2,4,6]]
    model_registry={
      "status":"FROZEN_BEFORE_PHASE14_FITTING",
      "model_selection":"For each asset x information set, select each tunable family configuration by pooled concatenated F1-F3 validation MAE, then select the system with lowest pooled development MAE across references and selected family configurations. Test data are not inputs to selection.",
      "models":{
       "TrainingMean":{"configs":[{"rule":"mean of training-fold Z labels"}],"complexity_rank":0},
       "Persistence":{"configs":[{"rule":"log1p current spot Amihud, using frozen Phase 6 persistence definition"}],"complexity_rank":1},
       "Ridge":{"configs":configs("Ridge"),"complexity_rank":2},
       "HistGradientBoosting":{"configs":configs("HistGradientBoosting"),"complexity_rank":3},
       "ExtraTrees":{"configs":configs("ExtraTrees"),"complexity_rank":4},
       "CatBoost":{"configs":cat_grid,"complexity_rank":5,"registry_status":"fixed grid authored for Phase 14 because Phase 6 excluded this family"},
       "XGBoost":{"configs":xgb_grid,"complexity_rank":6,"registry_status":"fixed grid authored for Phase 14 because Phase 6 excluded this family"},
       "MLP":{"fixed_configs":{k:v for k,v in fixed_neural.items() if "__MLP__" in k},"complexity_rank":7,"registry_status":"exact Phase 13 selected architecture per asset/information set; not retuned"},
       "GRU":{"fixed_configs":{k:v for k,v in fixed_neural.items() if "__GRU__" in k},"complexity_rank":8,"registry_status":"exact Phase 13 selected architecture per asset/information set; not retuned"}},
      "feature_preprocessing":"Phase 6 training-only StandardScaler. For GRU fit feature scaler on training rows x six time steps; for other models fit on flattened frozen lag features. No imputation, clipping, feature selection, or global standardization.",
      "target":"frozen Phase 6 future_amihud = arithmetic mean of spot_amihud[t+1..t+6]; Z=log1p(max(Y,0)). Only MLP/GRU use StandardScaler fitted on eligible training Z; inverse transform before scoring. Predictions from every model are scored raw on Z without clipping.",
      "models_to_evaluate":["TrainingMean","Persistence","Ridge","ExtraTrees","HistGradientBoosting","CatBoost","XGBoost","MLP","GRU"],
      "information_sets":["SPOT_ONLY","SPOT_PLUS_ALL_DERIVATIVES"],
      "primary_metric":"MAE on original log1p target Z; lower is better",
      "model_family_config_selection":"For each model family and information set, choose the registered config with lowest pooled chronological F1-F3 MAE. Ties use the locked configuration order.",
      "system_selection_complexity_order":["TrainingMean","Persistence","Ridge","HistGradientBoosting","ExtraTrees","CatBoost","XGBoost","MLP","GRU"],
      "seeds":{"base_seed":core.SEED,"development":"Phase 6 stable_seed(task_id, information_set, family, config_index, fold_name); fixed-config neural models use Phase 13 archived selected config index.","final":"Phase 6 stable_seed('final', asset, 'future_amihud', 6, information_set, model_family); the selected config is locked before test parsing."},
      "training_budget":{"Phase6_candidate_configs":3,"CatBoost_configs":3,"XGBoost_configs":3,"Phase13_MLP_configs_per_asset_info":1,"Phase13_GRU_configs_per_asset_info":1,"development_folds":3,"early_stopping":False,"training_per_fit_only":True},
      "neural":{"target_standardization":"StandardScaler on training labels only; inverse transform predictions before MAE.","MLP":{"architecture":"one hidden ReLU layer","optimizer":"Adam","alpha":0.001,"learning_rate_init":0.001,"max_iter":400,"early_stopping":False,"shuffle":True,"tol":1e-5,"n_iter_no_change":30},"GRU":{"layers":1,"sequence_length":6,"optimizer":"Adam","learning_rate":0.005,"weight_decay":0.001,"epochs":20,"batch":"full batch","output":"linear scalar from final hidden state","gradient_clipping":False,"early_stopping":False}},
      "inference":{"primary_block_length":42,"sensitivity_block_length":84,"replications":1000,"ci":"nominal percentile interval from moving-block resampling of paired loss differences","p":"two-sided null-centered moving-block bootstrap; center d_t by observed mean, inclusive absolute-tail comparison, add-one finite-resample p=(1+count)/(B+1)","holm_family":"two primary BTC/ETH tests, separately for block 42 and block 84; only 42-bar family is primary","seeds":{"BTC_block42":20261442,"ETH_block42":20261443,"BTC_block84":20261484,"ETH_block84":20261485}},
      "retrospective_status":"Historical final-test period was inspected in Phases 6/13. Phase 14 is a retrospective reanalysis and not independent confirmation.",
      "sources":{"phase6_protocol_sha256":p6hash,"phase6_selected_models_sha256":p6selected,"phase13_final_manifest_sha256":p13checksum}}

    # Confirm all referenced model grids and neural specs exist before locking.
    if set(core.INFOSETS["SPOT_ONLY"])!=set(protocol["features"]["spot_only"]): raise SystemExit("Spot feature list mismatch with frozen Phase 6 registry")
    if set(core.INFOSETS["SPOT_PLUS_ALL_DERIVATIVES"])!=set(protocol["features"]["spot_only"]+protocol["features"]["derivatives_all"]): raise SystemExit("S1 feature list mismatch with frozen Phase 6 registry")
    if not all("__MLP__" in k or "__GRU__" in k for k in fixed_neural): raise SystemExit("Unexpected neural registry key")

    import catboost,xgboost,sklearn,torch,pandas,numpy,matplotlib
    env={"python":sys.version,"numpy":numpy.__version__,"pandas":pandas.__version__,"scikit_learn":sklearn.__version__,"torch":torch.__version__,"catboost":catboost.__version__,"xgboost":xgboost.__version__,"matplotlib":matplotlib.__version__}
    manifest={"created_utc":__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),"source_files":source_files,"source_file_count":len(source_files),"dataset_full_sha256":sha(dataset),"phase6_protocol_sha256":p6hash,"phase6_selected_models_sha256":p6selected,"phase13_manifest_sha256":p13checksum,"python_environment":env}
    dump(OUT/"01_source_manifest_before.json",manifest)
    audit=("# Phase 14 source audit\n\nStatus: **PASS**. The frozen Phase 6 target/feature registries, chronological fold definitions, Phase 13 selected neural configurations, Phase 6 source code, certified 4-hour panel, and prior Phase 10/12/13 audit records were verified before any Phase 14 fitting.\n\n"
      f"The certified panel is `{dataset.relative_to(ROOT)}` (SHA-256 `{sha(dataset)}`). Phase 6 protocol hash: `{p6hash}`; selected-model registry hash: `{p6selected}`. Phase 13 final output manifest checksum verified: `{p13checksum}`; all {len(p13manifest.get('generated_output_files_excluding_final_manifest_and_checksum',[]))} files it covers match their recorded hashes. Phase 13 source audit flags all monitored sources unchanged.\n\n"
      f"The development-only loader parsed only timestamps before {core.SPLIT.isoformat()}; it contained {sum(v['n_complete_development_origins'] for v in task_counts.values())} complete BTC/ETH origins for this task. Fold sizes and target-end purges are recorded in `04_chronological_split_audit.md`. No test outcome was used to define the Phase 14 registry.\n\n"
      f"Frozen features match Phase 6 exactly: S0 has {len(core.INFOSETS['SPOT_ONLY'])} variables over six rows; S1 uses the same rows plus the four registered derivatives variables. The source implementation uses a future arithmetic mean of `spot_amihud[t+1:t+7]`, log1p target, oldest-to-newest t−5…t inputs, and requires target windows to end strictly before each development boundary.\n\n"
      f"Available libraries were verified before locking: CatBoost {catboost.__version__}, XGBoost {xgboost.__version__}, scikit-learn {sklearn.__version__}, PyTorch {torch.__version__}. No data download or model fitting occurred during this audit.\n\n"
      "Historical data/test outcomes were already inspected in prior phases. All Phase 14 findings must therefore be labeled retrospective, even though the Phase 14 model-selection choices are based only on its chronological development folds.\n")
    (OUT/"00_source_audit.md").write_text(audit,encoding="utf-8")
    dump(OUT/"03_frozen_model_registry.json",model_registry)
    split_md="# Phase 14 chronological split audit\n\n"
    split_md+=f"Development date range: {protocol['development_period'][0]} through {protocol['development_period'][1]}. Final test feature-origin range: {protocol['final_test_period'][0]} through {protocol['final_test_period'][1]}. Split timestamp: {core.SPLIT.isoformat()}.\n\n"
    split_md+="The exact original expanding validation folds are F1: 2022-10-01 to 2023-01-01; F2: 2023-01-01 to 2023-04-01; F3: 2023-04-01 to 2023-07-01. For each fold, training origins require target_end < validation_start; validation origins require origin < validation_end and target_end < validation_end. Final model training requires origin < 2023-07-01 and target_end < 2023-07-01. Test prediction origins require origin >= 2023-07-01 and target_end < 2024-03-04.\n\n"
    split_md+="All S0/S1 models use the same complete-case rows because Phase 6 requires all ten certified raw features to be finite over t−5…t, even for S0, and all six future Amihud bars to be present. No gaps are imputed. Development-only timestamps and fold sample counts:\n\n"
    for row in fold_rows: split_md+=f"- {row['asset']} {row['fold']}: train n={row['train_n']} ({row['train_first']}–{row['train_last']}, last target end {row['train_target_end_last']}); validation n={row['validation_n']} ({row['validation_first']}–{row['validation_last']}, last target end {row['validation_target_end_last']}).\n"
    split_md+="\nTest values are excluded from selection. Phase 14 selection is locked before the full-panel loader is called.\n"
    (OUT/"04_chronological_split_audit.md").write_text(split_md,encoding="utf-8")
    protocol_text=f"""# Phase 14 locked protocol — strong-baseline retrospective benchmark

## Status and question
This is a retrospective reanalysis of the BTC/ETH 24-hour future Amihud task. The historical Phase 6 test interval has already been examined. The primary question is whether a system selected using only pooled chronological development MAE, and allowed to use the frozen S1 features, has lower test MAE than the independently selected S0 system. No result is independent confirmation and no Phase 6–13 conclusion or artifact is replaced.

## Frozen data, target and information
Use the exact Phase 6 certified 4-hour panel and task constructor. Target Y_t is the archived mean of spot Amihud over t+1…t+6, where spot Amihud and all eligibility rules remain as implemented in `src/phase6_benchmark_core.py`; score Z_t=log1p(max(Y_t,0)). Inputs are exactly six frozen rows t−5,…,t, ordered oldest to newest. S0 is the six spot features; S1 is the same spot features plus the four frozen Funding/OI-change/futures-flow/basis features. All ten features must be complete in both information sets. No new features, imputation, or data download.

## Chronology and purging
Use the original F1/F2/F3 expanding development folds and final test dates from the Phase 6 protocol. In each fold, training label endpoints must be strictly before validation start, and validation target endpoints strictly before validation end. Final training labels must end before 2023-07-01. Test forecast origins must be in 2023-07-01 00:00 through 2024-03-03 20:00 UTC with target endpoints before 2024-03-04 00:00. Do not parse test values until all family configs and BEST_S0/BEST_S1 identities have been selected from development predictions and their freeze file is hashed.

## Models and bounded tuning
The exact candidate registry is `03_frozen_model_registry.json`. It reuses the Phase 6 three-config Ridge, ExtraTrees and HistGradientBoosting grids; uses the Phase 13 selected BTC/ETH MLP width-8 and GRU hidden-width-16 settings with their registered optimizers, regularization, iteration/epoch limits and no early stopping; and defines three fixed, small depth grids each for CatBoost and XGBoost because Phase 6 explicitly excluded those libraries. CatBoost/XGBoost package versions and parameters are locked. Training mean and persistence are included as fixed references and as eligible system candidates. Each model-family/config is evaluated under both information sets. No grid expansion, test tuning, new architecture, or seed search.

## Preprocessing and scoring
Use training-only StandardScaler for predictors as in Phase 6. For MLP/GRU only, standardize Z using a StandardScaler fit to eligible training labels; inverse-transform predictions with that fold's training-only mean and scale. A degenerate target scale is a logged failure, with no fallback. Score every model's un-clipped prediction on the original Z scale; no model-specific clipping or postprocessing. Record convergence, fit warnings, train loss, iteration/epoch count, runtime, and forecast distributions. All metrics are on identical origins.

## Development selection
For each model family × asset × information set, choose its configuration with the smallest pooled MAE over concatenated eligible F1–F3 validation predictions; ties use the frozen candidate order. BEST_S0 and BEST_S1 are selected independently for each asset as the lowest pooled development-MAE system among the nine registered families/references. If numerical ties remain, apply the fixed complexity order in the registry, then lexicographic model identifier. Save selection evidence and SHA-256 freeze before loading any final-test feature/target values. The test set is never used to change these choices.

## Primary endpoint and inference
For each asset, d_t=|Z_t−pBEST_S0,t|−|Z_t−pBEST_S1,t| and Delta_best=mean(d_t). Positive values favor S1. Primary metric is absolute MAE on Z; relative reduction is 100×Delta_best/MAE(BEST_S0). Use a 42-row moving-block bootstrap (1,000 draws) as primary and 84 rows as sensitivity, with fixed asset/block seeds in the registry. Sample contiguous non-circular blocks uniformly from all valid starts, concatenate ceil(n/block) blocks and truncate to n. Report nominal percentile 95% intervals. For the two-sided p-value, bootstrap d_t−mean(d), compare absolute replicate means to |Delta_best| using inclusive tails and add-one p=(1+tail count)/(B+1). Holm-adjust raw p-values across BTC/ETH separately for each block length; 42-block family is primary. Intervals are nominal and not multiplicity-adjusted. These are retrospective diagnostics on previously viewed test data.

## Descriptive result labels
For the primary 42-row analysis, label an asset `FAVORABLE` only when Delta_best > 0, the nominal 95% interval is entirely positive, and the two-asset Holm-adjusted p < 0.05. Label `ADVERSE` only when Delta_best < 0, the interval is entirely negative, and Holm-adjusted p < 0.05. Otherwise label `INCONCLUSIVE`, while retaining the point-estimate sign. These are retrospective inferential labels, not fresh confirmation.

## Reporting and integrity
Report every family under S0/S1, all selected configurations, development and test metrics, paired dates, convergence and runtime, baselines, the primary selected-system comparison, Phase 6 and Phase 13 results as different estimands, and the negative/uncertain outcomes. Preserve frozen Phase 6 outcome 2/18, Phase 7–8 generalization failure, historical QML record and Bybit `NO_EXTERNAL_QML_REPLICATION`. Do not modify previous outputs, data, QML, Bybit, or manuscript; do not claim causality, trading value, or quantum advantage. All Phase 14 results are retrospective reanalysis.
"""
    (OUT/"02_locked_experiment_protocol.md").write_text(protocol_text,encoding="utf-8")
    lock={"status":"LOCKED_BEFORE_FIT","created_utc":manifest["created_utc"],"files":{n:sha(OUT/n) for n in ["02_locked_experiment_protocol.md","03_frozen_model_registry.json","04_chronological_split_audit.md","00_source_audit.md","01_source_manifest_before.json"]}}
    dump(OUT/"pretraining_lock.json",lock)
    lock_hash=sha(OUT/"pretraining_lock.json")
    (OUT/"pretraining_lock.sha256").write_text(lock_hash+"  pretraining_lock.json\n",encoding="utf-8")
    print("PHASE14_SOURCE_AUDIT_PASS")
    print("PRETRAINING_LOCK_SHA256",lock_hash)
    print("SOURCE_FILE_COUNT",len(source_files))
    print("DEV_COUNTS",json.dumps(task_counts,sort_keys=True))
    print("FOLD_RECORDS",len(fold_rows))
    print("MODEL_REGISTRY_SHA256",sha(OUT/"03_frozen_model_registry.json"))

if __name__=="__main__": main()
