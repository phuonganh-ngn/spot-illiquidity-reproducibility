#!/usr/bin/env python3
"""Create and hash the Phase 13 pre-training lock; performs no model fitting."""
from __future__ import annotations
import csv, hashlib, json, platform, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/phase13_neural_target_scaling"
P6 = ROOT / "outputs/phase6_autonomous_benchmark"
P12 = ROOT / "outputs/phase12_technical_revision_audit"
P9 = ROOT / "outputs/phase9_external_qml_replication"

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)+"\n", encoding="utf-8")

def source_inventory():
    paths=set()
    for folder in [P6,P9,P12,ROOT/"manuscripts"]:
        if folder.exists(): paths.update(x for x in folder.rglob("*") if x.is_file() and "__pycache__" not in x.parts)
    paths.update([ROOT/"src/phase6_benchmark_core.py", ROOT/"src/phase6_run_development.py", ROOT/"src/phase6_evaluate_final.py", ROOT/"outputs/phase5_dynamic_discovery/dataset_4h.jsonl"])
    paths.update(x for x in (OUT/"scripts").rglob("*") if x.is_file())
    rows=[]
    for p in sorted(paths):
        if not p.is_file(): continue
        rows.append({"path":str(p.relative_to(ROOT)),"size_bytes":p.stat().st_size,"sha256":sha(p)})
    return rows

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"predictions").mkdir(exist_ok=True); (OUT/"training_logs").mkdir(exist_ok=True); (OUT/"figures").mkdir(exist_ok=True); (OUT/"scripts").mkdir(exist_ok=True)

    # Verify the prior Phase 12 evidence audit before using its recorded hashes.
    audit=json.loads((P12/"11_audit_manifest.json").read_text())
    prior=audit["final_source_integrity_check"]
    prior_fail=[]
    for r in prior:
        current=ROOT/r["source_path"]
        if not r["unchanged"] or not current.is_file() or sha(current)!=r["final_sha256"]:
            prior_fail.append(r["source_path"])
    if prior_fail: raise SystemExit(f"Phase 12 recorded source hashes do not verify: {prior_fail[:10]}")

    selected=json.loads((P6/"selected_models.json").read_text())
    dev=json.loads((P6/"development_results.json").read_text())
    protocol=json.loads((P6/"benchmark_protocol.json").read_text())
    folds=json.loads((P6/"temporal_folds.json").read_text())
    pred=json.loads((P6/"final_test_predictions.json").read_text())
    metrics=json.loads((P6/"final_test_metrics.json").read_text())
    if selected.get("status")!="FROZEN_AFTER_DEVELOPMENT_CV": raise SystemExit("selected-model registry is not frozen")
    if pred.get("prediction_scale")!="log1p target": raise SystemExit("unexpected historical prediction scale")
    protocol_hash=(P6/"benchmark_protocol.sha256").read_text().split()[0]
    selected_hash=(P6/"selected_models.sha256").read_text().split()[0]
    if sha(P6/"benchmark_protocol.json")!=protocol_hash or sha(P6/"selected_models.json")!=selected_hash:
        raise SystemExit("Phase 6 frozen protocol/config hash mismatch")
    if protocol.get("source_sha256")!=sha(ROOT/"outputs/phase5_dynamic_discovery/dataset_4h.jsonl"):
        raise SystemExit("certified 4h dataset hash differs from frozen Phase 6 protocol")
    if pred.get("protocol_sha256")!=protocol_hash or pred.get("selected_models_sha256")!=selected_hash:
        raise SystemExit("Phase 6 prediction file does not reference frozen protocol/config")
    configs={}
    for asset in ["BTC","ETH"]:
        task_id=f"{asset}__future_amihud__h6"
        for info in ["SPOT_ONLY","SPOT_PLUS_ALL_DERIVATIVES"]:
            key=f"{task_id}__{info}"
            for family in ["MLP","GRU"]:
                model=selected["selected"][key]["models"][family]
                configs[f"{asset}__{family}__{info}"]={
                    "asset":asset,"task_id":task_id,"target":"future_amihud","horizon_intervals":6,
                    "information_set":info,"model_family":family,
                    "selected_config":model["selected_config"],
                    "selected_config_index":model["selected_config_index"],
                    "selected_development_mae_log1p":model["selected_cv_mae_log1p"],
                    "selection_source":"selected_models.json; frozen development-only pooled log1p MAE",
                    "selection_status":"FROZEN_BEFORE_FINAL_TEST"
                }
    registry={
        "experiment":"Phase 13 retrospective neural target-scaling sensitivity",
        "source_protocol_sha256":protocol["source_sha256"],
        "source_selected_models_sha256":sha(P6/"selected_models.json"),
        "target":{"id":"future_amihud","formula":"mean(spot_amihud[t+1:t+6])","training_transform":"log1p(max(y,0))","scoring":"MAE on original log1p target"},
        "primary_assets":["BTC","ETH"],
        "primary_horizon_intervals":6,
        "information_sets":["SPOT_ONLY","SPOT_PLUS_ALL_DERIVATIVES"],
        "feature_order_per_interval":{"SPOT_ONLY":["spot_return","spot_realized_volatility","spot_downside_semivariance","spot_amihud","spot_taker_imbalance","spot_quote_volume_growth"],"SPOT_PLUS_ALL_DERIVATIVES":["spot_return","spot_realized_volatility","spot_downside_semivariance","spot_amihud","spot_taker_imbalance","spot_quote_volume_growth","funding_rate_state","oi_log_change","futures_taker_imbalance","basis_mean"],"sequence_order":"t-5,t-4,t-3,t-2,t-1,t; flattened MLP is row-major time then features"},
        "training_regimes":["CONTROL_REPLAY_UNSCALED_RAW_OUTPUT","TREATMENT_TRAIN_ONLY_STANDARDIZED_TARGET_RAW_OUTPUT"],
        "configs":configs,
        "seed_rule":{
            "development":"SEED + sum(sum(ord(c) for c in str(part)) for part in (task_id, information_set, family, selected_config_index, fold_name))",
            "final_test":"SEED + sum(sum(ord(c) for c in str(part)) for part in ('final', asset, target, horizon, information_set, family))",
            "base_seed":20261007,
            "source":"src/phase6_run_development.py and src/phase6_evaluate_final.py"
        },
        "preprocessing":{"predictors":"Original training-only StandardScaler; feature order and history unchanged","target_treatment":"Unscaled control or StandardScaler fit on training labels only; inverse using same training mean and population std","prediction_postprocessing":"No clipping of negative log-scale forecasts in either matched replay branch; a separate frozen-output-clipped control score is retained solely to compare exact historical implementation."},
        "exploratory_bootstrap":{"replications":1000,"block_lengths_intervals":[42,84],"rng_seed":20261013,"status":"POST_HOC_EXPLORATORY_INTERVALS_ONLY"},
        "architecture":{"MLP":"Original selected one-hidden-layer ReLU MLP; selected width from registry; Adam; alpha=0.001; learning_rate_init=0.001; max_iter=400; early_stopping=False; shuffle=True; tol=1e-5; n_iter_no_change=30","GRU":"Original selected one-layer GRU; selected hidden width from registry; sequence length 6 oldest-to-newest; linear scalar head on final hidden state; Adam lr=0.005; weight_decay=0.001; full batch; 20 epochs; no clipping or early stopping"},
        "folds":folds["folds"],
        "final_test_period":pred["final_test_period"],
        "phase6_frozen_decision":{"tasks":18,"model_rows":180,"passes":2,"passed_task_ids":["BTC__future_amihud__h6","ETH__future_amihud__h6"]},
        "historical_phase6_test_metrics":{
            "source":"final_test_metrics.json; values on common log1p target",
            "selected_rows":{
                f"{t['asset']}__{info}__{family}":t["metrics"][info][family]["mae_log1p"]
                for t in metrics["tasks"] if t["target"]=="future_amihud" and t["horizon_intervals"]==6 and t["asset"] in ["BTC","ETH"]
                for info in ["SPOT_ONLY","SPOT_PLUS_ALL_DERIVATIVES"] for family in ["MLP","GRU"]
            }
        },
        "experimental_status":"RETROSPECTIVE_SENSITIVITY_NOT_INDEPENDENT_CONFIRMATION"
    }
    write_json(OUT/"01_frozen_configuration_registry.json",registry)

    protocol_text="""# Phase 13 locked experiment protocol\n\n## Research question and status\nThis retrospective sensitivity experiment tests whether Phase 6 BTC/ETH 24-hour Amihud MLP and GRU forecasts depend on omitting training-target standardization, and whether the SpotPlusDerivatives versus SpotOnly point pattern persists. Phase 6 final-test outcomes have already been inspected; this is not independent confirmation and cannot replace the frozen 2/18 task decision.\n\n## Frozen scope\nOnly BTC and ETH, target `future_amihud`, six four-hour future bars, and the original Phase 6 `SPOT_ONLY` / `SPOT_PLUS_ALL_DERIVATIVES` information sets are included. Exactly the development-selected model configuration for each asset × information set × neural family is used. The source frozen registry is recorded in `01_frozen_configuration_registry.json`. No model selection, tuning, new features, market data, target redefinition, architecture change, altered split, or new seed is allowed.\n\n## Data, target, and timing\nUse only `outputs/phase5_dynamic_discovery/dataset_4h.jsonl` after verifying its frozen Phase 6 source SHA-256. The target is the existing Phase 6 mean Amihud over t+1,…,t+6, transformed as `log1p(max(y,0))`. Six input rows t−5,…,t are ordered oldest to newest and use the original complete-case rule. Final test origins are 2023-07-01 00:00 through 2024-03-03 20:00 UTC. Training labels must end strictly before the relevant validation/test boundary. Development folds are the three frozen expanding windows in the source registry; final training uses only labels ending strictly before 2023-07-01 00:00 UTC.\n\n## Single intervention and matched replay\nFor each fitting unit, the treatment fits `StandardScaler` to that unit’s eligible training `y_log` values only, trains on `(y_log−mu_train)/sigma_train`, and inverse-transforms predictions using the same training-only `mu_train` and population `sigma_train`. Degenerate sigma is a recorded fitting failure. The control replay fits the same selected architecture on unscaled `y_log`. Both branches use identical rows, predictors, predictor scaling, architecture, optimizer, training schedule, fixed seed, and raw/unclipped log-scale forecast scoring. Neither branch truncates negative model forecasts or clips to the observed target range. For audit comparability, the control replay also records the original Phase 6 lower-zero prediction transform as a separate derived score; this transformed score is not the paired scaling contrast.\n\n## Models and seeds\nMLP and GRU only. MLP settings, including original selected width/config, Adam, alpha 0.001, learning rate 0.001, max 400 iterations, no early stopping, shuffle, tolerance 1e−5, and no-iteration-change 30, remain fixed. GRU settings, including original selected hidden width, single GRU layer + linear output, sequence length six, Adam lr 0.005, weight decay 0.001, full batch, 20 epochs, and no early stopping/gradient clipping, remain fixed. Development and final seeds follow the exact deterministic Phase 6 formulas, recorded in the configuration registry.\n\n## Outcomes\nPrimary metric is MAE on the original `log1p` target scale, after inverse target-scaler transformation. Report historical saved MAE, unscaled replay MAE (raw output), target-scaled replay MAE, and separately the frozen Phase 6 clipped-output control replay. Also report relative change, training-mean, persistence, OLS, Ridge, RBF-SVR, ExtraTrees, HistGradientBoosting, test count, prediction distribution, fold metrics, warnings/convergence, training loss curves, and deterministic ensemble/category sensitivity. A test-selected strongest classical reference is explicitly an oracle-descriptive comparison.\n\n## Retrospective uncertainty\nIf prediction series are complete and matched, compute paired moving-block bootstrap intervals with fixed seed, 1,000 draws, block lengths 42 and 84. These are post-hoc exploratory intervals only. No new frozen PASS/FAIL status is created, and no inference supersedes the registered Phase 6 correction or decision.\n\n## Integrity and limits\nHash protocol/config/training plan/source manifest before any fit. Hash all outputs and recheck source hashes after the run. Preserve negative and failed fits. No manuscript is edited. Report any failure or reproduction discrepancy. This experiment can diagnose target-scaling sensitivity in this previously inspected test sample; it cannot establish independent generalization, causality, trading value, or change the historical Phase 6 decision.\n"""
    protocol_text=protocol_text.replace("## Integrity and limits", """## Prespecified diagnostics
An extreme absolute error is counted descriptively when it exceeds ten times the maximum eligible training `y_log` for that fit. Out-of-observed-range counts use the evaluation subset's observed target minimum and maximum and remain descriptive only.

## Prespecified descriptive classifications
Scaling response is CLEAR_ABSOLUTE_IMPROVEMENT if scaled MAE is below historical saved MAE in all four asset × family cells; MIXED_IMPROVEMENT if in one to three; NO_MEANINGFUL_IMPROVEMENT if in none; any failed fit is separately reported. Competitiveness is COMPETITIVE_WITH_STRONG_CLASSICAL only if every scaled neural cell beats its within-information-set best saved classical test-MAE oracle; COMPETITIVE_WITH_SIMPLE_BASELINE_ONLY if all four beat training-mean but the stronger-classical criterion fails; STILL_INFERIOR_TO_SIMPLE_BASELINE if all four are worse than training-mean; otherwise MIXED_ACROSS_ASSETS. Derivatives point pattern PERSISTS only if all four scaled S0-minus-S1 MAE differences are positive; DISAPPEARS if all are zero within 1e−15; REVERSES if all are negative; otherwise MIXED_ACROSS_ASSETS. Ensemble is SUBSTANTIALLY_IMPROVED if all four asset × information-set reconstructed ensembles lower MAE by at least 50% versus the saved ensemble; PARTIALLY_IMPROVED if one to three cells improve or all improve by less than 50%; UNCHANGED if all differences are within 1e−15; WORSENED if all four increase; otherwise MIXED (reported under PARTIALLY_IMPROVED with the cell pattern stated). These are descriptive conventions frozen before fitting, not inferential decision rules.

## Integrity and limits""")
    (OUT/"00_locked_experiment_protocol.md").write_text(protocol_text,encoding="utf-8")
    training_plan="""# Phase 13 training plan (locked)\n\nRun up to 64 fits: 2 assets × 2 model families × 2 information sets × (3 development folds + final test) × 2 target regimes, with a failed fit retained and no silent altered retry. Fold training/validation indices come from Phase 6 `fold_masks`; final training and test masks are recreated verbatim from `phase6_evaluate_final.py`. Before fitting, assert all fold labels end strictly before their split boundary and all test labels end before the frozen sample end.\n\nFor each fit, train-only scale predictors. Fit MLP on flattened six-step features, or GRU on `(n,6,n_features)` ordered oldest-to-current. The only paired training change is a training-label StandardScaler with inverse transform at prediction. Use exact Phase 6 fixed seeds and selected configs. Keep raw forecast outputs without truncation; separately derive the original Phase 6 nonnegative log-prediction transform for reproduction checking. Persist each origin-level prediction, warning, convergence/iteration information, GRU epoch losses and gradient norms, runtime, and fold/test metric.\n\nAfter every required fit completes, reconstruct baseline comparisons from frozen Phase 6 predictions, construct the equal-weight ensemble by replacing only MLP/GRU components, verify historical category aggregation, and then calculate sensitivity results. Use paired moving-block bootstrap intervals (1,000 draws, fixed seed 20261013, blocks 42/84) only for matched saved prediction vectors. Generate figures solely from those arrays. Recompute source hashes and verify protocol-lock hashes.\n"""
    (OUT/"02_training_plan.md").write_text(training_plan,encoding="utf-8")

    rows=source_inventory()
    write_json(OUT/"03_source_manifest_before.json",{
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "source_inventory":rows,
        "phase12_prior_audit":{"recorded_sources":len(prior),"all_recorded_source_hashes_verified":True},
        "phase6_original_protocol_sha256":(P6/"benchmark_protocol.sha256").read_text().split()[0],
        "phase6_selected_models_sha256":(P6/"selected_models.sha256").read_text().split()[0],
        "market_data_downloaded":False,"model_training_started":False,
        "scope":"Hash inventory of all files in frozen Phase 6, Phase 9, Phase 12 output directories, Phase 6 code, certified 4h input panel, and Phase 13 implementation source."
    })

    # Hash each lock file, plus a bundle hash of the four required artifacts.
    lock_files=["00_locked_experiment_protocol.md","01_frozen_configuration_registry.json","02_training_plan.md","03_source_manifest_before.json"]
    locks=[]
    for name in lock_files: locks.append({"file":name,"sha256":sha(OUT/name)})
    write_json(OUT/"pretraining_lock_manifest.json",{"locked_before_training":True,"files":locks})
    (OUT/"pretraining_lock_manifest.sha256").write_text(sha(OUT/"pretraining_lock_manifest.json")+"  pretraining_lock_manifest.json\n",encoding="utf-8")
    print(json.dumps({"status":"LOCK_CREATED_BEFORE_TRAINING","configs":len(configs),"source_files":len(rows),"phase12_hash_check":"PASS","lock_sha256":sha(OUT/"pretraining_lock_manifest.json"),"package_versions":{"python":platform.python_version()}},indent=2))

if __name__=="__main__": main()
