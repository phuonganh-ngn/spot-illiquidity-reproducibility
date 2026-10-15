from __future__ import annotations
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs'/'targeted_24h_amihud'
SRC=ROOT/'src'/'targeted_24h_amihud'
GUARD=SRC/'offline_guard'
SEED_FILES=[
 'analysis_panel.csv.gz','classical_group_checkpoints.jsonl','qml_residual_checkpoints.jsonl',
 'study_protocol.md','study_protocol.json','qml_environment.json','qml_fixture_validation.json',
 'qml_residual_execution_plan.json','qml_solver_amendment_001.md','qml_solver_amendment_001.sha256',
 'data_validation.json'
]
RESULT_FILES=[
 'classical_development_results.json','spot_only_models.json','spot_plus_derivatives_models.json',
 'direct_incremental_results.json','best_model_dominance.json','spot_oof_predictions.json',
 'spot_oof_residuals.json','paired_bootstrap.json','loss_differential_tests.json',
 'multiple_model_correction.json','model_confidence_set.json','component_ablation.json',
 'drop_one_ablation.json','seasonality_ablation.json','joint_state_results.json',
 'residual_learning_results.json','negative_controls.json','regime_exploration.json',
 'quantum_kernel_results.json','qnn_results.json','hybrid_qnn_results.json',
 'quantum_vs_classical.json','qml_paired_bootstrap.json','quantum_global_holm_sensitivity.json',
 'joint_state_classification.json','btc_classification.json','eth_classification.json',
 'paper_level_classification.json','research_summary.md'
]
VOLATILE_KEYS={'elapsed_seconds','fit_seconds','outer_fit_seconds','training_seconds','runtime_seconds'}

def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()

def canonical(value):
    if isinstance(value,dict):
        return {k:canonical(v) for k,v in sorted(value.items()) if k not in VOLATILE_KEYS}
    if isinstance(value,list): return [canonical(x) for x in value]
    return value

def result_fingerprint(name):
    p=OUT/name
    if p.suffix=='.json':
        obj=json.loads(p.read_text())
        blob=json.dumps(canonical(obj),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        return hashlib.sha256(blob).hexdigest()
    return file_hash(p)

def tree_hash(path,exclude_target=False):
    path=Path(path); h=hashlib.sha256(); count=0
    if not path.exists(): return {'sha256':None,'files':0}
    for p in sorted(x for x in path.rglob('*') if x.is_file()):
        rel=p.relative_to(path).as_posix()
        if exclude_target and rel.startswith('targeted_24h_amihud/'): continue
        h.update(rel.encode()); h.update(b'\0'); h.update(file_hash(p).encode()); h.update(b'\n'); count+=1
    return {'sha256':h.hexdigest(),'files':count}

def inputs_snapshot():
    paths=['data/raw','data/interim','data/processed','outputs/clean_lock_originals',
           'outputs/phase5_dynamic_discovery','outputs/phase5_dynamic_discovery_rerun',
           'outputs/phase6_autonomous_benchmark','outputs/phase7_liquidity_replication']
    snap={p:tree_hash(ROOT/p,exclude_target=(p.startswith('outputs/'))) for p in paths}
    # Include Stage-3 files at the output root, outside the directory snapshots.
    stage3=sorted(ROOT.glob('outputs/stage3*'))
    h=hashlib.sha256(); n=0
    for p in stage3:
        if p.is_file(): h.update(p.name.encode()); h.update(file_hash(p).encode()); n+=1
    snap['outputs/stage3_root']={'sha256':h.hexdigest(),'files':n}
    return snap

def frozen_hash_checks():
    pre=json.loads((OUT/'pre_model_freeze.json').read_text())
    checks={}
    for name,expected in pre.get('protocol_hashes',{}).items():
        checks[name]={'expected':expected,'actual':file_hash(OUT/name),'pass':file_hash(OUT/name)==expected}
    for name,key in [('qml_environment.json','qml_environment_sha256'),('qml_fixture_validation.json','qml_fixture_validation_sha256')]:
        expected=pre[key]; actual=file_hash(OUT/name)
        checks[name]={'expected':expected,'actual':actual,'pass':expected==actual}
    panel_expected=json.loads((OUT/'data_validation.json').read_text())['analysis_panel_sha256']
    panel_actual=file_hash(OUT/'analysis_panel.csv.gz')
    checks['analysis_panel.csv.gz']={'expected':panel_expected,'actual':panel_actual,'pass':panel_expected==panel_actual}
    qplan_expected=(OUT/'qml_residual_execution_plan.sha256').read_text().split()[0]
    qplan_actual=file_hash(OUT/'qml_residual_execution_plan.json')
    qml_amend=(OUT/'qml_solver_amendment_001.sha256').read_text().split()[0]
    qml_amend_actual=file_hash(OUT/'qml_solver_amendment_001.md')
    checks['qml_residual_execution_plan.json']={'expected':qplan_expected,'actual':qplan_actual,'pass':qplan_expected==qplan_actual}
    checks['qml_solver_amendment_001.md']={'expected':qml_amend,'actual':qml_amend_actual,'pass':qml_amend==qml_amend_actual}
    return checks

def main():
    start=time.time(); logs={}; baseline_inputs=inputs_snapshot(); before={n:result_fingerprint(n) for n in RESULT_FILES}
    env=os.environ.copy(); env['PYTHONPATH']=str(GUARD)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env['TARGETED_STUDY_OFFLINE_GUARD']='1'
    probe=subprocess.run([sys.executable,'-c','import sitecustomize; print(sitecustomize.NETWORK_GUARD_ACTIVE)'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=10)
    guard_active=probe.returncode==0 and probe.stdout.strip()=='True'
    if not guard_active: raise RuntimeError(f'Python socket guard failed to load: {probe.stderr}')
    commands=[
      ('classical_main','run_classical_development.py'),
      ('classical_followups','run_classical_followups.py'),
      ('qml_from_cached_fits','run_qml_residual.py'),
      ('summary','summarize_study.py'),
    ]
    for label,script in commands:
        log=OUT/f'offline_{label}.log'
        with open(log,'w') as f:
            proc=subprocess.run([sys.executable,'-u',str(SRC/script)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=3600)
        logs[label]={'returncode':proc.returncode,'log':log.name}
        if proc.returncode!=0: raise RuntimeError(f'{label} failed offline; inspect {log}')
    after={n:result_fingerprint(n) for n in RESULT_FILES}
    input_after=inputs_snapshot()
    frozen=frozen_hash_checks()
    checks={n:{'before':before[n],'after':after[n],'same':before[n]==after[n]} for n in RESULT_FILES}
    outputs_same=all(x['same'] for x in checks.values())
    inputs_same=baseline_inputs==input_after
    status='PASS' if outputs_same and inputs_same and all(x['pass'] for x in frozen.values()) else 'FAIL'
    out={'status':status,'mode':'offline Python socket guard; external IPv4/IPv6 connects, datagrams, and DNS denied; local AF_UNIX IPC allowed',
      'commands':logs,'network_guard_loaded':guard_active,'result_fingerprints_exclude_runtime_fields':sorted(VOLATILE_KEYS),
      'result_output_fingerprints':checks,'all_result_fingerprints_match':outputs_same,
      'input_tree_hashes_before':baseline_inputs,'input_tree_hashes_after':input_after,'all_certified_and_prior_stage_inputs_unchanged':inputs_same,
      'frozen_protocol_hash_checks':frozen,'all_frozen_hashes_match':all(x['pass'] for x in frozen.values()),
      'cached_fit_reuse':'Classical primary family fits are replayed from the deterministic 108-group checkpoint; post-fit aggregation and historical refits rerun. QML predictions are replayed from the 24 frozen QML fit checkpoints; paired inference and summaries rerun.',
      'elapsed_seconds':time.time()-start}
    (OUT/'reproducibility.json').write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'status':status,'outputs_match':outputs_same,'inputs_unchanged':inputs_same,'frozen_hashes_match':all(x['pass'] for x in frozen.values()),'elapsed_seconds':out['elapsed_seconds']},indent=2))
    if status!='PASS': raise SystemExit(1)

if __name__=='__main__': main()
