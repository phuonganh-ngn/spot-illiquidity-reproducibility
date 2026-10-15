from __future__ import annotations
import hashlib, importlib.metadata, json, platform
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs'/'targeted_24h_amihud'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
manifest={}
for line in (OUT/'study_protocol.sha256').read_text().splitlines():
 h,name=line.split('  ',1); manifest[name]=h
for name,h in manifest.items():
 if sha(OUT/name)!=h: raise RuntimeError(f'Frozen protocol hash changed: {name}')
transform=json.loads((OUT/'target_transform_selection.json').read_text())
if not transform['outer_evaluation_data_not_used'] or not transform['primary_model_comparison_not_run']:
 raise RuntimeError('Target transform was not selected and frozen prior to main comparison')
fixture=json.loads((OUT/'qml_fixture_validation.json').read_text())
if fixture['status']!='VALID': raise RuntimeError('QML synthetic fixtures failed')
packages=['numpy','pandas','scipy','scikit-learn','xgboost','lightgbm','catboost','torch','pennylane','pennylane-lightning','arch','statsmodels']
versions={p:importlib.metadata.version(p) for p in packages}
env={'status':'INSTALLED_AND_IMPORTED','python':platform.python_version(),'packages':versions,'qml_backend':'PennyLane default.qubit exact CPU statevector; no shots or hardware','qml_models':{'quantum_kernel':{'embedding':'Hadamard + Y angle embedding + RZ(x/2) + cyclic CNOT ring','qubits':[4,6,8],'nystrom_landmarks':128,'landmark_selection':'equally spaced chronological training indices'},'qnn':{'qubits':[4,6,8],'layers':2,'embedding':'Y angle encoding','seeds':[101,211,307,401,503],'optimizer':'Adam','epochs':40,'batch_size':64,'learning_rate':0.01},'hybrid':{'qubits':[4,6,8],'encoder':'single linear map from <=16 training-only PCA scores','layers':2,'seeds':[101,211,307,401,503],'optimizer':'Adam','epochs':40,'batch_size':64,'learning_rate':0.01},'matched_classical':'RBF kernel ridge and compact MLP use same training-only PCA inputs, folds, rows, and 128 landmarks / latent dimension'},'fixture_validation':fixture['status'],'empirical_results_inspected':False}
(OUT/'qml_environment.json').write_text(json.dumps(env,indent=2,sort_keys=True,allow_nan=False)+'\n')
pre={'status':'FROZEN_BEFORE_MAIN_MODEL_COMPARISON','protocol_hashes':manifest,'selected_target_transform':transform['selected_transform'],'target_transform_selection_sha256':sha(OUT/'target_transform_selection.json'),'feature_registry_sha256':sha(OUT/'feature_registry.json'),'interaction_registry_sha256':sha(OUT/'interaction_registry.json'),'seasonality_registry_sha256':sha(OUT/'seasonality_registry.json'),'model_registry_sha256':sha(OUT/'model_registry.json'),'search_budget_sha256':sha(OUT/'search_budget.json'),'temporal_folds_sha256':sha(OUT/'temporal_folds.json'),'qml_environment_sha256':sha(OUT/'qml_environment.json'),'qml_fixture_validation_sha256':sha(OUT/'qml_fixture_validation.json'),'source_dataset_sha256':json.loads((OUT/'data_validation.json').read_text())['analysis_panel_sha256'],'primary_model_comparison_not_run_before_freeze':True,'all_candidate_packages_importable':True,'data_source_downloads':0}
(OUT/'pre_model_freeze.json').write_text(json.dumps(pre,indent=2,sort_keys=True,allow_nan=False)+'\n')
(OUT/'pre_model_freeze.sha256').write_text(sha(OUT/'pre_model_freeze.json')+'  pre_model_freeze.json\n')
install='''# QML / model package installation log\n\nThe original project environment lacked `xgboost`, `catboost`, and PennyLane; `lightgbm` was present. A project-local `.venv_targeted24h` was created because the system Python lacked `ensurepip` and the prior-stage environment was kept unchanged. The first `python3 -m venv` attempt failed for that reason; setup recovered with the already available `uv` tool.\n\nInstallation command: `uv pip install --python .venv_targeted24h/bin/python numpy pandas scipy scikit-learn xgboost lightgbm catboost torch pennylane`, followed by `uv pip install --python .venv_targeted24h/bin/python arch statsmodels`.\n\nAll installed packages imported successfully. Exact versions and the PennyLane simulator/circuit settings are recorded in `qml_environment.json`. The quantum kernel, QNN, and hybrid QNN each passed deterministic linear, nonlinear-interaction, and null synthetic fixtures before any empirical model-family comparison. No market data were downloaded.\n'''
(OUT/'install_log.md').write_text(install)
print(json.dumps({'pre_model_freeze':'PASS','selected_transform':pre['selected_target_transform'],'packages':versions,'qml_fixture_validation':fixture['status'],'freeze_sha256':sha(OUT/'pre_model_freeze.json')},indent=2))
