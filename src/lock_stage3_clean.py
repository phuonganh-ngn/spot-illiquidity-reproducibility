#!/usr/bin/env python3
"""Hash clean-sample Stage-3 confirmatory outputs before any secondary/extension grid."""
import csv,hashlib
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
FILES=[
 'data/processed/derivatives_spot_daily_panel.csv',
 'outputs/protocol_manifest.csv','outputs/protocol_manifest_v2.csv',
 'outputs/protocol_amendment_001.md','outputs/primary_hypothesis_registry_v2.csv','outputs/multiple_testing_families_v2.csv',
 'outputs/table_stage3_primary_ts_dml.csv','outputs/table_stage3_forward.csv','outputs/table_stage3_reverse.csv',
 'outputs/table_stage3_falsification.csv','outputs/table_stage3_timing_robustness.csv','outputs/table_stage3_learner_robustness.csv',
 'outputs/table_stage3_forward_crossfit_robustness.csv','outputs/table_stage3_bootstrap.csv','outputs/table_stage3_granger.csv',
 'outputs/table_stage3_stationarity.csv','outputs/table_stage3_transfer_entropy.csv','outputs/table_stage3_net_information_flow.csv',
 'outputs/table_stage3_btc_eth.csv','outputs/table_stage3_time_variation.csv','outputs/stage3_rolling_directionality.csv',
 'outputs/stage3_temporal_endpoint_tests.csv','outputs/stage3_tvp_coefficients.csv','outputs/stage3_tvp_diagnostics.csv',
 'outputs/stage3_crossfit_realized.csv','outputs/table_stage3_robustness_matrix.csv',
 'outputs/channel_classification_summary.csv','outputs/directional_evidence_matrix.csv',
 'data/interim/stage3/primary_oof_scores.pkl',
 'src/stage3_common.py','src/run_stage3_dml.py','src/run_stage3_methods.py','src/run_stage3_temporal.py','src/stage3_primary_finalize.py',
]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for c in iter(lambda:f.read(1<<20),b''):h.update(c)
 return h.hexdigest()
# Stage-2 original and V2 manifests are immutable.
for m in ['outputs/protocol_manifest.csv','outputs/protocol_manifest_v2.csv']:
 with open(ROOT/m,newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
 for r in rows:
  if sha(ROOT/r['file'])!=r['SHA256']:raise SystemExit(f'Protocol hash mismatch: {r["file"]}')
for f in FILES:
 if not (ROOT/f).exists():raise SystemExit(f'Missing clean result at lock: {f}')
# Secondary results must not predate this lock.
for f in ['table_stage3_secondary.csv','table_stage3_extended_horizons.csv','table_stage3_extended_sample.csv']:
 if (OUT/f).exists():raise SystemExit(f'Secondary/extended output exists before clean lock: {f}')
now=datetime.now(timezone.utc).isoformat(timespec='seconds')
with (OUT/'stage3_clean_analysis_lock_manifest.csv').open('w',newline='',encoding='utf-8') as f:
 fields=['file','SHA256','lock_timestamp_utc','role','stage']
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for item in FILES:w.writerow({'file':item,'SHA256':sha(ROOT/item),'lock_timestamp_utc':now,'role':'FROZEN_CLEAN_SAMPLE_PRIMARY_RESULT_OR_INPUT','stage':'Stage 3 before secondary'})
print('CLEAN_PRIMARY_LOCK=PASS')
print('LOCKED_FILES',len(FILES))
print('SECONDARY_RUNS=NOT_YET_EXECUTED')
