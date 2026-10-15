#!/usr/bin/env python3
"""Deterministic offline recheck of post-lock Stage-3 outputs."""
from __future__ import annotations
import hashlib, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def run(script: str) -> None:
    subprocess.run([sys.executable, str(ROOT / 'src' / script)], cwd=ROOT,
                   check=True)


def snapshot(names):
    return {name: sha(OUT / name) for name in names}


def main():
    post_lock = ['table_stage3_secondary.csv', 'table_stage3_extended_horizons.csv',
                 'table_stage3_extended_sample.csv']
    before = snapshot(post_lock)
    run('run_stage3_secondary.py')
    run('run_stage3_extended_sample.py')
    after = snapshot(post_lock)
    if before != after:
        changed = [k for k in before if before[k] != after[k]]
        raise RuntimeError(f'Post-lock offline reproduction differs: {changed}')

    reports = ['table_stage3_multiple_testing.csv', 'table_stage3_robustness_matrix.csv',
               'stage3_model_manifest.csv', 'stage3_research_summary.md',
               'fig_channel1_funding_rv.png', 'fig_channel2_oi_rv.png',
               'fig_channel3_oi_downside.png', 'fig_channel4_flow_amihud.png',
               'fig_direction_summary_btc.png', 'fig_direction_summary_eth.png',
               'fig_time_variation_core_channels.png', 'fig_tvp_volatility_btc.png',
               'fig_tvp_volatility_eth.png']
    run('finalize_stage3_outputs.py')
    report_once = snapshot(reports)
    run('finalize_stage3_outputs.py')
    report_twice = snapshot(reports)
    if report_once != report_twice:
        changed = [k for k in reports if report_once[k] != report_twice[k]]
        raise RuntimeError(f'Final reporting is not byte-identical: {changed}')
    result = {
        'status': 'PASS',
        'network_access': 'none; invoked Stage-3 scripts use only cached local data',
        'secondary_extended_two_run_byte_identity': True,
        'reporting_two_run_byte_identity': True,
        'secondary_extended_sha256': after,
        'report_sha256': report_twice,
    }
    (OUT / 'stage3_reproduction_check.json').write_text(
        json.dumps(result, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    print('STAGE3_POST_LOCK_REPRODUCTION=PASS')


if __name__ == '__main__':
    main()
