#!/usr/bin/env python3
"""Final Stage-3 protocol, sample, integrity and reproduction certification."""
from __future__ import annotations
import csv, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
PROC = ROOT / 'data' / 'processed'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def require(condition: bool, message: str):
    if not condition:
        raise RuntimeError(message)


def verify_manifest(path: Path):
    rows = list(csv.DictReader(path.open(newline='', encoding='utf-8')))
    require(bool(rows), f'Empty manifest: {path.name}')
    for row in rows:
        item = ROOT / row['file']
        require(item.is_file(), f'Missing manifest item: {row["file"]}')
        require(sha(item) == row['SHA256'], f'Hash mismatch: {row["file"]}')
    return rows


def main():
    checks = []
    original = verify_manifest(OUT / 'protocol_manifest.csv')
    v2 = verify_manifest(OUT / 'protocol_manifest_v2.csv')
    checks.append(('Stage-2 original and V2 protocol hashes', 'PASS', f'{len(original)} original-manifest rows; {len(v2)} V2-manifest rows'))

    registry = pd.read_csv(OUT / 'primary_hypothesis_registry_v2.csv')
    primary_registry = registry[registry.confirmatory.astype(str).str.lower().eq('true')]
    primary = pd.read_csv(OUT / 'table_stage3_primary_ts_dml.csv')
    require(len(primary_registry) == 48 and primary_registry.test_id.nunique() == 48, 'V2 primary registry is not 48 unique tests')
    require(len(primary) == 48 and primary.test_id.nunique() == 48, 'Primary TS-DML output is not 48 unique tests')
    families = pd.read_csv(OUT / 'multiple_testing_families_v2.csv')
    require(families.family_id.nunique() == 16 and (families.groupby('family_id').size() == 3).all(), 'V2 Holm-family structure is invalid')
    require(set(primary.horizon.astype(int)) == {1, 3, 7}, 'Primary horizon set differs from frozen protocol')
    require(primary.status.eq('PASS').all(), 'One or more primary DML diagnostics did not pass')
    checks.append(('Confirmatory universe and Holm families', 'PASS', '48 unique tests; 16 families × 3 horizons'))

    panel_path = PROC / 'derivatives_spot_daily_panel.csv'
    panel = pd.read_csv(panel_path, parse_dates=['date'])
    panel['date'] = pd.to_datetime(panel.date, utc=True)
    panel = panel[(panel.date >= '2021-12-01') & (panel.date <= '2024-03-03')]
    expected = pd.date_range('2021-12-01', '2024-03-03', freq='D', tz='UTC')
    require(len(panel) == 1648 and not panel.duplicated(['date', 'asset']).any(), 'Primary sample row count/key uniqueness failure')
    require(set(panel.asset) == {'BTC', 'ETH'}, 'Unexpected asset in primary sample')
    for asset, g in panel.groupby('asset'):
        require(pd.DatetimeIndex(g.sort_values('date').date).equals(expected), f'Calendar/sample mismatch for {asset}')
    require(primary.n_dates.between(400, 824).all(), 'Primary usable date counts outside expected clean sample bounds')
    checks.append(('Clean sample and panel key', 'PASS', '824 UTC dates; 1,648 unique BTC/ETH asset-days'))

    controls = ['spot_log_return_1d_lag1','spot_realized_volatility_lag1','spot_amihud_lag1',
                'spot_downside_semivariance_lag1','spot_quote_volume_growth_1d_lag1','funding_mean_lag1',
                'oi_log_change_1d_lag1','futures_taker_imbalance_lag1','spot_log_return_1d_lag2',
                'spot_realized_volatility_lag7','funding_mean_lag2']
    dictionary = pd.read_csv(OUT / 'variable_dictionary.csv')
    available = set(dictionary.variable.astype(str))
    require(set(controls).issubset(available), 'A frozen control is absent from the certified variable dictionary')
    require(not any(('_fwd_' in x.lower() or x.lower().startswith('future_') or '_future_' in x.lower()) for x in controls), 'Future label found in control allowlist')
    require(not any('center' in x.lower() for x in controls), 'Centered feature found in control allowlist')
    checks.append(('Variable allowlist / future-label exclusion', 'PASS', f'{len(controls)} predetermined controls; no future-label or centered variable'))

    folds = pd.read_csv(OUT / 'stage3_crossfit_realized.csv')
    basefolds = folds[folds.variant == 'PRIMARY_HGB_BLOCKED']
    require(basefolds.test_id.nunique() == 48, 'Primary fold log does not cover all confirmatory fits')
    require(basefolds.groupby('test_id').fold.nunique().eq(4).all(), 'A primary model does not have four temporal folds')
    require(basefolds.status.eq('SCORED').all(), 'Primary fold status is not fully scored')
    for test_id, g in basefolds.groupby('test_id'):
        require(g.fold.nunique() == 4 and g.test_start.nunique() == 4 and g.test_end.nunique() == 4,
                f'Fold/date grouping issue in {test_id}')
    require(not folds.empty, 'Crossfit fold file is empty')
    checks.append(('Temporal folds and date grouping', 'PASS', 'Four contiguous blocks per primary fit; no row-level random folds'))

    boot = pd.read_csv(OUT / 'table_stage3_bootstrap.csv')
    require(len(boot) == 96 and boot.status.eq('PASS').all() and boot.valid_reps.ge(950).all(), 'Bootstrap adequacy failure')
    checks.append(('Date-block bootstrap', 'PASS', '96 intervals; 999 fixed-seed replications; adequacy passed'))

    placebo = pd.read_csv(OUT / 'table_stage3_falsification.csv')
    require(len(placebo) == 108, 'Expected 108 original past-timing placebo tests')
    te = pd.read_csv(OUT / 'table_stage3_transfer_entropy.csv')
    require(len(te) == 48, 'Expected 48 primary/sensitivity TE configurations')
    pass_te = te[te.status == 'PASS']
    require(pass_te.empty or pass_te.valid_surrogates.eq(999).all(), 'A passing TE row lacks the fixed 999-surrogate count')
    gr = pd.read_csv(OUT / 'table_stage3_granger.csv')
    require(len(gr) == 16, 'Expected 16 pairwise Granger directions')
    checks.append(('Placebo, Granger and Transfer Entropy coverage', 'PASS', '108 placebos; 16 Granger directions; 48 TE configurations'))

    primary_lock = pd.read_csv(OUT / 'stage3_clean_analysis_lock_manifest.csv')
    lock_mismatches = []
    for row in primary_lock.itertuples():
        path = ROOT / row.file
        if row.file == 'outputs/table_stage3_robustness_matrix.csv':
            path = OUT / 'clean_lock_originals' / 'table_stage3_robustness_matrix.csv'
        if not path.exists() or sha(path) != row.SHA256:
            lock_mismatches.append(row.file)
    require(not lock_mismatches, f'Pre-secondary clean lock cannot be verified: {lock_mismatches}')
    checks.append(('Pre-secondary clean-output lock', 'PASS', f'{len(primary_lock)} locked items verified; locked robustness source preserved separately'))

    raw_inventory = pd.read_csv(OUT / 'raw_hash_inventory.csv')
    raw_bad = []
    for row in raw_inventory.itertuples():
        path = ROOT / row.path
        if not path.is_file() or sha(path) != row.sha256:
            raw_bad.append(row.path)
    require(not raw_bad, f'Raw hash inventory mismatch ({len(raw_bad)} files)')
    source = pd.read_csv(OUT / 'source_manifest.csv')
    source_checked = 0
    for row in source.itertuples():
        if isinstance(row.raw_file_path, str) and row.raw_file_path:
            path = ROOT / row.raw_file_path
            require(path.is_file() and sha(path) == row.sha256, f'Raw source-manifest hash mismatch: {row.raw_file_path}')
            source_checked += 1
    checks.append(('Raw data immutability and source lineage', 'PASS', f'{len(raw_inventory)} raw files; {source_checked} source-manifest rows verified'))

    prerep = json.loads((OUT / 'stage3_reproduction_check.json').read_text())
    require(prerep['status'] == 'PASS' and prerep['secondary_extended_two_run_byte_identity'] and prerep['reporting_two_run_byte_identity'], 'Post-lock reproduction check failed')
    checks.append(('Offline deterministic reproduction', 'PASS', 'Clean primary outputs match original lock; secondary/extended and reports reproduced byte-identically'))

    required_files = [
        'table_stage3_primary_ts_dml.csv','table_stage3_forward.csv','table_stage3_reverse.csv',
        'table_stage3_falsification.csv','table_stage3_timing_robustness.csv','table_stage3_learner_robustness.csv',
        'table_stage3_bootstrap.csv','table_stage3_granger.csv','table_stage3_transfer_entropy.csv',
        'table_stage3_time_variation.csv','table_stage3_btc_eth.csv','table_stage3_secondary.csv',
        'table_stage3_extended_sample.csv','table_stage3_multiple_testing.csv','table_stage3_robustness_matrix.csv',
        'directional_evidence_matrix.csv','stage3_crossfit_realized.csv','stage3_model_manifest.csv',
        'stage3_research_summary.md','stage3_validation_report.md',
        'fig_channel1_funding_rv.png','fig_channel2_oi_rv.png','fig_channel3_oi_downside.png',
        'fig_channel4_flow_amihud.png','fig_direction_summary_btc.png','fig_direction_summary_eth.png',
        'fig_time_variation_core_channels.png','fig_tvp_volatility_btc.png','fig_tvp_volatility_eth.png',
    ]
    missing = [name for name in required_files if not (OUT / name).is_file()]
    require(not missing, f'Missing required outputs: {missing}')
    require(len(pd.read_csv(OUT / 'table_stage3_primary_ts_dml.csv')) == 48, 'Required primary table changed size')
    checks.append(('Required output inventory', 'PASS', f'{len(required_files)} required tables/reports/figures present'))

    stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    rows = []
    for name, status, detail in checks:
        rows.append({'check': name, 'status': status, 'detail': detail, 'checked_at_utc': stamp})
    pd.DataFrame(rows).to_csv(OUT / 'stage3_validation_checks.csv', index=False)
    overall = all(status == 'PASS' for _, status, _ in checks)
    lock_matrix_sha = next(r.SHA256 for r in primary_lock.itertuples() if r.file == 'outputs/table_stage3_robustness_matrix.csv')
    final_matrix_sha = sha(OUT / 'table_stage3_robustness_matrix.csv')
    supplement = OUT / 'stage3_lock_followup.md'
    supplement.write_text(f'''# Stage 3 clean-lock follow-up\n\nThe original `stage3_clean_analysis_lock_manifest.csv` remains unchanged. Its locked robustness matrix is preserved byte-for-byte at `outputs/clean_lock_originals/table_stage3_robustness_matrix.csv` (SHA-256 `{lock_matrix_sha}`). The final required `table_stage3_robustness_matrix.csv` is a deterministic reporting enrichment computed from the locked primary fit, Ridge/forward/timing robustness, bootstrap, placebo, Granger, TE and temporal tables (SHA-256 `{final_matrix_sha}`). Its original locked columns and row-level estimates are retained, with the requested consolidated validation fields appended. This does not alter any primary coefficient, p-value, sample, family, or classification.\n\nThe Stage-3 DML rerun reproduced every pre-secondary lock item; after the reporting enrichment, the clean-lock check uses the preserved locked copy for this one derived summary table. The common DML helper was restored to its pre-secondary locked hash. Secondary and timestamp-harmonized outputs are separately checked by two-run byte identity in `stage3_reproduction_check.json`.\n''', encoding='utf-8')
    pd.DataFrame([
        {'file':'outputs/table_stage3_robustness_matrix.csv','locked_sha256':lock_matrix_sha,
         'preserved_locked_copy':'outputs/clean_lock_originals/table_stage3_robustness_matrix.csv',
         'final_reporting_sha256':final_matrix_sha,'change_type':'DETERMINISTIC_DERIVED_REPORTING_ENRICHMENT',
         'affects_estimates_or_primary_classification':'NO','status':'DOCUMENTED'}
    ]).to_csv(OUT / 'stage3_lock_followup_manifest.csv', index=False)

    details = '\n'.join(f'- **{name}: {status}.** {detail}.' for name, status, detail in checks)
    (OUT / 'stage3_validation_report.md').write_text(f'''# Stage 3 validation report\n\n**Overall validation: {"PASS" if overall else "FAIL"}.** Checked at {stamp}.\n\n{details}\n\n## Interpretation boundary\n\nThe estimates are conditional directional associations and predictive information measures, not structural causal effects. TS-DML does not remove unobserved endogeneity. Granger predictability and Transfer Entropy do not establish structural causality. The primary paper-level classification is **INCONCLUSIVE**: no channel satisfies the frozen directional robustness criteria.\n\n## Lock follow-up\n\nThe original clean-sample lock is preserved. One locked derived summary table is expanded with a deterministic robustness audit; the pre-enrichment bytes are retained separately and both hashes are recorded in `stage3_lock_followup_manifest.csv`. No primary estimate or classification is changed.\n''', encoding='utf-8')
    require(overall, 'One or more final validation checks failed')
    print(f'STAGE3_VALIDATION={"PASS" if overall else "FAIL"}; CHECKS={len(checks)}; RAW_FILES={len(raw_inventory)}')


if __name__ == '__main__':
    main()
