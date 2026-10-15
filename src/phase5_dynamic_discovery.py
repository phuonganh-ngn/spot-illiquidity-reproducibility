#!/usr/bin/env python3
"""Phase 5: build/validate fixed 4-hour inputs and audit discovery-method readiness.

No network access. No Stage 1-3 files are written. The package audit is explicit:
the script does not relabel substitute estimators as PCMCI+/NOTEARS/LiNGAM.
"""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, importlib.util, json, math, platform, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import scipy, sklearn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'phase5_dynamic_discovery'
INTERIM = ROOT / 'data' / 'interim'
START = pd.Timestamp('2021-12-01 00:00:00', tz='UTC')
END_EXCLUSIVE = pd.Timestamp('2024-03-04 00:00:00', tz='UTC')
FREQ = '4h'
ASSETS = ('BTC', 'ETH')
LAGS = tuple(range(1, 7))
DISCOVERY_END_EXCLUSIVE = pd.Timestamp('2023-07-01 00:00:00', tz='UTC')
VARIABLES = [
    'funding_rate_state', 'oi_log_change', 'futures_taker_imbalance',
    'spot_return', 'spot_realized_volatility', 'spot_downside_semivariance',
    'spot_amihud', 'spot_taker_imbalance', 'spot_quote_volume_growth',
    'basis_mean',
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def json_bytes(obj) -> bytes:
    return (json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False,
                        separators=(',', ':')) + '\n').encode('utf-8')


def package_info(module_name: str, dist_name: str | None = None):
    present = importlib.util.find_spec(module_name) is not None
    version = None
    if present:
        try:
            version = importlib.metadata.version(dist_name or module_name)
        except importlib.metadata.PackageNotFoundError:
            version = 'PRESENT_VERSION_UNRESOLVED'
    return {'module': module_name, 'present': present, 'version': version}


def method_environment():
    packages = {
        'tigramite': package_info('tigramite'),
        'dynotears': package_info('dynotears'),
        'causalnex': package_info('causalnex'),
        'castle': package_info('castle'),
        'notears': package_info('notears'),
        'torch': package_info('torch'),
        'lingam': package_info('lingam'),
        'statsmodels': package_info('statsmodels'),
        'sklearn': package_info('sklearn', 'scikit-learn'),
    }
    return packages


def write_protocol(packages):
    # These choices are frozen before any discovery fit; no estimator is run in
    # this script when its validated implementation is not available.
    protocol = {
        'phase': 'PHASE_5_DYNAMIC_STRUCTURE_DISCOVERY',
        'status': 'FROZEN_BEFORE_METHOD_FITS',
        'created_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'primary_frequency': '4-hour UTC, intervals [period_start, period_end)',
        'sample_period_start': START.isoformat(),
        'sample_period_end_inclusive': '2024-03-03',
        'sample_period_end_exclusive': END_EXCLUSIVE.isoformat(),
        'discovery_period': {'start': START.isoformat(), 'end_exclusive': DISCOVERY_END_EXCLUSIVE.isoformat()},
        'confirmation_period': {'start': DISCOVERY_END_EXCLUSIVE.isoformat(), 'end_inclusive': '2024-03-03'},
        'assets_separate': list(ASSETS),
        'variables': VARIABLES,
        'optional_features_included': ['basis_mean', 'spot_quote_volume_growth'],
        'lag_set': list(LAGS),
        'edge_scope': 'lagged past-to-current only; no contemporaneous edge is confirmatory',
        'discovery_methods': ['PCMCI+', 'DYNOTEARS', 'NOTEARS-MLP', 'Neural Granger/cMLP', 'VAR-LiNGAM'],
        'pcmci': {'tau_min': 1, 'tau_max': 6, 'primary_ci_test': 'linear partial correlation', 'alpha_levels': [0.01, 0.05, 0.10], 'primary_adjusted_alpha': 0.05},
        'dynotears': {'lags': [1, 2, 3, 4, 5, 6], 'lambda_W_grid': [0.01, 0.05, 0.10, 0.20]},
        'notears_mlp': {'hidden_layers': 1, 'hidden_sizes': [10, 20], 'lagged_predictors_only': True, 'seed': 314159},
        'neural_granger': {'maximum_lag': 6, 'small_network': True, 'structured_group_penalty': True, 'seed': 314159},
        'var_lingam': {'bic_lag_search': [1, 2, 3, 4, 5, 6], 'diagnostic_gate_required': True},
        'bootstrap': {'primary_replications': 200, 'minimum_if_computationally_prohibitive': 100, 'block_lengths_observations': [42, 84], 'unit': 'contiguous four-hour rows'},
        'stable_edge_rules': {'stable': 0.70, 'very_stable': 0.85, 'weak_consensus': 2, 'moderate_consensus': 3, 'strong_consensus': 4},
        'holdout_rules': {'candidate_entry': 'moderate consensus OR one-method very-stable plus cross-asset replication', 'holm_scope': 'separately by asset'},
        'funding': {'no_jitter': True, 'representation': 'most recently observed settled funding rate at/before interval end; max allowed age 12 hours; no artificial noise'},
        'missing_data': 'No interpolation or synthetic values. Discovery/holdout estimation, if available, uses complete rows for the frozen variable set and required lag history.',
        'daily_sensitivity': 'NOT_PLANNED: optional only after a complete frozen 4-hour method analysis; no 4-hour method has a validated installed implementation.',
        'method_environment_at_freeze': packages,
        'network_data_downloads': 0,
    }
    raw = json_bytes(protocol)
    (OUT / 'phase5_protocol.json').write_bytes(raw)
    return hashlib.sha256(raw).hexdigest(), protocol


def dedupe_by_time(df: pd.DataFrame, times: pd.Series, value_cols: list[str]):
    z = df.copy()
    z['_ts'] = pd.DatetimeIndex(times)
    z = z.loc[z['_ts'].notna()].sort_values('_ts')
    duplicate_rows = int(z['_ts'].duplicated(keep=False).sum())
    conflicting = 0
    if duplicate_rows:
        for ts, g in z[z['_ts'].duplicated(keep=False)].groupby('_ts', sort=False):
            if len(g[value_cols].drop_duplicates()) > 1:
                conflicting += len(g)
        # Do not choose among conflicting duplicate values. Remove the entire
        # timestamp group so affected bars become missing in the expected grid.
        bad_ts = set(z[z['_ts'].duplicated(keep=False)].groupby('_ts').filter(
            lambda g: len(g[value_cols].drop_duplicates()) > 1)['_ts'])
        z = z[~z['_ts'].isin(bad_ts)].drop_duplicates('_ts', keep='first')
        z = z.set_index('_ts').sort_index()
    else:
        z = z.set_index('_ts').sort_index()
    return z, {'duplicate_rows': duplicate_rows, 'conflicting_duplicate_rows_removed': int(conflicting)}


def load_kline(asset: str, market: str, kind: str):
    path = INTERIM / f'{asset}USDT_{market}_{kind}_5m.pkl'
    d = pd.read_pickle(path)
    ts = pd.to_datetime(pd.to_numeric(d.open_time, errors='coerce'), unit='ms', utc=True, errors='coerce')
    cols = ['open', 'high', 'low', 'close', 'base_volume', 'quote_volume',
            'number_of_trades', 'taker_buy_base_volume']
    z, dup = dedupe_by_time(d, ts, cols)
    return z[cols], {'path': str(path.relative_to(ROOT)), 'sha256': sha(path),
                     'raw_rows': int(len(d)), 'first_timestamp': str(z.index.min()),
                     'last_timestamp': str(z.index.max()), **dup}


def valid_kline(z: pd.DataFrame):
    arr = z.to_numpy(dtype=float)
    finite = np.isfinite(arr).all(axis=1)
    positive_ohlc = (z[['open', 'high', 'low', 'close']] > 0).all(axis=1).to_numpy()
    nonnegative = (z[['base_volume', 'quote_volume', 'number_of_trades', 'taker_buy_base_volume']] >= 0).all(axis=1).to_numpy()
    buy_valid = (z.taker_buy_base_volume <= z.base_volume + 1e-7).to_numpy()
    return finite & positive_ohlc & nonnegative & buy_valid


def aggregate_kline_4h(z: pd.DataFrame, asset: str, market: str):
    grid5 = pd.date_range(START - pd.Timedelta(hours=4), END_EXCLUSIVE,
                          freq='5min', inclusive='left')
    z = z.reindex(grid5)
    good = valid_kline(z)
    clean = z.copy()
    clean.loc[~good, :] = np.nan
    interval = clean.index.floor('4h')
    counts = pd.Series(good.astype(int), index=clean.index).groupby(interval).sum()
    g = clean.groupby(interval)
    agg = pd.DataFrame(index=counts.index)
    agg['open'] = g.open.first(); agg['high'] = g.high.max(); agg['low'] = g.low.min(); agg['close'] = g.close.last()
    for col in ['base_volume','quote_volume','number_of_trades','taker_buy_base_volume']:
        agg[col] = g[col].sum(min_count=48)
    agg['valid_5m_bars'] = counts
    complete = counts.eq(48)
    agg.loc[~complete, ['open','high','low','close','base_volume','quote_volume','number_of_trades','taker_buy_base_volume']] = np.nan
    # Five-minute close-to-close log returns use the preceding 5-minute close;
    # the 4-hour RV is valid only with all 48 interval returns.
    r5 = np.log(clean.close / clean.close.shift(1))
    rg = r5.groupby(interval)
    rcount = rg.count()
    rv = np.sqrt(rg.apply(lambda x: np.square(x).sum(min_count=48)))
    down = rg.apply(lambda x: np.square(x.clip(upper=0)).sum(min_count=48))
    agg['rv_5m_return_count'] = rcount
    agg['spot_realized_volatility' if market == 'spot' else 'futures_realized_volatility'] = rv.where(rcount.eq(48) & complete)
    agg['downside_semivariance' if market == 'spot' else 'futures_downside_semivariance'] = down.where(rcount.eq(48) & complete)
    agg['imbalance'] = ((2 * agg.taker_buy_base_volume - agg.base_volume) /
                        agg.base_volume.where(agg.base_volume > 0)).where(complete)
    agg['quote_volume_growth'] = np.log(agg.quote_volume.where(agg.quote_volume > 0)).diff()
    if market == 'spot':
        agg['spot_return'] = np.log(agg.close / agg.close.shift(1))
        agg['spot_amihud'] = 1e6 * agg.spot_return.abs() / agg.quote_volume.where(agg.quote_volume > 0)
        agg['spot_taker_imbalance'] = agg.imbalance
        agg['spot_quote_volume_growth'] = agg.quote_volume_growth
    else:
        agg['futures_taker_imbalance'] = agg.imbalance
    agg = agg.replace([np.inf, -np.inf], np.nan)
    return agg, {'market': market, 'asset': asset, 'raw': None, 'expected_5m_rows': int(len(grid5)),
                 'valid_5m_rows': int(good.sum()), 'missing_or_invalid_5m_rows': int((~good).sum()),
                 'duplicate_rows': None}


def load_metric_oi(asset: str):
    path = INTERIM / f'{asset}USDT_futures_metrics_5m.pkl'
    d = pd.read_pickle(path)
    if 'create_time' not in d or 'sum_open_interest' not in d:
        raise RuntimeError(f'Unexpected OI schema: {path.name}')
    ts = pd.to_datetime(d.create_time, utc=True, errors='coerce')
    # Apply the Stage-1 empirically verified per-date effective time mapping.
    layout_path = OUT.parent / 'metrics_daily_layout_audit.csv'
    layout = pd.read_csv(layout_path)
    layout = layout[layout.symbol == f'{asset}USDT'].set_index('date')
    dates = pd.DatetimeIndex(ts).strftime('%Y-%m-%d')
    offset = pd.Series(dates).map(layout.stock_effective_offset_seconds).to_numpy()
    verified = np.isfinite(pd.to_numeric(pd.Series(offset), errors='coerce').to_numpy())
    ts = pd.DatetimeIndex(ts) + pd.to_timedelta(np.where(verified, offset, 0), unit='s')
    # Unknown mappings are excluded, not silently assigned a timestamp.
    value_cols = ['sum_open_interest', 'sum_open_interest_value']
    z, dup = dedupe_by_time(d, pd.Series(ts).where(verified), value_cols)
    z = z[value_cols]
    return z, {'path': str(path.relative_to(ROOT)), 'sha256': sha(path),
               'raw_rows': int(len(d)), 'mapped_rows': int(verified.sum()),
               'unknown_layout_rows_excluded': int((~verified).sum()), **dup}


def aggregate_oi_4h(z: pd.DataFrame):
    grid5 = pd.date_range(START - pd.Timedelta(hours=4), END_EXCLUSIVE,
                          freq='5min', inclusive='left')
    z = z.reindex(grid5)
    oi = pd.to_numeric(z.sum_open_interest, errors='coerce')
    value = pd.to_numeric(z.sum_open_interest_value, errors='coerce')
    good = np.isfinite(oi) & np.isfinite(value) & (oi > 0) & (value > 0)
    group = z.index.floor('4h')
    oi_close = oi.where(good).groupby(group).last()
    count = pd.Series(good.astype(int), index=z.index).groupby(group).sum()
    oi_close = oi_close.where(count.eq(48))
    result = pd.DataFrame({'oi_close': oi_close, 'oi_valid_5m_snapshots': count})
    result['oi_log_change'] = np.log(result.oi_close / result.oi_close.shift(1))
    return result, {'expected_5m_snapshots': int(len(grid5)), 'valid_5m_snapshots': int(good.sum()),
                    'missing_or_invalid_5m_snapshots': int((~good).sum())}


def aggregate_basis_4h(asset: str):
    p_mark = INTERIM / f'{asset}USDT_futures_markPriceKlines_5m.pkl'
    p_index = INTERIM / f'{asset}USDT_futures_indexPriceKlines_5m.pkl'
    dm = pd.read_pickle(p_mark); di = pd.read_pickle(p_index)
    tm = pd.to_datetime(pd.to_numeric(dm.open_time, errors='coerce'), unit='ms', utc=True, errors='coerce')
    ti = pd.to_datetime(pd.to_numeric(di.open_time, errors='coerce'), unit='ms', utc=True, errors='coerce')
    mark, md = dedupe_by_time(dm, tm, ['close'])
    index, ix = dedupe_by_time(di, ti, ['close'])
    grid5 = pd.date_range(START - pd.Timedelta(hours=4), END_EXCLUSIVE,
                          freq='5min', inclusive='left')
    m = pd.to_numeric(mark.close.reindex(grid5), errors='coerce')
    q = pd.to_numeric(index.close.reindex(grid5), errors='coerce')
    b = (m / q.where(q > 0) - 1).replace([np.inf, -np.inf], np.nan)
    group = b.index.floor('4h')
    count = b.groupby(group).count()
    mean = b.groupby(group).mean().where(count.eq(48))
    return pd.DataFrame({'basis_mean': mean, 'basis_valid_5m_bars': count}), {
        'mark_path': str(p_mark.relative_to(ROOT)), 'mark_sha256': sha(p_mark),
        'index_path': str(p_index.relative_to(ROOT)), 'index_sha256': sha(p_index),
        'valid_basis_5m_rows': int(b.notna().sum()), 'missing_basis_5m_rows': int(b.isna().sum()),
        'mark_duplicates': md, 'index_duplicates': ix,
    }


def funding_state_4h(asset: str, period_ends: pd.DatetimeIndex):
    path = INTERIM / f'{asset}USDT_futures_fundingRate_FUNDING_SETTLEMENT.pkl'
    d = pd.read_pickle(path)
    ts = pd.to_datetime(pd.to_numeric(d.calc_time, errors='coerce'), unit='ms', utc=True, errors='coerce')
    z, dup = dedupe_by_time(d, ts, ['last_funding_rate'])
    rate = pd.to_numeric(z.last_funding_rate, errors='coerce')
    event_ns = z.index.asi8
    end_ns = period_ends.asi8
    pos = np.searchsorted(event_ns, end_ns, side='right') - 1
    out = np.full(len(period_ends), np.nan, dtype=float)
    age = np.full(len(period_ends), np.nan, dtype=float)
    event_time = np.full(len(period_ends), np.datetime64('NaT'), dtype='datetime64[ns]')
    ok = pos >= 0
    ix = pos[ok]
    out[ok] = rate.to_numpy()[ix]
    event_time[ok] = z.index.to_numpy(dtype='datetime64[ns]')[ix]
    age[ok] = (end_ns[ok] - event_ns[ix]) / 3.6e12
    # A rate older than 12 hours is a true gap, not carried forward indefinitely.
    out[(age > 12) | (age < 0)] = np.nan
    return pd.DataFrame({'funding_rate_state': out,
                         'funding_age_hours': age,
                         'funding_last_event_utc': pd.to_datetime(event_time, utc=True)}, index=period_ends), {
        'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'raw_rows': int(len(d)),
        'frequency_hours_unique': sorted(pd.to_numeric(d.funding_interval_hours, errors='coerce').dropna().unique().astype(int).tolist()),
        'duplicate_rows': dup['duplicate_rows'], 'conflicting_duplicate_rows_removed': dup['conflicting_duplicate_rows_removed'],
        'rule': 'last settlement timestamp <= period end, only if age <= 12 hours; no jitter'}


def build_asset(asset: str):
    diagnostics = {'asset': asset, 'inputs': {}}
    spot, sdiag = load_kline(asset, 'spot', 'klines')
    fut, fdiag = load_kline(asset, 'futures', 'klines')
    spot4, _ = aggregate_kline_4h(spot, asset, 'spot')
    fut4, _ = aggregate_kline_4h(fut, asset, 'futures')
    diagnostics['inputs']['spot_klines'] = sdiag
    diagnostics['inputs']['futures_klines'] = fdiag

    oi, odiag = load_metric_oi(asset)
    oi4, oi_quality = aggregate_oi_4h(oi)
    diagnostics['inputs']['futures_metrics_oi'] = {**odiag, **oi_quality}

    basis4, bdiag = aggregate_basis_4h(asset)
    diagnostics['inputs']['basis'] = bdiag

    starts = pd.date_range(START, END_EXCLUSIVE, freq='4h', inclusive='left')
    ends = starts + pd.Timedelta(hours=4)
    p4 = pd.DataFrame({'period_start_utc': starts, 'period_end_utc': ends}).set_index('period_start_utc')
    p4 = p4.join(spot4.add_prefix('spot_raw_'), how='left')
    p4 = p4.join(fut4.add_prefix('futures_raw_'), how='left')
    p4 = p4.join(oi4, how='left')
    p4 = p4.join(basis4, how='left')
    fund, fndiag = funding_state_4h(asset, ends)
    fund.index = starts
    p4 = p4.join(fund, how='left')
    diagnostics['inputs']['funding'] = fndiag

    rename = {
        'spot_raw_spot_return':'spot_return',
        'spot_raw_spot_realized_volatility':'spot_realized_volatility',
        'spot_raw_downside_semivariance':'spot_downside_semivariance',
        'spot_raw_spot_amihud':'spot_amihud',
        'spot_raw_spot_taker_imbalance':'spot_taker_imbalance',
        'spot_raw_spot_quote_volume_growth':'spot_quote_volume_growth',
        'futures_raw_futures_taker_imbalance':'futures_taker_imbalance',
    }
    p4 = p4.rename(columns=rename)
    p4['asset'] = asset
    p4.index.name = 'period_start_utc'
    p4 = p4.reset_index()
    p4 = p4[(p4.period_start_utc >= START) & (p4.period_start_utc < END_EXCLUSIVE)].copy()
    p4['discovery_sample'] = p4.period_start_utc < DISCOVERY_END_EXCLUSIVE
    p4['sample_split'] = np.where(p4.discovery_sample, 'DISCOVERY', 'CONFIRMATION')
    p4['expected_5m_bars'] = 48
    p4['spot_valid_5m_bars'] = p4.spot_raw_valid_5m_bars
    p4['futures_valid_5m_bars'] = p4.futures_raw_valid_5m_bars
    # Only the frozen analysis features enter graph discovery. Raw aggregation
    # diagnostics remain in the JSONL for transparent validation.
    for col in VARIABLES:
        if col not in p4:
            p4[col] = np.nan
    p4[VARIABLES] = p4[VARIABLES].replace([np.inf, -np.inf], np.nan)
    diagnostics['feature_coverage'] = {}
    for col in VARIABLES:
        s = pd.to_numeric(p4[col], errors='coerce')
        valid = np.isfinite(s.to_numpy(dtype=float))
        q = p4.loc[valid, 'period_start_utc']
        diagnostics['feature_coverage'][col] = {
            'expected_bars': int(len(p4)), 'valid_bars': int(valid.sum()),
            'missing_or_invalid_bars': int(len(p4)-valid.sum()),
            'coverage_pct': float(100*valid.mean()) if len(valid) else 0.0,
            'first_valid_period_start': q.min().isoformat() if len(q) else None,
            'last_valid_period_start': q.max().isoformat() if len(q) else None,
            'zero_count': int((s[valid] == 0).sum()),
            'negative_count': int((s[valid] < 0).sum()),
        }
    cc = np.isfinite(p4[VARIABLES].to_numpy(dtype=float)).all(axis=1)
    diagnostics['complete_cases_all_frozen_variables'] = int(cc.sum())
    diagnostics['complete_case_coverage_pct'] = float(100*cc.mean())
    diagnostics['discovery_complete_cases'] = int((cc & p4.discovery_sample.to_numpy()).sum())
    diagnostics['confirmation_complete_cases'] = int((cc & ~p4.discovery_sample.to_numpy()).sum())
    diagnostics['periods'] = {
        'expected_total': int(len(p4)),
        'first_period_start': p4.period_start_utc.min().isoformat(),
        'last_period_start': p4.period_start_utc.max().isoformat(),
        'first_period_end': p4.period_end_utc.min().isoformat(),
        'last_period_end': p4.period_end_utc.max().isoformat(),
        'discovery_expected': int(p4.discovery_sample.sum()),
        'confirmation_expected': int((~p4.discovery_sample).sum()),
    }
    return p4, diagnostics


def json_safe(value):
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        if pd.isna(value): return None
        return pd.Timestamp(value).tz_localize('UTC') .isoformat() if pd.Timestamp(value).tzinfo is None else pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)): return bool(value)
    if pd.isna(value): return None
    return value


def dataframe_jsonl_bytes(df: pd.DataFrame):
    lines=[]
    for row in df.to_dict(orient='records'):
        item={k:json_safe(v) for k,v in row.items()}
        lines.append(json.dumps(item,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')))
    return ('\n'.join(lines)+'\n').encode('utf-8')


def verify_only():
    """Rebuild cached inputs in memory and compare to the saved Phase-5 panel."""
    dataset_path = OUT / 'dataset_4h.jsonl'
    protocol_path = OUT / 'phase5_protocol.json'
    if not dataset_path.is_file() or not protocol_path.is_file():
        raise RuntimeError('Cannot verify: saved Phase-5 dataset/protocol is missing')
    expected = dataset_path.read_bytes()
    frames=[]; audits=[]
    for asset in ASSETS:
        frame, audit = build_asset(asset)
        frames.append(frame); audits.append(audit)
    panel = pd.concat(frames,ignore_index=True).sort_values(['asset','period_start_utc']).reset_index(drop=True)
    actual = dataframe_jsonl_bytes(panel)
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    listed = []
    for audit in audits:
        for item in audit['inputs'].values():
            if not isinstance(item, dict):
                continue
            for pkey, hkey in [('path','sha256'),('mark_path','mark_sha256'),('index_path','index_sha256')]:
                if pkey in item:
                    listed.append({'path':item[pkey], 'sha256':item[hkey]})
    source_ok = all(sha(ROOT / item['path']) == item['sha256'] for item in listed)
    passed = actual == expected and source_ok and len(panel) == 9888
    result = {
        'status':'PASS' if passed else 'FAIL',
        'mode':'OFFLINE_VERIFY_ONLY',
        'dataset_sha256_saved':hashlib.sha256(expected).hexdigest(),
        'dataset_sha256_rebuilt':hashlib.sha256(actual).hexdigest(),
        'byte_identical':actual == expected,
        'rows':int(len(panel)), 'rows_by_asset':{k:int(v) for k,v in panel.groupby('asset').size().items()},
        'cached_source_hashes_unchanged':bool(source_ok),
        'source_count':len(listed),
        'protocol_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        'network_data_downloads':0, 'model_fits':0,
    }
    (OUT/'offline_reproduction.json').write_bytes(json_bytes(result))
    print(f"OFFLINE_VERIFY_ONLY={result['status']} byte_identical={result['byte_identical']} source_hashes={source_ok} rows={len(panel)}")
    if not passed:
        raise RuntimeError('Offline verification did not reproduce the saved panel exactly')


def output_status(method, package, reason):
    return {
        'method': method, 'status': 'FAIL', 'fit_attempted': False,
        'reason_code': 'VALIDATED_IMPLEMENTATION_NOT_INSTALLED',
        'reason': reason, 'required_or_checked_package': package,
        'funding_status': 'NOT_ASSESSED_METHOD_UNAVAILABLE',
        'assets': {asset: {'fit_status':'NOT_RUN','candidate_edges':[]} for asset in ASSETS},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-only', action='store_true', help='Rebuild cached inputs in memory and compare without overwriting existing artifacts')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.verify_only:
        verify_only()
        return
    log=[]
    def note(text):
        log.append(f'{datetime.now(timezone.utc).isoformat(timespec="seconds")} {text}')
        print(text, flush=True)

    if any(OUT.iterdir()):
        raise RuntimeError(f'Phase 5 output namespace is not empty; refusing to overwrite: {OUT}')

    package = method_environment()
    protocol_hash, protocol = write_protocol(package)
    note(f'PROTOCOL_FROZEN sha256={protocol_hash}')
    note('4H_BUILD_START cached_interim_only network_downloads=0')

    assets=[]; audits=[]
    for asset in ASSETS:
        frame, audit = build_asset(asset)
        assets.append(frame); audits.append(audit)
        note(f'4H_ASSET_VALIDATED asset={asset} rows={len(frame)} complete={audit["complete_cases_all_frozen_variables"]}')
    panel = pd.concat(assets, ignore_index=True).sort_values(['asset','period_start_utc']).reset_index(drop=True)
    expected_n = len(pd.date_range(START, END_EXCLUSIVE, freq='4h', inclusive='left'))
    if expected_n != 4944 or len(panel) != expected_n*2:
        raise RuntimeError(f'Unexpected 4h calendar: expected 4,944 periods per asset, got {len(panel)} rows')
    if panel.duplicated(['asset','period_start_utc']).any():
        raise RuntimeError('Duplicate asset × four-hour interval key')
    if set(panel.asset.unique()) != set(ASSETS):
        raise RuntimeError('Asset universe changed')

    # Validate no post-sample bar entered and source periods end no later than
    # 2024-03-04 00:00 UTC (the archive-layout boundary).
    if not (panel.period_start_utc.min() == START and panel.period_start_utc.max() == END_EXCLUSIVE-pd.Timedelta(hours=4)):
        raise RuntimeError('Unexpected primary 4-hour period bounds')
    if (panel.period_end_utc > END_EXCLUSIVE).any():
        raise RuntimeError('Post-window information found in 4-hour sample')
    if panel.groupby('asset').size().to_dict() != {'BTC':4944,'ETH':4944}:
        raise RuntimeError('Asset-specific period counts differ')
    note('4H_CALENDAR_VALID expected=4944_per_asset unique_keys=PASS end_boundary=2024-03-04T00:00Z')

    # The methods are only marked usable when their named, validated
    # implementation is installed. Do not substitute another algorithm under
    # one of these labels.
    methods = {
        'pcmci_results.json': output_status('PCMCI+', package['tigramite'], 'Tigramite is not installed; no PCMCI+ implementation was substituted.'),
        'dynotears_results.json': output_status('DYNOTEARS / temporal NOTEARS', {'dynotears':package['dynotears'],'causalnex':package['causalnex'],'castle':package['castle']}, 'No validated DYNOTEARS/temporal-NOTEARS implementation is installed; no sparse VAR was relabeled as DYNOTEARS.'),
        'notears_mlp_results.json': output_status('NOTEARS-MLP', {'torch':package['torch'],'notears':package['notears']}, 'PyTorch and a validated NOTEARS-MLP implementation are unavailable; no neural fit was attempted.'),
        'neural_granger_results.json': output_status('Neural Granger / cMLP', {'torch':package['torch']}, 'PyTorch and a validated group-sparse Neural Granger implementation are unavailable; sklearn MLP was not substituted.'),
        'var_lingam_results.json': output_status('VAR-LiNGAM', package['lingam'], 'The lingam package is not installed; no ordinary VAR/Granger output was relabeled as VAR-LiNGAM.'),
    }

    # Save a row-wise, machine-readable 4h panel in the new namespace only.
    ds_bytes = dataframe_jsonl_bytes(panel)
    (OUT/'dataset_4h.jsonl').write_bytes(ds_bytes)
    coverage = {
        'status': 'PASS' if len(panel)==9888 and not panel.duplicated(['asset','period_start_utc']).any() else 'FAIL',
        'source_frequency': '5-minute Binance archives from existing data/interim caches',
        'frequency': FREQ, 'timezone': 'UTC', 'assets': list(ASSETS),
        'expected_intervals_per_asset': expected_n,
        'sample': {'start':START.isoformat(),'last_period_start':'2024-03-03T20:00:00+00:00','last_period_end':END_EXCLUSIVE.isoformat()},
        'discovery_sample': {'start':START.isoformat(),'end_inclusive':'2023-06-30T20:00:00+00:00'},
        'confirmation_sample': {'start':DISCOVERY_END_EXCLUSIVE.isoformat(),'end_inclusive':'2024-03-03T20:00:00+00:00'},
        'asset_diagnostics': audits,
        'source_file_hashes': [],
        'limitations': [
            'No certified prebuilt 4-hour panel existed; panel was deterministically aggregated from cached certified 5-minute files.',
            'Funding is a last-known settlement state as of interval end, with a 12-hour maximum age; values repeat by design and are never jittered.',
            'OI is retained only when all 48 expected five-minute snapshots in the 4-hour interval are valid; OI log change requires adjacent valid intervals.',
            'No interpolation or synthetic observations were created; graph-fit complete cases are separately reported.',
            'Spot data are Binance BTCUSDT/ETHUSDT spot, not the global spot market.',
        ],
    }
    src_files=[]
    for audit in audits:
        for key, val in audit['inputs'].items():
            if not isinstance(val, dict): continue
            for pkey in ['path','mark_path','index_path']:
                if pkey in val:
                    rel=val[pkey]; src_files.append({'path':rel,'sha256':val.get('sha256') if pkey=='path' else val.get('mark_sha256' if pkey=='mark_path' else 'index_sha256')})
    coverage['source_file_hashes'] = sorted(src_files, key=lambda x:x['path'])
    (OUT/'dataset_validation.json').write_bytes(json_bytes(coverage))

    # Create the five required method JSONs, all explicitly unavailable.
    for filename, result in methods.items():
        result.update({'primary_frequency':'4h','discovery_sample_only':True,'lag_set':list(LAGS),
                       'protocol_sha256':protocol_hash,'dataset_sha256':hashlib.sha256(ds_bytes).hexdigest()})
        (OUT/filename).write_bytes(json_bytes(result))

    unavailable = [x['method'] for x in methods.values()]
    (OUT/'method_diagnostics.json').write_bytes(json_bytes({
        'status':'NO_VALIDATED_DISCOVERY_IMPLEMENTATION_AVAILABLE',
        'environment':package,
        'methods':{m['method']:{'status':m['status'],'fit_attempted':m['fit_attempted'],'reason_code':m['reason_code'],'reason':m['reason']} for m in methods.values()},
        'funding_notes':{
            'representation':'funding_rate_state; as-of carried state of observed settlement values, no jitter',
            'pcmci_plus':'NOT_ASSESSED_METHOD_UNAVAILABLE',
            'dynotears':'NOT_ASSESSED_METHOD_UNAVAILABLE',
            'notears_mlp':'NOT_ASSESSED_METHOD_UNAVAILABLE',
            'neural_granger':'NOT_ASSESSED_METHOD_UNAVAILABLE',
            'var_lingam':'NOT_ASSESSED_METHOD_UNAVAILABLE',
            'prior_daily_te_tie_gate':'unchanged; Phase 5 did not rerun or modify daily TE',
        },
        'stationarity_audit':'NOT_RUN: no VAR-LiNGAM implementation was available, so its pre-fit diagnostics were not invoked; no old Granger model was rerun.',
        'data_validation_status':coverage['status'],
    }))

    (OUT/'bootstrap_stability.json').write_bytes(json_bytes({
        'status':'NOT_RUN_NO_VALID_DISCOVERY_FITS', 'method_status':'All five named discovery methods unavailable in the active environment.',
        'primary_replications':200,'minimum_if_computationally_prohibitive':100,
        'block_lengths_observations':[42,84], 'bootstrap_unit':'contiguous four-hour blocks',
        'results':[], 'thresholds':{'stable':0.70,'very_stable':0.85},
    }))
    (OUT/'cross_method_consensus.json').write_bytes(json_bytes({
        'status':'INCONCLUSIVE_NO_VALID_METHOD_GRAPHS','valid_method_count':0,'edges':[],
        'support_thresholds':{'weak':2,'moderate':3,'strong':4},
        'note':'No consensus is inferred from absent method fits.'
    }))
    (OUT/'cross_asset_replication.json').write_bytes(json_bytes({
        'status':'NOT_ASSESSABLE_NO_FROZEN_ASSET_GRAPHS','replicated_edges':[],'asset_specific_candidates':[],
        'note':'BTC and ETH are kept separate; neither graph was estimable.'
    }))
    (OUT/'reverse_time_checks.json').write_bytes(json_bytes({
        'status':'NOT_RUN_NO_HIGH_CONSENSUS_EDGES','tests':[],
        'note':'Reverse-time checks require a frozen high-consensus edge; no eligible discovered edge exists.'
    }))
    (OUT/'holdout_confirmation.json').write_bytes(json_bytes({
        'status':'NOT_RUN_NO_ELIGIBLE_DISCOVERY_CANDIDATES','confirmation_start':DISCOVERY_END_EXCLUSIVE.isoformat(),
        'confirmation_end_inclusive':'2024-03-03','candidate_edges':[],'tests':[],
        'note':'No edge met the predeclared discovery entry rule. Holdout outcomes were not inspected for selection.'
    }))
    (OUT/'final_edges.json').write_bytes(json_bytes({
        'phase_classification':'INCONCLUSIVE','daily_stage3_classification_preserved':'NO_STABLE_INFORMATION_HIERARCHY_ESTABLISHED',
        'edges':[], 'pathways':[], 'cross_asset_replicated_edges':[],
        'note':'No structural or predictive edge claim is made because no validated discovery implementation was available.'
    }))

    # Rebuild once more in memory and require byte equality for offline data build.
    frames2=[]; audits2=[]
    for asset in ASSETS:
        f,a=build_asset(asset); frames2.append(f); audits2.append(a)
    panel2=pd.concat(frames2,ignore_index=True).sort_values(['asset','period_start_utc']).reset_index(drop=True)
    ds2=dataframe_jsonl_bytes(panel2)
    offline_pass=(ds_bytes==ds2 and audits==audits2)
    if not offline_pass:
        raise RuntimeError('Offline 4h rebuild did not reproduce byte-identically')

    # Verify prior-stage artifacts and all cached 5m inputs used above have not changed.
    prior_rel = [
        'data/processed/derivatives_spot_daily_panel.csv',
        'data/processed/derivatives_spot_extended_panel.csv',
        'data/interim/daily_features.pkl',
        'outputs/protocol_manifest.csv','outputs/protocol_manifest_v2.csv',
        'outputs/stage3_clean_analysis_lock_manifest.csv',
        'outputs/table_stage3_primary_ts_dml.csv','outputs/directional_evidence_matrix.csv',
        'outputs/stage3_research_summary.md','outputs/stage3_validation_report.md',
    ]
    prior_hashes={rel:sha(ROOT/rel) for rel in prior_rel}
    all_sources_unchanged=True
    for src in coverage['source_file_hashes']:
        if sha(ROOT/src['path']) != src['sha256']:
            all_sources_unchanged=False
    if not all_sources_unchanged:
        raise RuntimeError('A cached Stage-1 intraday source changed during Phase 5 build')

    checks={
        'status':'PASS' if coverage['status']=='PASS' and offline_pass and all_sources_unchanged else 'FAIL',
        'protocol_sha256':protocol_hash,
        'dataset_sha256':hashlib.sha256(ds_bytes).hexdigest(),
        'offline_rebuild_byte_identical':bool(offline_pass),
        'cached_intraday_sources_unchanged':bool(all_sources_unchanged),
        'previous_stage_artifacts_sha256':prior_hashes,
        'packages':package,
        'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
        'scipy':scipy.__version__,'scikit_learn':sklearn.__version__,
        'network_data_downloads':0,
        'fit_count':0,
        'note':'All named discovery methods were technically unavailable; no estimator or holdout fit was attempted.'
    }
    (OUT/'reproducibility.json').write_bytes(json_bytes(checks))

    summary = f'''# Phase 5 dynamic discovery summary

The cached five-minute Binance files support a mechanically constructed four-hour BTC/ETH panel. The panel contains 4,944 interval starts per asset, from {START.isoformat()} through 2024-03-03 20:00 UTC; each interval ends no later than 2024-03-04 00:00 UTC. Discovery is fixed through 2023-06-30 20:00 UTC and the untouched confirmation period begins 2023-07-01 00:00 UTC. The daily Stage-3 sample and its artifacts remain unchanged.

The panel was aggregated without downloading data or filling gaps. Funding is the last settled rate observable by each interval end, carried only while at most 12 hours old; no jitter was added. OI changes require adjacent complete four-hour intervals. Other intraday aggregates require all 48 five-minute rows. Coverage, gap counts, schema inputs, and hashes are in `dataset_validation.json`; row-level observations are in `dataset_4h.jsonl`.

The named discovery methods could not be executed in this environment: Tigramite/PCMCI+, validated DYNOTEARS, NOTEARS-MLP, structured Neural Granger, and VAR-LiNGAM implementations are not installed. The script did not substitute ordinary VAR/Granger or a generic MLP under those labels. Consequently method-specific bootstrap stability, cross-method consensus, cross-asset replication, reverse-time checks, and holdout confirmation are not estimable; their JSON outputs record these states rather than empty evidence of no edges.

The Phase-5 classification is **INCONCLUSIVE** because no validated discovery graph was fit. This does not overturn Stage 3’s daily conclusion, **NO STABLE INFORMATION HIERARCHY ESTABLISHED**. No edge, pathway, or nonlinear transmission claim is supported by this phase. Funding remains unassessed by the unavailable graph methods; the prior daily TE tie/atom gate was not changed.

Offline rebuilding of the four-hour panel reproduced byte-identically, and all cached intraday inputs and checked prior-stage artifacts retained their recorded hashes. No figures, presentation tables, new market data, daily sensitivity models, or discovery/holdout estimates were created.
'''
    (OUT/'research_summary.md').write_text(summary, encoding='utf-8')
    (OUT/'phase5_run.log').write_text('\n'.join(log)+'\n',encoding='utf-8')
    note('PHASE5_COMPLETE classification=INCONCLUSIVE fit_count=0 offline_rebuild=PASS')
    # Append final note after completion.
    (OUT/'phase5_run.log').write_text('\n'.join(log)+'\n',encoding='utf-8')
    print('PHASE5_CLASSIFICATION=INCONCLUSIVE')


if __name__ == '__main__':
    main()
