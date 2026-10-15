"""Frozen Stage-3 time-series DML primitives. No network access."""
from __future__ import annotations
import hashlib, math
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

START = pd.Timestamp('2021-12-01', tz='UTC')
END = pd.Timestamp('2024-03-03', tz='UTC')
ASSETS = ('BTC', 'ETH')
CONTROLS = [
    'spot_log_return_1d_lag1', 'spot_realized_volatility_lag1',
    'spot_amihud_lag1', 'spot_downside_semivariance_lag1',
    'spot_quote_volume_growth_1d_lag1', 'funding_mean_lag1',
    'oi_log_change_1d_lag1', 'futures_taker_imbalance_lag1',
    'spot_log_return_1d_lag2', 'spot_realized_volatility_lag7',
    'funding_mean_lag2',
]
BLOCKS = [
    (pd.Timestamp('2021-12-01', tz='UTC'), pd.Timestamp('2022-06-24', tz='UTC')),
    (pd.Timestamp('2022-06-25', tz='UTC'), pd.Timestamp('2023-01-16', tz='UTC')),
    (pd.Timestamp('2023-01-17', tz='UTC'), pd.Timestamp('2023-08-10', tz='UTC')),
    (pd.Timestamp('2023-08-11', tz='UTC'), pd.Timestamp('2024-03-03', tz='UTC')),
]

@dataclass
class FitResult:
    stats: dict
    scores: pd.DataFrame
    folds: list


def deterministic_seed(spec_id: str, base: int = 314159) -> int:
    digest = hashlib.sha256(spec_id.encode('utf-8')).digest()
    return (base + int.from_bytes(digest[:4], 'big')) % (2**32 - 1)


def make_model(name: str):
    if name == 'HGB':
        return HistGradientBoostingRegressor(
            loss='squared_error', learning_rate=0.05, max_iter=200,
            max_leaf_nodes=7, min_samples_leaf=25, l2_regularization=1.0,
            max_bins=255, early_stopping=False, random_state=314159,
        )
    if name == 'RIDGE':
        return Ridge(alpha=1.0, fit_intercept=True)
    raise ValueError(name)


def nuisance_predict(train_x, train_y, test_x, learner):
    train_x = np.asarray(train_x, dtype=float)
    train_y = np.asarray(train_y, dtype=float)
    test_x = np.asarray(test_x, dtype=float)
    ym = float(np.mean(train_y)); ys = float(np.std(train_y, ddof=0))
    if not np.isfinite(ys) or ys == 0: ys = 1.0
    y_scaled = (train_y - ym) / ys
    if learner == 'RIDGE':
        scaler = StandardScaler().fit(train_x)
        xtr = scaler.transform(train_x); xte = scaler.transform(test_x)
    else:
        xtr = train_x; xte = test_x
    model = make_model(learner)
    model.fit(xtr, y_scaled)
    return np.asarray(model.predict(xte), dtype=float) * ys + ym


def date_blocks(dates: pd.DatetimeIndex, window=None, predefined_blocks=None):
    if predefined_blocks is not None:
        return predefined_blocks
    if window is None:
        return BLOCKS
    selected = dates[(dates >= window[0]) & (dates <= window[1])]
    pieces = np.array_split(selected, 4)
    return [(pd.Timestamp(x[0]), pd.Timestamp(x[-1])) for x in pieces if len(x)]


def hac_from_scores(scores: pd.DataFrame, v: np.ndarray, u: np.ndarray,
                    theta: float, horizon: int, full_spine: pd.DatetimeIndex):
    n = len(scores)
    if n == 0: return (np.nan, np.nan, np.nan, np.nan)
    j = float(np.mean(v * v))
    if not np.isfinite(j) or j <= 0: return (np.nan, np.nan, np.nan, np.nan)
    q = v * (u - theta * v)
    loc = pd.Series(np.arange(len(full_spine)), index=full_spine)
    qspine = np.full(len(full_spine), np.nan, dtype=float)
    positions = loc.loc[pd.DatetimeIndex(scores['date'])].to_numpy(dtype=int)
    qspine[positions] = q
    lags = max(7, 2 * int(horizon) - 1)
    lrv = float(np.nansum(qspine * qspine) / n)
    for ell in range(1, lags + 1):
        left = qspine[ell:]; right = qspine[:-ell]
        ok = np.isfinite(left) & np.isfinite(right)
        gamma = float(np.sum(left[ok] * right[ok]) / n) if ok.any() else 0.0
        weight = 1.0 - ell / (lags + 1.0)
        lrv += 2.0 * weight * gamma
    variance = max(lrv, 0.0) / (n * j * j)
    se = math.sqrt(variance) if variance >= 0 else np.nan
    p = float(2 * norm.sf(abs(theta / se))) if np.isfinite(se) and se > 0 else np.nan
    return se, p, theta - 1.959963984540054 * se, theta + 1.959963984540054 * se


def fit_dml(df: pd.DataFrame, asset: str, treatment: str, outcome: str,
            horizon: int, learner='HGB', scheme='BLOCKED_4', control_shift=0,
            exposure_shift=0, embargo=None, window=None, require_primary=True,
            control_columns=None, fold_label='BASELINE', allowed_dates=None,
            sample_start=START, sample_end=END, predefined_blocks=None) -> FitResult:
    """Fit OOF partially-linear DML on calendar anchors; never compress dates."""
    control_columns = list(control_columns or CONTROLS)
    full = df[(df['asset'] == asset) & (df['date'] >= sample_start) & (df['date'] <= sample_end)].copy()
    full = full.sort_values('date').set_index('date')
    if window is not None:
        full = full.loc[(full.index >= window[0]) & (full.index <= window[1])]
    if full.index.has_duplicates:
        raise RuntimeError(f'duplicate date for {asset}')
    d = full[treatment].shift(exposure_shift) if exposure_shift else full[treatment]
    y = full[outcome]
    x = full[control_columns].shift(control_shift) if control_shift else full[control_columns]
    arr = np.column_stack([d.to_numpy(dtype=float), y.to_numpy(dtype=float), x.to_numpy(dtype=float)])
    complete = np.isfinite(arr).all(axis=1)
    if allowed_dates is not None:
        allowed = pd.DatetimeIndex(allowed_dates)
        complete &= full.index.isin(allowed)
    eligible_dates = pd.DatetimeIndex(full.index[complete])
    if embargo is None:
        embargo = int(horizon) + (9 if control_shift > 0 or exposure_shift > 0 else 8)
    blocks = date_blocks(full.index, window, predefined_blocks)
    mhat = pd.Series(np.nan, index=full.index, dtype=float)
    ghat = pd.Series(np.nan, index=full.index, dtype=float)
    fold_records = []
    for f, (a, b) in enumerate(blocks, start=1):
        test_date_mask = (eligible_dates >= a) & (eligible_dates <= b)
        if scheme == 'FORWARD_4' and f == 1:
            fold_records.append({'scheme': scheme, 'fold': f, 'test_start': a, 'test_end': b,
                                 'train_start': '', 'train_end': '', 'n_train': 0,
                                 'n_test': int(test_date_mask.sum()), 'status': 'BURN_IN_NOT_SCORED'})
            continue
        if scheme == 'FORWARD_4':
            train_date_mask = eligible_dates < (a - pd.Timedelta(days=embargo))
        elif scheme == 'BLOCKED_4':
            outside = ((eligible_dates < (a - pd.Timedelta(days=embargo))) |
                       (eligible_dates > (b + pd.Timedelta(days=embargo))))
            train_date_mask = outside
        else:
            raise ValueError(scheme)
        train_dates = eligible_dates[train_date_mask]
        test_dates = eligible_dates[test_date_mask]
        status = 'SCORED'
        minimum_train = 120 if window is not None else 150
        if len(train_dates) < minimum_train or len(test_dates) < (30 if window is not None else 50):
            status = 'NOT_ESTIMABLE_FOLD_SIZE'
            fold_records.append({'scheme': scheme, 'fold': f, 'test_start': a, 'test_end': b,
                                 'train_start': train_dates.min() if len(train_dates) else '',
                                 'train_end': train_dates.max() if len(train_dates) else '',
                                 'n_train': len(train_dates), 'n_test': len(test_dates), 'status': status})
            continue
        Xtr = x.loc[train_dates].to_numpy(dtype=float); Xte = x.loc[test_dates].to_numpy(dtype=float)
        mhat.loc[test_dates] = nuisance_predict(Xtr, d.loc[train_dates], Xte, learner)
        ghat.loc[test_dates] = nuisance_predict(Xtr, y.loc[train_dates], Xte, learner)
        fold_records.append({'scheme': scheme, 'fold': f, 'test_start': a, 'test_end': b,
                             'train_start': train_dates.min(), 'train_end': train_dates.max(),
                             'n_train': len(train_dates), 'n_test': len(test_dates), 'status': status})
    scored = mhat.notna() & ghat.notna() & complete
    dates = full.index[scored]
    dd = d.loc[dates].to_numpy(dtype=float); yy = y.loc[dates].to_numpy(dtype=float)
    mm = mhat.loc[dates].to_numpy(dtype=float); gg = ghat.loc[dates].to_numpy(dtype=float)
    v = dd - mm; u = yy - gg
    denom = float(np.dot(v, v))
    theta = float(np.dot(v, u) / denom) if denom > 0 else np.nan
    adequate = (np.isfinite(theta) and len(dates) >= (250 if window is not None else (400 if require_primary else 1))
                and all(x['status'] in {'SCORED', 'BURN_IN_NOT_SCORED'} for x in fold_records)
                and np.isfinite(np.var(dd)) and np.var(dd) > 0
                and np.var(v) / np.var(dd) >= 0.01)
    # Nuisance prediction R2 against a mean-only reference.
    r2d = 1 - float(np.sum((dd - mm)**2)) / float(np.sum((dd - dd.mean())**2)) if np.var(dd)>0 else np.nan
    r2y = 1 - float(np.sum((yy - gg)**2)) / float(np.sum((yy - yy.mean())**2)) if np.var(yy)>0 else np.nan
    sd = float(np.std(d.to_numpy(dtype=float)[np.isfinite(d.to_numpy(dtype=float))], ddof=1)) if np.isfinite(d.to_numpy(dtype=float)).sum()>1 else np.nan
    if adequate:
        se, p, lo, hi = hac_from_scores(pd.DataFrame({'date': dates}), v, u, theta, horizon, full.index)
    else:
        se = p = lo = hi = np.nan
    scores = pd.DataFrame({'date': dates, 'asset': asset, 'D': dd, 'Y': yy, 'mhat': mm, 'ghat': gg, 'v': v, 'u': u})
    stats = {
        'asset': asset, 'treatment': treatment, 'outcome': outcome, 'horizon': int(horizon),
        'learner': learner, 'scheme': scheme, 'control_shift': control_shift,
        'exposure_shift': exposure_shift, 'embargo_days': int(embargo),
        'theta': theta, 'hac_se': se, 'raw_p': p, 'ci_lower': lo, 'ci_upper': hi,
        'effect_1sd': theta * sd if np.isfinite(theta) and np.isfinite(sd) else np.nan,
        'exposure_sd_clean': sd, 'r2_treatment_oof': r2d, 'r2_outcome_oof': r2y,
        'residual_exposure_variance_ratio': float(np.var(v)/np.var(dd)) if np.var(dd)>0 else np.nan,
        'n_dates': len(dates), 'n_rows': len(dates),
        'first_date': dates.min().date().isoformat() if len(dates) else '',
        'last_date': dates.max().date().isoformat() if len(dates) else '',
        'n_folds_active': sum(x['status']=='SCORED' for x in fold_records),
        'fold_adequacy': all(x['status'] in {'SCORED','BURN_IN_NOT_SCORED'} for x in fold_records),
        'status': 'PASS' if adequate else 'NOT_ESTIMABLE_OR_DIAGNOSTIC_INADEQUACY',
    }
    return FitResult(stats, scores, fold_records)


def moving_block_bootstrap(scores: pd.DataFrame, full_spine: pd.DatetimeIndex,
                           horizon: int, spec_id: str, reps=999, block_length=None):
    if scores.empty: return {'valid_reps': 0, 'ci_lower': np.nan, 'ci_upper': np.nan, 'sign_probability': np.nan, 'status': 'NOT_ESTIMABLE'}
    dates = pd.DatetimeIndex(scores['date'])
    lookup = {d: i for i, d in enumerate(dates)}
    spine_pos = {d: i for i, d in enumerate(full_spine)}
    row_at_pos = np.full(len(full_spine), -1, dtype=int)
    for i, d in enumerate(dates): row_at_pos[spine_pos[d]] = i
    ncal = len(full_spine); b = int(block_length or max(7, int(horizon)+1))
    seed = deterministic_seed(spec_id)
    rng = np.random.default_rng(seed)
    v = scores.v.to_numpy(dtype=float); u = scores.u.to_numpy(dtype=float)
    n_draw_blocks = int(np.ceil(ncal / b)); vals=[]
    starts_max = ncal-b+1
    for _ in range(reps):
        starts = rng.integers(0, starts_max, size=n_draw_blocks)
        ix = np.concatenate([np.arange(s, s+b) for s in starts])[:ncal]
        picked = row_at_pos[ix]; picked = picked[picked >= 0]
        if len(picked) == 0: continue
        vv=v[picked]; uu=u[picked]; denom=float(np.dot(vv,vv))
        if denom <= 0: continue
        th=float(np.dot(vv,uu)/denom)
        if np.isfinite(th): vals.append(th)
    vals=np.asarray(vals,dtype=float)
    if len(vals) < 950:
        return {'valid_reps': len(vals), 'block_length': b, 'seed': seed,
                'ci_lower': np.nan, 'ci_upper': np.nan, 'sign_probability': np.nan,
                'status': 'BOOTSTRAP_ADEQUACY_FAIL'}
    base=float(np.dot(v,u)/np.dot(v,v))
    return {'valid_reps': len(vals), 'block_length': b, 'seed': seed,
            'ci_lower': float(np.quantile(vals,.025)), 'ci_upper': float(np.quantile(vals,.975)),
            'sign_probability': float(np.mean(np.sign(vals)==np.sign(base))),
            'status': 'PASS', 'replicate_sd':float(np.std(vals,ddof=1))}


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    p = np.where(np.isfinite(p), p, 1.0)
    n = len(p); order = np.argsort(p); adj = np.empty(n, dtype=float); running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (n-rank)*p[idx])
        running = max(running, val); adj[idx] = running
    return adj


def pvalue_normal(theta, se):
    if not np.isfinite(theta) or not np.isfinite(se) or se <= 0: return np.nan
    return float(2 * norm.sf(abs(theta/se)))
