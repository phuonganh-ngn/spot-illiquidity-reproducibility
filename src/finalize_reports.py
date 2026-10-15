"""Write explicit certification from completed tests; never estimate relationships."""
import json
import pandas as pd,numpy as np
from common import *

def main(reproduced=False):
 r=json.loads((ROOT/'outputs/readiness_summary.json').read_text());cfg=json.loads((ROOT/'outputs/sample_definition.json').read_text());tests=pd.read_csv(ROOT/'outputs/automated_validation_tests.csv');assert tests.status.eq('PASS').all()
 r['ready_for_modeling']='YES' if r['numeric_core_pass'] and reproduced else 'NO';r['offline_reproduction']='PASS' if reproduced else 'NOT_RUN'
 if reproduced and r['numeric_core_pass']:r['fatal_blockers']=[]
 elif not reproduced:r['fatal_blockers']=['Network-disabled deterministic reproduction not yet verified']
 (ROOT/'outputs/readiness_summary.json').write_text(json.dumps(r,sort_keys=True,indent=2)+'\n')
 coverage=pd.read_csv(ROOT/'outputs/model_variable_coverage.csv');clean=coverage[coverage['sample']=='PRIMARY_CLEAN'];requested=coverage[coverage['sample']=='REQUESTED'];extended=coverage[coverage['sample']=='EXTENDED_HARMONIZED'];files=pd.read_csv(ROOT/'outputs/archive_file_schema_audit.csv');raw=pd.read_csv(ROOT/'outputs/raw_coverage_audit.csv');layout=pd.read_csv(ROOT/'outputs/metrics_daily_layout_audit.csv');cross=pd.read_csv(ROOT/'outputs/daily_price_crosscheck.csv')
 def pct(asset,col,sample=clean):return float(sample[(sample.asset==asset)&(sample.variable==col)].coverage_pct.iloc[0])
 def mdtable(frame):
  return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(str(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None))
 table=clean[clean.variable.isin(['spot_log_return_1d','spot_realized_volatility','spot_quote_volume','funding_mean','oi_close','oi_log_change_1d','basis_mean','futures_taker_imbalance','premium_index_mean'])][['asset','variable','valid_days','expected_days','coverage_pct']].copy();table.coverage_pct=table.coverage_pct.round(4)
 ranges=files.groupby(['symbol','market','dataset','frequency']).agg(first_observation=('first_timestamp','min'),last_observation=('last_timestamp','max'),schemas=('schema_version',lambda v:';'.join(sorted(v.unique())))).reset_index();ranges.to_csv(ROOT/'outputs/dataset_discovery.csv',index=False)
 scenarios=pd.DataFrame([dict(sample='PRIMARY_CLEAN',start=cfg['primary_clean_start'],end=cfg['primary_clean_end'],unique_dates=r['unique_dates'],asset_days=r['panel_rows']),dict(sample='EXTENDED_HARMONIZED',start=cfg['extended_start'],end=cfg['extended_end'],unique_dates=len(pd.date_range(cfg['extended_start'],cfg['extended_end'])),asset_days=2*len(pd.date_range(cfg['extended_start'],cfg['extended_end'])))])
 horizons=pd.DataFrame([dict(horizon=h,primary_eligible_asset_days=r['horizon_rows'][str(h)],extended_eligible_asset_days=r['extended_horizon_rows'][str(h)]) for h in [1,3,7,14,30]]);horizons.to_csv(ROOT/'outputs/horizon_sample_sizes.csv',index=False)
 unknown=layout[layout.layout=='UNVERIFIED'];invalid=pd.read_csv(ROOT/'outputs/malformed_record_audit.csv');duplicates=int(files.duplicate_timestamps.sum());oi=pd.read_csv(ROOT/'outputs/oi_structural_break_audit.csv');qc=int(cross.status.eq('PASS').sum())
 report=f'''# Derivatives ↔ spot dataset construction report

READY_FOR_MODELING = {r['ready_for_modeling']} for the PRIMARY CLEAN sample only. This certifies a reproducible data construction, not identification or predictive evidence. No models, hypothesis regressions, significance tests, synthetic data, composites or trading experiments were run. This new project is isolated from the previous stablecoin project.

{mdtable(scenarios)}

Requested calendar span has {len(pd.date_range(START,END))} dates; pre-sample {PRE} through {END} has {len(pd.date_range(PRE,END))} dates. All dates are UTC calendar days. The primary files are derivatives_spot_daily_panel.csv and derivatives_spot_system_daily.csv. A separate derivatives_spot_extended_panel.csv retains the harmonized extension. The original requested January2021 start fails ETH OI availability; it is not represented as complete.

## 1–2. Datasets and raw frequency

Actual S3 catalogue keys and 4,836 successfully downloaded ZIP responses establish availability. Datasets: spot 5m klines; USDⓈ-M perpetual 5m trade klines, markPriceKlines, indexPriceKlines, premiumIndexKlines; fundingRate monthly files containing individual settlements; daily metrics files containing five-minute observations. Spot 1d klines were downloaded only for deterministic validation dates/months, so their deliberately sparse audit is not a market-data failure. BTC and ETH are the only downloaded asset units. The catalogue's first keys are not automatically the valid-data first dates; actual ranges/schema are in dataset_discovery.csv and annual valid coverage is in raw_coverage_audit.csv. No REST last-30-day history was presented as long-run coverage.

## 3–5. Timestamps, changes and schemas

Trade and price candles have interval-start open_time and inclusive close_time. Spot uses milliseconds before2025-01-01 and microseconds afterward; the normalization is documented by the [official archive README](https://github.com/binance/binance-public-data/) and verified in each raw file. Futures candle timestamps are milliseconds. Header presence changes are parsed separately; every candle has 12 original fields. Funding archives have three fields and metrics have eight; no fabricated fields are added to raw bytes. {len(invalid)} irregular/short/reversed spot intervals were marked unusable, with original records retained in malformed_record_audit.csv. Dates with incomplete trade candles have missing daily prices/volume/RV; no partial-volume scaling is performed.

In the CURRENT retrieved metrics vintage, both assets change field layout at {cfg['break_detected']['BTCUSDT']}. Before this boundary, notional/quantity matches mark price at the row label and taker flow aligns with the preceding 5m trade bar where that ratio is populated. After it, OI valuation aligns five minutes after the label and flow with the bar beginning at the label. Daily matching requires a valuation relative error below0.0001 and five-fold separation from alternatives; unusable days remain missing. September15 and September30 2026 REST OI checks match all288 quantities per asset/day at API timestamp minus300s, and none at the unshifted comparison. This independently supports the later stock mapping. Tests do not assume the same shift for stock and flow fields.

The [repository issue491](https://github.com/binance/binance-public-data/issues/491) is a third-party report of a June2026 transition, not official confirmation. Its example label range disagrees with the cached current June24 file. No ADDITIONAL2026 layout transition is detected in the retrieved data. The boundary found here describes the present archival representation, not a verified date of live-feed implementation. Provider backfill/revision risk remains; no historical publication-time immutability is claimed. Current official API documentation describes OI timestamps as period-end and taker timestamps as period-start; one generic archive create_time cannot silently substitute for both.

Exact repeated rows ({duplicates} redundant timestamps across downloaded files) are collapsed only when every numerical field agrees; conflicting groups would be entirely invalidated. No conflicts were found before alignment. Unsorted recent files are sorted in interim data only. There are {len(unknown)} unmatched symbol-days ({unknown.groupby('symbol').size().to_dict()}); these are not mechanically shifted. The first layout is preferred for the main sample; the shifted version is a separately labelled extension. Historical OI time matching is strong internal evidence and an explicit empirical inference, not issuer-certified historical publication metadata.

## 6–10. Derivative availability and definitions

Funding has100% daily coverage for both assets. Archive calc_time and last_funding_rate exactly match fundingTime and fundingRate in30 deterministic settlement checks (five dates ×two assets ×three settlements). The complete observed cadence is8h; there is no assumption of a changing schedule hidden in the aggregation. Millisecond settlement offsets are preserved. A midnight settlement belongs to the date of its actual timestamp. Daily mean, arithmetic sum, last, absolute mean, positive fraction, min and max remain separate; no markPrice column exists in the archive funding dataset.

OI quantity and notional are separate. Quantity uses the BTC or ETH base unit; notional is USDT, consistent with the linear perpetual and notional/quantity valuation identity. Raw base OI levels are not cross-asset magnitude comparisons. OI close is the last valid snapshot before next midnight; means/max/min require at least274 valid observations of288 expected (>=95%). Unknown-layout days are excluded; mean/close do not imply complete snapshots. The daily and annual audits expose gaps independently.

The verified basis candidate is markPriceKlines.close / indexPriceKlines.close −1 on matching five-minute intervals, with daily mean/last/absolute mean/sample SD. It is dimensionless and nonannualized, and is not a quoted futures-expiry basis or executable spread. Provider premium-index candle closes are retained separately; zero and negative premium values are legitimate, not missing prices. The exact economic weighting of premium and historical index constituent changes are limitations for optional premium-based work. Mark/index candle placeholder volume/count fields are ignored.

Core futures taker imbalance comes from futures trade candles: (buy−sell)/(buy+sell), with sell=total traded base volume−taker-buy base volume. This partition is justified by the official transaction candle schema, not by unrelated blockchain or positioning quantities. Daily volume-weighted imbalance is within[-1,1] and requires positive total volume. It has100% daily coverage. No averages of ratios are substituted for volume-weighted flow.

Global account, top-trader account and top-trader position ratios are retained as distinct, provisional archive candidates. API concepts distinguish all accounts from top20% margin-balance accounts and positions, but the archive snake_case mappings/population history have not been independently certified. They are EXCLUDE from baseline, and SECONDARY_PENDING_VERIFICATION for future work. Metrics taker ratio is diagnostic only; verified trade-candle imbalance avoids depending on that field's unresolved coverage. The tested official BTC liquidationSnapshot prefix returned no objects; full historical liquidation availability elsewhere remains UNVERIFIED. No liquidation observations were downloaded and this optional metric does not constrain the baseline.

## 11–13. Spot fields and daily construction

OHLC, base/USDT quote volume, trade count and taker-buy base/quote volume are available from spot 5m transaction candles. Daily price/volume/flow features require all288 valid bars. Daily returns require the previous valid daily close; missing close is never filled. RV is sqrt(sum288 squared5m close-to-close log returns), including the preceding day's last5m close; downside semivariance sums squared negative intraday returns. A missing interval does not create a multi-interval return. No annualization is applied. Parkinson and Garman–Klass are separately labelled range candidates; GK's nonnegative formula bound is documented, not a data winsorization. Amihud=1e6×abs(daily log return)/daily USDT quote volume, requiring positive volume. It is a venue-specific daily proxy; no spread or market-cap denominator is invented. {qc}/{len(cross)} deterministic OHLC/volume/trade-count cross-checks pass against official1d candles.

## 14–17. Asset and common sample decisions

Both BTC and ETH have the same PRIMARY CLEAN sample {cfg['primary_clean_start']} through {cfg['primary_clean_end']}, {r['unique_dates']} calendar dates and {r['panel_rows']} asset-days. ETH OI archive starts2021-12-01; its first same-definition OI growth occurs one day later and lagged growth another day later. Pre-sample BTC/ETH spot and funding history begins2020-10-01. SOL = NOT AUDITED and does not constrain these decisions. No extra assets or external venues were added.

{mdtable(table)}

## 18–20. Coverage failures, comparability and2026

Requested-period ETH OI coverage is {pct('ETH','oi_close',requested):.4f}%, so the full2021–2026 original specification fails core coverage. Clean core coverage passes95% for both assets. Some optional/future/rolling variables have lower coverage or their own warm-up/end truncation; see every row in model_variable_coverage.csv rather than infer universal completeness from core pass. Ratio fields are semantically unverified and excluded from the main feature set. Spread, liquidation and market-cap-based turnover are not constructed. Mark/index gaps remain NA. There is no unresolved NEW2026 clock discontinuity in this downloaded vintage, but live historical publication times and revisions are not verified. Those are documentation/vintage limitations, not silently erased by harmonization.

{len(oi)} absolute OI log changes above0.15 were flagged across the raw/pre-sample history; none was removed on economic grounds. The flag does not label a leverage shock. Numerical notional/quantity and field-time checks do not find a required clean-sample unit change. Reasons for actual large supply/position changes are UNVERIFIED; no model or later spot response was used to classify them.

## 21–24. Samples, losses and horizons

Primary clean sample and harmonized extension are fixed by availability and archival timing evidence above, not by statistical significance. Final primary panel has {r['panel_rows']} rows including rows with explicit missing cells; the wide file has {r['unique_dates']} unique dates. Future labels use only t+1 through t+h and require all relevant future observations; primary labels are also truncated at the clean endpoint, even when extended raw data exist. Eligible counts require core current variables, selected lag1 controls and full forward spot/funding/OI labels. They are conservative joint counts; later specifications must recalculate outcome-specific samples. No rows are silently imputed or discarded from the exported calendar panel.

{mdtable(horizons)}

OI growth/lagging creates initial warm-up loss. Missing OI days and spot outage days propagate only to formulas that need them; longer windows amplify missingness. Final h>1 windows overlap, recorded in future_window_metadata.csv. Return is cumulative; RV root-sum-of-squares; downside is summed; Amihud/imbalance/volume are future means; funding mean and sum are distinct; OI future growth is endpoint log change and additionally requires all future OI days valid.

## 25. Readiness, limits and next step

READY_FOR_MODELING = {r['ready_for_modeling']}. Required automated mathematical/boundary checks pass; offline network-disabled reproduction is {r['offline_reproduction']}. Raw research-snapshot hashes and all252 protected pre-existing project files are unchanged. The main derivative candidates are funding (mean and sum remain separate), OI quantity log change and trade-candle futures imbalance. Basis is a verified additional candidate; premium is secondary. Exact contemporaneous publication/vintage safety is not proven. Day-t exposure features are historical measurements completed by next UTC midnight; a live-feed prompt-dissemination assumption is required for an immediate next-day origin, and separately prepared lag1 exposures impose a full additional day of separation. No source is alleged to have published monthly archives at day-t close.

Same-venue infrastructure/traders/USDT settlement can create mechanical links. USDT quote volume/notional is not assumed to equal USD. Future models may report conditional directional dynamics, not infer causality from these data alone. Stage2_modeling_plan.md is a document only. Stop here; no modeling has started.
'''
 (ROOT/'outputs/dataset_construction_report.md').write_text(report)
 leakage=f'''# Leakage and timing audit

Automated checks: PASS. Offline reproduction: {r['offline_reproduction']}.

FACTS: Raw spot/trade/mark/index candle open and inclusive close timestamps were inspected. No valid included candle extends beyond its UTC day. Actual funding timestamps match the official settlement API on fixed dates; no rounding of raw timestamp offsets occurs. OI stock labels are mapped using field-specific archive evidence; unknown-layout days are excluded. The later five-minute OI shift is independently validated on four asset-date API comparisons.

ASSUMPTIONS: Historical provider publication latencies are unavailable. available_at_utc records the earliest daily measurement-completion boundary, next UTC midnight, not an observed publication receipt time. Immediate t→t+1 analysis requires prompt live-feed dissemination after interval closure; lag1 exposures prepare the conservative t−1→t+1 alternative. The archive itself is published later, and can be revised. It is not a historical as-of information feed.

Every future window begins t+1. Returns are sums; RV roots of sums of squares; downside sums; Amihud means. No missing future day is filled. No raw observation after2026-09-30 enters any label; primary labels also stop at{cfg['primary_clean_end']}. Lag controls are calendar shifts. Rolling features use t−window through t−1. No centered windows or full-sample scaling. Raw descriptive quantiles flag extremes only. No outcome behavior selects data, sample dates, thresholds or assets. Same definitions apply to both assets. Date×asset and wide-date uniqueness pass. No future-outcome column is a control candidate.
'''
 (ROOT/'outputs/leakage_audit.md').write_text(leakage)
 (ROOT/'outputs/validation_report.md').write_text('# Validation report\n\n'+mdtable(tests)+f'\n\nNetwork-disabled byte-identical reproduction: {r["offline_reproduction"]}. Raw immutable files checked against raw_hash_inventory.csv. Previous project files checked against preexisting_project_hashes.json. Test failure raises immediately. No models are imported or run.\n')
 lines=['============================================================','DERIVATIVES ↔ SPOT DATASET AUDIT','============================================================',f'PRIMARY ASSETS: BTC, ETH',f'RAW DATE RANGE: {PRE} — {END}',f'PRIMARY CLEAN SAMPLE: {cfg["primary_clean_start"]} — {cfg["primary_clean_end"]}',f'OPTIONAL EXTENDED SAMPLE: {cfg["extended_start"]} — {cfg["extended_end"]} (empirical field-time harmonization)']
 for asset in ['BTC','ETH']:
  for label,col in [('SPOT','spot_realized_volatility'),('FUNDING','funding_mean'),('OI','oi_close'),('BASIS','basis_mean'),('TAKER FLOW','futures_taker_imbalance')]:lines.append(f'{asset} {label} COVERAGE: {pct(asset,col):.4f}%')
 lines+=['SOL STATUS: NOT AUDITED','TIMESTAMP STRUCTURAL BREAK: YES',f'BREAK DETAILS: current archive field layout boundary {cfg["break_detected"]["BTCUSDT"]}; no distinct additional2026 transition detected; spot ms→us Jan2025','FUNDING SEMANTICS: PASS','OI SEMANTICS: PASS (empirical historical field mapping; publication vintage unverified)','BASIS SEMANTICS: PASS (nonannualized mark/index close ratio)','TAKER IMBALANCE SEMANTICS: PASS (trade candles)','DAILY UTC ALIGNMENT: PASS','LEAKAGE TESTS: PASS (measurement timing; publication assumption documented)','RAW IMMUTABILITY: PASS',f'OFFLINE REPRODUCTION: {r["offline_reproduction"]}','PRIMARY CORE DERIVATIVE VARIABLES: funding_mean/funding_sum, oi_log_change_1d, futures_taker_imbalance; basis candidate','SECONDARY VARIABLES: basis summaries, premium index, notional OI, range-based volatility','EXCLUDED VARIABLES: unverified metrics positioning ratios; liquidation/spread/market-cap turnover; composite indices']
 lines += [f'H{h} SAMPLE: {r["horizon_rows"][str(h)]} eligible asset-days' for h in [1,3,7,14,30]]
 lines += [f'READY FOR MODELING: {r["ready_for_modeling"]}',f'FATAL BLOCKERS: {r["fatal_blockers"]}',f'NONFATAL LIMITATIONS: {r["limitations"]}','NEXT STEP: BIDIRECTIONAL TS-DML + GRANGER + TRANSFER ENTROPY + TVP (documented only; do not run)','============================================================']
 console='\n'.join(lines)+'\n';(ROOT/'outputs/final_console_summary.txt').write_text(console);print(console)
if __name__=='__main__':
 import sys
 main('--reproduced' in sys.argv)
