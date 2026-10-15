# Reproducibility package

**Forecasting Cryptocurrency Spot Illiquidity with Perpetual-Futures Information: Multi-Model Benchmarks, Residual Corrections, and Cross-Venue Evaluation**

This directory collects the analysis source, frozen aggregate inputs, archived forecasts, protocols, and manuscript tables used in the study. It is intended to stand on its own as a repository.

## Quick check

Use Python 3.12. From this directory:

~~~sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r environment/requirements-report.txt
make reproduce
~~~

The check validates the saved files and headline values, then rebuilds the two summary figures in `build/figures/`. It does not fit models or contact an exchange.

## Contents

- `src/` contains the analysis code used for data preparation and estimation.
- `data/` contains the frozen, compact panels needed by the archived analysis scripts. Raw exchange archives are not included.
- `outputs/` contains the registered protocols, saved predictions, model results, and offline checks.
- `evidence/` contains the reconciled model-level and generalization files used for the manuscript audit.
- `tables/` contains cleanly numbered manuscript-ready tables in CSV and Markdown.
- `figures/` contains the two supplied summary figures; `scripts/render_figures.py` rebuilds them from the archived tables.
- `source_manifest.csv` maps each bundled research artifact to its original project path and SHA-256.
- `SHA256SUMS` checks the contents of this package.

## Re-running estimation

The archived stage-specific scripts and configurations retain their original relative paths so the recorded commands still work. Start from a fresh clone and follow the protocol and environment record beside the relevant results. Some stages write to their canonical output directories, so run them in a disposable working copy if the archived results need to remain untouched.

The model runs use different historical Python environments. There is no single combined lock file that accurately describes every stage; recorded versions are kept with the corresponding experiment. The reports and figures can be verified without installing those model environments.

The bundled panels support re-running the estimation stages. Rebuilding every panel from exchange archives is a separate, substantially larger data-ingestion task; the raw archives are deliberately excluded from Git.

## Evidence status

Historical benchmark results, retrospective neural-scaling diagnostics, retrospective development-selected comparisons, and the frozen external-venue QML evaluation are kept as separate evidence. The original registered decision remains 2 of 18 tasks passing. The broad generalization rule failed, and the external QML result is `NO_EXTERNAL_QML_REPLICATION`. A retrospective result is not an independent confirmation.

The final author-approved LaTeX manuscript and bibliography were not present in the project when this package was assembled. They are therefore not represented here. The older Markdown draft was not substituted.

## Data and reuse

See `DATA.md` for the bundled inputs, omissions, and data provenance. Check the applicable exchange data terms before redistributing this repository publicly.

No project software or data license was present in the source tree, so this package does not assign one. Add the authors' chosen license before public release.
