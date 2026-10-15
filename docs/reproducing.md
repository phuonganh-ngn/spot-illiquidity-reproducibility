# Reproducing the reported evidence

## 1. Verify the bundle

From the package root, run:

~~~sh
make verify
~~~

This checks every bundled file against `SHA256SUMS`, checks copied research files against `source_manifest.csv`, and verifies the main archived counts and estimates. It does not recalculate inference or fit a model.

## 2. Rebuild the supplied figures

~~~sh
make figures
~~~

The figures are generated from the CSV tables in `tables/`. The script writes to `build/figures/` so the supplied copies under `figures/` remain unchanged.

## 3. Re-run a model stage

The estimation code is preserved in `src/` and in the script directories under `outputs/phase13_neural_target_scaling/` and `outputs/phase14_strong_baseline_benchmark/`. Each stage has a separate protocol, data registry, and software record. Read those files before executing a stage. The saved results should be treated as immutable reference outputs; use a disposable clone for refitting.

The historical final test interval was examined in prior analyses. The development-selected benchmark is retrospective, and its test period is not an untouched confirmation set. The later Bybit evaluation is the external venue test, with the frozen conclusion `NO_EXTERNAL_QML_REPLICATION`.

This package does not provide a one-command refit of every historical study. The stages use different environments and have different run and freeze procedures. It does provide the original stage code, the frozen panels required by the estimation scripts, the registered configurations, saved predictions, and the checks needed to audit the published evidence.
