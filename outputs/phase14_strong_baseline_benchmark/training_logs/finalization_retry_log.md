# Reporting-only finalization retry

The first post-estimation audit pass stopped while writing `development_metric_reconstruction.csv`: the report builder selected a `match` column before adding it. The exception occurred after development and final-test training/prediction were complete. It did not alter any model, prediction, selected configuration, metric, bootstrap draw, source artifact, or frozen file. The reporting code was corrected to add the column after selecting the reconstructed metrics. No model was refit and no statistical comparison was changed.

A second audit pass then stopped at the Phase 13 saved-origin cross-check because that input names its timestamp column `origin_utc` rather than `origin`. The validation code was corrected to use the archived schema. Again, no model was refit and no forecast, selection, metric, inference result, or earlier-stage artifact was changed.
