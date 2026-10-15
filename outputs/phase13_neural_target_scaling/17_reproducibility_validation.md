# Reproducibility validation

Overall validation: **PASS**.

- all_primary_test_predictions_on_original_registered_origins: **PASS** — exact date-vector comparison against final_test_predictions.json for every asset/model/information-set/branch
- target_scaler_inverse_transform_and_control_identity: **PASS** — prediction_log1p = network_output × training-only scale + training-only mean; control output remains unscaled
- all_primary_metrics_reconstruct_from_saved_predictions: **PASS** — test MAE recomputed from saved origin-level predictions equals 08_fold_level_results.csv within 1e-14
- target_scaler_fit_training_labels_only: **PASS** — logged split masks end before evaluation; treatment scaler flag is true only for scaled branch
- original_model_configs_and_scope_retained: **PASS** — 8 frozen asset×family×information-set configs; six-bar Amihud task only
- all_locked_fit_units_completed: **PASS** — {'failed_fit_ids': [], 'fit_count': 64, 'status': 'PASS', 'wall_seconds_so_far': 17.373028704000035}
- all_phase6_phase9_phase12_dataset_code_and_manuscript_hashes_unchanged: **PASS** — 283 paths compared to pre-training SHA256 inventory
- pretraining_protocol_lock_unchanged: **PASS** — pretraining_lock_manifest.sha256 and all four frozen file hashes

Software: Python 3.12.3, NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.9.1, PyTorch 2.14.1+cpu, matplotlib 3.11.2.
The final-test predictions are checked against the frozen Phase 6 date vector; target-scaler inverse transformation and saved MAEs are independently reconstructed from CSV predictions. No second training pass was performed as part of this final integrity audit.
