# QML / model package installation log

The original project environment lacked `xgboost`, `catboost`, and PennyLane; `lightgbm` was present. A project-local `.venv_targeted24h` was created because the system Python lacked `ensurepip` and the prior-stage environment was kept unchanged. The first `python3 -m venv` attempt failed for that reason; setup recovered with the already available `uv` tool.

Installation command: `uv pip install --python .venv_targeted24h/bin/python numpy pandas scipy scikit-learn xgboost lightgbm catboost torch pennylane`, followed by `uv pip install --python .venv_targeted24h/bin/python arch statsmodels`.

All installed packages imported successfully. Exact versions and the PennyLane simulator/circuit settings are recorded in `qml_environment.json`. The quantum kernel, QNN, and hybrid QNN each passed deterministic linear, nonlinear-interaction, and null synthetic fixtures before any empirical model-family comparison. No market data were downloaded.
