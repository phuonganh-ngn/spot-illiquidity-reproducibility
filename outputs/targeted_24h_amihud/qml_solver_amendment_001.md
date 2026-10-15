# QML solver amendment 001

Before inspecting any QML-versus-classical predictive loss, the first empirical QML execution emitted repeated scikit-learn convergence warnings for the matched one-hidden-layer MLP at its fixed 500-iteration ceiling. The run had produced only intermediate prediction checkpoints; no QML/classical loss comparison or asset-level QML result was computed or viewed. Those incomplete checkpoints are preserved separately and excluded from the analysis.

For a fair matched nonlinear benchmark, the MLP optimizer ceiling is fixed at 2,000 iterations, with the already-used Adam solver, learning rate, L2 penalty, tolerance, architecture, training-only PCA/scaling, target scaling, data split, feature dimensions, and five seeds unchanged. The validation and empirical scripts record convergence status and warnings for every fit. The revised solver setting is validated on deterministic synthetic fixtures and hashed before restarting the QML evaluation.

This amendment does not change the target, target transformation, features, interactions, sample, temporal folds, holdout, quantum circuit, qubit counts, seed list, model-family set, paired comparison, inference, or classification rules. It does not use forecast scores to select the iteration ceiling.
