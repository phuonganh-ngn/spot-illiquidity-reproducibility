"""Validated Phase-5 rerun estimators and synthetic fixtures.

Method notes:
* PCMCI+ is Tigramite's PCMCI.run_pcmciplus.
* VAR-LiNGAM is lingam.VARLiNGAM (not DirectLiNGAM on levels).
* DYNOTEARS is a direct NumPy/SciPy port of the published objective and the
  reference CausalNex implementation pinned by commit in the run manifest.
* Temporal NOTEARS-MLP adapts the official Zheng et al. nonlinear NOTEARS MLP
  to a fixed past-to-current mask. The mask makes the time-unrolled graph a
  DAG by construction; it does not impose an instantaneous graph.
* cMLP uses the reference Neural-GC implementation pinned in the run manifest.
"""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
import torch
from lingam import DirectLiNGAM, VARLiNGAM
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / ".cache_phase5_rerun" / "repos"
sys.path.insert(0, str(REF / "notears"))
sys.path.insert(0, str(REF / "Neural-GC"))
from notears.nonlinear import NotearsMLP, notears_nonlinear
from models.cmlp import cMLP, train_model_gista

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.set_default_dtype(torch.float64)

VARS = [
    "funding_rate_state", "oi_log_change", "futures_taker_imbalance",
    "spot_return", "spot_realized_volatility", "spot_downside_semivariance",
    "spot_amihud", "spot_taker_imbalance", "spot_quote_volume_growth", "basis_mean",
]
LAGS = tuple(range(1, 7))
REG_GRID = (0.01, 0.05, 0.10, 0.20)
EDGE_THRESH = {
    "dynotears": 0.05,
    "notears_mlp": 0.30,  # reference NOTEARS-MLP default threshold
    "neural_granger": 1e-6,  # exact group-lasso zeros in reference cMLP
    "var_lingam": 0.05,
}


def ar_fixtures(kind: str, n: int = 2200, seed: int = 41329):
    """Stationary synthetic systems; all directions are known by construction."""
    rng = np.random.default_rng(seed)
    d = 3
    x = np.zeros((n, d), dtype=float)
    # Non-Gaussian innovations help validate LiNGAM assumptions.
    e = rng.laplace(0.0, 1 / math.sqrt(2), size=(n, d))
    if kind == "null":
        for t in range(1, n):
            x[t] = 0.45 * x[t - 1] + e[t]
        expected = []
    elif kind == "single_direction":
        for t in range(1, n):
            x[t, 0] = 0.35 * x[t - 1, 0] + e[t, 0]
            x[t, 1] = 0.35 * x[t - 1, 1] + 0.85 * x[t - 1, 0] + e[t, 1]
            x[t, 2] = 0.35 * x[t - 1, 2] + e[t, 2]
        expected = [(0, 1, 1)]
    elif kind == "chain":
        for t in range(1, n):
            x[t, 0] = 0.35 * x[t - 1, 0] + e[t, 0]
            x[t, 1] = 0.30 * x[t - 1, 1] + 0.75 * x[t - 1, 0] + e[t, 1]
            x[t, 2] = 0.30 * x[t - 1, 2] + 0.75 * x[t - 1, 1] + e[t, 2]
        expected = [(0, 1, 1), (1, 2, 1)]
    elif kind == "bidirectional":
        for t in range(1, n):
            x[t, 0] = 0.25 * x[t - 1, 0] + 0.50 * x[t - 1, 1] + e[t, 0]
            x[t, 1] = 0.25 * x[t - 1, 1] + 0.50 * x[t - 1, 0] + e[t, 1]
            x[t, 2] = 0.35 * x[t - 1, 2] + e[t, 2]
        expected = [(0, 1, 1), (1, 0, 1)]
    elif kind == "nonlinear":
        for t in range(1, n):
            x[t, 0] = 0.35 * x[t - 1, 0] + e[t, 0]
            x[t, 1] = 0.45 * x[t - 1, 1] + 0.8 * (x[t - 1, 0] ** 2 - 1.0) + e[t, 1]
            x[t, 2] = 0.35 * x[t - 1, 2] + e[t, 2]
        expected = [(0, 1, 1)]
    else:
        raise ValueError(kind)
    x = x[200:]  # remove initialization burn-in
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    return x, expected


def _as_realizations(data):
    if isinstance(data, (list, tuple)):
        return {i: np.asarray(block, dtype=float) for i, block in enumerate(data)}
    return np.asarray(data, dtype=float)


def pcmci_fit(data, names, alpha=0.05, ci="parcorr"):
    dta = _as_realizations(data)
    frame = pp.DataFrame(dta, var_names=names, analysis_mode="multiple" if isinstance(dta, dict) else "single")
    if ci == "parcorr":
        test = ParCorr(significance="analytic")
    elif ci == "cmiknn":
        from tigramite.independence_tests.cmiknn import CMIknn
        test = CMIknn(knn=0.2, significance="shuffle_test", shuffle_neighbors=5,
                      workers=1, transform="ranks")
    else:
        raise ValueError(ci)
    model = PCMCI(dataframe=frame, cond_ind_test=test, verbosity=0)
    result = model.run_pcmciplus(tau_min=1, tau_max=6, pc_alpha=alpha,
                                 max_conds_dim=None, max_combinations=1,
                                 fdr_method="none")
    q = model.get_corrected_pvalues(result["p_matrix"], fdr_method="fdr_bh",
                                   exclude_contemporaneous=True,
                                   tau_min=1, tau_max=6)
    edges = []
    graph = result["graph"]
    vals = result["val_matrix"]
    pvals = result["p_matrix"]
    for i, source in enumerate(names):
        for j, target in enumerate(names):
            for lag in LAGS:
                try:
                    g = str(graph[i, j, lag])
                    pv = float(q[i, j, lag])
                    value = float(vals[i, j, lag])
                    raw_p = float(pvals[i, j, lag])
                except Exception:
                    continue
                if ("->" in g or "-->" in g) and np.isfinite(pv) and pv < 0.05:
                    edges.append({"source": source, "target": target, "lag": lag,
                                  "weight": value, "sign": int(np.sign(value)),
                                  "raw_p": raw_p, "adjusted_p": pv, "graph_mark": g})
    return {"edges": edges, "result": result, "q_matrix": q, "ci_test": ci}


def dynotears_design(blocks, p=6):
    ys, xs = [], []
    for seq in blocks:
        seq = np.asarray(seq, float)
        for t in range(p, len(seq)):
            ys.append(seq[t])
            xs.append(np.concatenate([seq[t-k] for k in range(1, p+1)]))
    return np.asarray(ys), np.asarray(xs)


def dynotears_fit(blocks, lam=0.05, p=6, max_outer=20, max_inner=300, h_tol=1e-8):
    """DYNOTEARS objective/augmented Lagrangian, with the CausalNex convention."""
    Y, Xlags = dynotears_design(blocks, p)
    n, d = Y.shape
    if n <= 2 * d:
        raise ValueError("insufficient lagged rows")
    # Vector stores W+/- then lagged A+/- interleaved by lag, exactly as
    # CausalNex's _reshape_wa reference implementation.
    def reshape(v):
        wtilde = v.reshape(2 * (p + 1) * d, d)
        W = wtilde[:d] - wtilde[d:2*d]
        aa = wtilde[2*d:].reshape(2*p, d*d)
        Ap = aa[::2].reshape(p*d, d)
        Am = aa[1::2].reshape(p*d, d)
        A = Ap - Am
        return W, A
    bounds = []
    for _sgn in range(2):
        for i in range(d):
            for j in range(d):
                bounds.append((0, 0) if i == j else (0, None))
    for _lag in range(p):
        for _sgn in range(2):
            bounds.extend([(0, None)] * (d*d))
    v = np.zeros(2 * (p+1) * d*d)
    rho, alpha, h_old, h = 1.0, 0.0, np.inf, np.inf
    last_objective = None
    for _outer in range(max_outer):
        while rho < 1e20 and (h > 0.25 * h_old or not np.isfinite(h)):
            def fun_grad(vv):
                W, A = reshape(vv)
                residual = Y - Y @ W - Xlags @ A
                loss = 0.5 / n * np.sum(residual * residual)
                expW = slin.expm(W * W)
                hval = float(np.trace(expW) - d)
                obj = loss + 0.5 * rho * hval*hval + alpha*hval
                obj += lam * (np.sum(vv[:2*d*d]) + np.sum(vv[2*d*d:]))
                gradW = -Y.T @ residual / n + (rho*hval + alpha) * expW.T * W * 2
                gradA = -Xlags.T @ residual / n
                grad = np.concatenate([
                    gradW.ravel(), -gradW.ravel(),
                    np.stack([gradA.reshape(p, d*d), -gradA.reshape(p, d*d)], axis=1).ravel(),
                ]) + lam
                return float(obj), grad
            opt = sopt.minimize(fun_grad, v, method="L-BFGS-B", jac=True,
                                bounds=bounds,
                                options={"maxiter":max_inner, "ftol":1e-9, "maxls":30})
            v = opt.x
            W, A = reshape(v)
            h = float(np.trace(slin.expm(W * W)) - d)
            last_objective = float(opt.fun)
            if h > 0.25 * h_old:
                rho *= 10
        h_old = h
        alpha += rho * h
        if h <= h_tol or rho >= 1e20:
            break
    W, A = reshape(v)
    return {"W": W, "A": A.reshape(p, d, d), "acyclicity_h": h,
            "objective": last_objective, "converged": bool(h <= h_tol),
            "outer_iterations": _outer + 1, "n_rows": int(n)}


def temporal_notears_mlp_fit(blocks, names, hidden=10, lam=0.05, max_iter=40,
                             w_threshold=0.30):
    """Official NOTEARS-MLP with a fixed lag-node -> current-node DAG mask."""
    # Each row is [X_(t-1),...,X_(t-6), X_t]. Only lag nodes may parent current.
    d = len(names); p = 6
    y, lagged = dynotears_design(blocks, p)
    x = np.column_stack([lagged, y])
    n_nodes = (p + 1) * d
    model = NotearsMLP(dims=[n_nodes, hidden, 1], bias=True)
    # Bounds use official NOTEARS-MLP fc1 layout (target node, hidden unit, source node).
    bounds = []
    for target in range(n_nodes):
        for _h in range(hidden):
            for source in range(n_nodes):
                allowed = (target >= p*d and source < p*d)
                bounds.append((0, None) if allowed else (0, 0))
    model.fc1_pos.weight.bounds = bounds
    model.fc1_neg.weight.bounds = bounds
    W = notears_nonlinear(model, x.astype(np.float64), lambda1=lam, lambda2=0.01,
                          max_iter=max_iter, h_tol=1e-8, rho_max=1e12,
                          w_threshold=w_threshold)
    # Lagged input node = lag group [0..p*d), current targets=[p*d..].
    scores = W[:p*d, p*d:].reshape(p, d, d)  # lag, source, target
    return {"weights": scores, "acyclicity_h": float(model.h_func().detach().cpu()),
            "n_rows": int(n), "finite": bool(np.isfinite(scores).all())}


def cmlp_fit(blocks, names, lam=0.05, hidden=10, max_iter=300, seed=314159):
    """Reference Neural-GC cMLP with group-lasso proximal training."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = np.stack([np.asarray(b, dtype=np.float64) for b in blocks], axis=0)
    tensor = torch.as_tensor(data, dtype=torch.float64)
    model = cMLP(num_series=len(names), lag=6, hidden=[hidden], activation="relu")
    losses, mses = train_model_gista(model, tensor, lam=lam, lam_ridge=0.01,
                                    lr=0.01, penalty="GL", max_iter=max_iter,
                                    check_every=max_iter + 1, verbose=0)
    weights = model.GC(threshold=False, ignore_lag=False).detach().cpu().numpy()
    # Reference GC is target x source x lag; expose lag x source x target.
    return {"weights": np.transpose(weights, (2, 1, 0)),
            "loss": float(losses[-1].detach().cpu()),
            "mse": float(mses[-1].detach().cpu()), "finite": bool(np.isfinite(weights).all())}


def lagged_design_from_blocks(blocks, p=6):
    y, xlag = dynotears_design(blocks, p)
    return xlag, y


def segmented_var_lingam_fit(blocks, p, seed=314159):
    """VAR-LiNGAM stages with within-block VAR pairs, avoiding block seams."""
    xs, ys = [], []
    for seq in blocks:
        seq = np.asarray(seq, float)
        if len(seq) <= p:
            continue
        for t in range(p, len(seq)):
            xs.append(np.concatenate([seq[t-k] for k in range(1,p+1)]))
            ys.append(seq[t])
    Xlag, Y = np.asarray(xs), np.asarray(ys)
    coef = np.linalg.lstsq(Xlag, Y, rcond=None)[0]  # [p*d, target]
    pred = Xlag @ coef
    residual = Y - pred
    model = DirectLiNGAM(random_state=seed)
    model.fit(residual)
    B0 = np.asarray(model.adjacency_matrix_)
    d = Y.shape[1]
    M = np.stack([coef[k*d:(k+1)*d].T for k in range(p)])  # reduced-form target x source
    Blag = np.stack([(np.eye(d) - B0) @ M[k] for k in range(p)])
    return {"contemporaneous": B0, "lagged": Blag,
            "causal_order": [int(x) for x in model.causal_order_],
            "n_rows": int(len(Y)), "finite": bool(np.isfinite(Blag).all())}


def varlingam_fit(data, max_lag=6, seed=314159):
    model = VARLiNGAM(lags=max_lag, criterion="bic", prune=False, random_state=seed)
    model.fit(np.asarray(data, dtype=float))
    mats = np.asarray(model.adjacency_matrices_)
    return {"matrices": mats, "selected_lag": int(len(mats)-1),
            "causal_order": [int(x) for x in model.causal_order_],
            "residuals": np.asarray(model.residuals_), "finite": bool(np.isfinite(mats).all())}


def edges_from_weight_tensor(weights, names, threshold=0.05):
    arr = np.asarray(weights)
    edges = []
    for lag_idx in range(arr.shape[0]):
        for i, src in enumerate(names):
            for j, dst in enumerate(names):
                w = float(arr[lag_idx, i, j])
                if np.isfinite(w) and abs(w) >= threshold:
                    edges.append({"source": src, "target": dst, "lag": lag_idx+1,
                                  "weight": w, "sign": int(np.sign(w))})
    return edges


def edge_set(edges):
    return {(e["source"], e["target"], int(e["lag"])) for e in edges}


def expected_recovery(edges, expected, allow_adjacent=False):
    es = edge_set(edges)
    return {f"{s}->{t}@{lag}": any((s,t,l)==e or (allow_adjacent and s==e[0] and t==e[1] and abs(l-e[2])<=1) for e in es)
            for s,t,lag in expected}


def varlag_edges(matrices, names, threshold=0.05):
    # lingam matrices: [0] contemporaneous B[target,source], [k] lagged B[target,source].
    out = []
    for lag in range(1, len(matrices)):
        M = matrices[lag]
        for target, tname in enumerate(names):
            for source, sname in enumerate(names):
                w = float(M[target, source])
                if abs(w) >= threshold:
                    out.append({"source":sname,"target":tname,"lag":lag,
                                "weight":w,"sign":int(np.sign(w))})
    return out


def pcmci_fixture(kind, x, expected, ci="parcorr", alpha=0.05):
    names = ["X", "Y", "Z"]
    fit = pcmci_fit(x, names, alpha=alpha, ci=ci)
    rec = expected_recovery(fit["edges"], expected, allow_adjacent=True)
    return {"edge_count":len(fit["edges"]), "edges":fit["edges"],
            "expected_recovery":rec, "ci_test":ci}


def _fixture_dynotears(x, expected):
    fit = dynotears_fit([x], lam=0.05, p=2, max_outer=15, max_inner=200)
    W = fit["W"]
    lagged = fit["A"].reshape(2,3,3)
    edges=edges_from_weight_tensor(lagged,["X","Y","Z"],threshold=0.05)
    return {"edges":edges,"edge_count":len(edges),"converged":fit["converged"],
            "acyclicity_h":fit["acyclicity_h"],"expected_recovery":expected_recovery(edges,expected,allow_adjacent=True),
            "contemporaneous_nonzero":int(np.sum(np.abs(W)>=0.05))}


def _fixture_varlingam(x, expected):
    fit=varlingam_fit(x,max_lag=6,seed=7919)
    edges=varlag_edges(fit["matrices"],["X","Y","Z"],threshold=0.05)
    return {"edges":edges,"edge_count":len(edges),"selected_lag":fit["selected_lag"],
            "causal_order":fit["causal_order"],
            "expected_recovery":expected_recovery(edges,expected,allow_adjacent=True)}


def _fixture_neural(x, expected, method, lam=0.05):
    blocks=[x]
    if method=="neural_granger":
        fit=cmlp_fit(blocks,["X","Y","Z"],lam=lam,hidden=10,max_iter=250,seed=7919)
        edges=edges_from_weight_tensor(fit["weights"],["X","Y","Z"],threshold=1e-5)
        extras={"loss":fit["loss"],"mse":fit["mse"]}
    else:
        fit=temporal_notears_mlp_fit(blocks,["X","Y","Z"],hidden=10,lam=lam,max_iter=35,w_threshold=0.30)
        edges=edges_from_weight_tensor(fit["weights"],["X","Y","Z"],threshold=0.30)
        extras={"acyclicity_h":fit["acyclicity_h"]}
    return {"edges":edges,"edge_count":len(edges),"expected_recovery":expected_recovery(edges,expected,allow_adjacent=True),**extras}


def validate_method_fixtures():
    # All fixtures are generated and fitted before the empirical panel is opened.
    fixtures={name:ar_fixtures(name) for name in ["null","single_direction","chain","bidirectional","nonlinear"]}
    output={"seed":41329,"n_after_burnin":2000,"variables":["X","Y","Z"],
            "fixtures":{},"methods":{},"protocol_note":"Synthetic-only validation; no BTC/ETH feature values or empirical graph results are accessed."}
    for name,(x,expected) in fixtures.items():
        output["fixtures"][name]={"n":int(len(x)),"expected_edges":[{"source":s,"target":t,"lag":l} for s,t,l in expected]}
    method_rows={k:{} for k in ["pcmci_plus_parcorr","pcmci_plus_cmiknn","var_lingam","dynotears","notears_mlp","neural_granger_cmlp"]}
    for name,(x,expected) in fixtures.items():
        for method in method_rows:
            if method=="pcmci_plus_cmiknn" and name not in {"null","nonlinear"}:
                continue
            try:
                if method=="pcmci_plus_parcorr":
                    row=pcmci_fixture(name,x,expected,"parcorr")
                elif method=="pcmci_plus_cmiknn":
                    row=pcmci_fixture(name,x,expected,"cmiknn")
                elif method=="var_lingam":
                    row=_fixture_varlingam(x,expected)
                elif method=="dynotears":
                    row=_fixture_dynotears(x,expected)
                else:
                    row=_fixture_neural(x,expected,"neural_granger" if method.endswith("cmlp") else "notears_mlp")
                row["status"]="PASS"
            except Exception as exc:  # record technical fixture failures, do not hide
                row={"status":"FAIL","error":f"{type(exc).__name__}: {exc}"}
            method_rows[method][name]=row

    validations={}
    for method, rows in method_rows.items():
        if method=="pcmci_plus_cmiknn":
            applicable=["null","nonlinear"]
            test_recovery=(rows.get("nonlinear",{}).get("expected_recovery",{}).get("X->Y@1",False))
            null_sparse=rows.get("null",{}).get("edge_count",99)<=2
            valid=all(rows.get(f,{}).get("status")=="PASS" for f in applicable) and test_recovery and null_sparse
            validations[method]={"status":"VALID" if valid else "VALIDATION_FAIL","applicable_fixtures":applicable,
                                 "nonlinear_recovery":bool(test_recovery),"null_edge_count":rows.get("null",{}).get("edge_count")}
        else:
            required=["null","single_direction","chain","bidirectional"]
            passes={f: rows.get(f,{}).get("status")=="PASS" for f in required}
            null_sparse=rows.get("null",{}).get("edge_count",99)<=2
            single=rows.get("single_direction",{}).get("expected_recovery",{}).get("X->Y@1",False)
            chain=all(rows.get("chain",{}).get("expected_recovery",{}).values())
            bidi=all(rows.get("bidirectional",{}).get("expected_recovery",{}).values())
            valid=all(passes.values()) and null_sparse and single and chain and bidi
            validations[method]={"status":"VALID" if valid else "PARTIAL_VALIDATION" if all(passes.values()) and (single or chain or bidi) else "VALIDATION_FAIL",
                                 "applicable_fixtures":required,"fixture_runs_pass":passes,
                                 "null_edge_count":rows.get("null",{}).get("edge_count"),
                                 "single_direction_recovery":bool(single),"chain_recovery":bool(chain),"bidirectional_recovery":bool(bidi)}
    output["methods"]={name:{"validation":validations[name],"fixture_results":rows} for name,rows in method_rows.items()}
    output["all_primary_methods_valid"] = all(
        validations[m]["status"]=="VALID" for m in ["pcmci_plus_parcorr","var_lingam","dynotears","notears_mlp","neural_granger_cmlp"])
    return output

