"""Validated Phase-5 rerun estimators.

This module keeps temporal methods in named, auditable functions. DYNOTEARS,
temporal NOTEARS-MLP, and cMLP use reference formulations documented in the
rerun manifest; missing PyPI distributions are not replaced by unrelated
estimators.
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.stats import chi2
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import roc_auc_score

LAGS = 6
LAMBDAS = (0.01, 0.05, 0.10, 0.20)
EDGE_EPS = 1e-3


def make_lagged(segments: Sequence[np.ndarray], max_lag: int = LAGS):
    """Build current and lagged matrices without ever crossing a data gap."""
    ys, xs = [], []
    for a in segments:
        a = np.asarray(a, float)
        if len(a) <= max_lag:
            continue
        for t in range(max_lag, len(a)):
            ys.append(a[t])
            xs.append(np.concatenate([a[t - lag] for lag in range(1, max_lag + 1)]))
    if not ys:
        raise ValueError("No valid lagged rows")
    return np.asarray(ys), np.asarray(xs)


def _reshape_wa(wa_vec: np.ndarray, d: int, p: int):
    w_tilde = wa_vec.reshape([2 * (p + 1) * d, d])
    w_mat = w_tilde[:d] - w_tilde[d:2 * d]
    a_plus = w_tilde[2*d:].reshape(2*p, d*d)[::2].reshape(d*p, d)
    a_minus = w_tilde[2*d:].reshape(2*p, d*d)[1::2].reshape(d*p, d)
    return w_mat, a_plus - a_minus


def _dynotears_fit(x: np.ndarray, xlags: np.ndarray, lam: float,
                   max_outer: int = 30, max_inner: int = 250,
                   h_tol: float = 1e-8):
    """CausalNex DYNOTEARS reference objective, simplified to array inputs.

    Orientations follow the reference implementation: W[source,target] and
    A[lag-major source,target]. Intra-slice self loops are fixed to zero.
    """
    n, d = x.shape
    p = xlags.shape[1] // d
    bounds_w = 2 * [
        (0, 0) if i == j else (0, None)
        for i in range(d) for j in range(d)
    ]
    bounds_a = 2 * [(0, None) for _ in range(p * d * d)]
    bounds = bounds_w + bounds_a
    rho, alpha, hval, w = 1.0, 0.0, np.inf, np.zeros(2*(p+1)*d*d)

    def objective_gradient(v):
        wm, am = _reshape_wa(v, d, p)
        e = slin.expm(wm * wm)
        h = float(np.trace(e) - d)
        resid = x @ (np.eye(d) - wm) - xlags @ am
        value=(0.5 / n * np.square(np.linalg.norm(resid, "fro"))
               + 0.5 * rho * h * h + alpha * h + lam * np.abs(v).sum())
        grad_w = -x.T @ resid / n + (rho * (np.trace(e)-d) + alpha) * e.T * wm * 2
        grad_a = -xlags.T @ resid / n
        # Positive/negative variable split as in the published reference.
        gv = np.concatenate([grad_w.ravel(), -grad_w.ravel()]) + lam
        ga = grad_a.reshape(p, d*d)
        ga = np.concatenate([ga, -ga], axis=1).ravel() + lam
        return value, np.concatenate([gv, ga])

    converged = False
    outer_iter = 0
    for outer_iter in range(max_outer):
        hnew = np.inf
        while rho < 1e20 and (hnew > 0.25*hval or hnew == np.inf):
            res = sopt.minimize(objective_gradient, w, method="L-BFGS-B", jac=True,
                                bounds=bounds, options={"maxiter": max_inner, "ftol": 1e-10})
            wnew = res.x
            wm_new, _am_new = _reshape_wa(wnew,d,p)
            hnew = float(np.trace(slin.expm(wm_new*wm_new))-d)
            if hnew > 0.25*hval:
                rho *= 10
            else:
                break
        w, hval = wnew, hnew
        alpha += rho*hval
        if hval <= h_tol:
            converged = True
            break
    wm, am = _reshape_wa(w, d, p)
    return wm, am.reshape(p, d, d), {
        "converged": bool(converged), "h_acyclicity": float(hval),
        "outer_iterations": outer_iter + 1, "objective": float(objective_gradient(w)[0]),
        "optimizer_success": bool(res.success), "optimizer_message": str(res.message),
    }


def dynotears(segments: Sequence[np.ndarray], lambdas=LAMBDAS):
    y, xlag = make_lagged(segments)
    d = y.shape[1]
    fits = []
    for lam in lambdas:
        w, a, diag = _dynotears_fit(y, xlag, float(lam))
        fits.append({"lambda": float(lam), "contemporaneous": w,
                     "lagged": a, "diagnostics": diag})
    return fits


def _bh(pvalues):
    p = np.asarray(pvalues, float).ravel()
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n, float)
    q[order] = np.minimum.accumulate((p[order] * n / np.arange(1, n+1))[::-1])[::-1]
    return np.clip(q, 0, 1).reshape(np.asarray(pvalues).shape)


def pcmci_plus(segments: Sequence[np.ndarray], names: Sequence[str], pc_alpha=.05,
               ci_kind="ParCorr", seed=314159):
    from tigramite.pcmci import PCMCI
    from tigramite import data_processing as pp
    if ci_kind == "ParCorr":
        from tigramite.independence_tests.parcorr import ParCorr
        ci = ParCorr(significance="analytic")
    elif ci_kind == "CMIknn":
        from tigramite.independence_tests.cmiknn import CMIknn
        ci = CMIknn(knn=0.2, significance="shuffle_test", sig_samples=100,
                    workers=1, seed=seed)
    else:
        raise ValueError(ci_kind)
    data = {i: np.asarray(seg, float) for i, seg in enumerate(segments) if len(seg) > 12}
    if not data:
        raise ValueError("No complete segment long enough for PCMCI+")
    frame = pp.DataFrame(data=data, var_names=list(names),
                         analysis_mode="multiple" if len(data)>1 else "single")
    pcmci = PCMCI(dataframe=frame, cond_ind_test=ci, verbosity=0)
    res = pcmci.run_pcmciplus(tau_min=1, tau_max=LAGS, pc_alpha=pc_alpha)
    pmat, vmat = np.asarray(res["p_matrix"]), np.asarray(res["val_matrix"])
    qmat = np.full_like(pmat, np.nan, dtype=float)
    lagged = pmat[:, :, 1:LAGS+1]
    qmat[:, :, 1:LAGS+1] = _bh(lagged)
    edges = []
    for i, src in enumerate(names):
        for j, dst in enumerate(names):
            for lag in range(1, min(LAGS, pmat.shape[2]-1)+1):
                if np.isfinite(qmat[i,j,lag]) and qmat[i,j,lag] < .05:
                    edges.append({"source":src,"target":dst,"lag":lag,
                                  "statistic":float(vmat[i,j,lag]),"p":float(pmat[i,j,lag]),
                                  "q_bh":float(qmat[i,j,lag]),
                                  "graph_mark":str(res["graph"][i,j,lag])})
    return {"pc_alpha":pc_alpha,"ci_test":ci_kind,"edges":edges,
            "p_matrix":pmat,"val_matrix":vmat,"q_matrix":qmat,
            "n_segments":len(data),"n_observations":int(sum(len(x) for x in data.values()))}


def stationarity_audit(segments: Sequence[np.ndarray], names: Sequence[str]):
    x = np.concatenate(segments, axis=0)
    rows=[]
    for j,name in enumerate(names):
        s=x[:,j]
        try: adf_p=float(adfuller(s,autolag="AIC")[1])
        except Exception: adf_p=None
        try: kpss_p=float(kpss(s,regression="c",nlags="auto")[1])
        except Exception: kpss_p=None
        rows.append({"variable":name,"adf_p":adf_p,"kpss_p":kpss_p,
                     "stationary_gate":bool(adf_p is not None and kpss_p is not None and adf_p<.05 and kpss_p>.05)})
    return rows


def _bic_lag(segments: Sequence[np.ndarray], max_lag=LAGS):
    best=(np.inf,None)
    d=segments[0].shape[1]
    for p in range(1,max_lag+1):
        ys=[]; xs=[]
        for s in segments:
            for t in range(p,len(s)):
                ys.append(s[t]); xs.append(np.r_[1., *[s[t-k] for k in range(1,p+1)]])
        y,x=np.asarray(ys),np.asarray(xs)
        if len(y)<=x.shape[1]+d: continue
        coef=np.linalg.lstsq(x,y,rcond=None)[0]
        r=y-x@coef
        cov=r.T@r/len(r)
        sign,logdet=np.linalg.slogdet(cov+np.eye(d)*1e-12)
        if sign<=0: continue
        bic=len(y)*logdet + (d*x.shape[1])*np.log(len(y))
        if bic<best[0]: best=(float(bic),p)
    if best[1] is None: raise ValueError("BIC lag selection failed")
    return best[1],best[0]


def var_lingam(segments: Sequence[np.ndarray], names: Sequence[str], seed=314159):
    from lingam import VARLiNGAM
    stationary = stationarity_audit(segments, names)
    if not all(x["stationary_gate"] for x in stationary):
        d=len(names)
        return {"lag_order_bic":None,"bic":None,"segment_diagnostics":[],
                "valid_segment_fits":0,"eligible_segments":0,"diagnostic_pass":False,
                "stationarity":stationary,
                "diagnostic_failure":"stationarity_gate_failed",
                "contemporaneous":np.zeros((d,d)),"lagged":np.zeros((LAGS,d,d)),
                "lagged_segment_frequency":np.zeros((LAGS,d,d))}
    p,bic=_bic_lag(segments)
    d=len(names); fits=[]; diagnostics=[]
    for idx,s in enumerate(segments):
        if len(s)<max(100,12*(d*p+1)): continue
        try:
            model=VAR(s).fit(p,trend="c")
            stable=bool(model.is_stable())
            whiteness=model.test_whiteness(nlags=min(max(p+8,12),len(s)//4),adjusted=True)
            white_p=float(whiteness.pvalue)
            diag_ok=stable and white_p>=.01
            diag={"segment":idx,"n":len(s),"stable":stable,"whiteness_p":white_p,"diagnostic_pass":diag_ok}
            diagnostics.append(diag)
            if not diag_ok: continue
            ling=VARLiNGAM(lags=p,criterion="bic",prune=True,random_state=seed+idx).fit(s)
            mats=np.asarray(ling.adjacency_matrices_,float)
            if mats.shape[0] != p+1: raise RuntimeError(f"Unexpected VARLiNGAM adjacency shape {mats.shape}")
            fits.append(mats)
        except Exception as exc:
            diagnostics.append({"segment":idx,"n":len(s),"diagnostic_pass":False,"error":f"{type(exc).__name__}: {exc}"})
    if fits:
        # VAR-LiNGAM uses adjacency[target,source], with index 0 contemporaneous.
        med=np.median(np.stack(fits),axis=0)
        freq=np.mean(np.abs(np.stack(fits))>EDGE_EPS,axis=0)
    else:
        med=np.zeros((p+1,d,d)); freq=np.zeros_like(med)
    passes=sum(bool(x.get("diagnostic_pass")) for x in diagnostics)
    eligible=sum(1 for s in segments if len(s)>=max(100,12*(d*p+1)))
    gate=eligible>=1 and passes>=max(1,int(np.ceil(.5*eligible))) and len(fits)>=1
    # Return the lagged adjacency oriented source -> target for common output.
    lagged=np.transpose(med[1:],(0,2,1))
    lagfreq=np.transpose(freq[1:],(0,2,1))
    contemporaneous=med[0].T
    return {"lag_order_bic":int(p),"bic":float(bic),"segment_diagnostics":diagnostics,
            "valid_segment_fits":len(fits),"eligible_segments":eligible,"diagnostic_pass":bool(gate),
            "stationarity":stationary,"contemporaneous":contemporaneous,"lagged":lagged,"lagged_segment_frequency":lagfreq}


def _train_target_mlp(segments, lam, hidden, mode, seed, epochs=400, lr=.02):
    """Temporal DAG MLP / component-wise cMLP with proximal group sparsity."""
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32-1))
    y_np,x_np=make_lagged(segments)
    n,d=y_np.shape; k=LAGS
    x_np=x_np.reshape(n,k,d)
    dtype=torch.float32
    xt=torch.as_tensor(x_np,dtype=dtype); yt=torch.as_tensor(y_np,dtype=dtype)
    h=int(hidden)
    # Per-target first-layer weights, preserving lag/source groups.
    w1=torch.nn.Parameter(torch.randn(d,h,k,d,dtype=dtype)*.025)
    b1=torch.nn.Parameter(torch.zeros(d,h,dtype=dtype))
    w2=torch.nn.Parameter(torch.randn(d,h,dtype=dtype)*.025)
    b2=torch.nn.Parameter(torch.zeros(d,dtype=dtype))
    params=[w1,b1,w2,b2]
    # The reference cMLP uses proximal gradient (ISTA), not Adam followed by a
    # shrinkage heuristic. Plain full-batch SGD keeps the proximal update valid.
    opt=torch.optim.SGD(params,lr=lr)
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        pre=torch.einsum("nkd,ohkd->noh",xt,w1)+b1[None,:,:]
        if mode=="notears_mlp": act=torch.sigmoid(pre)
        else: act=torch.relu(pre)
        pred=(act*w2[None,:,:]).sum(dim=2)+b2[None,:]
        # Sum target-wise mean squared errors, matching the reference cMLP
        # objective rather than dividing the sparsity penalty by dimension.
        loss=torch.mean((pred-yt)**2)*d
        loss.backward(); opt.step()
        with torch.no_grad():
            if mode=="notears_mlp":
                # Temporal ordering makes all allowed edges point from a past
                # node to a current node; this is an acyclic constrained MLP.
                norms=torch.linalg.vector_norm(w1,dim=1,keepdim=True)
                factor=torch.clamp(1-lr*lam/(norms+1e-12),min=0)
                w1.mul_(factor)
            else:
                # cMLP group lasso: one group per (target,source), all lags.
                norms=torch.linalg.vector_norm(w1,dim=(1,2),keepdim=True)
                factor=torch.clamp(1-lr*lam/(norms+1e-12),min=0)
                w1.mul_(factor)
    with torch.no_grad():
        group_lag=torch.linalg.vector_norm(w1,dim=1).cpu().numpy() # target,lag,source
        group_total=np.linalg.norm(group_lag,axis=1) # target,source
        # Local finite-difference sign; nonlinear effects need not have a
        # globally constant sign, so this is only a descriptive statistic.
        base=pred.detach()
        delta=.05
        signs=np.zeros((d,k,d),float)
        for lag in range(k):
            xp=xt.clone(); xp[:,lag,:]+=delta
            ppred=( (torch.sigmoid(torch.einsum("nkd,ohkd->noh",xp,w1)+b1[None,:,:]) if mode=="notears_mlp" else torch.relu(torch.einsum("nkd,ohkd->noh",xp,w1)+b1[None,:,:])) * w2[None,:,:]).sum(dim=2)+b2[None,:]
            signs[:,lag,:]=((ppred-base).mean(dim=0).cpu().numpy()/delta)
    return {"lag_norms":group_lag.transpose(2,0,1), # source,target,lag
            "total_norms":group_total.T, # source,target
            "local_derivative":signs.transpose(2,0,1), # source,target,lag
            "final_loss":float(loss.detach().cpu()),"n":int(n),"hidden":h,"lambda":float(lam)}


def neural_graph(segments: Sequence[np.ndarray], mode: str,
                 lambdas=LAMBDAS, hidden_sizes=(10,20), seed=314159,
                 epochs=1000):
    fits=[]
    if mode=="cmlp": hidden_sizes=(10,)
    for h in hidden_sizes:
        for lam in lambdas:
            fits.append(_train_target_mlp(segments,lam,h,
                         "neural_granger" if mode=="cmlp" else "notears_mlp",
                         seed+int(1000*h+100*lam),epochs=epochs))
    return fits


def synthetic_fixtures(n=1500, seed=20261007):
    out={}
    rng=np.random.default_rng(seed)
    # Independent stationary AR(1) null.
    e=rng.laplace(size=(n,3)); x=np.zeros((n,3))
    for t in range(1,n): x[t]=.35*x[t-1]+e[t]
    out["null"]=(x,np.zeros((3,3),int))
    # Single direction X -> Y at one step.
    e=rng.laplace(size=(n,3)); x=np.zeros((n,3))
    for t in range(1,n):
        x[t,0]=.35*x[t-1,0]+e[t,0]
        x[t,1]=.4*x[t-1,1]+.85*x[t-1,0]+e[t,1]
        x[t,2]=.35*x[t-1,2]+e[t,2]
    truth=np.zeros((3,3),int);truth[0,1]=1
    out["single_direction"]=(x,truth)
    # Chain X -> Y -> Z.
    e=rng.laplace(size=(n,3));x=np.zeros((n,3))
    for t in range(1,n):
        x[t,0]=.3*x[t-1,0]+e[t,0]
        x[t,1]=.35*x[t-1,1]+.8*x[t-1,0]+e[t,1]
        x[t,2]=.35*x[t-1,2]+.8*x[t-1,1]+e[t,2]
    truth=np.zeros((3,3),int);truth[0,1]=1;truth[1,2]=1
    out["chain"]=(x,truth)
    # Stable bidirectional lag-one system.
    e=rng.laplace(size=(n,3));x=np.zeros((n,3))
    for t in range(1,n):
        x[t,0]=.25*x[t-1,0]+.55*x[t-1,1]+e[t,0]
        x[t,1]=.25*x[t-1,1]+.55*x[t-1,0]+e[t,1]
        x[t,2]=.35*x[t-1,2]+e[t,2]
    truth=np.zeros((3,3),int);truth[0,1]=truth[1,0]=1
    out["bidirectional"]=(x,truth)
    # Zero-linear-covariance nonlinear driver, plus target persistence.
    e=rng.normal(size=(n,3));x=np.zeros((n,3))
    x[:,0]=rng.normal(size=n)
    for t in range(1,n):
        x[t,1]=.6*x[t-1,1]+.8*(np.abs(x[t-1,0])-np.sqrt(2/np.pi))+e[t,1]
        x[t,2]=.35*x[t-1,2]+e[t,2]
    truth=np.zeros((3,3),int);truth[0,1]=1
    out["nonlinear"]=(x,truth)
    # Standardize columns separately inside each synthetic fixture.
    for key,(a,t) in list(out.items()):
        a=(a-a.mean(0))/a.std(0,ddof=1)
        out[key]=(a,t)
    return out
