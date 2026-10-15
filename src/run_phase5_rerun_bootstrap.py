from __future__ import annotations

"""Discovery-only temporal block-bootstrap stability for Phase 5 rerun."""
import json
import time
import hashlib
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from phase5_rerun_common import OUT, VARS, ASSETS, log, write_json, load_asset_discovery
from phase5_rerun_algorithms import pcmci_plus, dynotears, neural_graph

SEED = 8675309
BLOCKS = (42, 84)
REPS = {
    "PCMCI+": 200,
    "DYNOTEARS": 200,
    "NOTEARS-MLP": 100,
    "Neural Granger/cMLP": 100,
}
KEYS = {
    "PCMCI+": "pcmci_results.json",
    "DYNOTEARS": "dynotears_results.json",
    "NOTEARS-MLP": "notears_mlp_results.json",
    "Neural Granger/cMLP": "neural_granger_results.json",
}


def sample_blocks(segments, block_len, rng):
    """Draw moving blocks only within observed contiguous segments.

    Each selected block remains a separate realization, so neither a sampled
    join nor an original data gap creates a synthetic lag transition.
    """
    eligible = [(s, i) for s in segments for i in range(max(0, len(s)-block_len+1))]
    if not eligible:
        raise ValueError(f"No complete {block_len}-observation source blocks")
    target_n = sum(map(len, segments))
    nblocks = target_n // block_len
    if nblocks < 2:
        raise ValueError("Too few blocks for a temporal bootstrap")
    picks = rng.integers(0, len(eligible), size=nblocks)
    return [np.asarray(eligible[int(k)][0][eligible[int(k)][1]:eligible[int(k)][1]+block_len], float)
            for k in picks]


def selected_edges(method, segments, seed):
    edges = []
    if method == "PCMCI+":
        r = pcmci_plus(segments, VARS, pc_alpha=.05, ci_kind="ParCorr", seed=seed)
        for e in r["edges"]:
            edges.append((e["source"], e["target"], int(e["lag"]), float(e["statistic"])))
    elif method == "DYNOTEARS":
        fits = dynotears(segments, lambdas=(.05,))
        f = fits[0]
        if not f["diagnostics"]["converged"]:
            raise RuntimeError("fixed-lambda DYNOTEARS bootstrap fit did not converge")
        a = np.asarray(f["lagged"], float)
        for lag in range(a.shape[0]):
            for i, src in enumerate(VARS):
                for j, dst in enumerate(VARS):
                    w = float(a[lag, i, j])
                    if abs(w) >= .05:
                        edges.append((src, dst, lag+1, w))
    else:
        if method == "NOTEARS-MLP":
            mode, lam, hidden, method_seed = "notears_mlp", .0075, 10, 314159
        else:
            mode, lam, hidden, method_seed = "cmlp", .03, 10, 271828
        f = neural_graph(segments, mode, lambdas=(lam,), hidden_sizes=(hidden,),
                         seed=method_seed, epochs=1000)[0]
        norms = np.asarray(f["lag_norms"], float)
        effects = np.asarray(f["local_derivative"], float)
        for i, src in enumerate(VARS):
            for j, dst in enumerate(VARS):
                for k in range(norms.shape[2]):
                    if norms[i, j, k] >= .10:
                        edges.append((src, dst, k+1, float(effects[i, j, k])))
    return edges


def checkpoint_path(method, asset, block_len):
    slug = method.lower().replace("+", "plus").replace("/", "_").replace(" ", "_")
    return OUT / "bootstrap_checkpoints" / f"{slug}_{asset}_block{block_len}.json"


def bootstrap_one(method, asset, block_len, segments, nrep):
    path = checkpoint_path(method, asset, block_len)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        ck = json.loads(path.read_text())
        if (ck["method"] != method or ck["asset"] != asset or
            ck["block_length"] != block_len or ck["requested_replications"] != nrep):
            raise RuntimeError(f"Checkpoint config mismatch: {path}")
    else:
        ck = {"method": method, "asset": asset, "block_length": block_len,
              "requested_replications": nrep, "seed": SEED,
              "sampling": "moving contiguous blocks, each block a separate segment",
              "valid_replications": [], "failed_replications": []}
    done = {int(x["replication"]) for x in ck["valid_replications"] + ck["failed_replications"]}
    for rep in range(nrep):
        if rep in done:
            continue
        rng = np.random.default_rng(SEED + 100000*ASSETS.index(asset) + 10000*BLOCKS.index(block_len) + 100*list(REPS).index(method) + rep)
        bsegments = sample_blocks(segments, block_len, rng)
        try:
            edges = selected_edges(method, bsegments, SEED + rep)
            ck["valid_replications"].append({"replication": rep,
                "edges": [{"source":s,"target":t,"lag":l,"statistic":v} for s,t,l,v in edges]})
        except Exception as exc:
            ck["failed_replications"].append({"replication":rep,"error":f"{type(exc).__name__}: {exc}"})
        if rep % 5 == 0 or rep == nrep-1:
            write_json(path, ck)
            log(f"bootstrap checkpoint method={method} asset={asset} block={block_len} rep={rep+1}/{nrep} valid={len(ck['valid_replications'])} failed={len(ck['failed_replications'])}")
    write_json(path, ck)
    return ck


def summarize(ck):
    counts = {}
    stats = {}
    for rep in ck["valid_replications"]:
        seen = set()
        for e in rep["edges"]:
            key = (e["source"], e["target"], int(e["lag"]))
            if key in seen:
                continue
            seen.add(key)
            counts[key] = counts.get(key, 0) + 1
            stats.setdefault(key, []).append(float(e["statistic"]))
    denom = len(ck["valid_replications"])
    out = []
    for key in sorted(set(counts) | {(s,t,l) for s in VARS for t in VARS for l in range(1,7)}):
        vals = stats.get(key, [])
        freq = counts.get(key, 0) / denom if denom else None
        signs = np.sign(vals)
        sign_consistency = (float(max(np.mean(signs > 0), np.mean(signs < 0))) if len(signs) else None)
        out.append({"source":key[0],"target":key[1],"lag":key[2],
                    "selection_frequency":freq,"selected_replications":counts.get(key,0),
                    "valid_replications":denom,"failed_replications":len(ck["failed_replications"]),
                    "median_statistic_when_selected":float(np.median(vals)) if vals else None,
                    "sign_consistency":sign_consistency,
                    "stability_class":"VERY_STABLE" if freq is not None and freq>=.85 else
                        "STABLE" if freq is not None and freq>=.70 else "NOT_STABLE"})
    return out


def main():
    fixture = json.loads((OUT/"fixture_validation.json").read_text())
    if any(x["status"] != "VALID" for x in fixture["methods"].values()):
        raise RuntimeError("Synthetic validation gate failed")
    results = {}
    assets = load_asset_discovery()
    with threadpool_limits(limits=1):
        for method in REPS:
            results[method] = {}
            for asset in ASSETS:
                results[method][asset] = {}
                usable = json.loads((OUT/KEYS[method]).read_text())["assets"][asset]["status"] == "FIT_COMPLETE"
                for bl in BLOCKS:
                    if not usable:
                        results[method][asset][str(bl)] = {"status":"SKIPPED_EMPIRICAL_METHOD_GATE"}
                        continue
                    log(f"bootstrap start method={method} asset={asset} block={bl} replications={REPS[method]}")
                    ck = bootstrap_one(method, asset, bl, assets[asset]["segments"], REPS[method])
                    results[method][asset][str(bl)] = {
                        "status":"COMPLETE" if not ck["failed_replications"] else "PARTIAL",
                        "requested_replications":REPS[method],
                        "valid_replications":len(ck["valid_replications"]),
                        "failed_replications":len(ck["failed_replications"]),
                        "seed":SEED,"edge_stability":summarize(ck)}
                    write_json(OUT/"bootstrap_stability.json", {
                        "definition":"frequency of lagged edge selection across contiguous temporal block-bootstrap discovery refits",
                        "thresholds":{"STABLE":.70,"VERY_STABLE":.85},
                        "block_lengths_4h_observations":list(BLOCKS),
                        "bootstrap_status":results})
    write_json(OUT/"bootstrap_stability.json", {
        "definition":"frequency of lagged edge selection across contiguous temporal block-bootstrap discovery refits",
        "thresholds":{"STABLE":.70,"VERY_STABLE":.85},
        "block_lengths_4h_observations":list(BLOCKS),
        "bootstrap_status":results})
    log("ALL BOOTSTRAP METHODS COMPLETE")


if __name__ == "__main__":
    main()
