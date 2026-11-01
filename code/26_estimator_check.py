"""Validate the sampled delta_rel estimator against exact computation.

delta is a supremum, so an m-point subsample underestimates it. The question that
matters for this paper is not whether the estimator is biased (it is) but whether the
bias is stable enough across geometries that comparing two arms by their sampled
estimates is sound. We take frozen ESM-2 layers, which span the range of delta_rel the
paper reports, and compare the 48-point estimate against exact evaluation on the same
gallery."""
import argparse, json
import numpy as np
from scipy import stats
from common import DATA, RESULTS, get_logger, save_json
log = get_logger("estcheck")

def exact_drel(D):
    n = D.shape[0]; best = 0.0
    for w in range(n):
        A = 0.5 * (D[w][:, None] + D[w][None, :] - D)
        M = np.minimum(A[:, :, None], A[None, :, :]).max(axis=1)
        v = float((M - A).max())
        if v > best: best = v
    return 2.0 * best / float(D.max())

def dmat(Z):
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    G = Z @ Z.T
    D = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * G)); np.fill_diagonal(D, 0.0)
    return D

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="35M")
ap.add_argument("--dataset", default="scope")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--m", type=int, default=48)
ap.add_argument("--reps", type=int, default=300)
ap.add_argument("--layers", type=int, nargs="+", default=None)
x = ap.parse_args()

Z_all = np.load(DATA / "emb" / ("%s_%s.npy" % (x.dataset, x.model)), mmap_mode="r")
nL = Z_all.shape[0]
layers = x.layers if x.layers else list(range(0, nL, max(1, nL // 8)))
rng = np.random.default_rng(0)
idx = rng.choice(Z_all.shape[1], size=x.n, replace=False)
rows = []
for L in layers:
    D = dmat(np.asarray(Z_all[L][idx], dtype=np.float64))
    ex = exact_drel(D)
    vals = [exact_drel(D[np.ix_(s, s)]) for s in
            (rng.choice(x.n, size=x.m, replace=False) for _ in range(x.reps))]
    mu, sd = float(np.mean(vals)), float(np.std(vals))
    rows.append({"layer": int(L), "exact": ex, "sampled": mu, "sd": sd,
                 "bias": mu - ex, "ratio": mu / ex})
    log.info("layer %2d  exact=%.4f  sampled(m=%d)=%.4f+-%.4f  bias=%+.4f  ratio=%.3f",
             L, ex, x.m, mu, sd, mu - ex, mu / ex)
e = [r["exact"] for r in rows]; s = [r["sampled"] for r in rows]
sp = stats.spearmanr(e, s); pe = stats.pearsonr(e, s)
log.info("across %d layers: Spearman=%.3f (p=%.2g)  Pearson=%.3f  ratio range [%.3f, %.3f]",
         len(rows), sp.statistic, sp.pvalue, pe.statistic,
         min(r["ratio"] for r in rows), max(r["ratio"] for r in rows))
save_json({"rows": rows, "spearman": float(sp.statistic), "pearson": float(pe.statistic),
           "n": x.n, "m": x.m, "reps": x.reps, "model": x.model},
          RESULTS / ("estcheck_%s_%s.json" % (x.dataset, x.model)))
log.info("ESTCHECK_DONE")
