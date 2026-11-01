"""Does the 48-point estimate of delta_rel preserve the ORDER of two geometries that
differ by the amount our arms differ by (about 0.10 absolute)?

We build metric spaces that interpolate between a tree metric (delta = 0) and random
points on a sphere (large delta), sweep the mixing weight so that exact delta_rel spans
the range the paper reports, and compare exact against the 48-point estimator."""
import numpy as np
from scipy import stats
from common import RESULTS, get_logger, save_json
log = get_logger("estrange")

def exact_drel(D):
    n = D.shape[0]; best = 0.0
    for w in range(n):
        A = 0.5 * (D[w][:, None] + D[w][None, :] - D)
        M = np.minimum(A[:, :, None], A[None, :, :]).max(axis=1)
        v = float((M - A).max())
        if v > best: best = v
    return 2.0 * best / float(D.max())

def tree_metric(n, rng, depth=7):
    """Shortest-path metric of a random binary tree on n leaves."""
    codes = rng.integers(0, 2, size=(n, depth))
    eq = (codes[:, None, :] == codes[None, :, :])
    shared = np.zeros((n, n), dtype=int)
    for d in range(depth):
        shared += np.all(eq[:, :, :d + 1], axis=2)
    D = 2.0 * (depth - shared).astype(float)
    np.fill_diagonal(D, 0.0)
    return D / D.max()

def sphere_metric(n, rng, dim=64):
    Z = rng.normal(size=(n, dim)); Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    D = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * (Z @ Z.T))); np.fill_diagonal(D, 0.0)
    return D / D.max()

rng = np.random.default_rng(0)
N, M, REPS = 200, 48, 300
T, S = tree_metric(N, rng), sphere_metric(N, rng)
rows = []
for a in [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]:
    D = (1 - a) * T + a * S
    D = D / D.max(); np.fill_diagonal(D, 0.0)
    ex = exact_drel(D)
    vals = [exact_drel(D[np.ix_(s, s)]) for s in
            (rng.choice(N, size=M, replace=False) for _ in range(REPS))]
    mu, sd = float(np.mean(vals)), float(np.std(vals))
    rows.append({"mix": a, "exact": ex, "sampled": mu, "sd": sd, "ratio": mu / ex})
    log.info("mix=%.2f  exact=%.4f  sampled=%.4f+-%.4f  ratio=%.3f", a, ex, mu, sd, mu / ex)
e = [r["exact"] for r in rows]; s = [r["sampled"] for r in rows]
sp = stats.spearmanr(e, s); pe = stats.pearsonr(e, s)
log.info("exact range [%.3f, %.3f]  Spearman=%.3f (p=%.2g)  Pearson=%.3f  ratio [%.3f, %.3f]",
         min(e), max(e), sp.statistic, sp.pvalue, pe.statistic,
         min(r["ratio"] for r in rows), max(r["ratio"] for r in rows))
save_json({"rows": rows, "spearman": float(sp.statistic), "pearson": float(pe.statistic)},
          RESULTS / "estimator_range.json")
log.info("ESTRANGE_DONE")
