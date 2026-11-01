"""How does the reported statistic relate to the one the loss minimizes?

drel is base-pointed and normalized by the diameter of a 48-point neighbourhood; the
loss acts on a base-point-free slack normalized by the mean pairwise distance of a
quadruple. This computes both on the same galleries across frozen ESM-2 layers.
"""
import numpy as np
from scipy import stats
from common import DATA, RESULTS, get_logger, save_json
log = get_logger("surrogate")

def dmat(Z):
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    D = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * (Z @ Z.T)))
    np.fill_diagonal(D, 0.0)
    return D

def drel_sampled(D, rng, m=48, reps=200):
    n = D.shape[0]; out = []
    for _ in range(reps):
        s = rng.choice(n, size=m, replace=False)
        A = D[np.ix_(s, s)]; best = 0.0
        for w in range(m):
            B = 0.5 * (A[w][:, None] + A[w][None, :] - A)
            Mx = np.minimum(B[:, :, None], B[None, :, :]).max(axis=1)
            v = float((Mx - B).max())
            if v > best: best = v
        out.append(2.0 * best / float(A.max()))
    return float(np.mean(out))

def srel_quads(D, rng, nq=200000):
    n = D.shape[0]
    q = rng.integers(0, n, size=(int(nq * 1.6), 4))
    ok = np.ones(len(q), dtype=bool)
    for i in range(4):
        for j in range(i + 1, 4):
            ok &= q[:, i] != q[:, j]
    q = q[ok][:nq]
    a, b, c, e = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    S = np.stack([D[a, b] + D[c, e], D[a, c] + D[b, e], D[a, e] + D[b, c]], axis=1)
    S.sort(axis=1)
    v = (0.5 * (S[:, 2] - S[:, 1])) / (0.5 * S.mean(axis=1) + 1e-8)
    k = max(1, int(round(0.05 * len(v))))
    return float(v.mean()), float(np.sort(v)[-k:].mean()), float(v.max())

if __name__ == "__main__":
    rows = []
    for M in ("35M", "150M", "650M"):
        Z = np.load(DATA / "emb" / ("scope_%s.npy" % M), mmap_mode="r")
        rng = np.random.default_rng(0)
        idx = rng.choice(Z.shape[1], size=400, replace=False)
        for L in range(0, Z.shape[0], max(1, Z.shape[0] // 8)):
            D = dmat(np.asarray(Z[L][idx], dtype=np.float64))
            dr = drel_sampled(D, np.random.default_rng(L))
            sm, st, sx = srel_quads(D, np.random.default_rng(L))
            rows.append({"model": M, "layer": int(L), "drel": dr,
                         "srel_mean": sm, "srel_top5": st, "srel_max": sx})
            log.info("%s L%02d drel=%.4f mean=%.4f top5=%.4f max=%.4f",
                     M, L, dr, sm, st, sx)
    x = [r["drel"] for r in rows]; out = {"rows": rows}
    for k in ("srel_mean", "srel_top5", "srel_max"):
        y = [r[k] for r in rows]
        sp = stats.spearmanr(x, y); pe = stats.pearsonr(x, y)
        sl, ic = np.polyfit(x, y, 1)
        out[k] = {"spearman": float(sp.statistic), "pearson": float(pe.statistic),
                  "slope": float(sl), "intercept": float(ic)}
        log.info("drel vs %s: Spearman=%.3f Pearson=%.3f slope=%.3f",
                 k, sp.statistic, pe.statistic, sl)
    save_json(out, RESULTS / "surrogate_vs_reported.json")
    log.info("SURROGATE_DONE")
