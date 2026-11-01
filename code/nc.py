"""Neural-collapse statistics on a trained representation.

NC1 measures within-class variability collapse, NC2 the simplex equiangular
tight frame structure of the class means, and NC4 agreement between the
nearest class mean and the label. NC3, self-duality, compares class means with
the rows of a linear classifier; a contrastive objective has no classifier, so
NC3 is undefined here and we report no substitute for it.
"""
import numpy as np


def nc_stats(Z, labels, min_per_class=3):
    Z = np.asarray(Z, dtype=np.float64)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    _, inv = np.unique(labels, return_inverse=True)
    cnt = np.bincount(inv)
    keep = np.where(cnt >= min_per_class)[0]
    if len(keep) < 3:
        return {}
    mask = np.isin(inv, keep)
    Z = Z[mask]
    _, inv = np.unique(inv[mask], return_inverse=True)
    C = int(inv.max()) + 1
    d = Z.shape[1]
    mu = np.stack([Z[inv == c].mean(0) for c in range(C)])
    M = mu - mu.mean(0)
    Sw = np.zeros((d, d))
    for c in range(C):
        X = Z[inv == c] - mu[c]
        Sw += X.T @ X
    Sw /= len(Z)
    Sb = M.T @ M / C
    nc1 = float(np.trace(Sw @ np.linalg.pinv(Sb)) / C)
    nrm = np.linalg.norm(M, axis=1)
    Mn = M / (nrm[:, None] + 1e-12)
    G = Mn @ Mn.T
    iu = np.triu_indices(C, 1)
    cos = G[iu]
    nc4 = float(((Z @ mu.T).argmax(1) == inv).mean())
    return {"nc1": nc1,
            "nc2_equinorm_cv": float(nrm.std() / (nrm.mean() + 1e-12)),
            "nc2_cos_std": float(cos.std()),
            "nc2_etf_gap": float(np.abs(cos + 1.0 / (C - 1)).mean()),
            "nc4_ncm_agreement": nc4,
            "n_classes": C, "n_points": int(len(Z))}
