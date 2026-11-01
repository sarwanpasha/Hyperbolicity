"""Gromov four-point / delta-hyperbolicity utilities (numpy + torch)."""
import numpy as np
import torch


def gromov_products(D, base=0):
    """D: (B,m,m) pairwise distances. Returns Gromov product matrix wrt row `base`.
    (x|y)_w = 0.5 (d(x,w) + d(y,w) - d(x,y))."""
    dw = D[:, base, :]                       # (B,m)
    return 0.5 * (dw[:, :, None] + dw[:, None, :] - D)


def delta_from_D(D, base=0, chunk=64):
    """Batched Gromov delta wrt a base point using the max-min matrix product.
    delta = max_{x,y,z} [ min((x|y)_w,(y|z)_w) - (x|z)_w ].
    D: (B,m,m) numpy float32. Returns (B,) float32."""
    B = D.shape[0]
    out = np.zeros(B, dtype=np.float32)
    for s in range(0, B, chunk):
        A = gromov_products(D[s:s + chunk], base)          # (b,m,m)
        # max-min product: M[i,k] = max_j min(A[i,j], A[j,k])
        M = np.minimum(A[:, :, :, None], A[:, None, :, :]).max(axis=2)
        out[s:s + chunk] = (M - A).max(axis=(1, 2))
    return out


def delta_torch(D, base=0):
    """Differentiable Gromov delta surrogate ingredients (torch).
    D: (B,m,m). Returns delta (B,) using hard max/min (non-differentiable at ties)."""
    dw = D[:, base, :]
    A = 0.5 * (dw[:, :, None] + dw[:, None, :] - D)
    M = torch.minimum(A[:, :, :, None], A[:, None, :, :]).amax(dim=2)
    return (M - A).amax(dim=(1, 2))


def pdist_normalized(X):
    """X: (B,m,d) -> Euclidean distances on L2-normalized rows, (B,m,m)."""
    Xn = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)
    G = Xn @ np.swapaxes(Xn, -1, -2)
    d2 = np.clip(2.0 - 2.0 * G, 0.0, None)
    return np.sqrt(d2, dtype=np.float32)


def ultrametricity(D):
    """Fraction of triples that are (approximately) isosceles with the two long
    sides equal - a classical tree-likeness statistic. D:(B,m,m) -> (B,)"""
    B, m, _ = D.shape
    out = np.zeros(B, dtype=np.float32)
    iu = np.triu_indices(m, 1)
    for b in range(B):
        d = D[b]
        vals = []
        idx = np.arange(m)
        rng = np.random.default_rng(0)
        tri = rng.choice(m, size=(min(2000, m * m), 3))
        tri = tri[(tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])]
        a = d[tri[:, 0], tri[:, 1]]; bb = d[tri[:, 1], tri[:, 2]]; c = d[tri[:, 0], tri[:, 2]]
        s = np.sort(np.stack([a, bb, c], 1), axis=1)
        denom = np.clip(s[:, 2], 1e-8, None)
        out[b] = float(np.mean((s[:, 2] - s[:, 1]) / denom))
    return out
