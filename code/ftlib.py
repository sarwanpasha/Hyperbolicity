"""Shared training/eval helpers for the intervention experiments."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from geom import delta_from_D, pdist_normalized


class Head(nn.Module):
    def __init__(self, din, dout=256, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, hidden), nn.GELU(), nn.Linear(hidden, dout))

    def forward(self, x):
        return self.net(x)


class PKSampler:
    """P superfamilies x K domains per batch (needed for a contrastive task loss)."""
    def __init__(self, labels, P=32, K=4, seed=0, steps=200):
        self.by = {}
        for i, l in enumerate(labels):
            self.by.setdefault(l, []).append(i)
        self.keys = [k for k, v in self.by.items() if len(v) >= 2]
        self.P, self.K, self.steps = P, K, steps
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        for _ in range(self.steps):
            cls = self.rng.choice(len(self.keys), size=min(self.P, len(self.keys)), replace=False)
            idx = []
            for c in cls:
                pool = self.by[self.keys[c]]
                take = self.rng.choice(len(pool), size=min(self.K, len(pool)),
                                       replace=len(pool) < self.K)
                idx.extend([pool[t] for t in take])
            yield np.array(idx)

    def __len__(self):
        return self.steps


class FoldPKSampler:
    """P superfamilies x K domains, but the P superfamilies are drawn from as few
    folds as possible. Every negative pair in a batch is then same-fold, which is the
    condition the mechanism experiment needs."""
    def __init__(self, labels, folds, P=16, K=4, seed=0, steps=200):
        self.by = {}
        for i, l in enumerate(labels):
            self.by.setdefault(l, []).append(i)
        self.keys = [k for k, v in self.by.items() if len(v) >= 2]
        sf2fold = {}
        for l, f in zip(labels, folds):
            sf2fold[l] = f
        self.fold2sf = {}
        for k in self.keys:
            self.fold2sf.setdefault(sf2fold[k], []).append(k)
        self.folds = sorted(self.fold2sf.keys(), key=lambda f: -len(self.fold2sf[f]))
        self.P, self.K, self.steps = P, K, steps
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        for _ in range(self.steps):
            cls = []
            while len(cls) < min(self.P, len(self.keys)):
                f = self.folds[self.rng.integers(0, len(self.folds))]
                pool = self.fold2sf[f]
                take = list(self.rng.permutation(pool))[: self.P - len(cls)]
                cls.extend(take)
            idx = []
            for c in cls:
                pool = self.by[c]
                take = self.rng.choice(len(pool), size=min(self.K, len(pool)),
                                       replace=len(pool) < self.K)
                idx.extend([pool[t] for t in take])
            yield np.array(idx)

    def __len__(self):
        return self.steps


def retrieval_metrics(Z, labels, K=5, device="cuda", return_arrays=False):
    X = F.normalize(torch.tensor(np.asarray(Z, dtype=np.float32)), dim=1).to(device)
    N = X.shape[0]
    labels = np.asarray(labels)
    top1 = np.zeros(N, dtype=bool); top5 = np.zeros(N, dtype=bool)
    for s in range(0, N, 1024):
        e = min(s + 1024, N)
        S = X[s:e] @ X.T
        S[torch.arange(e - s, device=X.device), torch.arange(s, e, device=X.device)] = -2
        idx = torch.topk(S, K, dim=1).indices.cpu().numpy()
        lab = labels[idx]
        top1[s:e] = lab[:, 0] == labels[s:e]
        top5[s:e] = (lab == labels[s:e][:, None]).any(1)
    if return_arrays:
        return float(top1.mean()), float(top5.mean()), top1, top5
    return float(top1.mean()), float(top5.mean())


def measure_delta(Z, n=2000, m=48, seed=0):
    """Relative Gromov delta averaged over random m-point subsets."""
    rng = np.random.default_rng(seed)
    Zf = np.asarray(Z, dtype=np.float32)
    idx = rng.integers(0, Zf.shape[0], size=(n, m))
    D = pdist_normalized(Zf[idx])
    d = delta_from_D(D, base=0)
    diam = D.max(axis=(1, 2))
    return float(np.mean(d)), float(np.mean(2 * d / np.clip(diam, 1e-6, None)))


def poincare_supcon(Z, labels, c=1.0, temp=0.1, eps=1e-5):
    """Hyperbolic-head baseline: expmap at the origin, contrastive on -d_hyp."""
    n = Z.norm(dim=-1, keepdim=True).clamp_min(eps)
    sc = math.sqrt(c)
    X = torch.tanh(sc * n) * Z / (sc * n)
    X = X * (1 - 1e-4)
    x2 = (X * X).sum(-1)
    num = torch.cdist(X, X, p=2).pow(2)
    den = ((1 - c * x2)[:, None] * (1 - c * x2)[None, :]).clamp_min(eps)
    Dh = torch.acosh((1 + 2 * c * num / den).clamp_min(1 + 1e-7)) / sc
    S = -Dh / temp
    m = S.shape[0]
    eye = torch.eye(m, dtype=torch.bool, device=Z.device)
    S = S.masked_fill(eye, -1e9)
    pos = (labels[:, None] == labels[None, :]) & (~eye)
    logp = S - torch.logsumexp(S, dim=1, keepdim=True)
    npos = pos.sum(1); valid = npos > 0
    if valid.sum() == 0:
        return Z.sum() * 0.0
    return -((logp * pos).sum(1)[valid] / npos[valid]).mean()
