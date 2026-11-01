"""Differentiable delta-hyperbolicity regularizer and companion losses.

The Gromov four-point condition states that a metric space is delta-hyperbolic if
for every quadruple (a,b,c,d) the two largest of
    S1 = d(a,b)+d(c,d),  S2 = d(a,c)+d(b,d),  S3 = d(a,d)+d(b,c)
differ by at most 2*delta. We therefore define the per-quadruple slack
    delta(a,b,c,d) = 0.5 * (S_(1) - S_(2))          [sorted descending]
and penalise its scale-free version delta / (mean pairwise distance). The sort is
a permutation, so gradients flow everywhere except on a measure-zero tie set; we
also provide a temperature-smoothed variant for ablations.
"""
import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-8


def pairwise_dist(Z, normalize=True):
    X = F.normalize(Z, dim=-1) if normalize else Z
    D = torch.cdist(X, X, p=2)
    return D.clamp_min(0)


def sample_quads(B, Q, device, gen=None):
    """Sample Q index quadruples with four distinct members."""
    out = torch.randint(0, B, (int(Q * 1.6), 4), device=device, generator=gen)
    ok = torch.ones(out.shape[0], dtype=torch.bool, device=device)
    for i in range(4):
        for j in range(i + 1, 4):
            ok &= out[:, i] != out[:, j]
    out = out[ok]
    if out.shape[0] < Q:
        return out
    return out[:Q]


def quad_slack(D, quads, tau=0.0):
    """Returns (delta, delta_rel) per quadruple. tau>0 gives a smoothed top-2 gap."""
    a, b, c, d = quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3]
    s1 = D[a, b] + D[c, d]
    s2 = D[a, c] + D[b, d]
    s3 = D[a, d] + D[b, c]
    S = torch.stack([s1, s2, s3], dim=1)
    if tau > 0:
        m1 = tau * torch.logsumexp(S / tau, dim=1)
        w = torch.softmax(S / tau, dim=1)
        m2 = tau * torch.logsumexp((S - 1e4 * w) / tau, dim=1)
        gap = (m1 - m2).clamp_min(0)
    else:
        top2 = torch.topk(S, 2, dim=1).values
        gap = top2[:, 0] - top2[:, 1]
    delta = 0.5 * gap
    scale = 0.5 * S.mean(dim=1) + EPS      # ~ mean pairwise distance in the quad
    return delta, delta / scale


def hyperbolicity_loss(Z, n_quads=1024, tau=0.0, target=0.0, gen=None,
                       agg="topq", topq=0.05, lse_temp=0.05):
    """Penalty on the scale-free four-point slack of sampled quadruples.

    Gromov delta is a *supremum* over quadruples, so penalising the sample mean
    barely moves it: the mean can be driven to ~0 while the tail that defines delta
    is untouched. `agg` selects what is actually penalised:
      mean  - the sample mean (ablation; what a naive implementation does)
      topq  - the mean of the worst `topq` fraction of quadruples (default)
      lse   - a temperature-smoothed maximum, log-sum-exp over quadruples
    """
    D = pairwise_dist(Z)
    q = sample_quads(Z.shape[0], n_quads, Z.device, gen)
    if q.numel() == 0:
        return Z.sum() * 0.0, {}
    _, drel = quad_slack(D, q, tau)
    excess = torch.clamp(drel - target, min=0)
    if agg == "mean":
        loss = excess.pow(2).mean()
    elif agg == "lse":
        loss = (lse_temp * torch.logsumexp(excess / lse_temp, dim=0)
                - lse_temp * float(np.log(excess.numel()))).pow(2)
    else:
        k = max(1, int(round(topq * excess.numel())))
        loss = torch.topk(excess, k).values.pow(2).mean()
    with torch.no_grad():
        st = {"delta_rel_mean": float(drel.mean()),
              "delta_rel_p95": float(drel.quantile(0.95)),
              "delta_rel_max": float(drel.max())}
    return loss, st


def anticollapse_loss(Z, min_dist=1.0, unif_w=0.0):
    """Keeps the representation from shrinking to a point (which is trivially
    0-hyperbolic). Penalises mean pairwise distance below `min_dist` and adds a
    uniformity term on the unit sphere."""
    D = pairwise_dist(Z)
    n = D.shape[0]
    iu = torch.triu_indices(n, n, offset=1, device=Z.device)
    dv = D[iu[0], iu[1]]
    hinge = torch.clamp(min_dist - dv.mean(), min=0)
    loss = hinge
    if unif_w > 0:
        loss = loss + unif_w * torch.log(torch.exp(-2.0 * dv.pow(2)).mean() + EPS)
    with torch.no_grad():
        st = {"mean_pdist": float(dv.mean()), "std_pdist": float(dv.std())}
    return loss, st


def supcon_loss(Z, labels, temp=0.1):
    """Supervised contrastive loss on L2-normalised embeddings."""
    X = F.normalize(Z, dim=-1)
    S = X @ X.t() / temp
    n = S.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=Z.device)
    S = S.masked_fill(eye, -1e9)
    pos = (labels[:, None] == labels[None, :]) & (~eye)
    logZ = torch.logsumexp(S, dim=1, keepdim=True)
    logp = S - logZ
    npos = pos.sum(1)
    valid = npos > 0
    if valid.sum() == 0:
        return Z.sum() * 0.0
    lp = (logp * pos).sum(1)[valid] / npos[valid]
    return -lp.mean()
