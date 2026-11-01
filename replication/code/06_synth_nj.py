"""Tree recovery on simulated phylogenies, scored by neighbor joining.

Sequences are evolved down a known random binary tree, embedded with a frozen
ESM-2 35M, and a projection head is trained on clade labels from one half of the
tree with and without the four-point penalty. The other half, whose clades were
never labels, is then scored three ways: dendrogram purity, the normalized
Robinson-Foulds distance between a neighbor-joining tree built from embedding
distances and the generating tree, and the fraction of sampled quartets whose
resolved topology matches. The quartet score is the natural one here, since the
four-point condition the penalty acts on is itself a statement about quartets.
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F

AA = list("ACDEFGHIKLMNPQRSTVWY")


def simulate(depth, base_rate, seqlen, jitter, seed):
    """Evolve one ancestral sequence down a complete binary tree of given depth."""
    rng = np.random.default_rng(seed)
    root = rng.integers(0, 20, size=seqlen)
    level = [(root, 0.0)]
    for _ in range(depth):
        nxt = []
        for seq, dist in level:
            for _ in range(2):
                b = base_rate * np.exp(rng.normal(0, jitter))
                nmut = rng.binomial(seqlen, min(b, 0.9))
                child = seq.copy()
                if nmut:
                    sp = rng.choice(seqlen, nmut, replace=False)
                    child[sp] = rng.integers(0, 20, nmut)
                nxt.append((child, dist + b))
        level = nxt
    leaves = ["".join(AA[i] for i in s) for s, _ in level]
    n = len(leaves)
    # patristic distance from the complete-tree structure: 2 * (depth - lca_depth)
    idx = np.arange(n)
    D = np.zeros((n, n), dtype=np.float64)
    for d in range(1, depth + 1):
        blk = 1 << d
        same = (idx[:, None] // blk) == (idx[None, :] // blk)
        D[~same] += 2.0
    np.fill_diagonal(D, 0.0)
    return leaves, D


def true_splits(n, depth):
    out = []
    for d in range(1, depth):
        blk = 1 << d
        for s in range(0, n, blk):
            part = frozenset(range(s, s + blk))
            if 1 < len(part) < n - 1:
                out.append(part)
    return out


def neighbor_join(D):
    """Standard neighbor joining; returns the set of non-trivial splits."""
    n = D.shape[0]
    big = np.zeros((2 * n, 2 * n))
    big[:n, :n] = D
    active = list(range(n))
    members = {i: frozenset([i]) for i in range(n)}
    splits, nxt = [], n
    while len(active) > 2:
        idx = np.array(active)
        sub = big[np.ix_(idx, idx)]
        m = len(idx)
        r = sub.sum(axis=1)
        Q = (m - 2) * sub - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        i, j = np.unravel_index(np.argmin(Q), Q.shape)
        a, b = int(idx[i]), int(idx[j])
        new = members[a] | members[b]
        if 1 < len(new) < n - 1:
            splits.append(new)
        members[nxt] = new
        for k in idx:
            if k in (a, b):
                continue
            big[nxt, k] = big[k, nxt] = 0.5 * (big[a, k] + big[b, k] - big[a, b])
        active = [k for k in active if k not in (a, b)] + [nxt]
        nxt += 1
    return splits


def rf_distance(s1, s2, n):
    def canon(ss):
        out = set()
        for x in ss:
            y = frozenset(range(n)) - x
            out.add(x if 0 in x else y)
        return out
    a, b = canon(s1), canon(s2)
    tot = len(a) + len(b)
    return float(len(a ^ b)) / tot if tot else 0.0


def quartet_score(De, Dt, n_q=20000, seed=0):
    rng = np.random.default_rng(seed)
    n = De.shape[0]
    q = rng.integers(0, n, size=(n_q, 4))
    q = q[[len(set(r)) == 4 for r in q]]
    def topo(D):
        a, b, c, e = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return np.argmin(np.stack([D[a, b] + D[c, e], D[a, c] + D[b, e],
                                   D[a, e] + D[b, c]], axis=1), axis=1)
    return float((topo(De) == topo(Dt)).mean())


def dendrogram_purity(D, labels, pairs=20000, seed=0):
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    Z = linkage(squareform(D, checks=False), method="average")
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    uq = [np.where(labels == u)[0] for u in np.unique(labels)]
    uq = [g for g in uq if len(g) > 1]
    if not uq:
        return 0.0
    k = len(np.unique(labels))
    cl = fcluster(Z, k, criterion="maxclust")
    hits = 0
    for _ in range(pairs):
        g = uq[rng.integers(len(uq))]
        i, j = rng.choice(g, 2, replace=False)
        hits += int(cl[i] == cl[j])
    return hits / pairs


def supcon(z, y, temp=0.1):
    s = (z @ z.T) / temp
    n = z.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    s = s.masked_fill(eye, -1e9)
    pos = (y.view(-1, 1) == y.view(1, -1)) & ~eye
    logp = s - torch.logsumexp(s, dim=1, keepdim=True)
    cnt = pos.sum(1)
    ok = cnt > 0
    if not ok.any():
        return z.sum() * 0.0
    return -((logp * pos).sum(1)[ok] / cnt[ok]).mean()


def hyp_penalty(z, n_quads=2048, rho=0.05, eps=1e-6):
    n = z.shape[0]
    q = torch.randint(0, n, (n_quads, 4), device=z.device)
    keep = (q[:, 0] != q[:, 1]) & (q[:, 0] != q[:, 2]) & (q[:, 0] != q[:, 3]) & \
           (q[:, 1] != q[:, 2]) & (q[:, 1] != q[:, 3]) & (q[:, 2] != q[:, 3])
    q = q[keep]
    if q.shape[0] < 8:
        return z.sum() * 0.0
    a, b, c, e = [z[q[:, i]] for i in range(4)]
    d = lambda u, v: torch.linalg.norm(u - v, dim=-1)
    S1 = d(a, b) + d(c, e)
    S2 = d(a, c) + d(b, e)
    S3 = d(a, e) + d(b, c)
    S = torch.stack([S1, S2, S3], dim=1)
    top2 = torch.topk(S, 2, dim=1).values
    slack = 0.5 * (top2[:, 0] - top2[:, 1])
    sbar = S.sum(1) / 6.0
    u = slack / (sbar + eps)
    k = max(1, int(np.ceil(rho * u.shape[0])))
    return (torch.topk(u, k).values ** 2).mean()


def anticollapse(z, m0=1.0):
    n = z.shape[0]
    D = torch.cdist(z, z)
    mean = D.sum() / max(n * (n - 1), 1)
    return torch.clamp(m0 - mean, min=0.0)


def poincare_supcon(z, y, temp=0.1, eps=1e-5):
    nv = z.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    x = torch.tanh(nv) * z / nv
    sq = (x * x).sum(-1).clamp_max(1 - eps)
    num = torch.cdist(x, x).pow(2)
    den = ((1 - sq).view(-1, 1) * (1 - sq).view(1, -1)).clamp_min(1e-6)
    dh = torch.acosh((1 + 2 * num / den).clamp_min(1 + 1e-6))
    s = -dh / temp
    n = z.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    s = s.masked_fill(eye, -1e9)
    pos = (y.view(-1, 1) == y.view(1, -1)) & ~eye
    logp = s - torch.logsumexp(s, dim=1, keepdim=True)
    cnt = pos.sum(1)
    ok = cnt > 0
    if not ok.any():
        return z.sum() * 0.0
    return -((logp * pos).sum(1)[ok] / cnt[ok]).mean()


@torch.no_grad()
def embed_frozen(seqs, device, bs=32):
    from transformers import AutoModel, AutoTokenizer
    name = "facebook/esm2_t12_35M_UR50D"
    tok = AutoTokenizer.from_pretrained(name)
    mod = AutoModel.from_pretrained(name).to(device).eval()
    out = []
    for i in range(0, len(seqs), bs):
        b = tok(seqs[i:i + bs], return_tensors="pt", padding=True,
                truncation=True, max_length=512).to(device)
        h = mod(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1).clamp_min(1.0)).float().cpu())
    return torch.cat(out)


def train_head(X, y_train, train_mask, method, lam, steps, seed, device):
    torch.manual_seed(seed)
    din = X.shape[1]
    head = torch.nn.Sequential(torch.nn.Linear(din, 256), torch.nn.ReLU(),
                               torch.nn.Linear(256, 128)).to(device)
    if method == "frozen":
        with torch.no_grad():
            return F.normalize(X.to(device), dim=1).cpu().numpy()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    Xt = X.to(device)
    tr = np.where(train_mask)[0]
    yl = np.asarray(y_train)
    rng = np.random.default_rng(seed)
    uq = np.unique(yl[tr])
    for _ in range(steps):
        fams = rng.choice(uq, size=min(16, len(uq)), replace=False)
        idx = []
        for f in fams:
            pool = tr[yl[tr] == f]
            if len(pool) >= 2:
                idx.extend(rng.choice(pool, size=min(4, len(pool)), replace=False))
        if len(idx) < 8:
            continue
        idx = np.array(idx)
        z = F.normalize(head(Xt[idx]), dim=1)
        yb = torch.tensor(
            [int(np.where(uq == v)[0][0]) for v in yl[idx]], device=device)
        if method == "poincare":
            loss = poincare_supcon(head(Xt[idx]), yb)
        else:
            loss = supcon(z, yb)
            if method == "hyp":
                loss = loss + lam * hyp_penalty(z) + anticollapse(z)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return F.normalize(head(Xt), dim=1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--ntask", type=int, default=27)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seqlen", type=int, default=150)
    ap.add_argument("--lam", type=float, default=10.0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    grid = [(d, r, s) for d in (7, 9, 11) for r in (0.01, 0.02, 0.04)
            for s in (0, 1, 2)]
    cfgs = [g for i, g in enumerate(grid) if i % a.ntask == a.task]
    R = os.environ["HYP_ROOT"] + "/results/synthnj"
    os.makedirs(R, exist_ok=True)
    rows = []
    for depth, rate, seed in cfgs:
        leaves, Dt = simulate(depth, rate, a.seqlen, 0.5, seed)
        n = len(leaves)
        X = embed_frozen(leaves, dev)
        # clade labels at a mid depth; train on the left half of the tree only
        blk = 1 << max(1, depth - 4)
        clade = np.arange(n) // blk
        train_mask = np.arange(n) < (n // 2)
        held = ~train_mask
        ts = true_splits(n, depth)
        for method in ("frozen", "ft", "hyp", "poincare"):
            Z = train_head(X, clade, train_mask, method, a.lam, a.steps, seed, dev)
            Zh = Z[held]
            De = np.linalg.norm(Zh[:, None, :] - Zh[None, :, :], axis=-1)
            Dth = Dt[np.ix_(np.where(held)[0], np.where(held)[0])]
            nh = De.shape[0]
            nj = neighbor_join(De)
            tj = neighbor_join(Dth)
            rows.append(dict(depth=depth, rate=rate, seed=seed, method=method,
                             n_leaves=n, n_held=nh,
                             rf=rf_distance(nj, tj, nh),
                             quartet=quartet_score(De, Dth),
                             purity=dendrogram_purity(De, clade[held])))
            print(rows[-1], flush=True)
    with open(R + "/nj_%03d.json" % a.task, "w") as f:
        json.dump(rows, f, indent=1)
    print("wrote", len(rows), "rows")


if __name__ == "__main__":
    main()
