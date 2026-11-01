"""Controlled validation: sequences evolved along a KNOWN random tree.

We simulate a birth process that produces a rooted binary tree with branch
lengths, evolve an ancestral amino-acid sequence down the tree under a simple
substitution model, embed the leaves with a frozen ESM-2, and train a projection
head with and without the delta-hyperbolicity regularizer. Because the generating
tree is known, we can ask directly whether the regularizer makes the representation
a better tree: Spearman correlation between embedding distance and patristic
distance, dendrogram purity against the true clades, and the measured relative delta.

Clades are split into train/test halves, so the recovery metrics are reported on
clades whose labels were never used for training.
"""
import argparse, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from common import PROC, RESULTS, HF_CACHE, ESM_MODELS, get_logger, Timer, save_json
from hyploss import hyperbolicity_loss, anticollapse_loss, supcon_loss
from ftlib import Head, PKSampler, measure_delta, poincare_supcon

log = get_logger("synth")
AA = "ACDEFGHIKLMNPQRSTVWY"


def simulate_tree(depth=9, seqlen=150, base_rate=0.02, rate_jitter=0.5, seed=0,
                  root=None):
    """Complete binary tree of the given depth; returns leaf sequences, the
    patristic distance matrix, and the clade label of each leaf at every level."""
    rng = np.random.default_rng(seed)
    if root is None:
        root = rng.integers(0, 20, size=seqlen)
    seqs = {(): np.asarray(root)}
    blen = {(): 0.0}
    for d in range(depth):
        for node in [k for k in seqs if len(k) == d]:
            for child in (0, 1):
                key = node + (child,)
                b = base_rate * np.exp(rng.normal(0, rate_jitter))
                s = seqs[node].copy()
                nmut = rng.binomial(seqlen, min(b, 0.9))
                pos = rng.choice(seqlen, size=nmut, replace=False)
                s[pos] = rng.integers(0, 20, size=nmut)
                seqs[key] = s
                blen[key] = b
    leaves = sorted([k for k in seqs if len(k) == depth])
    strs = ["".join(AA[i] for i in seqs[k]) for k in leaves]
    n = len(leaves)
    # patristic distance = sum of branch lengths on the path between two leaves
    cum = {k: sum(blen[k[:i + 1]] for i in range(len(k))) for k in leaves}
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = leaves[i], leaves[j]
            c = 0
            while c < depth and a[c] == b[c]:
                c += 1
            anc = sum(blen[a[:t + 1]] for t in range(c))
            d = (cum[a] - anc) + (cum[b] - anc)
            D[i, j] = D[j, i] = d
    labels = {lev: np.array(["".join(map(str, k[:lev])) for k in leaves])
              for lev in range(1, depth)}
    return strs, D, labels, leaves


def dendrogram_purity(Z, labels, sample=20000, seed=0):
    """Fraction of same-label leaf pairs whose smallest common cluster in the
    average-linkage dendrogram of Z contains only that label (sampled)."""
    Zn = F.normalize(torch.tensor(np.asarray(Z, np.float32)), dim=1).numpy()
    Dm = np.sqrt(np.clip(2 - 2 * (Zn @ Zn.T), 0, None))
    L = linkage(squareform(Dm, checks=False), method="average")
    n = len(labels)
    rng = np.random.default_rng(seed)
    scores = []
    for k in np.unique(np.linspace(2, n // 2, 20).astype(int)):
        cl = fcluster(L, t=int(k), criterion="maxclust")
        df = pd.DataFrame({"c": cl, "y": labels})
        pur = df.groupby("c")["y"].apply(lambda s: (s.value_counts().iloc[0] / len(s)))
        w = df.groupby("c").size()
        scores.append(float((pur * w).sum() / w.sum()))
    return float(np.mean(scores))


def tree_recovery(Z, Dtrue):
    Zn = F.normalize(torch.tensor(np.asarray(Z, np.float32)), dim=1).numpy()
    De = np.sqrt(np.clip(2 - 2 * (Zn @ Zn.T), 0, None))
    iu = np.triu_indices(De.shape[0], 1)
    rho = spearmanr(De[iu], Dtrue[iu]).statistic
    return float(rho)


def embed_esm(seqs, model_key, device, batch=64):
    from transformers import AutoTokenizer, AutoModel
    name = ESM_MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(name, cache_dir=str(HF_CACHE))
    m = AutoModel.from_pretrained(name, cache_dir=str(HF_CACHE)).eval().to(device).to(torch.bfloat16)
    outs = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            enc = tok(seqs[i:i + batch], return_tensors="pt", padding=True, truncation=True,
                      max_length=1024)
            enc = {k: v.to(device) for k, v in enc.items()}
            h = m(**enc).last_hidden_state
            mask = enc["attention_mask"].clone(); mask[:, 0] = 0
            last = (enc["attention_mask"].sum(1) - 1).clamp(min=0)
            mask[torch.arange(mask.shape[0], device=device), last] = 0
            mm = mask.unsqueeze(-1).to(h.dtype)
            outs.append(((h * mm).sum(1) / mm.sum(1).clamp(min=1)).float().cpu().numpy())
    return np.concatenate(outs, 0)


def train_head(X, y_lab, method, a, device):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    head = Head(X.shape[1], a.dout).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    Xt = torch.tensor(X).to(device)
    uniq = {s: i for i, s in enumerate(pd.unique(y_lab))}
    yt = torch.tensor([uniq[s] for s in y_lab], device=device)
    sampler = PKSampler(y_lab, P=a.P, K=a.K, seed=a.seed, steps=a.steps)
    gen = torch.Generator(device=device); gen.manual_seed(a.seed)
    for ep in range(a.epochs):
        head.train()
        for bidx in sampler:
            zb = head(Xt[bidx]); yb = yt[bidx]
            if method == "poincare":
                loss = poincare_supcon(zb, yb, temp=a.temp)
            else:
                loss = supcon_loss(zb, yb, temp=a.temp)
            if method == "hyp":
                lh, _ = hyperbolicity_loss(zb, a.quads, gen=gen, agg=a.agg, topq=a.topq)
                lv, _ = anticollapse_loss(zb, a.min_dist)
                loss = loss + a.lam_hyp * lh + a.lam_ac * lv
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    return head


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="35M")
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--seqlen", type=int, default=150)
    ap.add_argument("--rate", type=float, default=0.02)
    ap.add_argument("--random-root", action="store_true", dest="random_root",
                    help="use a uniform random root instead of a real SCOPe domain")
    ap.add_argument("--train-level", type=int, default=5, dest="train_level")
    ap.add_argument("--dout", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--lam-hyp", type=float, default=30.0, dest="lam_hyp")
    ap.add_argument("--lam-ac", type=float, default=1.0, dest="lam_ac")
    ap.add_argument("--quads", type=int, default=4096)
    ap.add_argument("--agg", default="topq", choices=["mean", "topq", "lse"])
    ap.add_argument("--topq", type=float, default=0.05)
    ap.add_argument("--min-dist", type=float, default=1.0, dest="min_dist")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with Timer(log, "synthetic"):
        root = None
        if not a.random_root:
            sc = pd.read_parquet(PROC / "scope.parquet")
            cand = sc[(sc.length >= a.seqlen) & (sc.length <= a.seqlen + 60)]
            pick = cand.iloc[int(np.random.default_rng(a.seed).integers(0, len(cand)))]
            root = np.array([AA.index(c) for c in pick.seq[:a.seqlen] if c in AA])
            log.info("root = real SCOPe domain %s (len %d)" % (pick.sid, len(root)))
        seqs, Dtrue, labels, leaves = simulate_tree(a.depth, a.seqlen, a.rate,
                                                   seed=a.seed, root=root)
        log.info("simulated %d leaves, patristic dist mean=%.3f max=%.3f"
                 % (len(seqs), Dtrue[np.triu_indices(len(seqs), 1)].mean(), Dtrue.max()))
        # split the tree at the root: clade 0 = train, clade 1 = held out
        side = np.array([k[0] for k in leaves])
        tr, te = side == 0, side == 1
        y = labels[a.train_level]
        log.info("train leaves=%d test leaves=%d train labels=%d"
                 % (tr.sum(), te.sum(), len(np.unique(y[tr]))))

        X = embed_esm(seqs, a.model, device)
        log.info("embedded: %s" % (X.shape,))

        res = {}
        for method in ("none", "ft", "poincare", "hyp"):
            if method == "none":
                Zte = X[te]
            else:
                head = train_head(X[tr], y[tr], method, a, device)
                with torch.no_grad():
                    Zte = head(torch.tensor(X[te]).to(device)).cpu().numpy()
            Dte = Dtrue[np.ix_(te, te)]
            rho = tree_recovery(Zte, Dte)
            pur = dendrogram_purity(Zte, labels[a.train_level][te])
            pur_deep = dendrogram_purity(Zte, labels[a.depth - 2][te])
            d_abs, d_rel = measure_delta(Zte, n=1000, m=48)
            res[method] = dict(spearman_patristic=rho, purity_trainlevel=pur,
                               purity_deep=pur_deep, delta=d_abs, delta_rel=d_rel)
            log.info("%-9s rho=%.4f purity=%.4f purity_deep=%.4f delta_rel=%.4f"
                     % (method, rho, pur, pur_deep, d_rel))
        res["config"] = vars(a)
        save_json(res, RESULTS / ("synthetic_%s_d%d_s%d.json" % (a.model, a.depth, a.seed)))
    log.info("SYNTH_DONE")
