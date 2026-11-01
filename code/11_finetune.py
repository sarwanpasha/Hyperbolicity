"""Stage 6 (main contribution): train a projection of ESM-2 with and without the
differentiable delta-hyperbolicity regularizer, and evaluate remote-homology
generalisation on fold-disjoint held-out SCOPe folds and on CATH.

Two backbones:
  --backbone frozen : MLP head on precomputed layer embeddings (cheap, ablations)
  --backbone lora   : LoRA-adapted ESM-2 trunk + head (expensive, headline runs)
"""
import argparse, json, math, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import PROC, EMB, RESULTS, CKPT, HF_CACHE, ESM_MODELS, get_logger, Timer, save_json
from hyploss import hyperbolicity_loss, anticollapse_loss, supcon_loss, pairwise_dist
from geom import delta_from_D, pdist_normalized

log = get_logger("finetune")


class Head(nn.Module):
    def __init__(self, din, dout=256, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, hidden), nn.GELU(),
                                 nn.Linear(hidden, dout))
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
                take = self.rng.choice(len(pool), size=min(self.K, len(pool)), replace=len(pool) < self.K)
                idx.extend([pool[t] for t in take])
            yield np.array(idx)
    def __len__(self):
        return self.steps


def retrieval_metrics(Z, labels, K=5, device="cuda"):
    X = F.normalize(torch.tensor(np.asarray(Z, dtype=np.float32)), dim=1).to(device)
    N = X.shape[0]
    top1 = np.zeros(N, dtype=bool); top5 = np.zeros(N, dtype=bool)
    for s in range(0, N, 1024):
        e = min(s + 1024, N)
        S = X[s:e] @ X.T
        S[torch.arange(e - s, device=device), torch.arange(s, e, device=device)] = -2
        idx = torch.topk(S, K, dim=1).indices.cpu().numpy()
        lab = labels[idx]
        top1[s:e] = lab[:, 0] == labels[s:e]
        top5[s:e] = (lab == labels[s:e][:, None]).any(1)
    return float(top1.mean()), float(top5.mean())


def measure_delta(Z, n=2000, m=48, seed=0):
    """Global-ish delta estimate: random subsets of size m, relative delta."""
    rng = np.random.default_rng(seed)
    Zf = np.asarray(Z, dtype=np.float32)
    idx = rng.integers(0, Zf.shape[0], size=(n, m))
    D = pdist_normalized(Zf[idx])
    d = delta_from_D(D, base=0)
    diam = D.max(axis=(1, 2))
    return float(np.mean(d)), float(np.mean(2 * d / np.clip(diam, 1e-6, None)))


def poincare_supcon(Z, labels, c=1.0, temp=0.1, eps=1e-5):
    """Baseline competitor: map the head output into the Poincare ball with the
    exponential map at the origin and run the contrastive loss on -d_hyp."""
    n = Z.norm(dim=-1, keepdim=True).clamp_min(eps)
    sc = math.sqrt(c)
    X = torch.tanh(sc * n) * Z / (sc * n)
    X = X * (1 - 1e-4)
    x2 = (X * X).sum(-1)
    num = torch.cdist(X, X, p=2).pow(2)
    den = ((1 - c * x2)[:, None] * (1 - c * x2)[None, :]).clamp_min(eps)
    arg = (1 + 2 * c * num / den).clamp_min(1 + 1e-7)
    Dh = torch.acosh(arg) / sc
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


def run_frozen(a, device):
    meta = pd.read_parquet(PROC / "scope_splits.parquet")
    Z = np.load(EMB / ("scope_" + a.model + ".npy"), mmap_mode="r")
    ids = np.load(EMB / ("scope_" + a.model + "_ids.npy"), allow_pickle=True)
    meta = meta.set_index("sid").loc[ids].reset_index()
    layer = a.layer if a.layer >= 0 else Z.shape[0] - 2
    X = np.asarray(Z[layer], dtype=np.float32)
    din = X.shape[1]
    log.info("frozen backbone, layer %d, dim %d, N=%d" % (layer, din, len(meta)))

    tr = (meta.split == "train").to_numpy()
    va = (meta.split == "val").to_numpy()
    te = (meta.split == "test").to_numpy()
    lab_sf = meta.superfam.to_numpy()
    Xtr = torch.tensor(X[tr]).to(device)
    ytr_raw = lab_sf[tr]
    uniq = {s: i for i, s in enumerate(pd.unique(ytr_raw))}
    ytr = torch.tensor([uniq[s] for s in ytr_raw], device=device)

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    head = Head(din, a.dout).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    sampler = PKSampler(ytr_raw, P=a.P, K=a.K, seed=a.seed, steps=a.steps)
    gen = torch.Generator(device=device); gen.manual_seed(a.seed)

    hist = []
    for ep in range(a.epochs):
        head.train(); agg = {}
        for bidx in sampler:
            xb = Xtr[bidx]; yb = ytr[bidx]
            zb = head(xb)
            if a.method == "poincare":
                l_task = poincare_supcon(zb, yb, temp=a.temp)
            else:
                l_task = supcon_loss(zb, yb, temp=a.temp)
            loss = l_task
            stats = {"task": float(l_task)}
            if a.method == "hyp":
                l_h, s1 = hyperbolicity_loss(zb, a.quads, a.tau, a.target, gen,
                                                   agg=a.agg, topq=a.topq)
                l_v, s2 = anticollapse_loss(zb, a.min_dist)
                loss = loss + a.lam_hyp * l_h + a.lam_var * l_v
                stats.update({"hyp": float(l_h), "var": float(l_v)}); stats.update(s1); stats.update(s2)
            opt.zero_grad(); loss.backward(); opt.step()
            for k, v in stats.items():
                agg[k] = agg.get(k, 0.0) + v
        agg = {k: v / len(sampler) for k, v in agg.items()}
        head.eval()
        with torch.no_grad():
            Zva = head(torch.tensor(X[va]).to(device)).cpu().numpy()
        t1, t5 = retrieval_metrics(Zva, lab_sf[va], device=device)
        d_abs, d_rel = measure_delta(Zva)
        hist.append(dict(epoch=ep, val_top1=t1, val_top5=t5, delta=d_abs, delta_rel=d_rel, **agg))
        log.info("ep %d val_top1=%.4f val_top5=%.4f delta_rel=%.4f %s" % (ep, t1, t5, d_rel, agg))
    return head, X, meta, hist, layer, (tr, va, te)


def evaluate(a, head, X, meta, layer, masks, device):
    tr, va, te = masks
    lab_sf = meta.superfam.to_numpy(); lab_fold = meta.fold.to_numpy()
    out = {}
    with torch.no_grad():
        def emb(mask):
            Xi = torch.tensor(X[mask]).to(device)
            return Xi.cpu().numpy() if head is None else head(Xi).cpu().numpy()
        for split, mask in (("val", va), ("test", te)):
            Zs = emb(mask)
            t1, t5 = retrieval_metrics(Zs, lab_sf[mask], device=device)
            f1, f5 = retrieval_metrics(Zs, lab_fold[mask], device=device)
            d_abs, d_rel = measure_delta(Zs)
            out[split] = dict(sf_top1=t1, sf_top5=t5, fold_top1=f1, fold_top5=f5,
                              delta=d_abs, delta_rel=d_rel, n=int(mask.sum()))
        # CATH transfer: same layer, same head, never seen in training
        try:
            Zc = np.load(EMB / ("cath_" + a.model + ".npy"), mmap_mode="r")
            cids = np.load(EMB / ("cath_" + a.model + "_ids.npy"), allow_pickle=True)
            cm = pd.read_parquet(PROC / "cath.parquet").set_index("did").loc[cids].reset_index()
            Xc = torch.tensor(np.asarray(Zc[layer], dtype=np.float32)).to(device)
            Zc2 = Xc.cpu().numpy() if head is None else torch.cat(
                [head(Xc[i:i + 4096]).cpu() for i in range(0, Xc.shape[0], 4096)]).numpy()
            h1, h5 = retrieval_metrics(Zc2, cm.cath_h.to_numpy(), device=device)
            t1c, t5c = retrieval_metrics(Zc2, cm.cath_t.to_numpy(), device=device)
            d_abs, d_rel = measure_delta(Zc2)
            out["cath"] = dict(h_top1=h1, h_top5=h5, t_top1=t1c, t_top5=t5c,
                               delta=d_abs, delta_rel=d_rel, n=int(len(cm)))
        except FileNotFoundError:
            log.info("no CATH embeddings yet; skipping transfer eval")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="150M")
    ap.add_argument("--backbone", default="frozen", choices=["frozen", "lora"])
    ap.add_argument("--method", default="ft", choices=["none", "ft", "hyp", "poincare"])
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--dout", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--P", type=int, default=48)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--lam-hyp", type=float, default=1.0, dest="lam_hyp")
    ap.add_argument("--lam-var", type=float, default=1.0, dest="lam_var")
    ap.add_argument("--quads", type=int, default=4096)
    ap.add_argument("--agg", default="topq", choices=["mean", "topq", "lse"])
    ap.add_argument("--topq", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--target", type=float, default=0.0)
    ap.add_argument("--min-dist", type=float, default=1.0, dest="min_dist")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = "_".join([a.model, a.backbone, a.method, "lh%g" % a.lam_hyp,
                     "s%d" % a.seed] + ([a.tag] if a.tag else []))
    log.info("run %s on %s" % (name, device))

    with Timer(log, name):
        if a.method == "none":
            meta = pd.read_parquet(PROC / "scope_splits.parquet")
            Z = np.load(EMB / ("scope_" + a.model + ".npy"), mmap_mode="r")
            ids = np.load(EMB / ("scope_" + a.model + "_ids.npy"), allow_pickle=True)
            meta = meta.set_index("sid").loc[ids].reset_index()
            layer = a.layer if a.layer >= 0 else Z.shape[0] - 2
            X = np.asarray(Z[layer], dtype=np.float32)
            masks = ((meta.split == "train").to_numpy(), (meta.split == "val").to_numpy(),
                     (meta.split == "test").to_numpy())
            head, hist = None, []
        else:
            head, X, meta, hist, layer, masks = run_frozen(a, device)
        res = evaluate(a, head, X, meta, layer, masks, device)
        res["config"] = vars(a); res["history"] = hist; res["layer"] = layer
        save_json(res, RESULTS / ("run_" + name + ".json"))
        log.info("TEST %s" % res.get("test"))
        log.info("CATH %s" % res.get("cath"))
    log.info("RUN_DONE")
