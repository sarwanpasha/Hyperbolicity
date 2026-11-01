"""Does the inflation reproduce outside proteins?

CIFAR-100 has the same two-level shape as our protein setting: 100 fine classes nested
in 20 coarse superclasses. We supervise the fine level, exactly as we supervise SCOPe
superfamilies, and evaluate retrieval at the coarse level, which is never supervised and
plays the role CATH plays for proteins.
"""
import argparse, os, pickle, tarfile
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from transformers import ViTModel
from common import RESULTS, get_logger, save_json
from ftlib import Head, PKSampler, retrieval_metrics, measure_delta
from hyploss import hyperbolicity_loss, anticollapse_loss, supcon_loss

log = get_logger("cifdis")
ROOT = os.environ["HYP_DATA"] + "/cifar"
VIT = "google/vit-base-patch16-224-in21k"

def load_split(name):
    """CIFAR-100 from the HuggingFace parquet mirror (uoft-cs/cifar100)."""
    import glob, io, pandas as pd
    from PIL import Image
    from huggingface_hub import snapshot_download
    root = snapshot_download("uoft-cs/cifar100", repo_type="dataset")
    f = glob.glob(root + "/cifar100/%s-*.parquet" % name)
    assert f, (root, name)
    df = pd.concat([pd.read_parquet(x) for x in sorted(f)], ignore_index=True)
    icol = "img" if "img" in df.columns else "image"
    X = np.zeros((len(df), 3, 32, 32), dtype=np.uint8)
    for i, v in enumerate(df[icol].to_numpy()):
        b = v["bytes"] if isinstance(v, dict) else v
        X[i] = np.asarray(Image.open(io.BytesIO(b)).convert("RGB")).transpose(2, 0, 1)
    return X, df["fine_label"].to_numpy(), df["coarse_label"].to_numpy()

class Net(nn.Module):
    def __init__(self, dout=256, freeze=False):
        super().__init__()
        self.trunk = ViTModel.from_pretrained(VIT, add_pooling_layer=False)
        if freeze:
            for p in self.trunk.parameters():
                p.requires_grad = False
        self.head = Head(self.trunk.config.hidden_size, dout)
        self.freeze = freeze
    def forward(self, x):
        h = self.trunk(pixel_values=x).last_hidden_state[:, 0]
        return h if self.freeze else self.head(h)

def prep(X, idx, dev):
    x = torch.from_numpy(X[idx]).to(dev).float().div_(255.0)
    x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
    return (x - 0.5) / 0.5

@torch.no_grad()
def embed(model, X, dev, bs=256):
    out = []
    for i in range(0, len(X), bs):
        idx = np.arange(i, min(i + bs, len(X)))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(prep(X, idx, dev)).float().cpu())
    return torch.cat(out).numpy()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="ft", choices=["none", "ft", "hyp"])
    ap.add_argument("--lam-hyp", type=float, default=1.0, dest="lam_hyp")
    ap.add_argument("--lam-ac", type=float, default=1.0, dest="lam_ac")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--quads", type=int, default=4096)
    ap.add_argument("--topq", type=float, default=0.05)
    ap.add_argument("--min-dist", type=float, default=1.0, dest="min_dist")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda"
    Xtr, ytr, ctr = load_split("train")
    Xte, yte, cte = load_split("test")
    order = np.random.default_rng(1234).permutation(20)
    seen, held = set(order[:10].tolist()), set(order[10:].tolist())
    trm = np.array([c in seen for c in ctr])
    m_seen = np.array([c in seen for c in cte])
    m_held = np.array([c in held for c in cte])
    Xtr, ytr = Xtr[trm], ytr[trm]
    log.info("train=%d over %d superclasses; held-out test=%d over %d"
             % (len(Xtr), len(seen), int(m_held.sum()), len(held)))
    log.info("train=%d test=%d fine=%d coarse=%d" % (len(Xtr), len(Xte),
             len(set(ytr.tolist())), len(set(ctr.tolist()))))

    model = Net(freeze=(a.method == "none")).to(dev)
    name = "cifdis_%s_lh%g_s%d" % (a.method, a.lam_hyp, a.seed)

    if a.method != "none":
        model.trunk.gradient_checkpointing_enable()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
        sampler = PKSampler(ytr, P=a.P, K=a.K, seed=a.seed, steps=a.steps)
        total = a.epochs * a.steps
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total,
                                                    pct_start=0.1)
        gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
        for ep in range(a.epochs):
            model.train(); agg = {}
            for idx in sampler:
                yb = torch.from_numpy(ytr[idx]).to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    zb = model(prep(Xtr, idx, dev))
                zb = zb.float()
                l_task = supcon_loss(zb, yb, temp=a.temp)
                loss = l_task; stats = {"task": float(l_task)}
                if a.method == "hyp":
                    l_h, s1 = hyperbolicity_loss(zb, a.quads, 0.0, 0.0, gen,
                                                 agg="topq", topq=a.topq)
                    l_v, s2 = anticollapse_loss(zb, a.min_dist)
                    loss = loss + a.lam_hyp * l_h + a.lam_ac * l_v
                    stats.update({"hyp": float(l_h), "ac": float(l_v)})
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step()
                for k, v in stats.items():
                    agg[k] = agg.get(k, 0.0) + v
            log.info("ep %d %s" % (ep, {k: round(v / a.steps, 4) for k, v in agg.items()}))

    model.eval()
    res = {}
    for tag, msk in (("held", m_held), ("seen", m_seen)):
        Z = embed(model, Xte[msk], dev)
        f1, f5, fa1, _ = retrieval_metrics(Z, yte[msk], device=dev, return_arrays=True)
        c1, c5, ca1, _ = retrieval_metrics(Z, cte[msk], device=dev, return_arrays=True)
        d_abs, d_rel = measure_delta(Z)
        res[tag] = dict(fine_top1=f1, fine_top5=f5, coarse_top1=c1, coarse_top5=c5,
                        delta=d_abs, delta_rel=d_rel, n=int(msk.sum()))
        np.savez_compressed(RESULTS / ("perq_%s_%s.npz" % (name, tag)),
                            coarse=ca1, fine=fa1)
    res["config"] = vars(a)
    res["superclasses_seen"] = sorted(seen)
    res["superclasses_held"] = sorted(held)
    save_json(res, RESULTS / ("run_%s.json" % name))
    log.info("[%s] held=%s seen=%s" % (name, res["held"], res["seen"]))
    log.info("CIFDIS_DONE")
