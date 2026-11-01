"""End-to-end LoRA fine-tuning of ESM-2 with the delta-hyperbolicity regularizer.

This is the headline experiment: the regularizer shapes the trunk representation,
not just a head over frozen features. Evaluation is fold-disjoint retrieval on
held-out SCOPe folds plus zero-shot transfer to CATH.
"""
import argparse, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import PROC, RESULTS, CKPT, HF_CACHE, ESM_MODELS, get_logger, Timer, save_json
from hyploss import hyperbolicity_loss, anticollapse_loss, supcon_loss
from nc import nc_stats
from ftlib import (Head, PKSampler, FoldPKSampler, retrieval_metrics,
                   measure_delta, poincare_supcon)

log = get_logger("lora")


class EsmEncoder(nn.Module):
    def __init__(self, name, dout=256, lora_r=8, lora_alpha=16, freeze=False,
                 grad_ckpt=False):
        super().__init__()
        self.use_head = not freeze
        from transformers import AutoModel
        self.trunk = AutoModel.from_pretrained(name, cache_dir=str(HF_CACHE))
        d = self.trunk.config.hidden_size
        if freeze:
            for p in self.trunk.parameters():
                p.requires_grad = False
        else:
            if grad_ckpt:
                # activation checkpointing: required to fit 650M with backprop on 40GB
                self.trunk.gradient_checkpointing_enable()
                if hasattr(self.trunk, "enable_input_require_grads"):
                    self.trunk.enable_input_require_grads()
            if lora_r > 0:
                from peft import LoraConfig, get_peft_model
                cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
                                 bias="none",
                                 target_modules=["query", "key", "value"])
                self.trunk = get_peft_model(self.trunk, cfg)
        self.head = Head(d, dout)

    def forward(self, ids, mask, want_pooled=False):
        h = self.trunk(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.clone()
        m[:, 0] = 0
        last = (mask.sum(1) - 1).clamp(min=0)
        m[torch.arange(m.shape[0], device=m.device), last] = 0
        mm = m.unsqueeze(-1).to(h.dtype)
        pooled = (h * mm).sum(1) / mm.sum(1).clamp(min=1)
        z = self.head(pooled) if self.use_head else pooled
        return (z, pooled) if want_pooled else z


def encode_all(model, tok, seqs, device, maxlen, bs_tokens=8192):
    """Embed a list of sequences with the current model (length-sorted batching)."""
    model.eval()
    order = np.argsort([len(s) for s in seqs])
    out = None
    with torch.no_grad():
        i = 0
        while i < len(order):
            L = min(len(seqs[order[i]]) + 2, maxlen)
            bs = max(1, min(bs_tokens // L, 128, len(order) - i))
            batch = [seqs[j] for j in order[i:i + bs]]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=maxlen)
            ids = enc["input_ids"].to(device); mask = enc["attention_mask"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model(ids, mask).float()
            if out is None:
                out = np.zeros((len(order), z.shape[1]), dtype=np.float32)
            out[order[i:i + bs]] = z.cpu().numpy()
            i += bs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="35M")
    ap.add_argument("--method", default="ft", choices=["none", "ft", "hyp", "poincare", "unif", "cpcc", "drift", "hypstruct", "multilevel"])
    ap.add_argument("--dout", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--maxlen", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--lam-hyp", type=float, default=1.0, dest="lam_hyp")
    ap.add_argument("--lam-ac", type=float, default=1.0, dest="lam_ac")
    ap.add_argument("--quads", type=int, default=4096)
    ap.add_argument("--agg", default="topq", choices=["mean", "topq", "lse"])
    ap.add_argument("--topq", type=float, default=0.05)
    ap.add_argument("--lse-temp", type=float, default=0.05, dest="lse_temp")
    ap.add_argument("--sampler", default="random", choices=["random", "fold"])
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--target", type=float, default=0.0)
    ap.add_argument("--min-dist", type=float, default=1.0, dest="min_dist")
    ap.add_argument("--lora-r", type=int, default=8, dest="lora_r")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true", dest="grad_ckpt")
    ap.add_argument("--nc", action="store_true")
    ap.add_argument("--save-cath-emb", action="store_true", dest="save_cath_emb")
    a = ap.parse_args()

    device = "cuda"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    name = "_".join(["lora", a.model, a.method, "lh%g" % a.lam_hyp, "s%d" % a.seed]
                    + ([a.tag] if a.tag else []))
    log.info("run %s" % name)
    outfile = RESULTS / ("run_" + name + ".json")
    if outfile.exists() and not a.overwrite:
        log.info("already done, skipping: %s" % outfile)
        return

    meta = pd.read_parquet(PROC / "scope_splits.parquet")
    cath = pd.read_parquet(PROC / "cath.parquet")
    tr = meta[meta.split == "train"].reset_index(drop=True)
    va = meta[meta.split == "val"].reset_index(drop=True)
    te = meta[meta.split == "test"].reset_index(drop=True)
    log.info("train=%d val=%d test=%d cath=%d" % (len(tr), len(va), len(te), len(cath)))

    from transformers import AutoTokenizer
    hf = ESM_MODELS[a.model]
    tok = AutoTokenizer.from_pretrained(hf, cache_dir=str(HF_CACHE))
    model = EsmEncoder(hf, a.dout, lora_r=a.lora_r,
                       freeze=(a.method == "none"),
                       grad_ckpt=a.grad_ckpt).to(device)
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("trainable params: %.2fM" % (ntr / 1e6))

    EMB_HOLD = {}

    def eval_all(tag):
        res = {}
        per_query = {}
        for split, df, labs in (("val", va, ["superfam", "fold"]),
                                ("test", te, ["superfam", "fold"])):
            Z = encode_all(model, tok, list(df.seq.to_numpy()), device, a.maxlen)
            r = {}
            for lb in labs:
                t1, t5, a1, a5 = retrieval_metrics(Z, df[lb].to_numpy(), device=device,
                                                   return_arrays=True)
                r[lb + "_top1"], r[lb + "_top5"] = t1, t5
                per_query[split + "_" + lb + "_top1"] = a1
                per_query[split + "_" + lb + "_top5"] = a5
            d_abs, d_rel = measure_delta(Z)
            r.update(delta=d_abs, delta_rel=d_rel, n=len(df))
            res[split] = r
        Zc = encode_all(model, tok, list(cath.seq.to_numpy()), device, a.maxlen)
        if tag == "final":
            EMB_HOLD["Zc"] = Zc
        rc = {}
        for lb in ("cath_h", "cath_t"):
            t1, t5, a1, a5 = retrieval_metrics(Zc, cath[lb].to_numpy(), device=device,
                                               return_arrays=True)
            rc[lb + "_top1"], rc[lb + "_top5"] = t1, t5
            per_query["cath_" + lb + "_top1"] = a1
            per_query["cath_" + lb + "_top5"] = a5
        d_abs, d_rel = measure_delta(Zc)
        rc.update(delta=d_abs, delta_rel=d_rel, n=len(cath))
        res["cath"] = rc
        log.info("[%s] test=%s cath=%s" % (tag, res["test"], res["cath"]))
        return res, per_query

    hist = []
    if a.method != "none":
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
        if a.sampler == "fold":
            sampler = FoldPKSampler(tr.superfam.to_numpy(), tr.fold.to_numpy(),
                                    P=a.P, K=a.K, seed=a.seed, steps=a.steps)
        else:
            sampler = PKSampler(tr.superfam.to_numpy(), P=a.P, K=a.K, seed=a.seed, steps=a.steps)
        total = a.epochs * a.steps
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total,
                                                    pct_start=0.1)
        seqs = list(tr.seq.to_numpy())
        sf = tr.superfam.to_numpy()
        fL = tr.fold.to_numpy()
        uniqF = {f: i for i, f in enumerate(pd.unique(fL))}
        cL = tr.cls.to_numpy()
        uniqC = {c: i for i, c in enumerate(pd.unique(cL))}
        faL = tr.family.to_numpy()
        uniqFa = {f: i for i, f in enumerate(pd.unique(faL))}
        uniq = {s: i for i, s in enumerate(pd.unique(sf))}
        gen = torch.Generator(device=device); gen.manual_seed(a.seed)
        step = 0
        for ep in range(a.epochs):
            model.train(); agg = {}
            t0 = time.time()
            for bidx in sampler:
                batch = [seqs[i] for i in bidx]
                yb = torch.tensor([uniq[s] for s in sf[bidx]], device=device)
                enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                          max_length=a.maxlen)
                ids = enc["input_ids"].to(device); mask = enc["attention_mask"].to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if a.method == "drift":
                        zb, pl = model(ids, mask, want_pooled=True)
                    else:
                        zb = model(ids, mask)
                zb = zb.float()
                if a.method == "poincare":
                    l_task = poincare_supcon(zb, yb, temp=a.temp)
                else:
                    if a.method == "multilevel":
                        ycb = torch.tensor([uniqC[c] for c in cL[bidx]], device=device)
                        yfb = torch.tensor([uniqF[f] for f in fL[bidx]], device=device)
                        yab = torch.tensor([uniqFa[f] for f in faL[bidx]], device=device)
                        l_task = 0.25 * (supcon_loss(zb, yab, temp=a.temp)
                                         + supcon_loss(zb, yb, temp=a.temp)
                                         + supcon_loss(zb, yfb, temp=a.temp)
                                         + supcon_loss(zb, ycb, temp=a.temp))
                    else:
                        l_task = supcon_loss(zb, yb, temp=a.temp)
                loss = l_task
                stats = {"task": float(l_task)}
                if a.method == "hyp":
                    l_h, s1 = hyperbolicity_loss(zb, a.quads, a.tau, a.target, gen,
                                                   agg=a.agg, topq=a.topq,
                                          lse_temp=a.lse_temp)
                    l_v, s2 = anticollapse_loss(zb, a.min_dist)
                    loss = loss + a.lam_hyp * l_h + a.lam_ac * l_v
                    stats.update({"hyp": float(l_h), "ac": float(l_v)})
                    stats.update(s1); stats.update(s2)
                elif a.method == "unif":
                    zn = torch.nn.functional.normalize(zb, dim=-1)
                    sq = torch.cdist(zn, zn).pow(2)
                    iu = torch.triu_indices(zn.size(0), zn.size(0), 1, device=zn.device)
                    l_u = torch.log(torch.exp(-2.0 * sq[iu[0], iu[1]]).mean() + 1e-12)
                    loss = loss + a.lam_hyp * l_u
                    stats.update({"unif": float(l_u)})
                elif a.method == "cpcc":
                    yfb = torch.tensor([uniqF[f] for f in fL[bidx]], device=device)
                    zn = torch.nn.functional.normalize(zb, dim=-1)
                    de = torch.cdist(zn, zn)
                    ssf = (yb.view(-1, 1) == yb.view(1, -1))
                    sfo = (yfb.view(-1, 1) == yfb.view(1, -1))
                    one = torch.ones_like(de)
                    dt = torch.where(ssf, torch.zeros_like(de), torch.where(sfo, one, 2.0 * one))
                    iu = torch.triu_indices(zn.size(0), zn.size(0), 1, device=zn.device)
                    xv = de[iu[0], iu[1]]
                    yv = dt[iu[0], iu[1]]
                    xv = xv - xv.mean()
                    yv = yv - yv.mean()
                    rho = (xv * yv).sum() / (xv.norm() * yv.norm() + 1e-8)
                    l_c = 1.0 - rho
                    loss = loss + a.lam_hyp * l_c
                    stats.update({"cpcc": float(l_c)})
                elif a.method == "hypstruct":
                    yfb = torch.tensor([uniqF[f] for f in fL[bidx]], device=device)
                    nv = zb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    sc = 1.0 / nv.mean().detach().clamp_min(1e-6)
                    xh = torch.tanh(0.5 * sc * nv) * zb / nv
                    sq = (xh * xh).sum(-1).clamp_max(1.0 - 1e-5)
                    num = torch.cdist(xh, xh).pow(2)
                    den = ((1.0 - sq).view(-1, 1) * (1.0 - sq).view(1, -1)).clamp_min(1e-6)
                    dh = torch.acosh((1.0 + 2.0 * num / den).clamp_min(1.0 + 1e-6))
                    ssf = (yb.view(-1, 1) == yb.view(1, -1))
                    sfo = (yfb.view(-1, 1) == yfb.view(1, -1))
                    one = torch.ones_like(dh)
                    dt = torch.where(ssf, torch.zeros_like(dh), torch.where(sfo, one, 2.0 * one))
                    iu = torch.triu_indices(dh.size(0), dh.size(0), 1, device=dh.device)
                    xv = dh[iu[0], iu[1]]
                    yv = dt[iu[0], iu[1]]
                    xv = xv - xv.mean()
                    yv = yv - yv.mean()
                    rho = (xv * yv).sum() / (xv.norm() * yv.norm() + 1e-8)
                    l_hc = 1.0 - rho
                    l_ct = xh.mean(0).norm()
                    loss = loss + a.lam_hyp * (l_hc + l_ct)
                    stats.update({"hypcpcc": float(l_hc), "center": float(l_ct)})
                elif a.method == "drift":
                    with torch.no_grad():
                        with model.trunk.disable_adapter():
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                pb = model(ids, mask, want_pooled=True)[1]
                    pb = torch.nn.functional.normalize(pb.float(), dim=-1)
                    pa = torch.nn.functional.normalize(pl.float(), dim=-1)
                    db = torch.cdist(pb, pb)
                    da = torch.cdist(pa, pa)
                    iu = torch.triu_indices(da.size(0), da.size(0), 1, device=da.device)
                    l_d = ((da[iu[0], iu[1]] - db[iu[0], iu[1]]) ** 2).mean()
                    loss = loss + a.lam_hyp * l_d
                    stats.update({"drift": float(l_d)})
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); step += 1
                for k, v in stats.items():
                    agg[k] = agg.get(k, 0.0) + v
            agg = {k: v / len(sampler) for k, v in agg.items()}
            Zv = encode_all(model, tok, list(va.seq.to_numpy()), device, a.maxlen)
            t1, t5 = retrieval_metrics(Zv, va.superfam.to_numpy(), device=device)
            d_abs, d_rel = measure_delta(Zv)
            hist.append(dict(epoch=ep, val_sf_top1=t1, val_sf_top5=t5,
                             delta=d_abs, delta_rel=d_rel, secs=time.time() - t0, **agg))
            log.info("ep %d val_top1=%.4f val_top5=%.4f delta_rel=%.4f (%.0fs) %s"
                     % (ep, t1, t5, d_rel, time.time() - t0, agg))

    res, per_query = eval_all("final")
    np.savez_compressed(RESULTS / ("perq_" + name + ".npz"), **per_query)
    if a.save_cath_emb and "Zc" in EMB_HOLD:
        import os as _os
        _d = _os.environ["HYP_DATA"] + "/emb"
        _os.makedirs(_d, exist_ok=True)
        np.save(_d + "/ftcath_" + name + ".npy", EMB_HOLD["Zc"].astype(np.float16))
        log.info("saved CATH embeddings %s for %s", EMB_HOLD["Zc"].shape, name)

    res["config"] = vars(a); res["history"] = hist
    res["trainable_params"] = int(ntr)
    if a.nc:
        Ztr = encode_all(model, tok, list(tr.seq.to_numpy()), device, a.maxlen)
        res["nc"] = nc_stats(Ztr, tr.superfam.to_numpy())
        log.info("NC %s" % res["nc"])
    save_json(res, RESULTS / ("run_" + name + ".json"))
    log.info("RUN_DONE %s" % name)


if __name__ == "__main__":
    with Timer(log, "lora run"):
        main()
