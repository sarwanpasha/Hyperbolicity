"""LoRA fine-tuning of ESM-2 with the optional four-point penalty.

Rebuilt on this cluster from the configuration the paper specifies: LoRA rank 8,
alpha 16 on the attention query, key and value projections, a two-layer head to
256 dimensions, a P x K sampler over SCOPe superfamilies, 400 steps of AdamW on a
one-cycle schedule peaking at 3e-4, bfloat16 autocast and gradient clipping at 1.0.
Saves the CATH embedding matrix so any gallery can be scored afterwards.
"""
import argparse, json, os, time
import numpy as np, pandas as pd
import torch, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
_nj = import_module("06_synth_nj")
supcon, hyp_penalty, anticollapse = _nj.supcon, _nj.hyp_penalty, _nj.anticollapse

MODELS = {"35M": "facebook/esm2_t12_35M_UR50D",
          "150M": "facebook/esm2_t30_150M_UR50D",
          "650M": "facebook/esm2_t33_650M_UR50D",
          "3B": "facebook/esm2_t36_3B_UR50D"}
PK = {"35M": (16, 4), "150M": (16, 4), "650M": (12, 4), "3B": (8, 4)}


class LoRALinear(torch.nn.Module):
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = torch.nn.Parameter(torch.zeros(r, base.in_features))
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, r))
        torch.nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scale


def add_lora(model, r, alpha):
    n = 0
    for name, mod in list(model.named_modules()):
        for child in ("query", "key", "value"):
            if hasattr(mod, child):
                lin = getattr(mod, child)
                if isinstance(lin, torch.nn.Linear):
                    setattr(mod, child, LoRALinear(lin, r, alpha))
                    n += 1
    return n


class Encoder(torch.nn.Module):
    def __init__(self, name, r, alpha, dim=256, dtype=None):
        super().__init__()
        kw = {"torch_dtype": dtype} if dtype is not None else {}
        try:
            self.trunk = AutoModel.from_pretrained(name, attn_implementation="sdpa", **kw)
            self.attn = "sdpa"
        except Exception as e:
            print("sdpa unavailable (%s); using eager attention" % type(e).__name__, flush=True)
            self.trunk = AutoModel.from_pretrained(name, **kw)
            self.attn = "eager"
        if r > 0:
            # LoRA: freeze the whole trunk, then add trainable low-rank adapters
            for p in self.trunk.parameters():
                p.requires_grad = False
            self.n_lora = add_lora(self.trunk, r, alpha)
        else:
            self.n_lora = 0
            for p in self.trunk.parameters():
                p.requires_grad = True
        h = self.trunk.config.hidden_size
        self.head = torch.nn.Sequential(torch.nn.Linear(h, 1024),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(1024, dim))

    def forward(self, ids, mask):
        h = self.trunk(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
        return F.normalize(self.head(pooled.float()), dim=-1)


def encode_all(model, tok, seqs, device, maxlen, bs=32):
    model.eval()
    out = []
    order = np.argsort([-len(s) for s in seqs])
    res = [None] * len(seqs)
    with torch.no_grad():
        for i in range(0, len(order), bs):
            sel = order[i:i + bs]
            b = tok([seqs[j][:maxlen] for j in sel], return_tensors="pt",
                    padding=True, truncation=True, max_length=maxlen).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device.startswith("cuda")):
                z = model(b["input_ids"], b["attention_mask"])
            for k, j in enumerate(sel):
                res[j] = z[k].float().cpu().numpy()
    model.train()
    return np.stack(res)


def retrieval_top1(Z, labels, device, bs=512):
    X = F.normalize(torch.tensor(Z, dtype=torch.float32), dim=1).to(device)
    labels = np.asarray(labels)
    ok = np.zeros(len(labels), dtype=bool)
    for s in range(0, len(labels), bs):
        e = min(s + bs, len(labels))
        S = X[s:e] @ X.T
        S[torch.arange(e - s, device=X.device), torch.arange(s, e, device=X.device)] = -2
        nn = S.argmax(1).cpu().numpy()
        ok[s:e] = labels[nn] == labels[s:e]
    return ok


def measure_drel(Z, n=800, m=48, seed=0):
    rng = np.random.default_rng(seed)
    Zf = np.asarray(Z, dtype=np.float32)
    idx = rng.integers(0, Zf.shape[0], size=(n, m))
    vals = []
    for row in idx:
        P = Zf[row]
        D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
        diam = D.max()
        if diam <= 0:
            continue
        g = 0.5 * (D[0, :][:, None] + D[0, :][None, :] - D)
        A = np.minimum(g[:, :, None], g[None, :, :]).max(axis=1) - g
        vals.append(2.0 * A.max() / diam)
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="35M")
    ap.add_argument("--method", default="ft", choices=["ft", "hyp", "none"])
    ap.add_argument("--lam-hyp", type=float, default=10.0, dest="lam_hyp")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--maxlen", type=int, default=1022)
    ap.add_argument("--train-maxlen", type=int, default=512, dest="train_maxlen")
    ap.add_argument("--lora-r", type=int, default=8, dest="lora_r")
    ap.add_argument("--grad-ckpt", action="store_true", dest="grad_ckpt")
    ap.add_argument("--bf16-trunk", action="store_true", dest="bf16_trunk")
    a = ap.parse_args()
    H = os.environ["HYP_ROOT"]; D = os.environ["HYP_DATA"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name = "%s_%s_lh%g_s%d" % (a.model, a.method, a.lam_hyp, a.seed)
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    sc = pd.read_parquet(D + "/scope.parquet")
    tr = sc[sc.split == "train"].reset_index(drop=True)
    te = sc[sc.split == "test"].reset_index(drop=True)
    va = sc[sc.split == "val"].reset_index(drop=True)
    ca = pd.read_parquet(D + "/cath.parquet").reset_index(drop=True)
    print("train %d val %d test %d cath %d" % (len(tr), len(va), len(te), len(ca)), flush=True)

    tok = AutoTokenizer.from_pretrained(MODELS[a.model])
    model = Encoder(MODELS[a.model], a.lora_r, 16,
                    dtype=torch.bfloat16 if a.bf16_trunk else None).to(dev)
    if a.grad_ckpt:
        model.trunk.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    print("attention=%s grad_ckpt=%s bf16_trunk=%s" % (model.attn, a.grad_ckpt, a.bf16_trunk), flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    print("trainable %.2fM" % (sum(p.numel() for p in params) / 1e6), flush=True)

    t0 = time.time()
    if a.method != "none":
        opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=a.steps)
        P, K = PK[a.model]
        fams = tr.superfam.to_numpy()
        uq = np.array([f for f in pd.unique(fams) if (fams == f).sum() >= 2])
        rng = np.random.default_rng(a.seed)
        pool = {f: np.where(fams == f)[0] for f in uq}
        model.train()
        for step in range(a.steps):
            pick = rng.choice(uq, size=min(P, len(uq)), replace=False)
            idx = []
            for f in pick:
                p = pool[f]
                idx.extend(rng.choice(p, size=min(K, len(p)), replace=False))
            idx = np.array(idx)
            b = tok([tr.seq[j][:a.train_maxlen] for j in idx], return_tensors="pt",
                    padding=True, truncation=True, max_length=a.train_maxlen).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
                z = model(b["input_ids"], b["attention_mask"])
            y = torch.tensor([int(np.where(pick == f)[0][0]) for f in fams[idx]], device=dev)
            loss = supcon(z, y, temp=0.1)
            if a.method == "hyp":
                loss = loss + a.lam_hyp * hyp_penalty(z, n_quads=4096, rho=0.05) \
                            + anticollapse(z, m0=1.0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sch.step()
            if step % 100 == 0:
                print("step %d loss %.4f (%.0fs)" % (step, float(loss), time.time() - t0), flush=True)
    train_s = time.time() - t0

    res = {"name": name, "model": a.model, "method": a.method, "seed": a.seed,
           "train_maxlen": a.train_maxlen, "eval_maxlen": a.maxlen,
           "lam_hyp": a.lam_hyp, "steps": a.steps, "train_seconds": train_s,
           "n_train": len(tr), "n_val": len(va), "n_test": len(te), "n_cath": len(ca)}

    Zv = encode_all(model, tok, list(va.seq), dev, a.maxlen)
    res["val_sf_top1"] = float(retrieval_top1(Zv, va.superfam.to_numpy(), dev).mean())
    res["val_fold_top1"] = float(retrieval_top1(Zv, va.fold.to_numpy(), dev).mean())

    Zt = encode_all(model, tok, list(te.seq), dev, a.maxlen)
    res["test_sf_top1"] = float(retrieval_top1(Zt, te.superfam.to_numpy(), dev).mean())
    res["test_fold_top1"] = float(retrieval_top1(Zt, te.fold.to_numpy(), dev).mean())
    res["test_drel"] = measure_drel(Zt)

    Zc = encode_all(model, tok, list(ca.seq), dev, a.maxlen)
    okc = retrieval_top1(Zc, ca.cath_h.to_numpy(), dev)
    res["cath_h_top1"] = float(okc.mean())
    res["cath_drel"] = measure_drel(Zc)

    os.makedirs(H + "/results/runs", exist_ok=True)
    os.makedirs(H + "/data/emb", exist_ok=True)
    np.save(H + "/data/emb/ftcath_%s.npy" % name, Zc.astype(np.float16))
    np.savez_compressed(H + "/results/runs/perq_%s.npz" % name, cath_h_top1=okc)
    with open(H + "/results/runs/run_%s.json" % name, "w") as f:
        json.dump(res, f, indent=1)
    print("RESULT", json.dumps(res), flush=True)
    print("RUN_DONE", name, flush=True)


if __name__ == "__main__":
    main()
