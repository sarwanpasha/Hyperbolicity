"""End-to-end smoke test on the bundled subset.

Fine-tunes ESM-2 35M for a few steps with and without the four-point penalty on a
small SCOPe subset, then reports relative hyperbolicity and top-1 retrieval on a
small CATH subset. It checks that the installation works. The numbers it prints are
not the paper's numbers: the paper uses the full databases and 400 steps.

Usage:
    python smoke_test.py            # needs one GPU, about three minutes
    python smoke_test.py --cpu      # slower, no GPU needed
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TESTDATA = os.path.join(os.path.dirname(HERE), "testdata")

sys.path.insert(0, HERE)
from hyploss import hyperbolicity_loss, anticollapse_loss, supcon_loss  # noqa: E402
from ftlib import retrieval_metrics, measure_delta  # noqa: E402


def load(name):
    path = os.path.join(TESTDATA, name)
    if not os.path.exists(path):
        sys.exit("missing %s; run this from the repository checkout" % path)
    return pd.read_parquet(path)


def embed(model, tok, seqs, device, maxlen=512, bs=16):
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(seqs), bs):
            batch = [s[:maxlen] for s in seqs[i:i + bs]]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=maxlen).to(device)
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            z = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
            out.append(torch.nn.functional.normalize(z, dim=-1).float().cpu())
    return torch.cat(out).numpy()


def run(lam_hyp, tr, te, device, steps, seed):
    from transformers import AutoModel, AutoTokenizer
    torch.manual_seed(seed)
    np.random.seed(seed)
    name = "facebook/esm2_t12_35M_UR50D"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device)

    head = torch.nn.Sequential(
        torch.nn.Linear(model.config.hidden_size, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 128)).to(device)
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()), lr=1e-4)

    labels = tr.superfam.to_numpy()
    uniq = {s: i for i, s in enumerate(pd.unique(labels))}
    seqs = list(tr.seq)
    rng = np.random.default_rng(seed)

    model.train()
    for step in range(steps):
        fams = rng.choice(list(uniq), size=min(8, len(uniq)), replace=False)
        idx = []
        for f in fams:
            pool = np.where(labels == f)[0]
            idx.extend(rng.choice(pool, size=min(4, len(pool)), replace=False))
        idx = np.array(idx)
        batch = [seqs[i][:512] for i in idx]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to(device)
        h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        z = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
        z = torch.nn.functional.normalize(head(z), dim=-1)
        y = torch.tensor([uniq[labels[i]] for i in idx], device=device)

        loss = supcon_loss(z, y, temp=0.1)
        if lam_hyp > 0:
            l_hyp, _ = hyperbolicity_loss(z, n_quads=512, agg="topq", topq=0.05)
            l_ac, _ = anticollapse_loss(z, min_dist=1.0)
            loss = loss + lam_hyp * l_hyp + l_ac
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(head.parameters()), 1.0)
        opt.step()

    Z = embed(model, tok, list(te.seq), device)
    top1, _ = retrieval_metrics(Z, te.cath_h.to_numpy(), K=5,
                                device=str(device))
    _, drel = measure_delta(Z, n=200, m=24, seed=0)
    return drel, top1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    device = torch.device(
        "cpu" if a.cpu or not torch.cuda.is_available() else "cuda")

    tr = load("scope_mini.parquet")
    te = load("cath_mini.parquet")
    print("train %d domains over %d superfamilies; eval %d domains"
          % (len(tr), tr.superfam.nunique(), len(te)))
    print("device %s, %d steps\n" % (device, a.steps))

    d0, a0 = run(0.0, tr, te, device, a.steps, a.seed)
    print("task only            delta_rel %.4f   CATH top-1 %.4f" % (d0, a0))
    d1, a1 = run(10.0, tr, te, device, a.steps, a.seed)
    print("task + L_hyp (10)    delta_rel %.4f   CATH top-1 %.4f" % (d1, a1))

    print("\ndelta_rel change %+.4f" % (d1 - d0))
    if d1 < d0:
        print("PASS: the penalty lowered relative hyperbolicity")
    else:
        print("WARNING: no reduction at this tiny scale; rerun with --steps 60")


if __name__ == "__main__":
    main()
