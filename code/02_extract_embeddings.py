"""Stage 2: extract layerwise mean-pooled ESM-2 embeddings for SCOPe / CATH.

Writes EMB/<dataset>_<model>.npy with shape (n_layers+1, N, D) in float16,
plus EMB/<dataset>_<model>_ids.npy giving row order.
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import torch
from common import PROC, EMB, HF_CACHE, ESM_MODELS, get_logger, Timer

log = get_logger("extract")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="150M", choices=list(ESM_MODELS))
    ap.add_argument("--dataset", default="scope", choices=["scope", "cath"])
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    out = EMB / (a.dataset + "_" + a.model + ".npy")
    if out.exists() and not a.overwrite:
        log.info("exists, skipping: %s" % out); return

    df = pd.read_parquet(PROC / (a.dataset + ".parquet"))
    idcol = "sid" if a.dataset == "scope" else "did"
    log.info("loaded %d %s domains" % (len(df), a.dataset))

    from transformers import AutoTokenizer, AutoModel
    name = ESM_MODELS[a.model]
    log.info("loading %s" % name)
    tok = AutoTokenizer.from_pretrained(name, cache_dir=str(HF_CACHE))
    model = AutoModel.from_pretrained(name, cache_dir=str(HF_CACHE))
    model.eval().cuda().to(torch.bfloat16)
    nL = model.config.num_hidden_layers + 1
    D = model.config.hidden_size
    log.info("layers(incl embed)=%d hidden=%d" % (nL, D))

    order = np.argsort(df.length.to_numpy())  # length-sorted batching
    seqs = df.seq.to_numpy()[order]
    ids = df[idcol].to_numpy()[order]
    N = len(seqs)
    Z = np.zeros((nL, N, D), dtype=np.float32)

    i, done = 0, 0
    with torch.no_grad():
        while i < N:
            L = len(seqs[i]) + 2
            bs = max(1, min(a.max_tokens // L, 128, N - i))
            batch = list(seqs[i:i + bs])
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            enc = {k: v.cuda() for k, v in enc.items()}
            out_h = model(**enc, output_hidden_states=True).hidden_states
            mask = enc["attention_mask"].clone()
            mask[:, 0] = 0
            idx = mask.sum(1) - 1 + 1
            for b in range(mask.shape[0]):
                mask[b, idx[b]] = 0  # drop eos
            m = mask.unsqueeze(-1).to(torch.bfloat16)
            denom = m.sum(1).clamp(min=1)
            for li, h in enumerate(out_h):
                pooled = (h * m).sum(1) / denom
                Z[li, i:i + bs] = pooled.float().cpu().numpy()
            i += bs; done += bs
            if done % 2000 < bs:
                log.info("%d/%d" % (done, N))

    np.save(out, Z)
    np.save(EMB / (a.dataset + "_" + a.model + "_ids.npy"), ids)
    log.info("saved %s shape=%s" % (out, Z.shape))
    log.info("STAGE2_DONE")


if __name__ == "__main__":
    with Timer(log, "stage 2"):
        main()
