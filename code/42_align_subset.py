"""Smith-Waterman nearest-neighbor search restricted to the structure-matched gallery.

Same alignment settings as 31_align_baseline.py, but both the queries and the
targets are drawn from the 22,931 CATH domains that also have a structure, so the
result is comparable with Foldseek and with the embedding arms scored on that subset.
"""
import os, json
import numpy as np, pandas as pd, parasail
from multiprocessing import Pool
from common import PROC, RESULTS, get_logger

log = get_logger("alnsub")
S = os.environ["HYP_DATA"] + "/struct"
QN = 2000
MAT = parasail.blosum62

d = pd.read_parquet(PROC / "cath.parquet").reset_index(drop=True)
keep = set(x.strip() for x in open(S + "/overlap.list") if x.strip())
mask = d.did.isin(keep).to_numpy()
sub = d[mask].reset_index(drop=True)
seqs = list(sub.seq)
lab = sub.cath_h.to_numpy()


def best(i):
    s = seqs[i]
    bs = -1
    bj = -1
    for j in range(len(seqs)):
        if j == i:
            continue
        r = parasail.sw_striped_16(s, seqs[j], 11, 1, MAT)
        if r.score > bs:
            bs = r.score
            bj = j
    return i, bj


if __name__ == "__main__":
    cnt = sub.cath_h.value_counts()
    multi = set(cnt[cnt >= 2].index)
    retr = np.array([x in multi for x in lab])
    log.info("subset %d domains, %d retrievable", len(sub), int(retr.sum()))
    rng = np.random.default_rng(0)
    q = rng.choice(len(sub), min(QN, len(sub)), replace=False)
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 8))) as p:
        res = p.map(best, list(q), chunksize=8)
    qi = np.array([i for i, _ in res])
    hit = np.array([lab[i] == lab[j] for i, j in res])
    z = np.load(RESULTS / "cath_maxident.npz")["maxident"][mask]
    mi = z[qi]
    rq = retr[qi]
    out = {"n": int(len(qi)),
           "all": float(hit.mean()),
           "retrievable": float(hit[rq].mean()),
           "n_retr": int(rq.sum()),
           "lt20": float(hit[rq & (mi < .2)].mean()),
           "b2030": float(hit[rq & (mi >= .2) & (mi < .3)].mean()),
           "ge30": float(hit[rq & (mi >= .3)].mean()),
           "n_lt20": int((rq & (mi < .2)).sum()),
           "n_2030": int((rq & (mi >= .2) & (mi < .3)).sum()),
           "n_ge30": int((rq & (mi >= .3)).sum())}
    json.dump(out, open(RESULTS / "align_subset.json", "w"), indent=1)
    log.info("SW subset all=%.4f retr=%.4f | <20%% %.4f | 20-30%% %.4f | >=30%% %.4f",
             out["all"], out["retrievable"], out["lt20"], out["b2030"], out["ge30"])
