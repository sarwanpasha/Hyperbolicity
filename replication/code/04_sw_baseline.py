"""Smith-Waterman nearest-neighbour search over the full CATH gallery.

The alignment reference point: each sampled query is scored against all 37,169
domains with BLOSUM62 and affine gaps, and the highest-scoring other domain is
taken as the retrieved neighbour.
"""
import os, sys
import numpy as np, pandas as pd, parasail

CH = int(sys.argv[1]); NCH = int(sys.argv[2])
D = os.environ["HYP_DATA"]; R = os.environ["HYP_ROOT"] + "/results/sw"
os.makedirs(R, exist_ok=True)
MAT = parasail.blosum62
d = pd.read_parquet(D + "/cath.parquet").reset_index(drop=True)
seqs = list(d.seq)
q = np.load(os.environ["HYP_ROOT"] + "/hmm/qidx.npy")
mine = q[CH::NCH]
hit_q, hit_t = [], []
for i in mine:
    s = seqs[i]; bs = -1; bj = -1
    for j in range(len(seqs)):
        if j == i:
            continue
        sc = parasail.sw_striped_16(s, seqs[j], 11, 1, MAT).score
        if sc > bs:
            bs = sc; bj = j
    hit_q.append(int(i)); hit_t.append(int(bj))
np.savez_compressed(R + "/sw_%03d.npz" % CH,
                    q=np.array(hit_q, dtype=np.int32),
                    t=np.array(hit_t, dtype=np.int32))
print("chunk", CH, "queries", len(hit_q))
