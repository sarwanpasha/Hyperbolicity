"""Highest Smith-Waterman identity each domain shares with a true homolog.

For every CATH domain we align it against other members of its own homologous
superfamily and keep the best identity, matches divided by the length of the
shorter domain. Superfamilies with more than 40 members are compared against a
random reference subset of 40, which lower-bounds the true maximum. This defines
the remoteness strata the paper reports accuracy in.
"""
import os, sys
import numpy as np, pandas as pd, parasail

CH = int(sys.argv[1]); NCH = int(sys.argv[2])
D = os.environ["HYP_DATA"]; R = os.environ["HYP_ROOT"] + "/results/maxident"
os.makedirs(R, exist_ok=True)
MAT = parasail.blosum62
d = pd.read_parquet(D + "/cath.parquet").reset_index(drop=True)
seqs = list(d.seq)
groups = {}
for i, h in enumerate(d.cath_h.to_numpy()):
    groups.setdefault(h, []).append(i)
keys = sorted(groups)
mine = [k for n, k in enumerate(keys) if n % NCH == CH]
rng = np.random.default_rng(0)
out_i, out_v = [], []
for k in mine:
    idx = groups[k]
    if len(idx) < 2:
        for i in idx:
            out_i.append(i); out_v.append(0.0)
        continue
    refs = idx if len(idx) <= 40 else list(rng.choice(idx, 40, replace=False))
    for i in idx:
        best = 0.0
        for j in refs:
            if j == i:
                continue
            r = parasail.sw_stats_striped_16(seqs[i], seqs[j], 11, 1, MAT)
            den = min(len(seqs[i]), len(seqs[j]))
            if den:
                v = r.matches / den
                if v > best:
                    best = v
        out_i.append(i); out_v.append(float(best))
np.savez_compressed(R + "/mi_%03d.npz" % CH,
                    idx=np.array(out_i, dtype=np.int32),
                    val=np.array(out_v, dtype=np.float32))
print("chunk", CH, "domains", len(out_i))
