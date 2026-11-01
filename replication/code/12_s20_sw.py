"""Smith-Waterman nearest neighbour within the S20 benchmark gallery."""
import os, sys
import numpy as np, pandas as pd, parasail
CH = int(sys.argv[1]); NCH = int(sys.argv[2])
H = os.environ["HYP_ROOT"]; R = H + "/results/s20sw"
os.makedirs(R, exist_ok=True)
MAT = parasail.blosum62
d = pd.read_parquet(os.environ["HYP_DATA"] + "/cath.parquet").reset_index(drop=True)
z = np.load(H + "/s20/s20_index.npz")
gal = z["gallery"]; q = z["q_all"][CH::NCH]
seqs = d.seq.to_numpy()
hq, ht = [], []
for i in q:
    bs, bj = -1, -1
    for j in gal:
        if j == i:
            continue
        sc = parasail.sw_striped_16(seqs[i], seqs[j], 11, 1, MAT).score
        if sc > bs:
            bs, bj = sc, j
    hq.append(int(i)); ht.append(int(bj))
np.savez_compressed(R + "/sw_%03d.npz" % CH, q=np.array(hq), t=np.array(ht))
print("chunk", CH, "queries", len(hq))
