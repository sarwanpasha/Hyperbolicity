"""Gallery and query FASTAs. The query sample reproduces the paper's seed-0 draw."""
import os
import numpy as np, pandas as pd
D = os.environ["HYP_DATA"]
H = os.environ["HYP_ROOT"] + "/hmm"
os.makedirs(H, exist_ok=True)
d = pd.read_parquet(D + "/cath.parquet").reset_index(drop=True)
with open(H + "/cath.fasta", "w") as f:
    for r in d.itertuples():
        f.write(">%s\n%s\n" % (r.did, r.seq))
rng = np.random.default_rng(0)
q = rng.choice(len(d), 2000, replace=False)
np.save(H + "/qidx.npy", q)
with open(H + "/query.fasta", "w") as f:
    for i in q:
        f.write(">%s\n%s\n" % (d.did[i], d.seq[i]))
print("gallery", len(d), "queries", len(q), "first query", d.did[q[0]])
