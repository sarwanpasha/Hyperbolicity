"""The S20 remote-homology benchmark.

Gallery: the CATH v4.4.0 S20 non-redundant domains present in our S35 gallery. No two
of them share 20% sequence identity, so every correct retrieval is a remote homolog by
construction. Queries: every gallery domain whose homologous superfamily has another
member in the gallery. A balanced variant keeps one query per superfamily.
"""
import os
import numpy as np, pandas as pd
D = os.environ["HYP_DATA"]; S = os.environ["HYP_ROOT"] + "/s20"
os.makedirs(S + "/chunks", exist_ok=True)
d = pd.read_parquet(D + "/cath.parquet").reset_index(drop=True)
keep = set(x.strip() for x in open(D + "/S20.list") if x.strip())
gal = np.where(d.did.isin(keep).to_numpy())[0]
sub = d.iloc[gal]
cnt = sub.cath_h.value_counts()
multi = set(cnt[cnt >= 2].index)
q_all = np.array([i for i in gal if d.cath_h[i] in multi])
rng = np.random.default_rng(0)
q_one = np.array(sorted(rng.choice(q_all[d.cath_h.to_numpy()[q_all] == h], 1)[0]
                        for h in sorted(multi)))
np.savez(S + "/s20_index.npz", gallery=gal, q_all=q_all, q_one=q_one)
with open(S + "/s20_gallery.fasta", "w") as f:
    for i in gal:
        f.write(">%s\n%s\n" % (d.did[i], d.seq[i]))
recs = [">%s\n%s\n" % (d.did[i], d.seq[i]) for i in q_all]
N = 20; per = (len(recs) + N - 1) // N
for k in range(N):
    open(S + "/chunks/q%02d.fasta" % k, "w").write("".join(recs[k*per:(k+1)*per]))
print("gallery", len(gal), "queries_all", len(q_all), "queries_one_per_sf", len(q_one),
      "superfams", len(multi))
