"""SCOPe 2.08 ASTRAL at 40% identity, and the fold-disjoint splits.

Folds are assigned greedily to train, validation and test within each structural
class, so no fold is shared across splits and test domains come from folds never
seen in training.
"""
import os, re
import numpy as np, pandas as pd

D = os.environ["HYP_DATA"]
fa = D + "/astral-40-2.08.fa"
rows, did, seq, hdr = [], None, [], None
for line in open(fa):
    line = line.rstrip()
    if line.startswith(">"):
        if did and seq:
            rows.append((did, hdr, "".join(seq)))
        p = line[1:].split()
        did, hdr, seq = p[0], (p[1] if len(p) > 1 else ""), []
    else:
        seq.append(line)
if did and seq:
    rows.append((did, hdr, "".join(seq)))

recs = []
for d, h, s in rows:
    m = re.match(r"^([a-g])\.(\d+)\.(\d+)\.(\d+)$", h)
    if not m:
        continue
    cls = m.group(1)
    if cls not in "abcdef":
        continue
    if not (30 <= len(s) <= 1000):
        continue
    recs.append(dict(did=d, cls=cls, fold="%s.%s" % (m.group(1), m.group(2)),
                     superfam="%s.%s.%s" % (m.group(1), m.group(2), m.group(3)),
                     family=h, seq=s.upper(), length=len(s)))
df = pd.DataFrame(recs)
print("SCOPe domains", len(df), "folds", df.fold.nunique(),
      "superfams", df.superfam.nunique())

tgt = {"train": 0.666, "val": 0.101, "test": 0.233}
assign = {}
for cls, sub in df.groupby("cls"):
    tot = len(sub)
    have = {k: 0 for k in tgt}
    sizes = sub.groupby("fold").size()
    rng = np.random.default_rng(0)
    order = list(sizes.index)
    rng.shuffle(order)
    for fold in order:
        n = int(sizes[fold])
        deficit = {k: tgt[k] * tot - have[k] for k in tgt}
        k = max(deficit, key=deficit.get)
        assign[fold] = k
        have[k] += n
df["split"] = df.fold.map(assign)
print(df.groupby("split").agg(domains=("did", "size"), folds=("fold", "nunique")))
df.to_parquet(D + "/scope.parquet", index=False)
