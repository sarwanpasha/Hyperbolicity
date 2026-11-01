"""Top-1 homologous-superfamily retrieval for the alignment and profile baselines.

Reads the HMMER tblout files and the Smith-Waterman chunks, takes each query's
best non-self hit, and reports accuracy overall and split by how remote the
query's closest true homolog is.
"""
import glob, json, os
import numpy as np, pandas as pd

H = os.environ["HYP_ROOT"]
d = pd.read_parquet(os.environ["HYP_DATA"] + "/cath.parquet").reset_index(drop=True)
lab = d.cath_h.to_numpy()
pos = {q: i for i, q in enumerate(d.did.to_numpy())}
size = d.groupby("cath_h").cath_h.transform("size").to_numpy()

mi = np.zeros(len(d), dtype=np.float32)
got = np.zeros(len(d), dtype=bool)
for f in sorted(glob.glob(H + "/results/maxident/mi_*.npz")):
    z = np.load(f)
    mi[z["idx"]] = z["val"]; got[z["idx"]] = True
print("maxident coverage %d/%d" % (got.sum(), len(d)))

sing = size < 2
bins = [("lt20", (mi < .2) & ~sing), ("b2030", (mi >= .2) & (mi < .3) & ~sing),
        ("ge30", (mi >= .3) & ~sing), ("nohom", sing)]


def tbl_best(pattern):
    best = {}
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            if line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 6:
                continue
            t, q = p[0], p[2]
            if t == q or t not in pos or q not in pos:
                continue
            try:
                ev = float(p[4])
            except ValueError:
                continue
            if q not in best or ev < best[q][1]:
                best[q] = (t, ev)
    return best


def score(best_map, name, out):
    qi = np.array([pos[q] for q in best_map], dtype=int)
    ok = np.zeros(len(d), dtype=bool)
    has = np.zeros(len(d), dtype=bool)
    for q, (t, _) in best_map.items():
        i = pos[q]; has[i] = True; ok[i] = lab[i] == lab[pos[t]]
    qs = np.load(H + "/hmm/qidx.npy")
    qm = np.zeros(len(d), dtype=bool); qm[qs] = True
    rec = {"n_queries": int(qm.sum()), "with_hit": int((has & qm).sum()),
           "all": float(ok[qm].mean())}
    for k, m in bins:
        s = qm & m
        rec[k] = float(ok[s].mean()) if s.sum() else None
        rec["n_" + k] = int(s.sum())
    out[name] = rec
    print("%-10s all=%.4f lt20=%s b2030=%s ge30=%s n=%d hits=%d" %
          (name, rec["all"], rec["lt20"], rec["b2030"], rec["ge30"],
           rec["n_queries"], rec["with_hit"]))


out = {}
score(tbl_best(H + "/hmm/phmmer_*.tbl"), "phmmer", out)
score(tbl_best(H + "/hmm/jack_*.tbl"), "jackhmmer", out)

swb = {}
for f in sorted(glob.glob(H + "/results/sw/sw_*.npz")):
    z = np.load(f)
    for a, b in zip(z["q"], z["t"]):
        swb[d.did[int(a)]] = (d.did[int(b)], 0.0)
if swb:
    score(swb, "smith_waterman", out)

out["strata_sizes"] = {k: int(m.sum()) for k, m in bins}
json.dump(out, open(H + "/results/baselines_cpu.json", "w"), indent=1)
print("wrote results/baselines_cpu.json")
