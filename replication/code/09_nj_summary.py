import glob, json, os, collections
import numpy as np, scipy.stats as st
H = os.environ["HYP_ROOT"]
rows = []
for f in glob.glob(H + "/results/synthnj/nj_*.json"):
    rows += json.load(open(f))
print("rows", len(rows), "files", len(glob.glob(H + "/results/synthnj/nj_*.json")))
agg = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    for k in ("rf", "quartet", "purity"):
        agg[r["method"]][k].append(r[k])
for m in ("frozen", "ft", "hyp", "poincare"):
    a = agg[m]
    print("%-9s RF %.4f+-%.4f  quartet %.4f+-%.4f  purity %.4f+-%.4f  n=%d" % (
        m, np.mean(a["rf"]), np.std(a["rf"]), np.mean(a["quartet"]), np.std(a["quartet"]),
        np.mean(a["purity"]), np.std(a["purity"]), len(a["rf"])))
byc = {}
for r in rows:
    byc.setdefault((r["depth"], r["rate"], r["seed"]), {})[r["method"]] = r
for k in ("quartet", "purity", "rf"):
    hf = [c["hyp"][k] - c["ft"][k] for c in byc.values()]
    ff = [c["ft"][k] - c["frozen"][k] for c in byc.values()]
    hz = [c["hyp"][k] - c["frozen"][k] for c in byc.values()]
    pf = [c["poincare"][k] - c["ft"][k] for c in byc.values()]
    print("%-8s ft-frozen %+.4f p=%.2g | hyp-ft %+.4f p=%.2g wins %d/%d | hyp-frozen %+.4f | poin-ft %+.4f" % (
        k, np.mean(ff), st.ttest_1samp(ff, 0).pvalue, np.mean(hf), st.ttest_1samp(hf, 0).pvalue,
        sum(x > 0 for x in hf) if k != "rf" else sum(x < 0 for x in hf), len(hf),
        np.mean(hz), np.mean(pf)))
print("--- by depth (quartet, hyp-ft) ---")
for dpt in (7, 9, 11):
    v = [c["hyp"]["quartet"] - c["ft"]["quartet"] for (d, r, s), c in byc.items() if d == dpt]
    w = [c["ft"]["quartet"] - c["frozen"]["quartet"] for (d, r, s), c in byc.items() if d == dpt]
    print("depth %d  ft-frozen %+.4f  hyp-ft %+.4f  n=%d" % (dpt, np.mean(w), np.mean(v), len(v)))
print("--- by rate (quartet, hyp-ft) ---")
for rt in (0.01, 0.02, 0.04):
    v = [c["hyp"]["quartet"] - c["ft"]["quartet"] for (d, r, s), c in byc.items() if r == rt]
    w = [c["ft"]["quartet"] - c["frozen"]["quartet"] for (d, r, s), c in byc.items() if r == rt]
    print("rate %.2f  ft-frozen %+.4f  hyp-ft %+.4f  n=%d" % (rt, np.mean(w), np.mean(v), len(v)))
