"""Score every trained arm and every alignment baseline on the S20 benchmark."""
import glob, json, os, re
import numpy as np, pandas as pd, scipy.stats as st
H = os.environ["HYP_ROOT"]
d = pd.read_parquet(os.environ["HYP_DATA"] + "/cath.parquet").reset_index(drop=True)
lab = d.cath_h.to_numpy(); pos = {q: i for i, q in enumerate(d.did.to_numpy())}
z = np.load(H + "/s20/s20_index.npz")
gal, q_all, q_one = z["gallery"], z["q_all"], z["q_one"]
gset = set(gal.tolist())


def emb_top1(Z, queries):
    G = Z[gal].astype(np.float32)
    G /= np.linalg.norm(G, axis=1, keepdims=True) + 1e-9
    gp = {int(g): k for k, g in enumerate(gal)}
    ok = []
    for s in range(0, len(queries), 1024):
        qi = queries[s:s + 1024]
        Q = Z[qi].astype(np.float32)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9
        S = Q @ G.T
        for r, i in enumerate(qi):
            S[r, gp[int(i)]] = -9
        nn = gal[S.argmax(1)]
        ok.extend((lab[nn] == lab[qi]).tolist())
    return np.array(ok)


out = {"n_gallery": int(len(gal)), "n_q_all": int(len(q_all)), "n_q_one": int(len(q_one)),
       "runs": {}, "baselines": {}}
for f in sorted(glob.glob(H + "/data/emb/ftcath_*.npy")):
    name = os.path.basename(f)[7:-4]
    Z = np.load(f)
    ra = emb_top1(Z, q_all); ro = emb_top1(Z, q_one)
    rj = H + "/results/runs/run_%s.json" % name
    info = json.load(open(rj)) if os.path.exists(rj) else {}
    out["runs"][name] = {"s20_all": float(ra.mean()), "s20_one": float(ro.mean()),
                         "cath_full": info.get("cath_h_top1"),
                         "test_sf": info.get("test_sf_top1"),
                         "test_drel": info.get("test_drel")}


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
            if t == q or t not in pos or q not in pos or pos[t] not in gset:
                continue
            try:
                ev = float(p[4])
            except ValueError:
                continue
            if q not in best or ev < best[q][1]:
                best[q] = (t, ev)
    return best


def score_map(bm):
    ok = {}
    for q, (t, _) in bm.items():
        ok[pos[q]] = lab[pos[q]] == lab[pos[t]]
    a = np.array([ok.get(int(i), False) for i in q_all])
    o = np.array([ok.get(int(i), False) for i in q_one])
    return {"s20_all": float(a.mean()), "s20_one": float(o.mean()),
            "with_hit": int(sum(int(i) in ok for i in q_all))}


for nm, pat in [("phmmer", H + "/s20/phmmer_*.tbl"), ("jackhmmer", H + "/s20/jack_*.tbl")]:
    if glob.glob(pat):
        out["baselines"][nm] = score_map(tbl_best(pat))
swb = {}
for f in sorted(glob.glob(H + "/results/s20sw/sw_*.npz")):
    zz = np.load(f)
    for a_, b_ in zip(zz["q"], zz["t"]):
        swb[d.did[int(a_)]] = (d.did[int(b_)], 0.0)
if swb:
    out["baselines"]["smith_waterman"] = score_map(swb)

# per scale: mean over seeds, and paired penalty minus task-only
agg = {}
for name, r in out["runs"].items():
    m = re.match(r"(\w+?)_(ft|hyp)_lh[\d.]+_s(\d+)", name)
    if not m:
        continue
    agg.setdefault((m.group(1), m.group(2)), {})[int(m.group(3))] = r
summary = {}
for sc in ["35M", "150M", "650M", "3B"]:
    ft = agg.get((sc, "ft"), {}); hy = agg.get((sc, "hyp"), {})
    rec = {}
    for k in ["s20_all", "s20_one", "cath_full", "test_sf", "test_drel"]:
        fv = [ft[s][k] for s in sorted(ft) if ft[s].get(k) is not None]
        hv = [hy[s][k] for s in sorted(hy) if hy[s].get(k) is not None]
        rec[k] = {"ft": float(np.mean(fv)) if fv else None, "ft_sd": float(np.std(fv, ddof=1)) if len(fv) > 1 else None,
                  "hyp": float(np.mean(hv)) if hv else None, "hyp_sd": float(np.std(hv, ddof=1)) if len(hv) > 1 else None}
        common = sorted(set(ft) & set(hy))
        if len(common) >= 2:
            dd = np.array([100 * (hy[s][k] - ft[s][k]) for s in common if hy[s].get(k) is not None])
            if len(dd) >= 2 and dd.std() > 0:
                lo, hi = st.t.interval(0.95, len(dd) - 1, loc=dd.mean(), scale=st.sem(dd))
                rec[k].update({"delta_pts": float(dd.mean()), "ci": [float(lo), float(hi)],
                               "p": float(st.ttest_1samp(dd, 0).pvalue), "n": int(len(dd)),
                               "same_sign": bool(np.all(dd > 0) or np.all(dd < 0))})
    rec["seeds_ft"] = sorted(ft); rec["seeds_hyp"] = sorted(hy)
    summary[sc] = rec
out["summary"] = summary
json.dump(out, open(H + "/results/s20_eval.json", "w"), indent=1)
print("gallery", out["n_gallery"], "q_all", out["n_q_all"], "q_one", out["n_q_one"], "runs", len(out["runs"]))
for nm, b in out["baselines"].items():
    print("BASELINE %-15s all %.4f one %.4f hits %d" % (nm, b["s20_all"], b["s20_one"], b["with_hit"]))
for sc, rec in summary.items():
    r = rec["s20_all"]
    if r.get("ft") is not None:
        print("%-5s S20 FT %.4f -> hyp %s  d=%s CI=%s p=%s | seeds ft%s hyp%s" % (
            sc, r["ft"], "%.4f" % r["hyp"] if r.get("hyp") else "-", "%+.2f" % r["delta_pts"] if "delta_pts" in r else "-",
            r.get("ci"), "%.3g" % r["p"] if "p" in r else "-", rec["seeds_ft"], rec["seeds_hyp"]))
