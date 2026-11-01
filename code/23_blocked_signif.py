"""Cluster-robust significance. Plain McNemar treats leave-one-out queries as
independent; they cluster by structural group (shared gallery, one displaced
centroid flips a whole clade). Block-bootstrap and permute over GROUPS instead."""
import argparse, glob, json
import numpy as np, pandas as pd
from common import RESULTS, PROC, get_logger, save_json
log = get_logger("blocked")
SPEC = {"test_superfam_top1":("scope","superfam"), "test_superfam_top5":("scope","superfam"),
        "test_fold_top1":("scope","fold"), "test_fold_top5":("scope","fold"),
        "cath_cath_h_top1":("cath","cath_h"), "cath_cath_h_top5":("cath","cath_h"),
        "cath_cath_t_top1":("cath","cath_t")}
def blocks():
    m = pd.read_parquet(PROC/"scope_splits.parquet")
    return {"scope": m[m.split=="test"].reset_index(drop=True),
            "cath": pd.read_parquet(PROC/"cath.parquet").reset_index(drop=True)}
def run(a, b, nboot, nperm):
    fa, fb = RESULTS/("perq_%s.npz"%a), RESULTS/("perq_%s.npz"%b)
    if not (fa.exists() and fb.exists()):
        log.warning("missing %s or %s", a, b); return
    A, B, BL = np.load(fa), np.load(fb), blocks()
    out = {"a":a, "b":b, "metrics":{}}
    rng = np.random.default_rng(0)
    for k,(ds,lvl) in SPEC.items():
        if k not in A.files or k not in B.files: continue
        ya, yb = A[k].astype(float), B[k].astype(float)
        df = BL[ds]
        if len(df) != len(ya):
            log.warning("%s: meta %d vs arr %d -- SKIP", k, len(df), len(ya)); continue
        g = df[lvl].to_numpy()
        d = yb - ya
        uq, inv = np.unique(g, return_inverse=True); G = len(uq); n = len(d)
        obs = d.mean()
        sums = np.bincount(inv, weights=d, minlength=G)
        null = np.array([(sums*(rng.integers(0,2,G)*2-1)).sum()/n for _ in range(nperm)])
        p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (nperm + 1)
        idx = [np.where(inv==i)[0] for i in range(G)]
        bs = np.empty(nboot)
        for i in range(nboot):
            bs[i] = d[np.concatenate([idx[j] for j in rng.integers(0,G,G)])].mean()
        lo, hi = np.percentile(bs,2.5), np.percentile(bs,97.5)
        cen = np.bincount(inv, weights=d-obs, minlength=G)
        deff = float((cen**2).sum()/n**2 / (d.var(ddof=1)/n)) if d.var(ddof=1)>0 else float("nan")
        out["metrics"][k] = {"diff":float(obs), "ci_lo":float(lo), "ci_hi":float(hi),
            "p_cluster":float(p), "n":n, "n_groups":G, "design_effect":deff,
            "eff_n":float(n/deff), "fixed":int((d>0).sum()), "broken":int((d<0).sum())}
        log.info("%-20s %+.4f CI[%+.4f,%+.4f] p_cluster=%.3g  n=%d G=%d deff=%.1f eff_n=%.0f",
                 k, obs, lo, hi, p, n, G, deff, n/deff)
    save_json(out, RESULTS/("blocked_%s__vs__%s.json"%(a,b)))
ap = argparse.ArgumentParser()
ap.add_argument("--a"); ap.add_argument("--b"); ap.add_argument("--all", action="store_true")
ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--perm", type=int, default=10000)
x = ap.parse_args()
if x.all:
    names = sorted(p.stem[5:] for p in RESULTS.glob("perq_*.npz"))
    for nm in names:
        if "_ft_lh1_s" not in nm: continue
        mdl, sd = nm.split("_")[1], nm.split("_s")[-1]
        for m2 in names:
            if m2.startswith("lora_%s_hyp_lh30_s%s"%(mdl,sd)):
                log.info("=== %s vs %s ===", nm, m2); run(nm, m2, x.boot, x.perm)
else:
    run(x.a, x.b, x.boot, x.perm)
log.info("BLOCKED_DONE")
