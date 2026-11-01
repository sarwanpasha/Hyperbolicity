"""Minimum detectable effect for the paired McNemar comparison."""
import argparse
import numpy as np
from scipy import stats
from common import RESULTS, get_logger, save_json
log = get_logger("mde")
KEYS = ["test_superfam_top1","test_superfam_top5","test_fold_top1","cath_cath_h_top1"]
def mde_exact(n, pid, alpha=0.05, power=0.80, nsim=3000, rng=None):
    rng = rng or np.random.default_rng(0)
    lo, hi = 0.0, min(pid, 0.10)
    for _ in range(18):
        mid = (lo+hi)/2
        p10, p01 = (pid+mid)/2, (pid-mid)/2
        if p01 < 0: hi = mid; continue
        rej = 0
        n10s = rng.binomial(n, p10, nsim); n01s = rng.binomial(n, p01, nsim)
        for a_, b_ in zip(n10s, n01s):
            m = a_+b_
            if m and stats.binomtest(int(a_), int(m), 0.5).pvalue < alpha: rej += 1
        if rej/nsim >= power: hi = mid
        else: lo = mid
    return (lo+hi)/2
ap = argparse.ArgumentParser(); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
x = ap.parse_args()
A = np.load(RESULTS/("perq_%s.npz"%x.a)); B = np.load(RESULTS/("perq_%s.npz"%x.b))
out = {"a":x.a,"b":x.b,"metrics":{}}
for k in KEYS:
    if k not in A.files or k not in B.files: continue
    ya, yb = A[k].astype(int), B[k].astype(int)
    n = len(ya)
    n10 = int(((yb==1)&(ya==0)).sum()); n01 = int(((yb==0)&(ya==1)).sum())
    pid = (n10+n01)/n
    m = mde_exact(n, pid)
    obs = float(yb.mean()-ya.mean())
    out["metrics"][k] = {"n":n,"fixed":n10,"broken":n01,"discordant_rate":pid,
                         "observed_diff":obs,"mde80":float(m),"below_mde":bool(abs(obs)<m)}
    log.info("%-20s n=%6d pi_d=%.4f obs=%+.4f  MDE80=%.4f  %s", k, n, pid, obs, m,
             "BELOW" if abs(obs)<m else "above")
save_json(out, RESULTS/("mde_%s__vs__%s.json"%(x.a,x.b)))
log.info("MDE_DONE")
