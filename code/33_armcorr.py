"""Does cross-database transfer track the delta an arm reaches, or the loss that reached it?

Every 35M arm at a given seed contributes one point (drel on SCOPe, SCOPe SF top-1,
CATH H top-1). We report Spearman correlation between drel and CATH transfer, its
bootstrap CI over arms, and the partial correlation controlling for SCOPe SF top-1,
which replaces the post-hoc "retains accuracy" filter with a continuous control.
"""
import json, glob, os, re
import numpy as np
from scipy.stats import spearmanr, rankdata
from common import RESULTS, get_logger, save_json

log = get_logger("armcorr")

PATTERNS = [
    ("penalty",  "run_lora_35M_hyp_lh%s_s%%d_topq.json", ["1", "3", "10", "30", "100", "300"]),
    ("cpcc",     "run_lora_35M_cpcc_lh%s_s%%d.json",      ["0.1", "1", "10"]),
    ("unif",     "run_lora_35M_unif_lh%s_s%%d.json",      ["0.1", "1", "10"]),
    ("drift",    "run_lora_35M_drift_lh%s_s%%d.json",     ["0.1", "1", "10", "100", "1000", "10000"]),
]
EXTRA = [("r2", "run_lora_35M_ft_lh1_s%d_r2.json"),
         ("s200", "run_lora_35M_ft_lh1_s%d_s200.json"),
         ("poincare", "run_lora_35M_poincare_lh1_s%d.json")]
BASE = "run_lora_35M_ft_lh1_s%d.json"

def load(p):
    f = RESULTS / p
    if not f.exists():
        return None
    d = json.load(open(f))
    return (d["test"]["delta_rel"], d["test"]["superfam_top1"], d["cath"]["cath_h_top1"])

def arms_for(seed):
    out = []
    for fam, pat, lams in PATTERNS:
        for L in lams:
            v = load((pat % L) % seed)
            if v: out.append((fam, L, v))
    for fam, pat in EXTRA:
        v = load(pat % seed)
        if v: out.append((fam, "-", v))
    return out

def partial_spearman(x, y, z):
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    def resid(a, b):
        b1 = np.c_[np.ones(len(b)), b]
        return a - b1 @ np.linalg.lstsq(b1, a, rcond=None)[0]
    return float(np.corrcoef(resid(rx, rz), resid(ry, rz))[0, 1])

def boot(fn, n, rng, nb=10000):
    v = [fn(rng.integers(0, n, n)) for _ in range(nb)]
    v = np.array([x for x in v if np.isfinite(x)])
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

if __name__ == "__main__":
    out = {}
    for seed in (0, 1, 2):
        A = arms_for(seed)
        b = load(BASE % seed)
        if b is None or len(A) < 8:
            log.warning("seed %d: only %d arms, base=%s", seed, len(A), b is not None); continue
        dr = np.array([a[2][0] for a in A]); sf = np.array([a[2][1] for a in A])
        h1 = np.array([a[2][2] for a in A])
        dc = (h1 - b[2]) * 100.0
        keep = sf >= b[1] - 0.01
        rng = np.random.default_rng(0)
        r_all = float(spearmanr(dr, dc).statistic)
        ci_all = boot(lambda i: spearmanr(dr[i], dc[i]).statistic, len(dr), rng)
        r_k = float(spearmanr(dr[keep], dc[keep]).statistic)
        ci_k = boot(lambda i: spearmanr(dr[keep][i], dc[keep][i]).statistic, int(keep.sum()), rng)
        r_p = partial_spearman(dr, dc, sf)
        ci_p = boot(lambda i: partial_spearman(dr[i], dc[i], sf[i]), len(dr), rng)
        out["seed%d" % seed] = dict(n_arms=len(A), n_keep=int(keep.sum()),
            base_drel=b[0], base_sf=b[1], base_h1=b[2],
            rho_all=r_all, ci_all=ci_all, rho_keep=r_k, ci_keep=ci_k,
            rho_partial=r_p, ci_partial=ci_p,
            arms=[dict(fam=a[0], lam=a[1], drel=a[2][0], sf=a[2][1], h1=a[2][2]) for a in A])
        log.info("seed %d n=%d keep=%d | rho_all %+.3f CI[%+.3f,%+.3f] | rho_keep %+.3f CI[%+.3f,%+.3f] | rho_partial %+.3f CI[%+.3f,%+.3f]",
                 seed, len(A), int(keep.sum()), r_all, ci_all[0], ci_all[1],
                 r_k, ci_k[0], ci_k[1], r_p, ci_p[0], ci_p[1])
    save_json(out, RESULTS / "armcorr.json")
    log.info("ARMCORR_DONE")
