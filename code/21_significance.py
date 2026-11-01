"""Paired significance tests between two runs, using per-query retrieval outcomes.

Point estimates on 37k queries look precise but the runs are paired (same queries,
same gallery), so the honest comparison is a paired one: McNemar's exact test on the
discordant pairs, plus a paired bootstrap CI on the accuracy difference.
"""
import argparse, glob, json, itertools, os
import numpy as np
from scipy.stats import binomtest
from common import RESULTS, get_logger, save_json

log = get_logger("signif")


def paired(a, b, n_boot=10000, seed=0):
    a = np.asarray(a).astype(bool); b = np.asarray(b).astype(bool)
    assert a.shape == b.shape, (a.shape, b.shape)
    n01 = int((~a & b).sum())   # b right, a wrong
    n10 = int((a & ~b).sum())   # a right, b wrong
    mc = binomtest(n01, n01 + n10, 0.5).pvalue if (n01 + n10) else 1.0
    rng = np.random.default_rng(seed)
    d = b.astype(float) - a.astype(float)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(1)
    return {"acc_a": float(a.mean()), "acc_b": float(b.mean()),
            "diff": float(b.mean() - a.mean()),
            "ci_lo": float(np.percentile(boot, 2.5)),
            "ci_hi": float(np.percentile(boot, 97.5)),
            "n_b_only": n01, "n_a_only": n10, "mcnemar_p": float(mc),
            "n": int(len(a))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline run name (perq_<name>.npz)")
    ap.add_argument("--b", required=True, help="treatment run name")
    ap.add_argument("--keys", nargs="*", default=["cath_cath_h_top1", "cath_cath_h_top5",
                                                  "cath_cath_t_top1",
                                                  "test_superfam_top1", "test_superfam_top5",
                                                  "test_fold_top1"])
    a = ap.parse_args()
    A = np.load(RESULTS / ("perq_" + a.a + ".npz"))
    B = np.load(RESULTS / ("perq_" + a.b + ".npz"))
    out = {}
    for k in a.keys:
        if k not in A.files or k not in B.files:
            log.info("skip %s (missing)" % k); continue
        r = paired(A[k], B[k])
        out[k] = r
        log.info("%-22s %.4f -> %.4f  diff=%+.4f [%+.4f,%+.4f] mcnemar p=%.2e (n=%d)"
                 % (k, r["acc_a"], r["acc_b"], r["diff"], r["ci_lo"], r["ci_hi"],
                    r["mcnemar_p"], r["n"]))
    save_json(out, RESULTS / ("signif_%s__vs__%s.json" % (a.a, a.b)))
    log.info("SIGNIF_DONE")
