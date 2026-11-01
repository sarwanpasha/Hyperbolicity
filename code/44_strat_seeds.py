"""CATH retrieval by remoteness, averaged over seeds rather than read off one run.

The earlier stratified table reported seed 0 only while the text drew conclusions
from it. Every arm stores a per-query correctness array over the 37,169 CATH
domains, so the strata can be recomputed for every seed at no training cost, and
the per-stratum difference can be tested with the seed as the unit of replication.
"""
import glob, json, os, re
import numpy as np, pandas as pd
from scipy import stats
from common import PROC, RESULTS, get_logger

log = get_logger("strat2")
SC = ["35M", "150M", "650M", "3B"]


def load(name):
    p = RESULTS / ("perq_%s.npz" % name)
    if not p.exists():
        return None
    z = np.load(p)
    return z["cath_cath_h_top1"] if "cath_cath_h_top1" in z.files else None


def seeds_for(scale, arm):
    pat = str(RESULTS / ("perq_lora_%s_%s.npz" % (scale, arm)))
    out = {}
    for f in glob.glob(pat.replace("SEED", "*")):
        m = re.search(r"_s(\d)(?:_topq)?\.npz$", f)
        if m:
            out[int(m.group(1))] = os.path.basename(f)[5:-4]
    return out


def main():
    d = pd.read_parquet(PROC / "cath.parquet").reset_index(drop=True)
    lab = d.cath_h.to_numpy()
    size = d.groupby("cath_h").cath_h.transform("size").to_numpy()
    z = np.load(RESULTS / "cath_maxident.npz")["maxident"]
    sing = size < 2
    bins = [("lt20", (z < .2) & ~sing),
            ("b2030", (z >= .2) & (z < .3) & ~sing),
            ("ge30", (z >= .3) & ~sing),
            ("all", np.ones(len(d), dtype=bool))]
    log.info("stratum sizes: %s", {k: int(m.sum()) for k, m in bins})

    out = {"strata": {k: int(m.sum()) for k, m in bins}, "scales": {}}
    for sc in SC:
        ft = seeds_for(sc, "ft_lh1_sSEED")
        hy = seeds_for(sc, "hyp_lh10_sSEED_topq")
        common = sorted(set(ft) & set(hy))
        if not common:
            continue
        rec = {"seeds": common, "n_seeds": len(common)}
        for key, m in bins:
            a, b = [], []
            for s in common:
                fa, ha = load(ft[s]), load(hy[s])
                if fa is None or ha is None:
                    continue
                a.append(float(fa[m].mean()))
                b.append(float(ha[m].mean()))
            if len(a) < 2:
                continue
            diff = 100.0 * (np.array(b) - np.array(a))
            t, p = stats.ttest_rel(b, a)
            lo, hi = stats.t.interval(0.95, len(diff) - 1,
                                      loc=diff.mean(),
                                      scale=stats.sem(diff)) if diff.std() > 0 else (diff.mean(), diff.mean())
            rec[key] = {"ft": float(np.mean(a)), "ft_sd": float(np.std(a, ddof=1)),
                        "hyp": float(np.mean(b)), "hyp_sd": float(np.std(b, ddof=1)),
                        "delta_pts": float(diff.mean()), "delta_sd": float(diff.std(ddof=1)),
                        "ci": [float(lo), float(hi)], "p": float(p),
                        "all_same_sign": bool(np.all(diff > 0) or np.all(diff < 0))}
        out["scales"][sc] = rec
    with open(RESULTS / "strat_seeds.json", "w") as f:
        json.dump(out, f, indent=1)
    for sc, rec in out["scales"].items():
        for k in ["lt20", "b2030", "ge30", "all"]:
            if k in rec:
                r = rec[k]
                log.info("%-5s %-6s n=%d FT %.4f -> %.4f  d=%+.2f+-%.2f CI[%+.2f,%+.2f] p=%.3g same_sign=%s",
                         sc, k, rec["n_seeds"], r["ft"], r["hyp"], r["delta_pts"],
                         r["delta_sd"], r["ci"][0], r["ci"][1], r["p"], r["all_same_sign"])


if __name__ == "__main__":
    main()
