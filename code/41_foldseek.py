"""Structure-based reference on the CATH subset that has structures.

Foldseek is run all-against-all over the 22,931 domains that our CATH S35 gallery
shares with the CATH S40 structure release. Every embedding arm is then scored on
exactly that subset, by masking the stored per-query correctness arrays, so the
comparison is on one gallery and one query set.
"""
import os, json, glob
import numpy as np, pandas as pd
from common import PROC, RESULTS, get_logger

log = get_logger("fsk")
S = os.environ["HYP_DATA"] + "/struct"


def norm_id(x, valid):
    """Foldseek names entries after the file; recover the CATH domain id."""
    b = os.path.basename(x)
    for suf in (".pdb", ".ent", ".cif"):
        if b.endswith(suf):
            b = b[: -len(suf)]
    if b in valid:
        return b
    if "_" in b:
        c = b.rsplit("_", 1)[0]
        if c in valid:
            return c
    return None


def main():
    d = pd.read_parquet(PROC / "cath.parquet").reset_index(drop=True)
    did = d.did.to_numpy()
    lab = d.cath_h.to_numpy()
    pos = {q: i for i, q in enumerate(did)}
    keep = [x.strip() for x in open(S + "/overlap.list") if x.strip()]
    valid = set(keep)
    mask = np.zeros(len(d), dtype=bool)
    for q in keep:
        mask[pos[q]] = True
    log.info("subset %d of %d domains", mask.sum(), len(d))

    best = {}
    nline = 0
    with open(S + "/fs_hits.tsv") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 6:
                continue
            nline += 1
            q = norm_id(p[0], valid)
            t = norm_id(p[1], valid)
            if q is None or t is None or q == t:
                continue
            b = float(p[5])
            cur = best.get(q)
            if cur is None or b > cur[1]:
                best[q] = (t, b)
    log.info("parsed %d alignment lines, %d queries with a hit", nline, len(best))

    fs_ok = np.zeros(len(d), dtype=bool)
    fs_has = np.zeros(len(d), dtype=bool)
    for q, (t, _) in best.items():
        i = pos[q]
        fs_has[i] = True
        fs_ok[i] = lab[i] == lab[pos[t]]

    sub = d[mask]
    cnt = sub.cath_h.value_counts()
    multi = set(cnt[cnt >= 2].index)
    retr = mask & np.array([x in multi for x in lab])
    log.info("retrievable in subset: %d", retr.sum())

    z = np.load(RESULTS / "cath_maxident.npz")["maxident"]
    bins = {"lt20": (z < .2), "b2030": (z >= .2) & (z < .3), "ge30": (z >= .3)}

    def summarize(ok):
        o = {"all": float(ok[mask].mean()), "n_all": int(mask.sum()),
             "retrievable": float(ok[retr].mean()), "n_retr": int(retr.sum())}
        for k, b in bins.items():
            m = retr & b
            o[k] = float(ok[m].mean()) if m.sum() else None
            o["n_" + k] = int(m.sum())
        return o

    out = {"foldseek": summarize(fs_ok),
           "foldseek_query_coverage": float(fs_has[mask].mean()),
           "subset_n": int(mask.sum()),
           "subset_superfams": int(sub.cath_h.nunique())}

    arms = {}
    for f in sorted(glob.glob(str(RESULTS / "perq_*.npz"))):
        name = os.path.basename(f)[5:-4]
        try:
            zz = np.load(f)
        except Exception:
            continue
        if "cath_cath_h_top1" in zz.files:
            ok = zz["cath_cath_h_top1"]
        elif name.endswith("_cath_h") and "a1" in zz.files:
            ok = zz["a1"]
        else:
            continue
        if ok.shape[0] != len(d):
            continue
        arms[name] = summarize(ok)
    log.info("scored %d embedding arms on the subset", len(arms))
    out["arms"] = arms
    with open(RESULTS / "foldseek_subset.json", "w") as f:
        json.dump(out, f, indent=1)
    log.info("Foldseek all=%.4f retrievable=%.4f coverage=%.3f",
             out["foldseek"]["all"], out["foldseek"]["retrievable"],
             out["foldseek_query_coverage"])


if __name__ == "__main__":
    main()
