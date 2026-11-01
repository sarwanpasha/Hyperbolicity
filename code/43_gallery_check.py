"""How much does the gallery change when it is restricted to domains with structures?

The embedding arms were scored against the full CATH gallery; Foldseek can only
search the 22,931 domains that have a structure. This script measures the size of
that difference on the representations we have stored, so the comparison in the
paper can be put on one gallery.
"""
import os
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from common import PROC, EMB, RESULTS, get_logger

log = get_logger("galchk")
LAYER = {"35M": 10, "150M": 19, "650M": 17}
S = os.environ["HYP_DATA"] + "/struct"
dev = "cuda" if torch.cuda.is_available() else "cpu"


def top1(X, lab, gal, q):
    gi = np.where(gal)[0]
    G = X[gi]
    qi = np.where(q)[0]
    ok = np.zeros(len(qi), dtype=bool)
    gpos = {int(g): j for j, g in enumerate(gi)}
    for s in range(0, len(qi), 512):
        idx = qi[s:s + 512]
        Sm = X[idx] @ G.T
        for r, qq in enumerate(idx):
            j = gpos.get(int(qq))
            if j is not None:
                Sm[r, j] = -2
        nn = gi[Sm.argmax(1).cpu().numpy()]
        ok[s:s + 512] = lab[nn] == lab[idx]
    return float(ok.mean())


if __name__ == "__main__":
    d = pd.read_parquet(PROC / "cath.parquet").reset_index(drop=True)
    lab = d.cath_h.to_numpy()
    keep = set(x.strip() for x in open(S + "/overlap.list") if x.strip())
    mask = d.did.isin(keep).to_numpy()
    sub = d[mask]
    cnt = sub.cath_h.value_counts()
    multi = set(cnt[cnt >= 2].index)
    retr = mask & np.array([x in multi for x in lab])
    full = np.ones(len(d), dtype=bool)
    log.info("queries %d, full gallery %d, subset gallery %d",
             retr.sum(), full.sum(), mask.sum())
    for s in ["35M", "150M", "650M"]:
        p = EMB / ("cath_%s.npy" % s)
        if not p.exists():
            continue
        ids = np.load(EMB / ("cath_%s_ids.npy" % s), allow_pickle=True)
        arr = np.load(p, mmap_mode="r")
        if arr.ndim == 3:
            arr = np.asarray(arr[LAYER[s]])
        arr = np.asarray(arr)
        order = {str(q): j for j, q in enumerate(ids)}
        take = np.array([order[q] for q in d.did.to_numpy()])
        arr = arr[take]
        X = F.normalize(torch.tensor(arr, dtype=torch.float32), dim=1).to(dev)
        a = top1(X, lab, full, retr)
        b = top1(X, lab, mask, retr)
        log.info("frozen %-5s full=%.4f subset=%.4f diff=%+.4f", s, a, b, b - a)
        del X
