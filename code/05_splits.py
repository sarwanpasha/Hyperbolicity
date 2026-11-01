"""Stage 5: deterministic fold-disjoint splits of SCOPe.

We split at the SCOPe *fold* level so that test domains belong to folds never seen
in training - the remote-homology regime. Splits are balanced within structural
class so each class keeps its share of held-out folds.
Output: PROC/scope_splits.parquet with column `split` in {train,val,test}.
"""
import numpy as np
import pandas as pd
from common import PROC, get_logger, save_json

log = get_logger("splits")
SEED = 17
FRAC = {"train": 0.70, "val": 0.10, "test": 0.20}


def main():
    df = pd.read_parquet(PROC / "scope.parquet")
    rng = np.random.default_rng(SEED)
    assign = {}
    for cls, sub in df.groupby("cls"):
        sizes = sub.groupby("fold").size().sort_values(ascending=False)
        folds = list(sizes.index)
        rng.shuffle(folds)
        total = sizes.sum()
        target = {k: v * total for k, v in FRAC.items()}
        got = {k: 0 for k in FRAC}
        # greedy: give each fold to the split furthest below its target share
        for f in folds:
            deficit = {k: (target[k] - got[k]) / max(target[k], 1) for k in FRAC}
            pick = max(deficit, key=deficit.get)
            assign[f] = pick
            got[pick] += sizes[f]
        log.info("class %s: %d folds %d domains -> %s" % (cls, len(folds), total, got))
    df["split"] = df["fold"].map(assign)
    df.to_parquet(PROC / "scope_splits.parquet", index=False)
    st = df.groupby("split").agg(n=("sid", "size"), folds=("fold", "nunique"),
                                 sfs=("superfam", "nunique")).to_dict("index")
    log.info("split summary %s" % st)
    # sanity: no fold appears in two splits
    chk = df.groupby("fold")["split"].nunique().max()
    assert chk == 1, "fold leakage detected"
    save_json(st, PROC / "split_stats.json")
    log.info("STAGE5_DONE")


if __name__ == "__main__":
    main()
