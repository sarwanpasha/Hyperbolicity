"""For every CATH domain, the maximum sequence identity to another domain of the
same homologous superfamily. Used to stratify retrieval by remoteness. For
superfamilies with more than CAP members we compare every member against a random
reference subset of CAP members, which lower-bounds the true maximum."""
import numpy as np, pandas as pd, parasail
from collections import defaultdict
from common import PROC, RESULTS, get_logger
log = get_logger('ident')
CAP = 40
MAT = parasail.blosum62
d = pd.read_parquet(PROC / 'cath.parquet').reset_index(drop=True)
seqs = d.seq.to_numpy(); grp = d.cath_h.to_numpy()
best = np.zeros(len(d), dtype=np.float32)
rng = np.random.default_rng(0)
idx = defaultdict(list)
for i, g in enumerate(grp):
    idx[g].append(i)
for n, (g, members) in enumerate(idx.items()):
    if len(members) < 2:
        continue
    ref = members if len(members) <= CAP else list(rng.choice(members, CAP, replace=False))
    for a in members:
        la = len(seqs[a])
        for b in ref:
            if a == b:
                continue
            r = parasail.sw_stats_striped_16(seqs[a], seqs[b], 11, 1, MAT)
            f = r.matches / max(1, min(la, len(seqs[b])))
            if f > best[a]:
                best[a] = f
    if n % 1000 == 0:
        log.info('group %d/%d', n, len(idx))
np.savez_compressed(RESULTS / 'cath_maxident.npz', maxident=best, did=d.did.to_numpy())
sing = np.array([len(idx[g]) < 2 for g in grp])
log.info('done; singletons %d; among non-singletons: median %.3f frac<0.20 %.3f frac<0.30 %.3f',
         int(sing.sum()), float(np.median(best[~sing])),
         float((best[~sing] < 0.20).mean()), float((best[~sing] < 0.30).mean()))
