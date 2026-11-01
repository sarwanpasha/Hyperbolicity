"""Smith-Waterman nearest-neighbour baseline on CATH: an absolute reference point for
embedding retrieval. Samples QN queries and scans the full gallery."""
import numpy as np, pandas as pd, parasail, os, json
from multiprocessing import Pool
from common import PROC, RESULTS, get_logger
log = get_logger('align')
QN = 2000
d = pd.read_parquet(PROC / 'cath.parquet').reset_index(drop=True)
seqs = list(d.seq); lab = d.cath_h.to_numpy()
rng = np.random.default_rng(0)
q = rng.choice(len(d), QN, replace=False)
MAT = parasail.blosum62

def best(i):
    s = seqs[i]; bs = -1; bj = -1
    for j in range(len(seqs)):
        if j == i:
            continue
        r = parasail.sw_striped_16(s, seqs[j], 11, 1, MAT)
        if r.score > bs:
            bs = r.score; bj = j
    return i, bj

if __name__ == '__main__':
    with Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 8))) as p:
        res = p.map(best, list(q), chunksize=8)
    hit = np.array([lab[i] == lab[j] for i, j in res])
    z = np.load(RESULTS / 'cath_maxident.npz')['maxident']
    size = d.groupby('cath_h').cath_h.transform('size').to_numpy()
    qi = np.array([i for i, _ in res]); mi = z[qi]; sing = size[qi] < 2
    out = {'n': int(QN), 'top1': float(hit.mean()),
           'lt20': float(hit[(mi < .2) & ~sing].mean()),
           'b2030': float(hit[(mi >= .2) & (mi < .3) & ~sing].mean()),
           'ge30': float(hit[(mi >= .3) & ~sing].mean()),
           'n_lt20': int(((mi < .2) & ~sing).sum()),
           'n_2030': int(((mi >= .2) & (mi < .3) & ~sing).sum()),
           'n_ge30': int(((mi >= .3) & ~sing).sum()),
           'n_sing': int(sing.sum())}
    json.dump(out, open(RESULTS / 'align_baseline.json', 'w'), indent=1)
    log.info('SW top-1 %.4f | <20%% %.4f | 20-30%% %.4f | >=30%% %.4f | singletons %d',
             out['top1'], out['lt20'], out['b2030'], out['ge30'], out['n_sing'])
