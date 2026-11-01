"""Stratify CATH homologous-superfamily retrieval by how remote the closest
true homolog of each query is."""
import argparse, numpy as np, pandas as pd
from scipy.stats import binomtest
from common import PROC, RESULTS, get_logger
log = get_logger('strat')
ap = argparse.ArgumentParser(); ap.add_argument('--runs', nargs='+', required=True)
a = ap.parse_args()
z = np.load(RESULTS / 'cath_maxident.npz')
mi = z['maxident']
d = pd.read_parquet(PROC / 'cath.parquet').reset_index(drop=True)
size = d.groupby('cath_h').cath_h.transform('size').to_numpy()
sing = size < 2
BINS = [('<20%', (mi < 0.20) & ~sing), ('20-30%', (mi >= 0.20) & (mi < 0.30) & ~sing),
        ('>=30%', (mi >= 0.30) & ~sing), ('singleton', sing)]
out = {}
for r in a.runs:
    p = np.load(RESULTS / ('perq_%s.npz' % r))['cath_cath_h_top1'].astype(bool)
    out[r] = p
    log.info('%-38s overall %.4f', r, p.mean())
for name, m in BINS:
    row = '%-10s n=%6d' % (name, int(m.sum()))
    for r in a.runs:
        row += '  %s %.4f' % (r.split('_')[2] + r.split('_')[3], out[r][m].mean())
    if len(a.runs) == 2:
        x, y = out[a.runs[0]][m], out[a.runs[1]][m]
        b = int((~x & y).sum()); c = int((x & ~y).sum())
        pv = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
        row += '  diff %+.4f  b=%d c=%d p=%.3g' % (y.mean() - x.mean(), b, c, pv)
    log.info(row)
