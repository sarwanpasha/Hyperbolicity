"""Dump every 35M arm as (delta_rel, SCOPe SF top-1, CATH H top-1) for the mechanism plot."""
import json, glob, os
R = os.environ['HYP_ROOT'] + '/results'
fams = [('task', ['run_lora_35M_ft_lh1_s0.json']),
        ('hyp', sorted(glob.glob(R + '/run_lora_35M_hyp_lh*_s0_topq.json'))),
        ('cpcc', sorted(glob.glob(R + '/run_lora_35M_cpcc_lh*_s0.json'))),
        ('unif', sorted(glob.glob(R + '/run_lora_35M_unif_lh*_s0.json'))),
        ('drift', sorted(glob.glob(R + '/run_lora_35M_drift_lh*_s0.json'))),
        ('capacity', [R + '/run_lora_35M_ft_lh1_s0_r2.json',
                      R + '/run_lora_35M_ft_lh1_s0_s200.json']),
        ('poincare', [R + '/run_lora_35M_poincare_lh1_s0.json'])]
out = {}
for fam, fs in fams:
    pts = []
    for f in fs:
        f = f if f.startswith('/') else R + '/' + f
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        pts.append({'name': os.path.basename(f), 'drel': d['test']['delta_rel'],
                    'sf': d['test']['superfam_top1'], 'cath': d['cath']['cath_h_top1']})
    out[fam] = pts
    for p in pts:
        print('%-9s %-42s drel %.4f sf %.4f cath %.4f' % (fam, p['name'], p['drel'], p['sf'], p['cath']))
json.dump(out, open(R + '/arms35M.json', 'w'), indent=1)
