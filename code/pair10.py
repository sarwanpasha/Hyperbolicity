import json, os, glob
R = os.environ['HYP_ROOT'] + '/results'
for M in ['35M', '150M', '650M', '3B']:
    for sd in [0, 1, 2, 3, 4]:
        f = '%s/signif_lora_%s_ft_lh1_s%d__vs__lora_%s_hyp_lh10_s%d_topq.json' % (R, M, sd, M, sd)
        if not os.path.exists(f):
            print(M, sd, 'MISSING'); continue
        d = json.load(open(f))
        ks = [k for k in d.keys()]
        out = []
        for k in ks:
            v = d[k]
            if isinstance(v, dict) and 'diff' in v:
                out.append('%s %+0.4f p=%.3g' % (k, v['diff'], v['mcnemar_p']))
        print(M, sd, ' | '.join(out))
