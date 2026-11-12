"""Summarize the RRXOR plateau runs over seeds (rrxor_plateau.py outputs) and the pooled-positions head-to-head
(hh_pool.py outputs). Per architecture and residual point: mean (min-max over seeds) of the per-seed median over
history lengths.

    python agg_rrxor.py
"""
import glob
import json
import re

import numpy as np

ARCHS = ('L4_d64', 'L4_d64_rope', 'L2_d32', 'L2_d32_rope')
VARS = ('belief', 'last2', 'xor2', 'phase', 'ph_xor')


def fmt(v):
    v = np.asarray(v, dtype=float)
    return f"{v.mean():.2f} ({v.min():.2f}-{v.max():.2f})" if len(v) > 1 else f"{v.mean():.2f}"


def plateau():
    for arch in ARCHS:
        for tag in ('', '_untrained'):
            fs = sorted(f for f in glob.glob(f'results/plateau/rrxor_{arch}_eps0_s[0-9]{tag}.json'))
            if not fs:
                continue
            runs = [json.load(open(f)) for f in fs]
            L = max(int(re.match(r'p(\d+)_', k).group(1)) for k in runs[0]['points'])
            print(f"\n== {arch}{tag or ' trained'}: {len(runs)} seeds; test_kl " +
                  ' '.join(f"{r['test_kl']:.1e}" for r in runs))
            print("   point | groups | " + ' | '.join(f"{v}: H(V|C) H(C|V)" for v in VARS))
            for p in range(L + 1):
                def med(f):
                    return [np.median([f(r['points'][k]) for k in r['points'] if k.startswith(f'p{p}_')])
                            for r in runs]
                cells = [f"{v}: {fmt(med(lambda x: x['scores'][v]['V_given_C']))} "
                         f"{fmt(med(lambda x: x['scores'][v]['C_given_V']))}" for v in VARS]
                print(f"   p{p} | {fmt(med(lambda x: x['n_clusters']))} | " + ' | '.join(cells))
            for k in runs[0]['transitions']:
                print(f"   transition {k}: {fmt([np.median(r['transitions'][k]) for r in runs])}")


def pooled():
    for arch in ARCHS:
        fs = sorted(glob.glob(f'results/plateau/hhpool_{arch}_s[0-9]_n300.json'))
        if not fs:
            continue
        runs = [json.load(open(f)) for f in fs]
        print(f"\n== pooled positions {arch}: {len(runs)} seeds (error at tau 1e-2; lower is better)")
        for p in runs[0]['points']:
            R = [r['points'][p] for r in runs]
            g = lambda f: fmt([f(x) for x in R])
            print(f"   {p}: eta2 belief {g(lambda x: x['eta2']['belief'])} position {g(lambda x: x['eta2']['position'])}"
                  f" | groups ours {g(lambda x: x['n_groups_A']['ours'])} pca {g(lambda x: x['n_groups_A']['pca'])}")
            print("      error " + ' | '.join(f"{k} {g(lambda x, k=k: x['eval'][k]['tau0.01']['error'])}"
                                         for k in ('ours', 'pca', 'oracle_belief', 'oracle_position',
                                                   'oracle_belief_x_position', 'all_separate')))
            print("      ARI belief ours " + g(lambda x: x['ari_vs_belief']['ours']) + " pca " +
                  g(lambda x: x['ari_vs_belief']['pca']) + " | ARI position ours " +
                  g(lambda x: x['ari_vs_position']['ours']) + " pca " + g(lambda x: x['ari_vs_position']['pca']))
            print("      swap KL > 1e-2: " + ' | '.join(
                f"{k} {g(lambda x, k=k: x['swap_cost_split'][k]['frac_gt_1em2'])}"
                for k in ('sameB_sameP', 'sameB_diffP', 'diffB_sameP', 'diffB_diffP')) +
                  f" | probe position acc {g(lambda x: x['probe_position_acc'])}")


if __name__ == '__main__':
    plateau()
    pooled()
