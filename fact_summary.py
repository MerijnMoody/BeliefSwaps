"""Seed-averaged summary of results/fact/{gbs,swap,min}_*.json."""
import glob
import json
import re
from collections import defaultdict

import numpy as np


def load(pat):
    groups = defaultdict(list)
    for f in sorted(glob.glob(pat)):
        name = f.split('/')[-1]
        key = re.sub(r'_s\d+', '', name.replace('.json', ''))
        groups[key].append(json.load(open(f)))
    return groups


def m(v):
    return float(np.mean(v))


print('=== GBS bottleneck: KL(exact 2-step future || decoded), mean over seeds; dim = smallest d with mlp KL <= 2x its '
      'minimum')
for key, rs in load('results/fact/gbs_*.json').items():
    K = rs[0]['K']
    print(f'-- {key} ({len(rs)} seeds) K={K}, PSR rank {2 ** K - 1}, model 2-step KL {m([r["model_kl_2step"] for r in rs]):.1e}')
    for src in rs[0]['bottleneck']:
        rows = [r['bottleneck'][src] for r in rs]
        ds = list(rows[0]['mlp'].keys())
        mlp = {d: m([r['mlp'][d] for r in rows]) for d in ds}
        lin = {d: m([r['lin'][d] for r in rows]) for d in ds}
        soft = {d: m([r['soft'][d] for r in rows]) for d in ds}
        best = min(mlp.values())
        dim = next(int(d) for d in ds if mlp[d] <= 2 * best)
        lin_dim = next((int(d) for d in ds if lin[d] <= 2 * best + 1e-6), None)
        print(f'   {src:13s} mlp dim {dim:2d} | KL at d=K: mlp {mlp[str(K)]:.1e} soft {soft[str(K)]:.1e} lin {lin[str(K)]:.1e}'
              f' | lin at max d {lin[ds[-1]]:.1e} (lin reaches mlp-level at d={lin_dim})')
    print('   probes (R2, held-out): ' + ' | '.join(
        f"{s}: joint {m([r['probes'][s]['joint_belief'] for r in rs]):.2f} factors {m([r['probes'][s]['factor_marg'] for r in rs]):.2f}"
        for s in ['p1', 'p2', 'p3', 'p4', 'all']))
    for p in rs[0]['factor_swaps']:
        f = lambda mode, k: m([np.mean(r['factor_swaps'][p][mode][k]) for r in rs])
        print(f"   swap {p}: specificity ICA {f('ica_pattern', 'specificity'):.2f} vs random {f('random_pattern', 'specificity'):.2f}"
              f" | distinct bits ICA {f('ica_pattern', 'distinct_bits'):.1f}/{K} random {f('random_pattern', 'distinct_bits'):.1f}"
              f" | transfer r ICA {f('ica_pattern', 'transfer_matched'):.2f} random {f('random_pattern', 'transfer_matched'):.2f}"
              f" | R2 transfer all-K {f('ica_pattern', 'r2_transfer_allK'):.2f} (pinv {f('ica_pinv', 'r2_transfer_allK'):.2f})"
              f" vs whole vector {f('ica_pattern', 'r2_transfer_full_vector'):.2f}")

print('\n=== Swap groups / plateaus / output continuum')
for key, rs in load('results/fact/swap_*.json').items():
    disc = rs[0]['proc'] != 'tele'
    K = rs[0]['K']
    print(f"-- {key} ({len(rs)} seeds) distinct beliefs {rs[0]['n_distinct']}")
    for c in rs[0]['cells']:
        g = lambda k: m([r['cells'][c][k] for r in rs])
        s = (f"   {c:8s}: {g('n_groups'):5.1f} groups, sep {g('separation'):.2f}, H(joint|C) {g('H_joint_given_C'):.2f} "
             f"H(C|joint) {g('H_C_given_joint'):.2f} | " + ' '.join(f"H(f{k}|C) {g(f'H_f{k}_given_C'):.2f}" for k in range(K)))
        if disc:
            pr = lambda k: m([r['cells'][c]['product'][k] for r in rs])
            s += f" | product: gA {pr('n_gA'):.1f} x gB {pr('n_gB'):.1f}, H(C|prod) {pr('H_C_given_prod'):.2f}"
        else:
            s += ' | eta2 ' + ' '.join(f"{g(f'eta2_f{k}'):.2f}" for k in range(K))
        print(s)
    for p in rs[0]['interp']:
        parts = []
        for nd in rs[0]['interp'][p]:
            hist = defaultdict(int)
            for r in rs:
                for j, v in r['interp'][p].get(nd, {}).get('jumps_hist', {}).items():
                    hist[int(j)] += v
            tot = sum(hist.values())
            frac = ' '.join(f"{j}:{hist[j] / tot:.2f}" for j in sorted(hist))
            spr = [r['interp'][p][nd]['alpha_star_spread_q10_50_90'][1] for r in rs
                   if 'alpha_star_spread_q10_50_90' in r['interp'][p].get(nd, {})]
            parts.append(f"{nd.split('_')[0]} differ: jumps {frac}" + (f" (alpha* spread median {m(spr):.2f})" if spr else ''))
        print(f"   interp {p}: " + ' | '.join(parts))
    co = [r['continuum'] for r in rs]
    print(f"   continuum: clean next-token PCs 99.9% prob {m([c['model_prob_pcs'][1] for c in co]):.1f} log "
          f"{m([c['model_log_pcs'][1] for c in co]):.1f}; ICA best |corr| per comp "
          f"{np.round(np.mean([np.max(c['ica_corr'], 1) for c in co if 'ica_corr' in c and len(c['ica_corr']) == len(co[0]['ica_corr'])], 0), 2)}")

print('\n=== Minimality (per layer, last position)')
for key, rs in load('results/fact/min_*.json').items():
    print(f"-- {key} ({len(rs)} seeds) process: " + json.dumps({k: (np.round(v, 3).tolist() if isinstance(v, (float, list)) else v)
                                                                for k, v in rs[0]['process'].items()}))
    for p in rs[0]['layers']:
        g = lambda k: m([r['layers'][p][k] for r in rs])
        s = (f"   {p}: R2 belief {g('R2_full'):.3f} additive {g('R2_add'):.3f} interaction share {g('interaction_share'):.3f}"
             f" +3 tokens {g('R2_tok'):.3f}")
        if 'eta2_exact' in rs[0]['layers'][p]:
            s += f" eta2 {g('eta2_exact'):.3f}"
        s += f" | PCs95 R {m([r['layers'][p]['pcs_R'][0] for r in rs]):.1f} belief part {m([r['layers'][p]['pcs_belief_part'][0] for r in rs]):.1f}"
        if 'cmi' in rs[0]['layers'][p]:
            s += ' | I(R;x_t-l|b) ' + ' '.join(f"l{l}:{m([r['layers'][p]['cmi'][l]['cmi'] for r in rs]):.3f}"
                                               for l in rs[0]['layers'][p]['cmi'])
        print(s)
    if 'cmi_calibration_R_is_belief' in rs[0]:
        print('   calibration: ' + ' '.join(f"l{l}:{m([r['cmi_calibration_R_is_belief'][l] for r in rs]):.3f}"
                                         for l in rs[0]['cmi_calibration_R_is_belief']))
