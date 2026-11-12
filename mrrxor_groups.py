"""How do swap groups relate to the continuous MRRXOR beliefs? Reads rrxor_plateau.py outputs for eps > 0
(results/plateau/rrxor_{arch}_eps{eps}_s{seed}[_untrained]_swaps.json) and regenerates the same 200 contexts.

Per (layer, history length): number of groups, Otsu separation, and the share of variance explained by the groups
(eta^2) for the belief (5-dim), the exact 4-step future distribution (16-dim) and the next-token probability; compared
with (a) shuffled groups of the same sizes and (b) k-means on the belief with the same number of groups (the best a
partition of that size can do for the belief). Also the mean within-group spread of the belief (L1) vs overall.

    python mrrxor_groups.py --eps 0.05
"""
import argparse
import glob
import importlib.util
import json
import os

import numpy as np
import torch
from sklearn.cluster import KMeans

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('rp', os.path.join(_here, 'rrxor_plateau.py'))
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)


def eta2(X, lab):
    X = X - X.mean(0)
    tot = (X ** 2).sum()
    return float(sum(((X[lab == c].mean(0)) ** 2).sum() * (lab == c).sum() for c in np.unique(lab)) / tot) if tot > 0 else np.nan


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--eps', default='0.05')
    a = pa.parse_args()
    T = rp.transition_tensor(float(a.eps))
    toks = rp.sample(T, 200, 32, torch.Generator().manual_seed(4242))
    bel = rp.beliefs(T, toks)
    rng = np.random.default_rng(0)
    res = {}
    for arch in ('L4_d64_rope', 'L4_d64'):
        for tag in ('', '_untrained'):
            fs = sorted(glob.glob(f'results/plateau/rrxor_{arch}_eps{a.eps}_s[0-9]{tag}_swaps.json'))
            if not fs:
                continue
            runs = [json.load(open(f)) for f in fs]
            print(f'\n== {arch} eps {a.eps}{tag or " trained"}: {len(runs)} seeds | per cell: groups, separation, '
                  f'eta2 belief (shuffled / kmeans same k), eta2 future, eta2 next-token prob')
            for key in runs[0]['points']:
                t = int(key.split('_t')[1])
                b = bel[:, t].numpy()
                fut = rp.futures(T, bel[:, t], 4).numpy()
                nxt = (b @ T[1].sum(1).numpy())[:, None]
                rows = []
                for r in runs:
                    lab = np.array(r['points'][key]['labels'])
                    k = int(lab.max())
                    sh = np.mean([eta2(b, rng.permutation(lab)) for _ in range(20)])
                    km = KMeans(k, n_init=5, random_state=0).fit(b).labels_ if 1 < k < 200 else lab
                    rows.append([k, r['points'][key]['separation'], eta2(b, lab), sh, eta2(b, km), eta2(fut, lab),
                                 eta2(nxt, lab)])
                R = np.array(rows, dtype=float)
                mu = R.mean(0)
                res[f'{arch}{tag}_{key}'] = dict(zip(['groups', 'separation', 'eta2_belief', 'eta2_belief_shuffled',
                                                      'eta2_belief_kmeans', 'eta2_future', 'eta2_next'], mu.tolist()))
                print(f'   {key}: {mu[0]:.1f} groups, sep {mu[1]:.2f} | belief {mu[2]:.2f} (shuffled {mu[3]:.2f}, '
                      f'kmeans {mu[4]:.2f}) | future {mu[5]:.2f} | next {mu[6]:.2f}')
    json.dump(res, open(f'results/plateau/mrrxor_groups_eps{a.eps}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
