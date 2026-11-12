"""Share of variance explained by cache_swap.py groups for continuous (eps > 0) beliefs: belief, exact 4-step future,
next-token probability; baselines: shuffled groups and k-means on the belief with the same number of groups.

    python mrrxor_cache.py --arch L4_d64_rope --eps 0.05
"""
import argparse, glob, importlib.util, json, os
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
    return float(sum(((X[lab == c].mean(0)) ** 2).sum() * (lab == c).sum() for c in np.unique(lab)) / tot)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--arch', default='L4_d64_rope')
    pa.add_argument('--eps', default='0.05')
    a = pa.parse_args()
    T = rp.transition_tensor(float(a.eps))
    toks = rp.sample(T, 200, 32, torch.Generator().manual_seed(4242))
    bel = rp.beliefs(T, toks)
    rng = np.random.default_rng(0)
    res = {}
    for tag in ('', '_untrained'):
        fs = sorted(glob.glob(f'results/plateau/cache_{a.arch}_eps{a.eps}_s[0-9]{tag}.json'))
        if not fs:
            continue
        runs = [json.load(open(f)) for f in fs]
        print(f'\n== {a.arch} eps {a.eps}{tag or " trained"}: {len(runs)} seeds | groups, harmful share, eta2 belief '
              f'(shuffled / kmeans same k), eta2 future, eta2 next-token prob')
        for key in runs[0]['cells']:
            t = int(key.split('_')[0][1:])
            b = bel[:, t].numpy()
            fut = rp.futures(T, bel[:, t], 4).numpy()
            nxt = (b @ T[1].sum(1).numpy())[:, None]
            R = []
            for r in runs:
                lab = np.array(r['cells'][key]['labels'])
                k = int(lab.max())
                km = KMeans(k, n_init=5, random_state=0).fit(b).labels_ if 1 < k < 200 else lab
                R.append([k, r['cells'][key]['frac_harmful'], eta2(b, lab),
                          np.mean([eta2(b, rng.permutation(lab)) for _ in range(20)]), eta2(b, km), eta2(fut, lab),
                          eta2(nxt, lab)])
            mu = np.array(R).mean(0)
            res[f'{tag or "trained"}_{key}'] = mu.tolist()
            print(f'   {key}: {mu[0]:.1f} groups, harmful {mu[1]:.2f} | belief {mu[2]:.2f} (shuffled {mu[3]:.2f}, '
                  f'kmeans {mu[4]:.2f}) | future {mu[5]:.2f} | next {mu[6]:.2f}', flush=True)
    json.dump(res, open(f'results/plateau/mrrxor_cache_{a.arch}_eps{a.eps}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
