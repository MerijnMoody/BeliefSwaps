"""Pooled-positions head-to-head: swap groups vs PCA + k-means when contexts at different positions are pooled.

Position carries a large share of the residual variance in the RRXOR models (hh_explore.py: 36-43% at p1-p2 for
L4_d64), but for an eps = 0 RRXOR source the optimal future depends on the belief only, not on the position. If a layer
no longer uses position downstream, PCA should split by position and the swap test should not. If a layer still uses
position (e.g. positional attention lookups later on), the swap test should split by position too, and that would be
the right answer.

Contexts: n per set, each assigned a history length t_c from POOL (round robin). Cross-position swap: context j's
activation at (p, t_j) replaces context i's activation at (p, t_i); distance = max over both directions of
KL(i's clean exact K-step future || swapped future). Groups from Otsu + complete linkage as before.
Scoring on a held-out set B with the causal swap test (harmful_within / harmless_between, as in hh.py), plus ARI
against belief and position, eta^2 of belief and position, and the swap cost split by (same belief?, same position?).

    python hh_pool.py --arch L4_d64 --seed 0
"""
import argparse
import importlib.util
import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import adjusted_rand_score as ari

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('hh', os.path.join(_here, 'hh.py'))
hh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hh)
rp = hh.rp
dev = 'cuda'
POOL = (6, 9, 14, 20)


@torch.no_grad()
def acts(m, hist, tc, p):
    A = torch.zeros(len(hist), m.cfg.d_model, device=dev, dtype=torch.float64)
    for t in POOL:
        idx = np.where(tc == t)[0]
        _, a = rp.run(m, hist[idx, :t + 1], collect=(p, t))
        A[idx] = a.double()
    return A


@torch.no_grad()
def clean_lp(m, hist, tc, p, K):
    lp = torch.zeros(len(hist), 2 ** K, device=dev, dtype=torch.float64)
    for t in POOL:
        idx = np.where(tc == t)[0]
        lp[idx] = hh.lp_with(m, hist[idx], t, p, torch.zeros(len(idx), m.cfg.d_model, device=dev), K)
    return lp


@torch.no_grad()
def inject_kl(m, hist, tc, p, A_rec, A_don, lp0, K):
    """M[i, j] = KL(clean future of recipient i || future with donor activation j put into i at (p, t_i))."""
    n, nd = len(hist), len(A_don)
    M = torch.zeros(n, nd, device=dev, dtype=torch.float64)
    for t in POOL:
        R = np.where(tc == t)[0]
        ii = torch.tensor(R, device=dev).repeat_interleave(nd)
        jj = torch.arange(nd, device=dev).repeat(len(R))
        for s in range(0, len(ii), 2048):
            i, j = ii[s:s + 2048], jj[s:s + 2048]
            lpa = hh.lp_with(m, hist[i], t, p, A_don[j] - A_rec[i], K)
            M[i, j] = (lp0[i].exp() * (lp0[i] - lpa)).sum(1)
    return M.cpu().numpy()


def eta2(A, v):
    A = A - A.mean(0)
    tot = (A ** 2).sum()
    return float(sum(((A[v == c].mean(0)) ** 2).sum() * (v == c).sum() for c in np.unique(v)) / tot)


def make_set(T, n, ctx, seed, K):
    toks = rp.sample(T, n, ctx, torch.Generator().manual_seed(seed))
    bel = rp.beliefs(T, toks)
    tc = np.array([POOL[i % len(POOL)] for i in range(n)])
    b = torch.stack([bel[i, tc[i]] for i in range(n)]).numpy()
    x = toks.numpy()
    last2 = np.array([x[i, tc[i] - 2] * 2 + x[i, tc[i] - 1] for i in range(n)])
    bid = rp.ids(b)
    lab = dict(belief=bid, position=tc.copy(), belief_x_position=rp.ids(np.stack([bid, tc], 1), 0), last2=last2,
               last2_x_position=rp.ids(np.stack([last2, tc], 1), 0))
    return toks.to(dev), tc, lab, b


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--arch', default='L4_d64')
    pa.add_argument('--seed', type=int, default=0)
    pa.add_argument('--n', type=int, default=300)
    pa.add_argument('--K', type=int, default=4)
    a = pa.parse_args()
    ck = torch.load(os.path.join(rp.REPO, 'runs', a.arch, 'eps0', f's{a.seed}.pt'), map_location='cpu',
                    weights_only=True)
    cfg = rp.Config(**ck['config'])
    m = rp.Transformer(cfg)
    m.load_state_dict(ck['model'])
    m = m.to(dev).eval()
    for prm in m.parameters():
        prm.requires_grad_(False)
    T = rp.transition_tensor(0.)
    K, n, L = a.K, a.n, cfg.n_layers
    assert max(POOL) + K <= cfg.context
    hA, tcA, labA, bA = make_set(T, n, cfg.context, 4242, K)
    hB, tcB, labB, bB = make_set(T, n, cfg.context, 999, K)
    rng = np.random.default_rng(0)
    out = dict(arch=a.arch, seed=a.seed, n=n, K=K, pool=POOL, points={})
    for p in range(L + 1):
        AA, AB = acts(m, hA, tcA, p), acts(m, hB, tcB, p)
        lpA, lpB = clean_lp(m, hA, tcA, p, K), clean_lp(m, hB, tcB, p, K)
        MA = inject_kl(m, hA, tcA, p, AA, AA, lpA, K)
        MB = inject_kl(m, hB, tcB, p, AB, AB, lpB, K)
        DA, DB = np.maximum(MA, MA.T), np.maximum(MB, MB.T)
        ours_A, thr = hh.groups_from(DA)
        med = []
        for g in np.unique(ours_A):
            idx = np.where(ours_A == g)[0]
            med.append(idx[np.argmin(DA[np.ix_(idx, idx)].sum(1))])
        klm = inject_kl(m, hB, tcB, p, AB, AA[med], lpB, K)
        ours_B = np.unique(ours_A)[klm.argmin(1)]
        XA, XB = AA.cpu().numpy(), AB.cpu().numpy()
        pcaA, pcaB, r, kpca = hh.pca_groups(XA, XB)
        groupings = dict(ours=ours_B, pca=pcaB, **{f'oracle_{k}': v for k, v in labB.items()},
                         one_group=np.zeros(n, int), all_separate=np.arange(n))
        # swap cost split by (same belief?, same position?) on B
        iu = np.triu_indices(n, 1)
        sb = labB['belief'][iu[0]] == labB['belief'][iu[1]]
        sp = labB['position'][iu[0]] == labB['position'][iu[1]]
        dB = DB[iu]
        split = {}
        for nm, msk in (('sameB_sameP', sb & sp), ('sameB_diffP', sb & ~sp), ('diffB_sameP', ~sb & sp),
                        ('diffB_diffP', ~sb & ~sp)):
            split[nm] = dict(n=int(msk.sum()), median=float(np.median(dB[msk])) if msk.any() else None,
                             q90=float(np.quantile(dB[msk], 0.9)) if msk.any() else None,
                             frac_gt_1em2=float((dB[msk] > 1e-2).mean()) if msk.any() else None)
        pos_acc = LogisticRegression(max_iter=3000).fit(XA, labA['position']).score(XB, labB['position'])
        bel_R2 = Ridge(1e-3).fit(XA, bA).score(XB, bB)
        row = dict(threshold=thr, n_groups_A=dict(ours=int(len(np.unique(ours_A))), pca=int(kpca)), pca_dims=r,
                   n_beliefs_B=int(len(np.unique(labB['belief']))),
                   eta2=dict(belief=eta2(XB, labB['belief']), position=eta2(XB, labB['position']),
                             last2=eta2(XB, labB['last2'])),
                   eval={k: hh.causal_eval(DB, v) for k, v in groupings.items()},
                   eval_in_sample_A=dict(ours=hh.causal_eval(DA, ours_A), pca=hh.causal_eval(DA, pcaA), oracle_belief_x_position=hh.causal_eval(DA, labA["belief_x_position"])),
                   ari_vs_belief={k: float(ari(labB['belief'], v)) for k, v in groupings.items()},
                   ari_vs_position={k: float(ari(labB['position'], v)) for k, v in groupings.items()},
                   ari_vs_last2={k: float(ari(labB['last2'], v)) for k, v in groupings.items()},
                   swap_cost_split=split, probe_position_acc=float(pos_acc), probe_belief_R2=float(bel_R2))
        out['points'][f'p{p}'] = row
        e = row['eval']
        print(f"p{p}: eta2 belief {row['eta2']['belief']:.2f} position {row['eta2']['position']:.2f} | groups ours "
              f"{row['n_groups_A']['ours']} pca {kpca} (r={r}) | error@1e-2 " +
              ' '.join(f"{k} {v['tau0.01']['error']:.3f}" for k, v in e.items()) +
              "\n   ARI belief " + ' '.join(f"{k} {row['ari_vs_belief'][k]:.2f}" for k in ('ours', 'pca')) +
              " | ARI position " + ' '.join(f"{k} {row['ari_vs_position'][k]:.2f}" for k in ('ours', 'pca')) +
              " | swap KL median " + ' '.join(f"{k} {v['median']:.1e}" for k, v in split.items()) +
              f" | probe pos acc {pos_acc:.2f} belief R2 {bel_R2:.2f}", flush=True)
    os.makedirs('results/plateau', exist_ok=True)
    json.dump(out, open(f'results/plateau/hhpool_{a.arch}_s{a.seed}_n{n}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
