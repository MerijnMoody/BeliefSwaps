"""Ground truth for the factored processes: PSR rank vs GBS dimension, and how nonlinear the joint prediction is.

For each process: linear rank of the centred next-token / 2-step future sets (= PSR dimension seen in that window),
number of PCs for 95 / 99 / 99.9% of the variance, and the R^2 of the best linear (affine) map from the K factor
marginals to the joint next-token distribution (1 - R^2 = share of the PSR variance that is nonlinear in the factor
coordinates). Also R^2 of an affine map from the K factor log-odds to the next-token log-probabilities (exactly 1 for
a product process: logits are additive over factors).
"""
import sys

import numpy as np
import torch

import fact_proc as fp

dev = 'cuda'


def r2(X, Y):
    X1 = np.c_[X, np.ones(len(X))]
    coef, *_ = np.linalg.lstsq(X1, Y, rcond=None)
    res = Y - X1 @ coef
    return float(1 - (res ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def ncomp(X, fracs=(0.95, 0.99, 0.999)):
    s = np.linalg.svd(X - X.mean(0), compute_uv=False) ** 2
    c = np.cumsum(s) / s.sum()
    return [int(np.searchsorted(c, f) + 1) for f in fracs], int((s / s[0] > 1e-12).sum())


def report(name, K, c, q, e, t=24, n=20000):
    T = fp.transition_tensor(K, q, e, c)
    g = torch.Generator(device=dev).manual_seed(0)
    toks = fp.sample(T, n, t, g, dev)
    bel = fp.beliefs(T, toks)[:, t]
    m = fp.factor_marginals(bel, K).cpu().numpy()
    nx = fp.next_probs(T, bel).cpu().numpy()
    f2 = fp.futures(T, bel, 2).cpu().numpy()
    lo = np.log(m / (1 - m))
    print(f'{name}: K={K} c={c} q={q[:K]} e={e[:K]}')
    print(f'   next-token: PCs 95/99/99.9% {ncomp(nx)[0]}, rank {ncomp(nx)[1]} | 2-step: PCs {ncomp(f2)[0]}, rank {ncomp(f2)[1]}')
    print(f'   R2 affine(factor marginals -> joint next probs) {r2(m, nx):.3f} | affine(factor log-odds -> next log-probs) '
          f'{r2(lo, np.log(nx)):.4f} | affine(marginals -> 2-step) {r2(m, f2):.3f}')
    print(f'   factor marginal sd {np.round(m.std(0), 3)} | corr between factor log-odds (max |r|) '
          f'{np.abs(np.corrcoef(lo.T) - np.eye(K)).max():.3f}', flush=True)


if __name__ == '__main__':
    for K in (3, 4):
        report('product', K, 0.0, fp.DEFAULT_Q, fp.DEFAULT_E)
        for c in (0.3, 0.6):
            report('coupled', K, c, fp.DEFAULT_Q, fp.DEFAULT_E)
    for q, e in (((0.1,) * 4, (0.2,) * 4), ((0.02,) * 4, (0.05,) * 4)):
        report('alt', 4, 0.0, q, e)
