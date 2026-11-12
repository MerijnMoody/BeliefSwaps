"""Continuous read-out of what a swapped component carries: SVD of patched futures (MRRXOR / RRXOR).

For a component c (layer p at the last history position t, or layer p at all history positions 1..t), recipient i
and donor j: F[i, j] = the model's exact K-step future distribution (2^K probabilities, teacher-forced enumeration)
after j's activations at c are put into i's forward pass (as in cache_swap.py). For the true process the future is
linear in the belief (future = belief @ M), so the futures span at most rank(M) dimensions.

Per recipient i, over the 200 donors j:
  - effective rank of the centred matrix F[i, :, :] (number of singular directions for 99% / 99.9% of the variance)
  - label-free coordinates = top-k singular directions; R^2 of a linear fit from them to the donor's exact belief,
    exact future and next-token probability (k = 1..5). Labels are used only for this scoring.
  - R^2 of a linear fit of F[i, :, :] on the donor's own clean model future (how much of j's prediction gets through).
References (same scoring): the model's own clean futures over the 200 contexts (full transfer), and the exact process
futures. Also rank(M).

    python fut_svd.py --arch L4_d64_rope --eps 0.05 --seed 0 [--untrained] [--save]
"""
import argparse
import importlib.util
import json
import os

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('rp', os.path.join(_here, 'rrxor_plateau.py'))
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)
_spec2 = importlib.util.spec_from_file_location('cs', os.path.join(_here, 'cache_swap.py'))
cs = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(cs)
dev = 'cuda'
KS = (1, 2, 3, 4, 5, 8)


def r2(X, Y):
    """R^2 (pooled over Y's columns) of a linear fit with intercept of Y on X."""
    X1 = np.c_[X, np.ones(len(X))]
    coef, *_ = np.linalg.lstsq(X1, Y, rcond=None)
    res = Y - X1 @ coef
    tot = ((Y - Y.mean(0)) ** 2).sum()
    return float(1 - (res ** 2).sum() / tot) if tot > 0 else float('nan')


def eff_rank(X, frac):
    s = np.linalg.svd(X - X.mean(0), compute_uv=False) ** 2
    if s.sum() <= 0:
        return 0
    return int(np.searchsorted(np.cumsum(s) / s.sum(), frac) + 1)


def scores(X, labels):
    """X (n, d) futures over donors. Effective ranks and R^2 of top-k SVD coordinates for each label."""
    Xc = X - X.mean(0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    out = dict(rank99=eff_rank(X, 0.99), rank999=eff_rank(X, 0.999),
               sv_share=[float(v) for v in (S ** 2 / max((S ** 2).sum(), 1e-300))[:8]])
    for nm, Y in labels.items():
        out[f'r2_{nm}'] = {k: r2(U[:, :k] * S[:k], Y) for k in KS}
    return out


def mean_dict(ds):
    out = {}
    for k in ds[0]:
        v = [d[k] for d in ds]
        if isinstance(v[0], dict):
            out[k] = {kk: float(np.mean([x[kk] for x in v])) for kk in v[0]}
        elif isinstance(v[0], list):
            out[k] = np.mean(np.array(v), 0).tolist()
        else:
            out[k] = float(np.mean(v))
    return out


@torch.no_grad()
def patched_futures(m, hist, t, p, pos, A, K, br, rec_bt=8):
    n = len(hist)
    F = torch.empty(n, n, 2 ** K, dtype=torch.float64)
    for s in range(0, n, rec_bt):
        rec = torch.arange(s, min(s + rec_bt, n), device=dev)
        i = rec.repeat_interleave(n)
        j = torch.arange(n, device=dev).repeat(len(rec))
        lp = cs.flp(m, hist[i], t, p, pos, (A[j] - A[i]).float(), K, br)
        F[s:s + len(rec)] = lp.exp().view(len(rec), n, -1).cpu()
    return F.numpy()


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--arch', default='L4_d64_rope')
    pa.add_argument('--eps', default='0.05')
    pa.add_argument('--seed', type=int, default=0)
    pa.add_argument('--K', type=int, default=4)
    pa.add_argument('--ts', default='14,28')
    pa.add_argument('--untrained', action='store_true')
    pa.add_argument('--save', action='store_true', help='save patched futures (float32 npz)')
    a = pa.parse_args()
    ck = torch.load(os.path.join(rp.REPO, 'runs', a.arch, f'eps{a.eps}', f's{a.seed}.pt'), map_location='cpu',
                    weights_only=True)
    cfg = rp.Config(**ck['config'])
    if a.untrained:
        m = rp.untrained(cfg, float(ck['args']['init_std']), int(ck['args']['seed']))
    else:
        m = rp.Transformer(cfg); m.load_state_dict(ck['model'])
    m = m.to(dev).eval()
    for prm in m.parameters():
        prm.requires_grad_(False)
    K = a.K
    T = rp.transition_tensor(float(a.eps))
    br = torch.cartesian_prod(*[torch.arange(2, device=dev)] * K)
    Mx = rp.futures(T, torch.eye(5, dtype=torch.float64), K).numpy()        # (5 states, 2^K)
    toks = rp.sample(T, 200, cfg.context, torch.Generator().manual_seed(4242))
    bel = rp.beliefs(T, toks)
    hist = toks.to(dev)
    tag = ('_untrained' if a.untrained else '')
    out = dict(arch=a.arch, eps=a.eps, seed=a.seed, untrained=a.untrained, K=K,
               rank_M=int(np.linalg.matrix_rank(Mx - Mx.mean(0), tol=1e-10)), cells={})
    print(f'rank of the centred exact future map M (5 states x {2 ** K}): {out["rank_M"]}', flush=True)
    saved = {}
    for t in map(int, a.ts.split(',')):
        b = bel[:, t].numpy()
        labels = dict(belief=b, future=rp.futures(T, bel[:, t], K).numpy(),
                      next=(b @ T[1].sum(1).numpy())[:, None])
        zero = torch.zeros(200, cfg.d_model, device=dev)
        clean = cs.flp(m, hist, t, cfg.n_layers, t, zero, K, br).exp().double().cpu().numpy()
        out['cells'][f't{t}_clean_model'] = scores(clean, labels)
        out['cells'][f't{t}_exact_process'] = scores(labels['future'], labels)
        for nm in ('clean_model', 'exact_process'):
            c = out['cells'][f't{t}_{nm}']
            print(f"t{t} {nm}: rank99 {c['rank99']} rank999 {c['rank999']} | R2 belief by k "
                  + ' '.join(f"{k}:{v:.2f}" for k, v in c['r2_belief'].items()), flush=True)
        for p in range(1, cfg.n_layers + 1):
            for mode, pos in (('last', t), ('slice', slice(1, t + 1))):
                _, A = rp.run(m, hist[:, :t + 1], collect=(p, pos))
                F = patched_futures(m, hist, t, p, pos, A, K, br)
                per = []
                for i in range(200):
                    sc = scores(F[i], labels)
                    sc['r2_donor_clean'] = r2(clean, F[i])
                    per.append(sc)
                row = mean_dict(per)
                key = f't{t}_p{p}_{mode}'
                out['cells'][key] = row
                if a.save:
                    saved[key] = F.astype(np.float32)
                print(f"{key}: rank99 {row['rank99']:.1f} rank999 {row['rank999']:.1f} | donor-future R2 "
                      f"{row['r2_donor_clean']:.2f} | R2 belief by k "
                      + ' '.join(f"{k}:{v:.2f}" for k, v in row['r2_belief'].items())
                      + f" | next (k=1) {row['r2_next'][1]:.2f}", flush=True)
        json.dump(out, open(f'results/plateau/futsvd_{a.arch}_eps{a.eps}_s{a.seed}{tag}.json', 'w'), indent=1)
    if a.save:
        np.savez_compressed(f'results/plateau/futsvd_{a.arch}_eps{a.eps}_s{a.seed}{tag}_F.npz', **saved)


if __name__ == '__main__':
    main()
