"""Search for flat directions by gradient descent (RRXOR L4_d64, artemyk/belief-state checkpoints).

At residual point p and history length t, a move A_i -> A_i + s v (unit v, position t only, everything else
recomputed as in rrxor_plateau.py) is "flat" if KL(clean exact K-step future of i || moved future) < the swap-group
threshold of that cell. The radius s is the median distance between contexts in different swap groups, i.e. a move as
large as jumping to another group.

1. Per context: for k in K_LIST, learn an orthonormal k-dim subspace V_i (orthogonal to the all-ones direction, which
   every pre-LayerNorm read removes) that minimizes the mean KL over its basis vectors and random unit combinations, at
   radius s. Evaluate on 64 fresh random unit directions in V_i: fraction flat.
2. Shared: one subspace V for all contexts (train on 100 contexts, evaluate on the other 100).
   Baselines: random subspaces, and the k lowest-variance principal directions of the activations.
   Also: share of activation variance inside V, and share of within-group / between-group differences inside V.
3. Untrained init: shared subspace only.
Sanity: the all-ones direction at radius s.

    python grad_flat.py --seed 0
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
dev = 'cuda'
K_LIST = (1, 2, 4, 8, 16, 32, 48, 60)


def flp(m, h, t, p, delta, K, br):
    """Differentiable (n, 2^K) log-probs of every K-token continuation with delta (n, D) added at (p, t)."""
    n, nb = len(h), len(br)
    seq = torch.cat([h[:, :t].repeat_interleave(nb, 0), br.repeat(n, 1)], 1)
    lg, _ = rp.run(m, seq, inj=(p, t, delta.repeat_interleave(nb, 0)))
    lp = torch.log_softmax(lg.double(), -1)
    tgt = br.repeat(n, 1)
    return sum(lp[:, t + j].gather(1, tgt[:, j:j + 1])[:, 0] for j in range(K)).view(n, nb)


def kl_of(m, h, t, p, delta, lp0, K, br, bt=4096):
    out = []
    for s in range(0, len(h), bt):
        lpa = flp(m, h[s:s + bt], t, p, delta[s:s + bt], K, br)
        c = lp0[s:s + bt]
        out.append((c.exp() * (c - lpa)).sum(1))
    return torch.cat(out)


def basis(W, ones):
    """Orthonormal rows spanning W's rows, orthogonal to the all-ones direction. W (..., k, D)."""
    W = W - (W @ ones)[..., None] * ones
    Q, _ = torch.linalg.qr(W.transpose(-1, -2))
    return Q.transpose(-1, -2)


def rand_unit(shape, gen=None):
    u = torch.randn(*shape, device=dev, dtype=torch.float64, generator=gen)
    return u / u.norm(dim=-1, keepdim=True)


def moves_in(V, n_comb, gen=None):
    """Basis vectors plus n_comb random unit combinations: (..., k + n_comb, D)."""
    u = rand_unit((*V.shape[:-2], n_comb, V.shape[-2]), gen)
    return torch.cat([V, u @ V], -2)


def per_context(m, hist, t, p, A, lp0, idx, s, thr, K, br, ones, steps):
    res = {}
    for k in K_LIST:
        W = torch.randn(len(idx), k, A.shape[1], device=dev, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([W], lr=0.05)
        for _ in range(steps):
            V = basis(W, ones)
            mv = moves_in(V, 4)                                   # (n, k + 4, D)
            nm = mv.shape[1]
            kl = kl_of(m, hist[idx].repeat_interleave(nm, 0), t, p, s * mv.reshape(-1, A.shape[1]),
                       lp0[idx].repeat_interleave(nm, 0), K, br)
            loss = kl.mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            V = basis(W, ones)
            u = rand_unit((len(idx), 64, k)) @ V                  # 64 fresh directions in each V_i
            kl = kl_of(m, hist[idx].repeat_interleave(64, 0), t, p, s * u.reshape(-1, A.shape[1]),
                       lp0[idx].repeat_interleave(64, 0), K, br).view(len(idx), 64)
        res[k] = dict(frac_flat=float((kl < thr).double().mean()), median_kl=float(kl.median()))
    return res


def eval_shared(m, hist, t, p, lp0, idx, V, s, thr, K, br, n_dir=32, gen=None):
    with torch.no_grad():
        u = rand_unit((len(idx), n_dir, V.shape[0]), gen) @ V
        kl = kl_of(m, hist[idx].repeat_interleave(n_dir, 0), t, p, s * u.reshape(-1, V.shape[1]),
                   lp0[idx].repeat_interleave(n_dir, 0), K, br)
    return dict(frac_flat=float((kl < thr).double().mean()), median_kl=float(kl.median()))


def share_in(V, X):
    """Fraction of the squared norm of rows of X inside span(V)."""
    return float(((X @ V.T) ** 2).sum() / (X ** 2).sum())


def shared(m, hist, t, p, A, lp0, tr, te, s, thr, K, br, ones, steps, lab=None, pairs=None):
    D = A.shape[1]
    Xc = A - A.mean(0)
    Xc = Xc - (Xc @ ones)[:, None] * ones
    ev, evec = torch.linalg.eigh(Xc.T @ Xc)                       # ascending variance
    order = [i for i in range(D) if abs(float(evec[:, i] @ ones)) < 0.99]
    gen = torch.Generator(device=dev).manual_seed(1)
    res = {}
    for k in K_LIST:
        W = torch.randn(k, D, device=dev, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([W], lr=0.05)
        for it in range(steps):
            V = basis(W, ones)
            b = tr[torch.randint(len(tr), (32,))]
            mv = moves_in(V.expand(len(b), k, D), 4)
            nm = mv.shape[1]
            kl = kl_of(m, hist[b].repeat_interleave(nm, 0), t, p, s * mv.reshape(-1, D),
                       lp0[b].repeat_interleave(nm, 0), K, br)
            opt.zero_grad(); kl.mean().backward(); opt.step()
        V = basis(W, ones).detach()
        Vpca = basis(evec[:, order[:k]].T.contiguous(), ones)
        Vrnd = basis(torch.randn(k, D, device=dev, dtype=torch.float64, generator=gen), ones)
        row = dict(learned=eval_shared(m, hist, t, p, lp0, te, V, s, thr, K, br, gen=gen),
                   pca_lowest=eval_shared(m, hist, t, p, lp0, te, Vpca, s, thr, K, br, gen=gen),
                   random=eval_shared(m, hist, t, p, lp0, te, Vrnd, s, thr, K, br, gen=gen),
                   variance_share_learned=share_in(V, Xc), variance_share_pca_lowest=share_in(Vpca, Xc),
                   variance_share_random=share_in(Vrnd, Xc))
        if lab is not None:
            same, diff = pairs
            row['within_group_diff_share'] = share_in(V, A[same[:, 1]] - A[same[:, 0]])
            row['between_group_diff_share'] = share_in(V, A[diff[:, 1]] - A[diff[:, 0]])
        res[k] = row
    return res


def calib_radius(m, hist, t, p, lp0, s_ref, thr, K, br, ones, D, n_dir=4, target=0.9):
    """Smallest radius (25 log-spaced values in [0.01, 100] x s_ref) at which >= target of random unit moves
    (orthogonal to all-ones) exceed thr. Returns (radius, fraction broken)."""
    gen = torch.Generator(device=dev).manual_seed(7)
    u = torch.randn(200 * n_dir, D, device=dev, dtype=torch.float64, generator=gen)
    u = u - (u @ ones)[:, None] * ones
    u = u / u.norm(dim=1, keepdim=True)
    h, l0 = hist.repeat_interleave(n_dir, 0), lp0.repeat_interleave(n_dir, 0)
    for f in np.logspace(-2, 2, 25):
        with torch.no_grad():
            kl = kl_of(m, h, t, p, f * s_ref * u, l0, K, br)
        fr = float((kl > thr).double().mean())
        if fr >= target:
            return float(f * s_ref), fr
    return float(100 * s_ref), fr


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--arch', default='L4_d64')
    pa.add_argument('--seed', type=int, default=0)
    pa.add_argument('--K', type=int, default=4)
    pa.add_argument('--steps', type=int, default=300)
    pa.add_argument('--n_ctx', type=int, default=16)
    a = pa.parse_args()
    torch.manual_seed(a.seed)
    ck = torch.load(os.path.join(rp.REPO, 'runs', a.arch, 'eps0', f's{a.seed}.pt'), map_location='cpu', weights_only=True)
    cfg = rp.Config(**ck['config'])
    mt = rp.Transformer(cfg); mt.load_state_dict(ck['model'])
    mu = rp.untrained(cfg, float(ck['args']['init_std']), int(ck['args']['seed']))
    for m in (mt, mu):
        m.to(dev).eval()
        for prm in m.parameters():
            prm.requires_grad_(False)
    T = rp.transition_tensor(0.)
    toks = rp.sample(T, 200, cfg.context, torch.Generator().manual_seed(4242))
    hist = toks.to(dev)
    R = json.load(open(f'results/plateau/rrxor_{a.arch}_eps0_s{a.seed}.json'))
    K, D = a.K, cfg.d_model
    br = torch.cartesian_prod(*[torch.arange(2, device=dev)] * K)
    ones = torch.ones(D, device=dev, dtype=torch.float64) / D ** 0.5
    rng = np.random.default_rng(a.seed)
    perm = torch.tensor(rng.permutation(200), device=dev)
    tr, te = perm[:100], perm[100:]
    out = dict(arch=a.arch, seed=a.seed, K=K, k_list=K_LIST, cells={})
    for p in range(1, cfg.n_layers + 1):
        for t in (6, 14):
            key = f'p{p}_t{t}'
            lab = np.array(R['points'][key]['labels'])
            thr = R['points'][key]['threshold']
            with torch.no_grad():
                _, A = rp.run(mt, hist[:, :t + 1], collect=(p, t))
                A = A.double()
                lp0 = kl_dummy = None
                lp0 = flp(mt, hist, t, p, torch.zeros(200, D, device=dev, dtype=torch.float64), K, br)
            iu = np.array(np.triu_indices(200, 1)).T
            sm = lab[iu[:, 0]] == lab[iu[:, 1]]
            same, diff = iu[sm], iu[~sm]
            dd = (A[diff[:, 1]] - A[diff[:, 0]]).norm(dim=1)
            s_between = float(dd.median())
            s, fr = calib_radius(mt, hist, t, p, lp0, s_between, thr, K, br, ones, D)
            with torch.no_grad():
                kl1 = kl_of(mt, hist, t, p, (s * ones).expand(200, D).contiguous(), lp0, K, br)
            cell = dict(threshold=thr, radius=s, between_group_distance=s_between, random_broken_at_radius=fr,
                        n_groups=int(lab.max()), ones_direction_max_kl=float(kl1.max()))
            if t == 6:
                idx = te[:a.n_ctx]
                cell['per_context'] = per_context(mt, hist, t, p, A, lp0, idx, s, thr, K, br, ones, a.steps)
            cell['shared'] = shared(mt, hist, t, p, A, lp0, tr, te, s, thr, K, br, ones, a.steps, lab, (same, diff))
            # untrained: same radius rule from its own activations (random pairs), its own scale
            with torch.no_grad():
                _, Au = rp.run(mu, hist[:, :t + 1], collect=(p, t))
                Au = Au.double()
                lpu = flp(mu, hist, t, p, torch.zeros(200, D, device=dev, dtype=torch.float64), K, br)
            su_ref = float((Au[diff[:, 1]] - Au[diff[:, 0]]).norm(dim=1).median())
            su, fru = calib_radius(mu, hist, t, p, lpu, su_ref, thr, K, br, ones, D)   # same KL threshold
            cell['untrained'] = dict(threshold=thr, radius=su, random_broken_at_radius=fru,
                                     shared=shared(mu, hist, t, p, Au, lpu, tr, te, su, thr, K, br, ones, a.steps))
            out['cells'][key] = cell
            pc = cell.get('per_context', {})
            print(f"{key}: radius {s:.1f} ({s / s_between:.2f} x between-group distance; random moves broken "
                  f"{fr:.2f}), thr {thr:.1e}, all-ones max KL {cell['ones_direction_max_kl']:.1e} | untrained radius "
                  f"{su:.2f} (broken {fru:.2f})", flush=True)
            if pc:
                print('   per-context flat fraction by k: ' + ' '.join(f"{k}:{v['frac_flat']:.2f}" for k, v in pc.items()), flush=True)
            sh = cell['shared']
            print('   shared (held-out) flat fraction by k: ' + ' '.join(
                f"{k}: learned {v['learned']['frac_flat']:.2f} pca-low {v['pca_lowest']['frac_flat']:.2f} rand {v['random']['frac_flat']:.2f}"
                f" [var {v['variance_share_learned']:.3f}/{v['variance_share_pca_lowest']:.3f}; within {v['within_group_diff_share']:.2f} between {v['between_group_diff_share']:.2f}]"
                for k, v in sh.items()), flush=True)
            print('   untrained shared: ' + ' '.join(f"{k}: {v['learned']['frac_flat']:.2f}/rand {v['random']['frac_flat']:.2f}"
                                                for k, v in cell['untrained']['shared'].items()), flush=True)
            json.dump(out, open(f'results/plateau/gradflat_{a.arch}_s{a.seed}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
