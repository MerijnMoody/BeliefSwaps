"""Interpolation tests on the RRXOR swap groups (L4_d64, artemyk/belief-state checkpoints).

For context pairs (i, j) at residual point p and history length t, put A_i + alpha (A_j - A_i) into i's forward pass
(position t, point p; everything else recomputed as in rrxor_plateau.py) and measure
d(alpha) = KL(clean exact-K-step future of i || patched future).

Tests
1. Same-group paths: is d(alpha) below the group threshold along the whole path (not only at the ends / midpoint)?
2. Different-group paths: shape of d(alpha) / d(1): sharpness (max slope; a straight line gives 1), share of the path
   that is flat at each end, and the number of jumps (runs of consecutive steps whose step-KL between neighbouring
   alphas exceeds the group threshold).
3. Random-direction control: from A_i, move the same distance |A_j - A_i| along an isotropic random direction and
   along a direction drawn from the activation covariance; fraction of moves above the threshold, vs the real
   different-group direction.
4. Extrapolation along a same-group direction: alpha up to 16; at what multiple does d exceed the threshold, and how
   far is that compared to the typical between-group distance?
Trained vs untrained (same contexts, untrained init of the same checkpoint): sharpness of random pairs.

    python interp_rrxor.py --seed 0
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
ALPHA = np.linspace(0, 1, 41)
EXTRA = np.array([1, 1.5, 2, 3, 4, 6, 8, 12, 16])


@torch.no_grad()
def kl_moves(m, hist, t, p, rec, delta, lp0, K, bt=2048):
    """KL(clean future of recipient rec[k] || future with delta[k] added at (p, t)) for each row k."""
    out = torch.empty(len(rec), dtype=torch.float64, device=dev)
    for s in range(0, len(rec), bt):
        r = rec[s:s + bt]
        lpa = rp.future_logp(m, hist[r], t, p, delta[s:s + bt].float(), K)
        out[s:s + bt] = (lp0[r].exp() * (lp0[r] - lpa)).sum(1)
    return out.cpu().numpy()


@torch.no_grad()
def futures_along(m, hist, t, p, rec, delta, K, bt=2048):
    """Patched future log-probs (rows, 2^K)."""
    out = []
    for s in range(0, len(rec), bt):
        out.append(rp.future_logp(m, hist[rec[s:s + bt]], t, p, delta[s:s + bt].float(), K))
    return torch.cat(out)


def path_stats(m, hist, t, p, A, lp0, pairs, thr, K):
    """d(alpha) curves and step KLs for pairs (i, j): recipient i, path from A_i to A_j."""
    i, j = pairs[:, 0], pairs[:, 1]
    na, npair = len(ALPHA), len(pairs)
    rec = np.repeat(i, na)
    al = torch.tensor(np.tile(ALPHA, npair), device=dev, dtype=torch.float64)[:, None]
    delta = al * (A[np.repeat(j, na)] - A[rec])
    lpa = futures_along(m, hist, t, p, rec, delta, K).view(npair, na, -1)
    lc = lp0[i][:, None]
    d = (lc.exp() * (lc - lpa)).sum(-1).cpu().numpy()                                  # (npair, na)
    step = (lpa[:, :-1].exp() * (lpa[:, :-1] - lpa[:, 1:])).sum(-1).cpu().numpy()     # (npair, na - 1)
    to_end = (lpa.exp() * (lpa - lpa[:, -1:])).sum(-1).cpu().numpy()                  # KL(f(alpha) || f(1))
    return d, step, to_end


def jumps(step, thr):
    above = step > thr
    return (above[:, 1:] & ~above[:, :-1]).sum(1) + above[:, 0]


def summarize_diff(d, step, to_end, thr):
    d1 = np.maximum(d[:, -1], 1e-12)
    norm = d / d1[:, None]
    slope = np.diff(norm, axis=1).max(1) / (ALPHA[1] - ALPHA[0])
    flat_start = (np.cumprod(d < thr, axis=1)).sum(1) / len(ALPHA)
    flat_end = (np.cumprod((to_end < thr)[:, ::-1], axis=1)).sum(1) / len(ALPHA)
    nj = jumps(step, thr)
    q = lambda v: [float(np.quantile(v, x)) for x in (0.1, 0.5, 0.9)]
    return dict(sharpness_q10_50_90=q(slope), flat_start_q=q(flat_start), flat_end_q=q(flat_end),
                jumps_hist={int(k): int(v) for k, v in zip(*np.unique(nj, return_counts=True))},
                median_norm_curve=[float(x) for x in np.median(norm, 0)[::4]])


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--arch', default='L4_d64')
    pa.add_argument('--seed', type=int, default=0)
    pa.add_argument('--K', type=int, default=4)
    pa.add_argument('--npairs', type=int, default=150)
    a = pa.parse_args()
    ck = torch.load(os.path.join(rp.REPO, 'runs', a.arch, 'eps0', f's{a.seed}.pt'), map_location='cpu', weights_only=True)
    cfg = rp.Config(**ck['config'])
    models = {}
    m = rp.Transformer(cfg); m.load_state_dict(ck['model']); models['trained'] = m
    models['untrained'] = rp.untrained(cfg, float(ck['args']['init_std']), int(ck['args']['seed']))
    for k in models:
        models[k] = models[k].to(dev).eval()
        for prm in models[k].parameters():
            prm.requires_grad_(False)
    T = rp.transition_tensor(0.)
    toks = rp.sample(T, 200, cfg.context, torch.Generator().manual_seed(4242))
    hist = toks.to(dev)
    R = json.load(open(f'results/plateau/rrxor_{a.arch}_eps0_s{a.seed}.json'))
    K = a.K
    rng = np.random.default_rng(a.seed)
    out = dict(arch=a.arch, seed=a.seed, K=K, alpha=ALPHA.tolist(), extra=EXTRA.tolist(), cells={})
    for p in range(1, cfg.n_layers + 1):
        for t in (6, 14):
            key = f'p{p}_t{t}'
            lab = np.array(R['points'][key]['labels'])
            thr = R['points'][key]['threshold']
            mt = models['trained']
            _, A = rp.run(mt, hist[:, :t + 1], collect=(p, t))
            A = A.double()
            lp0 = rp.future_logp(mt, hist, t, p, torch.zeros(200, cfg.d_model, device=dev), K)
            hid = np.array([hash(tuple(toks[i, :t].tolist())) for i in range(200)])
            iu = np.array(np.triu_indices(200, 1)).T
            iu = iu[hid[iu[:, 0]] != hid[iu[:, 1]]]                         # different histories only
            same = lab[iu[:, 0]] == lab[iu[:, 1]]
            ps = iu[same][rng.permutation(same.sum())[:a.npairs]]
            pdf = iu[~same][rng.permutation((~same).sum())[:a.npairs]]
            ps = np.concatenate([ps, ps[:, ::-1]]); pdf = np.concatenate([pdf, pdf[:, ::-1]])  # both directions
            cell = dict(threshold=thr, n_groups=int(lab.max()), n_same=len(ps), n_diff=len(pdf))
            # 1. same-group paths
            if len(ps):
                d, step, _ = path_stats(mt, hist, t, p, A, lp0, ps, thr, K)
                mx = d.max(1)
                cell['same'] = dict(frac_path_below_thr=float((mx < thr).mean()),
                                    frac_endpoint_below_thr=float((d[:, -1] < thr).mean()),
                                    frac_mid_below_thr=float((d[:, 20] < thr).mean()),
                                    max_over_path_q50_90=[float(np.quantile(mx, 0.5)), float(np.quantile(mx, 0.9))],
                                    jumps_hist={int(k): int(v) for k, v in zip(*np.unique(jumps(step, thr), return_counts=True))})
            # 2. different-group paths
            d, step, to_end = path_stats(mt, hist, t, p, A, lp0, pdf, thr, K)
            harm = d[:, -1] > thr          # directed swaps that change the future (shape stats use only these)
            cell['diff_all'] = summarize_diff(d, step, to_end, thr)
            cell['diff'] = summarize_diff(d[harm], step[harm], to_end[harm], thr) if harm.any() else {}
            cell['diff']['frac_endpoint_above_thr'] = float(harm.mean())
            h2 = harm.reshape(2, -1)
            cell['diff']['frac_pairs_harmful_either_direction'] = float((h2[0] | h2[1]).mean())
            if harm.any():
                nh = d[harm] / d[harm][:, -1:]
                cross = ALPHA[np.argmax(nh >= 0.5, axis=1)]
                cell['diff']['alpha_half_q10_50_90'] = [float(np.quantile(cross, x)) for x in (0.1, 0.5, 0.9)]
            # 3. random directions of the same length (from A_i), vs the real direction to A_j
            Xc = (A - A.mean(0)).cpu().numpy()
            L = np.linalg.cholesky(np.cov(Xc.T) + 1e-9 * np.eye(cfg.d_model))
            for nm, pr in (('diff', pdf), ('same', ps)):
                if not len(pr):
                    continue
                i, j = pr[:, 0], pr[:, 1]
                dist = (A[j] - A[i]).norm(dim=1, keepdim=True)
                iso = torch.randn(len(pr), cfg.d_model, device=dev, dtype=torch.float64)
                cov = torch.tensor(rng.standard_normal((len(pr), cfg.d_model)) @ L.T, device=dev)
                res = {}
                for dn, v in (('real', A[j] - A[i]), ('random_iso', iso), ('random_cov', cov)):
                    v = v / v.norm(dim=1, keepdim=True) * dist
                    kl = kl_moves(mt, hist, t, p, i, v, lp0, K)
                    res[dn] = dict(frac_above_thr=float((kl > thr).mean()), median_kl=float(np.median(kl)))
                cell[f'random_control_{nm}'] = res
                cell[f'pair_distance_{nm}_median'] = float(dist.median())
            # 4. extrapolation along same-group directions
            if len(ps):
                i, j = ps[:, 0], ps[:, 1]
                ext = []
                for e in EXTRA:
                    ext.append(kl_moves(mt, hist, t, p, i, e * (A[j] - A[i]), lp0, K))
                ext = np.stack(ext, 1)
                first = np.array([EXTRA[np.argmax(r > thr)] if (r > thr).any() else np.inf for r in ext])
                dist = (A[j] - A[i]).norm(dim=1).cpu().numpy()
                cell['extrapolate_same'] = dict(
                    frac_above_thr_by_alpha={str(e): float((ext[:, k] > thr).mean()) for k, e in enumerate(EXTRA)},
                    median_breaking_distance=float(np.median(np.where(np.isfinite(first), first * dist, np.nan)[np.isfinite(first)])) if np.isfinite(first).any() else None,
                    frac_never_break_16x=float(np.isinf(first).mean()))
            # between-group scale for reference
            cell['between_group_distance_median'] = float((A[pdf[:, 1]] - A[pdf[:, 0]]).norm(dim=1).median())
            # untrained: random pairs, shape only
            mu = models['untrained']
            _, Au = rp.run(mu, hist[:, :t + 1], collect=(p, t))
            Au = Au.double()
            lpu = rp.future_logp(mu, hist, t, p, torch.zeros(200, cfg.d_model, device=dev), K)
            pr = iu[rng.permutation(len(iu))[:a.npairs]]
            du, su, teu = path_stats(mu, hist, t, p, Au, lpu, pr, thr, K)
            dn = du / np.maximum(du[:, -1:], 1e-30)
            cell['untrained_random_pairs'] = dict(
                sharpness_q10_50_90=[float(np.quantile(np.diff(dn, axis=1).max(1) / (ALPHA[1] - ALPHA[0]), x)) for x in (0.1, 0.5, 0.9)],
                median_norm_curve=[float(x) for x in np.median(dn, 0)[::4]],
                endpoint_kl_median=float(np.median(du[:, -1])))
            dt, st, tet = path_stats(mt, hist, t, p, A, lp0, pr, thr, K)
            dnt = dt / np.maximum(dt[:, -1:], 1e-30)
            cell['trained_random_pairs'] = dict(
                sharpness_q10_50_90=[float(np.quantile(np.diff(dnt, axis=1).max(1) / (ALPHA[1] - ALPHA[0]), x)) for x in (0.1, 0.5, 0.9)],
                median_norm_curve=[float(x) for x in np.median(dnt, 0)[::4]],
                endpoint_kl_median=float(np.median(dt[:, -1])))
            out['cells'][key] = cell
            s, df = cell.get('same', {}), cell['diff']
            print(f"{key}: {cell['n_groups']} groups, thr {thr:.1e} | SAME path flat {s.get('frac_path_below_thr', float('nan')):.2f} "
                  f"(endpoint {s.get('frac_endpoint_below_thr', float('nan')):.2f}, mid {s.get('frac_mid_below_thr', float('nan')):.2f}) | "
                  f"DIFF harmful {df['frac_endpoint_above_thr']:.2f} (either dir {df['frac_pairs_harmful_either_direction']:.2f}); "
                  f"harmful paths: sharpness {np.round(df.get('sharpness_q10_50_90', []), 1).tolist()} jumps {df.get('jumps_hist')} "
                  f"flat start/end {df.get('flat_start_q', [0, float('nan')])[1]:.2f}/{df.get('flat_end_q', [0, float('nan')])[1]:.2f} "
                  f"alpha at half {df.get('alpha_half_q10_50_90')}\n"
                  f"   random ctrl (diff, same length): " + ' '.join(f"{k} {v['frac_above_thr']:.2f}" for k, v in cell['random_control_diff'].items()) +
                  (f" | (same): " + ' '.join(f"{k} {v['frac_above_thr']:.2f}" for k, v in cell['random_control_same'].items()) if 'random_control_same' in cell else '') +
                  (f"\n   extrapolate same-group: break frac by alpha {cell['extrapolate_same']['frac_above_thr_by_alpha']} "
                   f"median break dist {cell['extrapolate_same']['median_breaking_distance']} vs between-group {cell['between_group_distance_median']:.1f}, "
                   f"same-pair dist {cell.get('pair_distance_same_median', float('nan')):.2f}" if 'extrapolate_same' in cell else '') +
                  f"\n   random pairs sharpness: trained {np.round(cell['trained_random_pairs']['sharpness_q10_50_90'], 1).tolist()} "
                  f"untrained {np.round(cell['untrained_random_pairs']['sharpness_q10_50_90'], 1).tolist()} | median curve trained "
                  f"{np.round(cell['trained_random_pairs']['median_norm_curve'], 2).tolist()} untrained "
                  f"{np.round(cell['untrained_random_pairs']['median_norm_curve'], 2).tolist()}", flush=True)
    os.makedirs('results/plateau', exist_ok=True)
    json.dump(out, open(f'results/plateau/interp_{a.arch}_s{a.seed}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
