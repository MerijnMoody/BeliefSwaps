"""Head-to-head: plateau (swap) partitions vs PCA + k-means vs linear probes, on the RRXOR models of artemyk/belief-state.

H2 (natural). At residual point p, history length t: build groups on context set A with
  ours    swap distance (j's real activation into i and back, max KL of the exact K-step future), Otsu + complete linkage
  pca     PCA (components for 90% of variance, max 20) + k-means, k by silhouette (2..40)
  oracle  hypothesis groupings from labels: belief, last2, xor2, (belief, last2); plus all-in-one and all-separate
and score every grouping on a HELD-OUT context set B with one causal test:
  harmful_within   fraction of same-group pairs whose swap changes the future by KL > tau
  harmless_between fraction of different-group pairs whose swap changes it by KL < tau
B contexts are assigned to groups by each method's own rule (ours: smallest swap KL to the group medoid in A;
pca: nearest k-means centroid; oracle: the true label). Probes (fit on A, scored on B): ridge R^2 to the belief,
to a shuffled belief (distinct beliefs permuted), logistic accuracy for last2.

H1 (planted, ground truth). Add a 6-valued nuisance z (independent of everything) along random directions in the
output-insensitive subspace (bottom eigenvectors of the Fisher matrix of the K-step future w.r.t. the activation),
scaled alpha x the activation RMS spread. Check the plant's own effect on the future (KL), then compare which
variable each method recovers (ARI vs belief and vs z) and the probe accuracy for z.

    python hh.py --arch L4_d64 --seed 0
"""
import argparse
import importlib.util
import json
import os

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import adjusted_rand_score as ari, silhouette_score

_spec = importlib.util.spec_from_file_location('rp', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                   'rrxor_plateau.py'))
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)
dev = 'cuda'
TAUS = (1e-3, 1e-2, 1e-1)


@torch.no_grad()
def lp_with(m, hist, t, p, delta, K):
    out = []
    for s in range(0, len(hist), 4096):
        out.append(rp.future_logp(m, hist[s:s + 4096], t, p, delta[s:s + 4096].float(), K))
    return torch.cat(out)


@torch.no_grad()
def pair_dist(m, hist, t, p, Areal, Aeff, K):
    """(n, n) swap distance: max over (j into i, i into j) of KL(own future || swapped future). Aeff = activation
    the context actually carries (planted or not); Areal = the model's own activation (the injection baseline)."""
    n = len(hist)
    lp0 = lp_with(m, hist, t, p, Aeff - Areal, K)
    ii, jj = torch.triu_indices(n, n, 1, device=dev)
    d = torch.zeros(len(ii), device=dev, dtype=torch.float64)
    B = 2048
    for s in range(0, len(ii), B):
        i, j = ii[s:s + B], jj[s:s + B]
        for src, dst in ((i, j), (j, i)):
            lpa = lp_with(m, hist[src], t, p, Aeff[dst] - Areal[src], K)
            kl = (lp0[src].exp() * (lp0[src] - lpa)).sum(1)
            d[s:s + B] = torch.maximum(d[s:s + B], kl)
    D = torch.zeros(n, n, device=dev, dtype=torch.float64)
    D[ii, jj] = d
    D[jj, ii] = d
    return D.cpu().numpy(), lp0


def groups_from(D):
    n = len(D)
    iu = np.triu_indices(n, 1)
    ld = np.log10(np.maximum(D[iu], 1e-12))
    thr, sep = rp.otsu(ld)
    off = ld.min() - 1e-6
    Z = linkage(ld - off, method='complete')
    return fcluster(Z, t=thr - off, criterion='distance'), float(10 ** thr)


def pca_groups(XA, XB):
    mu = XA.mean(0)
    U, S, Vt = np.linalg.svd(XA - mu, full_matrices=False)
    ev = np.cumsum(S ** 2) / (S ** 2).sum()
    r = int(min(20, np.searchsorted(ev, 0.9) + 1))
    ZA, ZB = (XA - mu) @ Vt[:r].T, (XB - mu) @ Vt[:r].T
    best = None
    for k in range(2, min(41, len(ZA) // 2)):
        km = KMeans(k, n_init=4, random_state=0).fit(ZA)
        if not 2 <= len(np.unique(km.labels_)) < len(ZA):
            continue
        s = silhouette_score(ZA, km.labels_)
        if best is None or s > best[0]:
            best = (s, k, km)
    return best[2].labels_, best[2].predict(ZB), r, best[1]


def causal_eval(DB, lab):
    n = len(DB)
    iu = np.triu_indices(n, 1)
    d = DB[iu]
    same = lab[iu[0]] == lab[iu[1]]
    out = dict(n_groups=int(len(np.unique(lab))))
    for tau in TAUS:
        hw = float((d[same] > tau).mean()) if same.any() else 0.0
        hb = float((d[~same] < tau).mean()) if (~same).any() else 0.0
        out[f'tau{tau:g}'] = dict(harmful_within=hw, harmless_between=hb, error=0.5 * (hw + hb))
    return out


def labels_for(T, toks, bel, t, K):
    x = toks.numpy()
    b = bel[:, t]
    bid = rp.ids(b)
    last2 = x[:, t - 2] * 2 + x[:, t - 1]
    return dict(belief=bid, last2=last2, xor2=x[:, t - 2] ^ x[:, t - 1],
                bel_last2=rp.ids(np.stack([bid, last2], 1), 0)), b.numpy()


def probes(XA, XB, labA, labB, bA, bB, rng):
    out = {}
    r = Ridge(1e-3).fit(XA, bA)
    out['belief_R2'] = float(r.score(XB, bB))
    # shuffled belief: permute the distinct beliefs among themselves (same in A and B)
    allb = np.round(np.concatenate([bA, bB]), 6)
    uniq, inv = np.unique(allb, axis=0, return_inverse=True)
    perm = rng.permutation(len(uniq))
    sh = uniq[perm][inv.reshape(-1)]
    r = Ridge(1e-3).fit(XA, sh[:len(bA)])
    out['shuffled_belief_R2'] = float(r.score(XB, sh[len(bA):]))
    for nm in ('last2', 'belief'):
        if len(np.unique(labA[nm])) > 1:
            c = LogisticRegression(max_iter=3000).fit(XA, labA[nm])
            out[f'{nm}_acc'] = float(c.score(XB, labB[nm]))
            vals, cnt = np.unique(labB[nm], return_counts=True)
            out[f'{nm}_acc_chance'] = float(cnt.max() / cnt.sum())
    return out


@torch.no_grad()
def fisher(m, hist, t, p, A, K):
    """Fisher matrix of the K-step future distribution w.r.t. the activation at (p, t), finite differences."""
    n, D = A.shape
    h = 1e-2 * A.std().item()
    lp0 = lp_with(m, hist, t, p, torch.zeros_like(A), K)
    P = lp0.exp()
    G = torch.zeros(D, D, device=dev, dtype=torch.float64)
    J = torch.zeros(n, P.shape[1], D, device=dev, dtype=torch.float64)
    for k in range(D):
        e = torch.zeros_like(A)
        e[:, k] = h
        J[:, :, k] = (lp_with(m, hist, t, p, e, K) - lp_with(m, hist, t, p, -e, K)) / (2 * h)
    for i in range(n):
        Jp = J[i] * P[i, :, None].sqrt()
        G += Jp.T @ Jp
    return (G / n).cpu().numpy()


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--arch', default='L4_d64')
    pa.add_argument('--seed', type=int, default=0)
    pa.add_argument('--n', type=int, default=200)
    pa.add_argument('--K', type=int, default=4)
    pa.add_argument('--parts', default='H2,H1')
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
    toksA = rp.sample(T, n, cfg.context, torch.Generator().manual_seed(4242))
    toksB = rp.sample(T, n, cfg.context, torch.Generator().manual_seed(999))
    belA, belB = rp.beliefs(T, toksA), rp.beliefs(T, toksB)
    hA, hB = toksA.to(dev), toksB.to(dev)
    rng = np.random.default_rng(0)
    out = dict(arch=a.arch, seed=a.seed, n=n, K=K, H2={}, H1={})

    # ---------------- H2: natural ----------------
    for p in (range(1, L + 1) if 'H2' in a.parts else []):
        for t in (6, 9, 14):
            _, AA = rp.run(m, hA[:, :t + 1], collect=(p, t))
            _, AB = rp.run(m, hB[:, :t + 1], collect=(p, t))
            AA, AB = AA.double(), AB.double()
            labA, bA = labels_for(T, toksA, belA, t, K)
            labB, bB = labels_for(T, toksB, belB, t, K)
            DA, _ = pair_dist(m, hA, t, p, AA, AA, K)
            DB, lp0B = pair_dist(m, hB, t, p, AB, AB, K)
            ours_A, thr = groups_from(DA)
            # assign B to our groups: smallest swap KL from the group medoid (in A) into the B context
            med = []
            for g in np.unique(ours_A):
                idx = np.where(ours_A == g)[0]
                med.append(idx[np.argmin(DA[np.ix_(idx, idx)].sum(1))])
            klm = np.zeros((n, len(med)))
            for gi, mi in enumerate(med):
                lpa = lp_with(m, hB, t, p, AA[mi][None].expand(n, -1) - AB, K)
                klm[:, gi] = (lp0B.exp() * (lp0B - lpa)).sum(1).cpu().numpy()
            ours_B = np.array(np.unique(ours_A))[klm.argmin(1)]
            XA, XB = AA.cpu().numpy(), AB.cpu().numpy()
            pcaA, pcaB, r, kpca = pca_groups(XA, XB)
            groupings = dict(ours=ours_B, pca=pcaB, **{f'oracle_{k}': v for k, v in labB.items()},
                             one_group=np.zeros(n, int), all_separate=np.arange(n))
            row = dict(threshold=thr, n_groups_A=dict(ours=int(len(np.unique(ours_A))), pca=int(kpca)), pca_dims=r,
                       n_beliefs_B=int(len(np.unique(labB['belief']))),
                       eval={k: causal_eval(DB, v) for k, v in groupings.items()},
                       ari_vs_belief={k: float(ari(labB['belief'], v)) for k, v in groupings.items()},
                       ari_vs_last2={k: float(ari(labB['last2'], v)) for k, v in groupings.items()},
                       probes=probes(XA, XB, labA, labB, bA, bB, rng))
            out['H2'][f'p{p}_t{t}'] = row
            e = row['eval']
            print(f"H2 p{p} t{t}: beliefs {row['n_beliefs_B']} | groups ours {row['n_groups_A']['ours']} pca {kpca} "
                  f"(r={r}) | error@1e-2 " + ' '.join(f"{k} {v['tau0.01']['error']:.3f}" for k, v in e.items()) +
                  " | ARI belief " + ' '.join(f"{k} {row['ari_vs_belief'][k]:.2f}" for k in ('ours', 'pca')) +
                  " | probes " + ' '.join(f"{k} {v:.2f}" for k, v in row['probes'].items()), flush=True)

    # ---------------- H1: planted nuisance ----------------
    for p, t in (((3, 9), (3, 14), (2, 9)) if 'H1' in a.parts else ()):
        _, AA = rp.run(m, hA[:, :t + 1], collect=(p, t))
        AA = AA.double()
        labA, bA = labels_for(T, toksA, belA, t, K)
        G = fisher(m, hA, t, p, AA, K)
        w, V = np.linalg.eigh(G)
        insens = V[:, w < 1e-6 * w.max()]
        if insens.shape[1] < 6:
            insens = V[:, :16]
        U = insens @ np.linalg.qr(rng.standard_normal((insens.shape[1], 6)))[0]     # 6 orthonormal directions
        z = rng.integers(0, 6, n)
        spread = float(np.sqrt(((AA - AA.mean(0)) ** 2).sum(1).mean().item()))
        res = dict(fisher_eigs=w.tolist(), n_insensitive=int((w < 1e-6 * w.max()).sum()), spread=spread, alphas={})
        lpA = lp_with(m, hA, t, p, torch.zeros_like(AA), K)
        for alpha in (0.0, 0.5, 1.0, 2.0, 4.0):
            plant = torch.tensor(alpha * spread * U[:, z].T, device=dev)
            Aeff = AA + plant
            lpP = lp_with(m, hA, t, p, plant, K)
            eff = (lpA.exp() * (lpA - lpP)).sum(1).cpu().numpy()
            D, _ = pair_dist(m, hA, t, p, AA, Aeff, K)
            ours, thr = groups_from(D)
            X = Aeff.cpu().numpy()
            pcaL, _, r, kpca = pca_groups(X, X)
            half = n // 2
            zacc = LogisticRegression(max_iter=3000).fit(X[:half], z[:half]).score(X[half:], z[half:])
            row = dict(plant_effect_KL_median=float(np.median(eff)), plant_effect_KL_max=float(eff.max()),
                       threshold=thr, groups=dict(ours=int(ours.max()), pca=int(kpca)), pca_dims=r,
                       ari_belief=dict(ours=float(ari(labA['belief'], ours)), pca=float(ari(labA['belief'], pcaL))),
                       ari_nuisance=dict(ours=float(ari(z, ours)), pca=float(ari(z, pcaL))),
                       probe_nuisance_acc=float(zacc))
            res['alphas'][str(alpha)] = row
            print(f"H1 p{p} t{t} alpha {alpha}: plant effect KL median {row['plant_effect_KL_median']:.1e} max "
                  f"{row['plant_effect_KL_max']:.1e} (cutoff {thr:.1e}) | groups ours {row['groups']['ours']} pca "
                  f"{kpca} | ARI belief ours {row['ari_belief']['ours']:.2f} pca {row['ari_belief']['pca']:.2f} | "
                  f"ARI nuisance ours {row['ari_nuisance']['ours']:.2f} pca {row['ari_nuisance']['pca']:.2f} | "
                  f"probe nuisance acc {zacc:.2f}", flush=True)
        out['H1'][f'p{p}_t{t}'] = res
    os.makedirs('results/plateau', exist_ok=True)
    json.dump(out, open(f'results/plateau/hh_{a.arch}_s{a.seed}_n{n}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
