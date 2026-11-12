"""Scout for the head-to-head test: what dominates activation variance in the RRXOR models (positions pooled)?
eta^2(V) = between-class variance / total variance of the residual at each point, for belief, position, last tokens.
Also PCA (top 3 / top 10) + k-means (k = number of true beliefs present) ARI against each variable.

    python hh_explore.py --arch L4_d64
"""
import argparse
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari
import importlib.util, os
spec = importlib.util.spec_from_file_location('rp', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rrxor_plateau.py'))
rp = importlib.util.module_from_spec(spec); spec.loader.exec_module(rp)

pa = argparse.ArgumentParser(); pa.add_argument('--arch', default='L4_d64'); pa.add_argument('--seed', type=int, default=0)
a = pa.parse_args()
ck = torch.load(os.path.join(rp.REPO, 'runs', a.arch, 'eps0', f's{a.seed}.pt'), map_location='cpu', weights_only=True)
m = rp.Transformer(rp.Config(**ck['config'])); m.load_state_dict(ck['model']); m = m.to('cuda').eval()
T = rp.transition_tensor(0.)
toks = rp.sample(T, 2000, 32, torch.Generator().manual_seed(7)); bel = rp.beliefs(T, toks)
P = list(range(3, 29))
x = toks.numpy()
lab = dict(belief=np.concatenate([rp.ids(bel[:, t]) * 0 for t in P]))
B = torch.cat([bel[:, t] for t in P]).numpy()
lab['belief'] = rp.ids(B)
lab['position'] = np.concatenate([np.full(len(x), t) for t in P])
lab['last1'] = np.concatenate([x[:, t - 1] for t in P])
lab['last2'] = np.concatenate([x[:, t - 2] * 2 + x[:, t - 1] for t in P])
lab['last3'] = np.concatenate([x[:, t - 3] * 4 + x[:, t - 2] * 2 + x[:, t - 1] for t in P])
lab['xor2'] = np.concatenate([x[:, t - 2] ^ x[:, t - 1] for t in P])
print(a.arch, 'distinct beliefs pooled:', len(np.unique(lab['belief'])))
for p in range(m.cfg.n_layers + 1):
    with torch.no_grad():
        acts = []
        for t in P:
            _, A = rp.run(m, toks[:, :t + 1].cuda(), collect=(p, t)); acts.append(A.double().cpu().numpy())
    A = np.concatenate(acts); A = A - A.mean(0); tot = (A ** 2).sum()
    eta = {}
    for k, v in lab.items():
        bw = sum(((A[v == c].mean(0)) ** 2).sum() * (v == c).sum() for c in np.unique(v))
        eta[k] = bw / tot
    U, S, Vt = np.linalg.svd(A[np.random.default_rng(0).choice(len(A), 8000, replace=False)], full_matrices=False)
    ev = S ** 2 / (S ** 2).sum()
    kb = len(np.unique(lab['belief']))
    res = {}
    for nc in (3, 10):
        Z = A @ Vt[:nc].T
        cl = KMeans(kb, n_init=4, random_state=0).fit_predict(Z)
        res[nc] = {k: ari(v, cl) for k, v in lab.items()}
    print(f"p{p}: eta^2 " + ' '.join(f'{k} {v:.2f}' for k, v in eta.items()) +
          f" | PC var top3 {ev[:3].sum():.2f} | PCA3+kmeans ARI " + ' '.join(f'{k} {v:.2f}' for k, v in res[3].items()) +
          " | PCA10 " + ' '.join(f'{k} {v:.2f}' for k, v in res[10].items()), flush=True)
