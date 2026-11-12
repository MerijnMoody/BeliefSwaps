"""Minimality quantities on factored processes (models from train_fact.py; process / loading as in fact_swap.py).

Process (ground truth): block entropies H(L), entropy rate h, excess entropy E(L) = H(L) - L h, for each factor and the
joint process (E and C_mu add over independent factors); for the discrete process RRXOR x Even also the statistical
complexity C_mu = H(stationary causal states) and crypticity chi = C_mu - E. State dimensions: HMM states, PSR rank,
GBS dimension.

Model, residual R at layer p = 0..L, last position t (n contexts, held-out scores):
  R2_full    variance of R explained by a flexible function of the exact belief (MLP, held-out). 1 = R is a function
             of the belief state (minimal in the sense of the renewal / free-rider sections); 1 - R2_full = past
             information R keeps that the belief does not contain (or estimator error).
  R2_add     explained by a SUM of per-factor functions g_k(b_k) (factored, additive code). (R2_full - R2_add) / R2_full
             = share of the belief-coded variance that is interaction between factors (product-space code).
  R2_tok     explained by belief + the last 3 tokens (what the free-rider variance is).
  eta2       (discrete process only) exact: share of R's variance between joint-belief groups (no regressor).
  pcs        PCs of R for 95 / 99% of the variance; of the belief-explained part; of the remainder.
  cmi[l]     I(R; x_{t-l} | belief) in nats = I(R, b; x_{t-l}) - I(b; x_{t-l}), torch_mist SMILE, l = 1, 2, 4, 8.
             Calibration: the same difference with R replaced by the belief itself (true value 0).

    python fact_min.py --run runs_fact/prod4/s0 --t 24
"""
import argparse
import itertools
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch_mist.estimators import smile

import fact_proc as fp
import fact_swap as fs

dev = 'cuda'


# ---------------- process quantities ----------------
def block_entropies(T, Lmax, max_words=1 << 20):
    """H(L) in bits for L = 1..Lmax (stops when V^L exceeds max_words)."""
    T = T.to(dev)
    V = T.shape[0]
    pi = fp.stationary(T.cpu()).to(dev)
    vecs = pi[None]                                         # (words, S) unnormalized forward vectors
    H = []
    for L in range(1, Lmax + 1):
        if len(vecs) * V > max_words:
            break
        vecs = torch.einsum('ws,xst->wxt', vecs, T).reshape(-1, T.shape[1])
        p = vecs.sum(1)
        p = p[p > 0]
        H.append(float(-(p * p.log2()).sum()))
    return H


def excess(H):
    h = H[-1] - H[-2]
    return dict(H=H, h=h, E=H[-1] - len(H) * h, L=len(H))


def process_quantities(proc, K, c):
    out = {}
    if proc == 'tele':
        out['hmm_states'], out['psr_rank'] = 2 ** K, 2 ** K - 1
        out['gbs_dim'] = K if c == 0 else None
        if c == 0:
            fac = [excess(block_entropies(fp.transition_tensor(1, q=fp.DEFAULT_Q[k:k + 1], e=fp.DEFAULT_E[k:k + 1]), 20))
                   for k in range(K)]
            out['factor_E'] = [f['E'] for f in fac]
            out['factor_h'] = [f['h'] for f in fac]
            out['E_sum_factors'] = sum(out['factor_E'])
        j = excess(block_entropies(fp.transition_tensor(K, c=c), 20))
        out['joint_E_finiteL'], out['joint_h'], out['joint_L'] = j['E'], j['h'], j['L']
        out['C_mu'] = 'infinite (continuum of causal states)'
    else:
        Ts = [fp.rrxor_tensor(), fp.even_tensor()]
        Cs, Es = [], []
        for Tk in Ts:
            pi = fp.stationary(Tk).numpy()
            Cs.append(float(-(pi[pi > 0] * np.log2(pi[pi > 0])).sum()))
            Es.append(excess(block_entropies(Tk, 20))['E'])
        J = fp.product_tensor(Ts)
        pj = fp.stationary(J).numpy()
        j = excess(block_entropies(J, 10))
        out.update(factor_C_mu=Cs, factor_E=Es, factor_chi=[a - b for a, b in zip(Cs, Es)],
                   C_mu=float(-(pj[pj > 0] * np.log2(pj[pj > 0])).sum()), C_mu_sum_factors=sum(Cs),
                   joint_E_finiteL=j['E'], joint_L=j['L'], E_sum_factors=sum(Es), chi=None,
                   hmm_states=10, causal_states=10)
        out['chi'] = out['C_mu'] - out['E_sum_factors']
    return out


# ---------------- model quantities ----------------
class Add(nn.Module):
    def __init__(self, dims, D):
        super().__init__()
        self.dims = dims
        self.g = nn.ModuleList(nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(),
                                             nn.Linear(128, D)) for d in dims)

    def forward(self, x):
        out, s = 0, 0
        for d, g in zip(self.dims, self.g):
            out = out + g(x[:, s:s + d]); s += d
        return out


def regress(Xtr, Ytr, Xte, Yte, dims=None, steps=3000, seed=0):
    """Held-out R^2 (pooled over Y's columns) and test predictions; MLP, or additive over input blocks `dims`."""
    torch.manual_seed(seed)
    mx, sx = Xtr.mean(0), Xtr.std(0).clamp_min(1e-8)
    my, sy = Ytr.mean(0), Ytr.std(0).clamp_min(1e-8)
    xtr, xte, ytr = (Xtr - mx) / sx, (Xte - mx) / sx, (Ytr - my) / sy
    D = Ytr.shape[1]
    net = (Add(dims, D) if dims else nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.GELU(), nn.Linear(256, 256),
                                                      nn.GELU(), nn.Linear(256, D))).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-4)
    for _ in range(steps):
        idx = torch.randint(len(xtr), (4096,), device=dev)
        loss = ((net(xtr[idx]) - ytr[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    with torch.no_grad():
        pred = net(xte) * sy + my
    r2 = float(1 - ((Yte - pred) ** 2).sum() / ((Yte - Ytr.mean(0)) ** 2).sum())
    return r2, pred


def n_pcs(X, fracs=(0.95, 0.99)):
    s = torch.linalg.svdvals(X - X.mean(0)).double() ** 2
    c = (s.cumsum(0) / s.sum()).cpu().numpy()
    return [int(np.searchsorted(c, f) + 1) for f in fracs]


def mi(x, y, ntr, steps=4000, seed=0):
    torch.manual_seed(seed)
    mx, sx = x[:ntr].mean(0), x[:ntr].std(0).clamp_min(1e-8)
    x = ((x - mx) / sx).float()
    y = y.float()
    est = smile(x_dim=x.shape[1], y_dim=y.shape[1], hidden_dims=[256, 256], neg_samples=16, tau=5.0).to(dev)
    opt = torch.optim.Adam(est.parameters(), lr=5e-4)
    for _ in range(steps):
        idx = torch.randint(ntr, (512,), device=dev)
        loss = est.loss(x[idx], y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    est.eval()
    with torch.no_grad():
        vals = []
        for _ in range(40):
            idx = torch.randint(ntr, len(x), (1024,), device=dev)
            vals.append(float(est.mutual_information(x[idx], y[idx])))
    return float(np.mean(vals)), float(np.std(vals) / math.sqrt(len(vals)))


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--run', required=True)
    pa.add_argument('--untrained', action='store_true')
    pa.add_argument('--t', type=int, default=24)
    pa.add_argument('--n', type=int, default=30000)
    pa.add_argument('--ntest', type=int, default=6000)
    pa.add_argument('--lags', default='1,2,4,8')
    pa.add_argument('--no_mi', action='store_true')
    a = pa.parse_args()
    m, cfg, T, K, sizes, proc = fs.load(a.run, a.untrained)
    ck = torch.load(a.run + '.pt', map_location='cpu', weights_only=True)
    c = float(ck['args'].get('c', 0))
    V, t, L = T.shape[0], a.t, cfg.n_layers
    tag = a.run.replace('runs_fact/', '').replace('/', '_') + ('_untrained' if a.untrained else '')
    out = dict(run=a.run, untrained=a.untrained, proc=proc, K=K, c=c, t=t, process=process_quantities(proc, K, c),
               layers={})
    print(f"{tag}: process {json.dumps({k: (np.round(v, 3).tolist() if isinstance(v, (list, float)) else v) for k, v in out['process'].items() if k not in ('H',)})}",
          flush=True)

    toks = fp.sample(T, a.n, t + 1, torch.Generator(device=dev).manual_seed(999), dev)
    bel = fp.beliefs(T, toks[:, :t])[:, t].float()
    fb = fs.factor_beliefs(bel.double(), sizes)
    if proc == 'tele':
        fin = torch.cat([f[:, 1:] for f in fb], 1).float()          # K factor marginals
        dims = [1] * K
        full_in = bel if c != 0 else fin                              # product: joint belief = function of fin
    else:
        fin = torch.cat(fb, 1).float()
        dims = sizes
        full_in = bel
    ntr = a.n - a.ntest
    tr, te = slice(0, ntr), slice(ntr, a.n)
    lastk = torch.nn.functional.one_hot(toks[:, t - 3:t], V).flatten(1).float()
    ids = fs.rp.ids(bel.cpu().numpy()) if proc != 'tele' else None
    for p in range(L + 1):
        R = torch.cat([fs.rp.run(m, toks[s:s + 4096, :t + 1], collect=(p, t))[1] for s in range(0, a.n, 4096)]).float()
        row = {}
        row['R2_full'], pred = regress(full_in[tr], R[tr], full_in[te], R[te])
        row['R2_add'], _ = regress(fin[tr], R[tr], fin[te], R[te], dims=dims)
        row['R2_tok'], _ = regress(torch.cat([full_in, lastk], 1)[tr], R[tr], torch.cat([full_in, lastk], 1)[te], R[te])
        row['interaction_share'] = (row['R2_full'] - row['R2_add']) / max(row['R2_full'], 1e-9)
        row['pcs_R'] = n_pcs(R[te])
        row['pcs_belief_part'] = n_pcs(pred)
        row['pcs_remainder'] = n_pcs(R[te] - pred)
        if ids is not None:
            Rn = R.cpu().numpy().astype(np.float64)
            mu = np.zeros_like(Rn)
            for g in np.unique(ids):
                sel = ids == g
                mu[sel] = Rn[sel].mean(0)
            row['eta2_exact'] = float(1 - ((Rn - mu) ** 2).sum() / ((Rn - Rn.mean(0)) ** 2).sum())
        if not a.no_mi:
            row['cmi'] = {}
            for l in map(int, a.lags.split(',')):
                y = torch.nn.functional.one_hot(toks[:, t - l], V)
                both, se1 = mi(torch.cat([R, full_in], 1), y, ntr)
                base, se0 = mi(full_in, y, ntr)
                row['cmi'][l] = dict(I_Rb_x=both, I_b_x=base, cmi=both - base, se=float(np.hypot(se1, se0)))
        out['layers'][f'p{p}'] = row
        print(f"  p{p}: R2 belief {row['R2_full']:.3f} additive {row['R2_add']:.3f} (interaction share "
              f"{row['interaction_share']:.3f}) +last3 tokens {row['R2_tok']:.3f}"
              + (f" | eta2 exact {row['eta2_exact']:.3f}" if 'eta2_exact' in row else '')
              + f" | PCs 95/99% R {row['pcs_R']} belief part {row['pcs_belief_part']} remainder {row['pcs_remainder']}"
              + (' | I(R;x_t-l|b) ' + ' '.join(f"l{l}:{v['cmi']:.3f}±{v['se']:.3f}" for l, v in row['cmi'].items())
                 if 'cmi' in row else ''), flush=True)
    if not a.no_mi:   # calibration: R = belief itself (true conditional MI 0)
        cal = {}
        for l in map(int, a.lags.split(',')):
            y = torch.nn.functional.one_hot(toks[:, t - l], V)
            both, _ = mi(torch.cat([full_in, full_in], 1), y, ntr, seed=1)
            base, _ = mi(full_in, y, ntr)
            cal[l] = both - base
        out['cmi_calibration_R_is_belief'] = cal
        print('  calibration (R = belief, true 0): ' + ' '.join(f"l{l}:{v:.3f}" for l, v in cal.items()), flush=True)
    os.makedirs('results/fact', exist_ok=True)
    json.dump(out, open(f'results/fact/min_{tag}_t{t}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
