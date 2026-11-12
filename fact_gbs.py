"""GBS vs PSR on factored processes (fact_proc.py, models from train_fact.py).

State sources, at history length t (the model's last position t): exact joint belief (reference), raw residual
at layer p = 0..L (input of block p; L = final), and the concatenation of layers 1..L.

1. Bottleneck sweep (Artemy's GBS construction, without labels): affine encoder state -> d dims, then
     lin  : affine decoder to the 2-step future PROBABILITIES (closed-form reduced-rank regression; this is the PSR /
            Hankel form: future linear in the state)
     soft : affine decoder to 2-step future LOGITS + softmax (log-linear)
     mlp  : MLP decoder (2 hidden layers of 256) + softmax
   Score: held-out KL(exact 2-step future || decoded), and the share of KL captured relative to predicting the mean.
2. Linear probes (held-out R^2) from each state to the PSR state (joint belief, 2^K), to the factor marginals (K) and
   the factor log-odds (K), next-token and 2-step future probabilities.
3. Causal, label-free factor test at layers 1..L: take the d = K mlp encoder, find independent coordinates of its K
   outputs by FastICA (factor beliefs of independent factors are independent random variables), and swap ONE
   coordinate from donor j into recipient i (residual moved along the pseudo-inverse direction, single position t).
   Labels (the token bits) are used only to score: the change of each bit's predicted log-odds. Also the swap of all
   K coordinates vs the swap of the whole residual vector. Control: a random rotation instead of ICA.

    python fact_gbs.py --run runs_fact/prod4/s0 [--untrained]
"""
import argparse
import importlib.util
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import FastICA

import fact_proc as fp

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('rp', os.path.join(_here, 'rrxor_plateau.py'))
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)
dev = 'cuda'


def kl_rows(p, logq):
    return (p * (p.clamp_min(1e-300).log() - logq)).sum(-1).mean().item()


def r2(Xtr, Ytr, Xte, Yte, lam=1e-4):
    mu, sd = Xtr.mean(0), Xtr.std(0).clip(1e-8)
    A = np.c_[(Xtr - mu) / sd, np.ones(len(Xtr))]
    B = np.c_[(Xte - mu) / sd, np.ones(len(Xte))]
    W = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ Ytr)
    res = Yte - B @ W
    return float(1 - (res ** 2).sum() / ((Yte - Ytr.mean(0)) ** 2).sum())


def rrr_kl(Xtr, Ytr, Xte, Yte, ds):
    """Reduced-rank regression state -> future probabilities (affine), rank d; KL after clipping."""
    mu, sd = Xtr.mean(0), Xtr.std(0).clip(1e-8)
    A = np.c_[(Xtr - mu) / sd, np.ones(len(Xtr))]
    B = np.c_[(Xte - mu) / sd, np.ones(len(Xte))]
    W = np.linalg.lstsq(A, Ytr, rcond=None)[0]
    Ftr, Fte = A @ W, B @ W
    ym = Ftr.mean(0)
    _, _, Vt = np.linalg.svd(Ftr - ym, full_matrices=False)
    out = {}
    for d in ds:
        P = (Fte - ym) @ Vt[:d].T @ Vt[:d] + ym
        P = np.clip(P, 1e-6, None)
        P /= P.sum(1, keepdims=True)
        out[d] = float((Yte * (np.log(np.clip(Yte, 1e-300, None)) - np.log(P))).sum(1).mean())
    return out


class Bottle(nn.Module):
    def __init__(self, D, d, nout, kind):
        super().__init__()
        self.enc = nn.Linear(D, d)
        if kind == 'soft':
            self.dec = nn.Linear(d, nout)
        else:
            self.dec = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, nout))

    def forward(self, x):
        return torch.log_softmax(self.dec(self.enc(x)), -1)


def fit(Xtr, Ytr, Xte, Yte, d, kind, steps=3000, seed=0):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-8)
    xtr, xte = (Xtr - mu) / sd, (Xte - mu) / sd
    net = Bottle(Xtr.shape[1], d, Ytr.shape[1], kind).to(dev).double()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-4)
    for _ in range(steps):
        loss = -(Ytr * net(xtr)).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    with torch.no_grad():
        return kl_rows(Yte, net(xte)), net, mu, sd


@torch.no_grad()
def collect(m, toks, t, p, bt=4096):
    return torch.cat([rp.run(m, toks[s:s + bt, :t + 1], collect=(p, t))[1] for s in range(0, len(toks), bt)]).double()


@torch.no_grad()
def model_fut2(m, hist, V, bt=2048):
    """(n, V^2) model's own 2-step future after hist (n, t)."""
    out = []
    for s in range(0, len(hist), bt):
        h = hist[s:s + bt]
        n, t = h.shape
        x1 = torch.arange(V, device=dev).repeat(n)
        seq = torch.cat([h.repeat_interleave(V, 0), x1[:, None], torch.zeros_like(x1)[:, None]], 1)
        lg, _ = rp.run(m, seq)
        lp = torch.log_softmax(lg.double(), -1)
        l1 = lp[::V, t]                                      # (n, V) p(x1)
        l2 = lp[:, t + 1].view(n, V, V)                      # p(x2 | x1)
        out.append((l1[:, :, None] + l2).reshape(n, V * V))
    return torch.cat(out)


@torch.no_grad()
def bit_logodds(m, hist_t1, p, t, delta, K):
    lg, _ = rp.run(m, hist_t1, inj=(p, t, delta))
    pr = torch.softmax(lg[:, t].double(), -1)
    Bm = fp.bits(2 ** K, K).to(dev).double()
    q = (pr @ Bm).clamp(1e-9, 1 - 1e-9)
    return (q / (1 - q)).log()


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--run', required=True)
    pa.add_argument('--untrained', action='store_true')
    pa.add_argument('--t', type=int, default=24)
    pa.add_argument('--n', type=int, default=24000)
    pa.add_argument('--ntest', type=int, default=4000)
    pa.add_argument('--steps', type=int, default=3000)
    a = pa.parse_args()
    ck = torch.load(a.run + '.pt', map_location='cpu', weights_only=True)
    K, c = int(ck['args']['K']), float(ck['args']['c'])
    cfg = rp.Config(**ck['config'])
    if a.untrained:
        m = rp.untrained(cfg, float(ck['args']['init_std']), int(ck['args']['seed']))
    else:
        m = rp.Transformer(cfg); m.load_state_dict(ck['model'])
    m = m.to(dev).eval()
    for prm in m.parameters():
        prm.requires_grad_(False)
    V, t, L = 2 ** K, a.t, cfg.n_layers
    T = fp.transition_tensor(K, c=c)
    toks = fp.sample(T, a.n, t + 1, torch.Generator(device=dev).manual_seed(777), dev)
    bel = fp.beliefs(T, toks[:, :t])[:, t]
    fut2 = fp.futures(T, bel, 2)
    nxt = fp.next_probs(T, bel)
    marg = fp.factor_marginals(bel, K).clamp(1e-9, 1 - 1e-9)
    lo = (marg / (1 - marg)).log()
    tr, te = slice(0, a.n - a.ntest), slice(a.n - a.ntest, a.n)
    S = {'exact_belief': bel}
    for p in range(L + 1):
        S[f'p{p}'] = collect(m, toks, t, p)
    S['all'] = torch.cat([S[f'p{p}'] for p in range(1, L + 1)], 1)
    mf2 = model_fut2(m, toks[te, :t], V)
    out = dict(run=a.run, untrained=a.untrained, K=K, c=c, t=t, psr_rank=V - 1, gbs_dim_truth=K if c == 0 else None,
               test_kl_train=ck.get('test_kl'), model_kl_2step=kl_rows(fut2[te], mf2),
               kl_mean_2step=kl_rows(fut2[te], fut2[tr].mean(0).log().expand(a.ntest, -1)))
    print(f"{a.run}{' untrained' if a.untrained else ''}: K={K} c={c} | model's own 2-step KL {out['model_kl_2step']:.2e}"
          f" | KL of the mean {out['kl_mean_2step']:.3f}", flush=True)

    # 2. linear probes
    out['probes'] = {}
    tg = dict(joint_belief=bel, factor_marg=marg, factor_logodds=lo, next=nxt, fut2=fut2)
    for sn, X in S.items():
        Xn = X.cpu().numpy()
        out['probes'][sn] = {k: r2(Xn[tr], Y.cpu().numpy()[tr], Xn[te], Y.cpu().numpy()[te]) for k, Y in tg.items()}
        print(f"  probe {sn:13s} " + ' '.join(f"{k} {v:.3f}" for k, v in out['probes'][sn].items()), flush=True)

    # 1. bottleneck sweep
    ds = list(range(1, min(V, 8) + 1)) + ([10, 12, 15] if V > 8 else [])
    out['bottleneck'] = {}
    encs = {}
    for sn in ['exact_belief'] + [f'p{p}' for p in range(1, L + 1)] + ['all']:
        X = S[sn]
        Xn, Yn = X.cpu().numpy(), fut2.cpu().numpy()
        row = dict(lin=rrr_kl(Xn[tr], Yn[tr], Xn[te], Yn[te], ds), soft={}, mlp={})
        for d in ds:
            row['soft'][d] = fit(X[tr], fut2[tr], X[te], fut2[te], d, 'soft', a.steps)[0]
            kl, net, mu, sd = fit(X[tr], fut2[tr], X[te], fut2[te], d, 'mlp', a.steps)
            row['mlp'][d] = kl
            if d == K:
                encs[sn] = (net, mu, sd)
        out['bottleneck'][sn] = row
        for kind in ('lin', 'soft', 'mlp'):
            print(f"  KL {sn:13s} {kind:4s} " + ' '.join(f"d{d}:{row[kind][d]:.1e}" for d in ds), flush=True)

    # 3. causal label-free factor swaps
    out['factor_swaps'] = {}
    rng = np.random.default_rng(0)
    idx = np.arange(a.n)[te]
    ii, jj = rng.choice(idx, 2000), rng.choice(idx, 2000)
    for p in range(1, L + 1):
        net, mu, sd = encs[f'p{p}']
        W = net.enc.weight.detach() / sd[None, :]                  # z = W x + const (raw residual)
        Z = (S[f'p{p}'] @ W.T).cpu().numpy()
        res = {}
        for mode, dmode in (('ica', 'pattern'), ('ica', 'pinv'), ('random', 'pattern')):
            if mode == 'ica':
                ica = FastICA(n_components=K, whiten='unit-variance', random_state=0, max_iter=2000)
                ica.fit(Z[tr])
                U = torch.tensor(ica.components_, device=dev)     # s = U (z - mean)
            else:
                C = np.cov(Z[tr].T)
                ev, evec = np.linalg.eigh(C)
                Wh = evec @ np.diag(ev.clip(1e-12) ** -0.5) @ evec.T
                Q, _ = np.linalg.qr(rng.standard_normal((K, K)))
                U = torch.tensor(Q @ Wh, device=dev)
            A = U @ W                                              # (K, D): s = A x + const
            s_all = S[f'p{p}'] @ A.T
            if dmode == 'pinv':                                    # minimal-norm move that changes s only
                Dirs = torch.linalg.pinv(A)                        # (D, K)
            else:                                                  # pattern: regression of x on s (moves the
                xc = S[f'p{p}'][tr] - S[f'p{p}'][tr].mean(0)        # redundant copies of the coordinate too)
                sc = s_all[tr] - s_all[tr].mean(0)
                Dirs = torch.linalg.lstsq(sc, xc).solution.T        # (D, K)
            xi, xj = S[f'p{p}'][ii], S[f'p{p}'][jj]
            base = bit_logodds(m, toks[ii, :t + 1], p, t, torch.zeros_like(xi).float(), K)
            donor = bit_logodds(m, toks[jj, :t + 1], p, t, torch.zeros_like(xj).float(), K)
            target = donor - base
            E = np.zeros((K, K)); TR = np.zeros((K, K))
            for k in range(K):
                dk = (Dirs[:, k:k + 1] * (s_all[jj, k] - s_all[ii, k])[None, :]).T
                ch = bit_logodds(m, toks[ii, :t + 1], p, t, dk.float(), K) - base
                E[k] = ch.abs().mean(0).cpu().numpy()
                for l in range(K):
                    x_, y_ = ch[:, l].cpu().numpy(), target[:, l].cpu().numpy()
                    TR[k, l] = float(np.corrcoef(x_, y_)[0, 1]) if x_.std() > 0 else 0.0
            dall = (s_all[jj] - s_all[ii]) @ Dirs.T
            ch_all = bit_logodds(m, toks[ii, :t + 1], p, t, dall.float(), K) - base
            ch_full = bit_logodds(m, toks[ii, :t + 1], p, t, (xj - xi).float(), K) - base
            fr = lambda ch: float(1 - ((ch - target) ** 2).sum() / (target ** 2).sum())
            spec = E.max(1) / E.sum(1)
            match = E.argmax(1)
            sp_ic = np.array([[abs(np.corrcoef(s_all[te, k].cpu().numpy(), lo[te, l].cpu().numpy())[0, 1])
                               for l in range(K)] for k in range(K)])
            res[f'{mode}_{dmode}'] = dict(effect=E.tolist(), transfer_corr=TR.tolist(), specificity=spec.tolist(),
                             matched_bit=match.tolist(), distinct_bits=int(len(set(match.tolist()))),
                             transfer_matched=[TR[k, match[k]] for k in range(K)],
                             r2_transfer_allK=fr(ch_all), r2_transfer_full_vector=fr(ch_full),
                             corr_coord_logodds=sp_ic.tolist())
            print(f"  swap p{p} {mode}/{dmode:7s}: specificity {np.round(spec, 2)} bits {match} | transfer r (matched) "
                  f"{np.round(res[f'{mode}_{dmode}']['transfer_matched'], 2)} | |corr(coord, factor log-odds)| max per coord "
                  f"{np.round(sp_ic.max(1), 2)} | R2 transfer all-K {res[f'{mode}_{dmode}']['r2_transfer_allK']:.2f} "
                  f"vs full vector {res[f'{mode}_{dmode}']['r2_transfer_full_vector']:.2f}", flush=True)
        out['factor_swaps'][f'p{p}'] = res
    os.makedirs('results/fact', exist_ok=True)
    name = a.run.replace('runs_fact/', '').replace('/', '_') + ('_untrained' if a.untrained else '')
    json.dump(out, open(f'results/fact/gbs_{name}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
