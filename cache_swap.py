"""Swap groups for other parts of the transformer state (RRXOR / MRRXOR, artemyk/belief-state checkpoints).

rrxor_plateau.py swaps one residual vector: layer p, the last history position t. Here, with the same distance
(max over both directions of KL(clean exact K-step future || swapped future), all pairs of 200 contexts, swaps only)
and the same grouping (Otsu threshold on log10 distance + complete linkage), two other targets:

  pos   one vector at layer p, history position q (1 <= q <= t; q = t is the original test). The future is still the
        K tokens after the full history of length t. Shows where in the history the distinctions the future needs sit.
  slice the vectors at layer p at ALL history positions 1..t at once (one horizontal slice of the cache). Below layer
        p the continuation still reads i's own cache. Compared with pos q = t, this shows whether distinctions merged
        by the single-vector swap sit at other positions of the same layer.

Swapping every layer at every position would turn i into j, so it is not a test; the slice at p = 0 is that case.

Labels (scoring only): belief / phase / last two tokens / XOR at the end of the history (t), and for pos q also the
token, last two tokens, XOR, belief and phase at q, the XOR check at q (is token q-1 the XOR of the two before it) and
the last three checks up to q.

    python cache_swap.py --arch L4_d64 --eps 0 --seed 0
"""
import argparse
import importlib.util
import json
import os

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, fcluster

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('rp', os.path.join(_here, 'rrxor_plateau.py'))
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)
dev = 'cuda'


@torch.no_grad()
def flp(m, hist, t, p, pos, delta, K, br):
    """(n, 2^K) log-probs of every K-token continuation after hist[:, :t], with delta added at layer p, position(s)
    pos (int: delta (n, D); slice: delta (n, len, D))."""
    n, nb = len(hist), len(br)
    seq = torch.cat([hist[:, :t].repeat_interleave(nb, 0), br.repeat(n, 1)], 1)
    lg, _ = rp.run(m, seq, inj=(p, pos, delta.repeat_interleave(nb, 0)))
    lp = torch.log_softmax(lg.double(), -1)
    tgt = br.repeat(n, 1)
    return sum(lp[:, t + j].gather(1, tgt[:, j:j + 1])[:, 0] for j in range(K)).view(n, nb)


@torch.no_grad()
def distances(m, hist, t, p, pos, A, K, br, bt=2048):
    n = len(hist)
    zero = torch.zeros_like(A)
    lp0 = flp(m, hist, t, p, pos, zero, K, br)
    ii, jj = torch.triu_indices(n, n, 1, device=dev)
    d = torch.zeros(len(ii), device=dev, dtype=torch.float64)
    for s in range(0, len(ii), bt):
        i, j = ii[s:s + bt], jj[s:s + bt]
        for src, dst in ((i, j), (j, i)):
            lpa = flp(m, hist[src], t, p, pos, (A[dst] - A[src]).float(), K, br)
            d[s:s + bt] = torch.maximum(d[s:s + bt], (lp0[src].exp() * (lp0[src] - lpa)).sum(1))
    return d.cpu().numpy()


def groups(d):
    ld = np.log10(np.clip(d, 1e-12, None))
    thr, sep = rp.otsu(ld)
    off = ld.min() - 1e-6
    Z = linkage(ld - off, method='complete')
    return fcluster(Z, t=thr - off, criterion='distance'), float(10 ** thr), float(sep)


def score(lab, cand):
    out = {}
    for nm, v in cand.items():
        if rp.H(v) < 1e-9:
            continue
        out[nm] = dict(V_given_C=round(rp.Hcond(v, lab) / rp.H(v), 3),
                       C_given_V=round(rp.Hcond(lab, v) / max(rp.H(lab), 1e-12), 3))
    return out


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--arch', default='L4_d64')
    pa.add_argument('--eps', default='0')
    pa.add_argument('--seed', type=int, default=0)
    pa.add_argument('--K', type=int, default=4)
    pa.add_argument('--ts', default='6,9')
    pa.add_argument('--positions', default='all', choices=['all', 'last'])
    pa.add_argument('--untrained', action='store_true')
    a = pa.parse_args()
    ck = torch.load(os.path.join(rp.REPO, 'runs', a.arch, f'eps{a.eps}', f's{a.seed}.pt'), map_location='cpu',
                    weights_only=True)
    cfg = rp.Config(**ck['config'])
    if a.untrained:
        m = rp.untrained(cfg, float(ck['args']['init_std']), int(ck['args']['seed'])).to(dev).eval()
    else:
        m = rp.Transformer(cfg); m.load_state_dict(ck['model']); m = m.to(dev).eval()
    for prm in m.parameters():
        prm.requires_grad_(False)
    T = rp.transition_tensor(float(a.eps))
    toks = rp.sample(T, 200, cfg.context, torch.Generator().manual_seed(4242))
    bel = rp.beliefs(T, toks)
    x = toks.numpy()
    hist = toks.to(dev)
    K = a.K
    br = torch.cartesian_prod(*[torch.arange(2, device=dev)] * K)
    phase = lambda b: rp.ids(torch.stack([b[:, 0], b[:, 1] + b[:, 2], b[:, 3] + b[:, 4]], 1))
    chk = lambda q: (x[:, q - 1] == (x[:, q - 3] ^ x[:, q - 2])).astype(int) if q >= 3 else np.full(200, -1)
    out = dict(arch=a.arch, eps=a.eps, seed=a.seed, K=K, cells={})
    for t in map(int, a.ts.split(',')):
        end = dict(belief_t=rp.ids(bel[:, t]), phase_t=phase(bel[:, t]), last2_t=x[:, t - 2] * 2 + x[:, t - 1],
                   xor_t=x[:, t - 2] ^ x[:, t - 1])
        for p in range(1, cfg.n_layers + 1):
            qs = range(1, t + 1) if a.positions == 'all' else [t]
            targets = [('pos', q) for q in qs] + [('slice', None)]
            for mode, q in targets:
                pos = q if mode == 'pos' else slice(1, t + 1)
                _, A = rp.run(m, hist[:, :t + 1], collect=(p, pos))
                d = distances(m, hist, t, p, pos, A.double(), K, br)
                lab, thr, sep = groups(d)
                cand = dict(end)
                if mode == 'pos':
                    cand.update(tok_q=x[:, q - 1], belief_q=rp.ids(bel[:, q]), phase_q=phase(bel[:, q]))
                    if q >= 2:
                        cand.update(last2_q=x[:, q - 2] * 2 + x[:, q - 1], xor_q=x[:, q - 2] ^ x[:, q - 1])
                    if q >= 3:
                        cand['check_q'] = chk(q)
                    if q >= 5:
                        cand['checks3_q'] = rp.ids(np.stack([chk(q), chk(q - 1), chk(q - 2)], 1), 0)
                key = f't{t}_p{p}_' + (f'q{q}' if mode == 'pos' else 'slice')
                row = dict(n_groups=int(lab.max()), threshold=thr, separation=sep,
                           d_q05_50_95=[float(v) for v in np.quantile(d, [0.05, 0.5, 0.95])],
                           frac_harmful=float((d > thr).mean()), scores=score(lab, cand), labels=lab.tolist())
                out['cells'][key] = row
                best = sorted(row['scores'].items(), key=lambda kv: kv[1]['V_given_C'] + kv[1]['C_given_V'])[:3]
                print(f"{key}: {row['n_groups']} groups, harmful {row['frac_harmful']:.2f}, median d {row['d_q05_50_95'][1]:.1e} | "
                      + ' '.join(f"{k} {v['V_given_C']:.2f}/{v['C_given_V']:.2f}" for k, v in best)
                      + f" | belief_t {row['scores']['belief_t']['V_given_C']:.2f}/{row['scores']['belief_t']['C_given_V']:.2f}",
                      flush=True)
        json.dump(out, open(f'results/plateau/cache_{a.arch}_eps{a.eps}_s{a.seed}' + ('_untrained' if a.untrained else '') + '.json', 'w'))


if __name__ == '__main__':
    main()
