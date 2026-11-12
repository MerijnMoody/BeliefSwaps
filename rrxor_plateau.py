"""Per-layer plateau partitions on the RRXOR models of artemyk/belief-state (Shai et al. 2024 architecture), pilot.

Same method as plateau_algo.py: at residual point p (before block p; p = L is the final residual, before ln_final) and
history length t (model position t has read t tokens), the distance between contexts i and j is the largest
KL(clean exact K-step future || intervened future) over (j's activation into i, i's into j, i halfway to j). Groups =
complete linkage at the Otsu threshold on log10 distance (no labels) and at a 10x stricter "fine" threshold.

Labels are used only to score the groups (normalized conditional entropies, as in plateau_algo.py):
  belief      exact Bayesian belief over the 5 hidden states (eps = 0: 36 mixed states incl. the start)
  future      the exact K-step future distribution (the most the K-step test can resolve)
  phase       belief summed within phases {A}, {B, C}, {D, E}
  last1, last2, xor2   the last token, the last two tokens, their XOR
  ph_last2    (phase, last2): a product of a phase machine and a last-two-tokens machine
  ph_xor      (phase, xor2)
  bel_last2   (belief, last2)
Random-grouping baseline: H(V | C) / H(V) for the partition's labels shuffled across contexts (same class sizes).

    python rrxor_plateau.py --arch L4_d64 --eps 0 --seed 0 [--untrained]
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, fcluster

import importlib.util

dev = 'cuda'
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'external', 'belief-state')
# their model.py clashes with ours by name, so load it under its own module name
_spec = importlib.util.spec_from_file_location('bs_model', os.path.join(REPO, 'model.py'))
bs_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bs_model)
Config, Transformer, untrained = bs_model.Config, bs_model.Transformer, bs_model.untrained


def otsu(x):  # same as plateau_algo.py
    x = np.sort(x)
    best, thr = -1, x[len(x) // 2]
    for i in range(1, len(x)):
        w0, w1 = i / len(x), 1 - i / len(x)
        v = w0 * w1 * (x[:i].mean() - x[i:].mean()) ** 2
        if v > best:
            best, thr = v, 0.5 * (x[i - 1] + x[i])
    return thr, best / x.var()


def H(v):
    _, c = np.unique(v, return_counts=True, axis=0)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def Hcond(a, b):
    """H(a | b)."""
    return H(np.stack([a, b], 1)) - H(b)
POS = (4, 6, 9, 14, 28)
A_, B_, C_, D_, E_ = range(5)


def transition_tensor(eps):
    emit = torch.zeros(2, 5, 5, dtype=torch.float64)
    emit[0, A_, B_] = emit[1, A_, C_] = 0.5
    emit[0, B_, E_] = emit[1, B_, D_] = 0.5
    emit[0, C_, D_] = emit[1, C_, E_] = 0.5
    emit[1, D_, A_] = emit[0, E_, A_] = 1.0
    jump = torch.full((5, 5), eps / 4, dtype=torch.float64)
    jump.fill_diagonal_(1 - eps)
    return jump @ emit


def stationary(T):
    vals, vecs = torch.linalg.eig(T.sum(0).T)
    pi = vecs[:, torch.argmin((vals - 1).abs())].real
    return pi / pi.sum()


def sample(T, n, length, gen):
    pi = stationary(T)
    joint = T.permute(1, 0, 2).reshape(5, 10)               # joint[s, 5x + t]
    s = torch.multinomial(pi, n, replacement=True, generator=gen)
    toks = torch.empty(n, length, dtype=torch.long)
    for i in range(length):
        o = torch.multinomial(joint[s], 1, generator=gen)[:, 0]
        toks[:, i], s = o // 5, o % 5
    return toks


def beliefs(T, toks):
    """(n, length + 1, 5): belief before the first token and after each prefix."""
    n, L = toks.shape
    b = stationary(T).expand(n, 5).clone()
    out = [b]
    for i in range(L):
        b = torch.einsum('ns,nst->nt', b, T[toks[:, i]])
        b = b / b.sum(1, keepdim=True)
        out.append(b)
    return torch.stack(out, 1)


def futures(T, b, K):
    """(n, 2^K) exact K-step future distribution from beliefs b (n, 5)."""
    br = torch.cartesian_prod(*[torch.arange(2)] * K)
    M = torch.stack([torch.linalg.multi_dot([T[c] for c in row]) if K > 1 else T[row[0]] for row in br])  # (2^K, 5, 5)
    return torch.einsum('ns,cst->nc', b, M)


def ids(x, dec=6):
    """Integer ids of the distinct rows of x (rounded)."""
    x = np.round(np.asarray(x, dtype=np.float64), dec).reshape(len(x), -1)
    _, inv = np.unique(x, axis=0, return_inverse=True)
    return inv.reshape(-1)


def reachable_counts(T, maxlen):
    """Exact numbers of distinct values of each label over all positive-probability histories of length 2..maxlen
    (breadth-first over (belief, last two tokens))."""
    key = lambda b: tuple(np.round(b.numpy(), 6))
    start = stationary(T)
    layer = {(key(start), (-1, -1)): start}
    seen = {nm: set() for nm in ('belief', 'phase', 'ph_last2', 'ph_xor', 'bel_last2')}
    for t in range(1, maxlen + 1):
        nxt = {}
        for (kb, l2), b in layer.items():
            for x in (0, 1):
                u = b @ T[x]
                if u.sum() < 1e-12:
                    continue
                u = u / u.sum()
                nl2 = (l2[1], x)
                nxt[(key(u), nl2)] = u
        layer = nxt
        if t >= 2:
            for (kb, l2), b in layer.items():
                ph = tuple(np.round([b[0].item(), (b[1] + b[2]).item(), (b[3] + b[4]).item()], 6))
                seen['belief'].add(kb)
                seen['phase'].add(ph)
                seen['ph_last2'].add((ph, l2))
                seen['ph_xor'].add((ph, l2[0] ^ l2[1]))
                seen['bel_last2'].add((kb, l2))
    return {k: len(v) for k, v in seen.items()}


def run(m, toks, inj=None, collect=None):
    """Logits of the belief-state Transformer with delta added to the residual at (point p, position t)."""
    cfg = m.cfg
    inputs = torch.cat([torch.full_like(toks[:, :1], cfg.vocab), toks[:, :-1]], 1)
    x = m.embed(inputs)
    if m.pos_embed is not None:
        x = x + m.pos_embed(torch.arange(toks.shape[1], device=toks.device))
    Ln = toks.shape[1]
    mask = torch.ones(Ln, Ln, dtype=torch.bool, device=toks.device).tril()
    mask[1:, 0] = False
    got = None
    for p in range(cfg.n_layers + 1):
        if inj is not None and inj[0] == p:
            x = x.clone()
            x[:, inj[1]] += inj[2]
        if collect is not None and collect[0] == p:
            got = x[:, collect[1]].clone()
        if p < cfg.n_layers:
            x, _ = m.blocks[p](x, mask)
    return (m.unembed(m.ln_final(x)), got)


@torch.no_grad()
def future_logp(m, hist, t, p, delta, K):
    """(n, 2^K) model log-probabilities of every K-token continuation after history hist[:, :t], with delta (n, D)
    added at point p, position t."""
    n = len(hist)
    br = torch.cartesian_prod(*[torch.arange(2, device=dev)] * K)
    nb = len(br)
    seq = torch.cat([hist[:, :t].repeat_interleave(nb, 0), br.repeat(n, 1)], 1)
    lg, _ = run(m, seq, inj=(p, t, delta.repeat_interleave(nb, 0)))
    lp = torch.log_softmax(lg.double(), -1)
    tgt = br.repeat(n, 1)
    s = sum(lp[:, t + j].gather(1, tgt[:, j:j + 1])[:, 0] for j in range(K))
    return s.view(n, nb)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--arch', default='L4_d64')
    pa.add_argument('--eps', default='0')
    pa.add_argument('--seed', type=int, default=0)
    pa.add_argument('--n', type=int, default=200)
    pa.add_argument('--K', type=int, default=4)
    pa.add_argument('--untrained', action='store_true')
    pa.add_argument('--n_shuffle', type=int, default=50)
    pa.add_argument('--swaps_only', action='store_true', help='drop the halfway point (only real activations)')
    a = pa.parse_args()
    ck = torch.load(os.path.join(REPO, 'runs', a.arch, f'eps{a.eps}', f's{a.seed}.pt'), map_location='cpu',
                    weights_only=True)
    cfg = Config(**ck['config'])
    if a.untrained:
        m = untrained(cfg, float(ck['args']['init_std']), int(ck['args']['seed']))
    else:
        m = Transformer(cfg)
        m.load_state_dict(ck['model'])
    m = m.to(dev).eval()
    for prm in m.parameters():
        prm.requires_grad_(False)
    eps = float(a.eps)
    T = transition_tensor(eps)
    K, n, L = a.K, a.n, cfg.n_layers
    assert max(POS) + K <= cfg.context

    # horizon check: does the K-step future separate all beliefs? (eps = 0, exact enumeration over histories)
    out = dict(arch=a.arch, eps=eps, seed=a.seed, n=n, K=K, untrained=a.untrained, test_kl=float(ck['test_kl']),
               points={})
    if eps == 0:
        allh = torch.cartesian_prod(*[torch.arange(2)] * 12)
        bb = beliefs(T, allh)
        ok = torch.isfinite(bb).all(-1)
        flat = bb[ok]
        out['n_beliefs_reachable_len<=12'] = int(len(np.unique(ids(flat))))
        out['n_futureK_classes_of_those'] = int(len(np.unique(ids(futures(T, flat, K)))))
        out['reachable_counts_len<=28'] = reachable_counts(T, 28)
        print('beliefs reachable (len<=12):', out['n_beliefs_reachable_len<=12'], '| distinct', K, '-step futures:',
              out['n_futureK_classes_of_those'], '| reachable label counts (len 2..28):',
              out['reachable_counts_len<=28'], flush=True)

    gen = torch.Generator().manual_seed(4242)
    toks = sample(T, n, cfg.context, gen)
    bel = beliefs(T, toks)                                      # bel[:, t] = belief after t tokens
    hist = toks.to(dev)
    with torch.no_grad():                                         # sanity: our forward = theirs; model vs Bayes
        lg_ours, _ = run(m, hist)
        lg_theirs = m(hist)
        assert (lg_ours - lg_theirs).abs().max() < 1e-4, (lg_ours - lg_theirs).abs().max()
        opt = torch.einsum('nls,sx->nlx', bel[:, :-1], T.sum(2).T)
        lq = torch.log_softmax(lg_ours.double().cpu(), -1)
        kl = (opt * (opt.clamp_min(1e-300).log() - lq)).sum(-1).mean()
        out['sanity_kl_to_bayes'] = float(kl)
        print(f'sanity: forward matches; mean next-token KL to Bayes {float(kl):.2e} (checkpoint test_kl '
              f'{out["test_kl"]:.2e})', flush=True)
    parts = {}
    for p in range(L + 1):
        for t in POS:
            _, Aact = run(m, hist[:, :t + 1], collect=(p, t))
            Aact = Aact.double()
            lp0 = future_logp(m, hist, t, p, torch.zeros(n, cfg.d_model, device=dev), K)
            ii, jj = torch.triu_indices(n, n, 1, device=dev)
            d = torch.zeros(len(ii), device=dev, dtype=torch.float64)
            Bt = 2048
            for s in range(0, len(ii), Bt):
                i, j = ii[s:s + Bt], jj[s:s + Bt]
                moves = ((i, j, 1.0), (j, i, 1.0)) if a.swaps_only else ((i, j, 1.0), (j, i, 1.0), (i, j, 0.5))
                for src, dst, al in moves:
                    lpa = future_logp(m, hist[src], t, p, (al * (Aact[dst] - Aact[src])).float(), K)
                    kl = (lp0[src].exp() * (lp0[src] - lpa)).sum(1)
                    d[s:s + Bt] = torch.maximum(d[s:s + Bt], kl)
            ld = np.log10(d.clamp_min(1e-12).cpu().numpy())
            thr, sep = otsu(ld)
            off = ld.min() - 1e-6
            Z = linkage(ld - off, method='complete')
            lab = fcluster(Z, t=thr - off, criterion='distance')
            lab_fine = fcluster(Z, t=thr - 1 - off, criterion='distance')
            parts[(p, t)] = lab
            # labels
            b = bel[:, t]
            x = toks.numpy()
            last2 = x[:, t - 2] * 2 + x[:, t - 1]
            xor2 = x[:, t - 2] ^ x[:, t - 1]
            ph = torch.stack([b[:, 0], b[:, 1] + b[:, 2], b[:, 3] + b[:, 4]], 1)
            bid, phid = ids(b), ids(ph)
            cand = dict(belief=bid, future=ids(futures(T, b, K)), phase=phid, last1=x[:, t - 1], last2=last2,
                        xor2=xor2, ph_last2=ids(np.stack([phid, last2], 1), 0), ph_xor=ids(np.stack([phid, xor2], 1), 0),
                        bel_last2=ids(np.stack([bid, last2], 1), 0))
            rng = np.random.default_rng(0)
            row = dict(threshold=float(10 ** thr), separation=float(sep), n_clusters=int(lab.max()),
                       n_clusters_fine=int(lab_fine.max()),
                       n_distinct={k: int(len(np.unique(v))) for k, v in cand.items()},
                       d_quantiles=[float(10 ** q) for q in np.quantile(ld, [0.05, 0.25, 0.5, 0.75, 0.95])],
                       scores={}, scores_fine={}, labels=lab.tolist(), labels_fine=lab_fine.tolist())
            for key_, lb in (('scores', lab), ('scores_fine', lab_fine)):
                for nm, v in cand.items():
                    if H(v) < 1e-9:
                        continue
                    shuf = np.mean([Hcond(v, rng.permutation(lb)) / H(v) for _ in range(a.n_shuffle)])
                    row[key_][nm] = dict(V_given_C=Hcond(v, lb) / H(v), C_given_V=Hcond(lb, v) / max(H(lb), 1e-12),
                                         V_given_C_shuffled=float(shuf))
            out['points'][f'p{p}_t{t}'] = row
            sc, scf = row['scores'], row['scores_fine']
            print(f"p{p} t{t}: {row['n_clusters']} / {row['n_clusters_fine']} fine groups; distinct "
                  f"belief {row['n_distinct']['belief']}, future {row['n_distinct']['future']}, ph_last2 "
                  f"{row['n_distinct']['ph_last2']}; sep {sep:.2f}\n   " +
                  ' | '.join(f"{nm}: H(V|C) {v['V_given_C']:.2f}/{scf[nm]['V_given_C']:.2f} "
                             f"[shuf {scf[nm]['V_given_C_shuffled']:.2f}] H(C|V) {v['C_given_V']:.2f}/"
                             f"{scf[nm]['C_given_V']:.2f}" for nm, v in sc.items()), flush=True)
    out['transitions'] = {}
    for p in range(L):
        vals = [Hcond(parts[(p + 1, t)], parts[(p, t)]) / max(H(parts[(p + 1, t)]), 1e-12) for t in POS]
        out['transitions'][f'{p}->{p + 1}'] = vals
        print(f"transition p{p} -> p{p + 1}: " + ' '.join(f'{v:.2f}' for v in vals), flush=True)
    os.makedirs('results/plateau', exist_ok=True)
    tag = ('_untrained' if a.untrained else '') + ('_swaps' if a.swaps_only else '')
    json.dump(out, open(f'results/plateau/rrxor_{a.arch}_eps{a.eps}_s{a.seed}{tag}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
