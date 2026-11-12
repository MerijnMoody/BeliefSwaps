"""Swap groups, activation plateaus and the output continuum on factored processes (models from train_fact.py).

Same swap as rrxor_plateau.py / cache_swap.py: recipient i, donor j, j's residual at layer p put into i's forward pass,
either at the last history position t only ('last') or at all history positions 1..t ('slice'). Distance = max over
both directions of KL(model's clean future of the recipient || patched future), over all V^Ks continuations of Ks
tokens (teacher-forced enumeration; V^Ks = 256). Groups: Otsu threshold on log10 distance + complete linkage.

A. Groups per component, scored against the joint belief and each factor's belief (labels only for scoring).
   Product test (discrete factors): is the partition C the product of a partition of factor-A beliefs and a partition
   of factor-B beliefs? gA merges two A-beliefs if, for every B-belief seen with both, the two contexts share a group
   (transitive closure); C is a product partition iff C == (gA, gB). Reported as normalized H(C | gA, gB) and
   H(gA, gB | C); 0 / 0 = exact product.
B. Plateaus: interpolate the last-position vector at layer p from i to j (41 steps). Total curve d(alpha) and, using
   the token bits only for this decomposition, the KL of each factor's marginal future (KL of a product distribution
   is the sum over factors). Jumps = runs of steps whose step-KL exceeds the group threshold. Question: if two factors
   differ, are there two jumps at different alpha (one per factor), or one?
C. Output continuum: the model's clean next-token distributions over many contexts: number of PCs (99 / 99.9%) in
   probability space (the PSR / Hankel form) vs log-probability space (additive over factors for a product process),
   then FastICA on the log-space PCs: |corr| of each component with each factor's log-odds (telegraph) or belief
   (discrete). Same for patched outputs (8 recipients x 2000 donors) per component: dims passed in each space.

    python fact_swap.py --run runs_fact/rrxe/s0 --t 6
"""
import argparse
import importlib.util
import itertools
import json
import os

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import FastICA

import fact_proc as fp

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('rp', os.path.join(_here, 'rrxor_plateau.py'))
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)
dev = 'cuda'
ALPHA = np.linspace(0, 1, 41)


def load(run, untrained):
    ck = torch.load(run + '.pt', map_location='cpu', weights_only=True)
    ar = ck['args']
    proc = ar.get('proc', 'tele')
    if proc == 'tele':
        K = int(ar['K']); T = fp.transition_tensor(K, c=float(ar['c'])); sizes = [2] * K
    else:
        K = 2; T = fp.product_tensor([fp.rrxor_tensor(), fp.even_tensor()]); sizes = [5, 2]
    cfg = rp.Config(**ck['config'])
    m = rp.untrained(cfg, float(ar['init_std']), int(ar['seed'])) if untrained else rp.Transformer(cfg)
    if not untrained:
        m.load_state_dict(ck['model'])
    m = m.to(dev).eval()
    for prm in m.parameters():
        prm.requires_grad_(False)
    return m, cfg, T, K, sizes, proc


def factor_beliefs(bel, sizes):
    """List of (n, S_k) factor beliefs; joint state index has factor 0 least significant."""
    n = len(bel)
    B = bel.reshape(n, *sizes[::-1])
    K = len(sizes)
    out = []
    for k in range(K):
        ax = K - k                                    # dim of factor k in B (dim 0 is batch)
        keep = [d for d in range(1, K + 1) if d != ax]
        out.append(B.sum(keep) if keep else B)
    return out


@torch.no_grad()
def flp(m, hist, t, p, pos, delta, conts, bt=1 << 15):
    """(n, nc) model log-probs of each continuation after hist[:, :t], with delta added at layer p, position(s) pos."""
    n, nc = len(hist), len(conts)
    out = []
    step = max(1, bt // nc)
    for s in range(0, n, step):
        h, dl = hist[s:s + step, :t], delta[s:s + step]
        nn_ = len(h)
        seq = torch.cat([h.repeat_interleave(nc, 0), conts.repeat(nn_, 1)], 1)
        lg, _ = rp.run(m, seq, inj=(p, pos, dl.repeat_interleave(nc, 0)))
        lp = torch.log_softmax(lg.double(), -1)
        tgt = conts.repeat(nn_, 1)
        out.append(sum(lp[:, t + q].gather(1, tgt[:, q:q + 1])[:, 0] for q in range(conts.shape[1])).view(nn_, nc))
    return torch.cat(out)


def groups(d):
    ld = np.log10(np.clip(d, 1e-12, None))
    thr, sep = rp.otsu(ld)
    off = ld.min() - 1e-6
    Z = linkage(ld - off, method='complete')
    return fcluster(Z, t=thr - off, criterion='distance'), float(10 ** thr), float(sep)


def nH(a, b):
    """H(a | b) / H(a)."""
    h = rp.H(a)
    return float(rp.Hcond(a, b) / h) if h > 1e-12 else 0.0


def product_test(C, a, b):
    """Induced partitions gA (of factor-A labels a) and gB, and how far C is from the product (gA, gB)."""
    def induced(x, y):
        xs = np.unique(x)
        parent = {v: v for v in xs}
        def find(v):
            while parent[v] != v:
                v = parent[v]
            return v
        for yv in np.unique(y):
            sel = y == yv
            for gval in np.unique(C[sel]):
                members = np.unique(x[sel & (C == gval)])
                for u in members[1:]:
                    parent[find(u)] = find(members[0])
        return np.array([find(v) for v in x])
    gA, gB = induced(a, b), induced(b, a)
    prod = rp.ids(np.stack([gA, gB], 1), 0)
    return dict(n_gA=int(len(np.unique(gA))), n_gB=int(len(np.unique(gB))), n_C=int(len(np.unique(C))),
                H_C_given_prod=nH(C, prod), H_prod_given_C=nH(prod, C))


def n_pcs(X, fracs=(0.99, 0.999)):
    s = np.linalg.svd(X - X.mean(0), compute_uv=False) ** 2
    c = np.cumsum(s) / max(s.sum(), 1e-300)
    return [int(np.searchsorted(c, f) + 1) for f in fracs]


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--run', required=True)
    pa.add_argument('--untrained', action='store_true')
    pa.add_argument('--t', type=int, default=6)
    pa.add_argument('--n', type=int, default=200)
    pa.add_argument('--npairs', type=int, default=400)
    a = pa.parse_args()
    m, cfg, T, K, sizes, proc = load(a.run, a.untrained)
    V, t, L = T.shape[0], a.t, cfg.n_layers
    Ks = int(round(np.log(256) / np.log(V)))
    conts = torch.tensor(list(itertools.product(range(V), repeat=Ks)), device=dev)
    nc = len(conts)
    # factor-k marginal of a continuation: its k-th bits over the Ks steps
    fidx = [(((conts >> k) & 1) * (2 ** torch.arange(Ks - 1, -1, -1, device=dev))).sum(1) for k in range(K)]
    Mk = [torch.nn.functional.one_hot(f, 2 ** Ks).double() for f in fidx]   # (nc, 2^Ks)

    toks = fp.sample(T, 20000, t + 1, torch.Generator(device=dev).manual_seed(4242), dev)
    bel_all = fp.beliefs(T, toks[:, :t])[:, t]
    fb_all = factor_beliefs(bel_all, sizes)
    hist = toks[:a.n]
    bel, fb = bel_all[:a.n], [f[:a.n] for f in fb_all]
    disc = proc != 'tele'
    lab = dict(joint=rp.ids(bel.cpu()), **{f'f{k}': rp.ids(fb[k].cpu()) for k in range(K)})
    tag = a.run.replace('runs_fact/', '').replace('/', '_') + ('_untrained' if a.untrained else '')
    out = dict(run=a.run, untrained=a.untrained, proc=proc, t=t, K=K, Ks=Ks,
               n_distinct=dict(joint=int(len(np.unique(lab['joint']))),
                               **{f'f{k}': int(len(np.unique(lab[f'f{k}']))) for k in range(K)}),
               cells={}, interp={}, continuum={})
    print(f"{tag}: proc {proc}, t={t}, distinct beliefs in the {a.n} contexts {out['n_distinct']}", flush=True)
    zeroD = torch.zeros(a.n, cfg.d_model, device=dev)
    lp0 = flp(m, hist, t, L, t, zeroD, conts)                               # clean model futures

    # A. swap groups
    comps = [(p, 'last') for p in range(1, L + 1)] + [(p, 'slice') for p in range(1, L)]
    thr_of = {}
    ii, jj = np.triu_indices(a.n, 1)
    for p, mode in comps:
        pos = t if mode == 'last' else slice(1, t + 1)
        _, A = rp.run(m, hist[:, :t + 1], collect=(p, pos))
        A = A.double()
        d = np.zeros(len(ii))
        bt = 4096
        for s in range(0, len(ii), bt):
            i_, j_ = ii[s:s + bt], jj[s:s + bt]
            for src, dst in ((i_, j_), (j_, i_)):
                lpa = flp(m, hist[src], t, p, pos, (A[dst] - A[src]).float(), conts)
                kl = (lp0[src].exp() * (lp0[src] - lpa)).sum(1).cpu().numpy()
                d[s:s + bt] = np.maximum(d[s:s + bt], kl)
        C, thr, sep = groups(d)
        thr_of[(p, mode)] = thr
        row = dict(n_groups=int(C.max()), threshold=thr, separation=sep, frac_harmful=float((d > thr).mean()),
                   d_q05_50_95=[float(v) for v in np.quantile(d, [0.05, 0.5, 0.95])],
                   H_joint_given_C=nH(lab['joint'], C), H_C_given_joint=nH(C, lab['joint']),
                   **{f'H_f{k}_given_C': nH(lab[f'f{k}'], C) for k in range(K)})
        if disc:
            row['product'] = product_test(C, lab['f0'], lab['f1'])
        else:   # continuous factors: eta^2 of each factor marginal by group
            for k in range(K):
                y = fb[k][:, 1].cpu().numpy()
                mu = np.array([y[C == g].mean() for g in C])
                row[f'eta2_f{k}'] = float(1 - ((y - mu) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        row['labels'] = C.tolist()
        out['cells'][f'p{p}_{mode}'] = row
        extra = (f" | product test: gA {row['product']['n_gA']} x gB {row['product']['n_gB']}, H(C|prod) "
                 f"{row['product']['H_C_given_prod']:.3f} H(prod|C) {row['product']['H_prod_given_C']:.3f}") if disc else \
            ' | eta2 ' + ' '.join(f"f{k} {row[f'eta2_f{k}']:.2f}" for k in range(K))
        print(f"  p{p}_{mode:5s}: {row['n_groups']} groups, sep {sep:.2f}, harmful {row['frac_harmful']:.2f}, "
              f"d q05/50/95 {' '.join(f'{v:.0e}' for v in row['d_q05_50_95'])} | H(joint|C) {row['H_joint_given_C']:.2f} "
              f"H(C|joint) {row['H_C_given_joint']:.2f} | " + ' '.join(f"H(f{k}|C) {row[f'H_f{k}_given_C']:.2f}" for k in range(K))
              + extra, flush=True)

    # B. plateaus along interpolation paths (last position)
    rng = np.random.default_rng(1)
    pairs = rng.integers(0, a.n, (a.npairs, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    na = len(ALPHA)
    for p in range(1, L + 1):
        thr = thr_of[(p, 'last')]
        _, A = rp.run(m, hist[:, :t + 1], collect=(p, t))
        A = A.double()
        i_, j_ = pairs[:, 0], pairs[:, 1]
        rec = np.repeat(i_, na)
        al = torch.tensor(np.tile(ALPHA, len(pairs)), device=dev, dtype=torch.float64)[:, None]
        delta = (al * (A[np.repeat(j_, na)] - A[rec])).float()
        lpa = flp(m, hist[rec], t, p, t, delta, conts).view(len(pairs), na, nc)
        pa_ = lpa.exp()
        tot = (pa_[:, :-1] * (lpa[:, :-1] - lpa[:, 1:])).sum(-1).cpu().numpy()      # step KL
        d_end = (lp0[i_].exp()[:, None] * (lp0[i_][:, None] - lpa)).sum(-1).cpu().numpy()
        fstep, fend, fstar = [], [], []
        for k in range(K):
            q = (pa_ @ Mk[k]).clamp_min(1e-300)                                   # (np, na, 2^Ks)
            lq = q.log()
            fstep.append((q[:, :-1] * (lq[:, :-1] - lq[:, 1:])).sum(-1).cpu().numpy())
            fend.append((q[:, :1] * (lq[:, :1] - lq)).sum(-1)[:, -1].cpu().numpy())
            fstar.append(ALPHA[1:][fstep[-1].argmax(1)])
        fend, fstar = np.stack(fend, 1), np.stack(fstar, 1)
        above = tot > thr
        njump = (above[:, 1:] & ~above[:, :-1]).sum(1) + above[:, 0]
        ndiff = (fend > thr).sum(1)
        res = {}
        for nd in range(K + 1):
            sel = ndiff == nd
            if sel.sum() < 5:
                continue
            r = dict(n_pairs=int(sel.sum()),
                     jumps_hist={int(u): int(v) for u, v in zip(*np.unique(njump[sel], return_counts=True))})
            if nd >= 2:   # spread of the per-factor jump locations among differing factors
                spr = [np.ptp(fstar[s_][fend[s_] > thr]) for s_ in np.where(sel)[0]]
                r['alpha_star_spread_q10_50_90'] = [float(v) for v in np.quantile(spr, [0.1, 0.5, 0.9])]
                # is each factor's own transition sharp? share of its KL change in its largest step
                r['factor_step_share_median'] = float(np.median(np.concatenate(
                    [(fstep[k][sel] .max(1) / np.maximum(fstep[k][sel].sum(1), 1e-12))[fend[sel, k] > thr]
                     for k in range(K)])))
            res[f'{nd}_factors_differ'] = r
        out['interp'][f'p{p}'] = res
        print(f"  interp p{p}: " + ' | '.join(
            f"{k_}: n {v['n_pairs']} jumps {v['jumps_hist']}" + (f" alpha* spread {np.round(v['alpha_star_spread_q10_50_90'], 2)}"
                                                                 if 'alpha_star_spread_q10_50_90' in v else '')
            for k_, v in res.items()), flush=True)

    # C. output continuum
    nn_ = 4000
    zero = torch.zeros(nn_, cfg.d_model, device=dev)
    lg, _ = rp.run(m, toks[:nn_, :t + 1])
    lpn = torch.log_softmax(lg[:, t].double(), -1).cpu().numpy()
    P, Lg = np.exp(lpn), lpn - lpn.mean(1, keepdims=True)
    ex = fp.next_probs(T, bel_all[:nn_]).cpu().numpy()
    cont = dict(model_prob_pcs=n_pcs(P), model_log_pcs=n_pcs(Lg), exact_prob_pcs=n_pcs(ex),
                exact_log_pcs=n_pcs(np.log(np.clip(ex, 1e-12, None)) - np.log(np.clip(ex, 1e-12, None)).mean(1, keepdims=True)))
    r = cont['model_log_pcs'][1]
    Xc = Lg - Lg.mean(0)
    U, S_, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = U[:, :r] * S_[:r]
    try:
        sres = FastICA(n_components=r, whiten='unit-variance', random_state=0, max_iter=2000).fit_transform(Z)
        if disc:
            tgts = [fb_all[k][:nn_, 1:].cpu().numpy() for k in range(K)]
        else:
            tgts = [np.log(fb_all[k][:nn_, 1:].clamp(1e-9, 1).cpu().numpy() / fb_all[k][:nn_, :1].clamp(1e-9, 1).cpu().numpy())
                    for k in range(K)]
        corr = np.array([[max(abs(np.corrcoef(sres[:, c], tg[:, u])[0, 1]) for u in range(tg.shape[1]))
                          for tg in tgts] for c in range(r)])
        cont['ica_corr'] = np.nan_to_num(corr).tolist()
    except Exception as e:  # noqa
        cont['ica_error'] = str(e)
    print(f"  continuum (clean next-token over {nn_} contexts): PCs 99/99.9% prob {cont['model_prob_pcs']} log "
          f"{cont['model_log_pcs']} | exact process prob {cont['exact_prob_pcs']} log {cont['exact_log_pcs']}"
          + (f" | ICA comps: best |corr| with a factor {np.round(np.max(cont['ica_corr'], 1), 2)}, which factor "
             f"{np.argmax(cont['ica_corr'], 1)}" if 'ica_corr' in cont else ''), flush=True)
    # patched outputs per component
    nd_ = 2000
    for p, mode in comps:
        pos = t if mode == 'last' else slice(1, t + 1)
        _, A = rp.run(m, toks[:nd_, :t + 1], collect=(p, pos))
        pp, ll = [], []
        for i0 in range(8):
            rec = torch.full((nd_,), i0, device=dev)
            lg, _ = rp.run(m, toks[rec, :t + 1], inj=(p, pos, (A - A[i0:i0 + 1])))
            l_ = torch.log_softmax(lg[:, t].double(), -1).cpu().numpy()
            pp.append(n_pcs(np.exp(l_))); ll.append(n_pcs(l_ - l_.mean(1, keepdims=True)))
        cont[f'p{p}_{mode}'] = dict(prob_pcs=np.mean(pp, 0).tolist(), log_pcs=np.mean(ll, 0).tolist())
        print(f"    patched p{p}_{mode:5s}: PCs 99/99.9% prob {np.round(np.mean(pp, 0), 1)} log {np.round(np.mean(ll, 0), 1)}",
              flush=True)
    out['continuum'] = cont
    os.makedirs('results/fact', exist_ok=True)
    json.dump(out, open(f'results/fact/swap_{tag}_t{t}.json', 'w'))


if __name__ == '__main__':
    main()
