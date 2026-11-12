"""Factored processes for the GBS-vs-PSR test.

Each factor is a 2-state "noisy telegraph" HMM: the hidden bit flips with probability q, then emits itself with
probability 1 - e (otherwise its complement). Its belief is one number in [0, 1] and its next-bit prediction is linear
in that number, so a single factor has PSR (linear) dimension 1.

Product of K factors: hidden state s = sum_k s_k 2^k, token x = sum_k x_k 2^k (vocab 2^K). The joint belief is the
tensor product of the factor beliefs, so the set of joint beliefs (and of next-token distributions) spans 2^K - 1
linear dimensions (the PSR / Hankel rank) while being a K-dimensional curved set: a GBS with K coordinates and a
nonlinear (multilinear) decoder suffices.

Coupled control (coupling c > 0): factor k's flip probability depends on factor k-1's current hidden bit
(q_k if s_{k-1} = 0, else q_k + c, cyclic). Emissions stay factored, but the posteriors of the factors become
dependent, so the joint belief is no longer a product of K numbers.

Edge-emitting convention as in mrrxor.py: T[x, s, t] = P(emit x, next state t | state s); the move happens first and
the token is emitted by the new state.
"""
import itertools

import torch

DEFAULT_Q = (0.04, 0.06, 0.08, 0.10, 0.05)
DEFAULT_E = (0.10, 0.15, 0.12, 0.08, 0.10)


def bits(n, K):
    return torch.tensor([[(v >> k) & 1 for k in range(K)] for v in range(n)])


def transition_tensor(K, q=DEFAULT_Q, e=DEFAULT_E, c=0.0):
    S = 2 ** K
    B = bits(S, K)                                     # (S, K)
    move = torch.ones(S, S, dtype=torch.float64)       # P(t | s)
    for k in range(K):
        prev = B[:, (k - 1) % K]                       # s_{k-1} for each source state s
        qk = q[k] + c * prev.double()                  # (S,)
        flip = (B[:, k][:, None] != B[:, k][None, :]).double()   # (s, t)
        move = move * (flip * qk[:, None] + (1 - flip) * (1 - qk[:, None]))
    emit = torch.ones(S, S, dtype=torch.float64)       # P(x | t), rows t, cols x
    for k in range(K):
        same = (B[:, k][:, None] == B[:, k][None, :]).double()
        emit = emit * (same * (1 - e[k]) + (1 - same) * e[k])
    # T[x, s, t] = move[s, t] * emit[t, x]
    return (move[None, :, :] * emit.T[:, None, :]).contiguous()


def stationary(T):
    vals, vecs = torch.linalg.eig(T.sum(0).T)
    pi = vecs[:, torch.argmin((vals - 1).abs())].real
    return pi / pi.sum()


def sample(T, n, length, gen=None, device='cpu'):
    """(n, length) tokens from the stationary process (GPU-friendly, loops over positions)."""
    V, S, _ = T.shape
    joint = T.permute(1, 0, 2).reshape(S, V * S).to(device)          # joint[s, V... ] index x * S + t
    st = torch.multinomial(stationary(T).to(device).expand(n, S), 1, generator=gen)[:, 0]
    out = torch.empty(n, length, dtype=torch.long, device=device)
    for i in range(length):
        o = torch.multinomial(joint[st], 1, generator=gen)[:, 0]
        out[:, i], st = o // S, o % S
    return out


def beliefs(T, toks):
    """(n, length + 1, S) exact joint beliefs, float64, on toks' device."""
    T = T.to(toks.device)
    n, L = toks.shape
    b = stationary(T).to(toks.device).expand(n, -1).clone()
    out = [b]
    for i in range(L):
        u = torch.einsum('ns,nst->nt', b, T[toks[:, i]])
        b = u / u.sum(1, keepdim=True)
        out.append(b)
    return torch.stack(out, 1)


def factor_marginals(bel, K):
    """(..., K) P(s_k = 1 | history) from joint beliefs (..., 2^K)."""
    B = bits(2 ** K, K).to(bel.device).double()
    return bel @ B


def next_probs(T, bel):
    return bel @ T.to(bel.device).sum(2).T


def futures(T, bel, L):
    """(n, V^L) exact probabilities of every L-token continuation (first token most significant)."""
    T = T.to(bel.device)
    V = T.shape[0]
    out = []
    for seq in itertools.product(range(V), repeat=L):
        v = bel
        for x in seq:
            v = v @ T[x]
        out.append(v.sum(-1))
    return torch.stack(out, -1)


# ---- discrete (unifilar) factors, for swap groups / plateaus -------------------------------------------------
def rrxor_tensor():
    """RRXOR (eps = 0), T[x, s, t], states A..E = 0..4."""
    T = torch.zeros(2, 5, 5, dtype=torch.float64)
    T[0, 0, 1] = T[1, 0, 2] = 0.5
    T[0, 1, 4] = T[1, 1, 3] = 0.5
    T[0, 2, 3] = T[1, 2, 4] = 0.5
    T[1, 3, 0] = T[0, 4, 0] = 1.0
    return T


def even_tensor():
    """Even process: from A emit 0 (stay A) or 1 (go to B) with prob 1/2; B emits 1 and returns to A."""
    T = torch.zeros(2, 2, 2, dtype=torch.float64)
    T[0, 0, 0] = T[1, 0, 1] = 0.5
    T[1, 1, 0] = 1.0
    return T


def product_tensor(Ts):
    """Joint T of independent factors: token x = sum_k x_k V^k... here all factors binary, x = sum_k x_k 2^k;
    state s = mixed radix with factor 0 least significant."""
    J = Ts[0]
    for Tk in Ts[1:]:
        V1, S1, _ = J.shape
        V2, S2, _ = Tk.shape
        # new index: x = x1 + V1 * x2, s = s1 + S1 * s2
        J = torch.einsum('aij,bkl->bakilj', J, Tk).reshape(V1 * V2, S1 * S2, S1 * S2)
    return J.contiguous()
