# Swap partitions and generalized belief states in small transformers

Merijn Moody, November 2026. Working notes, not a paper. The code was written with an AI coding assistant and has been
only partly reviewed by hand; every number below is read from the result files in `results/`.

## Summary

- **Method.** Put context j's residual vector (layer p, last history position) into context i's forward pass and
  measure how much the predicted distribution over the next few tokens changes (KL). Contexts whose swaps change
  nothing form groups. Labels are used only
  afterwards, to say what the groups are.
- **RRXOR (discrete beliefs).** On the belief-state transformers of artemyk/belief-state, the groups read out a
  per-layer algorithm without labels: last two tokens (layer 1), their XOR (layer 2), close to the belief state
  (layer 3). The pattern holds in 5/5 seeds. Untrained networks give groups that track none of these variables.
- **Plateaus.** Between contexts in the same group, the predicted future stays flat along the straight line between
  their activations. Between groups there is mostly one sharp jump. Untrained networks show no jumps.
- **Continuous beliefs (MRRXOR eps 0.05, noisy telegraph factors).** Groups and jumps largely disappear: the
  activations form a continuum. There the right object is a set of coordinates, not a partition.
- **Beyond the linear (PSR / Hankel) state.** For K independent noisy factors the linear predictive state needs
  2^K - 1 dimensions, but K numbers determine everything. Transformers trained on these processes store K per-factor
  coordinates. An affine bottleneck with a nonlinear decoder needs K (or K + 1) dimensions; a decoder linear in
  probabilities is still poor at 2^K - 1. ICA on the bottleneck gives one coordinate per factor, and swapping one
  coordinate changes only that factor's prediction.
- **Minimality.** About 99% of the residual variance is a function of the belief, additive over factors, in about 6
  dimensions. By information the representation is not minimal: 0.5 to 0.9 nats about past tokens beyond the belief
  remain, in low-variance directions.

## 1. Models and processes

**Shared checkpoints.** [artemyk/belief-state](https://github.com/artemyk/belief-state) at commit 61c2a55 (included as a
submodule): the Shai et al. (2024) architecture (pre-LayerNorm, one attention head of dimension 8, ReLU MLP of width
256, BOS token seen only by position 0), four architectures (L4_d64, L4_d64_rope, L2_d32, L2_d32_rope), 5 seeds each,
trained on RRXOR (eps 0) and MRRXOR (eps 0.05). Our forward pass reproduces his (next-token KL to the Bayes-optimal
predictor 9.6e-7 for L4_d64 seed 0).

**Our factored processes** (`fact_proc.py`), trained with the same architecture (L4_d64, learned positions) and his
training hyperparameters, 100k steps, 3 seeds each (`train_fact.py`):

| Name | Process | Vocab | Test KL to Bayes |
|---|---|---|---|
| prod3, prod4 | K = 3 or 4 independent noisy-telegraph factors. Each factor is a hidden bit that flips with probability q_k and is emitted with error e_k (q = 0.04, 0.06, 0.08, 0.10; e = 0.10, 0.15, 0.12, 0.08). The token is the K emitted bits. | 2^K | 3e-5, 5e-5 |
| coup4 | As prod4, but factor k's flip probability is q_k + 0.6 when factor k-1's bit is 1 (cyclic). The factors' posteriors become dependent. | 16 | 2.7e-4 |
| rrxe | RRXOR x Even process, two independent discrete factors. Token = (RRXOR bit, Even bit). | 4 | 5e-5 |

For prod4 the linear predictive state (PSR, equal to the rank of the Hankel matrix) has 15 dimensions, while the 4
factor beliefs determine everything; an affine map from the 4 factor beliefs explains only 62% of the joint
next-token distribution. Telegraph beliefs are continuous (every history gives a different belief). RRXOR and rrxe
beliefs are discrete: RRXOR at eps 0 has 36 reachable beliefs including transients, and rrxe has 26 distinct joint
beliefs at history length 12.

## 2. The swap test

**Distance.** For contexts i and j at history length t, replace i's residual at layer p (the vector entering block p;
p = 0 is the embedding, p = L the final residual) at position t with j's, and compute the model's probabilities of all
continuations of length K by teacher forcing. The distance is the largest of KL(reference future of i || patched
future) over the two swap directions. RRXOR / MRRXOR scripts: K = 4, reference = the exact future. Factored-process
scripts (`fact_swap.py`): reference = the model's own clean future, K chosen so that there are about 256
continuations (2 tokens for prod4, 3 for prod3, 4 for rrxe). (The RRXOR pilot also included the
halfway point between the two activations; rebuilding the partitions from swaps only gave the same groups, median ARI
1.00, minimum 0.69.)

**Groups.** Complete linkage on log10 distance, cut at an automatic (Otsu) threshold. No labels are involved.

**Scoring.** Afterwards, the groups C are compared with candidate variables V (last token, last two tokens, their XOR,
phase, belief, next-token class) by normalized conditional entropies: H(V|C)/H(V) = 0 means the groups determine V,
H(C|V)/H(C) = 0 means the groups are not finer than V. Shuffled groupings with the same sizes give 0.8 to 1.0.

**What a swap actually tests.** The network computes a grid of residual vectors x^n_l (layer l, position n). Block l
at position n reads only the layer-l residuals at positions <= n (through attention) and its own residual; nothing
else inside a block (queries, keys, values, MLP hidden units) is reused later. After replacing x^t_p, the cells that
change form a cone: position t at layers above p, and the later (continuation) positions from layer p + 1 up.
Everything else stays i's: all earlier history positions at every layer, position t below p, and the continuation
positions below p+1. So i's history enters the patched computation twice: through attention at position t (layers
>= p read i's history residuals), and through the continuation positions, whose lower layers are built from i's
history. A swap is harmless when j's vector, read together with i's history, gives the same future. It measures what
the vector contributes given the rest of i's computation, not what it holds in isolation.

**Method limit at the final layer.** The final residual at position t feeds only the next-token output, so swapping
it can only change the next-token prediction. Final-layer groups are next-token classes by construction.

## 3. RRXOR (eps 0): a per-layer algorithm

200 contexts at history lengths 4, 6, 9, 14 and 28, K = 4 (all 36 beliefs have pairwise future KL >= 34).

| Layer | L4_d64, 5 seeds (median over history lengths) | Reading |
|---|---|---|
| 0 | 2 groups = the current token | input |
| 1 | about 5.6 groups = exactly the last two tokens (H(last2\|C) 0.00 in every seed) | collect recent tokens |
| 2 | 9 to 16 groups; XOR of the last two tokens kept (0.07), the raw pair partly dropped (0.23, range 0.01 to 0.46); some phase information | compute the XOR |
| 3 | about 13 groups, closest to the belief (H(belief\|C) 0.22, H(C\|belief) 0.20) | phase, then belief |
| 4 (final) | next-token classes (method limit above) | readout |

- **Architectures.** L4_d64_rope shows the same order with finer groups (layer 3: belief 0.09 with 35 groups). L2_d32
  compresses it (layer 1: last two tokens / XOR, then the output). L2_d32_rope (worst fit, test KL 6e-3) has 47 groups
  at layer 1 that determine every candidate variable.
- **Untrained controls** (all architectures): 9 to 14 groups that track no candidate variable (belief 0.60 to 0.73,
  XOR 0.77 to 0.87), flat across layers; swap distances about 1000x smaller.
- **Interior positions** (`cache_swap.py`, one vector at position q < t, future still after t): by layer 2 the groups
  at q determine the XOR check at q (is token q-1 the XOR of the two before it) in most positions and all seeds; layer
  3 carries partial phase and belief. Read: each position computes its own check bit, and the last position combines
  them into phase and belief. The gathering step is inferred, not traced to specific heads.
- **Whole layer** (all history positions at layer p swapped together): groups equal the belief states at layers 1 to 3
  (H(belief|C) 0.03 to 0.08, H(C|belief) 0.00 to 0.21), against 0.15 to 0.72 for the single last-position vector. The
  merges of the single-vector test come from information held at other positions of the same layer.

This order (recent tokens, XOR, then phase gating into the belief) is consistent with the "XOR circuit, then phase
gating" description from the Shai group.

**Compared with PCA and probes** (`hh.py`, `hh_pool.py`; held-out causal error = mean of the failure rates "same group
but harmful swap" and "different group but harmless swap", threshold 1e-2):
- Linear probes decode the last two tokens at 0.98 to 1.00 at every layer, yet grouping by them is the worst grouping
  at layers 3 and 4 (error 0.44 to 0.48). Decodable is not the same as used.
- With an unused variable planted along output-insensitive directions, PCA + k-means groups follow the plant (ARI 0.19
  to 0.38 for small plants, 0.5 to 1.0 for large ones) and lose the belief; swap groups ignore it (ARI <= 0.08),
  except in one case where the plant does change the future through LayerNorm.
- Swap groups clearly beat PCA at the final layer in every architecture (error 0.00 to 0.05 vs 0.07 to 0.57). In the natural
  middle layers PCA is about as good, and better on held-out contexts when history lengths are pooled, because the
  variables used there also carry much of the variance (including absolute position, which these models do use).

## 4. Plateaus along interpolations

`interp_rrxor.py`, RRXOR L4_d64, 3 seeds, layers 1 to 4, t = 6 and 14. Move i's activation in a straight line toward
j's and follow the predicted future.

- Same-group paths: flat in 99 to 100% of pairs.
- Harmful between-group paths: mostly one jump (65 to 90%), sharpness 5 to 16 at layers 1 to 3 (1 = straight line), against about 2 in the untrained
  network. The jump sits at alpha 0.6 to 0.7.
- Final layer: no plateau (sharpness 2 to 5).
- Directions within a group are flatter than random directions of equal length; directions between groups are more
  sensitive than random. Caveat: 3 to 51% of different-group pairs are harmless in both directions (complete linkage;
  harmlessness is not fully transitive, worse at t = 14).
- Flat-direction search (`grad_flat.py`): flatness in the middle layers is local (each region has its own 1 or 2 flat
  directions); a shared ignored subspace exists in the untrained network but not after training.

**rrxe** (two discrete factors, final layer): when no factor's belief differs, 0 jumps; when one differs, 1 jump (77%);
when both differ, 2 jumps in 48% of paths, at separate points (median spread of the jump positions 0.57). Each jump
changes one factor's predicted future, so the middle plateau is a valid combination (one factor from j, one from i).
Untrained: no jumps.

**Softmax saturation as a confound.** A straight line in logits between two confident predictions already gives one
sharp step in probabilities. Three observations argue against this explaining the jumps: the untrained network shows
none, continuous-belief processes show none, and the final layer, which has only LayerNorm, unembedding and softmax
after it, shows the weakest plateaus. The clean test (measuring the path in centered log-probabilities, and in L2 at
the next layer) has not been run.

## 5. Continuous beliefs

**MRRXOR (eps 0.05, Artemy's setting, 5 seeds).** Groups separate less cleanly (Otsu separation 0.45 to 0.65 against
clear gaps at eps 0). Single last-position groups are mostly next-token classes plus part of the belief (rope: final
layer 13 groups explain 44% of belief variance and 99% of next-token probability; layer 3, 69 to 72% of belief). Swapping
the whole layer recovers much more at early layers: belief variance explained at layer 1 goes from 0.37 to 0.50 (one
vector) to 0.82 to 0.97 (whole layer), so in these models the belief is spread over positions early on.

**Patched-future SVD** (`fut_svd.py`, rope, t = 14 and 28). For each recipient, the matrix of patched 4-step futures
over 200 donors; label-free coordinates are its top singular directions. The model's own clean futures span 4
dimensions and give the belief exactly (R^2 1.00). Patched futures do worse: belief R^2 with 4 coordinates is 0.22 to
0.42 for the last-position vector and 0.52 to 0.64 for the whole layer. The R^2 of a linear fit of the patched future
on the donor's own clean future (t = 14, mean over 5 seeds):

| Layer | single last vector | whole layer |
|---|---|---|
| 1 | 0.40 | 0.76 |
| 2 | 0.58 | 0.85 |
| 3 | 0.85 | 0.93 |
| final | 1.00 | 1.00 |

The missing part is i's history entering through the two routes in section 2: patched futures are not linear in the
donor's belief. A nonlinear read-out has not been tried. (Untrained networks score 0.94 to 1.00 here, but their
patched futures barely change, so this R^2 is not meaningful for them.)

**Telegraph factors.** Swap groups form a continuum (separation 0.57 to 0.63; about 25 to 41 groups, each mixing many
beliefs) and interpolations show no jumps. The coordinates in section 6 are what work here.

## 6. Generalized belief states beyond the PSR

**Bottleneck** (`fact_gbs.py`). Input: the residual at the last history position (one layer, 64 numbers, or all four
layers concatenated), t = 24, 20k training and 4k test contexts. An affine encoder to d numbers, then a decoder to the
exact 2-step future distribution (V^2 entries), trained by KL. Three decoders: linear in probabilities (closed-form
reduced-rank regression; this is the PSR / Hankel form), affine to logits plus softmax, and an MLP (2 hidden layers of
256). GBS dimension = smallest d at which the MLP's KL is within 2x of its minimum. Mean over 3 seeds:

| | prod3 | prod4 | coup4 |
|---|---|---|---|
| PSR rank / number of factors | 7 / 3 | 15 / 4 | 15 / 4 |
| MLP dimension, layers 1-4 | 3, 4, 4, 4 | 4, 5, 5, 5 | 4 to 6, at lossy KL |
| KL at d = K: MLP / softmax / linear (final layer) | 3e-4 / 1e-2 / 0.20 | 4e-4 / 1e-2 / 0.27 | 2e-3 / 2e-2 / 0.13 |
| linear decoder at d = 2^K - 1 (layers 1-4) | 0.14 to 0.16 | 0.24 to 0.27 | 0.04 |

With the exact belief as input, the linear decoder reaches zero at exactly d = 2^K - 1, as it must. From the
network's residual it never gets there: the residual holds the K factor beliefs in a form that is not linear in the
joint belief. Probes agree: the joint belief is decodable at R^2 0.62 to 0.70 (prod4) and 0.89 to 0.91 (prod3) per
layer, the factor marginals at 0.99 to 1.00.

**Label-free factors.** FastICA on the d = K encoder outputs. To swap one ICA coordinate from j into i, move i's
residual along that coordinate's regression ("pattern") direction by the coordinate difference, and measure the change
in the model's next-bit log-odds of each factor:

| | prod3 / prod4 | random rotation of the same space | coup4, untrained |
|---|---|---|---|
| specificity (share of the change on one factor) | 0.95 to 0.97 | 0.44 to 0.62 | random level |
| correlation of the change with j minus i on the matched factor | 0.99 to 1.00 | 0.72 to 0.84 | random level |
| R^2 of all K coordinates together vs swapping the whole vector | 0.98 to 0.99 vs 0.99 to 1.00 | | |

Minimal-norm (pseudo-inverse) directions transfer only 0.06 to 0.33, because the network keeps redundant copies of
each coordinate; the regression direction moves all copies.

**Output geometry.** The model's clean next-token distributions need 2^K - 1 principal components in probability space
but exactly K in log space (prod3: 7 vs 3; prod4: 15 vs 4), and ICA on the log components gives one component per
factor (|corr| 0.99). This follows from log P(token) being a sum of per-factor terms, and reads off the GBS dimension
from outputs alone. coup4: 14 vs 11, no factors. Untrained: the same count in both spaces.

## 7. Minimality

`fact_min.py`, last position, t = 24 (telegraph) or 12 (rrxe), 3 seeds. prod4, final layer:

| | trained | untrained |
|---|---|---|
| R^2 of the residual from the exact belief (MLP) | 0.988 | 0.59 |
| interaction share: (full - additive per-factor) / full | 0.001 | 0.70 |
| R^2 with the last 3 tokens added | 0.996 | 0.70 |
| PCs for 95% of variance (of the belief-explained part) | 5.7 (5.3) | 22 (13) |
| I(R; token 2 / 4 / 8 back \| belief), nats | 0.81 / 0.93 / 0.53 | about 0 / 0.07 / 0.24 |

prod3 is the same picture (R^2 0.97, interaction share 0.001, 6 PCs, 0.55 / 0.80 / 0.52 nats). The conditional mutual
information uses torch_mist's SMILE estimator by the chain rule; the same estimator with the residual replaced by the
belief itself, where the true value is 0, gives at most 0.12 nats. The most recent token carries 0.23 to 0.29 nats at
every layer, including the embedding, because the residual stream always contains the input token.

Reading: the residual is almost entirely a function of the belief, factored (additive over factors, not the joint
form) and low-dimensional (about 6 PCs, against K = 4 and 15 for the PSR). It is not minimal by information: a clear
amount of past-token information beyond the belief sits in the remaining 1% of the variance. Whether the network uses
it has not been tested; it cannot improve predictions, since the belief is sufficient.

coup4 has an interaction share of 0.46 to 0.58, but this is a weak test there: with coupled factors the marginals do
not determine the joint belief, so an additive model on them fails even for a perfect network.

**rrxe per layer** (eta^2 = share of residual variance explained by the exact distinct belief):

| Layer | 1 | 2 | 3 | 4 (final) |
|---|---|---|---|---|
| eta^2 | 0.25 | 0.23 | 0.41 | 0.92 |
| PCs for 95% | 6 | 18 | 26 | 3 |
| I(R; token 2 back \| belief), nats | 1.22 | 1.22 | 1.21 | 1.12 |

The middle layers are a working space holding raw token history; the final layer compresses to the belief, keeping
some past-token information in the low-variance remainder. rrxe swap groups: final-layer groups are exactly a product
of an RRXOR partition and an Even partition (3/3 seeds, t = 6 and 12); middle layers are not a product.

## 8. Caveats and open questions

- **Single-vector swaps measure a contribution, not a state.** i's history re-enters (section 2), so information also
  stored elsewhere looks dropped. The whole-layer swap is closer to "the state at layer p", but still keeps i's lower
  layers. A cleaner version would also replace the lower-layer residuals at the swapped positions, close to swapping
  the whole KV cache as in Artemy's GBS construction.
- **The GBS here uses the last-position residual, not the whole KV cache.** That suffices for the telegraph factors
  (the whole-vector swap transfers the full prediction) but not for RRXOR / MRRXOR, where the belief is spread over
  positions.
- **What harmful swaps produce is untested for RRXOR.** For each harmful swap, the patched future could be j's, another
  valid belief's (for example j's phase with i's last tokens), or none. With 36 exact beliefs this is a cheap test and
  would show the rule by which the network combines the two contexts.
- **Metric.** The swap distance is KL on exact futures (locally the Fisher-Rao metric). It is coordinate-free and splits
  over independent factors, but the grouping step assumes discrete cells. For continuous beliefs an embedding of the
  distance matrix (MDS / Isomap) would be the natural replacement; for products of independent factors it should give
  a flat K-dimensional box. Not run.
- **Seeds.** 5 seeds for the RRXOR / MRRXOR checkpoints, 3 for the factored processes, one architecture for the latter.

## 9. Related work

- Predictive state representations (Littman, Sutton and Singh 2002) and spectral learning of weighted automata (Balle,
  Carreras, Luque and Quattoni, Machine Learning 2014): the minimal linear state equals the Hankel rank and is unique up
  to an invertible linear map. Our patched-future matrices are Hankel matrices with interventions on one layer in place
  of string prefixes; section 6 shows a nonlinear state that is much smaller than the linear one.
- Shai et al. 2024 (belief-state geometry in the residual stream, linear probes); Shai et al. 2026, "Transformers learn
  factored representations" (arXiv 2602.02385): factored processes, linear regression and PCA, no nonlinear decoders or
  interventions.
- Causal partitions (Chalupka, Perona and Eberhardt, arXiv 1412.2309, 1512.07942): grouping micro-states by their
  interventional effect. Our groups are this partition with activations as micro-states and swaps as interventions.
- Vafa et al. 2024 (arXiv 2406.03689): Myhill-Nerode tests of implicit world models from model outputs only.
- Causal abstraction and interchange interventions (Geiger et al.): test a hypothesis supplied by the researcher; here
  the partition is found without one.
- Grant, Han, Tartaglini and Potts 2025 (arXiv 2511.04638): interventions can push activations off-distribution.
- Artemy Kolchinsky, "Generalized belief states in transformers" (LessWrong): the GBS construction (affine encoder of
  the whole KV cache, nonlinear decoder) that section 6 adapts.
