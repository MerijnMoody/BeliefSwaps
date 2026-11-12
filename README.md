# Swap partitions and generalized belief states in small transformers

Activation swaps ("put context j's activation into context i's forward pass and see whether the predicted future
changes") used to read out, label-free, what each layer of a small transformer represents. Applied to the belief-state
transformers of [artemyk/belief-state](https://github.com/artemyk/belief-state) (RRXOR, MRRXOR) and to transformers
with the same architecture trained on factored processes, where a nonlinear low-dimensional state (a generalized belief
state, GBS) is much smaller than the linear predictive state (PSR / Hankel rank).

**Start with [WRITEUP.md](WRITEUP.md)** for the methods, results and caveats.

Status: the code was written with an AI coding assistant (Claude Code) and has been only partly reviewed by hand. The
numbers in the writeup come from the result files in `results/`, which are included.

## Setup

```bash
git clone --recursive https://github.com/MerijnMoody/BeliefSwaps.git
cd BeliefSwaps
pip install -r requirements.txt
```

`external/belief-state` is a git submodule pinned to commit 61c2a55 of Artemy Kolchinsky's repository (MIT license).
It provides the model code (`model.py`) and the trained RRXOR / MRRXOR checkpoints (`runs/<arch>/eps<eps>/s<seed>.pt`)
that most scripts load. All experiments need a CUDA GPU. Run every script from the repository root: scripts write to
`results/` and read `runs_fact/` with relative paths.

## Layout

| Path | What |
|---|---|
| `rrxor_plateau.py` | Core swap machinery (forward pass with an injected residual, exact K-step futures, swap distance, Otsu grouping, scoring against labels). Per-layer swap groups on the RRXOR / MRRXOR checkpoints. Other scripts import it. |
| `agg_rrxor.py` | Aggregates `rrxor_plateau.py` results over seeds and architectures. |
| `hh.py`, `hh_explore.py`, `hh_pool.py` | Head-to-head of swap groups vs PCA + k-means vs linear probes, including a planted unused variable; pooled history lengths. |
| `interp_rrxor.py` | Straight-line interpolation between two contexts' activations: plateaus and jumps. |
| `grad_flat.py` | Search for flat (output-insensitive) directions around each context. |
| `cache_swap.py` | Swaps of other parts of the state: one vector at an interior position q, or a whole layer at all history positions ("slice"). |
| `mrrxor_groups.py`, `mrrxor_cache.py` | The same analyses on MRRXOR (eps 0.05), scored by belief variance explained. |
| `fut_svd.py` | SVD of patched futures (rows: donors, columns: continuations), a linear read-out for continuous beliefs. |
| `fact_proc.py` | Factored processes: K noisy-telegraph factors, coupled control, RRXOR x Even. Exact beliefs, sampling, futures. |
| `fact_params.py` | Ground-truth process quantities (PSR rank, excess entropy, statistical complexity). |
| `train_fact.py` | Trains Artemy's architecture, with his hyperparameters, on a factored process. |
| `fact_gbs.py` | GBS bottleneck (affine encoder, linear / softmax / MLP decoders), probes, ICA factor coordinates and their swaps. |
| `fact_swap.py` | Swap groups, product-partition test, interpolation jumps per factor, output continuum (probability vs log space). |
| `fact_min.py` | Minimality quantities: R^2 of the residual from the belief, additive vs interaction share, PCs, conditional MI about past tokens (torch_mist). |
| `fact_summary.py` | Seed-averaged printout of all `results/fact` files. |
| `runs_fact/` | The 12 trained factored-process models (prod3, prod4, coup4, rrxe; 3 seeds each) with training logs. |
| `results/plateau/`, `results/fact/` | Result JSON files behind every number in the writeup. |
| `slurm/` | The SLURM batch scripts used for the runs (cluster-specific paths, see `slurm/README.md`). |

## Reproducing

RRXOR / MRRXOR on the shared checkpoints (archs `L4_d64`, `L4_d64_rope`, `L2_d32`, `L2_d32_rope`; seeds 0-4):

```bash
python rrxor_plateau.py --arch L4_d64 --eps 0 --seed 0 [--untrained] [--swaps_only]
python agg_rrxor.py
python hh.py --arch L4_d64 --seed 0;  python hh_pool.py --arch L4_d64 --seed 0
python interp_rrxor.py --seed 0;      python grad_flat.py --seed 0
python cache_swap.py --arch L4_d64 --eps 0 --seed 0 [--positions last]
python mrrxor_groups.py --eps 0.05;   python mrrxor_cache.py --arch L4_d64_rope --eps 0.05
python fut_svd.py --arch L4_d64_rope --eps 0.05 --seed 0 [--untrained] [--save]
```

Factored processes (training took 18-56 minutes per model on an A100, with several models sharing one GPU; the trained
models are included):

```bash
python train_fact.py --K 4 --c 0   --seed 0 --out runs_fact/prod4/s0      # prod3: --K 3; coup4: --K 4 --c 0.6
python train_fact.py --proc rrxor_even --seed 0 --out runs_fact/rrxe/s0
python fact_gbs.py  --run runs_fact/prod4/s0 [--untrained]
python fact_swap.py --run runs_fact/prod4/s0 --t 12     # rrxe: --t 6 and --t 12
python fact_min.py  --run runs_fact/prod4/s0 --t 24     # rrxe: --t 12
python fact_summary.py
```

## Credits

Model code, architecture (Shai et al. 2024) and the RRXOR / MRRXOR checkpoints: Artemy Kolchinsky,
[artemyk/belief-state](https://github.com/artemyk/belief-state), MIT license.
