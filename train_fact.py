"""Train Artemy's belief-state transformer (external/belief-state/model.py, same hyperparameters as its train.py)
on a factored telegraph process (fact_proc.py). Vocab 2^K. Sampling on the GPU.

    python train_fact.py --K 4 --c 0 --seed 0 --out runs_fact/prod4/s0
"""
import argparse
import importlib.util
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import fact_proc as fp

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'external', 'belief-state')
_spec = importlib.util.spec_from_file_location('bs_model', os.path.join(REPO, 'model.py'))
bs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bs)

pa = argparse.ArgumentParser()
pa.add_argument('--K', type=int, default=4)
pa.add_argument('--c', type=float, default=0.0)
pa.add_argument('--proc', default='tele', choices=['tele', 'rrxor_even'])
pa.add_argument('--seed', type=int, default=0)
pa.add_argument('--layers', type=int, default=4)
pa.add_argument('--d-model', type=int, default=64)
pa.add_argument('--rope', action='store_true')
pa.add_argument('--init-std', type=float, default=0.1)
pa.add_argument('--context', type=int, default=32)
pa.add_argument('--batch', type=int, default=512)
pa.add_argument('--lr', type=float, default=1e-3)
pa.add_argument('--warmup', type=int, default=1000)
pa.add_argument('--steps', type=int, default=100_000)
pa.add_argument('--log-every', type=int, default=2000)
pa.add_argument('--ckpt-every', type=int, default=25000)
pa.add_argument('--out', type=Path, required=True)
args = pa.parse_args()

dev = 'cuda'
T = fp.transition_tensor(args.K, c=args.c) if args.proc == 'tele' else fp.product_tensor([fp.rrxor_tensor(), fp.even_tensor()])
if args.proc == 'rrxor_even':
    args.K = 2
gen = torch.Generator(device=dev).manual_seed(args.seed)
cfg = bs.Config(vocab=2 ** args.K, context=args.context, n_layers=args.layers, d_model=args.d_model,
                d_mlp=4 * args.d_model, rope=args.rope)
model = bs.untrained(cfg, args.init_std, args.seed).to(dev)
forward = torch.compile(model, mode='reduce-overhead')
opt = torch.optim.Adam(model.parameters(), lr=args.lr, fused=True)
sched = torch.optim.lr_scheduler.SequentialLR(opt, [
    torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-3, total_iters=args.warmup),
    torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.steps - args.warmup), eta_min=args.lr / 100),
], milestones=[args.warmup])

test_toks = fp.sample(T, 4096, args.context, torch.Generator(device=dev).manual_seed(12345), dev)
test_opt = fp.next_probs(T, fp.beliefs(T, test_toks)[:, :-1])
ent = float(-(test_opt * test_opt.clamp_min(1e-300).log()).sum(-1).mean())


def test_kl():
    with torch.no_grad():
        lq = torch.log_softmax(model(test_toks).double(), -1)
        return float((test_opt * (test_opt.clamp_min(1e-300).log() - lq)).sum(-1).mean())


args.out.mkdir(parents=True, exist_ok=True)


def save(path, step, **kw):
    torch.save({'config': vars(cfg), 'args': {k: str(v) for k, v in vars(args).items()}, 'step': step,
                'model': {k: v.cpu() for k, v in model.state_dict().items()}, **kw}, path)


print(f'K={args.K} c={args.c} seed={args.seed}: optimal next-token entropy {ent:.4f} nats', flush=True)
t0 = time.time()
BUF = 64  # draw 64 batches at once: the sampler loops over positions, so this amortizes kernel launches
for step in range(1, args.steps + 1):
    if (step - 1) % BUF == 0:
        buf = fp.sample(T, BUF * args.batch, args.context, gen, dev).view(BUF, args.batch, args.context)
    toks = buf[(step - 1) % BUF]
    loss = F.cross_entropy(forward(toks).flatten(0, 1), toks.flatten())
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    sched.step()
    if step % args.log_every == 0:
        print(f'step {step:7d} loss {loss.item():.4f} test KL {test_kl():.2e} ({step / (time.time() - t0):.0f} steps/s)',
              flush=True)
    if step % args.ckpt_every == 0:
        save(args.out / f'step_{step:06d}.pt', step)
kl = test_kl()
save(Path(f'{args.out}.pt'), args.steps, test_kl=kl, entropy=ent, seconds=time.time() - t0)
print(f'done: test KL {kl:.2e} (entropy {ent:.4f})', flush=True)
