#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=06:00:00
#SBATCH --job-name=cache_mrr3
#SBATCH --output=logs/cache_mrr3_%j.out
source ~/venvs/torch_env/bin/activate
cd ~/factored_lgssm
for s in 0 1 2 3 4; do python -u cache_swap.py --arch L4_d64 --eps 0.05 --seed $s --ts 9,14,28 --positions last > logs/cache_learned_eps0.05_s$s.out 2>&1 || echo FAILED $s; done
python -u cache_swap.py --arch L4_d64 --eps 0.05 --seed 0 --ts 9,14,28 --positions last --untrained > logs/cache_learned_eps0.05_s0_untrained.out 2>&1 || echo FAILED untrained
python -u mrrxor_cache.py --arch L4_d64 --eps 0.05 > logs/mrrxor_cache_learned.out 2>&1 || echo FAILED analysis
echo done
