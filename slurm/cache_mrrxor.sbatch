#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=06:00:00
#SBATCH --job-name=cache_mrr
#SBATCH --output=logs/cache_mrrxor_%j.out
source ~/venvs/torch_env/bin/activate
cd ~/factored_lgssm
# 1. MRRXOR swap groups (his setting: rope, eps 0.05) + learned-position comparison + untrained
for arch in L4_d64_rope L4_d64; do for s in 0 1 2 3 4; do
  python -u rrxor_plateau.py --arch $arch --eps 0.05 --seed $s --swaps_only > logs/mrrxor_${arch}_s$s.out 2>&1 || echo FAILED $arch $s
done; python -u rrxor_plateau.py --arch $arch --eps 0.05 --seed 0 --swaps_only --untrained > logs/mrrxor_${arch}_s0_untrained.out 2>&1 || echo FAILED untrained $arch; done
python -u mrrxor_groups.py --eps 0.05 > logs/mrrxor_groups.out 2>&1 || echo FAILED groups
# 2+3. interior positions and layer slices, RRXOR L4_d64, 3 seeds
for s in 0 1 2; do python -u cache_swap.py --arch L4_d64 --eps 0 --seed $s > logs/cache_L4_d64_s$s.out 2>&1 || echo FAILED cache $s; done
echo done
