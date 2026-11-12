#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=06:00:00
#SBATCH --job-name=futsvd
#SBATCH --output=logs/futsvd_%j.out
source ~/venvs/torch_env/bin/activate
cd ~/factored_lgssm
python -u fut_svd.py --arch L4_d64_rope --eps 0.05 --seed 0 --save > logs/futsvd_rope_s0.out 2>&1 || echo FAILED s0
for s in 1 2 3 4; do python -u fut_svd.py --arch L4_d64_rope --eps 0.05 --seed $s > logs/futsvd_rope_s$s.out 2>&1 || echo FAILED $s; done
python -u fut_svd.py --arch L4_d64_rope --eps 0.05 --seed 0 --untrained > logs/futsvd_rope_s0_untrained.out 2>&1 || echo FAILED untrained
echo done
