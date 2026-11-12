#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=05:00:00
#SBATCH --job-name=flg_hhpool
#SBATCH --output=logs/hhpool_%j.out
source ~/venvs/torch_env/bin/activate
cd ~/factored_lgssm
for arch in L4_d64 L4_d64_rope L2_d32 L2_d32_rope; do
  for s in 0 1 2 3 4; do
    echo "$(date +%T) $arch s$s"
    python hh_pool.py --arch $arch --seed $s > logs/hhpool_${arch}_s${s}.out 2>&1 || echo "FAILED $arch s$s"
  done
done
echo done
