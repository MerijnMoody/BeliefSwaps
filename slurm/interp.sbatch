#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --job-name=interp
#SBATCH --output=logs/interp_%j.out
source ~/venvs/torch_env/bin/activate
cd ~/factored_lgssm
for s in 0 1 2; do python -u interp_rrxor.py --seed $s > logs/interp_s$s.out 2>&1 || echo FAILED $s; done
echo done
