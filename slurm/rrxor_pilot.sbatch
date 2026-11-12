#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --job-name=flg_rrxor
#SBATCH --output=logs/rrxor_pilot_%j.out
source ~/venvs/torch_env/bin/activate
cd ~/factored_lgssm
python rrxor_plateau.py --arch L4_d64 --eps 0 --seed 0 > logs/rrxor_L4_d64_eps0_s0.out 2>&1
python rrxor_plateau.py --arch L4_d64 --eps 0 --seed 0 --untrained > logs/rrxor_L4_d64_eps0_s0_untrained.out 2>&1
