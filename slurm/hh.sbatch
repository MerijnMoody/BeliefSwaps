#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --job-name=flg_hh
#SBATCH --output=logs/hh_%j.out
source ~/venvs/torch_env/bin/activate
cd ~/factored_lgssm
python -W ignore hh.py --arch L4_d64 --seed 0 --n 200 > logs/hh_L4_d64_s0.out 2>&1
echo ALLDONE
