#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=08:00:00
#SBATCH --job-name=gradflat
#SBATCH --output=logs/gradflat_%j.out
source ~/venvs/torch_env/bin/activate
cd ~/factored_lgssm
python -u grad_flat.py --seed ${SEED:-0} --steps ${STEPS:-300} > logs/gradflat_s${SEED:-0}_st${STEPS:-300}.out 2>&1 || echo FAILED
echo done
