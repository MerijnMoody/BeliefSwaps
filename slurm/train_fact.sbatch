#!/bin/bash
#SBATCH -p gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH -c 18
#SBATCH -t 08:00:00
#SBATCH -o logs/train_fact_%j.out
source ~/venvs/torch_env/bin/activate
cd ~/factored_lgssm
for cfg in "3 0 prod3" "4 0 prod4" "4 0.6 coup4"; do
  set -- $cfg
  mkdir -p runs_fact/$3
  for s in 0 1 2; do
    python -u train_fact.py --K $1 --c $2 --seed $s --out runs_fact/$3/s$s > runs_fact/$3/s$s.log 2>&1 &
  done
done
wait
echo all done
