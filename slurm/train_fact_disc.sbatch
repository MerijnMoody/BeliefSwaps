#!/bin/bash
#SBATCH -p gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH -c 9
#SBATCH -t 06:00:00
#SBATCH -o logs/train_fact_disc_%j.out
source ~/venvs/torch_env/bin/activate
cd ~/factored_lgssm
mkdir -p runs_fact/rrxe
for s in 0 1 2; do
  python -u train_fact.py --proc rrxor_even --seed $s --out runs_fact/rrxe/s$s > runs_fact/rrxe/s$s.log 2>&1 &
done
wait
echo all done
