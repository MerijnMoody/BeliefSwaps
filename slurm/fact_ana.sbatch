#!/bin/bash
#SBATCH -p gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH -c 18
#SBATCH -t 06:00:00
#SBATCH -o logs/fact_ana_%j.out
source ~/venvs/torch_env/bin/activate
cd ~/factored_lgssm
mkdir -p logs/fact
for r in prod3 prod4 coup4; do
  for s in 0 1 2; do
    python -u fact_gbs.py --run runs_fact/$r/s$s > logs/fact/gbs_${r}_s$s.out 2>&1 &
  done
  python -u fact_gbs.py --run runs_fact/$r/s0 --untrained > logs/fact/gbs_${r}_s0_untrained.out 2>&1 &
done
wait
for r in prod3 prod4 coup4; do
  for s in 0 1 2; do
    python -u fact_swap.py --run runs_fact/$r/s$s --t 12 > logs/fact/swap_${r}_s${s}_t12.out 2>&1 &
  done
  python -u fact_swap.py --run runs_fact/$r/s0 --t 12 --untrained > logs/fact/swap_${r}_s0_untrained_t12.out 2>&1 &
done
wait
echo all done
