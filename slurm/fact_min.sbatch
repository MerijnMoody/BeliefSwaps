#!/bin/bash
#SBATCH -p gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH -c 18
#SBATCH -t 08:00:00
#SBATCH -o logs/fact_min_%j.out
source ~/venvs/torch_env/bin/activate
cd ~/factored_lgssm
mkdir -p logs/fact
for r in prod3 prod4 coup4; do
  for s in 0 1 2; do
    python -u fact_min.py --run runs_fact/$r/s$s --t 24 > logs/fact/min_${r}_s$s.out 2>&1 &
  done
  python -u fact_min.py --run runs_fact/$r/s0 --t 24 --untrained > logs/fact/min_${r}_s0_untrained.out 2>&1 &
done
for s in 0 1 2; do
  python -u fact_min.py --run runs_fact/rrxe/s$s --t 12 > logs/fact/min_rrxe_s$s.out 2>&1 &
done
python -u fact_min.py --run runs_fact/rrxe/s0 --t 12 --untrained > logs/fact/min_rrxe_s0_untrained.out 2>&1 &
wait
echo all done
