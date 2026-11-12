#!/bin/bash
#SBATCH -p gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH -c 9
#SBATCH -t 04:00:00
#SBATCH -o logs/fact_ana_disc_%j.out
source ~/venvs/torch_env/bin/activate
cd ~/factored_lgssm
mkdir -p logs/fact
for t in 6 12; do
  for s in 0 1 2; do
    python -u fact_swap.py --run runs_fact/rrxe/s$s --t $t > logs/fact/swap_rrxe_s${s}_t$t.out 2>&1 &
  done
  python -u fact_swap.py --run runs_fact/rrxe/s0 --t $t --untrained > logs/fact/swap_rrxe_s0_untrained_t$t.out 2>&1 &
done
wait
echo all done
