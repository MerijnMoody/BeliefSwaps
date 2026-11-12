#!/bin/bash
#SBATCH -p gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH -c 9
#SBATCH -t 03:00:00
#SBATCH -o logs/fact_gbs_redo_%j.out
source ~/venvs/torch_env/bin/activate
cd ~/factored_lgssm
python -u fact_gbs.py --run runs_fact/coup4/s0 > logs/fact/gbs_coup4_s0.out 2>&1 &
python -u fact_gbs.py --run runs_fact/coup4/s2 > logs/fact/gbs_coup4_s2.out 2>&1 &
python -u fact_gbs.py --run runs_fact/prod3/s2 > logs/fact/gbs_prod3_s2.out 2>&1 &
wait
