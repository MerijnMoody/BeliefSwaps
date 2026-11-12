#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=05:00:00
#SBATCH --job-name=flg_rrseeds
#SBATCH --output=logs/rrxor_seeds_%j.out
source ~/venvs/torch_env/bin/activate
cd ~/factored_lgssm
for arch in L4_d64 L4_d64_rope L2_d32 L2_d32_rope; do
  for s in 0 1 2 3 4; do
    for u in "" "--untrained"; do
      tag=${u:+_untrained}
      f=results/plateau/rrxor_${arch}_eps0_s${s}${tag}.json
      if [ -f "$f" ]; then echo "skip $f"; continue; fi
      echo "$(date +%T) $arch s$s $u"
      python rrxor_plateau.py --arch $arch --eps 0 --seed $s $u > logs/rrxor_${arch}_eps0_s${s}${tag}.out 2>&1 || echo "FAILED $arch s$s $u"
    done
  done
done
echo done
