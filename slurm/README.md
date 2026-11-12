# SLURM scripts

The batch scripts used for the runs in `results/`, copied unchanged from the original working directory. They are
specific to the cluster they ran on (Snellius, partition `gpu_a100`): they activate a virtual environment in the home
directory and `cd` into the original project folder, and they write logs to `logs/`. To reuse them, change the
`source` and `cd` lines to your environment and this repository, and submit from the repository root.

They mostly run several seeds as parallel background processes on one GPU, which is why some training runs took longer
than they would alone.
