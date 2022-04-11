#!/bin/bash

#SBATCH --job-name=run_task
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:5:00
#SBATCH --output=results/logs/%x.%j.out
#SBATCH --array=0-200

# Example submission: sbatch --export=task_name="fraziergaussian" scripts/run_task.sh

module load lang/python/miniconda/3.9.7
module load lang/gcc/9.3.0
source activate rnpe_env
python -m scripts.run_task --task-name=$task_name --seed=$SLURM_ARRAY_TASK_ID
