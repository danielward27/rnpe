#!/bin/bash

#SBATCH --job-name=fraziergaussian
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:10:00
#SBATCH --output=results/logs/%x.%j.out
#SBATCH --array=0-999

# Example submission: sbatch scripts/jobs/fraziergaussian.sh

module load lang/python/miniconda/3.9.7
module load lang/gcc/9.3.0
source activate rnpe_env
python -m scripts.run_task --task-name="fraziergaussian" --seed=$SLURM_ARRAY_TASK_ID
