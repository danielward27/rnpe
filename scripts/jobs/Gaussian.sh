#!/bin/bash

#SBATCH --job-name=Gaussian
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/user/work/dw16200/rnpe/logs/%A_%a.out
#SBATCH --time=0:30:00
#SBATCH --array=0-5

# Example submission: sbatch scripts/jobs/Gaussian.sh

module load lang/python/miniconda/3.9.7
module load lang/gcc/9.3.0
source activate rnpe_env
python -m scripts.run_task --seed=$SLURM_ARRAY_TASK_ID --task-name="${SLURM_JOB_NAME}" --results-dir="/user/work/dw16200/rnpe/${SLURM_JOB_NAME}" --slab-scale=0.5 
