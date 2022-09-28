#!/bin/bash

#SBATCH --job-name=GaussianLinear
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/user/work/dw16200/project/misspecification/rnpe/results/logs/%x_%A_%a.out
#SBATCH --time=1:45:00
#SBATCH --array=0-1000

# Example submission: sbatch scripts/jobs/GaussianLinear.sh

module load lang/python/miniconda/3.9.7
module load lang/gcc/9.3.0
source activate rnpe_env
python -m scripts.run_task --seed=$SLURM_ARRAY_TASK_ID --task-name="${SLURM_JOB_NAME}" --results-dir="/user/work/dw16200/project/misspecification/rnpe/results/${SLURM_JOB_NAME}" --well-specified
