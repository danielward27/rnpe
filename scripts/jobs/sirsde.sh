#!/bin/bash

#SBATCH --job-name=sirsde
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8GB
#SBATCH --time=0:45:00
#SBATCH --output=results/logs/%x.%j.out
#SBATCH --array=0-999

# Example submission: sbatch scripts/jobs/sirsde.sh

module load lang/python/miniconda/3.9.7
module load lang/gcc/9.3.0
source activate rnpe_env
python -m scripts.run_task --task-name="sirsde" --seed=$SLURM_ARRAY_TASK_ID
