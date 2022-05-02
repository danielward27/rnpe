#!/bin/bash

#SBATCH --job-name=cancer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/user/work/dw16200/rnpe/logs/%x.%j.out
#SBATCH --time=0:59:59
#SBATCH --mem=8GB
#SBATCH --array=17,27,49,51,55,63,67,68,77,83,94,99,104,113,115,122,124,143,154,160,165,172,178,182,184,190,194,195,196,205,209,212,229,253,261,281,301,306,320,323,342,351,363,367,369,377,382,392,403,433,436,437,443,446,447,450,461,471,494,504,509,521,524,525,532,560,564,565,570,577,593,614,615,619,625,631,640,645,647,650,661,665,669,671,680,681,687,689,702,704,715,717,722,724,732,741,747,760,763,767,770,786,788,809,813,822,858,860,868,882,891,892,900,910,921,923,927,928,939,943,950,957,965,968,979,986,993

# Example submission: sbatch scripts/jobs/cancer.sh

module load lang/python/miniconda/3.9.7
module load lang/gcc/9.3.0
source activate rnpe_env
python -m scripts.run_task --seed=$SLURM_ARRAY_TASK_ID --task-name="${SLURM_JOB_NAME}" --results-dir="/user/work/dw16200/rnpe/${SLURM_JOB_NAME}" 


