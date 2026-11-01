#!/bin/bash
#SBATCH --job-name=hyperbolicity_maxident
#SBATCH --partition=ondemand
#SBATCH --constraint=cpu8mem64a
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-39
#SBATCH --output=logs/%x_%A_%a.log
source ${HYP_ROOT:-.}/slurm/env.sh
export PYTHONPATH=$HYP_ROOT/pylibs
set -x
python3 $HYP_ROOT/code/03_maxident.py $SLURM_ARRAY_TASK_ID 40
echo MAXIDENT_DONE_$SLURM_ARRAY_TASK_ID
