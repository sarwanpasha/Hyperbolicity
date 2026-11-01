#!/bin/bash
#SBATCH --job-name=hyperbolicity_synthnj
#SBATCH --partition=ondemand
#SBATCH --constraint=cpu8mem64a
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-26
#SBATCH --output=logs/%x_%A_%a.log
source ${HYP_ROOT:-.}/slurm/env.sh
export PYTHONPATH=$HYP_ROOT/pylibs
export OMP_NUM_THREADS=8
set -x
cd $HYP_ROOT/code
python3 06_synth_nj.py --task $SLURM_ARRAY_TASK_ID --ntask 27 --steps 400
echo SYNTHNJ_DONE_$SLURM_ARRAY_TASK_ID
