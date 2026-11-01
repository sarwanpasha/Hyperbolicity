#!/bin/bash
#SBATCH --job-name=hyperbolicity_alnsub
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-node=1
#SBATCH --mem=120g
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.log
source ${HYP_ROOT:-.}/slurm/env.sh
cd $HYP_ROOT/code
$PY 42_align_subset.py
echo ALN_DONE
