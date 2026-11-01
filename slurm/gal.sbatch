#!/bin/bash
#SBATCH --job-name=hyperbolicity_galchk
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem=120g
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.log
source ${HYP_ROOT:-.}/slurm/env.sh
cd $HYP_ROOT/code
$PY 43_gallery_check.py
echo GAL_DONE
