#!/bin/bash
#SBATCH --job-name=hyperbolicity_hyp_run
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
source ${HYP_ROOT:-.}/slurm/env.sh
nvidia-smi -L
eval "$CMD"
