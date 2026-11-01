#!/bin/bash
#SBATCH --job-name=hyperbolicity_hyp_cpu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
source ${HYP_ROOT:-.}/slurm/env.sh
eval "$CMD"
