#!/bin/bash
#SBATCH --job-name=hyperbolicity_hyp_smoke
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=00:55:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
set -x
source ${HYP_ROOT:-.}/slurm/env.sh
nvidia-smi -L
$PY -u 02_extract_embeddings.py --model 35M --dataset scope || exit 1
$PY -u 03_features.py --model 35M --dataset scope --level superfam || exit 1
$PY -u 12_finetune_lora.py --model 35M --method hyp --lam-hyp 1.0 --epochs 1 --steps 5 --tag smoke || exit 1
echo SMOKE_DONE
