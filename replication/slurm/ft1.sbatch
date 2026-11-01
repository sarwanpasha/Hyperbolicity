#!/bin/bash
#SBATCH --job-name=hyperbolicity_ft1
#SBATCH --partition=gpu-a10g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.log
# usage: sbatch ft1.sbatch SCALE METHOD SEED [EXTRA]
source ${HYP_ROOT:-.}/slurm/env.sh
export PYTHONPATH=$HYP_ROOT/pylibs
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -x
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd $HYP_ROOT/code
python3 08_finetune.py --model $1 --method $2 --seed $3 --steps 400 --grad-ckpt ${4:-}
