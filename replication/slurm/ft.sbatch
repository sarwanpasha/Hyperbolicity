#!/bin/bash
#SBATCH --job-name=hyperbolicity_ft
#SBATCH --partition=gpu-a10g
#SBATCH --constraint=gpu1mem64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=56G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.log
source ${HYP_ROOT:-.}/slurm/env.sh
export PYTHONPATH=$HYP_ROOT/pylibs
set -x
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd $HYP_ROOT/code
python3 08_finetune.py --model $1 --method $2 --seed $3 \
    --steps ${4:-400} ${5:-}
