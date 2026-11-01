#!/bin/bash
#SBATCH --job-name=hyperbolicity_hyp_int8
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=00:58:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
set -x
source ${HYP_ROOT:-.}/slurm/env.sh
nvidia-smi -L
$PY -u 02_extract_embeddings.py --model 150M --dataset scope || exit 1
$PY -u 02_extract_embeddings.py --model 150M --dataset cath || exit 1
$PY -u 03_features.py --model 150M --dataset scope --level superfam || exit 1
LAYER=$($PY -c "import json;print(json.load(open('$HYP_ROOT/results/layerscan_scope_150M_superfam.json'))['best'])")
$PY -u 03_features.py --model 150M --dataset cath --level superfam --layer $LAYER || exit 1
$PY -u 04_selective.py --model 150M --level superfam --outcome y_top1 || exit 1
$PY -u 04_selective.py --model 150M --level superfam --outcome y_top5 || exit 1
$PY -u 20_summarize.py
echo INT8_DONE
