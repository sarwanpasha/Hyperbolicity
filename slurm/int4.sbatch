#!/bin/bash
#SBATCH --job-name=hyperbolicity_hyp_int4
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
$PY -u 02_extract_embeddings.py --model 35M --dataset cath || exit 1
LAYER=$($PY -c "import json;print(json.load(open('$HYP_ROOT/results/layerscan_scope_35M_superfam.json'))['best'])")
echo "using layer $LAYER"
$PY -u 03_features.py --model 35M --dataset cath --level superfam --layer $LAYER || exit 1
$PY -u 03_features.py --model 35M --dataset scope --level fold || exit 1
LAYERF=$($PY -c "import json;print(json.load(open('$HYP_ROOT/results/layerscan_scope_35M_fold.json'))['best'])")
$PY -u 03_features.py --model 35M --dataset cath --level fold --layer $LAYERF || exit 1
for L in superfam fold; do
  for O in y_top1 y_top5; do
    $PY -u 04_selective.py --model 35M --level $L --outcome $O || exit 1
  done
done
$PY -u 20_summarize.py
echo INT4_DONE
