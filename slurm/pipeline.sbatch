#!/bin/bash
#SBATCH --job-name=hyperbolicity_hyp_pipe
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --mem=96g
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
set -x
source ${HYP_ROOT:-.}/slurm/env.sh
nvidia-smi -L

# ---- stage 2: embeddings ----
for M in 35M 150M; do
  for D in scope cath; do
    $PY -u 02_extract_embeddings.py --model $M --dataset $D || exit 1
  done
done

# ---- stage 3: retrieval + local geometry features ----
for M in 35M 150M; do
  for L in superfam fold; do
    $PY -u 03_features.py --model $M --dataset scope --level $L || exit 1
    SC=$HYP_ROOT/results/layerscan_scope_${M}_${L}.json
    LAYER=$($PY -c "import json,sys;print(json.load(open('$SC'))['best'])")
    $PY -u 03_features.py --model $M --dataset cath --level $L --layer $LAYER || exit 1
  done
done

# ---- stage 4: selective prediction ----
for M in 35M 150M; do
  for L in superfam fold; do
    for O in y_top1 y_top5; do
      $PY -u 04_selective.py --model $M --level $L --outcome $O || exit 1
    done
  done
done

# ---- stage 6: intervention (frozen-backbone head) ----
for M in 35M 150M; do
  for METH in none ft poincare; do
    $PY -u 11_finetune.py --model $M --method $METH --seed 0 || exit 1
  done
  for LH in 0.1 0.3 1.0 3.0 10.0; do
    $PY -u 11_finetune.py --model $M --method hyp --lam-hyp $LH --seed 0 || exit 1
  done
done
echo PIPELINE_DONE
