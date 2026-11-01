#!/bin/bash
#SBATCH --job-name=hyperbolicity_emb
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=200g
#SBATCH --time=14:00:00
#SBATCH --output=logs/%x_%j.log
source ${HYP_ROOT:-.}/slurm/env.sh
set -x
M=${M:-35M}
SEEDS=${SEEDS:-"0 1 2 3 4"}
EXTRA=""
if [ "$M" = "650M" ]; then EXTRA="--grad-ckpt"; fi
if [ "$M" = "3B" ]; then EXTRA="--grad-ckpt"; fi
for S in $SEEDS; do
  $PY 12_finetune_lora.py --model $M --method ft  --seed $S --overwrite --save-cath-emb $EXTRA
  $PY 12_finetune_lora.py --model $M --method hyp --lam-hyp 10 --seed $S --overwrite --save-cath-emb $EXTRA
done
echo EMB_DONE_$M
