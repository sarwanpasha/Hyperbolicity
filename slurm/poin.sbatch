#!/bin/bash
#SBATCH --job-name=hyperbolicity_hyp_poin
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
E=4; S=100
for SD in 1 2; do
  $PY -u 12_finetune_lora.py --model 35M  --method poincare --epochs $E --steps $S --seed $SD
done
for SD in 1 2; do
  $PY -u 12_finetune_lora.py --model 150M --method poincare --epochs $E --steps $S --seed $SD
done
$PY -u 12_finetune_lora.py --model 35M --method poincare --epochs $E --steps $S --seed 0
for SD in 0 1 2; do
  $PY -u 21_significance.py --a lora_35M_poincare_lh1_s${SD} --b lora_35M_hyp_lh30_s${SD}_topq
  $PY -u 21_significance.py --a lora_150M_poincare_lh1_s${SD} --b lora_150M_hyp_lh30_s${SD}_topq
done
$PY -u 20_summarize.py
echo POIN_DONE
