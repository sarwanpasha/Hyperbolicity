#!/bin/bash
#SBATCH --job-name=hyperbolicity_hyp_int6
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
$PY -u 12_finetune_lora.py --model 150M --method poincare --epochs $E --steps $S --seed 0
for SD in 1 2; do
  $PY -u 12_finetune_lora.py --model 150M --method ft --epochs $E --steps $S --seed $SD
  $PY -u 12_finetune_lora.py --model 150M --method hyp --lam-hyp 30.0 --agg topq --epochs $E --steps $S --seed $SD --tag topq
done
$PY -u 20_summarize.py
echo INT6_DONE
