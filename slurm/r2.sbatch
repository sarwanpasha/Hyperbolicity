#!/bin/bash
#SBATCH --job-name=hyperbolicity_hyp_r2
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
E=4; S=100; P=12; K=4
for SD in 0 1 2; do
  $PY -u 12_finetune_lora.py --model 150M --epochs 4 --steps 100 --P 16 --K 4 --method hyp --lam-hyp 10 --agg topq --tag topq --seed $SD
  $PY -u 21_significance.py --a lora_150M_ft_lh1_s${SD} --b lora_150M_hyp_lh10_s${SD}_topq || true
done
$PY -u 20_summarize.py
echo R2_DONE
