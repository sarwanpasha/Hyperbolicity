#!/bin/bash
#SBATCH --job-name=hyperbolicity_hyp_int7
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
# paired significance on the 150M runs (per-query outcomes already saved)
for SD in 0 1 2; do
  $PY -u 21_significance.py --a lora_150M_ft_lh1_s${SD} --b lora_150M_hyp_lh30_s${SD}_topq
done
$PY -u 21_significance.py --a lora_150M_poincare_lh1_s0 --b lora_150M_hyp_lh30_s0_topq
$PY -u 21_significance.py --a lora_150M_none_lh1_s0 --b lora_150M_ft_lh1_s0
# regenerate 35M runs so their per-query outcomes exist for paired tests
E=4; S=100
for SD in 0 1 2; do
  $PY -u 12_finetune_lora.py --model 35M --method ft --epochs $E --steps $S --seed $SD
done
for SD in 0 1; do
  $PY -u 12_finetune_lora.py --model 35M --method hyp --lam-hyp 30.0 --agg topq --epochs $E --steps $S --seed $SD --tag topq
done
for SD in 0 1 2; do
  $PY -u 21_significance.py --a lora_35M_ft_lh1_s${SD} --b lora_35M_hyp_lh30_s${SD}_topq
done
$PY -u 20_summarize.py
echo INT7_DONE
