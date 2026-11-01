#!/bin/bash
#SBATCH --job-name=hyperbolicity_hyp_ac
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
E=4; S=100; P=12; K=4
for SD in 0 1 2; do
  # (a) anti-collapse ONLY: isolates the uniformity term from the geometry term
  $PY -u 12_finetune_lora.py --model 35M --method hyp --lam-hyp 0 --agg topq \
      --epochs $E --steps $S --P 16 --K 4 --tag aconly16 --seed $SD
  # (b) geometry WITHOUT anti-collapse: does the delta term need the floor at all?
  $PY -u 12_finetune_lora.py --model 35M --method hyp --lam-hyp 30 --lam-ac 0 --agg topq \
      --epochs $E --steps $S --P 16 --K 4 --tag noac16 --seed $SD
  $PY -u 21_significance.py --a lora_35M_ft_lh1_s${SD} --b lora_35M_hyp_lh0_s${SD}_aconly16 || true
  $PY -u 21_significance.py --a lora_35M_hyp_lh0_s${SD}_aconly16 --b lora_35M_hyp_lh30_s${SD}_topq || true
done
$PY -u 20_summarize.py
echo IAC_DONE
