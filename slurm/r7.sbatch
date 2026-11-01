#!/bin/bash
#SBATCH --job-name=hyperbolicity_hyp_3b
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --mem=112g
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
set -x
source ${HYP_ROOT:-.}/slurm/env.sh
nvidia-smi -L
export HF_HUB_OFFLINE=1
E=4; S=100; P=8; K=4
B="--model 3B --epochs $E --steps $S --P $P --K $K --grad-ckpt"
$PY -u 12_finetune_lora.py $B --method none --seed 0
for SD in 0 1; do
  $PY -u 12_finetune_lora.py $B --method ft --seed $SD
  $PY -u 12_finetune_lora.py $B --method hyp --lam-hyp 10 --agg topq --tag topq --seed $SD
  $PY -u 21_significance.py --a lora_3B_ft_lh1_s${SD} --b lora_3B_hyp_lh10_s${SD}_topq || true
done
$PY -u 20_summarize.py
echo R7_DONE
