#!/bin/bash
#SBATCH --job-name=hyperbolicity_hyp_r8
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
E=4; S=100; P=16; K=4
B="--model 35M --epochs $E --steps $S --P $P --K $K"
# push the drift control until the constraint actually binds
for LAM in 100 1000 10000; do
  $PY -u 12_finetune_lora.py $B --method drift --lam-hyp $LAM --seed 0
done
# extend the main arms to five seeds
for SD in 3 4; do
  $PY -u 12_finetune_lora.py $B --method ft --seed $SD
  $PY -u 12_finetune_lora.py $B --method hyp --lam-hyp 10 --agg topq --tag topq --seed $SD
  $PY -u 21_significance.py --a lora_35M_ft_lh1_s${SD} --b lora_35M_hyp_lh10_s${SD}_topq || true
done
$PY -u 20_summarize.py
echo R8_DONE
