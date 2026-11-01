#!/bin/bash
#SBATCH --job-name=hyperbolicity_e1
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --mem=96g
#SBATCH --time=00:58:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
set -x
source ${HYP_ROOT:-.}/slurm/env.sh
export HF_HUB_OFFLINE=1
B="--model 35M --epochs 4 --steps 100 --P 16 --K 4"
if [ "$E" = "lse" ]; then
  for T in 0.005 0.01 0.02 0.05 0.15 0.5; do
    $PY -u 12_finetune_lora.py $B --method hyp --lam-hyp 10 --agg lse --lse-temp $T --tag lse$T --seed 0
  done
elif [ "$E" = "temp" ]; then
  for T in 0.03 0.05 0.07 0.2 0.5; do
    $PY -u 12_finetune_lora.py $B --method ft --temp $T --tag tau$T --seed 0
  done
elif [ "$E" = "full" ]; then
  $PY -u 12_finetune_lora.py $B --method ft --lora-r 0 --tag full --seed 0
  $PY -u 12_finetune_lora.py $B --method hyp --lam-hyp 10 --agg topq --lora-r 0 --tag fullhyp --seed 0
  $PY -u 12_finetune_lora.py --model 150M --epochs 4 --steps 100 --P 16 --K 4 --grad-ckpt --method ft --lora-r 0 --tag full --seed 0
  $PY -u 12_finetune_lora.py --model 150M --epochs 4 --steps 100 --P 16 --K 4 --grad-ckpt --method hyp --lam-hyp 10 --agg topq --lora-r 0 --tag fullhyp --seed 0
else
  for SD in 0 1 2; do
    $PY -u 12_finetune_lora.py $B --method ft --sampler fold --tag foldblk --seed $SD
  done
  $PY -u 12_finetune_lora.py $B --method hyp --lam-hyp 10 --agg topq --sampler fold --tag foldblkhyp --seed 0
fi
echo E1_DONE
