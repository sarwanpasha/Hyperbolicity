#!/bin/bash
#SBATCH --job-name=hyperbolicity_b3
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --mem=112g
#SBATCH --time=00:58:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
set -x
source ${HYP_ROOT:-.}/slurm/env.sh
nvidia-smi -L
export HF_HUB_OFFLINE=1
B="--model 3B --epochs 4 --steps 100 --P 8 --K 4 --grad-ckpt"
if [ "$M" = "none" ]; then
  $PY -u 12_finetune_lora.py $B --method none --seed 0
elif [ "$M" = "ft" ]; then
  $PY -u 12_finetune_lora.py $B --method ft --seed $SD
else
  $PY -u 12_finetune_lora.py $B --method hyp --lam-hyp 10 --agg topq --tag topq --seed $SD
fi
echo B3_DONE
