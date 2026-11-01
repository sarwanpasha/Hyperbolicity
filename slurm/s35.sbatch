#!/bin/bash
#SBATCH --job-name=hyperbolicity_s35
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
E=4; S=100; P=16; K=4
B="--model 35M --epochs $E --steps $S --P $P --K $K"
if [ "$HALF" = "a" ]; then
  for LAM in 1 3 10 30 100 300; do
    $PY -u 12_finetune_lora.py $B --method hyp --lam-hyp $LAM --agg topq --tag topq --seed $SD
  done
  for LAM in 0.1 1.0 10.0; do
    $PY -u 12_finetune_lora.py $B --method cpcc --lam-hyp $LAM --seed $SD
  done
  $PY -u 12_finetune_lora.py $B --method poincare --seed $SD
else
  for LAM in 0.1 1.0 10.0; do
    $PY -u 12_finetune_lora.py $B --method unif --lam-hyp $LAM --seed $SD
  done
  for LAM in 0.1 1.0 10.0 100 1000 10000; do
    $PY -u 12_finetune_lora.py $B --method drift --lam-hyp $LAM --seed $SD
  done
  $PY -u 12_finetune_lora.py $B --method ft --lora-r 2 --tag r2 --seed $SD
  $PY -u 12_finetune_lora.py --model 35M --epochs 2 --steps $S --P $P --K $K --method ft --tag s200 --seed $SD
fi
echo S35_DONE
