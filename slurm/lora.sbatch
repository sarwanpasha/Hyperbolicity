#!/bin/bash
#SBATCH --job-name=hyperbolicity_hyp_lora
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --mem=96g
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
set -x
source ${HYP_ROOT:-.}/slurm/env.sh
nvidia-smi -L
M=${M:-35M}
$PY -u 12_finetune_lora.py --model $M --method none  --seed 0 || exit 1
$PY -u 12_finetune_lora.py --model $M --method ft    --seed 0 || exit 1
$PY -u 12_finetune_lora.py --model $M --method poincare --seed 0 || exit 1
for LH in 0.3 1.0 3.0; do
  $PY -u 12_finetune_lora.py --model $M --method hyp --lam-hyp $LH --seed 0 || exit 1
done
echo LORA_DONE
