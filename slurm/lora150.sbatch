#!/bin/bash
#SBATCH --job-name=hyperbolicity_hyp_l150
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --mem=96g
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
set -x
source ${HYP_ROOT:-.}/slurm/env.sh
nvidia-smi -L
E=4; S=100
for SD in 0 1 2; do
  $PY -u 12_finetune_lora.py --model 150M --method none --epochs $E --steps $S --seed $SD
  $PY -u 12_finetune_lora.py --model 150M --method ft   --epochs $E --steps $S --seed $SD
  $PY -u 12_finetune_lora.py --model 150M --method poincare --epochs $E --steps $S --seed $SD
  for LH in 10.0 30.0 100.0; do
    $PY -u 12_finetune_lora.py --model 150M --method hyp --lam-hyp $LH --agg topq --epochs $E --steps $S --seed $SD --tag topq
  done
done
for SD in 0 1 2; do
  $PY -u 13_synthetic.py --model 35M --depth 9 --seed $SD
done
$PY -u 20_summarize.py
echo L150_DONE
