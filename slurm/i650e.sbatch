#!/bin/bash
#SBATCH --job-name=hyperbolicity_hyp_650e
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
$PY -u 12_finetune_lora.py --model 650M --method poincare --epochs $E --steps $S --P $P --K $K --grad-ckpt --seed 0
for LAM in 10 100; do
  $PY -u 12_finetune_lora.py --model 650M --method hyp --lam-hyp $LAM --agg topq --epochs $E --steps $S --P $P --K $K --grad-ckpt --tag topq --seed 0
done
$PY -u 20_summarize.py
echo I650E_DONE
