#!/bin/bash
#SBATCH --job-name=hyperbolicity_hyp_r4
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
# regularizer baselines: uniformity (Wang and Isola) and CPCC tree-metric correlation
for MM in unif cpcc; do
  for LAM in 0.1 1.0 10.0; do
    $PY -u 12_finetune_lora.py $B --method $MM --lam-hyp $LAM --seed 0
  done
done
$PY -u 20_summarize.py
echo R4_DONE
