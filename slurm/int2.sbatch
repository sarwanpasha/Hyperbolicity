#!/bin/bash
#SBATCH --job-name=hyperbolicity_hyp_int2
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
E=4; S=100
for LH in 1.0 3.0 10.0 30.0; do
  $PY -u 12_finetune_lora.py --model 35M --method hyp --lam-hyp $LH --agg topq \
      --epochs $E --steps $S --seed 0 --tag topq
done
$PY -u 12_finetune_lora.py --model 35M --method hyp --lam-hyp 10.0 --agg mean \
    --epochs $E --steps $S --seed 0 --tag aggmean
$PY -u 13_synthetic.py --model 35M --depth 9 --seed 0
$PY -u 20_summarize.py
echo INT2_DONE
