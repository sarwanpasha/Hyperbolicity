#!/bin/bash
#SBATCH --job-name=hyperbolicity_hyp_r1
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
E=4; S=100; P=16; K=4
B="--model 35M --epochs $E --steps $S --P $P --K $K"
# (1) lambda=10 selected on the SCOPe validation split, three seeds
for SD in 0 1 2; do
  $PY -u 12_finetune_lora.py $B --method hyp --lam-hyp 10 --agg topq --tag topq --seed $SD
  $PY -u 21_significance.py --a lora_35M_ft_lh1_s${SD} --b lora_35M_hyp_lh10_s${SD}_topq || true
done
# (2) LSE aggregation, the DeltaZero smoothing, at matched weight
for LAM in 10 30; do
  $PY -u 12_finetune_lora.py $B --method hyp --lam-hyp $LAM --agg lse --tag lse --seed 0
done
# (3) rho sweep: is it the tail, or just an effective weight rescaling?
for RQ in 0.01 0.20 1.00; do
  $PY -u 12_finetune_lora.py $B --method hyp --lam-hyp 10 --agg topq --topq $RQ --tag rq$RQ --seed 0
done
# (4) longer schedule: is the delta inflation a property of the converged objective?
$PY -u 12_finetune_lora.py --model 35M --epochs 12 --steps 100 --P $P --K $K --method ft  --tag long --seed 0
$PY -u 12_finetune_lora.py --model 35M --epochs 12 --steps 100 --P $P --K $K --method hyp --lam-hyp 10 --agg topq --tag longtopq --seed 0
$PY -u 20_summarize.py
echo R1_DONE
