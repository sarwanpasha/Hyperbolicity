#!/bin/bash
#SBATCH --job-name=hyperbolicity_hyp_650
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --mem=96g
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
set -x
source ${HYP_ROOT:-.}/slurm/env.sh
nvidia-smi -L
E=4; S=100; P=12; K=4
# headline comparison first, so partial completion is still useful
$PY -u 12_finetune_lora.py --model 650M --method none --epochs $E --steps $S --P $P --K $K --grad-ckpt --seed 0
$PY -u 12_finetune_lora.py --model 650M --method ft   --epochs $E --steps $S --P $P --K $K --grad-ckpt --seed 0
$PY -u 12_finetune_lora.py --model 650M --method hyp --lam-hyp 30.0 --agg topq --epochs $E --steps $S --P $P --K $K --grad-ckpt --seed 0 --tag topq
$PY -u 21_significance.py --a lora_650M_ft_lh1_s0 --b lora_650M_hyp_lh30_s0_topq
$PY -u 20_summarize.py
$PY -u 12_finetune_lora.py --model 650M --method poincare --epochs $E --steps $S --P $P --K $K --grad-ckpt --seed 0
for SD in 1 2; do
  $PY -u 12_finetune_lora.py --model 650M --method ft  --epochs $E --steps $S --P $P --K $K --grad-ckpt --seed $SD
  $PY -u 12_finetune_lora.py --model 650M --method hyp --lam-hyp 30.0 --agg topq --epochs $E --steps $S --P $P --K $K --grad-ckpt --seed $SD --tag topq
  $PY -u 21_significance.py --a lora_650M_ft_lh1_s${SD} --b lora_650M_hyp_lh30_s${SD}_topq
done
$PY -u 20_summarize.py
# diagnostic half at 650M, to test whether the negative result is a trend
$PY -u 02_extract_embeddings.py --model 650M --dataset scope
$PY -u 02_extract_embeddings.py --model 650M --dataset cath
$PY -u 03_features.py --model 650M --dataset scope --level superfam
LAYER=$($PY -c "import json;print(json.load(open('$HYP_ROOT/results/layerscan_scope_650M_superfam.json'))['best'])")
$PY -u 03_features.py --model 650M --dataset cath --level superfam --layer $LAYER
$PY -u 04_selective.py --model 650M --level superfam --outcome y_top1
$PY -u 04_selective.py --model 650M --level superfam --outcome y_top5
$PY -u 20_summarize.py
echo M650_DONE
