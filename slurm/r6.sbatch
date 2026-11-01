#!/bin/bash
#SBATCH --job-name=hyperbolicity_r6
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --mem=96g
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log
source ${HYP_ROOT:-.}/slurm/env.sh
cd $HYP_ROOT/code
set -x
if [ "$E" = "hypstruct" ]; then
  for L in 0.1 1 10; do
    $PY 12_finetune_lora.py --model 35M --method hypstruct --lam-hyp $L --seed 0
  done
elif [ "$E" = "multilevel" ]; then
  $PY 12_finetune_lora.py --model 35M --method multilevel --lam-hyp 1 --seed 0
  $PY 12_finetune_lora.py --model 35M --method multilevel --lam-hyp 1 --seed 1
elif [ "$E" = "nc" ]; then
  $PY 12_finetune_lora.py --model 35M --method none --seed 0 --nc --tag ncf
  $PY 12_finetune_lora.py --model 35M --method ft --seed 0 --nc --tag nc
  $PY 12_finetune_lora.py --model 35M --method hyp --lam-hyp 10 --seed 0 --nc --tag nc
elif [ "$E" = "cifdis" ]; then
  $PY 37_cifar_disjoint.py --method none --seed 0
  for S in 0 1 2; do $PY 37_cifar_disjoint.py --method ft --seed $S; done
elif [ "$E" = "cifdish" ]; then
  for S in 0 1 2; do $PY 37_cifar_disjoint.py --method hyp --lam-hyp 1 --seed $S; done
  $PY 37_cifar_disjoint.py --method hyp --lam-hyp 10 --seed 0
fi
echo R6_DONE
if [ "$E" = "hs2" ]; then
  $PY 12_finetune_lora.py --model 35M --method hypstruct --lam-hyp 30 --seed 0
  $PY 12_finetune_lora.py --model 35M --method hypstruct --lam-hyp 100 --seed 0
fi
if [ "$E" = "hs3" ]; then
  $PY 12_finetune_lora.py --model 35M --method hypstruct --lam-hyp 10 --seed 1
  $PY 12_finetune_lora.py --model 35M --method hypstruct --lam-hyp 10 --seed 2
  $PY 12_finetune_lora.py --model 35M --method multilevel --lam-hyp 1 --seed 2
fi
B="--epochs 4 --steps 100"
if [ "$E" = "hsA" ]; then
  for L in 0.1 1 10; do
    $PY 12_finetune_lora.py --model 35M --method hypstruct --lam-hyp $L --seed 0 $B --overwrite
  done
fi
if [ "$E" = "hsB" ]; then
  for L in 30 100; do
    $PY 12_finetune_lora.py --model 35M --method hypstruct --lam-hyp $L --seed 0 $B --overwrite
  done
  $PY 12_finetune_lora.py --model 35M --method multilevel --lam-hyp 1 --seed 0 $B --overwrite
fi
if [ "$E" = "hsC" ]; then
  for S in 1 2; do
    $PY 12_finetune_lora.py --model 35M --method hypstruct --lam-hyp 10 --seed $S $B --overwrite
    $PY 12_finetune_lora.py --model 35M --method multilevel --lam-hyp 1 --seed $S $B --overwrite
  done
fi
if [ "$E" = "hsD" ]; then
  $PY 12_finetune_lora.py --model 35M --method none --seed 0 --nc --tag ncf $B --overwrite
  $PY 12_finetune_lora.py --model 35M --method ft --seed 0 --nc --tag nc $B --overwrite
  $PY 12_finetune_lora.py --model 35M --method hyp --lam-hyp 10 --seed 0 --nc --tag nc $B --overwrite
fi
