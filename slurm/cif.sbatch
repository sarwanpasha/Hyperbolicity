#!/bin/bash
#SBATCH --job-name=hyperbolicity_cif
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
export HF_HUB_OFFLINE=1
if [ "$C" = "a" ]; then
  $PY -u 34_cifar.py --method none --seed 0
  for SD in 0 1 2; do $PY -u 34_cifar.py --method ft --seed $SD; done
else
  for SD in 0 1 2; do $PY -u 34_cifar.py --method hyp --lam-hyp 1 --seed $SD; done
  $PY -u 34_cifar.py --method hyp --lam-hyp 10 --seed 0
fi
echo CIF_DONE
