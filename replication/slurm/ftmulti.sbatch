#!/bin/bash
#SBATCH --job-name=hyperbolicity_ftmulti
#SBATCH --partition=gpu-a10g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=14:00:00
#SBATCH --output=logs/%x_%j.log
# usage: sbatch --constraint=gpu4mem96 --gpus-per-node=4 ftmulti.sbatch SCALE [STEPS] [EXTRA]
source ${HYP_ROOT:-.}/slurm/env.sh
export PYTHONPATH=$HYP_ROOT/pylibs
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8
M=$1; STEPS=${2:-400}; EXTRA=${3:-}
NG=$(nvidia-smi -L | wc -l)
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "scale=$M gpus=$NG steps=$STEPS extra=$EXTRA"
cd $HYP_ROOT/code
i=0
for METH in ft hyp; do
  for S in 0 1 2; do
    G=$((i % NG))
    L=$HYP_ROOT/logs/hyperbolicity_run_${M}_${METH}_s${S}.log
    CUDA_VISIBLE_DEVICES=$G python3 08_finetune.py --model $M --method $METH \
        --seed $S --steps $STEPS $EXTRA > $L 2>&1 &
    echo "launched $M $METH s$S on gpu $G"
    i=$((i + 1))
    if [ $((i % NG)) -eq 0 ]; then wait; fi
  done
done
wait
grep -h RUN_DONE $HYP_ROOT/logs/hyperbolicity_run_${M}_*.log
echo MULTI_DONE_$M
