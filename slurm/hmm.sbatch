#!/bin/bash
#SBATCH --job-name=hyperbolicity_hmm
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-node=1
#SBATCH --mem=120g
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.log
source ${HYP_ROOT:-.}/slurm/env.sh
set -x
D=$HYP_DATA/hmm
cd $D
if [ ! -x $D/hmmer/bin/jackhmmer ]; then
  curl -L -o hmmer.tar.gz http://eddylab.org/software/hmmer/hmmer.tar.gz
  tar xzf hmmer.tar.gz
  cd hmmer-*
  ./configure --prefix=$D/hmmer > conf.log 2>&1
  make -j 16 > make.log 2>&1
  make install > inst.log 2>&1
  cd $D
fi
$D/hmmer/bin/jackhmmer -h | head -2
# phmmer: single-sequence profile search, the fast reference
$D/hmmer/bin/phmmer --cpu 48 -E 1000 --noali \
  --tblout $D/phmmer.tbl -o /dev/null $D/query.fasta $D/cath.fasta
wc -l $D/phmmer.tbl
echo PHMMER_DONE
# jackhmmer: 3 iterations, builds a profile from the gallery itself
$D/hmmer/bin/jackhmmer --cpu 48 -N 3 -E 1000 --noali \
  --tblout $D/jack.tbl -o /dev/null $D/query.fasta $D/cath.fasta
wc -l $D/jack.tbl
echo JACK_DONE
