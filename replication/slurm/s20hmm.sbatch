#!/bin/bash
#SBATCH --job-name=hyperbolicity_s20hmm
#SBATCH --partition=ondemand
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --constraint=cpu8mem64a
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/%x_%A_%a.log
source ${HYP_ROOT:-.}/slurm/env.sh
set -x
H=$HYP_ROOT/hmm; S=$HYP_ROOT/s20
B=$H/hmmer/bin
I=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
Q=$S/chunks/q$I.fasta
$B/phmmer --cpu 8 -E 1000 --noali --tblout $S/phmmer_$I.tbl -o /dev/null $Q $S/s20_gallery.fasta
echo PHMMER_DONE_$I
$B/jackhmmer --cpu 8 -N 3 -E 1000 --noali --tblout $S/jack_$I.tbl -o /dev/null $Q $S/s20_gallery.fasta
echo JACK_DONE_$I
