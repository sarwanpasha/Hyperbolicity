#!/bin/bash
#SBATCH --job-name=hyperbolicity_fski
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem=120g
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log
source ${HYP_ROOT:-.}/slurm/env.sh
set -x
S=$HYP_DATA/struct
FS=$(ls $S/foldseek/bin/foldseek 2>/dev/null || ls $S/foldseek/foldseek)
mkdir -p $S/sub $S/db
$PY - <<'PYX'
import os
S = os.environ['HYP_DATA'] + '/struct'
os.makedirs(S + '/sub', exist_ok=True)
n = 0
miss = 0
for line in open(S + '/overlap.list'):
    d = line.strip()
    if not d:
        continue
    src = S + '/dompdb/' + d
    dst = S + '/sub/' + d + '.pdb'
    if not os.path.exists(src):
        miss += 1
        continue
    if not os.path.exists(dst):
        os.symlink(src, dst)
    n += 1
print('linked', n, 'missing', miss)
PYX
ls $S/sub | wc -l
rm -rf $S/db/sdb* $S/db/res* $S/db/tmp
$FS createdb $S/sub $S/db/sdb
$FS search $S/db/sdb $S/db/sdb $S/db/res $S/db/tmp -a --threads 32 -s 9.5 --max-seqs 500 -e 1000
$FS convertalis $S/db/sdb $S/db/sdb $S/db/res $S/fs_hits.tsv --threads 32 --format-output query,target,fident,alnlen,evalue,bits
wc -l $S/fs_hits.tsv
echo FS_SEARCH_DONE
