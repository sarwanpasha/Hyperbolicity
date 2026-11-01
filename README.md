# Hyperbolicity

Code for measuring and controlling the tree-likeness of protein language model
embeddings. The repository measures Gromov delta-hyperbolicity of ESM-2 embeddings,
fine-tunes them with supervised contrastive learning on a structural hierarchy, and adds
a differentiable four-point penalty that keeps the representation tree-like during
fine-tuning. It also contains the remote homology benchmarks (CATH S35 and S20), the
alignment and profile HMM baselines, and the tree-reconstruction experiments on
simulated phylogenies.

Everything runs from public data.

## Layout

```
code/          main pipeline: data, embeddings, fine-tuning, baselines, statistics
slurm/         batch scripts for the main pipeline, plus env.sh
testdata/      a small subset of SCOPe and CATH for the smoke test
replication/   an independent reimplementation of the fine-tuning, the profile HMM
               baselines, the CATH S20 benchmark and neighbor-joining tree recovery
  code/
  slurm/
```

## Requirements

Python 3.10 or newer. Fine-tuning needs one NVIDIA GPU; everything else runs on CPU.

```bash
python -m venv venv && . venv/bin/activate
pip install torch transformers pandas pyarrow numpy scipy scikit-learn parasail matplotlib
```

Optional external tools:

- HMMER 3 (`phmmer`, `jackhmmer`) for the profile baselines, http://hmmer.org
- Foldseek for the structure-based reference, https://github.com/steineggerlab/foldseek

## Configuration

```bash
export HYP_ROOT=/path/to/Hyperbolicity     # this repository; results go here
export HYP_DATA=/path/to/scratch           # databases, model caches, embeddings
```

`slurm/env.sh` and `replication/slurm/env.sh` set these for batch jobs. The Slurm scripts
need their `--account` and `--partition` lines adapted to your cluster.

## Smoke test

Runs end to end on the bundled subset in a few minutes and checks the installation.

```bash
cd code
python smoke_test.py          # one GPU
python smoke_test.py --cpu    # no GPU, slower
```

It fine-tunes ESM-2 35M for 20 steps on `testdata/scope_mini.parquet` (359 SCOPe domains,
60 superfamilies) with and without the penalty, and reports relative hyperbolicity and
top-1 homologous-superfamily retrieval on `testdata/cath_mini.parquet` (600 CATH domains,
120 superfamilies). The penalty run should show a lower relative hyperbolicity. Absolute
numbers are not comparable to full-scale runs, which use the full databases and 400 steps.

## Main pipeline (`code/`)

```bash
cd code
python 01_download_data.py                     # SCOPe 2.08 ASTRAL 40% and CATH v4.4.0 S35
python 02_extract_embeddings.py --model 35M     # frozen ESM-2 embeddings
python 05_splits.py                             # fold-disjoint SCOPe splits

# LoRA fine-tuning: task-only, and with the four-point penalty
python 12_finetune_lora.py --model 35M --method ft  --seed 0
python 12_finetune_lora.py --model 35M --method hyp --lam-hyp 10 --seed 0
# slurm/lora*.sbatch and slurm/m650.sbatch hold the exact schedule, batch shape and
# seeds used at each model size

# baselines on the same gallery
python 31_align_baseline.py      # Smith-Waterman
python 32_prottucker.py          # ProtT5 and ProtTucker
python 41_foldseek.py            # Foldseek, after slurm/fs.sbatch

# statistics
python 20_summarize.py
python 21_significance.py --a <run_a> --b <run_b>
python 23_blocked_signif.py
python 44_strat_seeds.py         # retrieval stratified by remoteness, across seeds
```

`12_finetune_lora.py --help` lists every method, including the Poincare head, uniformity,
CPCC, HypStructure and multi-level supervision baselines. Other scripts cover the
estimator checks (`26_`, `27_`, `40_`), neural collapse (`nc.py`), CIFAR-100 (`34_`, `37_`),
and the LoRA simulation on a known tree (`13_synthetic.py`).

## Replication and remote homology (`replication/`)

A second implementation written from the method description, used for the S20 benchmark
and the tree-reconstruction experiment.

```bash
cd replication/code
python 01_build_cath.py          # rebuild the CATH v4.4.0 S35 gallery
python 02_make_fastas.py         # gallery and query FASTA files
python 03_maxident.py 0 1        # highest identity to a true homolog (task, ntasks)
python 04_sw_baseline.py 0 1     # Smith-Waterman nearest neighbor
# profile baselines: see ../slurm/hmmsearch.sbatch (phmmer, jackhmmer -N 3, E < 1000)
python 05_analyse.py             # alignment and profile retrieval, stratified

python 07_scope.py               # SCOPe split
python 10_fetch_models.py        # cache ESM-2 weights
python 08_finetune.py --model 35M --method ft  --seed 0 --grad-ckpt
python 08_finetune.py --model 35M --method hyp --seed 0 --grad-ckpt --lam-hyp 10

python 11_s20_build.py           # CATH S20 benchmark
python 12_s20_sw.py 0 1          # Smith-Waterman on S20
python 13_s20_eval.py            # score all arms and baselines on S20

python 06_synth_nj.py --task 0   # one of 27 simulated trees
python 09_nj_summary.py          # Robinson-Foulds, quartet agreement, purity
```

Use `--bf16-trunk` for the 3B model on GPUs with less than 40 GB of memory.

## The penalty on its own

`code/hyploss.py` is self-contained:

```python
from hyploss import hyperbolicity_loss, anticollapse_loss

# z: (B, d) unit-norm embeddings
l_hyp, _ = hyperbolicity_loss(z, n_quads=4096, agg="topq", topq=0.05)
l_ac, _ = anticollapse_loss(z, min_dist=1.0)
loss = loss_task + 10.0 * l_hyp + l_ac
```

It needs no tree and no labels, so it can be added to any objective that produces a
Euclidean embedding.

## Data sources

- SCOPe 2.08 ASTRAL, https://scop.berkeley.edu
- CATH v4.4.0, https://www.cathdb.info
- ESM-2, https://huggingface.co/facebook
- ProtT5 and ProtTucker, https://github.com/Rostlab/EAT
- CIFAR-100, https://www.cs.toronto.edu/~kriz/cifar.html

## License

MIT for the code in this repository. Databases and pretrained models keep their own
licenses.
