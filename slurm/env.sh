module load python/3.13.5-gcc13.3.1 >/dev/null 2>&1
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export HYP_ROOT=/path/to/Hyperbolicity
export HYP_DATA=/path/to/scratch
export VENV=/path/to/venv
export PY=$VENV/bin/python
export HF_HOME=$HYP_DATA/hf_cache
export TOKENIZERS_PARALLELISM=false
cd $HYP_ROOT/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
