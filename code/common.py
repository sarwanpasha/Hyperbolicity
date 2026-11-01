"""Shared paths, logging and small utilities for the hyperbolicity project."""
import os, sys, logging, time, json
from pathlib import Path

ROOT = Path(os.environ.get("HYP_ROOT", "/path/to/Hyperbolicity"))
DATA = Path(os.environ.get("HYP_DATA", "/path/to/scratch"))
RAW = DATA / "raw"
PROC = DATA / "proc"
EMB = DATA / "emb"
CKPT = DATA / "ckpt"
HF_CACHE = DATA / "hf_cache"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
for _p in (RAW, PROC, EMB, CKPT, HF_CACHE, RESULTS, LOGS):
    _p.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE))

ESM_MODELS = {
    "35M": "facebook/esm2_t12_35M_UR50D",
    "150M": "facebook/esm2_t30_150M_UR50D",
    "650M": "facebook/esm2_t33_650M_UR50D",
    "3B": "facebook/esm2_t36_3B_UR50D",
}

def get_logger(name):
    lg = logging.getLogger(name)
    if lg.handlers:
        return lg
    lg.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s " + name + " %(message)s",
                                     datefmt="%Y-%m-%d %H:%M:%S"))
    lg.addHandler(h)
    return lg

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)

class Timer:
    def __init__(self, log, msg):
        self.log, self.msg = log, msg
    def __enter__(self):
        self.t = time.time(); return self
    def __exit__(self, *a):
        self.log.info(self.msg + " took %.1fs" % (time.time() - self.t))
