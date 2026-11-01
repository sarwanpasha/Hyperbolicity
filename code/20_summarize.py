"""Collect every result file into one compact summary for reporting."""
import json, glob, os
import numpy as np
import pandas as pd
from common import RESULTS, PROC, get_logger

log = get_logger("summary")
L = []
def w(s=""):
    L.append(s)

w("# Hyperbolicity results summary")
try:
    st = json.load(open(PROC / "stage1_stats.json"))
    sp = json.load(open(PROC / "split_stats.json"))
    w("\n## Data\n")
    w("stage1: " + json.dumps(st))
    w("splits: " + json.dumps(sp))
except Exception as e:
    w("data stats missing: %s" % e)

w("\n## Layer scans\n")
for f in sorted(glob.glob(str(RESULTS / "layerscan_*.json"))):
    d = json.load(open(f))
    acc = d["accs"]
    w("- %s best=%d acc=%.4f  all=[%s]" % (os.path.basename(f), d["best"],
      acc[d["best"]], ", ".join("%.3f" % a for a in acc)))

w("\n## Selective prediction (post-hoc delta signal)\n")
for f in sorted(glob.glob(str(RESULTS / "selective_*.csv"))):
    df = pd.read_csv(f)
    w("\n### " + os.path.basename(f))
    w(df.round(4).to_markdown(index=False))

w("\n## Bootstrap deltas (full vs NN_norm)\n")
for f in sorted(glob.glob(str(RESULTS / "bootstrap_*.json"))):
    d = json.load(open(f))
    w("- %s: %s" % (os.path.basename(f), json.dumps(d)))

w("\n## Bucketed AUROC\n")
for f in sorted(glob.glob(str(RESULTS / "bucketed_*.csv"))):
    df = pd.read_csv(f)
    w("\n### " + os.path.basename(f))
    w(df.round(4).to_markdown(index=False))

w("\n## Intervention runs\n")
rows = []
for f in sorted(glob.glob(str(RESULTS / "run_*.json"))):
    d = json.load(open(f))
    c = d.get("config", {})
    r = {"run": os.path.basename(f)[4:-5],
         "model": c.get("model"), "backbone": c.get("backbone", "lora"),
         "method": c.get("method"), "lam_hyp": c.get("lam_hyp"),
         "seed": c.get("seed")}
    for split in ("val", "test", "cath"):
        for k, v in (d.get(split) or {}).items():
            if isinstance(v, (int, float)):
                r[split + "_" + k] = round(float(v), 4)
    rows.append(r)
if rows:
    df = pd.DataFrame(rows)
    w(df.to_markdown(index=False))
    df.to_csv(RESULTS / "all_runs.csv", index=False)

txt = "\n".join(L)
open(RESULTS / "summary.md", "w").write(txt)
log.info("wrote %s (%d chars)" % (RESULTS / "summary.md", len(txt)))
