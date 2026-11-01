"""Rebuild the CATH v4.4.0 S35 gallery used throughout the paper."""
import os, re, sys
import pandas as pd

D = os.environ["HYP_DATA"]
cls = {}
for line in open(D + "/cath-list.txt"):
    if line.startswith("#"):
        continue
    p = line.split()
    if len(p) < 10:
        continue
    cls[p[0]] = (p[1], p[2], p[3], p[4])

rows, skipped = [], {"nocls": 0, "len": 0, "empty": 0}
did, seq = None, []
def flush():
    global did, seq
    if did is None:
        return
    s = "".join(seq)
    if not s:
        skipped["empty"] += 1
    elif did not in cls:
        skipped["nocls"] += 1
    elif not (30 <= len(s) <= 1000):
        skipped["len"] += 1
    else:
        C, A, T, H = cls[did]
        rows.append(dict(did=did, cath_c=C, cath_a=C + "." + A,
                         cath_t=C + "." + A + "." + T,
                         cath_h=C + "." + A + "." + T + "." + H,
                         seq=s, length=len(s)))
    did, seq = None, []

for line in open(D + "/cath-S35.fa"):
    line = line.strip()
    if line.startswith(">"):
        flush()
        m = re.search(r"\|([0-9A-Za-z_]+)/", line) or re.search(r"\|([0-9A-Za-z_]+)$", line)
        did = m.group(1) if m else line[1:].split()[0].split("|")[-1].split("/")[0]
    else:
        seq.append(line)
flush()
df = pd.DataFrame(rows)
print("kept", len(df), "superfams", df.cath_h.nunique(), "skipped", skipped)
df.to_parquet(D + "/cath.parquet", index=False)
