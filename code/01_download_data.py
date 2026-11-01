"""Stage 1: download and parse SCOPe (ASTRAL) and CATH S40 into parquet tables.

All downloads are scriptable and public. Outputs:
  PROC/scope.parquet : sid, sccs, cls, fold, superfam, family, seq, length
  PROC/cath.parquet  : did, C, A, T, H, cath_c, cath_a, cath_t, cath_h, seq, length
"""
import re, sys, urllib.request, gzip, shutil
import pandas as pd
from common import RAW, PROC, get_logger, Timer, save_json

log = get_logger("download")

SCOPE_VER = "2.08"
SCOPE_FA = ("https://scop.berkeley.edu/downloads/scopeseq-" + SCOPE_VER +
            "/astral-scopedom-seqres-gd-sel-gs-bib-40-" + SCOPE_VER + ".fa")
CATH_VER = "v4_4_0"
CATH_BASE = "https://download.cathdb.info/cath/releases/all-releases/" + CATH_VER
CATH_FA = CATH_BASE + "/sequence-data/cath-domain-seqs-S35-" + CATH_VER + ".fa"
CATH_LIST = CATH_BASE + "/cath-classification-data/cath-domain-list-" + CATH_VER + ".txt"

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN, MAX_LEN = 30, 1000


def fetch(url, dest, force=False):
    if dest.exists() and dest.stat().st_size > 0 and not force:
        log.info("cached %s (%.1f MB)" % (dest.name, dest.stat().st_size / 1e6))
        return dest
    log.info("downloading %s" % url)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 research-script"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    log.info("wrote %s (%.1f MB)" % (dest.name, dest.stat().st_size / 1e6))
    return dest


def read_fasta(path):
    name, buf = None, []
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(buf)
                name, buf = line[1:], []
            elif line:
                buf.append(line)
    if name is not None:
        yield name, "".join(buf)


def clean(seq):
    s = seq.upper().replace("-", "").replace(".", "")
    return "".join(c if c in VALID_AA else "X" for c in s)


SCOPE_CLASSES = tuple("abcdef")  # true structural classes; drop g-k (small/coiled/low-res/peptide/designed)
SCCS_RE = re.compile(r"^([a-k])\.(\d+)\.(\d+)\.(\d+)$")


def build_scope():
    fa = fetch(SCOPE_FA, RAW / ("astral-scope-40-" + SCOPE_VER + ".fa"))
    rows, skipped = [], {"bad_sccs": 0, "class": 0, "length": 0, "dup": 0}
    seen = set()
    for hdr, seq in read_fasta(fa):
        parts = hdr.split()
        if len(parts) < 2:
            skipped["bad_sccs"] += 1; continue
        sid, sccs = parts[0], parts[1]
        m = SCCS_RE.match(sccs)
        if m is None:
            skipped["bad_sccs"] += 1; continue
        if m.group(1) not in SCOPE_CLASSES:
            skipped["class"] += 1; continue
        s = clean(seq)
        if not (MIN_LEN <= len(s) <= MAX_LEN):
            skipped["length"] += 1; continue
        if sid in seen:
            skipped["dup"] += 1; continue
        seen.add(sid)
        c, f, sf, fam = m.groups()
        rows.append(dict(sid=sid, sccs=sccs, cls=c, fold=c + "." + f,
                         superfam=c + "." + f + "." + sf, family=sccs,
                         seq=s, length=len(s)))
    df = pd.DataFrame(rows)
    log.info("SCOPe: %d domains kept, skipped=%s" % (len(df), skipped))
    log.info("SCOPe: %d classes, %d folds, %d superfamilies, %d families" %
             (df.cls.nunique(), df.fold.nunique(), df.superfam.nunique(), df.family.nunique()))
    df.to_parquet(PROC / "scope.parquet", index=False)
    return df


def build_cath():
    fa = fetch(CATH_FA, RAW / ("cath-S35-" + CATH_VER + ".fa"))
    lst = fetch(CATH_LIST, RAW / ("cath-domain-list-" + CATH_VER + ".txt"))
    cls = {}
    with open(lst, "r", errors="ignore") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            t = line.split()
            if len(t) < 5:
                continue
            cls[t[0]] = (t[1], t[2], t[3], t[4])
    log.info("CATH classification entries: %d" % len(cls))
    rows, skipped = [], {"noclass": 0, "length": 0, "dup": 0}
    seen = set()
    for hdr, seq in read_fasta(fa):
        did = hdr.split("|")[-1].split("/")[0].strip() if "|" in hdr else hdr.split()[0]
        if did not in cls:
            skipped["noclass"] += 1; continue
        s = clean(seq)
        if not (MIN_LEN <= len(s) <= MAX_LEN):
            skipped["length"] += 1; continue
        if did in seen:
            skipped["dup"] += 1; continue
        seen.add(did)
        C, A, T, H = cls[did]
        rows.append(dict(did=did, cath_c=C, cath_a=C + "." + A,
                         cath_t=C + "." + A + "." + T,
                         cath_h=C + "." + A + "." + T + "." + H,
                         seq=s, length=len(s)))
    df = pd.DataFrame(rows)
    log.info("CATH: %d domains kept, skipped=%s" % (len(df), skipped))
    log.info("CATH: %d C, %d A, %d T, %d H" % (df.cath_c.nunique(), df.cath_a.nunique(),
                                               df.cath_t.nunique(), df.cath_h.nunique()))
    df.to_parquet(PROC / "cath.parquet", index=False)
    return df


if __name__ == "__main__":
    with Timer(log, "stage 1"):
        sc = build_scope()
        ca = build_cath()
    save_json({"scope_n": int(len(sc)), "cath_n": int(len(ca)),
               "scope_superfam": int(sc.superfam.nunique()),
               "scope_fold": int(sc.fold.nunique()),
               "cath_h": int(ca.cath_h.nunique())}, PROC / "stage1_stats.json")
    log.info("STAGE1_DONE")
