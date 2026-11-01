"""Stage 4: selective-prediction analysis of the local delta-hyperbolicity signal.

Fits a logistic abstention model on SCOPe (grouped CV by fold) and transfers it to
CATH. Compares nested feature sets to isolate the marginal value of the geometry
features over similarity/norm baselines. Reports AUROC, AUARC (area under the
accuracy-rejection curve) and accuracy at fixed coverage, with bootstrap CIs.
"""
import argparse, itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from common import RESULTS, get_logger, Timer, save_json

log = get_logger("selective")

FEATSETS = {
    "NN_only":  ["s1"],
    "NN_margin": ["s1", "margin"],
    "NN_norm":  ["s1", "margin", "s_mean", "s_std", "qnorm", "length"],
    "full":     ["s1", "margin", "s_mean", "s_std", "qnorm", "length",
                 "delta", "delta_rel", "diam", "umet"],
    "geom_only": ["delta", "delta_rel", "diam", "umet"],
}
COVGRID = np.linspace(0.05, 1.0, 96)


def auarc(y, conf):
    """Area under the accuracy-rejection curve: mean selective accuracy over a grid
    of coverages, keeping the most confident fraction at each point."""
    o = np.argsort(-conf)
    ys = np.asarray(y)[o]
    csum = np.cumsum(ys)
    n = len(ys)
    accs = []
    for c in COVGRID:
        k = max(1, int(round(c * n)))
        accs.append(csum[k - 1] / k)
    return float(np.mean(accs)), np.array(accs)


def acc_at(y, conf, cov):
    o = np.argsort(-conf)
    ys = np.asarray(y)[o]
    k = max(1, int(round(cov * len(ys))))
    return float(ys[:k].mean())


def fit_predict_cv(df, feats, ycol, groups, n_splits=5, seed=0):
    X = df[feats].to_numpy().astype(np.float64)
    y = df[ycol].to_numpy().astype(int)
    oof = np.zeros(len(df))
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, C=1.0))
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof, y


def fit_full(df, feats, ycol, seed=0):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    m.fit(df[feats].to_numpy().astype(np.float64), df[ycol].to_numpy().astype(int))
    return m


def metrics(y, conf):
    a, _ = auarc(y, conf)
    return {"auroc": float(roc_auc_score(y, conf)),
            "auarc": a,
            "acc@50": acc_at(y, conf, 0.5),
            "acc@70": acc_at(y, conf, 0.7),
            "acc@90": acc_at(y, conf, 0.9),
            "base_acc": float(np.mean(y))}


def bootstrap_delta(y, ca, cb, n=1000, seed=0):
    """Bootstrap CI for metric(cb) - metric(ca)."""
    rng = np.random.default_rng(seed)
    N = len(y); y = np.asarray(y)
    dr, da = [], []
    for _ in range(n):
        i = rng.integers(0, N, N)
        if y[i].min() == y[i].max():
            continue
        dr.append(roc_auc_score(y[i], cb[i]) - roc_auc_score(y[i], ca[i]))
        da.append(auarc(y[i], cb[i])[0] - auarc(y[i], ca[i])[0])
    q = lambda v: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return {"d_auroc": q(dr), "d_auarc": q(da)}


def bucketed(df, y, ca, cb, name_a, name_b):
    rows = []
    q = pd.qcut(df.length, 4, labels=["L1", "L2", "L3", "L4"], duplicates="drop")
    sfsize = df.groupby("superfam")["id"].transform("size")
    sb = pd.cut(sfsize, [0, 2, 5, 20, 10 ** 9], labels=["sf<=2", "sf3-5", "sf6-20", "sf>20"])
    for bname, series in (("length", q), ("class", df["cls"]), ("sf_size", sb)):
        for lev in pd.unique(series.dropna()):
            m = (series == lev).to_numpy()
            if m.sum() < 100 or len(np.unique(y[m])) < 2:
                continue
            rows.append({"bucket": bname, "level": str(lev), "n": int(m.sum()),
                         "base_acc": float(y[m].mean()),
                         ("auroc_" + name_a): float(roc_auc_score(y[m], ca[m])),
                         ("auroc_" + name_b): float(roc_auc_score(y[m], cb[m]))})
    out = pd.DataFrame(rows)
    if len(out):
        out["d_auroc"] = out["auroc_" + name_b] - out["auroc_" + name_a]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="150M")
    ap.add_argument("--level", default="superfam")
    ap.add_argument("--outcome", default="y_top1", choices=["y_top1", "y_top5"])
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    tag = a.model + "_" + a.level + "_" + a.outcome

    with Timer(log, "stage 4 " + tag):
        sc = pd.read_parquet(RESULTS / ("feats_scope_" + a.model + "_" + a.level + ".parquet"))
        ca_df = pd.read_parquet(RESULTS / ("feats_cath_" + a.model + "_" + a.level + ".parquet"))
        log.info("SCOPe n=%d base=%.4f | CATH n=%d base=%.4f" %
                 (len(sc), sc[a.outcome].mean(), len(ca_df), ca_df[a.outcome].mean()))

        rows, oof_store, tr_store = [], {}, {}
        for fs, feats in FEATSETS.items():
            oof, y = fit_predict_cv(sc, feats, a.outcome, sc["fold"].to_numpy())
            m_in = metrics(y, oof)
            model = fit_full(sc, feats, a.outcome)
            conf = model.predict_proba(ca_df[feats].to_numpy().astype(np.float64))[:, 1]
            m_tr = metrics(ca_df[a.outcome].to_numpy().astype(int), conf)
            oof_store[fs] = oof; tr_store[fs] = conf
            rows.append(dict(featset=fs, split="scope_cv", **m_in))
            rows.append(dict(featset=fs, split="cath_transfer", **m_tr))
            log.info("%-10s scope AUROC=%.4f AUARC=%.4f | cath AUROC=%.4f AUARC=%.4f" %
                     (fs, m_in["auroc"], m_in["auarc"], m_tr["auroc"], m_tr["auarc"]))
        res = pd.DataFrame(rows)
        res.to_csv(RESULTS / ("selective_" + tag + ".csv"), index=False)

        y_sc = sc[a.outcome].to_numpy().astype(int)
        y_ca = ca_df[a.outcome].to_numpy().astype(int)
        boot = {
            "scope_cv": bootstrap_delta(y_sc, oof_store["NN_norm"], oof_store["full"], a.boot),
            "cath_transfer": bootstrap_delta(y_ca, tr_store["NN_norm"], tr_store["full"], a.boot),
        }
        save_json(boot, RESULTS / ("bootstrap_" + tag + ".json"))
        log.info("bootstrap %s" % boot)

        bk = bucketed(sc, y_sc, oof_store["NN_norm"], oof_store["full"], "NN_norm", "full")
        bk.to_csv(RESULTS / ("bucketed_scope_" + tag + ".csv"), index=False)
        bk2 = bucketed(ca_df, y_ca, tr_store["NN_norm"], tr_store["full"], "NN_norm", "full")
        bk2.to_csv(RESULTS / ("bucketed_cath_" + tag + ".csv"), index=False)

        curves = {}
        for fs in FEATSETS:
            curves["scope_" + fs] = auarc(y_sc, oof_store[fs])[1].tolist()
            curves["cath_" + fs] = auarc(y_ca, tr_store[fs])[1].tolist()
        curves["coverage"] = COVGRID.tolist()
        save_json(curves, RESULTS / ("curves_" + tag + ".json"))
    log.info("STAGE4_DONE")
