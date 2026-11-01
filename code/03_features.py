"""Stage 3: retrieval outcomes + local geometry features per query domain.

For a chosen ESM-2 layer we do leave-one-out nearest-neighbour retrieval inside a
dataset, record whether the retrieved neighbour shares the target label
(superfamily / fold for SCOPe, homologous-superfamily / topology for CATH), and
compute local Gromov delta statistics of each query's neighbourhood.
Output: RESULTS/feats_<dataset>_<model>_<level>.parquet
"""
import argparse, json
import numpy as np
import pandas as pd
import torch
from common import PROC, EMB, RESULTS, ESM_MODELS, get_logger, Timer, save_json
from geom import delta_from_D, pdist_normalized, ultrametricity

log = get_logger("feats")

LEVELS = {
    "scope": {"superfam": "superfam", "fold": "fold", "family": "family"},
    "cath": {"superfam": "cath_h", "fold": "cath_t", "family": "cath_h"},
}


def load(dataset, model):
    Z = np.load(EMB / (dataset + "_" + model + ".npy"), mmap_mode="r")
    ids = np.load(EMB / (dataset + "_" + model + "_ids.npy"), allow_pickle=True)
    df = pd.read_parquet(PROC / (dataset + ".parquet"))
    idcol = "sid" if dataset == "scope" else "did"
    df = df.set_index(idcol).loc[ids].reset_index()
    return Z, ids, df


def topk_retrieval(X, K, device, chunk=1024):
    """X: (N,D) float32 torch on cpu (already L2-normalised). Returns sims,idx (N,K)
    excluding self."""
    N = X.shape[0]
    Xg = X.to(device)
    sims = torch.zeros(N, K); idxs = torch.zeros(N, K, dtype=torch.long)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        S = Xg[s:e] @ Xg.T
        S[torch.arange(e - s, device=device), torch.arange(s, e, device=device)] = -2.0
        v, i = torch.topk(S, K, dim=1)
        sims[s:e] = v.cpu(); idxs[s:e] = i.cpu()
        del S
    return sims.numpy(), idxs.numpy()


def layer_scan(Z, labels, device, K=5, sub=4000, seed=0):
    """Top-1 retrieval accuracy per layer on a random subsample (for layer choice)."""
    rng = np.random.default_rng(seed)
    N = Z.shape[1]
    sel = np.sort(rng.choice(N, size=min(sub, N), replace=False))
    lab = labels[sel]
    accs = []
    for li in range(Z.shape[0]):
        X = torch.tensor(np.asarray(Z[li][sel], dtype=np.float32))
        X = torch.nn.functional.normalize(X, dim=1)
        _, idx = topk_retrieval(X, K, device)
        a1 = float(np.mean(lab[idx[:, 0]] == lab))
        accs.append(a1)
        log.info("layer %2d top1=%.4f" % (li, a1))
    return np.array(accs), sel


def build_features(Z, df, dataset, level, layer, device, K=5, M=32, batch=256):
    labcol = LEVELS[dataset][level]
    labels = df[labcol].to_numpy()
    Xraw = np.asarray(Z[layer], dtype=np.float32)
    qnorm = np.linalg.norm(Xraw, axis=1)
    X = torch.tensor(Xraw)
    X = torch.nn.functional.normalize(X, dim=1)
    KK = max(K, M)
    sims, idx = topk_retrieval(X, KK, device)
    log.info("retrieval done, K=%d" % KK)

    nbr_lab = labels[idx[:, :K]]
    y_top1 = (nbr_lab[:, 0] == labels).astype(np.int8)
    y_top5 = (nbr_lab == labels[:, None]).any(axis=1).astype(np.int8)

    Xn = X.numpy()
    N = Xn.shape[0]
    delta = np.zeros(N, np.float32); drel = np.zeros(N, np.float32)
    diam = np.zeros(N, np.float32); umet = np.zeros(N, np.float32)
    for s in range(0, N, batch):
        e = min(s + batch, N)
        nb = np.concatenate([np.arange(s, e)[:, None], idx[s:e, :M]], axis=1)
        P = Xn[nb]                                  # (b, M+1, d)
        D = pdist_normalized(P)
        d = delta_from_D(D, base=0)
        dm = D.max(axis=(1, 2))
        delta[s:e] = d
        diam[s:e] = dm
        drel[s:e] = 2.0 * d / np.clip(dm, 1e-6, None)
        umet[s:e] = ultrametricity(D)
        if s % 4096 == 0:
            log.info("delta %d/%d" % (s, N))

    out = pd.DataFrame({
        "id": df.iloc[:, 0].to_numpy(),
        "label": labels,
        "length": df.length.to_numpy(),
        "y_top1": y_top1,
        "y_top5": y_top5,
        "s1": sims[:, 0],
        "s2": sims[:, 1],
        "margin": sims[:, 0] - sims[:, 1],
        "s_mean": sims[:, :K].mean(1),
        "s_std": sims[:, :K].std(1),
        "qnorm": qnorm,
        "delta": delta,
        "delta_rel": drel,
        "diam": diam,
        "umet": umet,
    })
    if dataset == "scope":
        out["cls"] = df.cls.to_numpy()
        out["fold"] = df.fold.to_numpy()
        out["superfam"] = df.superfam.to_numpy()
    else:
        out["cls"] = df.cath_c.to_numpy()
        out["fold"] = df.cath_t.to_numpy()
        out["superfam"] = df.cath_h.to_numpy()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="150M", choices=list(ESM_MODELS))
    ap.add_argument("--dataset", default="scope", choices=["scope", "cath"])
    ap.add_argument("--level", default="superfam", choices=["superfam", "fold", "family"])
    ap.add_argument("--layer", type=int, default=-1, help="-1 = pick by layer scan")
    ap.add_argument("--M", type=int, default=32)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("device=%s" % device)
    with Timer(log, "stage 3"):
        Z, ids, df = load(a.dataset, a.model)
        labels = df[LEVELS[a.dataset][a.level]].to_numpy()
        layer = a.layer
        scanfile = RESULTS / ("layerscan_" + a.dataset + "_" + a.model + "_" + a.level + ".json")
        if layer < 0:
            accs, _ = layer_scan(Z, labels, device)
            layer = int(np.argmax(accs))
            save_json({"accs": accs.tolist(), "best": layer}, scanfile)
        log.info("using layer %d" % layer)
        feats = build_features(Z, df, a.dataset, a.level, layer, device, M=a.M)
        fn = RESULTS / ("feats_" + a.dataset + "_" + a.model + "_" + a.level + ".parquet")
        feats.to_parquet(fn, index=False)
        log.info("wrote %s  base_top1=%.4f base_top5=%.4f" %
                 (fn, feats.y_top1.mean(), feats.y_top5.mean()))
    log.info("STAGE3_DONE")
