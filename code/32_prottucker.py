"""Published protein-retrieval baselines evaluated on our galleries.

ProtT5-XL-U50 mean-pooled, and ProtTucker(ProtT5), the contrastive CATH head of
Heinzinger et al. 2022, using the authors' released weights. Same splits, same
leave-one-out retrieval and the same delta estimator as every other arm.
"""
import argparse, re, os
import numpy as np, pandas as pd, torch
from transformers import T5EncoderModel, T5Tokenizer
from common import PROC, RESULTS, HF_CACHE, get_logger, save_json
from ftlib import retrieval_metrics, measure_delta

log = get_logger("ptuck")
MODEL = "Rostlab/prot_t5_xl_half_uniref50-enc"
CKPT = os.environ["HYP_DATA"] + "/ckpt/ProtTucker_ProtT5.pt"

class Tucker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tucker = torch.nn.Sequential(
            torch.nn.Linear(1024, 256), torch.nn.Tanh(), torch.nn.Linear(256, 128))
    def single_pass(self, x):
        return self.tucker(x.float())

@torch.no_grad()
def embed(seqs, tok, enc, dev, maxlen, bs=8):
    order = np.argsort([-len(s) for s in seqs])
    out = np.zeros((len(seqs), 1024), dtype=np.float32)
    i = 0
    while i < len(order):
        idx = order[i:i + bs]
        raw = [re.sub(r"[UZOB]", "X", seqs[j][:maxlen]) for j in idx]
        sp = [" ".join(list(s)) for s in raw]
        b = tok(sp, add_special_tokens=True, padding="longest",
                                  return_tensors="pt").to(dev)
        h = enc(input_ids=b.input_ids, attention_mask=b.attention_mask).last_hidden_state
        for k, j in enumerate(idx):
            L = len(raw[k])
            out[j] = h[k, :L].float().mean(0).cpu().numpy()
        i += bs
        if (i // bs) % 200 == 0:
            log.info("embed %d/%d" % (i, len(order)))
    return out

def norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxlen", type=int, default=1022)
    a = ap.parse_args()
    dev = "cuda"
    tok = T5Tokenizer.from_pretrained(MODEL, do_lower_case=False, legacy=True)
    enc = T5EncoderModel.from_pretrained(MODEL).to(dev).eval().half()
    head = Tucker()
    sd = torch.load(CKPT, map_location="cpu")
    head.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
    head = head.to(dev).eval()

    sp = pd.read_parquet(PROC / "scope_splits.parquet")
    te = sp[sp.split == "test"].reset_index(drop=True)
    cath = pd.read_parquet(PROC / "cath.parquet")
    log.info("test=%d cath=%d" % (len(te), len(cath)))

    Rt = embed(list(te.seq.to_numpy()), tok, enc, dev, a.maxlen)
    log.info("scope embedded")
    Rc = embed(list(cath.seq.to_numpy()), tok, enc, dev, a.maxlen)
    log.info("cath embedded")

    for name, fn in (("prott5", lambda X: norm(X)),
                     ("prottucker", lambda X: norm(head.single_pass(
                         torch.from_numpy(X).to(dev)).detach().cpu().numpy()))):
        Zt, Zc = fn(Rt), fn(Rc)
        res = {}
        r = {}
        for lb in ("superfam", "fold"):
            t1, t5, a1, a5 = retrieval_metrics(Zt, te[lb].to_numpy(), device=dev,
                                               return_arrays=True)
            r[lb + "_top1"], r[lb + "_top5"] = t1, t5
            np.savez_compressed(RESULTS / ("perq_%s_test_%s.npz" % (name, lb)), a1=a1)
        d_abs, d_rel = measure_delta(Zt)
        r.update(delta=d_abs, delta_rel=d_rel, n=len(te))
        res["test"] = r
        rc = {}
        for lb in ("cath_h", "cath_t"):
            t1, t5, a1, a5 = retrieval_metrics(Zc, cath[lb].to_numpy(), device=dev,
                                               return_arrays=True)
            rc[lb + "_top1"], rc[lb + "_top5"] = t1, t5
            np.savez_compressed(RESULTS / ("perq_%s_%s.npz" % (name, lb)), a1=a1)
        d_abs, d_rel = measure_delta(Zc)
        rc.update(delta=d_abs, delta_rel=d_rel, n=len(cath))
        res["cath"] = rc
        save_json(res, RESULTS / ("run_%s.json" % name))
        log.info("[%s] test=%s cath=%s" % (name, res["test"], res["cath"]))
    log.info("PTUCK_DONE")
