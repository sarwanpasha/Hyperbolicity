"""Collects the four reviewer-requested experiments into one printable record."""
import json, glob, os
import numpy as np
R = os.environ["HYP_ROOT"] + "/results"

def g(pat):
    f = R + "/" + pat
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    t, c = d["test"], d.get("cath", {})
    return dict(drel=t["delta_rel"], sf=t.get("superfam_top1"), fold=t.get("fold_top1"),
                cath=c.get("cath_h_top1"), fine=t.get("fine_top1"),
                coarse=t.get("coarse_top1"), delta=t["delta"])

print("== E1  aggregation: tuned LSE temperature vs top-k rho, 35M lam=10 seed 0 ==")
b = g("run_lora_35M_ft_lh1_s0.json")
print("%-22s %-8s %-8s %-8s" % ("arm", "drel", "SF1", "CATH"))
print("%-22s %-8.4f %-8.4f %-8.4f" % ("task-only", b["drel"], b["sf"], b["cath"]))
for rho in ["0.01", "0.05", "0.1", "0.25", "0.5", "1.0"]:
    for pat in ["run_lora_35M_hyp_lh10_s0_q%s.json" % rho,
                "run_lora_35M_hyp_lh10_s0_topq%s.json" % rho]:
        v = g(pat)
        if v: print("%-22s %-8.4f %-8.4f %-8.4f" % ("topk rho=" + rho, v["drel"], v["sf"], v["cath"]))
for t in ["0.005", "0.01", "0.02", "0.05", "0.15", "0.5"]:
    v = g("run_lora_35M_hyp_lh10_s0_lse%s.json" % t)
    if v: print("%-22s %-8.4f %-8.4f %-8.4f" % ("lse  tau=" + t, v["drel"], v["sf"], v["cath"]))
v = g("run_lora_35M_hyp_lh10_s0_lse.json")
if v: print("%-22s %-8.4f %-8.4f %-8.4f" % ("lse  default", v["drel"], v["sf"], v["cath"]))

print()
print("== E2a  mechanism: SupCon temperature, 35M task-only seed 0 ==")
print("%-22s %-8s %-8s %-8s" % ("tau_supcon", "drel", "SF1", "CATH"))
for t in ["0.03", "0.05", "0.07", "0.2", "0.5"]:
    v = g("run_lora_35M_ft_lh1_s0_tau%s.json" % t)
    if v: print("%-22s %-8.4f %-8.4f %-8.4f" % (t, v["drel"], v["sf"], v["cath"]))
print("%-22s %-8.4f %-8.4f %-8.4f" % ("0.1 (default)", b["drel"], b["sf"], b["cath"]))

print()
print("== E2b  mechanism: fold-blocked batches, 35M ==")
print("%-22s %-8s %-8s %-8s" % ("arm", "drel", "SF1", "CATH"))
for sd in [0, 1, 2]:
    v = g("run_lora_35M_ft_lh1_s%d_foldblk.json" % sd)
    r = g("run_lora_35M_ft_lh1_s%d.json" % sd)
    if v: print("%-22s %-8.4f %-8.4f %-8.4f" % ("foldblock s%d" % sd, v["drel"], v["sf"], v["cath"]))
    if r: print("%-22s %-8.4f %-8.4f %-8.4f" % ("  random s%d" % sd, r["drel"], r["sf"], r["cath"]))
v = g("run_lora_35M_hyp_lh10_s0_foldblkhyp.json")
if v: print("%-22s %-8.4f %-8.4f %-8.4f" % ("foldblock+penalty", v["drel"], v["sf"], v["cath"]))

print()
print("== E3  full fine-tune, no adapters ==")
print("%-22s %-8s %-8s %-8s" % ("arm", "drel", "SF1", "CATH"))
for m in ["35M", "150M"]:
    for tag, lab in [("full", "full FT"), ("fullhyp", "full FT + penalty")]:
        pat = ("run_lora_%s_ft_lh1_s0_%s.json" % (m, tag)) if tag == "full" \
              else ("run_lora_%s_hyp_lh10_s0_%s.json" % (m, tag))
        v = g(pat)
        if v: print("%-22s %-8.4f %-8.4f %-8.4f" % (m + " " + lab, v["drel"], v["sf"], v["cath"]))
    v = g("run_lora_%s_none_lh1_s0.json" % m)
    if v: print("%-22s %-8.4f %-8.4f %-8.4f" % (m + " frozen", v["drel"], v["sf"], v["cath"]))

print()
print("== E4  CIFAR-100: supervise fine, evaluate coarse ==")
print("%-24s %-8s %-8s %-8s %-8s" % ("arm", "drel", "delta", "fine1", "coarse1"))
rows = [("frozen ViT", "run_cifar_none_lh1_s0.json")]
for sd in [0, 1, 2]:
    rows.append(("SupCon s%d" % sd, "run_cifar_ft_lh1_s%d.json" % sd))
for sd in [0, 1, 2]:
    rows.append(("SupCon+pen l=1 s%d" % sd, "run_cifar_hyp_lh1_s%d.json" % sd))
rows.append(("SupCon+pen l=10 s0", "run_cifar_hyp_lh10_s0.json"))
for lab, pat in rows:
    v = g(pat)
    if v: print("%-24s %-8.4f %-8.4f %-8.4f %-8.4f"
                % (lab, v["drel"], v["delta"], v["fine"], v["coarse"]))
print("R5_DONE")
