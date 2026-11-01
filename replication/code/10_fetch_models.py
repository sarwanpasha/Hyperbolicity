from transformers import AutoModel, AutoTokenizer
for n in ["facebook/esm2_t12_35M_UR50D", "facebook/esm2_t30_150M_UR50D",
          "facebook/esm2_t33_650M_UR50D", "facebook/esm2_t36_3B_UR50D"]:
    AutoTokenizer.from_pretrained(n)
    AutoModel.from_pretrained(n)
    print("FETCHED", n, flush=True)
