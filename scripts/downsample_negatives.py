import sys, pandas as pd, numpy as np, pathlib as p
csv_in  = p.Path(sys.argv[1] if len(sys.argv)>1 else r"data\sensi_preco_dataset_v2.csv")
csv_out = p.Path(sys.argv[2] if len(sys.argv)>2 else r"data\sensi_preco_dataset_bal.csv")
target_pos = float(sys.argv[3]) if len(sys.argv)>3 else 0.20  # 20% SIM
df = pd.read_csv(csv_in, sep=";", encoding="utf-8-sig")
pos = df[df["label"]==1]
neg = df[df["label"]==0]
need_neg = int(max(len(pos)/max(target_pos,1e-6) - len(pos), 0))
neg_sample = neg.sample(min(need_neg, len(neg)), random_state=42)
out = pd.concat([pos, neg_sample]).sample(frac=1.0, random_state=42).reset_index(drop=True)
out.to_csv(csv_out, sep=";", index=False, encoding="utf-8-sig", lineterminator="\r\n")
print(f"✅ {csv_out} | linhas={len(out)} | SIM={len(pos)} | NÃO={len(neg_sample)} | pct_SIM={len(pos)/len(out):.1%}")
