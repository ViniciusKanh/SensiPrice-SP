import pandas as pd
from pathlib import Path

src = Path("data/Relacionamento_e_NPS.xlsx")
dst = src.with_name(src.stem + "_lower.xlsx")

df = pd.read_excel(src)
col = "Resumo"
if col in df.columns:
    df[col] = df[col].astype(str).fillna("").str.lower()

df.to_excel(dst, index=False)
print("✅ Gerado:", dst.resolve())
