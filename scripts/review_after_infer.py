import sys, pandas as pd, pathlib as p

arq = p.Path(sys.argv[1] if len(sys.argv) > 1 else r"data\Rel_sensi_035.xlsx")
df  = pd.read_excel(arq)

print("Arquivo:", arq.resolve())
print("Contagem SIM/NÃO:\n", df["sensibilidade a preço"].value_counts(dropna=False))
print("\nEstatísticas 'confianca':\n", df["confianca"].describe(percentiles=[0.5,0.7,0.8,0.9,0.95]))

out_top = p.Path("data/Rel_review_SIM_top200.xlsx")
out_bd  = p.Path("data/Rel_review_borderline_030_045.xlsx")

df[df["sensibilidade a preço"]=="SIM"].sort_values("confianca", ascending=False).head(200).to_excel(out_top, index=False)
df[(df["confianca"]>=0.30) & (df["confianca"]<0.45)].sort_values("confianca", ascending=False).head(300).to_excel(out_bd, index=False)

print(f"\n✅ Salvos: {out_top}  e  {out_bd}")
