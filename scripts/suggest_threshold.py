import sys, pandas as pd

# uso: python scripts/suggest_threshold.py data\Rel_sensi_035.xlsx 0.05
arq = sys.argv[1] if len(sys.argv) > 1 else r"data\Rel_sensi_035.xlsx"
target_sim = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05  # 5% por padrão

df = pd.read_excel(arq)
conf = df["confianca"].astype(float)
thr  = conf.quantile(1 - target_sim)  # top-X% por confiança
sim_at = (conf >= thr).mean()

print(f"Arquivo: {arq}")
print(f"Alvo de SIM: {target_sim*100:.1f}%")
print(f"Sugestão de limiar: {thr:.3f}")
print(f"Com esse limiar, SIM estimado: {sim_at*100:.2f}%")
