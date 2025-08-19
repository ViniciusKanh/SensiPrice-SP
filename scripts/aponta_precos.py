import pandas as pd
from pathlib import Path
try:
    from app.utils.lexico import load_lexico, weak_label_price_sensitive, mark_price_spans
except Exception:
    from lexico import load_lexico, weak_label_price_sensitive, mark_price_spans

xlsx = Path("data/Relacionamento_e_NPS.xlsx")
saida = Path("data/Relacionamento_e_NPS_flag.xlsx")
col  = "Resumo"
lex  = load_lexico(Path("lexico/preco.json"))

df = pd.read_excel(xlsx)
if col not in df.columns:
    raise SystemExit(f"Coluna '{col}' não encontrada. Colunas: {list(df.columns)}")

def infer(txt):
    y = weak_label_price_sensitive(str(txt), lex)
    m = mark_price_spans(str(txt), lex)
    return y, m

df["preco_regra"], df["trechos_marcados"] = zip(*df[col].fillna("").map(infer))
df.to_excel(saida, index=False)
print(f"✅ Salvo: {saida.resolve()}  | SIM={int(df['preco_regra'].sum())} de {len(df)}")
