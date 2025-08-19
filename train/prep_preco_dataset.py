
# train/prep_preco_dataset.py
# -----------------------------------------------------------
# Gera dataset para "sensibilidade a preço".
# MODOS:
#  (A) --from_xlsx (padrão): lê Excel/Resumo, rotula com léxico (rótulo fraco).
#  (B) --synthetic: gera dataset SÓ A PARTIR DO LÉXICO (sem Excel).
#
# Saída: data/sensi_preco_dataset.csv com colunas: texto,label
# (UTF-8 BOM, ; como separador, CRLF, aspas em tudo)
# -----------------------------------------------------------

import os
import sys
import re
import csv
import json
import random
import argparse
import unicodedata as ud
from pathlib import Path
import pandas as pd

# utilitários de léxico (rotulador e marcação)
try:
    from lexico import (
        load_lexico,
        load_lexico_sintetico,
        weak_label_price_sensitive,
        mark_price_spans,
    )  # type: ignore
except Exception as e:
    print("❌ Import de 'lexico.py' falhou:", e, file=sys.stderr)
    raise

RAW_XLSX_DEFAULT = Path("data/Relacionamento_e_NPS.xlsx")
OUT_CSV_DEFAULT  = Path("data/sensi_preco_dataset.csv")
LEXICO_DEFAULT   = Path("lexico/preco.json")

# ------------------------------
# Utilidades comuns
# ------------------------------
def _strip_accents(s: str) -> str:
    if s is None:
        return ""
    return ud.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

def _slug(s: str) -> str:
    s = "" if s is None else str(s)
    s = _strip_accents(s).lower()
    s = re.sub(r"\s+", "", s)
    return s

def _clean_text(s: str, remove_accents: bool = False) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = re.sub(r"_x000D_\s*", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"[\r\n]+", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    s = re.sub(r'^\s*(ol[áa],?\s*)?(car[oa]s?|prezad[oa]s?)\s*[,:]\s*', "", s, flags=re.IGNORECASE)
    if remove_accents:
        s = _strip_accents(s)
    return s

def _find_text_col(df: pd.DataFrame, preferida: str = "Resumo") -> str:
    alvo_slug = _slug(preferida)
    mapa = {col: _slug(col) for col in df.columns}
    for col, s in mapa.items():
        if s == alvo_slug:
            return col
    candidatos = {"observacao", "comentario", "descricao", "texto", "mensagem", "detalhes"}
    for col, s in mapa.items():
        if s in candidatos:
            return col
    raise ValueError(f"Coluna '{preferida}' não encontrada. Colunas disponíveis: {list(df.columns)}")

# ------------------------------
# (B) Gerador sintético a partir do léxico
# ------------------------------
def generate_synthetic_dataset(lexico_path: Path, out_csv: Path, n_pos: int, n_neg: int, seed: int = 42):
    sint = load_lexico_sintetico(lexico_path)

    rng = random.Random(seed)
    hard = list(sint.get("hard_phrases", []))
    soft = list(sint.get("soft_phrases", []))
    bill = list(sint.get("billing_phrases", []))
    macro = list(sint.get("macro_phrases", []))
    negs = list(sint.get("neg_safe_phrases", []))

    pos_templates  = list(sint.get("pos_templates", ["o serviço está [PRICE]{x}[/PRICE]."]))
    soft_templates = list(sint.get("soft_templates", ["estamos com [PRICE]{x}[/PRICE]."]))
    neg_templates  = list(sint.get("neg_templates", ["{x}."]))

    rows = []

    def mk_pos():
        bucket = rng.choice(["HARD","SOFT","BILLING","MACRO"])
        if bucket == "HARD" and hard:
            x = rng.choice(hard)
            tpl = rng.choice(pos_templates)
        elif bucket == "SOFT" and soft:
            x = rng.choice(soft)
            tpl = rng.choice(soft_templates or pos_templates)
        elif bucket == "BILLING" and bill:
            # Reforça intensidade para casar com regra de "billing"
            x = rng.choice(bill)
            # injeta alguma intensidade/reajuste na frase
            intensifiers = ["alta", "elevada", "caro", "com reajuste", "com aumento"]
            x = f"{x} {rng.choice(intensifiers)}"
            tpl = rng.choice(pos_templates)
        else:  # MACRO
            x = rng.choice(macro or ["condições comerciais desfavoráveis"])
            tpl = rng.choice(pos_templates)
        sent = tpl.format(x=x)
        return sent, 1

    def mk_neg():
        x = rng.choice(negs or ["preço justo", "bom custo-benefício"])
        sent = rng.choice(neg_templates).format(x=x)
        return sent, 0

    for _ in range(int(n_pos)):
        rows.append(mk_pos())
    for _ in range(int(n_neg)):
        rows.append(mk_neg())

    rng.shuffle(rows)

    # salva CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";", quoting=csv.QUOTE_ALL, lineterminator="\r\n")
        w.writerow(["texto","label"])
        for texto, label in rows:
            w.writerow([texto, label])

    print(f"✅ Gerado dataset sintético: {out_csv.resolve()}  (pos={n_pos}, neg={n_neg})")

# ------------------------------
# (A) Modo Excel: rotulação fraca
# ------------------------------
def run_from_xlsx(raw_xlsx: Path, out_csv: Path, lexico_p: Path, sheet: str|None, text_col: str, lowercase: bool, remove_accents: bool, min_len: int, limit: int|None):
    if not raw_xlsx.exists():
        raise FileNotFoundError(f"Não encontrei o Excel de entrada: {raw_xlsx.resolve()}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not lexico_p.exists():
        raise FileNotFoundError(f"Léxico não encontrado: {lexico_p.resolve()}")

    lexico = load_lexico(lexico_p)
    df = pd.read_excel(raw_xlsx, sheet_name=sheet) if sheet else pd.read_excel(raw_xlsx)
    col_texto = _find_text_col(df, text_col)

    serie_txt = (
        df[col_texto]
        .fillna("")
        .astype(str)
        .apply(lambda x: _clean_text(x, remove_accents=remove_accents))
    )
    if lowercase:
        serie_txt = serie_txt.str.lower()

    serie_txt = serie_txt[serie_txt.str.len() >= int(min_len)]
    if limit is not None:
        serie_txt = serie_txt.head(int(limit))

    serie_txt = serie_txt.drop_duplicates()

    df2 = pd.DataFrame({"texto_raw": serie_txt.values})
    df2["label"] = df2["texto_raw"].apply(lambda t: weak_label_price_sensitive(t, lexico))
    df2["texto_marcado"] = df2["texto_raw"].apply(lambda t: mark_price_spans(t, lexico))
    out_df = df2[["texto_marcado", "label"]].rename(columns={"texto_marcado": "texto"})

    out_df.to_csv(
        out_csv,
        index=False,
        sep=";",
        encoding="utf-8-sig",
        lineterminator="\r\n",
        quoting=csv.QUOTE_ALL,
    )

    total = int(out_df.shape[0])
    pos = int((out_df["label"] == 1).sum())
    neg = int((out_df["label"] == 0).sum())
    pct_pos = (pos / total * 100) if total else 0.0
    print(f"✅ Dataset salvo em {out_csv.resolve()} — linhas={total}, SIM={pos} ({pct_pos:.1f}%), NÃO={neg}")
    if pct_pos < 10 or pct_pos > 60:
        print("⚠️ Observação: distribuição de SIM está desbalanceada. Ajuste o léxico/exclusões.")

# ------------------------------
# CLI
# ------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Gera CSV (texto,label) para sensibilidade a preço. Use --synthetic para treinar SÓ com o dicionário."
    )
    # modos
    ap.add_argument("--synthetic", action="store_true", help="Gera dataset 100% sintético a partir do léxico (sem Excel).")
    # comuns
    ap.add_argument("--out", "-o", type=str, default=str(OUT_CSV_DEFAULT), help=f"Caminho do CSV de saída (padrão: {OUT_CSV_DEFAULT})")
    ap.add_argument("--lexico", type=str, default=str(LEXICO_DEFAULT), help=f"Caminho do léxico JSON (padrão: {LEXICO_DEFAULT})")
    # Excel
    ap.add_argument("--input", "-i", type=str, default=str(RAW_XLSX_DEFAULT), help=f"(modo Excel) Caminho do Excel de entrada")
    ap.add_argument("--sheet", "-s", type=str, default=None, help="(modo Excel) Nome da planilha (aba).")
    ap.add_argument("--text_col", "-c", type=str, default="Resumo", help="(modo Excel) Nome preferido da coluna de texto.")
    ap.add_argument("--lowercase", action="store_true", help="(modo Excel) Força o texto para minúsculas após limpeza.")
    ap.add_argument("--remove_accents", action="store_true", help="(modo Excel) Remove acentos do texto após limpeza.")
    ap.add_argument("--min_len", type=int, default=1, help="(modo Excel) Descarta linhas com texto final menor que N caracteres.")
    ap.add_argument("--limit", type=int, default=None, help="(modo Excel) Processa no máximo N linhas (debug).")
    # Sintético
    ap.add_argument("--n_pos", type=int, default=None, help="(modo sintético) Quantidade de exemplos positivos.")
    ap.add_argument("--n_neg", type=int, default=None, help="(modo sintético) Quantidade de exemplos negativos.")
    ap.add_argument("--seed", type=int, default=None, help="(modo sintético) Seed para aleatoriedade.")

    args = ap.parse_args()

    out_csv  = Path(args.out)
    lexico_p = Path(args.lexico)

    if args.synthetic:
        # parâmetros padrão vindos do próprio JSON (se não informados)
        cfg = {}
        try:
            raw = json.loads(Path(lexico_p).read_text(encoding="utf-8"))
            cfg = dict(raw.get("sintetico", {}))
        except Exception:
            cfg = {}
        n_pos = args.n_pos if args.n_pos is not None else int(cfg.get("samples", {}).get("n_pos", 1000))
        n_neg = args.n_neg if args.n_neg is not None else int(cfg.get("samples", {}).get("n_neg", 1000))
        seed  = args.seed  if args.seed  is not None else int(cfg.get("samples", {}).get("seed", 42))
        generate_synthetic_dataset(lexico_p, out_csv, n_pos=n_pos, n_neg=n_neg, seed=seed)
    else:
        run_from_xlsx(
            raw_xlsx=Path(args.input),
            out_csv=out_csv,
            lexico_p=lexico_p,
            sheet=args.sheet,
            text_col=args.text_col,
            lowercase=args.lowercase,
            remove_accents=args.remove_accents,
            min_len=args.min_len,
            limit=args.limit,
        )

if __name__ == "__main__":
    main()
