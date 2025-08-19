# app/inferir_preco_excel.py
# -----------------------------------------------------------
# Inference local no Excel (validação/predição):
# Lê um .xlsx com coluna "Resumo" e adiciona:
#   - "sensibilidade a preço" (SIM/NÃO)
#   - "confianca" (probabilidade da classe SIM)
# Salva arquivo *_sensi_preco.xlsx ao lado do original.
# -----------------------------------------------------------

import argparse
from pathlib import Path
import sys
import unicodedata
import pandas as pd
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# Tenta importar utilidades de léxico do projeto; cai no lexico.py da raiz se necessário
try:
    from app.utils.lexico import load_lexico, mark_price_spans  # type: ignore
except Exception:
    try:
        from lexico import load_lexico, mark_price_spans  # type: ignore
    except Exception:
        load_lexico = None
        mark_price_spans = None

MODEL_DIR_DEFAULT = Path("app/model/bert-sensi-preco")
LEXICO_PATH_DEFAULT = Path("lexico/preco.json")

# ---------------------------
# Utilidades auxiliares
# ---------------------------

def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

def _norm_upper(s: str) -> str:
    return _strip_accents(str(s)).upper().strip()

def _discover_sim_index(model) -> int:
    """
    Descobre de forma robusta qual índice corresponde ao rótulo 'SIM'.
    Retorna 1 por padrão (binário padrão).
    """
    sim_idx = 1
    try:
        # 1) label2id explícito no config
        l2i = getattr(model.config, "label2id", None)
        if isinstance(l2i, dict) and len(l2i) > 0:
            norm_keys = { _norm_upper(k): v for k, v in l2i.items() }
            if "SIM" in norm_keys:
                return int(norm_keys["SIM"])
        # 2) id2label explícito no config
        i2l = getattr(model.config, "id2label", None)
        if isinstance(i2l, dict) and len(i2l) > 0:
            for k, v in i2l.items():
                if _norm_upper(v) == "SIM":
                    return int(k)
        # 3) fallback: 1
    except Exception:
        pass
    return sim_idx

def _mark_texts_with_lexicon(texts, lexico):
    """Aplica marcação [PRICE]...[/PRICE] se mark_price_spans estiver disponível."""
    if mark_price_spans is None or lexico is None:
        # Sem utilitário: retorna os textos como vieram
        return texts
    out = []
    for t in texts:
        t = "" if pd.isna(t) else str(t)
        out.append(mark_price_spans(t, lexico))
    return out

# ---------------------------
# Inferência em lote
# ---------------------------

def inferir_batch(textos, tokenizer, model, device, max_len=256, sim_idx=1):
    inputs = tokenizer(
        textos,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_len
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = softmax(logits, dim=1).cpu().numpy()
    confs_sim = probs[:, sim_idx]
    return confs_sim  # retornamos só a confiança de SIM (0..1)

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Classifica 'sensibilidade a preço' em um Excel adicionando colunas 'sensibilidade a preço' e 'confianca'."
    )
    ap.add_argument("--arquivo", "-f", required=True, help="Caminho do .xlsx de entrada.")
    ap.add_argument("--coluna", "-c", default="Resumo", help='Nome da coluna de texto (padrão: "Resumo").')
    ap.add_argument("--modelo", "-m", default=str(MODEL_DIR_DEFAULT), help=f"Diretório do modelo HF local (padrão: {MODEL_DIR_DEFAULT}).")
    ap.add_argument("--saida", "-o", default=None, help="Caminho do .xlsx de saída (padrão: <arquivo>_sensi_preco.xlsx).")
    ap.add_argument("--sheet", "-s", default=None, help="Nome da planilha (aba). Se omitido, usa a primeira.")
    ap.add_argument("--batch", "-b", type=int, default=64, help="Tamanho do batch na inferência (padrão: 64).")
    ap.add_argument("--max_len", type=int, default=256, help="Comprimento máximo de tokens (padrão: 256).")
    ap.add_argument("--limiar", type=float, default=0.60, help="Prob mínima para marcar SIM (padrão: 0.60).")
    ap.add_argument("--minusculas", action="store_true", help="Força o texto para minúsculas antes de inferir.")
    ap.add_argument("--lexico", default=str(LEXICO_PATH_DEFAULT), help=f"Caminho do léxico JSON (padrão: {LEXICO_PATH_DEFAULT}). Use --lexico '' para desativar.")
    args = ap.parse_args()

    xlsx_path = Path(args.arquivo)
    if not xlsx_path.exists():
        print(f"❌ Arquivo não encontrado: {xlsx_path}", file=sys.stderr)
        sys.exit(1)

    # Carrega DataFrame
    try:
        if args.sheet:
            df = pd.read_excel(xlsx_path, sheet_name=args.sheet)
        else:
            # Carrega primeira aba por padrão
            df = pd.read_excel(xlsx_path)
    except Exception as e:
        print(f"❌ Erro lendo Excel: {e}", file=sys.stderr)
        sys.exit(1)

    if args.coluna not in df.columns:
        print(f"❌ Coluna '{args.coluna}' não encontrada no arquivo. Colunas disponíveis: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    textos = df[args.coluna].astype(str).fillna("").tolist()
    if args.minusculas:
        textos = [t.lower() for t in textos]

    # Carrega léxico (opcional)
    lexico_obj = None
    if args.lexico and Path(args.lexico).exists():
        if load_lexico is None:
            print("ℹ️ Aviso: utilitário load_lexico indisponível; seguindo sem marcação léxica.")
        else:
            try:
                lexico_obj = load_lexico(Path(args.lexico))
            except Exception as e:
                print(f"ℹ️ Aviso: falha ao carregar léxico ({e}); seguindo sem marcação léxica.")
    elif args.lexico:
        print(f"ℹ️ Aviso: léxico não encontrado em {args.lexico}; seguindo sem marcação léxica.")

    # Marca spans [PRICE]...[/PRICE] (se disponível)
    if lexico_obj is not None and mark_price_spans is not None:
        textos_marked = _mark_texts_with_lexicon(textos, lexico_obj)
    else:
        textos_marked = textos

    # Carrega modelo/tokenizer locais
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.modelo, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(args.modelo, local_files_only=True).to(device)
    except Exception as e:
        print(f"❌ Erro carregando modelo/tokenizer em '{args.modelo}': {e}", file=sys.stderr)
        sys.exit(1)
    model.eval()

    # Determina índice da classe SIM de forma robusta
    sim_idx = _discover_sim_index(model)
    num_labels = getattr(model.config, "num_labels", None)
    if num_labels is not None and (sim_idx < 0 or sim_idx >= int(num_labels)):
        print(f"ℹ️ Aviso: índice de SIM ({sim_idx}) fora do intervalo [0, {num_labels-1}]. Usando fallback 1.")
        sim_idx = 1

    # Inferência em lotes
    confs_all = []
    BATCH = max(1, int(args.batch))
    for i in tqdm(range(0, len(textos_marked), BATCH), desc="🔎 Inferindo"):
        batch = textos_marked[i:i+BATCH]
        confs_sim = inferir_batch(
            batch,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_len=args.max_len,
            sim_idx=sim_idx
        )
        confs_all.extend(confs_sim.tolist())

    # Decisão por limiar
    df["confianca"] = [round(float(c), 4) for c in confs_all]
    df["sensibilidade a preço"] = ["SIM" if c >= float(args.limiar) else "NÃO" for c in confs_all]

    # Salva arquivo
    out_path = Path(args.saida) if args.saida else xlsx_path.with_name(xlsx_path.stem + "_sensi_preco.xlsx")
    try:
        df.to_excel(out_path, index=False)
    except Exception as e:
        print(f"❌ Erro ao salvar Excel de saída: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"✅ Resultado salvo em: {out_path.resolve()}")
    print(f"ℹ️ Limiar usado: {args.limiar} | Classe SIM @ index: {sim_idx} | Device: {device}")

if __name__ == "__main__":
    main()
