
# app/utils/lexico.py
# -----------------------------------------------------------
# Utilitários de léxico para "sensibilidade a preço"
# - Carrega e compila padrões do JSON (hard/soft/billing/macro)
# - Rotulador fraco (0=NAO, 1=SIM) com janela de negação local
# - Marcação de spans relevantes com [PRICE]...[/PRICE]
# - (NOVO) load_lexico_sintetico: lê blocos de geração sintética
# -----------------------------------------------------------

from pathlib import Path
import json
import re
import os
import unicodedata
from typing import Dict, List, Tuple, Any, Pattern, Set

__all__ = [
    "load_lexico",
    "weak_label_price_sensitive",
    "mark_price_spans",
    "load_lexico_sintetico",
]

# ------------------------------
# Normalização e regex helpers
# ------------------------------

def _strip_accents(s: str) -> str:
    """Remove acentos sem alterar demais caracteres (para comparação)."""
    s = unicodedata.normalize("NFD", s or "")
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

def _norm(s: str) -> str:
    """Normaliza para minúsculas sem acento (não usar para offsets)."""
    return _strip_accents(str(s).lower()).strip()

def _compile_regex(rx: str) -> Pattern:
    """
    Compila padrões do léxico. Aceita:
      - 'r:...'  => expressão regular
      - caso contrário => literal (escapado)
    """
    if rx.startswith("r:"):
        return re.compile(rx[2:], re.IGNORECASE)
    return re.compile(re.escape(rx), re.IGNORECASE)

def _split_sentences(s: str) -> List[Tuple[int, int, str]]:
    """
    Divide o texto em sentenças simples preservando offsets (a,b,conteúdo).
    Usa pontuação comum (. ! ? ;) e quebras de linha como delimitadores.
    """
    spans: List[Tuple[int, int, str]] = []
    for m in re.finditer(r"[^\.!\?\n;]+[\.!\?;]?", s):
        a, b = m.span()
        spans.append((a, b, s[a:b]))
    return spans or [(0, len(s), s)]

# ------------------------------
# Carregamento do léxico
# ------------------------------

def load_lexico(path: Path) -> Dict[str, Any]:
    """
    Lê o JSON do léxico e retorna estrutura compilada:
      {
        'positivos': {
           'hard':   [regex, ...],
           'soft_budget': [...],
           'billing': [...],
           'macro':  [...]
        },
        'negacoes': set(normalizado),
        'exclusoes': [regex, ...],
        'janela_negacao_palavras': int,
        'intensity_adj_re': regex
      }
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    # Compat: se "positivos" for lista, trate como "hard"
    pos_raw = data.get("positivos", {})
    if isinstance(pos_raw, list):
        pos_raw = {"hard": pos_raw}

    positivos: Dict[str, List[Pattern]] = {}
    for cat, arr in pos_raw.items():
        positivos[cat] = [_compile_regex(p) for p in (arr or [])]

    comp: Dict[str, Any] = {
        "positivos": positivos,
        "negacoes": set(_norm(x) for x in data.get("negacoes", [])),
        "exclusoes": [_compile_regex(r) for r in data.get("exclusoes", [])],
        "janela_negacao_palavras": int(data.get("janela_negacao_palavras", 6)),
    }

    # Adjetivos de intensidade e termos correlatos; ajudam a validar "billing"
    comp["intensity_adj_re"] = re.compile(
        r"\b(alto(s)?|alta(s)?|elevad[oa]s?|car[oa]s?|carissim[oa]s?|salgad[oa]s?|baix[oa]s?)\b",
        re.IGNORECASE
    )

    # Guardar também o "raw" para acesso à seção 'sintetico'
    comp["_raw_json"] = data
    return comp

def load_lexico_sintetico(path: Path) -> Dict[str, Any]:
    """Retorna o bloco 'sintetico' do JSON (com defaults sensatos)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    sint = dict(data.get("sintetico", {}))

    # defaults
    sint.setdefault("hard_phrases", [])
    sint.setdefault("soft_phrases", [])
    sint.setdefault("billing_phrases", [])
    sint.setdefault("macro_phrases", [])
    sint.setdefault("neg_safe_phrases", [])
    sint.setdefault("pos_templates", ["[PRICE]{x}[/PRICE]."])
    sint.setdefault("soft_templates", ["[PRICE]{x}[/PRICE]."])
    sint.setdefault("neg_templates", ["{x}."])
    sint.setdefault("samples", {"n_pos": 1000, "n_neg": 1000, "seed": 42})
    return sint

# ------------------------------
# Negação local
# ------------------------------

def _has_local_negation(
    text_original: str,
    match_span: Tuple[int, int],
    neg_set: Set[str],
    window_words: int
) -> bool:
    """
    Verifica se há negação "próxima" ao gatilho (na MESMA sentença),
    dentro de uma janela de N palavras à esquerda/direita.
    Importante: usa TEXTO ORIGINAL para offsets (match_span).
    Apenas as PALAVRAS comparadas são normalizadas (minúsculas, sem acento).
    """
    a, b = match_span
    # Identifica a sentença onde está o match
    for sa, sb, sent in _split_sentences(text_original):
        if sa <= a < sb:
            left = text_original[sa:a]
            right = text_original[b:sb]

            def words_norm(t: str) -> List[str]:
                return re.findall(r"\b\w+\b", _norm(t))

            lw, rw = words_norm(left), words_norm(right)

            # Procura termos de negação até 'window_words' em ambas direções
            max_w = int(max(0, window_words))
            for i in range(1, max_w + 1):
                if i <= len(lw) and lw[-i] in neg_set:
                    return True
                if i <= len(rw) and rw[i - 1] in neg_set:
                    return True
            return False
    # Se não localizou sentença, seja conservador: sem negação detectada
    return False

# ------------------------------
# Rotulador fraco
# ------------------------------

def weak_label_price_sensitive(text: str, lexico: Dict[str, Any]) -> int:
    """
    Regras de rótulo fraco (0/1) para 'sensibilidade a preço':

    1) Se houver gatilho 'hard' NAO NEGADO => SIM (1)
    2) Caso contrário, soma de evidências:
       - soft_budget: +1 por ocorrência válida (não negada)
       - macro:       +1 por ocorrência válida (não negada)
       - billing:     +1 SE houver adjetivo de intensidade ou termos de
                      'reajuste/aumento' na MESMA sentença do gatilho
       => SIM se score >= 2
    3) Qualquer 'exclusão' global derruba para NÃO (0).
    """
    if not text:
        return 0

    t = str(text)

    # Exclusões globais
    for rx in lexico.get("exclusoes", []):
        if rx.search(t):
            return 0

    window = int(lexico.get("janela_negacao_palavras", 6))
    pos: Dict[str, List[Pattern]] = lexico.get("positivos", {})
    intensity_adj_re: Pattern = lexico["intensity_adj_re"]
    negs: Set[str] = lexico.get("negacoes", set())

    def _valid_hits(cat: str) -> List[Tuple[int, int, re.Match]]:
        hits: List[Tuple[int, int, re.Match]] = []
        for rx in pos.get(cat, []):
            for m in rx.finditer(t):
                if not _has_local_negation(t, m.span(), negs, window):
                    hits.append((m.start(), m.end(), m))
        return hits

    # 1) hard
    if _valid_hits("hard"):
        return 1

    # 2) soft/macro/billing combinados
    soft = len(_valid_hits("soft_budget"))
    macro = len(_valid_hits("macro"))

    # billing exige intensidade na MESMA sentença
    billing_hits = 0
    for a, b, _ in _valid_hits("billing"):
        for sa, sb, sent in _split_sentences(t):
            if sa <= a < sb:
                if (intensity_adj_re.search(sent)
                    or re.search(r"\breajust", sent, re.IGNORECASE)
                    or re.search(r"\baument", sent, re.IGNORECASE)):
                    billing_hits += 1
                break

    score = soft + macro + (1 if billing_hits > 0 else 0)
    min_score = int(os.environ.get("WEAK_MIN_SCORE", "2"))
    return 1 if score >= min_score else 0

# ------------------------------
# Marcação de spans
# ------------------------------

def mark_price_spans(text: str, lexico: Dict[str, Any]) -> str:
    """
    Marca spans que contribuem para a decisão com [PRICE]...[/PRICE].
    Política simples:
      - Sempre marca todos 'hard' (não negados);
      - Marca 'soft_budget' (útil para auditoria humana);
      - Marca 'billing' SOMENTE se houver intensidade na MESMA sentença;
      - 'macro' não é marcado (normalmente demasiado genérico).
    """
    if not text:
        return ""

    t = str(text)
    pieces: List[Tuple[int, int]] = []

    window = int(lexico.get("janela_negacao_palavras", 6))
    pos: Dict[str, List[Pattern]] = lexico.get("positivos", {})
    negs: Set[str] = lexico.get("negacoes", set())
    intensity_adj_re: Pattern = lexico["intensity_adj_re"]

    def collect(cat: str):
        for rx in pos.get(cat, []):
            for m in rx.finditer(t):
                if not _has_local_negation(t, m.span(), negs, window):
                    pieces.append((m.start(), m.end()))

    # 'hard' sempre
    collect("hard")
    # 'soft_budget' auxilia auditoria
    collect("soft_budget")

    # 'billing' somente com intensidade na MESMA sentença
    for rx in pos.get("billing", []):
        for m in rx.finditer(t):
            a, b = m.span()
            if _has_local_negation(t, (a, b), negs, window):
                continue
            for sa, sb, sent in _split_sentences(t):
                if sa <= a < sb and (
                    intensity_adj_re.search(sent)
                    or re.search(r"\breajust", sent, re.IGNORECASE)
                    or re.search(r"\baument", sent, re.IGNORECASE)
                ):
                    pieces.append((a, b))
                    break

    if not pieces:
        return t

    # Fundir intervalos sobrepostos/adjacentes
    pieces.sort()
    merged: List[Tuple[int, int]] = []
    cur_a, cur_b = pieces[0]
    for a, b in pieces[1:]:
        if a <= cur_b:
            cur_b = max(cur_b, b)
        else:
            merged.append((cur_a, cur_b))
            cur_a, cur_b = a, b
    merged.append((cur_a, cur_b))

    # Inserir tags do fim pro começo para não quebrar offsets
    out = t
    for a, b in reversed(merged):
        out = out[:b] + "[/PRICE]" + out[b:]
        out = out[:a] + "[PRICE]" + out[a:]
    return out

# ------------------------------
# CLI de teste
# ------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        lex_path = Path(sys.argv[1])
        txt = " ".join(sys.argv[2:])
        lex = load_lexico(lex_path)
        y = weak_label_price_sensitive(txt, lex)
        print("Label SIM?" , y)
        print(mark_price_spans(txt, lex))
    else:
        print("Uso: python app/utils/lexico.py <caminho_lexico.json> <texto>")
