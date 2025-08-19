
# train/treina_preco.py
# -----------------------------------------------------------
# Treino para sensibilidade a preço (PT-BR)
# (NOVO) --synthetic: gera dataset SÓ com o dicionário antes do treino.
# -----------------------------------------------------------

import os
import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Para geração sintética se necessário
def _maybe_generate_synthetic(csv_path: Path, lexico_path: Path, n_pos: int, n_neg: int, seed: int):
    """
    Gera dataset sintético reutilizando o gerador do prep_preco_dataset.py.
    Se o CSV já existir, não faz nada (a menos que n_pos/n_neg sejam -1, forçando).
    """
    if csv_path.exists() and n_pos != -1 and n_neg != -1:
        print(f"ℹ️ CSV já existe: {csv_path}. Pulando geração sintética.")
        return
    try:
        from prep_preco_dataset import generate_synthetic_dataset
        generate_synthetic_dataset(lexico_path, csv_path, n_pos=n_pos, n_neg=n_neg, seed=seed)
    except Exception as e:
        # Fallback: gerar aqui um dataset mínimo
        print(f"⚠️ Falha ao importar gerador sintético ({e}). Usando fallback mínimo.")
        import csv, random, json
        rng = random.Random(seed)
        data = json.loads(Path(lexico_path).read_text(encoding='utf-8'))
        sint = data.get("sintetico", {})
        hard = list(sint.get("hard_phrases", ["preço alto", "muito caro", "reajuste"]))
        negs = list(sint.get("neg_safe_phrases", ["preço justo", "bom custo-benefício"]))
        pos_templates = list(sint.get("pos_templates", ["o serviço está [PRICE]{x}[/PRICE]."]))
        neg_templates = list(sint.get("neg_templates", ["{x}."]))
        rows = []
        for _ in range(max(1, n_pos)):
            x = rng.choice(hard)
            tpl = rng.choice(pos_templates)
            rows.append((tpl.format(x=x), 1))
        for _ in range(max(1, n_neg)):
            x = rng.choice(negs)
            tpl = rng.choice(neg_templates)
            rows.append((tpl.format(x=x), 0))
        rng.shuffle(rows)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";", quoting=csv.QUOTE_ALL, lineterminator="\r\n")
            w.writerow(["texto","label"])
            for t,l in rows:
                w.writerow([t,l])
        print(f"✅ Gerado (fallback) dataset sintético: {csv_path}")

# ------------------------------
# Config padrão
# ------------------------------
MODEL_BASE = "neuralmind/bert-base-portuguese-cased"
DATA_CSV_DEFAULT = "data/sensi_preco_dataset.csv"
OUT_DIR_DEFAULT = "app/model/bert-sensi-preco"
SEED_DEFAULT = 42
MAX_LEN_DEFAULT = 128  # suficiente p/ maioria dos "Resumo"
LEXICO_DEFAULT = "lexico/preco.json"

LABEL2ID = {"NAO": 0, "SIM": 1}
ID2LABEL = {0: "NAO", 1: "SIM"}

# ------------------------------
# Métricas
# ------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0, pos_label=1
    )
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

# ------------------------------
# Focal Loss opcional
# ------------------------------
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, target):
        ce = torch.nn.functional.cross_entropy(
            logits, target.long(), weight=self.alpha, reduction="none"
        )
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

# ------------------------------
# Congelar/descongelar camadas
# ------------------------------
def freeze_and_unfreeze_last(model, unfreeze_last=2):
    backbone = getattr(model, "bert", None) or getattr(model, "base_model", None)
    if backbone is None or not hasattr(backbone, "encoder"):
        return
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(backbone, "embeddings"):
        for p in backbone.embeddings.parameters():
            p.requires_grad = True
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True
    if unfreeze_last and hasattr(backbone.encoder, "layer"):
        for layer in backbone.encoder.layer[-int(unfreeze_last):]:
            for p in layer.parameters():
                p.requires_grad = True

# ------------------------------
# Utilidades dataset
# ------------------------------
def normalize_label(x: Any) -> int:
    if x is None: return 0
    if isinstance(x, bool): return 1 if x else 0
    s = str(x).strip().upper()
    s = (s.replace("Ã","A").replace("Õ","O").replace("Ó","O")
           .replace("Á","A").replace("Â","A").replace("Ê","E")
           .replace("Í","I").replace("Ú","U"))
    if s in ("1","SIM","S","TRUE"): return 1
    if s in ("0","NAO","N","FALSE","NAO ","NÃO","NAO.", "NÃO."): return 0
    try:
        v = int(float(s)); return 1 if v==1 else 0
    except Exception:
        return 0

def load_and_prepare_dataset(csv_path: str, delimiter: str, text_col: str, val_size: float, seed: int, lowercase: bool) -> DatasetDict:
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Dataset não localizado: {csv_path}")
    ds = load_dataset("csv", data_files=csv_path, delimiter=delimiter)["train"]
    cols = set(ds.column_names)
    if text_col not in cols:
        raise ValueError(f"Coluna de texto '{text_col}' não encontrada. Colunas: {sorted(cols)}")
    if "label" not in cols:
        raise ValueError("Coluna 'label' não encontrada no dataset.")
    def _map_row(batch):
        textos_orig = batch[text_col]
        labels_orig = batch["label"]
        textos = [("" if t is None else str(t)) for t in textos_orig]
        if lowercase:
            textos = [t.lower() for t in textos]
        return {"texto": textos, "label": [normalize_label(l) for l in labels_orig]}
    ds = ds.map(_map_row, batched=True, remove_columns=[c for c in ds.column_names if c not in (text_col, "label")])
    if text_col != "texto":
        ds = ds.remove_columns([text_col])
    ds = ds.train_test_split(test_size=val_size, seed=seed)
    return ds

# ------------------------------
# Principal
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=DATA_CSV_DEFAULT, help="Caminho do CSV (padrão: data/sensi_preco_dataset.csv)")
    ap.add_argument("--delimiter", type=str, default=";", help="Delimitador do CSV (padrão: ';')")
    ap.add_argument("--text_col", type=str, default="texto", help="Nome da coluna de texto no CSV (padrão: 'texto')")
    ap.add_argument("--val_size", type=float, default=0.2, help="Proporção de validação (padrão: 0.2)")
    ap.add_argument("--epochs", type=int, default=1, help="Épocas (padrão=1, rápido)")
    ap.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    ap.add_argument("--batch", type=int, default=None, help="Batch size (auto por GPU/CPU)")
    ap.add_argument("--use_focal", action="store_true", help="Ativa Focal Loss")
    ap.add_argument("--unfreeze_last", type=int, default=2, help="N camadas finais a descongelar")
    ap.add_argument("--max_len", type=int, default=MAX_LEN_DEFAULT, help="Comprimento máx. de tokens")
    ap.add_argument("--out_dir", type=str, default=OUT_DIR_DEFAULT, help="Diretório de saída do modelo")
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT, help="Seed")
    ap.add_argument("--lowercase", action="store_true", help="Força texto minúsculo no treino")
    ap.add_argument("--patience", type=int, default=3, help="Early stopping patience (épocas)")
    # NOVOS
    ap.add_argument("--synthetic", action="store_true", help="Gera dataset 100% sintético a partir do dicionário antes do treino.")
    ap.add_argument("--lexico", type=str, default=LEXICO_DEFAULT, help="Caminho do léxico JSON.")
    ap.add_argument("--n_pos", type=int, default=1200, help="(synthetic) n exemplos positivos.")
    ap.add_argument("--n_neg", type=int, default=1200, help="(synthetic) n exemplos negativos.")

    args = ap.parse_args()

    # Seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    has_cuda = torch.cuda.is_available()
    train_bs = args.batch or (16 if has_cuda else 4)
    eval_bs  = train_bs

    # --- opcional: gerar dataset sintético
    data_path = Path(args.data)
    if args.synthetic:
        _maybe_generate_synthetic(
            csv_path=data_path,
            lexico_path=Path(args.lexico),
            n_pos=args.n_pos,
            n_neg=args.n_neg,
            seed=args.seed
        )

    # --------------------------
    # Dataset
    # --------------------------
    ds = load_and_prepare_dataset(
        csv_path=str(data_path),
        delimiter=args.delimiter,
        text_col=args.text_col,
        val_size=args.val_size,
        seed=args.seed,
        lowercase=args.lowercase,
    )

    # --------------------------
    # Tokenizer
    # --------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE, use_fast=True)
    special_tokens = {"additional_special_tokens": ["[PRICE]", "[/PRICE]"]}
    tokenizer.add_special_tokens(special_tokens)

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if has_cuda else None,
    )

    def tok(batch):
        textos = [str(t) if t is not None else "" for t in batch["texto"]]
        return tokenizer(textos, truncation=True, padding=False, max_length=args.max_len)

    map_num_proc = 1 if os.name == "nt" else max(os.cpu_count() // 2, 1)

    ds_tok = DatasetDict()
    for split in ds:
        ds_tok[split] = ds[split].map(
            tok,
            batched=True,
            num_proc=map_num_proc,
            remove_columns=[c for c in ds[split].column_names if c not in ("label",)],
        )

    # --------------------------
    # Modelo
    # --------------------------
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_BASE, num_labels=2)
    model.config.label2id = {"NAO":0, "SIM":1}
    model.config.id2label = {0:"NAO", 1:"SIM"}

    model.resize_token_embeddings(len(tokenizer))
    freeze_and_unfreeze_last(model, unfreeze_last=args.unfreeze_last)

    # --------------------------
    # Pesos de classe + Sampler balanceado
    # --------------------------
    y_train = np.array(ds_tok["train"]["label"])
    c0, c1 = (y_train == 0).sum(), (y_train == 1).sum()
    tot = len(y_train)
    w0 = tot / (2 * c0) if c0 > 0 else 1.0
    w1 = tot / (2 * c1) if c1 > 0 else 1.0
    class_weights = torch.tensor([w0, w1], dtype=torch.float)

    per_sample_w = np.where(y_train == 1, w1, w0).astype("float64")
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(per_sample_w, dtype=torch.double),
        num_samples=len(per_sample_w),
        replacement=True,
    )

    # --------------------------
    # Trainer custom (loss + sampler)
    # --------------------------
    class MyTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits  = outputs.get("logits")
            if args.use_focal:
                loss_f = FocalLoss(alpha=class_weights.to(model.device), gamma=2.0)
                loss   = loss_f(logits.view(-1, 2), labels.view(-1))
            else:
                loss_f = torch.nn.CrossEntropyLoss(weight=class_weights.to(model.device))
                loss   = loss_f(logits.view(-1, 2), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

        def get_train_dataloader(self):
            return DataLoader(
                self.train_dataset,
                batch_size=train_bs,
                sampler=sampler,
                collate_fn=collator,
                pin_memory=has_cuda,
                drop_last=False,
                num_workers=0 if os.name == "nt" else max(os.cpu_count() // 2, 1),
            )

    # --------------------------
    # Args de treino
    # --------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args_tr = TrainingArguments(
        output_dir=str(out_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        learning_rate=args.lr,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",
        logging_steps=100,
        report_to=[],
        fp16=has_cuda,
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=0 if os.name == "nt" else max(os.cpu_count() // 2, 1),
    )

    trainer = MyTrainer(
        model=model,
        args=args_tr,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["test"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    trainer.train()

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    cfg_path = out_dir / "config.json"
    if cfg_path.exists():
        print(f"📝 Config salvo em: {cfg_path}")

    print(f"✅ Modelo salvo em {out_dir.resolve()} (épocas={args.epochs}, batch={train_bs}, max_len={args.max_len})")
    print(f"Distribuição treino — 0:{c0}  1:{c1}  (w0={w0:.3f}, w1={w1:.3f})")

if __name__ == "__main__":
    main()
