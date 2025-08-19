import sys, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

csv = sys.argv[1] if len(sys.argv)>1 else r"data\sensi_preco_dataset_bal.csv"
delimiter = sys.argv[2] if len(sys.argv)>2 else ";"
model_dir = sys.argv[3] if len(sys.argv)>3 else r"app\model\bert-sensi-preco"
seed = 42

df = pd.read_csv(csv, sep=delimiter, encoding="utf-8-sig")
df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
cut = int(len(df)*0.8)
val = df.iloc[cut:].reset_index(drop=True)

tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
mdl = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)

def predict_proba(texts):
    probs = []
    for i in range(0, len(texts), 64):
        batch = texts[i:i+64].tolist()
        x = tok(batch, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad(): p = torch.softmax(mdl(**x).logits, dim=1).numpy()[:,1]
        probs.extend(p.tolist())
    return np.array(probs)

p = predict_proba(val["texto"])
y = val["label"].values.astype(int)

best, best_thr = (-1, 0.5), 0.5
for thr in np.linspace(0.20, 0.80, 61):
    yhat = (p >= thr).astype(int)
    acc = accuracy_score(y, yhat)
    pr, rc, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    score = f1  # pode trocar p/ sua métrica preferida
    if score > best[0]:
        best, best_thr = (score, acc, pr, rc), thr

print(f"✅ melhor limiar = {best_thr:.3f} | F1={best[0]:.3f} | Acc={best[1]:.3f} | P={best[2]:.3f} | R={best[3]:.3f}")
