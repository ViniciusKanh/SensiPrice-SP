from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("app/model/bert-sensi-preco", local_files_only=True)
mdl = AutoModelForSequenceClassification.from_pretrained("app/model/bert-sensi-preco", local_files_only=True)

tests = [
    "o preço está muito alto para o que entrega.",
    "ficou mais caro depois do reajuste.",
    "não está caro, está ok pra nós."
]

x = tok([t.lower() for t in tests], return_tensors="pt", truncation=True, padding=True, max_length=256)

with torch.no_grad():
    p = torch.softmax(mdl(**x).logits, dim=1).detach().numpy()[:, 1]

for t, prob in zip(tests, p):
    print(f"{t} -> {prob:.3f}")
