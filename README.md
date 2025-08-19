# SensiPrice-SP

Classificador **PT-BR** de **sensibilidade a preço** em textos de NPS/Relacionamento, utilizando **sinal fraco** (léxico/dicionário de termos) para gerar rótulos iniciais e **fine-tuning** de **BERT (neuralmind/bert-large-portuguese-cased)** para capturar contexto.  
**Entrada:** planilha Excel com coluna `Resumo`.  
**Saída:** mesma planilha com colunas `sensibilidade a preço` (SIM/NÃO) e `confianca` (probabilidade da classe SIM).

---

## Sumário
- [Motivação](#motivação)
- [Arquitetura e Fluxo](#arquitetura-e-fluxo)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Configuração do Léxico](#configuração-do-léxico)
- [Geração do Dataset (sinal fraco)](#geração-do-dataset-sinal-fraco)
- [Treinamento](#treinamento)
- [Inferência em Planilha](#inferência-em-planilha)
- [Scripts de Automação](#scripts-de-automação)
- [Exemplo de Saída](#exemplo-de-saída)
- [Boas Práticas e Dicas](#boas-práticas-e-dicas)
- [Resolução de Problemas](#resolução-de-problemas)
- [Roadmap](#roadmap)
- [Créditos](#créditos)
- [Licença](#licença)

---

## Motivação
Detectar **sensibilidade a preço** em comentários de NPS/relacionamento permite priorizar ações de **Customer Success**, **Comercial** e **Produtos** (revisão de proposta, argumentação de valor, bundles, descontos) com base em evidências textuais. Este repositório entrega um pipeline **local**, **reproduzível** e **extensível** para esse fim, com custo reduzido de rotulação.

---

## Arquitetura e Fluxo

1. **Léxico (dicionário)** define termos/expressões indicativos de sensibilidade a preço.
2. **Sinal fraco**: os textos são rotulados automaticamente (SIM/NÃO) e os trechos relevantes são marcados com `[PRICE] ... [/PRICE]` para orientar a atenção do BERT.
3. **Fine-tuning** do **BERT (neuralmind/bert-large-portuguese-cased)** com **pesos por classe** para lidar com desbalanceamento.
4. **Inferência** em planilha: o Excel de entrada recebe as novas colunas `sensibilidade a preço` e `confianca` (probabilidade).

**Benefícios técnicos:** injeção de “features fracas” via spans, aprendizado contextual do modelo, e capacidade de adaptação via simples edição do léxico.

---

## Requisitos

- **Python 3.10+** (recomendado 3.11)
- **Windows** (scripts `.bat` fornecidos). Em Linux/Mac, execute os módulos Python diretamente.
- **GPU opcional** (treinamento acelera; CPU também funciona).
- Pacotes em `requirements.txt` (Transformers, Datasets, Torch, Pandas, etc.).

---

## Instalação

```bash
# criar ambiente virtual
python -m venv .venv

# Windows
.\.venv\Scriptsctivate
# Linux/Mac
# source .venv/bin/activate

# instalar dependências
pip install -r requirements.txt
```

> Se você já tem o ambiente com os pacotes instalados, pode gerar o `requirements.txt` com:
> ```bash
> python -m pip freeze > requirements.txt
> ```

---

## Estrutura do Projeto

```
SensiPrice-SP/
├─ app/
│  ├─ inferir_preco_excel.py
│  └─ utils/
│     └─ lexico.py
├─ train/
│  ├─ prep_preco_dataset.py
│  └─ treina_preco.py
├─ lexico/
│  └─ preco.json
├─ data/
│  └─ Relacionamento_e_NPS.xlsx           # (entrada)
├─ app/model/
│  └─ bert-sensi-preco/                   # (modelo salvo após treinamento)
├─ scripts/
│  ├─ run_prep.bat
│  ├─ run_train.bat
│  └─ run_infer.bat
├─ requirements.txt
├─ .gitignore
└─ README.md
```

---

## Configuração do Léxico

Arquivo: `lexico/preco.json` (UTF-8, **JSON válido**). Exemplo:

```json
{
  "positivos": [
    "caro", "caríssima", "caríssimo", "preço", "preço alto", "preço elevado",
    "subiu o preço", "aumento de preço", "reajuste", "ficou mais caro",
    "mensalidade alta", "valor alto", "está caro", "muito caro",
    "custoso", "oneroso", "não vale a pena", "não cabe no orçamento",
    "sem orçamento", "orçamento apertado", "corte de custos", "reduzir custo",
    "mais barato", "mais em conta", "proposta cara", "precisa de desconto",
    "negociar preço", "desconto", "valores elevados", "são muito caras",
    "baratas", "cortando gastos"
  ],
  "negacoes": ["não", "sem", "nunca", "jamais"],
  "exclusoes": ["^\\s*caro\\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\\wÁÉÍÓÚÂÊÔÃÕÇ]+[,:-]"],
  "janela_negacao_palavras": 3
}
```

- **positivos**: termos/expressões gatilho do domínio.
- **negacoes**: lida com “não está caro” etc.
- **exclusoes**: evita falsos positivos como “Caro João,” (regex precisa de barras duplas em JSON).
- **janela_negacao_palavras**: tamanho da janela para checar negação (padrão 3).

> Ajuste o léxico à sua realidade (SaaS, serviços gerenciados, hardware). Evite termos genéricos (“valor” isolado) que geram falsos positivos; prefira “valor cobrado”, “valor da mensalidade”.

---

## Geração do Dataset (sinal fraco)

Coloque sua planilha em `data/Relacionamento_e_NPS.xlsx` (deve conter a coluna **`Resumo`**).  
Depois execute:

```bash
# Windows
scripts\run_prep.bat
# ou indicando um Excel alternativo
scripts\run_prep.bat "C:\caminho\MeuNPS.xlsx"
```

Saída: `data\sensi_preco_dataset.csv` com colunas:
- `texto` — texto **marcado** com `[PRICE] ... [/PRICE]`
- `label` — 0 (NÃO) ou 1 (SIM)

---

## Treinamento

Fine-tuning do BERT com **pesos por classe** (desbalanceamento tratado):

```bash
scripts\run_train.bat
```

O modelo será salvo em: `app\model\bert-sensi-preco\` (tokenizer + pesos).

---

## Inferência em Planilha

Anota um Excel adicionando as colunas **`sensibilidade a preço`** e **`confianca`**:

```bash
# defaults: arquivo=data\Relacionamento_e_NPS.xlsx, coluna=Resumo, modelo=app\model\bert-sensi-preco
scripts\run_infer.bat

# argumentos: <arquivo> <coluna> <saida> <pasta_modelo>
scripts\run_infer.bat "data\MeuNPS.xlsx" "Resumo" "data\MeuNPS_sensi.xlsx" "app\model\bert-sensi-preco"
```

---

## Scripts de Automação

| Script                 | Função                                                            | Observações                                   |
|------------------------|-------------------------------------------------------------------|-----------------------------------------------|
| `scripts\run_prep.bat`  | Gera dataset fraco `sensi_preco_dataset.csv`                     | Aceita Excel opcional como argumento.         |
| `scripts\run_train.bat` | Treina BERT e salva em `app\model\bert-sensi-preco`            | Usa GPU se disponível.                        |
| `scripts\run_infer.bat` | Anota Excel com `sensibilidade a preço` + `confianca`            | Parâmetros opcionais (arquivo/coluna/saída).  |

---

## Exemplo de Saída

Colunas adicionadas ao Excel:

| Resumo                                                          | sensibilidade a preço | confianca |
|-----------------------------------------------------------------|-----------------------|-----------|
| “O serviço é **muito caro** para o que entrega.”                | SIM                   | 0.91      |
| “**Não** está caro, mas o suporte precisa melhorar.”            | NÃO                   | 0.22      |
| “Vamos ter que **cortar custos** no próximo trimestre.”         | SIM                   | 0.68      |

> Limiar prático: usar `confianca >= 0.6` para priorizar casos (ajuste ao seu contexto).

---

## Boas Práticas e Dicas

- Ajuste o **léxico** ao domínio (e-mails, contratos, negociação, etc.).
- **Audite** amostras com baixa confiança para *active learning*.
- Mantenha **pesos por classe** quando houver desbalanceamento (já implementado).
- Para produção/latência, considere **destilar** para um BERT menor.
- Garanta que `preco.json` esteja **válido** (sem vírgulas finais/comentários).

---

## Resolução de Problemas

- **JSONDecodeError no `preco.json`**: arquivo vazio/malformado. Use JSON **válido** e **UTF-8**.
- **“Coluna 'Resumo' não encontrada”**: confirme o nome exato da coluna no Excel.
- **OSError ao carregar modelo**: verifique se `app\model\bert-sensi-preco\` foi criado após o treino.
- **Acentos/encoding**: salve `preco.json` e Excel em **UTF-8**; mantenha `openpyxl` atualizado.

---

## Roadmap

- Calibração de probabilidades (Platt/Isotônica) para limiar mais estável.
- Léxicos por setor (perfis) com seleção automática por cliente/segmento.
- Explicabilidade local (destaque de spans influentes no score).
- Pipeline CI (tests + lint + treino mínimo de fumaça).
- Distilação do modelo para produção (trade-off entre acurácia e latência).

---

## Créditos

- Modelo base: **neuralmind/bert-large-portuguese-cased** (BERTimbau Large).
- Ferramentas: **HuggingFace Transformers & Datasets**, **PyTorch**, **Pandas**.

---

## Licença

Escolha a licença conforme sua necessidade (por exemplo, **MIT**).  
Inclua um arquivo `LICENSE` na raiz do repositório.

---

### Citação (opcional)

Ao publicar resultados/relatórios, cite este repositório e o modelo base (BERTimbau) conforme diretrizes dos autores.
