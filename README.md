# Manutenção Preditiva Industrial — AI4I 2020

Projeto completo da disciplina **Aplicação de Ciência de Dados** (Fatec Pompeia Shunji Nishimura — 2026), cobrindo desde a preparação de dados industriais e engenharia de atributos físicos até redução de dimensionalidade, comparação e otimização de modelos, publicação via API REST + interface web conteinerizada e relatório técnico em conformidade com as normas da ABNT.

## Integrantes do Grupo

- Hadryan Fortinis de Oliveira Lima
- Raul Victor Matsunaga de Carvalho
- Washingtom Ferreira Gonçalves

**Docente:** Prof. Dr. João Ricardo Favan

---

## Estrutura do Projeto

```
Tarefa_Ciencia_de _Dados/
├── app/
│   ├── __init__.py
│   ├── main.py                     # API REST FastAPI (/predict, /predict/batch, /model-info, /health)
│   └── static/
│       └── index.html              # Painel web industrial integrado à API
├── data/
│   └── ai4i2020.csv                # Dataset AI4I 2020 Predictive Maintenance (10.000 amostras)
├── models/
│   ├── modelo_manutencao_otimizado.joblib
│   └── model_metadata.json
├── notebooks/
│   └── experimento_manutencao_preditiva.ipynb
├── reports/
│   ├── figures/                    # Gráficos 300 DPI usados no relatório
│   ├── gerar_relatorio.py          # Gerador automático ABNT (.docx e .pdf)
│   ├── metricas_experimento.json
│   └── Relatorio_Manutencao_Preditiva_ABNT.docx
├── src/
│   ├── __init__.py
│   ├── features.py                 # Transformador sklearn com 4 atributos físicos derivados
│   └── train_pipeline.py           # Experimento reproduzível (EDA, ablação, Rank/PCA, tuning, export)
├── tests/
│   └── test_pipeline_and_api.py
├── Dockerfile
├── README.md
├── Relatorio_Manutencao_Preditiva_ABNT.pdf
└── requirements.txt
```

---

## Execução Local

### 1. Treinar o pipeline e gerar artefatos/figuras
```bash
.venv/bin/python -m src.train_pipeline
```

### 2. Executar testes automatizados
```bash
.venv/bin/pytest -v
```

### 3. Iniciar a API e Interface Web em Produção Local
```bash
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Acesse no navegador:
- Painel operacional integrado: `http://127.0.0.1:8000/`
- Documentação interativa Swagger/OpenAPI: `http://127.0.0.1:8000/docs`

### 4. Execução via Docker
```bash
docker build -t manutencao-preditiva-ai4i .
docker run -p 8000:8000 manutencao-preditiva-ai4i
```
