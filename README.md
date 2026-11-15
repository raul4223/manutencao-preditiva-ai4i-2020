# Manutenção Preditiva Industrial — AI4I 2020

Trabalho da disciplina **Aplicação de Ciência de Dados** (Fatec Pompeia Shunji Nishimura — 2026).
Prof. Dr. João Ricardo Favan.

**Integrantes:** Hadryan Fortinis de Oliveira Lima, Raul Victor Matsunaga de Carvalho e Washingtom Ferreira Gonçalves.

## Arquivos de entrega

| Arquivo | O que é |
|---|---|
| `treinamento_modelo.ipynb` | Notebook com todo o experimento: análise, pré-processamento, engenharia de atributos, Rank × PCA, comparação de modelos, otimização e exportação do modelo |
| `app.py` | Código-fonte da aplicação (API FastAPI + página web) em um único arquivo |
| `Relatorio_Manutencao_Preditiva_ABNT.pdf` | Relatório no padrão ABNT |

Arquivos de apoio: `data/ai4i2020.csv` (dataset), `models/` (modelo treinado gerado pelo notebook),
`Dockerfile` e `requirements.txt` (ambiente da API), `reports/` (figuras e gerador do relatório).

## 1. Treinar o modelo (notebook)

Abra `treinamento_modelo.ipynb` no Jupyter / VS Code e execute todas as células.
No final ele salva `models/modelo_manutencao_otimizado.joblib` e `models/model_metadata.json`.

> O modelo já treinado está incluído em `models/`, então este passo é opcional para rodar a API.
> Se treinar de novo, use a mesma versão do scikit-learn do `requirements.txt` (1.9.1).

## 2. Rodar a aplicação (Docker)

Não precisa criar ambiente virtual: tudo roda dentro do contêiner.

```bash
docker build -t manutencao-preditiva .
docker run -p 8000:8000 manutencao-preditiva
```

Acesse:
- Página web: http://localhost:8000
- Documentação da API (Swagger): http://localhost:8000/docs

Exemplo de chamada direta:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"type":"L","air_temperature_k":300.8,"process_temperature_k":309.4,"rotational_speed_rpm":1342,"torque_nm":62.4,"tool_wear_min":113}'
```

## Resultados principais

| Modelo | F1 (teste) | Precisão | Recall |
|---|---|---|---|
| HistGradientBoosting padrão | 0,8819 | 0,9492 | 0,8235 |
| HistGradientBoosting otimizado (GridSearchCV, 216 combinações) | **0,9048** | **0,9828** | **0,8382** |
