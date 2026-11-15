"""API de Manutenção Preditiva Industrial (AI4I 2020).

Faculdade de Tecnologia de Pompeia "Shunji Nishimura" (Fatec Pompeia)
Curso: Tecnologia em Ciência de Dados
Disciplina: Aplicação de Ciência de Dados — Prof. Dr. João Ricardo Favan

Integrantes:
    - Hadryan Fortinis de Oliveira Lima
    - Raul Victor Matsunaga de Carvalho
    - Washingtom Ferreira Gonçalves

Arquivo único com a API REST (FastAPI) e a página web que consomem o modelo
treinado no notebook `treinamento_modelo.ipynb`.

Rotas:
    GET  /            página web para testar o modelo
    POST /predict     recebe a leitura da máquina e devolve a previsão de falha
    GET  /model-info  informações e métricas do modelo
    GET  /health      verificação de funcionamento
    GET  /docs        documentação automática (Swagger)

Executar (Docker):
    docker build -t manutencao-preditiva .
    docker run -p 8000:8000 manutencao-preditiva
"""

import json
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

PASTA = Path(__file__).resolve().parent
MODELO = joblib.load(PASTA / "models" / "modelo_manutencao_otimizado.joblib")
METADADOS = json.loads((PASTA / "models" / "model_metadata.json").read_text(encoding="utf-8"))
LIMIAR = float(METADADOS.get("decision_threshold", 0.5))
COLUNAS_ENTRADA = METADADOS["input_columns"]


def criar_atributos(dados: pd.DataFrame) -> pd.DataFrame:
    """Mesmos 4 atributos físicos criados no notebook de treinamento."""
    dados = dados.copy()
    dados["Temp_Diff_K"] = dados["Process temperature [K]"] - dados["Air temperature [K]"]
    dados["Mechanical_Power_W"] = dados["Torque [Nm]"] * dados["Rotational speed [rpm]"] * (2 * np.pi / 60)
    dados["Wear_Torque_Nm_min"] = dados["Tool wear [min]"] * dados["Torque [Nm]"]
    dados["Torque_Speed_Ratio"] = dados["Torque [Nm]"] / dados["Rotational speed [rpm]"]
    return dados


class LeituraMaquina(BaseModel):
    type: Literal["L", "M", "H"] = Field(..., description="Qualidade do produto: L, M ou H")
    air_temperature_k: float = Field(..., ge=250, le=360, description="Temperatura do ar [K]")
    process_temperature_k: float = Field(..., ge=250, le=400, description="Temperatura do processo [K]")
    rotational_speed_rpm: float = Field(..., gt=0, le=10000, description="Velocidade de rotação [rpm]")
    torque_nm: float = Field(..., ge=0, le=250, description="Torque [Nm]")
    tool_wear_min: float = Field(..., ge=0, le=600, description="Desgaste da ferramenta [min]")

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "L",
                "air_temperature_k": 300.8,
                "process_temperature_k": 309.4,
                "rotational_speed_rpm": 1342,
                "torque_nm": 62.4,
                "tool_wear_min": 113,
            }
        }
    }


def alertas_fisicos(linha: pd.Series) -> list[str]:
    """Explica ao operador quais grandezas estão fora da faixa normal."""
    alertas = []
    if linha["Temp_Diff_K"] < 8.6 and linha["Rotational speed [rpm]"] < 1380:
        alertas.append("Dissipação térmica reduzida (ΔT < 8,6 K com baixa rotação)")
    if not 3500 <= linha["Mechanical_Power_W"] <= 9000:
        alertas.append("Potência mecânica fora da faixa nominal (3.500–9.000 W)")
    if linha["Wear_Torque_Nm_min"] > 10500:
        alertas.append("Esforço desgaste × torque elevado (> 10.500 N·m·min)")
    if linha["Tool wear [min]"] >= 200:
        alertas.append("Desgaste crítico da ferramenta (≥ 200 min)")
    return alertas or ["Parâmetros físicos dentro da faixa operacional nominal"]


app = FastAPI(
    title="API de Manutenção Preditiva (AI4I 2020)",
    description="Prevê falhas em máquinas industriais com o modelo HistGradientBoosting otimizado.",
    version="2.0.0",
)


@app.post("/predict")
def predict(leitura: LeituraMaquina):
    dados = pd.DataFrame([{
        "Type": leitura.type,
        "Air temperature [K]": leitura.air_temperature_k,
        "Process temperature [K]": leitura.process_temperature_k,
        "Rotational speed [rpm]": leitura.rotational_speed_rpm,
        "Torque [Nm]": leitura.torque_nm,
        "Tool wear [min]": leitura.tool_wear_min,
    }])
    dados = criar_atributos(dados)[COLUNAS_ENTRADA]
    linha = dados.iloc[0]

    prob = float(MODELO.predict_proba(dados)[0, 1])
    falha = prob >= LIMIAR
    risco = "CRITICO" if falha else "MODERADO" if prob >= 0.20 else "BAIXO"

    return {
        "prediction": int(falha),
        "status": "FALHA_PREVISTA" if falha else "NORMAL",
        "failure_probability": round(prob, 6),
        "risk_level": risco,
        "threshold": LIMIAR,
        "engineered_features": {
            "temp_diff_k": round(float(linha["Temp_Diff_K"]), 4),
            "mechanical_power_w": round(float(linha["Mechanical_Power_W"]), 2),
            "wear_torque_nm_min": round(float(linha["Wear_Torque_Nm_min"]), 2),
            "torque_speed_ratio": round(float(linha["Torque_Speed_Ratio"]), 6),
        },
        "physical_alerts": alertas_fisicos(linha),
        "model_name": METADADOS["best_model_name"],
    }


@app.get("/model-info")
def model_info():
    return METADADOS


@app.get("/health")
def health():
    return {"status": "ok", "model": METADADOS["best_model_name"]}


PAGINA = """<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Manutenção Preditiva</title>
<style>
  :root { --azul:#1B365D; --verde:#0F766E; --vermelho:#B91C1C; --amarelo:#D97706; --cinza:#475569; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, sans-serif; background:#F1F5F9; color:#0F172A; }
  header { background:var(--azul); color:#fff; padding:20px 16px; text-align:center; }
  header h1 { margin:0; font-size:1.4rem; }
  header p { margin:6px 0 0; opacity:.8; font-size:.9rem; }
  main { max-width:900px; margin:24px auto; padding:0 16px; display:grid; gap:20px; grid-template-columns:1fr 1fr; }
  @media (max-width: 720px) { main { grid-template-columns:1fr; } }
  .card { background:#fff; border-radius:12px; padding:20px; box-shadow:0 1px 4px #0002; }
  h2 { margin:0 0 14px; font-size:1.05rem; color:var(--azul); }
  label { display:block; font-size:.85rem; margin:10px 0 4px; color:var(--cinza); }
  input, select { width:100%; padding:9px 10px; border:1px solid #CBD5E1; border-radius:8px; font-size:.95rem; }
  .botoes { display:flex; gap:8px; margin-top:16px; flex-wrap:wrap; }
  button { flex:1; padding:11px; border:0; border-radius:8px; font-weight:600; cursor:pointer; color:#fff; background:var(--verde); }
  button.sec { background:#E2E8F0; color:#0F172A; font-weight:500; }
  .status { font-size:1.6rem; font-weight:700; margin:4px 0; }
  .barra { height:12px; background:#E2E8F0; border-radius:6px; overflow:hidden; margin:8px 0 16px; }
  .barra div { height:100%; width:0; transition:width .4s; }
  table { width:100%; border-collapse:collapse; font-size:.88rem; }
  td { padding:6px 0; border-bottom:1px solid #E2E8F0; }
  td:last-child { text-align:right; font-variant-numeric:tabular-nums; }
  ul { padding-left:18px; font-size:.88rem; }
  .vazio { color:var(--cinza); font-size:.9rem; }
  footer { text-align:center; font-size:.8rem; color:var(--cinza); padding:10px 16px 24px; }
  a { color:var(--verde); }
</style>
</head>
<body>
<header>
  <h1>Manutenção Preditiva Industrial</h1>
  <p>Modelo HistGradientBoosting otimizado · dataset AI4I 2020</p>
</header>
<main>
  <form class="card" id="form">
    <h2>Leitura da máquina</h2>
    <label>Tipo de produto</label>
    <select name="type"><option>L</option><option>M</option><option>H</option></select>
    <label>Temperatura do ar [K]</label><input name="air_temperature_k" type="number" step="0.1" value="300.8" required>
    <label>Temperatura do processo [K]</label><input name="process_temperature_k" type="number" step="0.1" value="309.4" required>
    <label>Rotação [rpm]</label><input name="rotational_speed_rpm" type="number" step="1" value="1342" required>
    <label>Torque [Nm]</label><input name="torque_nm" type="number" step="0.1" value="62.4" required>
    <label>Desgaste da ferramenta [min]</label><input name="tool_wear_min" type="number" step="1" value="113" required>
    <div class="botoes">
      <button type="button" class="sec" onclick="exemplo('normal')">Exemplo normal</button>
      <button type="button" class="sec" onclick="exemplo('falha')">Exemplo falha</button>
    </div>
    <div class="botoes"><button type="submit">Prever</button></div>
  </form>
  <section class="card" id="resultado">
    <h2>Resultado</h2>
    <p class="vazio">Preencha a leitura e clique em <b>Prever</b>.</p>
  </section>
</main>
<footer>Desenvolvido por Hadryan Fortinis de Oliveira Lima, Raul Victor Matsunaga de Carvalho e Washingtom Ferreira Gonçalves<br>Fatec Pompeia · Aplicação de Ciência de Dados · <a href="/docs">documentação da API</a></footer>
<script>
const EXEMPLOS = {
  normal: {type:"M", air_temperature_k:298.1, process_temperature_k:308.6, rotational_speed_rpm:1551, torque_nm:42.8, tool_wear_min:0},
  falha:  {type:"L", air_temperature_k:300.8, process_temperature_k:309.4, rotational_speed_rpm:1342, torque_nm:62.4, tool_wear_min:113}
};
const form = document.getElementById("form");
const saida = document.getElementById("resultado");
const fmt = (v, d) => Number(v).toLocaleString("pt-BR", {maximumFractionDigits: d});

function exemplo(nome) {
  for (const [k, v] of Object.entries(EXEMPLOS[nome])) form.elements[k].value = v;
  form.requestSubmit();
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const dados = Object.fromEntries(new FormData(form));
  for (const k in dados) if (k !== "type") dados[k] = Number(dados[k]);
  const resp = await fetch("/predict", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(dados)});
  const r = await resp.json();
  if (!resp.ok) { saida.innerHTML = "<h2>Resultado</h2><p>Dados inválidos: verifique os valores.</p>"; return; }
  const cor = r.risk_level === "CRITICO" ? "var(--vermelho)" : r.risk_level === "MODERADO" ? "var(--amarelo)" : "var(--verde)";
  const f = r.engineered_features;
  saida.innerHTML = `
    <h2>Resultado</h2>
    <div class="status" style="color:${cor}">${r.status === "NORMAL" ? "Operação normal" : "Falha prevista"}</div>
    <div>Probabilidade de falha: <b>${fmt(r.failure_probability * 100, 2)}%</b> · risco ${r.risk_level}</div>
    <div class="barra"><div style="width:${r.failure_probability * 100}%;background:${cor}"></div></div>
    <table>
      <tr><td>Diferença de temperatura</td><td>${fmt(f.temp_diff_k, 2)} K</td></tr>
      <tr><td>Potência mecânica</td><td>${fmt(f.mechanical_power_w, 0)} W</td></tr>
      <tr><td>Desgaste × torque</td><td>${fmt(f.wear_torque_nm_min, 0)} N·m·min</td></tr>
      <tr><td>Torque / rotação</td><td>${fmt(f.torque_speed_ratio, 4)}</td></tr>
    </table>
    <ul>${r.physical_alerts.map(a => `<li>${a}</li>`).join("")}</ul>`;
});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def pagina():
    return PAGINA


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


# =============================================================================
#  Trabalho desenvolvido por:
#    Hadryan Fortinis de Oliveira Lima
#    Raul Victor Matsunaga de Carvalho
#    Washingtom Ferreira Gonçalves
#  Fatec Pompeia "Shunji Nishimura" · Aplicação de Ciência de Dados · 2026
# =============================================================================
