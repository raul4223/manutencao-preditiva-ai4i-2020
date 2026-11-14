"""API REST (FastAPI) e interface web integrada para predição de falhas industriais."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src.features import RAW_FEATURE_COLUMNS, compute_physical_features

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "modelo_manutencao_otimizado.joblib"
DEFAULT_METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
STATIC_INDEX_PATH = Path(__file__).resolve().parent / "static" / "index.html"


class MachineTelemetryInput(BaseModel):
    """Contrato de entrada da telemetria bruta da máquina (esquema AI4I 2020)."""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "type": "L",
                "air_temperature_k": 300.8,
                "process_temperature_k": 309.4,
                "rotational_speed_rpm": 1342.0,
                "torque_nm": 62.4,
                "tool_wear_min": 113.0,
            }
        },
    )

    type: Literal["L", "M", "H"] = Field(
        ...,
        description="Variante de qualidade do produto/operação: L (Low), M (Medium) ou H (High).",
    )
    air_temperature_k: float = Field(
        ...,
        ge=250.0,
        le=360.0,
        description="Temperatura ambiente do ar em Kelvin [K].",
    )
    process_temperature_k: float = Field(
        ...,
        ge=250.0,
        le=400.0,
        description="Temperatura do processo de usinagem em Kelvin [K].",
    )
    rotational_speed_rpm: float = Field(
        ...,
        gt=0.0,
        le=10000.0,
        description="Velocidade de rotação do eixo [rpm].",
    )
    torque_nm: float = Field(
        ...,
        ge=0.0,
        le=250.0,
        description="Torque aplicado no eixo [Nm].",
    )
    tool_wear_min: float = Field(
        ...,
        ge=0.0,
        le=600.0,
        description="Desgaste acumulado da ferramenta [min].",
    )


class EngineeredFeaturesOut(BaseModel):
    temp_diff_k: float
    mechanical_power_w: float
    wear_torque_nm_min: float
    torque_speed_ratio: float


class PredictionResponse(BaseModel):
    prediction: int
    status_label: Literal["NORMAL", "FALHA_PREVISTA"]
    failure_probability: float
    risk_level: Literal["BAIXO", "MODERADO", "CRITICO"]
    threshold_used: float
    engineered_features: EngineeredFeaturesOut
    physical_alerts: list[str]
    model_name: str


class BatchTelemetryRequest(BaseModel):
    items: list[MachineTelemetryInput] = Field(..., min_length=1, max_length=500)


class BatchPredictionResponse(BaseModel):
    total: int
    predicted_failures: int
    failure_rate: float
    results: list[PredictionResponse]


def _load_artifacts() -> tuple[object | None, dict]:
    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    metadata_path = Path(os.getenv("METADATA_PATH", str(DEFAULT_METADATA_PATH)))

    model = joblib.load(model_path) if model_path.exists() else None
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    return model, metadata


def _to_dataframe(items: list[MachineTelemetryInput]) -> pd.DataFrame:
    rows = [
        {
            "Type": item.type,
            "Air temperature [K]": item.air_temperature_k,
            "Process temperature [K]": item.process_temperature_k,
            "Rotational speed [rpm]": item.rotational_speed_rpm,
            "Torque [Nm]": item.torque_nm,
            "Tool wear [min]": item.tool_wear_min,
        }
        for item in items
    ]
    return pd.DataFrame(rows, columns=RAW_FEATURE_COLUMNS)


def _diagnose_physical_alerts(row: pd.Series) -> list[str]:
    alerts: list[str] = []
    if row["Temp_Diff_K"] < 8.6 and row["Rotational speed [rpm]"] < 1380:
        alerts.append("Dissipação térmica reduzida (ΔT < 8,6 K com baixa rotação)")
    if row["Mechanical_Power_W"] < 3500 or row["Mechanical_Power_W"] > 9000:
        alerts.append("Potência mecânica fora da faixa nominal (3.500–9.000 W)")
    if row["Wear_Torque_Nm_min"] > 10500:
        alerts.append("Esforço desgaste×torque elevado (> 10.500 N·m·min)")
    if row["Tool wear [min]"] >= 200:
        alerts.append("Desgaste crítico da ferramenta (>= 200 min)")
    if not alerts:
        alerts.append("Parâmetros físicos dentro da faixa operacional nominal")
    return alerts


def _classify_risk(prob: float, threshold: float) -> Literal["BAIXO", "MODERADO", "CRITICO"]:
    if prob >= threshold:
        return "CRITICO"
    if prob >= max(0.20, threshold * 0.4):
        return "MODERADO"
    return "BAIXO"


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    model, metadata = _load_artifacts()
    app_instance.state.model = model
    app_instance.state.metadata = metadata
    yield


app = FastAPI(
    title="API de Manutenção Preditiva Industrial (AI4I 2020)",
    description=(
        "Serviço em produção para predição de falhas em máquinas industriais "
        "com engenharia de atributos físicos, seleção por Rank e modelo otimizado."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _get_loaded_model() -> tuple[object, dict]:
    model = getattr(app.state, "model", None)
    metadata = getattr(app.state, "metadata", None)
    if model is None:
        model, metadata = _load_artifacts()
        app.state.model = model
        app.state.metadata = metadata
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo otimizado não carregado.")
    return model, metadata or {}


@app.get("/", include_in_schema=False)
def serve_web_app():
    if STATIC_INDEX_PATH.exists():
        return FileResponse(STATIC_INDEX_PATH)
    return JSONResponse({"message": "API de Manutenção Preditiva ativa. Acesse /docs."})


@app.get("/health")
def health():
    model = getattr(app.state, "model", None)
    metadata = getattr(app.state, "metadata", {}) or {}
    if model is None:
        model, metadata = _load_artifacts()
        app.state.model = model
        app.state.metadata = metadata
    return {
        "status": "ok" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_name": metadata.get("best_model_name", "unknown"),
        "version": app.version,
    }


@app.get("/model-info")
def model_info():
    _, metadata = _get_loaded_model()
    return metadata


def _build_prediction_response(
    df_fe_row: pd.Series,
    prob: float,
    threshold: float,
    model_name: str,
) -> PredictionResponse:
    pred = int(prob >= threshold)
    return PredictionResponse(
        prediction=pred,
        status_label="FALHA_PREVISTA" if pred == 1 else "NORMAL",
        failure_probability=round(float(prob), 6),
        risk_level=_classify_risk(float(prob), threshold),
        threshold_used=threshold,
        engineered_features=EngineeredFeaturesOut(
            temp_diff_k=round(float(df_fe_row["Temp_Diff_K"]), 4),
            mechanical_power_w=round(float(df_fe_row["Mechanical_Power_W"]), 2),
            wear_torque_nm_min=round(float(df_fe_row["Wear_Torque_Nm_min"]), 2),
            torque_speed_ratio=round(float(df_fe_row["Torque_Speed_Ratio"]), 6),
        ),
        physical_alerts=_diagnose_physical_alerts(df_fe_row),
        model_name=model_name,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: MachineTelemetryInput):
    model, metadata = _get_loaded_model()
    df_raw = _to_dataframe([payload])
    df_fe = compute_physical_features(df_raw)
    prob = float(model.predict_proba(df_raw)[0, 1])
    threshold = float(metadata.get("decision_threshold", 0.5))
    model_name = str(metadata.get("best_model_name", "HistGradientBoostingClassifier (Otimizado)"))
    return _build_prediction_response(df_fe.iloc[0], prob, threshold, model_name)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchTelemetryRequest):
    model, metadata = _get_loaded_model()
    df_raw = _to_dataframe(payload.items)
    df_fe = compute_physical_features(df_raw)
    probs = model.predict_proba(df_raw)[:, 1]
    threshold = float(metadata.get("decision_threshold", 0.5))
    model_name = str(metadata.get("best_model_name", "HistGradientBoostingClassifier (Otimizado)"))

    results = [
        _build_prediction_response(df_fe.iloc[i], float(probs[i]), threshold, model_name)
        for i in range(len(payload.items))
    ]
    failures = sum(r.prediction for r in results)
    total = len(results)
    return BatchPredictionResponse(
        total=total,
        predicted_failures=failures,
        failure_rate=round(failures / total, 4) if total else 0.0,
        results=results,
    )
