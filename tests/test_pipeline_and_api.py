"""Testes automatizados para engenharia de atributos, pipeline exportado e API REST."""

from __future__ import annotations

import math
from pathlib import Path

import joblib
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from src.features import ENGINEERED_FEATURE_COLUMNS, RAW_FEATURE_COLUMNS, compute_physical_features

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "modelo_manutencao_otimizado.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
METRICS_PATH = BASE_DIR / "reports" / "metricas_experimento.json"


def test_compute_physical_features_values():
    df = pd.DataFrame(
        [
            {
                "Type": "M",
                "Air temperature [K]": 300.0,
                "Process temperature [K]": 310.0,
                "Rotational speed [rpm]": 1500.0,
                "Torque [Nm]": 40.0,
                "Tool wear [min]": 100.0,
            }
        ],
        columns=RAW_FEATURE_COLUMNS,
    )
    out = compute_physical_features(df)
    for col in ENGINEERED_FEATURE_COLUMNS:
        assert col in out.columns

    assert math.isclose(out.loc[0, "Temp_Diff_K"], 10.0, rel_tol=1e-6)
    expected_power = 40.0 * 1500.0 * (2.0 * math.pi / 60.0)
    assert math.isclose(out.loc[0, "Mechanical_Power_W"], expected_power, rel_tol=1e-6)
    assert math.isclose(out.loc[0, "Wear_Torque_Nm_min"], 4000.0, rel_tol=1e-6)
    assert math.isclose(out.loc[0, "Torque_Speed_Ratio"], 40.0 / 1500.0, rel_tol=1e-6)


def test_exported_artifacts_and_roundtrip_inference():
    assert MODEL_PATH.exists(), "Modelo .joblib não foi exportado."
    assert METADATA_PATH.exists(), "Arquivo model_metadata.json não foi exportado."
    assert METRICS_PATH.exists(), "Arquivo metricas_experimento.json não foi gerado."

    pipeline = joblib.load(MODEL_PATH)
    sample = pd.DataFrame(
        [
            {
                "Type": "M",
                "Air temperature [K]": 298.6,
                "Process temperature [K]": 309.2,
                "Rotational speed [rpm]": 1520.0,
                "Torque [Nm]": 38.5,
                "Tool wear [min]": 72.0,
            },
            {
                "Type": "L",
                "Air temperature [K]": 302.5,
                "Process temperature [K]": 310.4,
                "Rotational speed [rpm]": 1320.0,
                "Torque [Nm]": 54.0,
                "Tool wear [min]": 214.0,
            },
        ],
        columns=RAW_FEATURE_COLUMNS,
    )
    probs = pipeline.predict_proba(sample)
    assert probs.shape == (2, 2)
    assert probs[1, 1] > probs[0, 1]


def test_api_endpoints_end_to_end():
    with TestClient(app) as client:
        # Página web integrada
        r_root = client.get("/")
        assert r_root.status_code == 200
        assert "Manutenção Preditiva" in r_root.text

        # Saúde e metadados
        r_health = client.get("/health")
        assert r_health.status_code == 200
        health_json = r_health.json()
        assert health_json["status"] == "ok"
        assert health_json["model_loaded"] is True

        r_meta = client.get("/model-info")
        assert r_meta.status_code == 200
        meta_json = r_meta.json()
        assert len(meta_json["selected_features"]) == 10
        assert meta_json["metrics"]["tuned_test"]["test_f1"] >= meta_json["metrics"]["default_test"]["test_f1"]

        # Predição nominal -> normal
        r_nom = client.post(
            "/predict",
            json={
                "type": "M",
                "air_temperature_k": 298.6,
                "process_temperature_k": 309.2,
                "rotational_speed_rpm": 1520.0,
                "torque_nm": 38.5,
                "tool_wear_min": 72.0,
            },
        )
        assert r_nom.status_code == 200
        nom = r_nom.json()
        assert nom["prediction"] == 0
        assert nom["status_label"] == "NORMAL"

        # Predição crítica (dissipação térmica HDF + desgaste) -> falha
        r_fail = client.post(
            "/predict",
            json={
                "type": "L",
                "air_temperature_k": 302.5,
                "process_temperature_k": 310.4,
                "rotational_speed_rpm": 1320.0,
                "torque_nm": 58.0,
                "tool_wear_min": 214.0,
            },
        )
        assert r_fail.status_code == 200
        fail = r_fail.json()
        assert fail["prediction"] == 1
        assert fail["status_label"] == "FALHA_PREVISTA"
        assert fail["risk_level"] == "CRITICO"

        # Lote
        r_batch = client.post(
            "/predict/batch",
            json={
                "items": [
                    {
                        "type": "M",
                        "air_temperature_k": 298.6,
                        "process_temperature_k": 309.2,
                        "rotational_speed_rpm": 1520.0,
                        "torque_nm": 38.5,
                        "tool_wear_min": 72.0,
                    },
                    {
                        "type": "L",
                        "air_temperature_k": 302.5,
                        "process_temperature_k": 310.4,
                        "rotational_speed_rpm": 1320.0,
                        "torque_nm": 58.0,
                        "tool_wear_min": 214.0,
                    },
                ]
            },
        )
        assert r_batch.status_code == 200
        b = r_batch.json()
        assert b["total"] == 2
        assert b["predicted_failures"] == 1

        # Validação de payload inválido (rotação <= 0)
        r_bad = client.post(
            "/predict",
            json={
                "type": "M",
                "air_temperature_k": 298.6,
                "process_temperature_k": 309.2,
                "rotational_speed_rpm": 0.0,
                "torque_nm": 38.5,
                "tool_wear_min": 72.0,
            },
        )
        assert r_bad.status_code == 422
