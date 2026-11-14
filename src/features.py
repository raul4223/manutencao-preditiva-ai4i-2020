"""Engenharia de atributos e esquema de entrada para manutenção preditiva (AI4I 2020)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif

TARGET_COLUMN = "Machine failure"
LEAKAGE_AND_ID_COLUMNS = [
    "UDI",
    "Product ID",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

RAW_FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

ENGINEERED_FEATURE_COLUMNS = [
    "Temp_Diff_K",
    "Mechanical_Power_W",
    "Wear_Torque_Nm_min",
    "Torque_Speed_Ratio",
]


def compute_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula atributos físicos derivados para leitura de telemetria industrial."""
    out = df.copy()
    air_temp = pd.to_numeric(out["Air temperature [K]"], errors="coerce")
    proc_temp = pd.to_numeric(out["Process temperature [K]"], errors="coerce")
    rpm = pd.to_numeric(out["Rotational speed [rpm]"], errors="coerce")
    torque = pd.to_numeric(out["Torque [Nm]"], errors="coerce")
    wear = pd.to_numeric(out["Tool wear [min]"], errors="coerce")

    out["Temp_Diff_K"] = proc_temp - air_temp
    out["Mechanical_Power_W"] = torque * rpm * (2.0 * np.pi / 60.0)
    out["Wear_Torque_Nm_min"] = wear * torque
    out["Torque_Speed_Ratio"] = np.where(rpm > 0, torque / rpm, np.nan)
    return out


class PredictiveMaintenanceFeatureEngineer(BaseEstimator, TransformerMixin):
    """Transformador scikit-learn que gera 4 atributos físicos de operação."""

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        return compute_physical_features(X)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        base = (
            list(input_features)
            if input_features is not None
            else list(getattr(self, "feature_names_in_", RAW_FEATURE_COLUMNS))
        )
        for col in ENGINEERED_FEATURE_COLUMNS:
            if col not in base:
                base.append(col)
        return np.asarray(base, dtype=object)


def mutual_info_ranker(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calcula Informação Mútua determinística para ranqueamento de atributos."""
    return mutual_info_classif(X, y, random_state=42)
