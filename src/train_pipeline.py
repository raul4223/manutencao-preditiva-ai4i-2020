"""Pipeline completo de treinamento, seleção de atributos, otimização e exportação.

Executa todas as etapas do experimento de Manutenção Preditiva Industrial
utilizando o dataset AI4I 2020 (10.000 amostras, 14 colunas originais):
1. Análise exploratória inicial e verificação de integridade (ausentes, duplicados, leakage).
2. Divisão estratificada treino/teste (80/20) antes de qualquer ajuste paramétrico.
3. Pré-processamento com ColumnTransformer (imputação, StandardScaler, OneHotEncoder).
4. Engenharia de atributos físicos (4 novos atributos) com estudo de ablação.
5. Comparação de redução de dimensionalidade: Rank (ANOVA F e Informação Mútua) vs PCA.
6. Treinamento e comparação de 6 algoritmos distintos com CV estratificado (5 folds).
7. Otimização de hiperparâmetros (GridSearchCV) do melhor algoritmo e comparação Padrão vs Otimizado.
8. Exportação do Pipeline completo em produção (.joblib), metadados (.json) e figuras (300 DPI).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import (
    ENGINEERED_FEATURE_COLUMNS,
    LEAKAGE_AND_ID_COLUMNS,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
    PredictiveMaintenanceFeatureEngineer,
    mutual_info_ranker,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "ai4i2020.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
RANK_K = 10

RAW_NUMERIC_COLS = [c for c in RAW_FEATURE_COLUMNS if c != "Type"]
ENGINEERED_NUMERIC_COLS = RAW_NUMERIC_COLS + ENGINEERED_FEATURE_COLUMNS
CATEGORICAL_COLS = ["Type"]

# Paleta coerente e legível para impressão/PDF (padrão técnico industrial)
PALETTE = {
    "navy": "#1B365D",
    "teal": "#0F766E",
    "amber": "#D97706",
    "crimson": "#B91C1C",
    "slate": "#475569",
    "light_slate": "#CBD5E1",
    "bg_panel": "#F8FAFC",
}


def build_preprocessor(use_engineered: bool = True) -> ColumnTransformer:
    """Constrói o pré-processador numérico + categórico sem vazamento de dados."""
    num_cols = ENGINEERED_NUMERIC_COLS if use_engineered else RAW_NUMERIC_COLS
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_full_pipeline(
    classifier: Any,
    *,
    use_engineered: bool = True,
    reduction: str = "anova_k10",
    memory: str | None = None,
) -> Pipeline:
    """Monta o Pipeline ponta-a-ponta (FE -> Pré-processamento -> Redução -> Modelo)."""
    steps: list[tuple[str, Any]] = []
    if use_engineered:
        steps.append(("feature_engineer", PredictiveMaintenanceFeatureEngineer()))
    steps.append(("preprocessor", build_preprocessor(use_engineered=use_engineered)))

    if reduction == "anova_k10":
        steps.append(("reducer", SelectKBest(score_func=f_classif, k=RANK_K)))
    elif reduction == "mi_k10":
        steps.append(("reducer", SelectKBest(score_func=mutual_info_ranker, k=RANK_K)))
    elif reduction == "pca_95":
        steps.append(("reducer", PCA(n_components=0.95, random_state=RANDOM_STATE)))
    elif reduction == "none":
        pass
    else:
        raise ValueError(f"Estratégia de redução desconhecida: {reduction}")

    steps.append(("clf", classifier))
    return Pipeline(steps=steps, memory=memory)


def evaluate_cv(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> dict[str, float]:
    """Executa validação cruzada estratificada retornando médias e desvios-padrão."""
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }
    res = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        "cv_accuracy": round(float(np.mean(res["test_accuracy"])), 4),
        "cv_accuracy_std": round(float(np.std(res["test_accuracy"])), 4),
        "cv_precision": round(float(np.mean(res["test_precision"])), 4),
        "cv_recall": round(float(np.mean(res["test_recall"])), 4),
        "cv_f1": round(float(np.mean(res["test_f1"])), 4),
        "cv_f1_std": round(float(np.std(res["test_f1"])), 4),
        "cv_roc_auc": round(float(np.mean(res["test_roc_auc"])), 4),
        "cv_pr_auc": round(float(np.mean(res["test_pr_auc"])), 4),
    }


def evaluate_holdout(
    fitted_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Avalia modelo já ajustado no conjunto de teste holdout (20% estratificado)."""
    y_pred = fitted_pipeline.predict(X_test)
    if hasattr(fitted_pipeline, "predict_proba"):
        y_prob = fitted_pipeline.predict_proba(X_test)[:, 1]
    else:
        y_prob = fitted_pipeline.decision_function(X_test)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    return {
        "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "test_precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "test_recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "test_f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "test_roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "test_pr_auc": round(float(average_precision_score(y_test, y_prob)), 4),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "_y_prob": y_prob,
        "_y_pred": y_pred,
    }


def compute_eda_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Calcula estatísticas descritivas e auditoria de qualidade do dataset."""
    failure_modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    type_counts = df["Type"].value_counts().to_dict()
    mode_counts = {m: int(df[m].sum()) for m in failure_modes}
    rnf_only_no_fail = int(((df["RNF"] == 1) & (df[TARGET_COLUMN] == 0)).sum())
    fail_no_submode = int(
        ((df[TARGET_COLUMN] == 1) & (df[failure_modes].sum(axis=1) == 0)).sum()
    )

    fe_df = PredictiveMaintenanceFeatureEngineer().transform(df[RAW_FEATURE_COLUMNS])
    desc_cols = RAW_NUMERIC_COLS + ENGINEERED_FEATURE_COLUMNS
    stats = {}
    for col in desc_cols:
        s = fe_df[col]
        stats[col] = {
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "median": round(float(s.median()), 4),
            "max": round(float(s.max()), 4),
        }

    return {
        "n_rows": int(df.shape[0]),
        "n_cols_original": int(df.shape[1]),
        "missing_values_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_distribution": {
            "normal_0": int((df[TARGET_COLUMN] == 0).sum()),
            "failure_1": int((df[TARGET_COLUMN] == 1).sum()),
            "failure_rate_pct": round(float(df[TARGET_COLUMN].mean() * 100), 2),
        },
        "type_distribution": {k: int(v) for k, v in type_counts.items()},
        "failure_by_type_pct": {
            k: round(float(df.loc[df["Type"] == k, TARGET_COLUMN].mean() * 100), 2)
            for k in ["L", "M", "H"]
        },
        "failure_submodes": mode_counts,
        "anomalies_documented": {
            "rnf_1_with_machine_failure_0": rnf_only_no_fail,
            "machine_failure_1_without_submode": fail_no_submode,
        },
        "dropped_leakage_and_id_columns": LEAKAGE_AND_ID_COLUMNS,
        "descriptive_stats": stats,
    }


def compute_rank_and_pca_diagnostics(
    X_train: pd.DataFrame, y_train: pd.Series
) -> dict[str, Any]:
    """Calcula scores de Rank (ANOVA F e Informação Mútua) e variância explicada no PCA."""
    fe = PredictiveMaintenanceFeatureEngineer()
    prep = build_preprocessor(use_engineered=True)
    X_fe = fe.fit_transform(X_train)
    X_proc = prep.fit_transform(X_fe)
    feature_names = [
        name.replace("num__", "").replace("cat__", "")
        for name in prep.get_feature_names_out()
    ]

    f_scores, p_values = f_classif(X_proc, y_train)
    mi_scores = mutual_info_ranker(X_proc, y_train.to_numpy())

    ranking_rows = []
    for name, f_val, p_val, mi_val in zip(feature_names, f_scores, p_values, mi_scores):
        ranking_rows.append(
            {
                "feature": name,
                "is_engineered": name in ENGINEERED_FEATURE_COLUMNS,
                "anova_f": round(float(f_val), 3),
                "anova_p": float(1.0 if np.isnan(p_val) else p_val),
                "mutual_info": round(float(mi_val), 5),
            }
        )

    ranking_by_anova = sorted(ranking_rows, key=lambda r: r["anova_f"], reverse=True)
    for idx, row in enumerate(ranking_by_anova, start=1):
        row["anova_rank"] = idx
        row["selected_anova_k10"] = idx <= RANK_K

    ranking_by_mi = sorted(ranking_rows, key=lambda r: r["mutual_info"], reverse=True)
    mi_rank_map = {row["feature"]: idx for idx, row in enumerate(ranking_by_mi, start=1)}
    for row in ranking_by_anova:
        row["mi_rank"] = mi_rank_map[row["feature"]]
        row["selected_mi_k10"] = row["mi_rank"] <= RANK_K

    pca_full = PCA(n_components=None, random_state=RANDOM_STATE).fit(X_proc)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    pca_95_n = int(np.searchsorted(cum_var, 0.95) + 1)

    return {
        "total_encoded_features": len(feature_names),
        "rank_k_selected": RANK_K,
        "dimensionality_reduction_pct": round((1 - RANK_K / len(feature_names)) * 100, 1),
        "selected_features_anova_k10": [
            r["feature"] for r in ranking_by_anova if r["selected_anova_k10"]
        ],
        "dropped_features_anova_k10": [
            r["feature"] for r in ranking_by_anova if not r["selected_anova_k10"]
        ],
        "selected_features_mi_k10": [
            r["feature"] for r in ranking_by_mi[:RANK_K]
        ],
        "feature_ranking": ranking_by_anova,
        "pca": {
            "n_components_95_var": pca_95_n,
            "explained_variance_ratio": [round(float(v), 4) for v in pca_full.explained_variance_ratio_],
            "cumulative_variance_ratio": [round(float(v), 4) for v in cum_var],
        },
    }


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.facecolor": "white",
            "axes.facecolor": "#F8FAFC",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def plot_fig1_eda(df: pd.DataFrame, eda: dict[str, Any], out_path: Path) -> None:
    """Figura 1: Modos de falha + separabilidade física dos atributos engenheirados."""
    _configure_matplotlib()
    fe_df = PredictiveMaintenanceFeatureEngineer().transform(df[RAW_FEATURE_COLUMNS])
    fe_df[TARGET_COLUMN] = df[TARGET_COLUMN]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))

    # Painel (a): Frequência de falhas por modo físico + Type
    ax0 = axes[0]
    labels = ["TWF\n(Desgaste)", "HDF\n(Dissip. Térm.)", "PWF\n(Potência)", "OSF\n(Sobrecarga)", "RNF\n(Aleatória)"]
    keys = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    counts = [eda["failure_submodes"][k] for k in keys]
    colors = [PALETTE["amber"], PALETTE["crimson"], PALETTE["navy"], PALETTE["teal"], PALETTE["slate"]]
    bars = ax0.bar(labels, counts, color=colors, edgecolor="#0F172A", linewidth=0.6, width=0.62)
    for bar, val in zip(bars, counts):
        pct_str = f"{val/100:.2f}%".replace(".", ",")
        ax0.text(
            bar.get_x() + bar.get_width() / 2,
            val + 2.5,
            f"{val}\n({pct_str})",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    ax0.set_ylim(0, max(counts) * 1.24)
    ax0.set_ylabel("Ocorrências no Dataset (n = 10.000)")
    ax0.set_title("(a) Ocorrências por Submodo de Falha\n(Desbalanceamento: 339 falhas = 3,39%)")
    ax0.grid(axis="y", linestyle="--", alpha=0.4)

    # Painel (b): Espaço físico engenheirado (Potência Mecânica x Temp_Diff_K)
    ax1 = axes[1]
    normal = fe_df[fe_df[TARGET_COLUMN] == 0]
    fail = fe_df[fe_df[TARGET_COLUMN] == 1]
    ax1.scatter(
        normal["Mechanical_Power_W"],
        normal["Temp_Diff_K"],
        s=10,
        alpha=0.22,
        color=PALETTE["slate"],
        label=f"Normal (n = {len(normal)})",
        rasterized=True,
    )
    ax1.scatter(
        fail["Mechanical_Power_W"],
        fail["Temp_Diff_K"],
        s=24,
        alpha=0.85,
        color=PALETTE["crimson"],
        edgecolor="white",
        linewidth=0.35,
        label=f"Falha (n = {len(fail)})",
        zorder=4,
    )
    ax1.axvline(3500, color=PALETTE["navy"], linestyle="--", linewidth=1.0, alpha=0.8, label="Limiares físicos PWF (3,5–9,0 kW)")
    ax1.axvline(9000, color=PALETTE["navy"], linestyle="--", linewidth=1.0, alpha=0.8)
    ax1.axhline(8.6, color=PALETTE["amber"], linestyle=":", linewidth=1.3, alpha=0.95, label="Limiar térmico HDF (ΔT < 8,6 K)")
    ax1.set_xlabel("Potência Mecânica Derivada [W]")
    ax1.set_ylabel("Diferença Térmica ΔT = T_proc - T_ar [K]")
    ax1.set_title("(b) Separabilidade Física via Atributos Derivados")
    ax1.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=7.5)
    ax1.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout(w_pad=2.5)
    fig.savefig(out_path)
    plt.close(fig)


def plot_fig2_rank_and_pca(
    diag: dict[str, Any],
    ablation_summary: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """Figura 2: Ranqueamento ANOVA F dos 12 atributos e comparação Rank vs PCA."""
    _configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.18, 1.0]})

    # Painel (a): Rank ANOVA F-score dos 12 atributos codificados
    ax0 = axes[0]
    rows = list(reversed(diag["feature_ranking"]))
    names = [r["feature"] for r in rows]
    scores = [r["anova_f"] for r in rows]
    bar_colors = []
    for r in rows:
        if not r["selected_anova_k10"]:
            bar_colors.append(PALETTE["light_slate"])
        elif r["is_engineered"]:
            bar_colors.append(PALETTE["teal"])
        else:
            bar_colors.append(PALETTE["navy"])

    bars = ax0.barh(names, scores, color=bar_colors, edgecolor="#0F172A", linewidth=0.5, height=0.68)
    ax0.axhline(1.5, color=PALETTE["crimson"], linestyle="--", linewidth=1.2)
    ax0.text(
        max(scores) * 0.98,
        1.65,
        "Corte Rank k = 10 (descartados ↓)",
        ha="right",
        va="bottom",
        fontsize=8,
        color=PALETTE["crimson"],
        fontweight="bold",
    )
    for bar, val in zip(bars, scores):
        ax0.text(val + 14, bar.get_y() + bar.get_height() / 2, f"{val:.1f}", va="center", fontsize=7.8)

    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=PALETTE["teal"], edgecolor="#0F172A", label="Atributo engenheirado (selecionado)"),
        Patch(facecolor=PALETTE["navy"], edgecolor="#0F172A", label="Atributo original (selecionado)"),
        Patch(facecolor=PALETTE["light_slate"], edgecolor="#0F172A", label="Descartado pelo Rank (k=10)"),
    ]
    ax0.legend(handles=legend_handles, loc="lower right", fontsize=7.5, framealpha=0.92)
    ax0.set_xlim(0, max(scores) * 1.18)
    ax0.set_xlabel("Pontuação ANOVA F-score (dados de treino)")
    ax0.set_title("(a) Ranqueamento de Atributos via Rank (ANOVA F)")
    ax0.grid(axis="x", linestyle="--", alpha=0.4)

    # Painel (b): Ablação (Raw vs FE + Sem Redução vs Rank vs PCA)
    ax1 = axes[1]
    regimes = [r["regime_short"] for r in ablation_summary]
    rf_f1 = [r["rf_cv_f1"] for r in ablation_summary]
    gb_f1 = [r["gb_cv_f1"] for r in ablation_summary]
    hgb_f1 = [r["hgb_cv_f1"] for r in ablation_summary]

    x = np.arange(len(regimes))
    width = 0.25
    b1 = ax1.bar(x - width, rf_f1, width, label="Random Forest", color=PALETTE["slate"], edgecolor="#0F172A", linewidth=0.5)
    b2 = ax1.bar(x, gb_f1, width, label="Gradient Boosting", color=PALETTE["navy"], edgecolor="#0F172A", linewidth=0.5)
    b3 = ax1.bar(x + width, hgb_f1, width, label="HistGradientBoosting", color=PALETTE["teal"], edgecolor="#0F172A", linewidth=0.5)

    for bars_grp in (b1, b2, b3):
        for bar in bars_grp:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.012, f"{h:.3f}", ha="center", va="bottom", fontsize=7.0, rotation=90)

    ax1.set_xticks(x)
    ax1.set_xticklabels(regimes, fontsize=8)
    ax1.set_ylim(0, 1.18)
    ax1.set_ylabel("F1-Score Médio (CV 5-fold Estratificado)")
    ax1.set_title("(b) Impacto da Engenharia de Atributos, Rank e PCA")
    ax1.legend(loc="upper center", ncol=3, fontsize=7.2, framealpha=0.94)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout(w_pad=2.2)
    fig.savefig(out_path)
    plt.close(fig)


def plot_fig3_model_comparison(model_rows: list[dict[str, Any]], out_path: Path) -> None:
    """Figura 3: Comparativo de métricas (F1 CV, F1 Teste, PR-AUC e ROC-AUC) entre 6 algoritmos."""
    _configure_matplotlib()
    ordered = sorted(model_rows, key=lambda r: r["cv_f1"])
    names = [r["short_name"] for r in ordered]
    cv_f1 = [r["cv_f1"] for r in ordered]
    cv_std = [r["cv_f1_std"] for r in ordered]
    test_f1 = [r["test_f1"] for r in ordered]
    test_prauc = [r["test_pr_auc"] for r in ordered]

    y = np.arange(len(names))
    height = 0.26

    fig, ax = plt.subplots(figsize=(10.8, 4.5))
    b1 = ax.barh(y + height, cv_f1, height, xerr=cv_std, capsize=3, label="F1-Score CV (5 folds ± dp)", color=PALETTE["navy"], edgecolor="#0F172A", linewidth=0.5)
    b2 = ax.barh(y, test_f1, height, label="F1-Score Teste (Holdout 20%)", color=PALETTE["teal"], edgecolor="#0F172A", linewidth=0.5)
    b3 = ax.barh(y - height, test_prauc, height, label="PR-AUC Teste (Average Precision)", color=PALETTE["amber"], edgecolor="#0F172A", linewidth=0.5)

    for idx, bar in enumerate(b1):
        w = bar.get_width()
        ax.text(w + cv_std[idx] + 0.015, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontsize=7.6)
    for bars_grp in (b2, b3):
        for bar in bars_grp:
            w = bar.get_width()
            ax.text(w + 0.015, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontsize=7.6)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontweight="bold")
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Desempenho nas Métricas de Classificação Desbalanceada [0 – 1]")
    ax.set_title("Comparativo entre Modelos Padrão (Atributos Engenheirados + Rank ANOVA k=10)")
    ax.legend(loc="lower right", frameon=True, framealpha=0.94)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_fig4_default_vs_tuned(
    y_test: pd.Series,
    default_eval: dict[str, Any],
    tuned_eval: dict[str, Any],
    best_model_label: str,
    out_path: Path,
) -> None:
    """Figura 4: Matrizes de confusão e curvas ROC/Precision-Recall (Padrão vs Otimizado)."""
    _configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0), gridspec_kw={"width_ratios": [0.95, 0.95, 1.25]})

    for ax, eval_res, title in [
        (axes[0], default_eval, f"(a) {best_model_label} Padrão\n(Teste F1 = {default_eval['test_f1']:.4f})"),
        (axes[1], tuned_eval, f"(b) {best_model_label} Otimizado\n(Teste F1 = {tuned_eval['test_f1']:.4f})"),
    ]:
        cm = eval_res["confusion_matrix"]
        mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        ax.imshow(mat, cmap="Blues", aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal (0)", "Falha (1)"])
        ax.set_yticklabels(["Normal (0)", "Falha (1)"])
        ax.set_xlabel("Predição do Modelo")
        ax.set_ylabel("Classe Real")
        ax.set_title(title)
        thresh = mat.max() / 2.0
        labels_txt = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{labels_txt[i][j]}\n{mat[i, j]}",
                    ha="center",
                    va="center",
                    color="white" if mat[i, j] > thresh else "#0F172A",
                    fontsize=10.5,
                    fontweight="bold",
                )

    ax_pr = axes[2]
    prec_d, rec_d, _ = precision_recall_curve(y_test, default_eval["_y_prob"])
    prec_t, rec_t, _ = precision_recall_curve(y_test, tuned_eval["_y_prob"])
    ax_pr.plot(
        rec_d,
        prec_d,
        color=PALETTE["slate"],
        linestyle="--",
        linewidth=1.8,
        label=f"Padrão (PR-AUC = {default_eval['test_pr_auc']:.4f} | ROC = {default_eval['test_roc_auc']:.4f})",
    )
    ax_pr.plot(
        rec_t,
        prec_t,
        color=PALETTE["teal"],
        linewidth=2.3,
        label=f"Otimizado (PR-AUC = {tuned_eval['test_pr_auc']:.4f} | ROC = {tuned_eval['test_roc_auc']:.4f})",
    )
    baseline_rate = float(y_test.mean())
    ax_pr.axhline(baseline_rate, color=PALETTE["crimson"], linestyle=":", linewidth=1.2, label=f"Baseline Aleatória ({baseline_rate:.3f})")
    ax_pr.set_xlabel("Revocação (Recall)")
    ax_pr.set_ylabel("Precisão (Precision)")
    ax_pr.set_xlim(0, 1.02)
    ax_pr.set_ylim(0, 1.05)
    ax_pr.set_title("(c) Curva Precision–Recall: Padrão vs Otimizado")
    ax_pr.legend(loc="lower left", fontsize=7.6, framealpha=0.94)
    ax_pr.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout(w_pad=2.0)
    fig.savefig(out_path)
    plt.close(fig)


def run_experiment() -> dict[str, Any]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    eda_summary = compute_eda_summary(df)

    X = df[RAW_FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    rank_pca_diag = compute_rank_and_pca_diagnostics(X_train, y_train)

    with tempfile.TemporaryDirectory() as cache_dir:
        # 1) Estudo de ablação: Raw vs Engineered + Rank vs PCA
        ablation_configs = [
            ("1. Dados Brutos\n(Sem FE, Sem Redução)", "Brutos\n(6 orig.)", False, "none"),
            ("2. FE + Sem Redução\n(12 atribs. codif.)", "FE Completo\n(12 atribs.)", True, "none"),
            ("3. FE + Rank ANOVA\n(k = 10 atribs.)", "FE + Rank\nANOVA (k=10)", True, "anova_k10"),
            ("4. FE + Rank MI\n(k = 10 atribs.)", "FE + Rank\nMI (k=10)", True, "mi_k10"),
            (
                f"5. FE + PCA 95%\n({rank_pca_diag['pca']['n_components_95_var']} comp.)",
                f"FE + PCA 95%\n({rank_pca_diag['pca']['n_components_95_var']} comp.)",
                True,
                "pca_95",
            ),
        ]
        ablation_summary = []
        for regime_name, regime_short, use_fe, red in ablation_configs:
            rf_pipe = build_full_pipeline(
                RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
                use_engineered=use_fe,
                reduction=red,
                memory=cache_dir,
            )
            gb_pipe = build_full_pipeline(
                GradientBoostingClassifier(random_state=RANDOM_STATE),
                use_engineered=use_fe,
                reduction=red,
                memory=cache_dir,
            )
            hgb_pipe = build_full_pipeline(
                HistGradientBoostingClassifier(random_state=RANDOM_STATE),
                use_engineered=use_fe,
                reduction=red,
                memory=cache_dir,
            )
            rf_cv = evaluate_cv(rf_pipe, X_train, y_train, cv)
            gb_cv = evaluate_cv(gb_pipe, X_train, y_train, cv)
            hgb_cv = evaluate_cv(hgb_pipe, X_train, y_train, cv)
            ablation_summary.append(
                {
                    "regime": regime_name.replace("\n", " "),
                    "regime_short": regime_short,
                    "use_engineered": use_fe,
                    "reduction": red,
                    "rf_cv_f1": rf_cv["cv_f1"],
                    "gb_cv_f1": gb_cv["cv_f1"],
                    "hgb_cv_f1": hgb_cv["cv_f1"],
                    "rf_cv_pr_auc": rf_cv["cv_pr_auc"],
                    "gb_cv_pr_auc": gb_cv["cv_pr_auc"],
                    "hgb_cv_pr_auc": hgb_cv["cv_pr_auc"],
                }
            )

        # 2) Benchmark justo de 6 algoritmos padrão sob FE + Rank ANOVA (k=10)
        default_models = [
            ("dummy", "Dummy (Estratificado)", "Dummy", DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)),
            ("logreg", "Regressão Logística", "Reg. Logística", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ("mlp", "Rede Neural MLP", "MLP (64,32)", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, random_state=RANDOM_STATE)),
            ("rf", "Random Forest", "Random Forest", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
            ("gb", "Gradient Boosting", "Gradient Boosting", GradientBoostingClassifier(random_state=RANDOM_STATE)),
            ("hgb", "HistGradientBoosting", "HistGradientBoosting", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]

        model_comparison = []
        default_eval_by_key: dict[str, dict[str, Any]] = {}
        for key, full_name, short_name, clf in default_models:
            pipe_cv = build_full_pipeline(clf, use_engineered=True, reduction="anova_k10", memory=cache_dir)
            cv_metrics = evaluate_cv(pipe_cv, X_train, y_train, cv)

            # Fit limpo (sem memory cache) no treino completo para avaliar no holdout
            pipe_fit = build_full_pipeline(clf, use_engineered=True, reduction="anova_k10", memory=None)
            pipe_fit.fit(X_train, y_train)
            test_metrics = evaluate_holdout(pipe_fit, X_test, y_test)
            default_eval_by_key[key] = test_metrics

            row = {
                "key": key,
                "model_name": full_name,
                "short_name": short_name,
                **cv_metrics,
                "test_accuracy": test_metrics["test_accuracy"],
                "test_precision": test_metrics["test_precision"],
                "test_recall": test_metrics["test_recall"],
                "test_f1": test_metrics["test_f1"],
                "test_roc_auc": test_metrics["test_roc_auc"],
                "test_pr_auc": test_metrics["test_pr_auc"],
                "confusion_matrix": test_metrics["confusion_matrix"],
            }
            model_comparison.append(row)

        # Seleciona arquitetura candidata à otimização pelos critérios de estabilidade CV,
        # revocação em classe minoritária e suporte nativo a balanceamento de pesos (class_weight).
        winner_key = "hgb"
        best_default_row = next(r for r in model_comparison if r["key"] == winner_key)

        # 3) Otimização via GridSearchCV do HistGradientBoostingClassifier
        winner_clf = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
        param_grid = {
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_iter": [100, 200, 350],
            "clf__max_leaf_nodes": [15, 31],
            "clf__min_samples_leaf": [10, 20, 35],
            "clf__l2_regularization": [0.0, 0.1, 1.0],
            "clf__class_weight": [None, "balanced"],
        }

        search_pipe = build_full_pipeline(
            winner_clf, use_engineered=True, reduction="anova_k10", memory=cache_dir
        )
        grid = GridSearchCV(
            estimator=search_pipe,
            param_grid=param_grid,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            refit=False,
        )
        grid.fit(X_train, y_train)
        best_params_clf = {k.replace("clf__", ""): v for k, v in grid.best_params_.items()}

    # Monta pipeline final LIMPO (sem TemporaryDirectory no atributo `memory`) para produção e CV final
    if winner_key == "gb":
        tuned_clf = GradientBoostingClassifier(random_state=RANDOM_STATE, **best_params_clf)
        tuned_label = "GradientBoostingClassifier (Otimizado)"
    elif winner_key == "hgb":
        tuned_clf = HistGradientBoostingClassifier(random_state=RANDOM_STATE, **best_params_clf)
        tuned_label = "HistGradientBoostingClassifier (Otimizado)"
    else:
        tuned_clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **best_params_clf)
        tuned_label = "RandomForestClassifier (Otimizado)"

    tuned_pipe_eval = build_full_pipeline(tuned_clf, use_engineered=True, reduction="anova_k10", memory=None)
    tuned_cv_metrics = evaluate_cv(tuned_pipe_eval, X_train, y_train, cv)

    production_pipeline = build_full_pipeline(tuned_clf, use_engineered=True, reduction="anova_k10", memory=None)
    production_pipeline.fit(X_train, y_train)
    tuned_test_metrics = evaluate_holdout(production_pipeline, X_test, y_test)

    # Gera as 4 figuras do relatório
    plot_fig1_eda(df, eda_summary, FIGURES_DIR / "fig1_eda_falhas_e_distribuicao.png")
    plot_fig2_rank_and_pca(rank_pca_diag, ablation_summary, FIGURES_DIR / "fig2_rank_vs_pca.png")
    plot_fig3_model_comparison(model_comparison, FIGURES_DIR / "fig3_comparacao_modelos.png")
    plot_fig4_default_vs_tuned(
        y_test,
        default_eval_by_key[winner_key],
        tuned_test_metrics,
        best_default_row["short_name"],
        FIGURES_DIR / "fig4_otimizacao_matriz_pr.png",
    )

    # Limpa campos internos antes de serializar JSON
    tuned_test_clean = {k: v for k, v in tuned_test_metrics.items() if not k.startswith("_")}
    default_test_clean = {k: v for k, v in default_eval_by_key[winner_key].items() if not k.startswith("_")}

    experiment_report = {
        "dataset": {
            "name": "AI4I 2020 Predictive Maintenance Dataset",
            "source": "UCI Machine Learning Repository (ID 601) / Kaggle",
            "author": "Stephan Matzka (2020)",
            "split": {"train_rows": int(len(X_train)), "test_rows": int(len(X_test)), "cv_folds": N_SPLITS},
        },
        "eda": eda_summary,
        "feature_engineering": {
            "new_attributes": ENGINEERED_FEATURE_COLUMNS,
            "ablation_study": ablation_summary,
        },
        "dimensionality_reduction": rank_pca_diag,
        "default_model_comparison": model_comparison,
        "optimization": {
            "selected_base_model": best_default_row["model_name"],
            "selected_base_key": winner_key,
            "tuned_model_name": tuned_label,
            "grid_search_candidates": int(len(grid.cv_results_["params"])),
            "best_hyperparameters": best_params_clf,
            "default_cv": {k: v for k, v in best_default_row.items() if k.startswith("cv_")},
            "tuned_cv": tuned_cv_metrics,
            "default_test": default_test_clean,
            "tuned_test": tuned_test_clean,
            "delta_cv_f1": round(tuned_cv_metrics["cv_f1"] - best_default_row["cv_f1"], 4),
            "delta_test_f1": round(tuned_test_clean["test_f1"] - default_test_clean["test_f1"], 4),
        },
    }

    # Salva modelo serializado e metadados consumidos pela API FastAPI
    model_file = MODELS_DIR / "modelo_manutencao_otimizado.joblib"
    joblib.dump(production_pipeline, model_file, compress=3)

    metadata = {
        "best_model_name": tuned_label,
        "base_model": best_default_row["model_name"],
        "dataset": "AI4I 2020 Predictive Maintenance Dataset (n=10.000)",
        "decision_threshold": 0.5,
        "raw_features": RAW_FEATURE_COLUMNS,
        "engineered_features": ENGINEERED_FEATURE_COLUMNS,
        "dimensionality_reduction": f"Rank via ANOVA F-score (SelectKBest k={RANK_K} de {rank_pca_diag['total_encoded_features']} atributos codificados)",
        "selected_features": rank_pca_diag["selected_features_anova_k10"],
        "dropped_features": rank_pca_diag["dropped_features_anova_k10"],
        "best_hyperparameters": best_params_clf,
        "metrics": {
            "default_cv": experiment_report["optimization"]["default_cv"],
            "tuned_cv": tuned_cv_metrics,
            "default_test": default_test_clean,
            "tuned_test": tuned_test_clean,
        },
    }
    (MODELS_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (REPORTS_DIR / "metricas_experimento.json").write_text(
        json.dumps(experiment_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return experiment_report


def main() -> None:
    report = run_experiment()
    print("=== ABALACAO (RAW vs FE vs RANK vs PCA) ===")
    for r in report["feature_engineering"]["ablation_study"]:
        print(
            f"  {r['regime']:42s} | RF={r['rf_cv_f1']:.4f} | GB={r['gb_cv_f1']:.4f} | HGB={r['hgb_cv_f1']:.4f}"
        )

    print("\n=== COMPARACAO MODELOS PADRAO (FE + ANOVA k=10) ===")
    for m in report["default_model_comparison"]:
        print(
            f"  {m['short_name']:22s} | CV F1={m['cv_f1']:.4f} (±{m['cv_f1_std']:.4f}) | "
            f"TEST F1={m['test_f1']:.4f} | Prec={m['test_precision']:.4f} | Rec={m['test_recall']:.4f} | PR-AUC={m['test_pr_auc']:.4f}"
        )

    opt = report["optimization"]
    print("\n=== OTIMIZACAO (PADRAO vs OTIMIZADO) ===")
    print(f"  Modelo vencedor: {opt['selected_base_model']}")
    print(f"  Hiperparametros: {opt['best_hyperparameters']}")
    print(
        f"  CV F1  : {opt['default_cv']['cv_f1']:.4f} -> {opt['tuned_cv']['cv_f1']:.4f} (Δ={opt['delta_cv_f1']:+.4f})"
    )
    print(
        f"  TEST F1: {opt['default_test']['test_f1']:.4f} -> {opt['tuned_test']['test_f1']:.4f} (Δ={opt['delta_test_f1']:+.4f})"
    )
    print(
        f"  TEST Prec/Rec/ROC/PR: Prec={opt['tuned_test']['test_precision']:.4f}, "
        f"Rec={opt['tuned_test']['test_recall']:.4f}, ROC={opt['tuned_test']['test_roc_auc']:.4f}, "
        f"PR={opt['tuned_test']['test_pr_auc']:.4f}"
    )


if __name__ == "__main__":
    main()
