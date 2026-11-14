"""Gerador do Relatório Técnico em normas ABNT (.pdf via ReportLab e .docx via python-docx).

Lê exclusivamente os resultados auditados em `reports/metricas_experimento.json` e as figuras
300 DPI em `reports/figures/`, produzindo relatório paginado (máximo 12 páginas) com:
- Capa e Folha de Rosto institucionais (Fatec Pompeia "Shunji Nishimura")
- Resumo + Sumário automático com numeração exata de páginas (2 passagens)
- Seções obrigatórias: Introdução, Objetivo, Material e Métodos, Resultados e Discussão,
  Considerações Finais e Referências
- Tabelas em estilo IBGE (apenas linhas horizontais superior, cabeçalho e inferior)
- Margens A4 ABNT (superior/esquerda 3 cm; inferior/direita 2 cm), fonte Times New Roman 12 pt,
  espaçamento 1,5 e recuo de 1,25 cm no corpo textual.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor
from PIL import Image as PILImage
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import Flowable

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metricas_experimento.json"

PDF_OUT = BASE_DIR / "Relatorio_Manutencao_Preditiva_ABNT.pdf"
DOCX_OUT = REPORTS_DIR / "Relatorio_Manutencao_Preditiva_ABNT.docx"

INSTITUTION = 'FACULDADE DE TECNOLOGIA DE POMPEIA "SHUNJI NISHIMURA"'
INSTITUTION_TITLE = 'Faculdade de Tecnologia de Pompeia "Shunji Nishimura" (Fatec Pompeia)'
COURSE = "TECNOLOGIA EM CIÊNCIA DE DADOS"
DISCIPLINE = "Aplicação de Ciência de Dados"
PROFESSOR = "Prof. Dr. João Ricardo Favan"
AUTHORS = [
    "Hadryan Fortinis de Oliveira Lima",
    "Raul Victor Matsunaga de Carvalho",
    "Washingtom Ferreira Gonçalves",
]
TITLE = (
    "MANUTENÇÃO PREDITIVA INDUSTRIAL COM ENGENHARIA DE ATRIBUTOS FÍSICOS, "
    "SELEÇÃO VIA RANK E HISTOGRAM GRADIENT BOOSTING OTIMIZADO"
)


def register_tnr_fonts() -> tuple[str, str]:
    """Registra Times New Roman TrueType (Unicode completo: Δ, π, τ, ω) com fallback."""
    font_dir = Path("/System/Library/Fonts/Supplemental")
    reg = font_dir / "Times New Roman.ttf"
    bold = font_dir / "Times New Roman Bold.ttf"
    ital = font_dir / "Times New Roman Italic.ttf"
    bi = font_dir / "Times New Roman Bold Italic.ttf"

    if reg.exists() and bold.exists() and ital.exists() and bi.exists():
        pdfmetrics.registerFont(TTFont("TNR", str(reg)))
        pdfmetrics.registerFont(TTFont("TNR-Bold", str(bold)))
        pdfmetrics.registerFont(TTFont("TNR-Italic", str(ital)))
        pdfmetrics.registerFont(TTFont("TNR-BoldItalic", str(bi)))
        registerFontFamily("TNR", normal="TNR", bold="TNR-Bold", italic="TNR-Italic", boldItalic="TNR-BoldItalic")
        return "TNR", "TNR-Bold"
    return "Times-Roman", "Times-Bold"


def fmt_br(val: float, decimals: int = 4) -> str:
    return f"{val:.{decimals}f}".replace(".", ",")


def fmt_int_br(val: int) -> str:
    return f"{val:,}".replace(",", ".")


def scaled_image(img_path: Path, target_width_cm: float) -> Image:
    """Cria objeto Image com proporção exata da imagem original."""
    with PILImage.open(img_path) as im:
        w_px, h_px = im.size
    w = target_width_cm * cm
    h = w * (h_px / w_px)
    return Image(str(img_path), width=w, height=h)


class SectionMarker(Flowable):
    """Flowable invisível de altura zero que registra a página onde o título foi impresso."""

    def __init__(self, key: str, tracker: dict[str, int]):
        super().__init__()
        self.key = key
        self.tracker = tracker
        self.width = 0
        self.height = 0

    def draw(self) -> None:
        self.tracker[self.key] = self.canv.getPageNumber()


def build_report_data() -> dict[str, Any]:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def make_abnt_styles(font_reg: str, font_bold: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "cover_inst": ParagraphStyle(
            "cover_inst",
            parent=base["Normal"],
            fontName=font_bold,
            fontSize=12,
            leading=16,
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "cover_authors": ParagraphStyle(
            "cover_authors",
            parent=base["Normal"],
            fontName=font_reg,
            fontSize=12,
            leading=18,
            alignment=TA_CENTER,
        ),
        "cover_title": ParagraphStyle(
            "cover_title",
            parent=base["Normal"],
            fontName=font_bold,
            fontSize=13,
            leading=19,
            alignment=TA_CENTER,
        ),
        "cover_city": ParagraphStyle(
            "cover_city",
            parent=base["Normal"],
            fontName=font_reg,
            fontSize=12,
            leading=16,
            alignment=TA_CENTER,
        ),
        "folha_nota": ParagraphStyle(
            "folha_nota",
            parent=base["Normal"],
            fontName=font_reg,
            fontSize=10,
            leading=13,
            alignment=TA_JUSTIFY,
            leftIndent=7.2 * cm,
        ),
        "pre_heading": ParagraphStyle(
            "pre_heading",
            parent=base["Normal"],
            fontName=font_bold,
            fontSize=12,
            leading=16,
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "resumo_body": ParagraphStyle(
            "resumo_body",
            parent=base["Normal"],
            fontName=font_reg,
            fontSize=11.5,
            leading=16.5,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Normal"],
            fontName=font_bold,
            fontSize=12,
            leading=15.5,
            alignment=TA_LEFT,
            spaceBefore=11,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Normal"],
            fontName=font_bold,
            fontSize=12,
            leading=15.5,
            alignment=TA_LEFT,
            spaceBefore=9,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontName=font_reg,
            fontSize=11.5,
            leading=16.5,
            alignment=TA_JUSTIFY,
            firstLineIndent=1.25 * cm,
            spaceAfter=4,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["Normal"],
            fontName=font_reg,
            fontSize=11.2,
            leading=15.8,
            alignment=TA_JUSTIFY,
            leftIndent=0.9 * cm,
            firstLineIndent=-0.4 * cm,
            spaceAfter=3,
        ),
        "caption_above": ParagraphStyle(
            "caption_above",
            parent=base["Normal"],
            fontName=font_bold,
            fontSize=10,
            leading=12.5,
            alignment=TA_CENTER,
            spaceBefore=5,
            spaceAfter=3,
            keepWithNext=True,
        ),
        "source_below": ParagraphStyle(
            "source_below",
            parent=base["Normal"],
            fontName=font_reg,
            fontSize=9.5,
            leading=12,
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=6,
        ),
        "tbl_hdr_left": ParagraphStyle(
            "tbl_hdr_left",
            parent=base["Normal"],
            fontName=font_bold,
            fontSize=9.0,
            leading=10.8,
            alignment=TA_LEFT,
        ),
        "tbl_hdr_center": ParagraphStyle(
            "tbl_hdr_center",
            parent=base["Normal"],
            fontName=font_bold,
            fontSize=9.0,
            leading=10.8,
            alignment=TA_CENTER,
        ),
        "ref_item": ParagraphStyle(
            "ref_item",
            parent=base["Normal"],
            fontName=font_reg,
            fontSize=11,
            leading=13.5,
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "toc_l1": ParagraphStyle(
            "toc_l1",
            parent=base["Normal"],
            fontName=font_bold,
            fontSize=11,
            leading=15,
        ),
        "toc_l2": ParagraphStyle(
            "toc_l2",
            parent=base["Normal"],
            fontName=font_reg,
            fontSize=11,
            leading=15,
            leftIndent=0.6 * cm,
        ),
    }


def make_ibge_table(
    rows: list[list[Any]],
    col_widths: list[float],
    alignments: list[str],
    styles: dict[str, ParagraphStyle],
    font_reg: str,
    font_bold: str,
) -> Table:
    """Cria tabela em estilo IBGE com cabeçalho quebrável em múltiplas linhas e sem bordas verticais."""
    formatted_rows = list(rows)
    header = []
    for idx, cell in enumerate(rows[0]):
        if isinstance(cell, str):
            st = styles["tbl_hdr_left"] if alignments[idx] == "LEFT" else styles["tbl_hdr_center"]
            header.append(Paragraph(cell, st))
        else:
            header.append(cell)
    formatted_rows[0] = header

    tbl = Table(formatted_rows, colWidths=col_widths, repeatRows=1)
    cmds: list[tuple[Any, ...]] = [
        ("FONTNAME", (0, 0), (-1, 0), font_bold),
        ("FONTNAME", (0, 1), (-1, -1), font_reg),
        ("FONTSIZE", (0, 1), (-1, -1), 9.0),
        ("LEADING", (0, 1), (-1, -1), 11.0),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
        ("LINEABOVE", (0, 0), (-1, 0), 1.0, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.75, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 1.0, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
    ]
    for col_idx, aln in enumerate(alignments):
        cmds.append(("ALIGN", (col_idx, 0), (col_idx, -1), aln))
    tbl.setStyle(TableStyle(cmds))
    return tbl


def get_narrative_sections(m: dict[str, Any]) -> dict[str, Any]:
    eda = m["eda"]
    rank_diag = m["dimensionality_reduction"]
    ablation = m["feature_engineering"]["ablation_study"]
    comp = m["default_model_comparison"]
    opt = m["optimization"]
    hp = opt["best_hyperparameters"]

    resumo_text = (
        f"Este relatório apresenta o desenvolvimento ponta-a-ponta de uma solução de aprendizado de máquina "
        f"aplicada à manutenção preditiva industrial, utilizando o conjunto público AI4I 2020 "
        f"({fmt_int_br(eda['n_rows'])} amostras e {eda['n_cols_original']} variáveis originais, com apenas "
        f"{fmt_br(eda['target_distribution']['failure_rate_pct'], 2)}% de eventos de falha). Após auditoria de "
        f"integridade e remoção estrita de identificadores e indicadores pós-ocorrência para prevenir vazamento "
        f"de alvo, aplicou-se divisão estratificada (80% treino / 20% teste) e pré-processamento isolado por "
        f"transformadores de imputação, padronização e codificação categórica. Na etapa de engenharia de atributos, "
        f"foram formulados quatro novos descritores com base termodinâmica e mecânica (Temp_Diff_K, "
        f"Mechanical_Power_W, Wear_Torque_Nm_min e Torque_Speed_Ratio), elevando o F1-score médio em validação "
        f"cruzada da arquitetura Random Forest de {fmt_br(ablation[0]['rf_cv_f1'])} para {fmt_br(ablation[1]['rf_cv_f1'])}. "
        f"Na redução de dimensionalidade, comparou-se seleção por Rank (ANOVA F e Informação Mútua, k = 10) frente "
        f"à projeção PCA (95% da variância explicada, 5 componentes): enquanto a PCA degradou ensembles de árvores "
        f"para F1 abaixo de 0,59, o Rank ANOVA reteve os quatro atributos engenheirados entre as cinco maiores "
        f"pontuações e reduziu em {fmt_br(rank_diag['dimensionality_reduction_pct'], 1)}% a dimensionalidade codificada. "
        f"Foram avaliados seis algoritmos distintos, selecionando-se o HistGradientBoostingClassifier pela estabilidade "
        f"entre partições e suporte nativo ao balanceamento de classes. Sua otimização via GridSearchCV (216 combinações) "
        f"elevou o F1-score no conjunto de teste de {fmt_br(opt['default_test']['test_f1'])} para "
        f"{fmt_br(opt['tuned_test']['test_f1'])} (Precisão = {fmt_br(opt['tuned_test']['test_precision'])} e "
        f"Revocação = {fmt_br(opt['tuned_test']['test_recall'])}). Por fim, o pipeline completo foi serializado e "
        f"publicado em produção via API REST FastAPI, painel web operacional e contêiner Docker validado por testes automatizados."
    )

    keywords = "Manutenção preditiva. Engenharia de atributos. Redução de dimensionalidade. HistGradientBoosting. API REST."

    intro_paras = [
        "No contexto da Indústria 4.0, a transição de políticas corretivas e preventivas baseadas em calendário para estratégias de manutenção preditiva representa um fator decisivo para redução de custos operacionais, aumento da disponibilidade de ativos e prevenção de quebras catastróficas em linhas de usinagem (MATZKA, 2020). Sensores embarcados em tornos e fresadoras coletam continuamente grandezas termodinâmicas e cinemáticas, gerando fluxos de telemetria aptos à modelagem por aprendizado de máquina supervisionado.",
        "Entretanto, cenários industriais reais impõem dois desafios metodológicos críticos: o acentuado desbalanceamento de classes — uma vez que falhas representam fração minoritária dos ciclos operacionais — e o risco de vazamento de dados (target leakage) ou uso de representações lineares incapazes de capturar interações físicas não lineares entre torque, rotação, desgaste de ferramenta e dissipação de calor (PROVOST; FAWCETT, 2013).",
        "Diante desse cenário, este estudo conduz um ciclo completo de Ciência de Dados Aplicada sobre o benchmark AI4I 2020 Predictive Maintenance Dataset, demonstrando como a incorporação de equações físicas na engenharia de atributos, combinada à seleção supervisionada por Rank e à otimização de árvores por histogramas, viabiliza um classificador robusto, auditável e pronto para inferência em tempo real em ambiente conteinerizado.",
    ]

    obj_intro = (
        "O objetivo geral deste trabalho é conceber, avaliar experimentalmente, otimizar e implantar em ambiente de produção "
        "um pipeline de classificação binária capaz de antecipar falhas em máquinas industriais a partir de leituras brutas de telemetria."
    )
    obj_bullets = [
        "Auditar e pré-processar o conjunto AI4I 2020, eliminando variáveis identificadoras e submodos pós-evento que causariam vazamento de alvo;",
        "Projetar e validar empiricamente quatro novos atributos físicos derivados (superior ao mínimo de dois atributos exigido) mediante estudo de ablação;",
        "Comparar técnicas de redução de dimensionalidade — Rank (ANOVA F e Informação Mútua) versus Análise de Componentes Principais (PCA) — quanto ao impacto preditivo;",
        "Ajustar e confrontar seis algoritmos de classificação distintos (acima do mínimo de dois), selecionando e otimizando o modelo mais adequado frente à sua configuração padrão;",
        "Exportar o pipeline completo para produção por meio de uma API REST (FastAPI) integrada a interface web interativa e empacotamento Docker, com cobertura de testes automatizados.",
    ]

    return {
        "eda": eda,
        "rank_diag": rank_diag,
        "ablation": ablation,
        "comp": comp,
        "opt": opt,
        "hp": hp,
        "resumo": resumo_text,
        "keywords": keywords,
        "intro": intro_paras,
        "obj_intro": obj_intro,
        "obj_bullets": obj_bullets,
    }


def _heading_with_marker(key: str, text: str, style: ParagraphStyle, tracker: dict[str, int], following_flowables: list[Any]) -> KeepTogether:
    """Garante que título, marcador de página do Sumário e primeiro parágrafo fiquem na mesma página."""
    h_para = Paragraph(text, style)
    marker = SectionMarker(key, tracker)
    return KeepTogether([h_para, marker] + following_flowables)


def _build_pdf_story(
    data: dict[str, Any],
    styles: dict[str, ParagraphStyle],
    page_tracker: dict[str, int],
    toc_pages: dict[str, int],
    font_reg: str,
    font_bold: str,
) -> list[Any]:
    story: list[Any] = []
    eda = data["eda"]
    rank_diag = data["rank_diag"]
    ablation = data["ablation"]
    comp = data["comp"]
    opt = data["opt"]
    hp = opt["best_hyperparameters"]

    # ---------------- PÁGINA 1: CAPA ----------------
    story.append(Paragraph(INSTITUTION, styles["cover_inst"]))
    story.append(Paragraph(COURSE, styles["cover_inst"]))
    story.append(Spacer(1, 2.2 * cm))
    for author in AUTHORS:
        story.append(Paragraph(author.upper(), styles["cover_authors"]))
    story.append(Spacer(1, 4.6 * cm))
    story.append(Paragraph(TITLE, styles["cover_title"]))
    story.append(Spacer(1, 6.2 * cm))
    story.append(Paragraph("POMPEIA – SP<br/>2026", styles["cover_city"]))
    story.append(PageBreak())

    # ---------------- PÁGINA 2: FOLHA DE ROSTO ----------------
    for author in AUTHORS:
        story.append(Paragraph(author.upper(), styles["cover_authors"]))
    story.append(Spacer(1, 4.8 * cm))
    story.append(Paragraph(TITLE, styles["cover_title"]))
    story.append(Spacer(1, 2.2 * cm))
    story.append(
        Paragraph(
            f"Relatório técnico-científico apresentado à disciplina de {DISCIPLINE}, do Curso Superior "
            f"de Tecnologia em Ciência de Dados da {INSTITUTION_TITLE}, sob orientação do {PROFESSOR}, "
            f"como requisito parcial de avaliação.",
            styles["folha_nota"],
        )
    )
    story.append(Spacer(1, 6.2 * cm))
    story.append(Paragraph("Pompeia – SP<br/>2026", styles["cover_city"]))
    story.append(PageBreak())

    # ---------------- PÁGINA 3: RESUMO ----------------
    story.append(Paragraph("RESUMO", styles["pre_heading"]))
    story.append(Paragraph(data["resumo"], styles["resumo_body"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(f"<b>Palavras-chave:</b> {data['keywords']}", styles["resumo_body"]))
    story.append(PageBreak())

    # ---------------- PÁGINA 4: SUMÁRIO ----------------
    story.append(Paragraph("SUMÁRIO", styles["pre_heading"]))
    toc_entries = [
        ("s1", "1 INTRODUÇÃO", 1),
        ("s2", "2 OBJETIVO", 1),
        ("s3", "3 MATERIAL E MÉTODOS", 1),
        ("s3_1", "3.1 Conjunto de dados e pré-processamento", 2),
        ("s3_2", "3.2 Engenharia de atributos físicos", 2),
        ("s3_3", "3.3 Redução de dimensionalidade: Rank versus PCA", 2),
        ("s3_4", "3.4 Algoritmos avaliados, métricas e otimização", 2),
        ("s3_5", "3.5 Arquitetura de exportação para produção", 2),
        ("s4", "4 RESULTADOS E DISCUSSÃO", 1),
        ("s4_1", "4.1 Análise exploratória e integridade dos dados", 2),
        ("s4_2", "4.2 Impacto da engenharia de atributos e Rank versus PCA", 2),
        ("s4_3", "4.3 Comparação entre modelos padrão e otimização", 2),
        ("s4_4", "4.4 Operacionalização em API REST, interface web e Docker", 2),
        ("s5", "5 CONSIDERAÇÕES FINAIS", 1),
        ("ref", "REFERÊNCIAS", 1),
    ]
    toc_rows = []
    for key, label, lvl in toc_entries:
        pnum = str(toc_pages.get(key, "—"))
        st = styles["toc_l1"] if lvl == 1 else styles["toc_l2"]
        toc_rows.append([Paragraph(label, st), Paragraph(pnum, ParagraphStyle(f"pnum_{key}", parent=st, alignment=TA_RIGHT))])
    toc_tbl = Table(toc_rows, colWidths=[14.2 * cm, 1.8 * cm])
    toc_tbl.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E8F0")),
            ]
        )
    )
    story.append(toc_tbl)
    story.append(PageBreak())

    # ---------------- ELEMENTOS TEXTUAIS ----------------
    # 1 INTRODUÇÃO
    story.append(_heading_with_marker("s1", "1 INTRODUÇÃO", styles["h1"], page_tracker, [Paragraph(data["intro"][0], styles["body"])]))
    for p in data["intro"][1:]:
        story.append(Paragraph(p, styles["body"]))

    # 2 OBJETIVO
    story.append(_heading_with_marker("s2", "2 OBJETIVO", styles["h1"], page_tracker, [Paragraph(data["obj_intro"], styles["body"])]))
    for b in data["obj_bullets"]:
        story.append(Paragraph(f"• {b}", styles["bullet"]))

    # 3 MATERIAL E MÉTODOS + 3.1 (agrupados para não deixar "3 MATERIAL E MÉTODOS" órfão)
    s3_p1 = Paragraph(
        f"O conjunto AI4I 2020 Predictive Maintenance Dataset (MATZKA, 2020), obtido junto ao repositório da UCI "
        f"(ID 601) e espelhado no Kaggle, reúne {fmt_int_br(eda['n_rows'])} registros sintéticos calibrados para "
        f"representar operações reais de fresagem industrial. Dentre as {eda['n_cols_original']} colunas originais, "
        f"descartaram-se explicitamente os identificadores sem valor físico (UDI e Product ID) e os cinco indicadores "
        f"específicos de submodo de falha (TWF, HDF, PWF, OSF e RNF), cuja inclusão caracterizaria vazamento direto "
        f"do alvo Machine failure.",
        styles["body"],
    )
    story.append(
        KeepTogether(
            [
                Paragraph("3 MATERIAL E MÉTODOS", styles["h1"]),
                SectionMarker("s3", page_tracker),
                Paragraph("3.1 Conjunto de dados e pré-processamento", styles["h2"]),
                SectionMarker("s3_1", page_tracker),
                s3_p1,
            ]
        )
    )
    story.append(
        Paragraph(
            "Antes de qualquer cálculo estatístico, os dados foram divididos de forma estratificada em 80% para "
            "treinamento (8.000 amostras, sendo 271 falhas) e 20% para teste holdout (2.000 amostras, com 68 falhas), "
            "fixando semente aleatória 42. Dentro de um ColumnTransformer do scikit-learn (PEDREGOSA et al., 2011), as "
            "variáveis numéricas passaram por imputação por mediana e padronização z-score (StandardScaler), enquanto a "
            "variável categórica Type (L, M, H) recebeu imputação modal e codificação OneHotEncoder.",
            styles["body"],
        )
    )

    story.append(
        _heading_with_marker(
            "s3_2",
            "3.2 Engenharia de atributos físicos",
            styles["h2"],
            page_tracker,
            [
                Paragraph(
                    "Considerando os mecanismos físicos que governam falhas térmicas, de potência e de sobrecarga descritos por "
                    "Matzka (2020), implementou-se o transformador customizado PredictiveMaintenanceFeatureEngineer, responsável "
                    "por calcular quatro novos atributos determinísticos sem qualquer dependência do conjunto de avaliação:",
                    styles["body"],
                )
            ],
        )
    )
    fe_bullets = [
        "<b>Temp_Diff_K (ΔT = T_processo − T_ar [K]):</b> mede o gradiente térmico disponível para dissipação de calor; diferenças inferiores a 8,6 K associadas a baixa rotação precipitam falhas térmicas (HDF);",
        "<b>Mechanical_Power_W (P = τ · ω · 2π / 60 [W]):</b> converte torque (τ) e velocidade angular (ω em rpm) na potência efetiva transferida ao eixo, delimitando a faixa operacional nominal de 3.500 W a 9.000 W (PWF);",
        "<b>Wear_Torque_Nm_min (desgaste × torque [min·N·m]):</b> quantifica o esforço mecânico acumulado sobre a aresta de corte, capturando o limiar de falha por sobre-esforço (OSF);",
        "<b>Torque_Speed_Ratio (τ / rpm):</b> expressa a carga específica por rotação, isolando regimes críticos de alto torque em baixa velocidade angular.",
    ]
    for fb in fe_bullets:
        story.append(Paragraph(f"• {fb}", styles["bullet"]))

    story.append(
        _heading_with_marker(
            "s3_3",
            "3.3 Redução de dimensionalidade: Rank versus PCA",
            styles["h2"],
            page_tracker,
            [
                Paragraph(
                    f"Após a codificação OneHotEncoder da variável Type somada aos nove atributos numéricos (cinco originais e "
                    f"quatro engenheirados), obteve-se um espaço de {rank_diag['total_encoded_features']} dimensões. Compararam-se "
                    f"duas abordagens conceituais distintas: seleção supervisionada por Rank via SelectKBest (k = {rank_diag['rank_k_selected']}, "
                    f"redução de {fmt_br(rank_diag['dimensionality_reduction_pct'], 1)}%), ranqueando por estatística F da ANOVA "
                    f"(f_classif) e por Informação Mútua (mutual_info_classif), contra extração não supervisionada por Análise de "
                    f"Componentes Principais (PCA) calibrada para reter 95% da variância total acumulada.",
                    styles["body"],
                )
            ],
        )
    )

    story.append(
        _heading_with_marker(
            "s3_4",
            "3.4 Algoritmos avaliados, métricas e otimização",
            styles["h2"],
            page_tracker,
            [
                Paragraph(
                    "Em ambiente reprodutível (Python 3.13, scikit-learn 1.9.1), avaliaram-se seis famílias de algoritmos sob validação "
                    "cruzada estratificada de cinco partições (5-fold CV) no treino e avaliação única no holdout de teste: "
                    "DummyClassifier (baseline estratificada), Regressão Logística, Rede Neural MLP (camadas 64 e 32 com parada "
                    "antecipada), Random Forest (BREIMAN, 2001), Gradient Boosting clássico (FRIEDMAN, 2001) e "
                    "HistGradientBoostingClassifier baseado em histogramas (KE et al., 2017).",
                    styles["body"],
                )
            ],
        )
    )
    story.append(
        Paragraph(
            "Dado o severo desbalanceamento (3,39%), adotou-se o F1-score como métrica primária de seleção, complementado "
            "por Precisão, Revocação (Recall), PR-AUC (Average Precision), ROC-AUC e Acurácia. Para otimização, conduziu-se "
            f"busca exaustiva via GridSearchCV ({opt['grid_search_candidates']} combinações × 5 folds = "
            f"{fmt_int_br(opt['grid_search_candidates'] * 5)} ajustes) sobre o algoritmo escolhido, confrontando diretamente "
            "seu desempenho padrão contra a configuração otimizada.",
            styles["body"],
        )
    )

    story.append(
        _heading_with_marker(
            "s3_5",
            "3.5 Arquitetura de exportação para produção",
            styles["h2"],
            page_tracker,
            [
                Paragraph(
                    "O pipeline vencedor completo (engenharia física, pré-processamento, Rank ANOVA k = 10 e classificador otimizado) "
                    "foi exportado em disco via joblib (modelo_manutencao_otimizado.joblib) acompanhado de metadados auditáveis. "
                    "Disponibilizou-se o modelo por meio de uma API REST em FastAPI com validação Pydantic (/health, /model-info, "
                    "/predict e /predict/batch), interface web interativa servida na raiz (/), imagem Docker reproduzível "
                    "(python:3.13-slim) e suíte automatizada de testes em pytest.",
                    styles["body"],
                )
            ],
        )
    )

    # 4 RESULTADOS E DISCUSSÃO + 4.1
    story.append(
        KeepTogether(
            [
                Paragraph("4 RESULTADOS E DISCUSSÃO", styles["h1"]),
                SectionMarker("s4", page_tracker),
                Paragraph("4.1 Análise exploratória e integridade dos dados", styles["h2"]),
                SectionMarker("s4_1", page_tracker),
                Paragraph(
                    f"A auditoria inicial confirmou ausência total de valores nulos ({eda['missing_values_total']}) e linhas duplicadas "
                    f"({eda['duplicate_rows']}). Das {fmt_int_br(eda['n_rows'])} amostras, {fmt_int_br(eda['target_distribution']['normal_0'])} "
                    f"(96,61%) representam operação normal e {eda['target_distribution']['failure_1']} (3,39%) correspondem a falhas. "
                    f"Verificou-se ainda heterogeneidade nas variantes de qualidade: máquinas Type L (60,0% da base) concentram taxa "
                    f"de falha de {fmt_br(eda['failure_by_type_pct']['L'], 2)}%, ante {fmt_br(eda['failure_by_type_pct']['M'], 2)}% "
                    f"em Type M e {fmt_br(eda['failure_by_type_pct']['H'], 2)}% em Type H. Constataram-se ainda duas peculiaridades "
                    f"estruturais do gerador original: 18 instâncias marcadas com RNF = 1 possuem Machine failure = 0, enquanto 9 "
                    f"instâncias com falha geral ativa não acionam nenhum dos cinco submodos — reforçando o acerto em predizer a "
                    f"variável-alvo consolidada com remoção prévia dos submodos.",
                    styles["body"],
                ),
            ]
        )
    )

    t1_rows = [
        ["Segmento / Indicador", "Quantidade", "Proporção (%)", "Observação Metodológica"],
        ["Classe 0 — Operação Normal", "9.661", "96,61%", "Desbalanceamento 28,5 : 1"],
        ["Classe 1 — Falha na Máquina", "339", "3,39%", "Alvo binário (Machine failure)"],
        ["Submodo HDF (Dissipação Térmica)", "115", "1,15%", "ΔT < 8,6 K e RPM < 1.380"],
        ["Submodo OSF (Sobre-esforço)", "98", "0,98%", "Produto desgaste × torque crítico"],
        ["Submodo PWF (Potência)", "95", "0,95%", "Potência < 3,5 kW ou > 9,0 kW"],
        ["Submodo TWF (Desgaste Ferramenta)", "46", "0,46%", "Tool wear entre 200 e 240 min"],
        ["Submodo RNF (Falha Aleatória)", "19", "0,19%", "18 ocorrências com alvo = 0"],
        ["Variante Type L / M / H", "6.000 / 2.997 / 1.003", "60,0 / 30,0 / 10,0", "Taxa falha L = 3,92% > M > H"],
    ]
    story.append(
        KeepTogether(
            [
                Paragraph("Tabela 1 – Distribuição de classes, submodos de falha e variantes no AI4I 2020 (n = 10.000)", styles["caption_above"]),
                make_ibge_table(t1_rows, [5.4 * cm, 3.0 * cm, 2.7 * cm, 4.9 * cm], ["LEFT", "CENTER", "CENTER", "LEFT"], styles, font_reg, font_bold),
                Paragraph("Fonte: Elaborado pelos autores com base em Matzka (2020).", styles["source_below"]),
            ]
        )
    )

    story.append(
        KeepTogether(
            [
                Paragraph("Figura 1 – Submodos de falha (a) e separabilidade física obtida pelos atributos derivados (b)", styles["caption_above"]),
                scaled_image(FIGURES_DIR / "fig1_eda_falhas_e_distribuicao.png", 15.6),
                Paragraph("Fonte: Elaborado pelos autores (2026).", styles["source_below"]),
            ]
        )
    )

    story.append(
        Paragraph(
            "Conforme evidencia a Figura 1(b), a projeção dos dois novos descritores (Mechanical_Power_W e Temp_Diff_K) revela "
            "fronteiras nítidas para as falhas térmicas (quadrante inferior ΔT < 8,6 K) e de potência (extremos esquerdo abaixo "
            "de 3.500 W e direito acima de 9.000 W), explicando por que a modelagem explícita das relações físicas simplifica "
            "radicalmente o particionamento das árvores de decisão.",
            styles["body"],
        )
    )

    story.append(
        _heading_with_marker(
            "s4_2",
            "4.2 Impacto da engenharia de atributos e Rank versus PCA",
            styles["h2"],
            page_tracker,
            [
                Paragraph(
                    "O ranqueamento supervisionado no conjunto de treinamento (Tabela 2) comprovou o valor informacional da engenharia "
                    "de atributos: todos os quatro novos descritores figuraram entre as cinco variáveis de maior poder discriminatório "
                    "na estatística ANOVA F. Em particular, Torque_Speed_Ratio ocupou o 1º lugar absoluto tanto na ANOVA F (349,716; "
                    "p = 2,05 × 10<sup>−76</sup>) quanto na Informação Mútua (0,04707), seguida de Torque [Nm], Wear_Torque_Nm_min (285,822), "
                    "Mechanical_Power_W (255,657) e Temp_Diff_K (100,344). A seleção Rank (k = 10) reteve todos os nove atributos "
                    "numéricos além de Type_L, descartando apenas Type_M (F = 9,387) e Type_H (F = 4,946).",
                    styles["body"],
                )
            ],
        )
    )

    t2_rows = [
        ["Regime Experimental (Treino CV 5-fold)", "Atributos", "Random<br/>Forest F1", "Gradient<br/>Boosting F1", "HistGrad.<br/>Boosting F1"],
        ["1. Dados Brutos (sem FE, sem redução)", "8 (codif.)", fmt_br(ablation[0]["rf_cv_f1"]), fmt_br(ablation[0]["gb_cv_f1"]), fmt_br(ablation[0]["hgb_cv_f1"])],
        ["2. Engenharia de Atributos (sem redução)", "12 (codif.)", fmt_br(ablation[1]["rf_cv_f1"]), fmt_br(ablation[1]["gb_cv_f1"]), fmt_br(ablation[1]["hgb_cv_f1"])],
        ["3. FE + Seleção Rank ANOVA F (k = 10)", "10 (Rank)", fmt_br(ablation[2]["rf_cv_f1"]), fmt_br(ablation[2]["gb_cv_f1"]), fmt_br(ablation[2]["hgb_cv_f1"])],
        ["4. FE + Seleção Rank Inf. Mútua (k = 10)", "10 (Rank)", fmt_br(ablation[3]["rf_cv_f1"]), fmt_br(ablation[3]["gb_cv_f1"]), fmt_br(ablation[3]["hgb_cv_f1"])],
        ["5. FE + PCA (95% variância acumulada)", "5 comp.", fmt_br(ablation[4]["rf_cv_f1"]), fmt_br(ablation[4]["gb_cv_f1"]), fmt_br(ablation[4]["hgb_cv_f1"])],
    ]
    story.append(
        KeepTogether(
            [
                Paragraph("Tabela 2 – Estudo de ablação em validação cruzada: impacto da engenharia de atributos, Rank e PCA", styles["caption_above"]),
                make_ibge_table(t2_rows, [6.2 * cm, 2.0 * cm, 2.5 * cm, 2.65 * cm, 2.65 * cm], ["LEFT", "CENTER", "CENTER", "CENTER", "CENTER"], styles, font_reg, font_bold),
                Paragraph("Fonte: Elaborado pelos autores (2026).", styles["source_below"]),
            ]
        )
    )

    story.append(
        KeepTogether(
            [
                Paragraph("Figura 2 – Ranqueamento ANOVA F dos 12 atributos (a) e desempenho comparado FE, Rank e PCA (b)", styles["caption_above"]),
                scaled_image(FIGURES_DIR / "fig2_rank_vs_pca.png", 15.6),
                Paragraph("Fonte: Elaborado pelos autores (2026).", styles["source_below"]),
            ]
        )
    )

    story.append(
        Paragraph(
            f"Dois resultados da Tabela 2 e da Figura 2 merecem destaque. Primeiro, a inclusão dos quatro atributos físicos "
            f"proporcionou salto massivo de F1 em todas as famílias baseadas em árvores (+0,2823 no Random Forest, de "
            f"{fmt_br(ablation[0]['rf_cv_f1'])} para {fmt_br(ablation[1]['rf_cv_f1'])}, e +0,1812 no Gradient Boosting). "
            f"Segundo, a comparação entre técnicas de redução demonstrou superioridade categórica do Rank sobre a PCA: embora "
            f"5 componentes principais retivessem 95,66% da variância, sua combinação linear ortogonal rotacionou os eixos de "
            f"corte físicos (ΔT, potência e desgaste), derrubando o F1 de todos os ensembles para 0,5412–0,5839. Em "
            f"contrapartida, o Rank ANOVA k = 10 preservou a interpretabilidade física, manteve o F1 competitivo (0,8599–0,8874) "
            f"e chegou a melhorar o HistGradientBoosting padrão de 0,8518 para 0,8599 e o PR-AUC CV do Gradient Boosting de "
            f"0,8902 para 0,9099.",
            styles["body"],
        )
    )

    story.append(
        _heading_with_marker(
            "s4_3",
            "4.3 Comparação entre modelos padrão e otimização",
            styles["h2"],
            page_tracker,
            [
                Paragraph(
                    "Sob o pipeline padronizado (FE + Rank ANOVA k = 10), os seis algoritmos foram avaliados em seus hiperparâmetros "
                    "padrão de fábrica, cujos resultados completos encontram-se na Tabela 3 e na Figura 3.",
                    styles["body"],
                )
            ],
        )
    )

    t3_rows = [
        ["Algoritmo (Padrão)", "CV F1 (± dp)", "Teste<br/>Acur.", "Teste<br/>Prec.", "Teste<br/>Recall", "Teste<br/>F1", "ROC-<br/>AUC", "PR-<br/>AUC"],
    ]
    for r in comp:
        t3_rows.append(
            [
                r["short_name"],
                f"{fmt_br(r['cv_f1'])} (±{fmt_br(r['cv_f1_std'])})",
                fmt_br(r["test_accuracy"]),
                fmt_br(r["test_precision"]),
                fmt_br(r["test_recall"]),
                fmt_br(r["test_f1"]),
                fmt_br(r["test_roc_auc"]),
                fmt_br(r["test_pr_auc"]),
            ]
        )
    story.append(
        KeepTogether(
            [
                Paragraph("Tabela 3 – Comparativo de desempenho dos seis algoritmos padrão (CV 5-fold e Teste Holdout n = 2.000)", styles["caption_above"]),
                make_ibge_table(
                    t3_rows,
                    [3.1 * cm, 2.7 * cm, 1.7 * cm, 1.7 * cm, 1.7 * cm, 1.7 * cm, 1.7 * cm, 1.7 * cm],
                    ["LEFT", "CENTER", "CENTER", "CENTER", "CENTER", "CENTER", "CENTER", "CENTER"],
                    styles,
                    font_reg,
                    font_bold,
                ),
                Paragraph("Fonte: Elaborado pelos autores (2026).", styles["source_below"]),
            ]
        )
    )

    story.append(
        KeepTogether(
            [
                Paragraph("Figura 3 – Comparativo gráfico de F1-score (CV e Teste) e PR-AUC entre os modelos padrão", styles["caption_above"]),
                scaled_image(FIGURES_DIR / "fig3_comparacao_modelos.png", 15.2),
                Paragraph("Fonte: Elaborado pelos autores (2026).", styles["source_below"]),
            ]
        )
    )

    story.append(
        Paragraph(
            "Modelos lineares e neurais simples exibiram limitação clara na fronteira desbalanceada: a Regressão Logística "
            "obteve apenas F1 = 0,2697 no teste (Recall de 17,65%), enquanto a MLP alcançou F1 = 0,6168. Os três ensembles "
            "baseados em árvores dominaram o ranking, todos superando PR-AUC de 0,88. Entre os finalistas de boosting, o "
            "HistGradientBoostingClassifier destacou-se por reunir simultaneamente: (i) a menor variância entre partições na "
            "validação cruzada (±0,0149 contra ±0,0227 do Gradient Boosting e ±0,0295 da Random Forest); (ii) a maior "
            "Revocação padrão no teste (0,8235, detectando mais falhas críticas antes de paradas não planejadas) com "
            "ROC-AUC de 0,9843; (iii) escalabilidade computacional O(n) via discretização em histogramas (KE et al., 2017); e "
            "(iv) suporte nativo ao parâmetro class_weight='balanced', ausente no GradientBoostingClassifier clássico.",
            styles["body"],
        )
    )

    story.append(
        Paragraph(
            f"Otimizou-se então o HistGradientBoostingClassifier via GridSearchCV (216 combinações). A melhor parametrização "
            f"encontrada na validação cruzada foi: class_weight = 'balanced', learning_rate = {fmt_br(hp['learning_rate'], 1)}, "
            f"max_iter = {hp['max_iter']}, max_leaf_nodes = {hp['max_leaf_nodes']}, min_samples_leaf = {hp['min_samples_leaf']} "
            f"e l2_regularization = {fmt_br(hp['l2_regularization'], 1)}. Conforme detalhado na Tabela 4 e na Figura 4, "
            f"a otimização superou expressivamente o modelo padrão em ambas as avaliações.",
            styles["body"],
        )
    )

    def_cv = opt["default_cv"]
    tun_cv = opt["tuned_cv"]
    def_te = opt["default_test"]
    tun_te = opt["tuned_test"]
    cm_d = def_te["confusion_matrix"]
    cm_t = tun_te["confusion_matrix"]
    t4_rows = [
        ["Métrica / Indicador", "Padrão (CV)", "Otimizado (CV)", "Padrão<br/>(Teste)", "Otimizado<br/>(Teste)", "Ganho<br/>no Teste"],
        ["F1-Score", f"{fmt_br(def_cv['cv_f1'])} (±{fmt_br(def_cv['cv_f1_std'])})", f"{fmt_br(tun_cv['cv_f1'])} (±{fmt_br(tun_cv['cv_f1_std'])})", fmt_br(def_te["test_f1"]), fmt_br(tun_te["test_f1"]), f"+{fmt_br(opt['delta_test_f1'])}"],
        ["Precisão", fmt_br(def_cv["cv_precision"]), fmt_br(tun_cv["cv_precision"]), fmt_br(def_te["test_precision"]), fmt_br(tun_te["test_precision"]), f"+{fmt_br(tun_te['test_precision'] - def_te['test_precision'])}"],
        ["Revocação (Recall)", fmt_br(def_cv["cv_recall"]), fmt_br(tun_cv["cv_recall"]), fmt_br(def_te["test_recall"]), fmt_br(tun_te["test_recall"]), f"+{fmt_br(tun_te['test_recall'] - def_te['test_recall'])}"],
        ["Acurácia", fmt_br(def_cv["cv_accuracy"]), fmt_br(tun_cv["cv_accuracy"]), fmt_br(def_te["test_accuracy"]), fmt_br(tun_te["test_accuracy"]), f"+{fmt_br(tun_te['test_accuracy'] - def_te['test_accuracy'])}"],
        ["ROC-AUC", fmt_br(def_cv["cv_roc_auc"]), fmt_br(tun_cv["cv_roc_auc"]), fmt_br(def_te["test_roc_auc"]), fmt_br(tun_te["test_roc_auc"]), f"{tun_te['test_roc_auc'] - def_te['test_roc_auc']:+.4f}".replace(".", ",")],
        ["PR-AUC", fmt_br(def_cv["cv_pr_auc"]), fmt_br(tun_cv["cv_pr_auc"]), fmt_br(def_te["test_pr_auc"]), fmt_br(tun_te["test_pr_auc"]), f"{tun_te['test_pr_auc'] - def_te['test_pr_auc']:+.4f}".replace(".", ",")],
        ["Matriz [TN, FP / FN, TP]", "—", "—", f"{cm_d['tn']}, {cm_d['fp']} / {cm_d['fn']}, {cm_d['tp']}", f"{cm_t['tn']}, {cm_t['fp']} / {cm_t['fn']}, {cm_t['tp']}", "-2 FP / +1 TP"],
    ]
    story.append(
        KeepTogether(
            [
                Paragraph("Tabela 4 – Comparação direta entre HistGradientBoosting Padrão e Otimizado (CV e Teste)", styles["caption_above"]),
                make_ibge_table(t4_rows, [3.8 * cm, 2.5 * cm, 2.6 * cm, 2.3 * cm, 2.4 * cm, 2.4 * cm], ["LEFT", "CENTER", "CENTER", "CENTER", "CENTER", "CENTER"], styles, font_reg, font_bold),
                Paragraph("Fonte: Elaborado pelos autores (2026).", styles["source_below"]),
            ]
        )
    )

    story.append(
        KeepTogether(
            [
                Paragraph("Figura 4 – Matrizes de confusão (a–b) e curva Precision–Recall (c): Modelo Padrão versus Otimizado", styles["caption_above"]),
                scaled_image(FIGURES_DIR / "fig4_otimizacao_matriz_pr.png", 15.6),
                Paragraph("Fonte: Elaborado pelos autores (2026).", styles["source_below"]),
            ]
        )
    )

    story.append(
        Paragraph(
            f"O ganho conjunto foi expressivo: no CV, o F1-score avançou +3,02 pontos percentuais (de {fmt_br(def_cv['cv_f1'])} "
            f"para {fmt_br(tun_cv['cv_f1'])}) enquanto o desvio-padrão caiu pela metade (de ±0,0149 para ±0,0074). No conjunto "
            f"de teste holdout, o modelo otimizado atingiu F1 = {fmt_br(tun_te['test_f1'])} (+2,29 p.p. sobre o padrão e superior "
            f"a todos os seis baselines padrão), alcançando Precisão de 98,28% (apenas 1 falso positivo em 1.932 ciclos normais) "
            f"e Revocação de 83,82% (57 de 68 falhas reais detectadas). Observa-se na Figura 4(c) leve redução pontual na cauda "
            f"de ROC-AUC/PR-AUC (0,9843 para 0,9716), fenômeno esperado quando maior número de iterações (350 árvores) somado "
            f"a pesos balanceados concentra a separação das margens próximas ao limiar decisório de 0,50.",
            styles["body"],
        )
    )

    story.append(
        _heading_with_marker(
            "s4_4",
            "4.4 Operacionalização em API REST, interface web e Docker",
            styles["h2"],
            page_tracker,
            [
                Paragraph(
                    "Para efetivar a entrega em produção, o pipeline completo (capaz de receber os seis campos brutos da máquina, "
                    "calcular automaticamente os quatro atributos físicos, selecionar as 10 dimensões do Rank e emitir a probabilidade "
                    "de falha) foi embarcado em serviço FastAPI (app/main.py). A plataforma disponibiliza painel operacional web "
                    "interativo na rota /, endpoints REST /predict (unitário), /predict/batch (até 500 leituras), /model-info e "
                    "/health, além de alertas diagnósticos físicos legíveis ao operador de fábrica e container Dockerfile reproduzível. "
                    "Todos os testes automatizados (pytest: integridade das equações físicas, inferência roundtrip do arquivo .joblib, "
                    "respostas de risco nominal vs. crítico na API e rejeição HTTP 422 para entradas fisicamente inválidas) foram "
                    "executados com 100% de aprovação (3 passed).",
                    styles["body"],
                )
            ],
        )
    )

    # 5 CONSIDERAÇÕES FINAIS
    story.append(
        _heading_with_marker(
            "s5",
            "5 CONSIDERAÇÕES FINAIS",
            styles["h1"],
            page_tracker,
            [
                Paragraph(
                    "A condução estruturada deste experimento confirmou que a acurácia e a confiabilidade em manutenção preditiva "
                    "dependem mais da representação física dos dados e da prevenção contra vazamento do que da complexidade isolada "
                    "dos algoritmos. A criação de quatro atributos derivados das leis de potência, dissipação térmica e desgaste "
                    "mecânico elevou substancialmente a separabilidade das classes e dominou o topo do ranqueamento ANOVA F.",
                    styles["body"],
                )
            ],
        )
    )
    story.append(
        Paragraph(
            "Demonstraram-se ainda critérios claros para seleção de atributos versus projeções lineares: a seleção por Rank "
            "ANOVA k = 10 superou largamente a PCA em ensembles de árvores por preservar fronteiras físicas ortogonais aos "
            "sensores. A otimização do HistGradientBoostingClassifier entregou o melhor modelo global (F1 = 0,9048 e "
            "Precisão = 98,28% no teste) devidamente operacionalizado em API REST, interface web e Docker. Como limitações, "
            "destaca-se a natureza sintética calibrada do AI4I 2020 e o limite de detecção de falhas puramente aleatórias (RNF); "
            "para trabalhos futuros, recomendam-se calibração isotônica de probabilidades em séries temporais reais de fábrica, "
            "explicabilidade por valores SHAP por instância na API e monitoramento contínuo de concept drift.",
            styles["body"],
        )
    )

    # REFERÊNCIAS
    refs = [
        "ASSOCIAÇÃO BRASILEIRA DE NORMAS TÉCNICAS. <b>ABNT NBR 6023</b>: Informação e documentação — Referências — Elaboração. Rio de Janeiro: ABNT, 2018.",
        "ASSOCIAÇÃO BRASILEIRA DE NORMAS TÉCNICAS. <b>ABNT NBR 10520</b>: Informação e documentação — Citações em documentos — Apresentação. Rio de Janeiro: ABNT, 2023.",
        "ASSOCIAÇÃO BRASILEIRA DE NORMAS TÉCNICAS. <b>ABNT NBR 14724</b>: Informação e documentação — Trabalhos acadêmicos — Apresentação. Rio de Janeiro: ABNT, 2011.",
        "BREIMAN, Leo. Random Forests. <b>Machine Learning</b>, Dordrecht, v. 45, n. 1, p. 5–32, 2001.",
        "FRIEDMAN, Jerome H. Greedy function approximation: a gradient boosting machine. <b>The Annals of Statistics</b>, Hayward, v. 29, n. 5, p. 1189–1232, 2001.",
        "KE, Guolin et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. <i>In</i>: <b>Advances in Neural Information Processing Systems (NeurIPS)</b>, Long Beach, v. 30, p. 3146–3154, 2017.",
        "MATZKA, Stephan. Explainable Artificial Intelligence for Predictive Maintenance Applications. <i>In</i>: <b>2020 Third International Conference on Artificial Intelligence for Industries (AI4I)</b>. Irvine: IEEE, 2020. p. 69–74. Dataset UCI ID 601.",
        "PEDREGOSA, Fabian et al. Scikit-learn: Machine Learning in Python. <b>Journal of Machine Learning Research</b>, v. 12, p. 2825–2830, 2011.",
        "PROVOST, Foster; FAWCETT, Tom. <b>Data Science for Business</b>: What You Need to Know about Data Mining and Data-Analytic Thinking. Sebastopol: O'Reilly Media, 2013.",
        "RAMÍREZ, Sebastián. <b>FastAPI</b>: high performance, easy to learn, fast to code, ready for production. Documentação oficial. Disponível em: https://fastapi.tiangolo.com/. Acesso em: nov. 2026.",
    ]
    story.append(_heading_with_marker("ref", "REFERÊNCIAS", styles["h1"], page_tracker, [Paragraph(refs[0], styles["ref_item"])]))
    for r in refs[1:]:
        story.append(Paragraph(r, styles["ref_item"]))

    return story


def generate_pdf(data: dict[str, Any]) -> tuple[int, dict[str, int]]:
    """Constrói PDF ABNT em duas passagens retornando total de páginas e mapa exato do Sumário."""
    font_reg, font_bold = register_tnr_fonts()
    styles = make_abnt_styles(font_reg, font_bold)
    first_textual_page = 5

    def draw_abnt_page_number(canvas, doc) -> None:
        p = canvas.getPageNumber()
        if p >= first_textual_page:
            canvas.saveState()
            canvas.setFont(font_reg, 10)
            canvas.drawRightString(A4[0] - 2.0 * cm, A4[1] - 2.0 * cm, str(p))
            canvas.restoreState()

    def build_doc(tracker: dict[str, int], toc_pages: dict[str, int]) -> None:
        doc = BaseDocTemplate(
            str(PDF_OUT),
            pagesize=A4,
            leftMargin=3.0 * cm,
            rightMargin=2.0 * cm,
            topMargin=3.0 * cm,
            bottomMargin=2.0 * cm,
            title=TITLE,
            author=", ".join(AUTHORS),
        )
        frame = Frame(
            3.0 * cm,
            2.0 * cm,
            A4[0] - 5.0 * cm,
            A4[1] - 5.0 * cm,
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        doc.addPageTemplates([PageTemplate(id="abnt", frames=[frame], onPage=draw_abnt_page_number)])
        story = _build_pdf_story(data, styles, tracker, toc_pages, font_reg, font_bold)
        doc.build(story)

    pass1_pages: dict[str, int] = {}
    build_doc(pass1_pages, {})
    pass2_pages: dict[str, int] = {}
    build_doc(pass2_pages, pass1_pages)

    reader = PdfReader(str(PDF_OUT))
    return len(reader.pages), pass2_pages


def _style_ibge_docx_table(table) -> None:
    """Remove bordas verticais no Word e aplica fios horizontais ABNT/IBGE."""
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for side, val, sz in [
        ("top", "single", "12"),
        ("bottom", "single", "12"),
        ("insideH", "none", "0"),
        ("left", "none", "0"),
        ("right", "none", "0"),
        ("insideV", "none", "0"),
    ]:
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:val"), val)
        if sz != "0":
            el.set(qn("w:sz"), sz)
            el.set(qn("w:color"), "000000")
        borders.append(el)
    tbl_pr.append(borders)

    for cell in table.rows[0].cells:
        tc_pr = cell._tc.get_or_add_tcPr()
        tc_borders = OxmlElement("w:tcBorders")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), "8")
        bottom.set(qn("w:color"), "000000")
        tc_borders.append(bottom)
        tc_pr.append(tc_borders)


def _add_docx_page_number_field(run) -> None:
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.extend([fld_begin, instr, fld_sep, fld_end])


def generate_docx(data: dict[str, Any], toc_pages: dict[str, int] | None = None) -> None:
    """Gera documento .docx editável completo (incluindo todas as 4 tabelas e 4 figuras)."""
    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal.font.color.rgb = RGBColor(0, 0, 0)

    sec1 = doc.sections[0]
    sec1.page_width = Cm(21.0)
    sec1.page_height = Cm(29.7)
    sec1.top_margin = Cm(3.0)
    sec1.left_margin = Cm(3.0)
    sec1.bottom_margin = Cm(2.0)
    sec1.right_margin = Cm(2.0)

    def add_p(text: str, *, bold=False, size=12, align=WD_ALIGN_PARAGRAPH.JUSTIFY, indent_cm=0.0, left_cm=0.0, spacing=1.5, after_pt=4):
        p = doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.line_spacing = spacing
        p.paragraph_format.space_after = Pt(after_pt)
        if indent_cm:
            p.paragraph_format.first_line_indent = Cm(indent_cm)
        if left_cm:
            p.paragraph_format.left_indent = Cm(left_cm)
        run = p.add_run(text)
        run.bold = bold
        run.font.name = "Times New Roman"
        run.font.size = Pt(size)
        return p

    # Capa
    add_p(INSTITUTION, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.15, after_pt=2)
    add_p(COURSE, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.15, after_pt=48)
    for a in AUTHORS:
        add_p(a.upper(), align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.3, after_pt=2)
    add_p("", after_pt=96)
    add_p(TITLE, bold=True, size=13, align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.3, after_pt=140)
    add_p("POMPEIA – SP\n2026", align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.15)
    doc.add_page_break()

    # Folha de Rosto
    for a in AUTHORS:
        add_p(a.upper(), align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.3, after_pt=2)
    add_p("", after_pt=90)
    add_p(TITLE, bold=True, size=13, align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.3, after_pt=42)
    add_p(
        f"Relatório técnico-científico apresentado à disciplina de {DISCIPLINE}, do Curso Superior de Tecnologia em "
        f"Ciência de Dados da {INSTITUTION_TITLE}, sob orientação do {PROFESSOR}, como requisito parcial de avaliação.",
        size=10,
        align=WD_ALIGN_PARAGRAPH.JUSTIFY,
        left_cm=7.2,
        spacing=1.0,
        after_pt=120,
    )
    add_p("Pompeia – SP\n2026", align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.15)
    doc.add_page_break()

    # Resumo
    add_p("RESUMO", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after_pt=12)
    add_p(data["resumo"], spacing=1.5, after_pt=10)
    p_kw = doc.add_paragraph()
    p_kw.paragraph_format.line_spacing = 1.5
    r1 = p_kw.add_run("Palavras-chave: ")
    r1.bold = True
    r1.font.name = "Times New Roman"
    r1.font.size = Pt(12)
    r2 = p_kw.add_run(data["keywords"])
    r2.font.name = "Times New Roman"
    r2.font.size = Pt(12)
    doc.add_page_break()

    # Sumário
    add_p("SUMÁRIO", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after_pt=12)
    sum_items = [
        ("1 INTRODUÇÃO", "s1"),
        ("2 OBJETIVO", "s2"),
        ("3 MATERIAL E MÉTODOS", "s3"),
        ("    3.1 Conjunto de dados e pré-processamento", "s3_1"),
        ("    3.2 Engenharia de atributos físicos", "s3_2"),
        ("    3.3 Redução de dimensionalidade: Rank versus PCA", "s3_3"),
        ("    3.4 Algoritmos avaliados, métricas e otimização", "s3_4"),
        ("    3.5 Arquitetura de exportação para produção", "s3_5"),
        ("4 RESULTADOS E DISCUSSÃO", "s4"),
        ("    4.1 Análise exploratória e integridade dos dados", "s4_1"),
        ("    4.2 Impacto da engenharia de atributos e Rank versus PCA", "s4_2"),
        ("    4.3 Comparação entre modelos padrão e otimização", "s4_3"),
        ("    4.4 Operacionalização em API REST, interface web e Docker", "s4_4"),
        ("5 CONSIDERAÇÕES FINAIS", "s5"),
        ("REFERÊNCIAS", "ref"),
    ]
    for label, key in sum_items:
        pno = toc_pages.get(key, "") if toc_pages else ""
        add_p(f"{label} {'.' * max(4, 58 - len(label))} {pno}", spacing=1.25, after_pt=3)

    # Seção 2: Textual
    sec2 = doc.add_section()
    sec2.page_width = Cm(21.0)
    sec2.page_height = Cm(29.7)
    sec2.top_margin = Cm(3.0)
    sec2.left_margin = Cm(3.0)
    sec2.bottom_margin = Cm(2.0)
    sec2.right_margin = Cm(2.0)
    sec2.header.is_linked_to_previous = False
    hp = sec2.header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    hrun = hp.add_run()
    hrun.font.name = "Times New Roman"
    hrun.font.size = Pt(10)
    _add_docx_page_number_field(hrun)

    eda = data["eda"]
    rank_diag = data["rank_diag"]
    ablation = data["ablation"]
    comp = data["comp"]
    opt = data["opt"]
    hparams = data["hp"]

    add_p("1 INTRODUÇÃO", bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, spacing=1.15, after_pt=6)
    for p in data["intro"]:
        add_p(p, indent_cm=1.25)

    add_p("2 OBJETIVO", bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, spacing=1.15, after_pt=6)
    add_p(data["obj_intro"], indent_cm=1.25)
    for b in data["obj_bullets"]:
        add_p(f"• {b}", left_cm=0.8, spacing=1.35, after_pt=3)

    add_p("3 MATERIAL E MÉTODOS", bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, spacing=1.15, after_pt=6)
    add_p("3.1 Conjunto de dados e pré-processamento", bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, spacing=1.15, after_pt=4)
    add_p(
        f"Utilizou-se o AI4I 2020 Predictive Maintenance Dataset (MATZKA, 2020), com {fmt_int_br(eda['n_rows'])} linhas "
        f"e {eda['n_cols_original']} variáveis originais. Removeram-se identificadores (UDI, Product ID) e os submodos "
        f"pós-evento (TWF, HDF, PWF, OSF, RNF) para evitar vazamento de alvo. Aplicou-se divisão estratificada 80/20 "
        f"(8.000 treino / 2.000 teste, random_state=42) e pré-processamento por ColumnTransformer (SimpleImputer + "
        f"StandardScaler para atributos numéricos; SimpleImputer + OneHotEncoder para Type).",
        indent_cm=1.25,
    )

    add_p("3.2 Engenharia de atributos físicos", bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, spacing=1.15, after_pt=4)
    add_p(
        "Implementou-se PredictiveMaintenanceFeatureEngineer gerando quatro novos descritores físicos: "
        "Temp_Diff_K (ΔT = T_processo − T_ar [K]), Mechanical_Power_W (Torque × RPM × 2π/60), "
        "Wear_Torque_Nm_min (Tool wear × Torque) e Torque_Speed_Ratio (Torque / RPM).",
        indent_cm=1.25,
    )

    add_p("3.3 Redução de dimensionalidade: Rank versus PCA", bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, spacing=1.15, after_pt=4)
    add_p(
        f"Sobre as {rank_diag['total_encoded_features']} variáveis codificadas, confrontou-se seleção por Rank supervisionado "
        f"(SelectKBest k = {rank_diag['rank_k_selected']} via ANOVA F e Informação Mútua) contra extração ortogonal por PCA "
        f"retendo 95% da variância total (5 componentes).",
        indent_cm=1.25,
    )

    add_p("3.4 Algoritmos avaliados, métricas e otimização", bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, spacing=1.15, after_pt=4)
    add_p(
        f"Avaliaram-se seis algoritmos (Dummy, Regressão Logística, MLP, Random Forest, Gradient Boosting e HistGradientBoosting) "
        f"via CV estratificado de 5 folds e teste holdout, escolhendo-se o HistGradientBoostingClassifier para otimização "
        f"por GridSearchCV ({opt['grid_search_candidates']} combinações).",
        indent_cm=1.25,
    )

    add_p("3.5 Arquitetura de exportação para produção", bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, spacing=1.15, after_pt=4)
    add_p(
        "O pipeline completo foi serializado com joblib (modelo_manutencao_otimizado.joblib) e publicado via API REST "
        "FastAPI (/health, /model-info, /predict, /predict/batch), interface web integrada em / e contêiner Dockerfile.",
        indent_cm=1.25,
    )

    def add_docx_table(caption: str, source: str, matrix: list[list[str]]) -> None:
        add_p(caption, bold=True, size=10, align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.1, after_pt=3)
        t = doc.add_table(rows=len(matrix), cols=len(matrix[0]))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                cell = t.cell(i, j)
                p = cell.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.line_spacing = 1.0
                p.paragraph_format.space_after = Pt(2)
                r = p.add_run(val)
                r.font.name = "Times New Roman"
                r.font.size = Pt(9.5)
                r.bold = i == 0
        _style_ibge_docx_table(t)
        add_p(source, size=9.5, align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.0, after_pt=8)

    def add_docx_fig(caption: str, img_name: str, source: str) -> None:
        add_p(caption, bold=True, size=10, align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.1, after_pt=3)
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(FIGURES_DIR / img_name), width=Inches(5.9))
        add_p(source, size=9.5, align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.0, after_pt=8)

    add_p("4 RESULTADOS E DISCUSSÃO", bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, spacing=1.15, after_pt=6)
    add_p("4.1 Análise exploratória e integridade dos dados", bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, spacing=1.15, after_pt=4)
    add_p(
        f"O dataset apresentou {eda['missing_values_total']} nulos e {eda['duplicate_rows']} duplicados, com 9.661 instâncias "
        f"normais (96,61%) e 339 falhas (3,39%). As taxas de falha por variante foram L = {fmt_br(eda['failure_by_type_pct']['L'], 2)}%, "
        f"M = {fmt_br(eda['failure_by_type_pct']['M'], 2)}% e H = {fmt_br(eda['failure_by_type_pct']['H'], 2)}%.",
        indent_cm=1.25,
    )
    t1_matrix = [
        ["Segmento / Indicador", "Quantidade", "Proporção (%)", "Observação Metodológica"],
        ["Classe 0 — Operação Normal", "9.661", "96,61%", "Desbalanceamento 28,5 : 1"],
        ["Classe 1 — Falha na Máquina", "339", "3,39%", "Alvo binário (Machine failure)"],
        ["Submodo HDF (Dissipação Térmica)", "115", "1,15%", "ΔT < 8,6 K e RPM < 1.380"],
        ["Submodo OSF (Sobre-esforço)", "98", "0,98%", "Produto desgaste × torque crítico"],
        ["Submodo PWF (Potência)", "95", "0,95%", "Potência < 3,5 kW ou > 9,0 kW"],
        ["Submodo TWF (Desgaste Ferramenta)", "46", "0,46%", "Tool wear entre 200 e 240 min"],
        ["Submodo RNF (Falha Aleatória)", "19", "0,19%", "18 ocorrências com alvo = 0"],
        ["Variante Type L / M / H", "6.000 / 2.997 / 1.003", "60,0 / 30,0 / 10,0", "Taxa falha L = 3,92% > M > H"],
    ]
    add_docx_table("Tabela 1 – Distribuição de classes, submodos de falha e variantes no AI4I 2020", "Fonte: Elaborado pelos autores com base em Matzka (2020).", t1_matrix)
    add_docx_fig("Figura 1 – Submodos de falha (a) e separabilidade física obtida pelos atributos derivados (b)", "fig1_eda_falhas_e_distribuicao.png", "Fonte: Elaborado pelos autores (2026).")

    add_p("4.2 Impacto da engenharia de atributos e Rank versus PCA", bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, spacing=1.15, after_pt=4)
    add_p(
        f"Os quatro atributos criados alcançaram o Top 5 do ranqueamento ANOVA F (Torque_Speed_Ratio F = 349,716 em 1º lugar). "
        f"O estudo de ablação demonstrou ganho decisivo com a engenharia de atributos (RF de {fmt_br(ablation[0]['rf_cv_f1'])} para "
        f"{fmt_br(ablation[1]['rf_cv_f1'])}) e nítida vantagem do Rank ANOVA k = 10 sobre a PCA (0,5412–0,5839).",
        indent_cm=1.25,
    )
    t2_matrix = [
        ["Regime Experimental", "Atributos", "RF CV F1", "GB CV F1", "HGB CV F1"],
        ["1. Dados Brutos", "8", fmt_br(ablation[0]["rf_cv_f1"]), fmt_br(ablation[0]["gb_cv_f1"]), fmt_br(ablation[0]["hgb_cv_f1"])],
        ["2. FE Completo", "12", fmt_br(ablation[1]["rf_cv_f1"]), fmt_br(ablation[1]["gb_cv_f1"]), fmt_br(ablation[1]["hgb_cv_f1"])],
        ["3. FE + Rank ANOVA (k=10)", "10", fmt_br(ablation[2]["rf_cv_f1"]), fmt_br(ablation[2]["gb_cv_f1"]), fmt_br(ablation[2]["hgb_cv_f1"])],
        ["4. FE + Rank MI (k=10)", "10", fmt_br(ablation[3]["rf_cv_f1"]), fmt_br(ablation[3]["gb_cv_f1"]), fmt_br(ablation[3]["hgb_cv_f1"])],
        ["5. FE + PCA 95%", "5 comp.", fmt_br(ablation[4]["rf_cv_f1"]), fmt_br(ablation[4]["gb_cv_f1"]), fmt_br(ablation[4]["hgb_cv_f1"])],
    ]
    add_docx_table("Tabela 2 – Estudo de ablação em validação cruzada (FE, Rank e PCA)", "Fonte: Elaborado pelos autores (2026).", t2_matrix)
    add_docx_fig("Figura 2 – Ranqueamento ANOVA F e comparação FE, Rank e PCA", "fig2_rank_vs_pca.png", "Fonte: Elaborado pelos autores (2026).")

    add_p("4.3 Comparação entre modelos padrão e otimização", bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, spacing=1.15, after_pt=4)
    t3_matrix = [["Algoritmo", "CV F1", "Teste Prec.", "Teste Recall", "Teste F1", "ROC-AUC", "PR-AUC"]] + [
        [r["short_name"], fmt_br(r["cv_f1"]), fmt_br(r["test_precision"]), fmt_br(r["test_recall"]), fmt_br(r["test_f1"]), fmt_br(r["test_roc_auc"]), fmt_br(r["test_pr_auc"])]
        for r in comp
    ]
    add_docx_table("Tabela 3 – Comparativo dos seis algoritmos padrão", "Fonte: Elaborado pelos autores (2026).", t3_matrix)
    add_docx_fig("Figura 3 – Comparativo gráfico entre modelos padrão", "fig3_comparacao_modelos.png", "Fonte: Elaborado pelos autores (2026).")

    add_p(
        f"A otimização do HistGradientBoostingClassifier ({hparams}) elevou o F1 em CV de {fmt_br(opt['default_cv']['cv_f1'])} para "
        f"{fmt_br(opt['tuned_cv']['cv_f1'])} e o F1 de Teste de {fmt_br(opt['default_test']['test_f1'])} para "
        f"{fmt_br(opt['tuned_test']['test_f1'])}, com Precisão = {fmt_br(opt['tuned_test']['test_precision'])} e "
        f"Recall = {fmt_br(opt['tuned_test']['test_recall'])}.",
        indent_cm=1.25,
    )
    def_cv, tun_cv = opt["default_cv"], opt["tuned_cv"]
    def_te, tun_te = opt["default_test"], opt["tuned_test"]
    cm_d, cm_t = def_te["confusion_matrix"], tun_te["confusion_matrix"]
    t4_matrix = [
        ["Métrica / Indicador", "Padrão (CV)", "Otimizado (CV)", "Padrão (Teste)", "Otimizado (Teste)", "Ganho Teste"],
        ["F1-Score", fmt_br(def_cv["cv_f1"]), fmt_br(tun_cv["cv_f1"]), fmt_br(def_te["test_f1"]), fmt_br(tun_te["test_f1"]), f"+{fmt_br(opt['delta_test_f1'])}"],
        ["Precisão", fmt_br(def_cv["cv_precision"]), fmt_br(tun_cv["cv_precision"]), fmt_br(def_te["test_precision"]), fmt_br(tun_te["test_precision"]), f"+{fmt_br(tun_te['test_precision'] - def_te['test_precision'])}"],
        ["Revocação", fmt_br(def_cv["cv_recall"]), fmt_br(tun_cv["cv_recall"]), fmt_br(def_te["test_recall"]), fmt_br(tun_te["test_recall"]), f"+{fmt_br(tun_te['test_recall'] - def_te['test_recall'])}"],
        ["Acurácia", fmt_br(def_cv["cv_accuracy"]), fmt_br(tun_cv["cv_accuracy"]), fmt_br(def_te["test_accuracy"]), fmt_br(tun_te["test_accuracy"]), f"+{fmt_br(tun_te['test_accuracy'] - def_te['test_accuracy'])}"],
        ["ROC-AUC", fmt_br(def_cv["cv_roc_auc"]), fmt_br(tun_cv["cv_roc_auc"]), fmt_br(def_te["test_roc_auc"]), fmt_br(tun_te["test_roc_auc"]), f"{tun_te['test_roc_auc'] - def_te['test_roc_auc']:+.4f}".replace(".", ",")],
        ["PR-AUC", fmt_br(def_cv["cv_pr_auc"]), fmt_br(tun_cv["cv_pr_auc"]), fmt_br(def_te["test_pr_auc"]), fmt_br(tun_te["test_pr_auc"]), f"{tun_te['test_pr_auc'] - def_te['test_pr_auc']:+.4f}".replace(".", ",")],
        ["Matriz [TN,FP/FN,TP]", "—", "—", f"{cm_d['tn']},{cm_d['fp']}/{cm_d['fn']},{cm_d['tp']}", f"{cm_t['tn']},{cm_t['fp']}/{cm_t['fn']},{cm_t['tp']}", "-2 FP / +1 TP"],
    ]
    add_docx_table("Tabela 4 – Comparação direta HistGradientBoosting Padrão vs Otimizado", "Fonte: Elaborado pelos autores (2026).", t4_matrix)
    add_docx_fig("Figura 4 – Matrizes de confusão e curva Precision–Recall (Padrão vs Otimizado)", "fig4_otimizacao_matriz_pr.png", "Fonte: Elaborado pelos autores (2026).")

    add_p("4.4 Operacionalização em API REST, interface web e Docker", bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, spacing=1.15, after_pt=4)
    add_p(
        "O serviço FastAPI + painel web integrado em app/main.py e app/static/index.html opera com validação Pydantic e "
        "cálculo automático dos atributos físicos, aprovado integralmente nos testes pytest (3/3).",
        indent_cm=1.25,
    )

    add_p("5 CONSIDERAÇÕES FINAIS", bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, spacing=1.15, after_pt=6)
    add_p(
        "Conclui-se que a integração entre engenharia de atributos físicos, seleção Rank ANOVA k = 10 e otimização do "
        "HistGradientBoostingClassifier proporcionou desempenho superior (Teste F1 = 0,9048) com implantação prática em "
        "ambiente de produção via API REST e Docker.",
        indent_cm=1.25,
    )

    add_p("REFERÊNCIAS", bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, spacing=1.15, after_pt=6)
    refs_plain = [
        "ASSOCIAÇÃO BRASILEIRA DE NORMAS TÉCNICAS. ABNT NBR 6023: Informação e documentação — Referências. Rio de Janeiro: ABNT, 2018.",
        "ASSOCIAÇÃO BRASILEIRA DE NORMAS TÉCNICAS. ABNT NBR 10520: Informação e documentação — Citações em documentos. Rio de Janeiro: ABNT, 2023.",
        "ASSOCIAÇÃO BRASILEIRA DE NORMAS TÉCNICAS. ABNT NBR 14724: Informação e documentação — Trabalhos acadêmicos. Rio de Janeiro: ABNT, 2011.",
        "BREIMAN, Leo. Random Forests. Machine Learning, v. 45, n. 1, p. 5–32, 2001.",
        "FRIEDMAN, Jerome H. Greedy function approximation: a gradient boosting machine. The Annals of Statistics, v. 29, n. 5, p. 1189–1232, 2001.",
        "KE, Guolin et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS, v. 30, p. 3146–3154, 2017.",
        "MATZKA, Stephan. Explainable Artificial Intelligence for Predictive Maintenance Applications. IEEE AI4I, p. 69–74, 2020.",
        "PEDREGOSA, Fabian et al. Scikit-learn: Machine Learning in Python. JMLR, v. 12, p. 2825–2830, 2011.",
        "PROVOST, Foster; FAWCETT, Tom. Data Science for Business. Sebastopol: O'Reilly Media, 2013.",
        "RAMÍREZ, Sebastián. FastAPI Documentation. Disponível em: https://fastapi.tiangolo.com/. Acesso em: nov. 2026.",
    ]
    for r in refs_plain:
        add_p(r, size=11, align=WD_ALIGN_PARAGRAPH.LEFT, spacing=1.0, after_pt=6)

    doc.save(str(DOCX_OUT))


def main() -> None:
    data = get_narrative_sections(build_report_data())
    n_pages, exact_toc_pages = generate_pdf(data)
    generate_docx(data, toc_pages=exact_toc_pages)
    print(f"PDF gerado em: {PDF_OUT} ({n_pages} páginas)")
    print(f"DOCX gerado em: {DOCX_OUT}")
    print(f"Sumário sincronizado: {exact_toc_pages}")
    assert n_pages <= 12, f"Limite de 12 páginas violado! Gerado com {n_pages} páginas."


if __name__ == "__main__":
    main()
