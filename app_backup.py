"""
MIZU: Intelligent Water Systems Monitoring Dashboard
===================================================
A decision-support application for unsupervised anomaly detection,
operational regime clustering, and explainability in wastewater treatment plants.

Dataset: UCI Water Treatment Plant (ID: 106)
Frameworks: Streamlit, Pandas, Plotly, Scikit-Learn

NOTE: This version only changes the presentation layer (layout, styling,
navigation, charts, interactivity). The model loading, ingestion calls and
inference pipeline (`load_all_models`, `load_baseline_raw_data`,
`run_mizu_inference`) are unchanged.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# MIZU core modules
from src.data_loader import (
    COLUMN_NAMES,
    get_column_names,
    get_feature_groups,
    get_missing_summary,
    load_water_treatment_data,
)
from src.preprocessing import MizuPreprocessor, get_feature_columns
from src.anomaly_detection import IsolationForestDetector
from src.lof_detection import LOFDetector
from src.clustering import MizuKMeansClusterer
from src.dimensionality_reduction import MizuPCA
from src.explainability import AnomalyExplainer, SENSOR_METADATA
from src.ingestion import (
    MultiFormatIngestionEngine,
    IngestionReport,
    ExtractedData,
    CANONICAL_FEATURES,
    CANONICAL_UNITS,
    FEATURE_ALIASES
)


# =============================================================================
# 1. CONSTANTS & METADATA
# =============================================================================

# Cluster operational regime taxonomy (names/descriptions unchanged; colours tuned for dark UI)
REGIME_METADATA = {
    0: {
        "name": "Regime 0: Baseline Normal Operations",
        "short": "Baseline normal operations",
        "description": "Stable biological and physical clarification. Standard influent flow and organic loading with high removal efficiencies across all treatment stages.",
        "color": "#38BDF8"
    },
    1: {
        "name": "Regime 1: Secondary Settler Solids Washout",
        "short": "Solids washout",
        "description": "Elevated effluent settleable solids and suspended solids. Biological clarification impaired, causing sub-optimal final sedimentation.",
        "color": "#F5B942"
    },
    2: {
        "name": "Regime 2: Hydraulic & Organic Surge",
        "short": "Hydraulic / organic surge",
        "description": "Severe volumetric inflow spike or industrial organic slug. Elevated influent COD and primary solids with stressed downstream stages.",
        "color": "#A78BFA"
    },
    3: {
        "name": "Regime 3: Critical Solids Separation Breakdown",
        "short": "Separation breakdown",
        "description": "Extreme physical separation failure in both primary and secondary sedimentation stages. Severe plant performance collapse.",
        "color": "#F0525B"
    }
}

# Palette (single source of truth for charts and CSS)
C_BG = "#0B1220"
C_PANEL = "#131C31"
C_LINE = "#22304D"
C_TEXT = "#E6EDF7"
C_MUTED = "#8FA0BC"
C_TEAL = "#2DD4BF"
C_AMBER = "#F5B942"
C_RED = "#F0525B"
C_BLUE = "#4C8DFF"
C_GRID = "#1B2742"

PAGES = [
    "Overview",
    "Upload Data",
    "Flagged Readings",
    "Patterns",
    "Why Flagged",
    "About",
]

# Plain-language labels for the model concordance categories
STATUS_LABELS = {
    "Consensus Anomaly": "Flagged by both models",
    "iForest Only": "Isolation Forest only",
    "LOF Only": "LOF only",
    "Normal Baseline": "Normal",
}
STATUS_COLORS = {
    "Flagged by both models": C_RED,
    "Isolation Forest only": "#FB923C",
    "LOF only": C_AMBER,
    "Normal": "#5B6B88",
}

# Friendly column names for tables (only used for display)
TABLE_LABELS = {
    "Date": "Date",
    "status": "Status",
    "iforest_score": "iForest score",
    "lof_score": "LOF score",
    "regime_short": "Operating regime",
    "Q-E": "Inflow (m3/day)",
    "ZN-E": "Zinc in (mg/L)",
    "SED-S": "Settleable solids out (mL/L)",
    "RD-DBO-G": "BOD removal (%)",
    "RD-SED-G": "Solids removal (%)",
}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
    --bg: #0B1220;
    --panel: #131C31;
    --panel-2: #0F1729;
    --line: #22304D;
    --text: #E6EDF7;
    --muted: #8FA0BC;
    --teal: #2DD4BF;
    --amber: #F5B942;
    --red: #F0525B;
    --blue: #4C8DFF;
}

/* ---------- Base ---------- */
html, body, .stApp, [class*="css"] {
    font-family: 'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, sans-serif;
    font-size: 13px;
}
.stApp { background: var(--bg); color: var(--text); }
#MainMenu, footer, [data-testid="stDeployButton"], .stDeployButton { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }

[data-testid="stMainBlockContainer"], .block-container {
    padding: 1.3rem 2rem 2.5rem 2rem;
    max-width: 1400px;
}

h1 { font-size: 22px !important; font-weight: 600 !important; margin: 0 0 2px 0 !important; padding: 0 !important; letter-spacing: -0.01em; }
h2 { font-size: 17px !important; font-weight: 600 !important; margin: 14px 0 4px 0 !important; padding: 0 !important; }
h3 { font-size: 15px !important; font-weight: 600 !important; margin: 10px 0 4px 0 !important; padding: 0 !important; }
h4, h5 { font-size: 13px !important; font-weight: 600 !important; margin: 8px 0 2px 0 !important; padding: 0 !important; }
p, li, label, .stMarkdown { font-size: 13px; line-height: 1.55; }
[data-testid="stCaptionContainer"], .stCaption, small { font-size: 12px !important; color: var(--muted) !important; }
hr { margin: 14px 0 !important; border-color: var(--line) !important; }

/* Short fade when a page's content appears (answers a navigation click) */
[data-testid="stMainBlockContainer"] .element-container {
    animation: mizuFade .28s ease both;
}
@keyframes mizuFade { from { opacity: 0; } to { opacity: 1; } }

@media (prefers-reduced-motion: reduce) {
    * { animation: none !important; transition: none !important; }
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: var(--panel-2);
    border-right: 1px solid var(--line);
}
@media (min-width: 769px) {
    section[data-testid="stSidebar"] { width: 240px !important; min-width: 240px !important; }
}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding: 1rem 0.9rem; }

.brand { font-size: 20px; font-weight: 600; letter-spacing: 0.02em; color: var(--text); line-height: 1.1; }
.brand-sub { font-size: 12px; color: var(--muted); margin-top: 2px; }

/* Sidebar navigation: turn the radio group into a vertical menu */
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 2px; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    width: 100%;
    min-height: 40px;
    padding: 8px 12px;
    margin: 0;
    border-radius: 8px;
    border-left: 3px solid transparent;
    display: flex;
    align-items: center;
    cursor: pointer;
    transition: background .18s ease, border-color .18s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: #18233D; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display: none; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: #16304A;
    border-left-color: var(--teal);
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label p {
    font-size: 13px; margin: 0; white-space: nowrap; color: var(--text);
}

/* ---------- KPI cards ---------- */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px;
    margin: 8px 0 14px 0;
}
.kpi {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 12px 14px;
    min-height: 94px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: border-color .2s ease, transform .2s ease;
}
.kpi:hover { border-color: #345083; transform: translateY(-1px); }
.kpi-label { font-size: 12px; color: var(--muted); white-space: nowrap; }
.kpi-value { font-size: 28px; font-weight: 600; line-height: 1.1; font-variant-numeric: tabular-nums; }
.kpi-sub { font-size: 12px; color: var(--muted); }
.t-teal .kpi-value { color: var(--teal); }
.t-red .kpi-value { color: var(--red); }
.t-amber .kpi-value { color: var(--amber); }
.t-blue .kpi-value { color: var(--blue); }
.t-plain .kpi-value { color: var(--text); }

/* One staggered reveal for the KPI row */
.kpi-grid .kpi { animation: kpiIn .5s ease both; }
.kpi-grid .kpi:nth-child(2) { animation-delay: .06s; }
.kpi-grid .kpi:nth-child(3) { animation-delay: .12s; }
.kpi-grid .kpi:nth-child(4) { animation-delay: .18s; }
.kpi-grid .kpi:nth-child(5) { animation-delay: .24s; }
@keyframes kpiIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }

/* ---------- Status strip ---------- */
.status {
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
    padding: 10px 14px;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 10px;
    margin: 8px 0 4px 0;
}
.status-text { color: var(--muted); font-size: 12.5px; }
.dot { width: 10px; height: 10px; border-radius: 50%; flex: 0 0 auto; }
.dot.ok { background: var(--teal); animation: pulseOk 2.2s ease-out infinite; }
.dot.warn { background: var(--amber); animation: pulseWarn 1.8s ease-out infinite; }
.dot.bad { background: var(--red); animation: pulseBad 1.2s ease-out infinite; }
@keyframes pulseOk { 0% { box-shadow: 0 0 0 0 rgba(45,212,191,.55); } 100% { box-shadow: 0 0 0 9px rgba(45,212,191,0); } }
@keyframes pulseWarn { 0% { box-shadow: 0 0 0 0 rgba(245,185,66,.55); } 100% { box-shadow: 0 0 0 9px rgba(245,185,66,0); } }
@keyframes pulseBad { 0% { box-shadow: 0 0 0 0 rgba(240,82,91,.65); } 100% { box-shadow: 0 0 0 10px rgba(240,82,91,0); } }
.badge { font-weight: 600; font-size: 13px; padding: 2px 10px; border-radius: 999px; }
.badge.ok { background: rgba(45,212,191,.14); color: var(--teal); }
.badge.warn { background: rgba(245,185,66,.14); color: var(--amber); }
.badge.bad { background: rgba(240,82,91,.14); color: var(--red); }

/* Sidebar chip */
.chip { display: inline-flex; align-items: center; gap: 8px; font-size: 12px; padding: 5px 10px; border-radius: 999px; border: 1px solid var(--line); background: var(--panel); }
.src-box { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; font-size: 12px; color: var(--muted); margin-bottom: 8px; line-height: 1.5; }
.src-box b { color: var(--text); font-weight: 500; }

/* ---------- Notices ---------- */
.notice, .abstain {
    border-radius: 8px; padding: 10px 14px; margin: 8px 0 12px 0;
    font-size: 12.5px; line-height: 1.55;
}
.notice { background: #101B33; border-left: 3px solid var(--blue); color: #B6C4DE; }
.abstain { background: #2A2210; border-left: 3px solid var(--amber); color: #E7CF9A; }
.notice b, .abstain b { color: var(--text); font-weight: 600; }

/* ---------- Panels ---------- */
.panel {
    background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
    padding: 12px 14px; height: 100%;
}
.panel-title { font-weight: 600; font-size: 13px; margin-bottom: 4px; color: var(--text); }
.panel-text { font-size: 12.5px; color: #AEBBD3; line-height: 1.5; }

/* ---------- Native widgets ---------- */
[data-testid="stMetric"] {
    background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
    padding: 10px 14px;
}
[data-testid="stMetricLabel"] p { font-size: 12px !important; color: var(--muted) !important; white-space: nowrap; }
[data-testid="stMetricValue"] { font-size: 24px !important; font-variant-numeric: tabular-nums; }

.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
    border-radius: 8px; border: 1px solid var(--line); background: var(--panel);
    color: var(--text); min-height: 40px; font-size: 13px;
    transition: border-color .18s ease, background .18s ease, transform .12s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--teal); background: #16243F; }
.stButton > button:active, .stDownloadButton > button:active { transform: scale(.98); }

button[data-baseweb="tab"] { font-size: 13px; padding: 8px 14px; transition: color .18s ease; }
[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 10px; background: var(--panel); }
[data-testid="stExpander"] summary { font-size: 13px; }
[data-testid="stFileUploader"] section { border: 1px dashed #345083; border-radius: 10px; background: var(--panel); transition: border-color .2s ease; }
[data-testid="stFileUploader"] section:hover { border-color: var(--teal); }
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }

/* ---------- Mobile ---------- */
@media (max-width: 768px) {
    [data-testid="stMainBlockContainer"], .block-container { padding: 0.8rem 0.8rem 2rem 0.8rem; }
    h1 { font-size: 19px !important; }
    h2 { font-size: 16px !important; }
    .kpi-grid { grid-template-columns: repeat(2, 1fr); gap: 8px; }
    .kpi { min-height: 84px; padding: 10px 12px; }
    .kpi-value { font-size: 23px; }
    .status { padding: 8px 12px; }
    .stButton > button, .stDownloadButton > button { min-height: 44px; }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label { min-height: 46px; }
    [data-testid="stDataFrame"] { overflow-x: auto; }
}
@media (max-width: 420px) {
    .kpi-grid { grid-template-columns: 1fr 1fr; }
    .kpi-label { font-size: 11px; }
}
</style>
"""


# =============================================================================
# 2. CACHED MODEL & DATA LOADING  (UNCHANGED)
# =============================================================================

@st.cache_resource(show_spinner="Loading trained MIZU ML artifacts...")
def load_all_models() -> Tuple[
    MizuPreprocessor,
    IsolationForestDetector,
    LOFDetector,
    MizuKMeansClusterer,
    MizuPCA,
    AnomalyExplainer
]:
    """Load all fitted scikit-learn models and explainer from disk."""
    preprocessor = MizuPreprocessor.load("models/preprocessor.joblib")
    iforest = IsolationForestDetector.load("models/isolation_forest.joblib")
    lof = LOFDetector.load("models/lof_model.joblib")
    kmeans = MizuKMeansClusterer.load("models/kmeans_model.joblib")
    pca = MizuPCA.load("models/pca_model.joblib")
    explainer = AnomalyExplainer(iforest_detector=iforest, lof_detector=lof)
    return preprocessor, iforest, lof, kmeans, pca, explainer


@st.cache_data(show_spinner="Loading historical baseline dataset...")
def load_baseline_raw_data() -> pd.DataFrame:
    """Load the historical 527-record UCI Water Treatment dataset."""
    df = load_water_treatment_data()
    return df


# =============================================================================
# 3. PIPELINE INFERENCE ENGINE  (UNCHANGED)
# =============================================================================

def run_mizu_inference(
    df_raw: pd.DataFrame,
    preprocessor: MizuPreprocessor,
    iforest: IsolationForestDetector,
    lof: LOFDetector,
    kmeans: MizuKMeansClusterer,
    pca: MizuPCA
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[List[str]]]:
    """
    Execute full inference pipeline on validated 38-feature sensor records:
    1. Verify presence of all 38 continuous sensor columns.
    2. Handle missing/null values via fitted median imputer.
    3. Scale features via fitted RobustScaler.
    4. Predict Isolation Forest flags & scores.
    5. Predict Novelty LOF flags & scores.
    6. Predict K-Means operational cluster regimes.
    7. Project into 2D PCA subspace.
    """
    required_features = preprocessor.feature_names
    missing_cols = [col for col in required_features if col not in df_raw.columns]
    if missing_cols:
        return None, None, missing_cols

    # Clean numeric features
    df_clean = df_raw.copy()
    for col in required_features:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Preserve or generate Date identifier
    if "Date" not in df_clean.columns:
        df_clean["Date"] = [f"Record_{i+1:04d}" for i in range(len(df_clean))]
    else:
        df_clean["Date"] = df_clean["Date"].astype(str)

    # Scale using pre-fitted preprocessor (Strictly in inference mode - NO retraining)
    X_scaled, _ = preprocessor.transform(df_clean)

    # 1. Isolation Forest inference
    iforest_flags = iforest.predict(X_scaled)
    iforest_scores = iforest.compute_anomaly_scores(X_scaled)

    # 2. Local Outlier Factor inference (novelty mode)
    lof_flags = lof.predict(X_scaled)
    lof_scores = lof.compute_lof_scores(X_scaled)

    # 3. K-Means clustering inference
    clusters = kmeans.predict(X_scaled)

    # 4. PCA 2D projection inference
    pca_coords = pca.transform(X_scaled)

    # Combine into unified results dataframe
    results = df_clean.copy()
    results["iforest_flag"] = iforest_flags
    results["iforest_score"] = np.round(iforest_scores, 4)
    results["lof_flag"] = lof_flags
    results["lof_score"] = np.round(lof_scores, 4)
    results["cluster"] = clusters
    results["regime_name"] = [REGIME_METADATA.get(c, {}).get("name", f"Cluster {c}") for c in clusters]
    results["PC1"] = np.round(pca_coords[:, 0], 4)
    results["PC2"] = np.round(pca_coords[:, 1], 4)

    # Categorize concordance
    def get_consensus_label(row):
        if row["iforest_flag"] == 1 and row["lof_flag"] == 1:
            return "Consensus Anomaly"
        elif row["iforest_flag"] == 1:
            return "iForest Only"
        elif row["lof_flag"] == 1:
            return "LOF Only"
        else:
            return "Normal Baseline"

    results["concordance"] = results.apply(get_consensus_label, axis=1)

    return results, X_scaled, None


# =============================================================================
# 4. UI HELPERS (presentation only)
# =============================================================================

def banner(kind: str, title: str, body: str) -> None:
    """Render a slim notice. kind = 'notice' | 'abstain'."""
    st.markdown(
        f'<div class="{kind}"><b>{title}</b> {body}</div>',
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str, tone: str = "plain", tip: str = "") -> str:
    tip_attr = f' title="{tip}"' if tip else ""
    return (
        f'<div class="kpi t-{tone}"{tip_attr}>'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'</div>'
    )


def kpi_grid(cards: List[str]) -> None:
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f"# {title}")
    st.caption(subtitle)


def with_status(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of results with plain-language status and short regime name (display only)."""
    out = df.copy()
    if "concordance" in out.columns:
        out["status"] = out["concordance"].map(STATUS_LABELS).fillna(out["concordance"])
    if "cluster" in out.columns:
        out["regime_short"] = out["cluster"].map(
            lambda c: REGIME_METADATA.get(c, {}).get("short", f"Cluster {c}")
        )
    return out


def style_fig(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", size=12, color="#C9D4E8"),
        margin=dict(l=8, r=8, t=36, b=8),
        hoverlabel=dict(bgcolor=C_PANEL, bordercolor=C_LINE, font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor=C_GRID, zeroline=False, linecolor=C_LINE)
    fig.update_yaxes(gridcolor=C_GRID, zeroline=False, linecolor=C_LINE)
    return fig


def overall_status(df: pd.DataFrame) -> Tuple[str, str, str]:
    """Return (label, css class, explanation) based on the most recent reading."""
    last = df.iloc[-1]
    flagged = int((df["concordance"] != "Normal Baseline").sum())
    both = int((df["concordance"] == "Consensus Anomaly").sum())
    when = str(last["Date"])
    if last["concordance"] == "Consensus Anomaly":
        return "Alert", "bad", f"The latest reading ({when}) is flagged by both models."
    if last["concordance"] in ("iForest Only", "LOF Only"):
        return "Watch", "warn", f"The latest reading ({when}) is flagged by one model."
    return "Stable", "ok", f"The latest reading ({when}) looks normal. {both} readings in this dataset were flagged by both models; {flagged} by at least one."


def flagged_table(df_res: pd.DataFrame, limit: Optional[int] = None):
    """Build a plain-language table of flagged readings (display only)."""
    d = with_status(df_res)
    d = d[d["concordance"] != "Normal Baseline"].sort_values(
        by=["iforest_score", "lof_score"], ascending=[False, False]
    )
    if limit:
        d = d.head(limit)
    cols = [c for c in TABLE_LABELS if c in d.columns]
    d = d[cols].rename(columns=TABLE_LABELS)
    fmt = {
        "iForest score": "{:.3f}",
        "LOF score": "{:.2f}",
        "Inflow (m3/day)": "{:,.0f}",
        "Zinc in (mg/L)": "{:.2f}",
        "Settleable solids out (mL/L)": "{:.2f}",
        "BOD removal (%)": "{:.1f}",
        "Solids removal (%)": "{:.1f}",
    }
    fmt = {k: v for k, v in fmt.items() if k in d.columns}
    return d.style.format(fmt, na_rep="-")


def _jump_to_explain(date: str) -> None:
    """Callback: open the Why Flagged page with a reading preselected."""
    st.session_state["page"] = "Why Flagged"
    st.session_state["explain_date_sel"] = date


def inspect_reading(df_res: pd.DataFrame, pos: int) -> None:
    """Detail panel for one reading (display only)."""
    row = df_res.iloc[pos]
    d = with_status(df_res.iloc[[pos]]).iloc[0]
    tone = {"Consensus Anomaly": "red", "iForest Only": "amber", "LOF Only": "amber"}.get(row["concordance"], "teal")
    st.markdown(f"##### Reading {row['Date']}")
    cards = [
        kpi_card("Status", d["status"], "Model agreement", tone),
        kpi_card("iForest score", f"{row['iforest_score']:.3f}", "Higher = more unusual", "plain"),
        kpi_card("LOF score", f"{row['lof_score']:.2f}", "Above about 1.5 = unusual", "plain"),
        kpi_card("Operating regime", d["regime_short"], f"Cluster {row['cluster']}", "blue"),
    ]
    kpi_grid(cards)
    key_cols = [c for c in ["Q-E", "ZN-E", "SED-S", "RD-DBO-G", "RD-SED-G"] if c in df_res.columns]
    if key_cols:
        vals = pd.DataFrame(
            {
                "Measurement": [TABLE_LABELS.get(c, c) for c in key_cols],
                "Value": [row[c] for c in key_cols],
            }
        )
        st.dataframe(vals, hide_index=True, width="stretch")
    if row["concordance"] != "Normal Baseline":
        st.button(
            "See which sensors drove this flag",
            key=f"jump_{pos}",
            on_click=_jump_to_explain,
            args=(str(row["Date"]),),
        )


def replay_block(df_res: pd.DataFrame) -> None:
    """Live replay: streams readings one batch at a time into a chart."""
    n = len(df_res)
    c1, c2, c3 = st.columns([1, 1.4, 2.6])
    with c1:
        live = st.toggle(
            "Live replay",
            key="replay_on",
            help="Plays the readings back in order, as if they were arriving from the plant.",
        )
    with c2:
        step = st.select_slider(
            "Speed (readings per update)", options=[1, 2, 5, 10, 20], value=5, key="replay_step"
        )
    with c3:
        st.caption("Turn on Live replay to watch the readings arrive and see anomalies appear as they happen.")

    if not live:
        st.session_state["replay_idx"] = 0
        st.session_state["replay_done"] = False
        return

    done = st.session_state.get("replay_done", False)
    if done:
        if st.button("Restart replay"):
            st.session_state["replay_idx"] = 0
            st.session_state["replay_done"] = False
            st.rerun()

    @st.fragment(run_every=None if done else 0.6)
    def _replay():
        idx = st.session_state.get("replay_idx", 0)
        if not st.session_state.get("replay_done", False):
            new_idx = min(idx + step, n)
            # Toast for any reading flagged by both models in this step
            window = df_res.iloc[idx:new_idx]
            hit = window[window["concordance"] == "Consensus Anomaly"]
            for _, r in hit.head(1).iterrows():
                st.toast(f"Both models flagged {r['Date']}", icon=":material/warning:")
            idx = new_idx
            st.session_state["replay_idx"] = idx
        shown = df_res.iloc[:idx]
        x = np.arange(1, idx + 1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=shown["iforest_score"], mode="lines", name="iForest score",
            line=dict(color=C_TEAL, width=1.6), customdata=shown["Date"],
            hovertemplate="%{customdata}<br>Score %{y:.3f}<extra></extra>",
        ))
        cons = shown[shown["concordance"] == "Consensus Anomaly"]
        fig.add_trace(go.Scatter(
            x=np.where((df_res["concordance"] == "Consensus Anomaly").values[:idx])[0] + 1,
            y=cons["iforest_score"], mode="markers", name="Flagged by both models",
            marker=dict(color=C_RED, size=10, symbol="diamond", line=dict(color="#FFFFFF", width=1)),
            customdata=cons["Date"],
            hovertemplate="%{customdata}<br>Score %{y:.3f}<extra></extra>",
        ))
        style_fig(fig, 300)
        fig.update_xaxes(range=[0, n], title="Reading number")
        fig.update_yaxes(range=[float(df_res["iforest_score"].min()) * 0.95, float(df_res["iforest_score"].max()) * 1.05],
                         title="iForest score")
        st.plotly_chart(fig, width="stretch", key="replay_chart")
        st.progress(idx / n, text=f"Reading {idx} of {n}")
        if idx >= n and not st.session_state.get("replay_done", False):
            st.session_state["replay_done"] = True
            st.rerun()

    _replay()


# =============================================================================
# 5. APPLICATION ENTRYPOINT
# =============================================================================

def main():
    st.set_page_config(
        page_title="MIZU | Intelligent Water Systems Monitoring",
        layout="wide",
        initial_sidebar_state="auto"
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    preprocessor, iforest, lof, kmeans, pca, explainer = load_all_models()
    ingestion_engine = MultiFormatIngestionEngine()

    if "data_source" not in st.session_state:
        st.session_state["data_source"] = "baseline"

    if "is_ml_permitted" not in st.session_state:
        st.session_state["is_ml_permitted"] = True

    if "active_results" not in st.session_state:
        raw_baseline = load_baseline_raw_data()
        results, X_scaled, _ = run_mizu_inference(
            raw_baseline, preprocessor, iforest, lof, kmeans, pca
        )
        st.session_state["active_results"] = results
        st.session_state["active_scaled"] = X_scaled
        st.session_state["active_raw"] = raw_baseline
        st.session_state["active_name"] = "UCI Baseline Dataset (527 Historical Records)"
        st.session_state["active_report"] = None
        st.session_state["is_ml_permitted"] = True

    st.session_state.setdefault("page", "Overview")

    # ------------------------------------------------------------------ Sidebar
    with st.sidebar:
        st.markdown(
            '<div class="brand">MIZU</div><div class="brand-sub">Water systems monitoring</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        page = st.radio("Navigation", options=PAGES, key="page", label_visibility="collapsed")

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="src-box"><b>Data in use</b><br>{st.session_state["active_name"]}<br>'
            f'{len(st.session_state["active_results"])} rows</div>',
            unsafe_allow_html=True,
        )
        if st.session_state.get("is_ml_permitted", True):
            st.markdown(
                '<span class="chip"><span class="dot ok"></span>All 38 sensors present. Models running.</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="chip"><span class="dot warn"></span>Sensors missing. Scoring skipped.</span>',
                unsafe_allow_html=True,
            )

        if st.session_state["data_source"] != "baseline":
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if st.button("Restore baseline data (527 records)", width="stretch"):
                raw_baseline = load_baseline_raw_data()
                results, X_scaled, _ = run_mizu_inference(
                    raw_baseline, preprocessor, iforest, lof, kmeans, pca
                )
                st.session_state["active_results"] = results
                st.session_state["active_scaled"] = X_scaled
                st.session_state["active_raw"] = raw_baseline
                st.session_state["active_name"] = "UCI Baseline Dataset (527 Historical Records)"
                st.session_state["active_report"] = None
                st.session_state["data_source"] = "baseline"
                st.session_state["is_ml_permitted"] = True
                st.rerun()

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        with st.expander("Model settings"):
            st.caption("Contamination rate: **3.0%** (about 16 events)")
            st.caption("LOF neighbourhood (k): **20 neighbours**")
            st.caption("Operating regimes: **K = 4 clusters**")
            st.caption("PCA retained variance: **81.83%**")
        st.caption("MIZU v1.2.0 | Decision-support prototype")

    # ------------------------------------------------------------------ Shared top line
    st.markdown(
        '<div class="notice" style="margin-top:0;">Decision-support tool for screening plant readings. '
        'It is not a drinking-water compliance check or a pathogen test. See <b>About</b> for limits.</div>',
        unsafe_allow_html=True,
    )

    abstain_note = (
        "Some of the 38 sensors this tool needs are missing, so the models did not score anything. "
        "Scoring on partial data would produce misleading results. Check <b>Upload Data</b> for the full report, "
        "or restore the baseline data from the sidebar."
    )

    # -------------------------------------------------------------------------
    # PAGE: OVERVIEW
    # -------------------------------------------------------------------------
    if page == "Overview":
        page_header(
            "Overview",
            "Flags unusual water-quality readings using two anomaly-detection models.",
        )

        if not st.session_state.get("is_ml_permitted", True):
            banner("abstain", "Scoring skipped.", abstain_note)
            st.markdown("##### Measurements extracted from your file")
            st.dataframe(st.session_state["active_results"].head(50), width="stretch")
        else:
            df_res = st.session_state["active_results"]
            total_obs = len(df_res)
            n_iforest = int(df_res["iforest_flag"].sum()) if "iforest_flag" in df_res.columns else 0
            n_lof = int(df_res["lof_flag"].sum()) if "lof_flag" in df_res.columns else 0
            n_consensus = int((df_res["concordance"] == "Consensus Anomaly").sum()) if "concordance" in df_res.columns else 0
            n_normal = int((df_res["concordance"] == "Normal Baseline").sum()) if "concordance" in df_res.columns else total_obs

            label, cls, why = overall_status(df_res)
            st.markdown(
                f'<div class="status"><span class="dot {cls}"></span>'
                f'<span class="badge {cls}">{label}</span>'
                f'<span class="status-text">{why}</span></div>',
                unsafe_allow_html=True,
            )

            kpi_grid([
                kpi_card("Readings checked", f"{total_obs:,}", "Rows analysed", "plain",
                         "Number of readings (rows) scored by the models."),
                kpi_card("Flagged by both", f"{n_consensus}", f"{n_consensus/total_obs*100:.1f}% of readings", "red",
                         "Readings that both models consider unusual. Highest confidence."),
                kpi_card("Isolation Forest", f"{n_iforest}", f"{n_iforest/total_obs*100:.1f}% flagged", "amber",
                         "Readings flagged by Isolation Forest, which catches readings that are extreme overall."),
                kpi_card("LOF", f"{n_lof}", f"{n_lof/total_obs*100:.1f}% flagged", "amber",
                         "Readings flagged by Local Outlier Factor, which catches readings that sit far from similar days."),
                kpi_card("Normal", f"{n_normal}", f"{n_normal/total_obs*100:.1f}% of readings", "teal",
                         "Readings neither model flagged."),
            ])

            st.markdown("##### Anomaly scores over time")
            st.caption(
                "Each point is one reading. Red diamonds are flagged by both models. "
                "Hover to see the date. Drag the slider under the chart to zoom."
            )

            x = np.arange(1, total_obs + 1)
            cons_mask = (df_res["concordance"] == "Consensus Anomaly").values
            fig_timeline = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                subplot_titles=(
                    "Isolation Forest score (0 = normal, 1 = extreme)",
                    "LOF score (about 1 = normal, above 1.5 = unusual)",
                ),
            )
            fig_timeline.add_trace(go.Scatter(
                x=x, y=df_res["iforest_score"], mode="lines", name="iForest score",
                line=dict(color=C_TEAL, width=1.4), customdata=df_res["Date"],
                hovertemplate="%{customdata}<br>iForest %{y:.3f}<extra></extra>"), row=1, col=1)
            fig_timeline.add_trace(go.Scatter(
                x=x[cons_mask], y=df_res.loc[cons_mask, "iforest_score"], mode="markers",
                name="Flagged by both models",
                marker=dict(color=C_RED, size=9, symbol="diamond", line=dict(color="#FFFFFF", width=1)),
                customdata=df_res.loc[cons_mask, "Date"],
                hovertemplate="%{customdata}<br>iForest %{y:.3f}<extra></extra>"), row=1, col=1)
            fig_timeline.add_trace(go.Scatter(
                x=x, y=df_res["lof_score"], mode="lines", name="LOF score",
                line=dict(color=C_BLUE, width=1.4), customdata=df_res["Date"],
                hovertemplate="%{customdata}<br>LOF %{y:.2f}<extra></extra>"), row=2, col=1)
            fig_timeline.add_trace(go.Scatter(
                x=x[cons_mask], y=df_res.loc[cons_mask, "lof_score"], mode="markers",
                name="Flagged by both (LOF)", showlegend=False,
                marker=dict(color=C_RED, size=9, symbol="diamond", line=dict(color="#FFFFFF", width=1)),
                customdata=df_res.loc[cons_mask, "Date"],
                hovertemplate="%{customdata}<br>LOF %{y:.2f}<extra></extra>"), row=2, col=1)
            style_fig(fig_timeline, 460)
            fig_timeline.update_annotations(font=dict(size=12, color=C_MUTED), x=0, xanchor="left")
            fig_timeline.update_xaxes(title_text="Reading number (date order)", row=2, col=1,
                                      rangeslider=dict(visible=True, thickness=0.06, bgcolor=C_PANEL))
            st.plotly_chart(fig_timeline, width="stretch")

            st.markdown("##### Live replay")
            replay_block(df_res)

            st.markdown("##### Top 10 most unusual readings")
            st.caption("Ranked by Isolation Forest score. Open Flagged Readings for the full list.")
            if int((df_res["concordance"] != "Normal Baseline").sum()) == 0:
                st.success("No unusual readings found in this data.")
            else:
                st.dataframe(flagged_table(df_res, limit=10), hide_index=True, width="stretch")

            with st.expander("Limitations and disclaimer"):
                st.markdown(
                    "MIZU is an unsupervised decision-support tool for secondary operational screening. "
                    "It does **not** certify potable drinking water compliance, detect microbiological pathogens "
                    "(for example *E. coli*, protozoan cysts, enteric viruses), or trigger automated physical action in the "
                    "treatment plant. Anomaly scores and sensor attributions show mathematical deviation from baseline "
                    "operating conditions and must be verified by accredited laboratory tests."
                )

    # -------------------------------------------------------------------------
    # PAGE: UPLOAD DATA
    # -------------------------------------------------------------------------
    elif page == "Upload Data":
        page_header(
            "Upload data",
            "Add a CSV, Excel or PDF file of plant readings. MIZU matches your column names to its 38 sensors and checks the data before scoring.",
        )

        col_upload, col_sample = st.columns([2, 1])

        with col_upload:
            uploaded_file = st.file_uploader(
                "Choose a file (CSV, Excel, PDF)",
                type=["csv", "xlsx", "xls", "pdf"],
                help="Supports CSV tables, Excel spreadsheets (horizontal and vertical), and digital PDF tables or reports."
            )

        with col_sample:
            with st.expander("Test files you can try"):
                st.caption(
                    "Synthetic files in `tests/fixtures/`:\n\n"
                    "- `valid_38_features.csv` (complete CSV)\n"
                    "- `valid_38_features.xlsx` (complete Excel)\n"
                    "- `vertical_lab_report.xlsx` (single lab sheet)\n"
                    "- `table_report.pdf` (PDF table log)\n"
                    "- `key_value_report.pdf` (PDF lab certificate)\n"
                    "- `incomplete_24_features.csv` (shows skipped scoring)\n"
                    "- `scanned_doc_simulated.pdf` (scanned PDF demo)"
                )

        # Process uploaded file
        if uploaded_file is not None:
            outcome = None
            extracted = None
            report = None
            results_up = None
            scaled_up = None
            try:
                with st.status("Reading file and matching sensors...", expanded=False) as status:
                    extracted = ingestion_engine.ingest_file(uploaded_file, uploaded_file.name)
                    report = extracted.report
                    if report.error_message:
                        outcome = "error"
                        status.update(label="Could not read this file", state="error")
                    elif report.is_complete_38:
                        status.update(label="Scoring readings with both models...")
                        results_up, scaled_up, _ = run_mizu_inference(
                            extracted.standardized_df, preprocessor, iforest, lof, kmeans, pca
                        )
                        outcome = "complete"
                        status.update(label="Done. All 38 sensors found and scored.", state="complete")
                    else:
                        outcome = "incomplete"
                        status.update(label="Some sensors are missing. Scoring skipped.", state="error")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

            if outcome == "error":
                st.error(report.error_message)
                if report.file_type == "scanned_pdf":
                    st.info(
                        "This looks like a scanned PDF. Text recognition (OCR, for example Tesseract) is not available "
                        "in this environment. Export the document as a text PDF, CSV or Excel file and upload that."
                    )
            elif outcome in ("complete", "incomplete"):
                st.session_state["active_report"] = report
                st.session_state["extracted_raw"] = extracted.raw_df
                st.session_state["extracted_payload"] = extracted

                if outcome == "complete":
                    st.success(
                        f"All 38 sensors matched ({report.coverage_pct}% coverage) from {report.file_type.upper()} "
                        f"across {report.total_rows} reading(s). Models ran successfully."
                    )
                    st.session_state["active_results"] = results_up
                    st.session_state["active_scaled"] = scaled_up
                    st.session_state["active_raw"] = extracted.raw_df
                    st.session_state["active_name"] = f"Uploaded File: {uploaded_file.name} ({report.total_rows} rows)"
                    st.session_state["data_source"] = "uploaded"
                    st.session_state["is_ml_permitted"] = True
                else:
                    st.warning(
                        f"Found {report.detected_canonical_count} of 38 sensors ({report.coverage_pct}% coverage)."
                    )
                    banner(
                        "abstain",
                        "Scoring skipped.",
                        "The models were trained on all 38 sensors covering the full treatment process. "
                        "Filling in missing sensors would produce made-up scores, so MIZU does not score incomplete data. "
                        "The report below lists exactly which sensors are missing.",
                    )
                    st.session_state["active_results"] = extracted.standardized_df if extracted.standardized_df is not None else extracted.raw_df
                    st.session_state["active_raw"] = extracted.raw_df
                    st.session_state["active_name"] = f"Uploaded (Incomplete): {uploaded_file.name} ({report.detected_canonical_count}/38 features)"
                    st.session_state["data_source"] = "uploaded_incomplete"
                    st.session_state["is_ml_permitted"] = False

        st.markdown("---")

        df_active = st.session_state["active_results"]
        report_active: Optional[IngestionReport] = st.session_state.get("active_report", None)

        st.markdown(f"##### Data in use: {st.session_state['active_name']}")

        tab_preview, tab_audit, tab_stages, tab_download = st.tabs([
            "Data preview", "Quality report", "By plant stage", "Download"
        ])

        with tab_preview:
            st.dataframe(df_active.head(50), width="stretch")

        with tab_audit:
            if report_active:
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Sensors found", f"{report_active.detected_canonical_count} / 38")
                with m2:
                    st.metric("Coverage", f"{report_active.coverage_pct}%")
                with m3:
                    st.metric("File type", report_active.file_type.upper())
                with m4:
                    ml_status_text = "Scored" if report_active.is_ml_permitted else "Skipped"
                    st.metric("Model scoring", ml_status_text)

                # Missing features breakdown by stage
                if report_active.missing_features:
                    st.markdown("##### Missing sensors by plant stage")
                    st.caption("These sensors are part of the 38 the models need but were not found in your file.")

                    for stage, missing_list in report_active.missing_by_stage.items():
                        stage_title = {
                            "influent": "Stage 1: Influent intake (-E)",
                            "primary": "Stage 2: Primary settler (-P)",
                            "secondary": "Stage 3: Secondary clarifier (-D)",
                            "effluent": "Stage 4: Effluent discharge (-S)",
                            "efficiency": "Stage 5: Removal efficiencies (RD-)"
                        }.get(stage, stage.capitalize())

                        missing_details = []
                        for code in missing_list:
                            meta = SENSOR_METADATA.get(code, {"name": code, "unit": ""})
                            missing_details.append(f"`{code}` ({meta['name']}, {meta['unit']})")
                        st.markdown(f"- **{stage_title}** ({len(missing_list)} missing): {', '.join(missing_details)}")

                # Ambiguous features
                if report_active.ambiguous_features:
                    st.markdown("##### Columns that need your confirmation")
                    st.caption("These columns do not say which treatment stage they belong to. Pick the sensor each one measures.")
                    with st.form("resolve_ambiguous_columns_form"):
                        resolutions = {}
                        for amb in report_active.ambiguous_features:
                            col_orig = amb["original_name"]
                            cands = ["(Leave Unmapped)"] + amb["candidate_canonicals"]
                            sel = st.selectbox(
                                f"Map column '{col_orig}' to:",
                                options=cands,
                                key=f"sel_amb_{col_orig}"
                            )
                            if sel != "(Leave Unmapped)":
                                resolutions[col_orig] = sel
                        submitted = st.form_submit_button("Confirm and re-check")
                        if submitted and resolutions:
                            cur_payload = st.session_state.get("extracted_payload", None)
                            if cur_payload is not None:
                                updated_extracted = ingestion_engine.apply_user_mappings(
                                    cur_payload,
                                    resolutions,
                                    filename=st.session_state.get("active_name", "uploaded_file")
                                )
                                st.session_state["extracted_payload"] = updated_extracted
                                st.session_state["active_report"] = updated_extracted.report
                                if updated_extracted.report.is_complete_38:
                                    st.success("All 38 sensors confirmed. Running the models...")
                                    res_ml, sc_ml, _ = run_mizu_inference(
                                        updated_extracted.standardized_df, preprocessor, iforest, lof, kmeans, pca
                                    )
                                    st.session_state["active_results"] = res_ml
                                    st.session_state["active_scaled"] = sc_ml
                                    st.session_state["is_ml_permitted"] = True
                                else:
                                    st.session_state["active_results"] = updated_extracted.standardized_df
                                    st.session_state["is_ml_permitted"] = False
                                st.rerun()

                # Unrecognised features
                if report_active.unrecognised_features:
                    st.markdown("##### Columns that were ignored")
                    st.caption("These columns are not among the 38 sensors, so they were left out.")
                    st.info(f"Ignored: {', '.join([f'`{u}`' for u in report_active.unrecognised_features])}")

                # Unit warnings
                if report_active.unit_warnings:
                    st.markdown("##### Unit conversions")
                    for w in report_active.unit_warnings:
                        st.caption(f"- {w}")

                # Feature Mapping Audit Table
                if report_active.mapped_features:
                    st.markdown("##### Sensors matched to your columns")
                    mapping_rows = []
                    for canon, orig in report_active.mapped_features.items():
                        meta = SENSOR_METADATA.get(canon, {"name": canon, "stage": "Unknown", "unit": ""})
                        mapping_rows.append({
                            "Sensor code": canon,
                            "Description": meta["name"],
                            "Plant stage": meta["stage"],
                            "Unit": meta["unit"],
                            "Your column": orig
                        })
                    st.dataframe(pd.DataFrame(mapping_rows), hide_index=True, width="stretch")

            else:
                # Baseline Historical Dataset missing summary
                st.caption(
                    "Gaps in the baseline data are filled with the median value. "
                    "Biochemical oxygen demand (BOD, shown as DBO) has more gaps because the lab test takes 5 days."
                )
                missing_summary = get_missing_summary(st.session_state["active_raw"])
                st.dataframe(missing_summary.head(15), width="stretch")

        with tab_stages:
            stages = get_feature_groups()
            selected_stage = st.selectbox(
                "Plant treatment stage:",
                options=list(stages.keys()),
                format_func=lambda s: {
                    "influent": "Stage 1: Plant influent / raw intake (-E)",
                    "primary": "Stage 2: Primary settler sedimentation (-P)",
                    "secondary": "Stage 3: Aeration and secondary clarification (-D)",
                    "effluent": "Stage 4: Treated discharge / effluent (-S)",
                    "efficiency": "Stage 5: Removal efficiencies (RD-)"
                }.get(s, s)
            )
            stage_cols = ["Date"] + [c for c in stages[selected_stage] if c in df_active.columns]
            st.dataframe(df_active[stage_cols].head(30), width="stretch")

        with tab_download:
            st.caption("Download the standardized table (your data with matched sensor names and units).")
            csv_bytes = df_active.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download standardized CSV",
                data=csv_bytes,
                file_name="mizu_standardized_telemetry.csv",
                mime="text/csv",
                width="stretch"
            )

    # -------------------------------------------------------------------------
    # PAGE: FLAGGED READINGS
    # -------------------------------------------------------------------------
    elif page == "Flagged Readings":
        page_header(
            "Flagged readings",
            "Where the two models agree and disagree. A reading flagged by both models is the strongest signal.",
        )

        if not st.session_state.get("is_ml_permitted", True):
            banner("abstain", "Scoring skipped.", abstain_note)
            st.dataframe(st.session_state["active_results"].head(30), width="stretch")
        else:
            df_res = st.session_state["active_results"]

            cat_counts = df_res["concordance"].value_counts()
            n_both = cat_counts.get("Consensus Anomaly", 0)
            n_iforest_only = cat_counts.get("iForest Only", 0)
            n_lof_only = cat_counts.get("LOF Only", 0)

            union_flags = n_both + n_iforest_only + n_lof_only
            jaccard = (n_both / union_flags) if union_flags > 0 else 1.0

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Flagged by both", f"{n_both}", delta=f"{n_both/len(df_res)*100:.1f}% of readings", delta_color="inverse")
            with c2:
                st.metric("Isolation Forest only", f"{n_iforest_only}", help="Flagged by Isolation Forest, normal under LOF")
            with c3:
                st.metric("LOF only", f"{n_lof_only}", help="Flagged by LOF, normal under Isolation Forest")
            with c4:
                st.metric("Agreement", f"{jaccard*100:.1f}%", help="Share of all flagged readings that both models flagged (Jaccard similarity)")

            st.markdown("##### Model scores compared")
            st.caption(
                "Each dot is a reading. Click a dot to inspect it, or pick a date below. "
                "Isolation Forest finds readings that are extreme overall. LOF finds readings that are far from similar days."
            )

            plot_df = with_status(df_res)
            plot_df["_row"] = np.arange(len(plot_df))
            hover_cols = [c for c in ["Date", "Q-E", "ZN-E", "SED-S", "regime_short"] if c in plot_df.columns]
            fig_scatter = px.scatter(
                plot_df,
                x="iforest_score",
                y="lof_score",
                color="status",
                color_discrete_map=STATUS_COLORS,
                category_orders={"status": list(STATUS_COLORS.keys())},
                custom_data=["_row"],
                hover_data=hover_cols,
                labels={
                    "iforest_score": "Isolation Forest score",
                    "lof_score": "LOF score",
                    "status": "Status",
                    "regime_short": "Regime",
                },
            )
            fig_scatter.update_traces(marker=dict(size=8, opacity=0.9, line=dict(width=0)))
            fig_scatter.add_vline(x=0.55, line_width=1, line_dash="dash", line_color="#FB923C",
                                  annotation_text="iForest borderline", annotation_font_size=11)
            fig_scatter.add_hline(y=1.5, line_width=1, line_dash="dash", line_color=C_AMBER,
                                  annotation_text="LOF boundary", annotation_font_size=11)
            style_fig(fig_scatter, 460)

            clicked_pos = None
            try:
                event = st.plotly_chart(
                    fig_scatter, width="stretch", key="concordance_scatter",
                    on_select="rerun", selection_mode="points",
                )
                pts = event["selection"]["points"] if event and event.get("selection") else []
                if pts:
                    clicked_pos = int(pts[0]["customdata"][0])
            except Exception:
                st.plotly_chart(fig_scatter, width="stretch")

            flagged_dates = list(df_res.loc[df_res["concordance"] != "Normal Baseline", "Date"])
            picked = st.selectbox(
                "Or pick a flagged reading to inspect",
                options=flagged_dates, index=None, placeholder="Choose a date",
                key="inspect_date",
            )
            if picked is not None:
                matches = np.where(df_res["Date"].values == picked)[0]
                if len(matches):
                    clicked_pos = int(matches[0])

            if clicked_pos is not None:
                st.markdown("---")
                inspect_reading(df_res, clicked_pos)

            st.markdown("##### All flagged readings")
            if union_flags == 0:
                st.success("No unusual readings found in this data.")
            else:
                st.dataframe(flagged_table(df_res), hide_index=True, width="stretch")

            with st.expander("Why do the models sometimes disagree?"):
                col_agree, col_disagree = st.columns(2)
                with col_agree:
                    st.markdown(
                        "**Flagged by both (for example `D-13/3/90`)**\n\n"
                        "- A severe breakdown that is extreme overall and also unlike similar days.\n"
                        "- Example: a clarifier failure pushed settleable solids out (`SED-S`) to 3.5 mL/L, "
                        "against a historical median of 0.02 mL/L.\n"
                        "- Both models flag these independently, so confidence is highest."
                    )
                with col_disagree:
                    st.markdown(
                        "**Only one model flags it (for example `D-12/8/90` vs `D-7/2/91`)**\n\n"
                        "- **Isolation Forest only (`D-12/8/90`)**: global extremes such as a storm surge "
                        "(`Q-E` above 50,000 m3/day).\n"
                        "- **LOF only (`D-7/2/91`)**: a reading far from its 20 nearest similar days, such as a zinc spike "
                        "(`ZN-E` = 10.0 mg/L), even if it is not extreme overall."
                    )

    # -------------------------------------------------------------------------
    # PAGE: PATTERNS
    # -------------------------------------------------------------------------
    elif page == "Patterns":
        page_header(
            "Patterns",
            "Groups readings into four operating regimes and shows them on a 2D map of the 38 sensors.",
        )

        if not st.session_state.get("is_ml_permitted", True):
            banner("abstain", "Scoring skipped.", abstain_note)
            st.dataframe(st.session_state["active_results"].head(30), width="stretch")
        else:
            df_res = with_status(st.session_state["active_results"])

            col_pca, col_summary = st.columns([2, 1])

            with col_pca:
                hover_cols = [c for c in ["Date", "status", "SED-S", "Q-E", "ZN-E"] if c in df_res.columns]
                fig_pca = px.scatter(
                    df_res,
                    x="PC1",
                    y="PC2",
                    color="regime_name",
                    symbol="status",
                    symbol_map={
                        "Flagged by both models": "diamond",
                        "Isolation Forest only": "triangle-up",
                        "LOF only": "triangle-down",
                        "Normal": "circle",
                    },
                    color_discrete_map={REGIME_METADATA[k]["name"]: REGIME_METADATA[k]["color"] for k in REGIME_METADATA},
                    hover_data=hover_cols,
                    labels={
                        "PC1": "Axis 1 (77.2% of variation)",
                        "PC2": "Axis 2 (4.6% of variation)",
                        "regime_name": "Operating regime",
                        "status": "Status",
                    },
                )
                fig_pca.update_traces(marker=dict(size=8, opacity=0.85))
                style_fig(fig_pca, 500)
                fig_pca.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0, font=dict(size=11)))
                st.plotly_chart(fig_pca, width="stretch")
                st.caption(
                    "PCA (principal component analysis) squeezes the 38 sensors into two axes. "
                    "Readings close together behave alike."
                )

            with col_summary:
                st.markdown("##### What the two axes mean")
                st.markdown(
                    "- **Axis 1 (77.2%)**: mostly clarification and effluent solids (`SED-S`, `RD-SED-G`). "
                    "High values point to clarification failure.\n"
                    "- **Axis 2 (4.6%)**: mostly influent suspended solids (`SS-E`, `SS-P`), the raw solids load.\n"
                    "- Together they keep **81.8%** of the variation in the data."
                )

                st.markdown("##### Days in each regime")
                cluster_counts = df_res["cluster"].value_counts().sort_index()
                for c_id in sorted(REGIME_METADATA.keys()):
                    count = int(cluster_counts.get(c_id, 0))
                    pct = count / len(df_res) * 100
                    st.markdown(
                        f'<span class="dot" style="display:inline-block;background:{REGIME_METADATA[c_id]["color"]};margin-right:8px;"></span>'
                        f'**{REGIME_METADATA[c_id]["short"]}**: {count} days ({pct:.1f}%)',
                        unsafe_allow_html=True,
                    )

            st.markdown("##### The four regimes")
            cols = st.columns(4)
            for i, col in enumerate(cols):
                meta = REGIME_METADATA[i]
                with col:
                    st.markdown(
                        f'<div class="panel" style="border-top:3px solid {meta["color"]};">'
                        f'<div class="panel-title">{meta["short"]}</div>'
                        f'<div class="panel-text">{meta["description"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # -------------------------------------------------------------------------
    # PAGE: WHY FLAGGED
    # -------------------------------------------------------------------------
    elif page == "Why Flagged":
        page_header(
            "Why flagged",
            "See which sensors pushed a reading toward being flagged.",
        )

        if not st.session_state.get("is_ml_permitted", True):
            banner("abstain", "Scoring skipped.", abstain_note)
            st.dataframe(st.session_state["active_results"].head(30), width="stretch")
        else:
            st.caption(
                "Method: for each sensor, the tool swaps that reading for the normal (median) value and checks how much the "
                "anomaly score drops. A bigger drop means that sensor mattered more to the flag."
            )

            df_res = st.session_state["active_results"]
            scaled_df = st.session_state["active_scaled"]
            raw_df = st.session_state["active_raw"]

            anomalies_only = df_res[df_res["concordance"] != "Normal Baseline"]

            col_select1, col_select2, col_select3 = st.columns([2, 1, 1])

            with col_select1:
                date_options = list(anomalies_only["Date"]) if not anomalies_only.empty else list(df_res["Date"])
                if st.session_state.get("explain_date_sel") not in date_options:
                    st.session_state.pop("explain_date_sel", None)

                def format_date_option(d):
                    row = df_res[df_res["Date"] == d].iloc[0]
                    return f"{d}  |  {STATUS_LABELS.get(row['concordance'], row['concordance'])}  |  iForest {row['iforest_score']:.3f}, LOF {row['lof_score']:.2f}"

                selected_date = st.selectbox(
                    "Reading to explain",
                    options=date_options,
                    format_func=format_date_option,
                    key="explain_date_sel",
                )

            with col_select2:
                model_choice = st.radio(
                    "Model",
                    options=["iforest", "lof"],
                    format_func=lambda m: "Isolation Forest" if m == "iforest" else "LOF",
                    horizontal=True
                )

            with col_select3:
                top_n = st.slider("Sensors to show", min_value=3, max_value=8, value=5)

            if selected_date:
                idx = df_res[df_res["Date"] == selected_date].index[0]
                sample_scaled = scaled_df.loc[idx:idx]
                sample_raw = raw_df.loc[idx] if idx < len(raw_df) else None

                with st.spinner("Working out which sensors mattered..."):
                    top_df, metadata = explainer.explain_sample(
                        sample=sample_scaled,
                        raw_row=sample_raw,
                        sample_id=selected_date,
                        model_type=model_choice,
                        top_n=top_n
                    )

                fig_bar = px.bar(
                    top_df,
                    x="score_reduction",
                    y="sensor_name",
                    orientation="h",
                    color="stage",
                    text="score_reduction",
                    color_discrete_map={
                        "Influent (-E)": C_BLUE,
                        "Primary (-P)": C_TEAL,
                        "Secondary (-D)": "#A78BFA",
                        "Effluent (-S)": C_AMBER,
                        "Efficiency (RD-)": C_RED
                    },
                    labels={
                        "score_reduction": "Score drop when set to normal",
                        "sensor_name": "Sensor",
                        "stage": "Plant stage"
                    },
                    title=f"Top {top_n} sensors for {selected_date} ({metadata['model_name']})"
                )
                fig_bar.update_traces(
                    texttemplate="%{text:.4f}",
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>Stage: %{customdata[0]}<br>Score drop: %{x:.6f}<br>Raw value: %{customdata[1]} %{customdata[2]}<extra></extra>",
                    customdata=top_df[["stage", "raw_value", "unit"]].values
                )
                style_fig(fig_bar, 380)
                fig_bar.update_layout(yaxis=dict(autorange="reversed"), title_font=dict(size=13))
                st.plotly_chart(fig_bar, width="stretch")

                st.markdown("##### Sensor details")
                display_attribution = top_df[[
                    "feature_code", "sensor_name", "stage", "raw_value", "unit",
                    "scaled_value", "original_score", "ablated_score", "score_reduction"
                ]].copy()
                display_attribution = display_attribution.rename(columns={
                    "feature_code": "Code", "sensor_name": "Sensor", "stage": "Stage",
                    "raw_value": "Reading", "unit": "Unit", "scaled_value": "Distance from normal",
                    "original_score": "Score", "ablated_score": "Score if normal",
                    "score_reduction": "Score drop",
                })
                st.dataframe(
                    display_attribution.style.format({
                        "Reading": "{:.2f}",
                        "Distance from normal": "{:+.3f} IQR",
                        "Score": "{:.4f}",
                        "Score if normal": "{:.4f}",
                        "Score drop": "{:.6f}"
                    }),
                    hide_index=True,
                    width="stretch"
                )

                banner(
                    "notice",
                    "How to read this.",
                    f"Setting <b>{metadata['top_feature']}</b> (<code>{metadata['top_feature_code']}</code>) to normal lowers the "
                    f"{metadata['model_name']} score by <b>{metadata['top_score_reduction']:.6f}</b>. "
                    f"That means the model was most sensitive to this measurement. It does not prove a chemical failure "
                    f"without a laboratory check.",
                )

    # -------------------------------------------------------------------------
    # PAGE: ABOUT
    # -------------------------------------------------------------------------
    elif page == "About":
        page_header(
            "About MIZU",
            "How the pipeline works, why it was built this way, and what it cannot do.",
        )

        tab_how, tab_why, tab_limits, tab_terms = st.tabs([
            "How it works", "Design choices", "Limits", "Glossary"
        ])

        with tab_how:
            st.markdown("##### Pipeline")
            st.code(
                "Upload (CSV / XLSX / PDF)\n"
                "   |\n"
                "   v\n"
                "Ingestion engine\n"
                " - detects file format (CSV, Excel, PDF tables or text)\n"
                " - matches column names to sensors, flags ambiguous ones\n"
                " - converts units (for example m3/h to m3/day, ppm to mg/L)\n"
                "   |\n"
                "   +-- all 38 sensors present --------+-- fewer than 38 sensors\n"
                "   |                                   |\n"
                "   v                                   v\n"
                "Median imputer                    Scoring skipped\n"
                "Robust scaler (median, IQR)       Missing-sensor report\n"
                "   |\n"
                "   +-- Isolation Forest (contamination 3.0%)\n"
                "   +-- Novelty LOF (k = 20)\n"
                "   +-- K-Means regimes (K = 4)\n"
                "   +-- PCA 2D map (81.83% variance kept)\n"
                "   |\n"
                "   v\n"
                "Agreement between the two anomaly models\n"
                "   |\n"
                "   v\n"
                "Feature ablation: which sensors drove the flag",
                language=None,
            )

        with tab_why:
            q1, q2 = st.columns(2)
            with q1:
                st.markdown(
                    "**Why skip scoring when sensors are missing?**\n\n"
                    "The models learned patterns across all 38 sensors from every treatment stage. Filling in missing sensors with "
                    "constants or medians adds bias and distorts the distance calculations LOF and Isolation Forest depend on. "
                    "MIZU would rather say \"not enough data\" than invent a score.\n\n"
                    "**Why a robust scaler?**\n\n"
                    "Wastewater has shock loads, such as zinc spikes up to 33.5 mg/L against a median of 1.5 mg/L. "
                    "A standard scaler lets those spikes stretch the scale and hide normal variation. The robust scaler uses the "
                    "median and interquartile range, which spikes barely move.\n\n"
                    "**Why novelty-mode LOF?**\n\n"
                    "Standard LOF cannot score new data. Novelty mode keeps the training points so incoming or uploaded readings "
                    "can be scored."
                )
            with q2:
                st.markdown(
                    "**Why four regimes instead of two?**\n\n"
                    "Two clusters scored a high silhouette (0.939) only by splitting 3 extreme days from 524 normal ones. "
                    "Four clusters show real structure: normal operation (510 days), clarifier solids washout (14 days), "
                    "hydraulic overload (1 day) and critical separation failure (2 days).\n\n"
                    "**Why feature ablation instead of SHAP or LIME?**\n\n"
                    "SHAP and LIME rely on approximations that struggle with distance-based unsupervised models. Ablation is "
                    "exact and easy to explain: replace one reading with the historical median and see how the score changes."
                )

        with tab_limits:
            st.markdown(
                "1. **Not for drinking water.** MIZU monitors industrial and municipal wastewater treatment. It cannot certify "
                "potable water against EPA or WHO guidelines.\n"
                "2. **No pathogen detection.** The data covers physical and chemical measures (pH, solids, flow, oxygen demand). "
                "It does not detect *Cryptosporidium*, *Giardia*, *Legionella* or *Norovirus*.\n"
                "3. **A person decides.** Every flag needs a physical sample and lab confirmation by a certified operator "
                "before any process change."
            )
            banner(
                "notice",
                "Regulatory and engineering notice.",
                "MIZU is an unsupervised decision-support tool for secondary operational screening. It does not trigger "
                "automated physical action in the treatment plant. Scores show mathematical deviation from baseline "
                "operating conditions and must be verified by accredited laboratory tests.",
            )

        with tab_terms:
            st.markdown(
                "- **Anomaly**: a reading that looks unlike normal plant behaviour.\n"
                "- **Isolation Forest**: finds readings that are easy to separate from the rest, meaning extreme overall.\n"
                "- **LOF (Local Outlier Factor)**: finds readings that are far from the days most similar to them.\n"
                "- **Flagged by both models**: the two models agree. This is the strongest signal.\n"
                "- **Regime**: a typical way the plant operates, found automatically by grouping similar days (K-Means).\n"
                "- **PCA**: a method that squeezes many sensors into two axes so they can be drawn on a map.\n"
                "- **Feature ablation**: swapping one sensor for a normal value to see how much it mattered to a flag.\n"
                "- **IQR**: interquartile range, a spread measure that ignores extreme values."
            )


if __name__ == "__main__":
    main()