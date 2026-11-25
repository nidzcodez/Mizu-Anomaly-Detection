"""
MIZU: Intelligent Water Systems Monitoring Dashboard
===================================================
A real SaaS / SCADA decision-support application for unsupervised anomaly detection,
operational regime clustering, and explainability in wastewater treatment plants.

Dataset: UCI Water Treatment Plant (ID: 106)
Frameworks: Streamlit, Pandas, Plotly, Scikit-Learn

Strict Constraints:
- load_all_models and run_mizu_inference interface, signatures and return values are preserved.
- Model logic, thresholds, data processing, and computed outputs are 100% unchanged.
- Multipage top navigation bar with isolated callable pages.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

import base64
import hashlib
import io
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
)


# =============================================================================
# 1. CONSTANTS & METADATA (SINGLE SOURCE OF TRUTH)
# =============================================================================

# Design System Palette (Tokens mirrored from assets/style.css)
C_BG = "#F6F9FC"
C_SURFACE = "#FFFFFF"
C_SURFACE_2 = "#F0F6F9"
C_TOPBAR = "#FFFFFF"
C_BORDER = "#D9E4EA"
C_GRID = "#E7EEF2"
C_TEXT = "#17324D"
C_TEXT_MUTED = "#60778B"
C_TEXT_FAINT = "#7F93A3"
C_ACCENT = "#1976D2"      # Water blue: Primary / Selection / OK
C_ANOMALY = "#EF4444"     # Red: Flagged-by-both / Critical Anomaly only
C_WARN = "#F5B83D"        # Amber: Single-model flags / Watch
C_INFO = "#2F80C8"        # Blue: Neutral telemetry
C_OK = "#2F80C8"

# Cluster operational regime taxonomy
REGIME_METADATA = {
    0: {
        "name": "Regime 0: Baseline Normal Operations",
        "short": "Baseline normal operations",
        "description": "Stable biological and physical clarification. Standard influent flow and organic loading with high removal efficiencies across all treatment stages.",
        "color": "#1E88E5"
    },
    1: {
        "name": "Regime 1: Secondary Settler Solids Washout",
        "short": "Solids washout",
        "description": "Elevated effluent settleable solids and suspended solids. Biological clarification impaired, causing sub-optimal final sedimentation.",
        "color": "#64B5F6"
    },
    2: {
        "name": "Regime 2: Hydraulic & Organic Surge",
        "short": "Hydraulic / organic surge",
        "description": "Severe volumetric inflow spike or industrial organic slug. Elevated influent COD and primary solids with stressed downstream stages.",
        "color": "#1565C0"
    },
    3: {
        "name": "Regime 3: Critical Solids Separation Breakdown",
        "short": "Separation breakdown",
        "description": "Extreme physical separation failure in both primary and secondary sedimentation stages. Severe plant performance collapse.",
        "color": C_ANOMALY
    }
}

# Single stage color map for Plotly charts
STAGE_COLOR_MAP = {
    "Influent (-E)": "#1E88E5",
    "Primary (-P)": "#42A5F5",
    "Secondary (-D)": "#1565C0",
    "Effluent (-S)": "#90CAF9",
    "Efficiency (RD-)": "#274C77"
}

STATUS_LABELS = {
    "Consensus Anomaly": "Flagged by both models",
    "iForest Only": "Isolation Forest only",
    "LOF Only": "LOF only",
    "Normal Baseline": "Normal",
}

STATUS_COLORS = {
    "Flagged by both models": C_ANOMALY,
    "Isolation Forest only": "#1976D2",
    "LOF only": "#64B5F6",
    "Normal": "#718096",
}

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

ABSTAIN_NOTE = (
    "Some of the 38 sensors this tool needs are missing, so the models did not score anything. "
    "Scoring on partial data would produce misleading results. Check <b>Upload Data</b> for the full report, "
    "or restore the baseline data from the sidebar."
)


# =============================================================================
# 2. CACHED MODEL & DATA LOADING  (EXACT SIGNATURES PRESERVED)
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
# 3. PIPELINE INFERENCE ENGINE  (EXACT SIGNATURES PRESERVED)
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
# 4. PRESENTATION HELPERS (PHASE 1-4 COMPONENT SYSTEM)
# =============================================================================

def apply_plot_theme(fig: go.Figure, height: int = 440, legend: str = "top") -> go.Figure:
    """The single source of truth for all Plotly charts."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, sans-serif", size=12, color=C_TEXT),
        margin=dict(l=55, r=15, t=45, b=40),
        hoverlabel=dict(bgcolor=C_SURFACE_2, bordercolor=C_BORDER, font_size=12, font_family="Inter, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.06,
            xanchor="left",
            x=0,
            font=dict(size=11, color=C_TEXT_MUTED),
            itemsizing="constant"
        ),
        title=None,
        modebar=dict(
            bgcolor="rgba(0,0,0,0)",
            color=C_TEXT_MUTED,
            activecolor=C_ACCENT,
            orientation="h",
        )
    )
    fig.update_xaxes(
        gridcolor=C_GRID,
        gridwidth=1,
        zeroline=False,
        linecolor=C_BORDER,
        tickfont=dict(size=11, color=C_TEXT_MUTED),
        title_font=dict(size=12, color=C_TEXT),
        title_standoff=8,
        automargin=True
    )
    fig.update_yaxes(
        gridcolor=C_GRID,
        gridwidth=1,
        zeroline=False,
        linecolor=C_BORDER,
        tickfont=dict(size=11, color=C_TEXT_MUTED),
        title_font=dict(size=12, color=C_TEXT),
        title_standoff=8,
        automargin=True
    )
    return fig


def page_header(title: str, subtitle: str, how_to_read_text: Optional[str] = None) -> None:
    """Header block with title, subtitle, and an optional 'How to read this' popover button."""
    c_title, c_pop = st.columns([10, 2], vertical_alignment="top")
    with c_title:
        st.markdown(f"<h1>{title}</h1>", unsafe_allow_html=True)
        st.markdown(f'<div class="header-subtitle">{subtitle}</div>', unsafe_allow_html=True)
    with c_pop:
        if how_to_read_text:
            with st.popover("How to read", icon=":material/help_outline:", width="stretch"):
                st.markdown("##### How to read this view")
                st.markdown(how_to_read_text)
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)


def zone_eyebrow(label: str) -> None:
    """Render zone eyebrow label."""
    st.markdown(f'<div class="zone-eyebrow">{label}</div>', unsafe_allow_html=True)


def status_banner(level: str, badge_text: str, text: str) -> None:
    """Render a plain-language operational summary without decorative status dots."""
    st.markdown(
        f'<div class="status-banner {level}">'
        f'<div><span class="status-badge {level}">{badge_text}</span>'
        f'<div class="status-text">{text}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, caption: str, tone: str = "plain", tip: str = "") -> str:
    """HTML string for equal-height KPI card aligned to 12-column grid."""
    tip_attr = f' title="{tip}"' if tip else ""
    return (
        f'<div class="kpi-card tone-{tone}"{tip_attr}>'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{caption}</div>'
        f'</div>'
    )


def kpi_grid(cards: List[str]) -> None:
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def display_df(df: pd.DataFrame, limit: Optional[int] = None) -> pd.DataFrame:
    """Return a display-only copy with missing values rendered as em-dash, without altering source data."""
    d = df.copy()
    if limit is not None:
        d = d.head(limit)
    return d.fillna("—")


def with_status(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of results with plain-language status and short regime name."""
    out = df.copy()
    if "concordance" in out.columns:
        out["status"] = out["concordance"].map(STATUS_LABELS).fillna(out["concordance"])
    if "cluster" in out.columns:
        out["regime_short"] = out["cluster"].map(
            lambda c: REGIME_METADATA.get(c, {}).get("short", f"Cluster {c}")
        )
    return out


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
    return "Operating normally", "ok", (
        f"The latest reading ({when}) is within the established operating baseline. "
        f"{both} readings were flagged by both models and {flagged} by at least one model; "
        "review Flagged Readings when investigating historical events."
    )


def flagged_table(df_res: pd.DataFrame, limit: Optional[int] = None):
    """Build a plain-language table of flagged readings."""
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
    return d.style.format(fmt, na_rep="—")


def _jump_to_explain(date: str) -> None:
    """Callback: open the Why Flagged page with a reading preselected."""
    st.session_state["explain_date_sel"] = date
    st.switch_page("why-flagged")


def inspect_reading(df_res: pd.DataFrame, pos: int) -> None:
    """Detail panel for one reading (display only)."""
    row = df_res.iloc[pos]
    d = with_status(df_res.iloc[[pos]]).iloc[0]
    tone = {"Consensus Anomaly": "red", "iForest Only": "amber", "LOF Only": "amber"}.get(row["concordance"], "teal")
    st.markdown(f"##### Reading {row['Date']}")
    cards = [
        kpi_card("Status", d["status"], "Model agreement", tone),
        kpi_card("iForest score", f"{row['iforest_score']:.3f}", "Higher = more unusual", "plain"),
        kpi_card("LOF score", f"{row['lof_score']:.2f}", "Above ~1.5 = unusual", "plain"),
        kpi_card("Operating regime", d["regime_short"], f"Cluster {row['cluster']}", "plain"),
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
    c1, c2, c3 = st.columns([1, 1.4, 2.6], vertical_alignment="center")
    with c1:
        live = st.toggle(
            "Live replay",
            key="replay_on",
            help="Plays the readings back in order, as if they were arriving from the plant.",
        )
    with c2:
        step = st.select_slider(
            "Speed (readings / update)", options=[1, 2, 5, 10, 20], value=5, key="replay_step"
        )
    with c3:
        st.caption("Turn on Live replay to watch readings stream in sequence and observe anomalies as they occur.")

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
            line=dict(color=C_ACCENT, width=2), customdata=shown["Date"],
            hovertemplate="%{customdata}<br>Score %{y:.3f}<extra></extra>",
        ))
        cons = shown[shown["concordance"] == "Consensus Anomaly"]
        fig.add_trace(go.Scatter(
            x=np.where((df_res["concordance"] == "Consensus Anomaly").values[:idx])[0] + 1,
            y=cons["iforest_score"], mode="markers", name="Flagged by both models",
            marker=dict(color=C_ANOMALY, size=10, symbol="diamond", line=dict(color="#FFFFFF", width=1)),
            customdata=cons["Date"],
            hovertemplate="%{customdata}<br>Score %{y:.3f}<extra></extra>",
        ))
        apply_plot_theme(fig, 320)
        fig.update_xaxes(range=[0, n], title="Reading number")
        fig.update_yaxes(
            range=[float(df_res["iforest_score"].min()) * 0.95, float(df_res["iforest_score"].max()) * 1.05],
            title="iForest score"
        )
        st.plotly_chart(fig, width="stretch", key="replay_chart", config={"displaylogo": False})
        st.progress(idx / n, text=f"Reading {idx} of {n}")
        if idx >= n and not st.session_state.get("replay_done", False):
            st.session_state["replay_done"] = True
            st.rerun()

    _replay()


# =============================================================================
# 5. SIX ISOLATED PAGE CALLABLES
# =============================================================================

def page_overview():
    page_header(
        "Overview",
        "Plant health at a glance. Unsupervised anomaly screening across 38 water-treatment sensors.",
        how_to_read_text=(
            "- **Top KPIs**: Summarize total readings checked and agreement between models.\n"
            "- **Anomaly Timeline**: Plots Isolation Forest and LOF scores over time. Red diamonds mark consensus anomalies.\n"
            "- **Plant at a glance**: Displays current raw sensor values across Influent, Primary, Secondary, and Effluent stages."
        )
    )

    if not st.session_state.get("is_ml_permitted", True):
        status_banner("warn", "Scoring Skipped", ABSTAIN_NOTE)
        zone_eyebrow("EXTRACTED MEASUREMENTS")
        st.dataframe(display_df(st.session_state["active_results"].head(50)), width="stretch")
        return

    df_res = st.session_state["active_results"]
    total_obs = len(df_res)
    n_iforest = int(df_res["iforest_flag"].sum()) if "iforest_flag" in df_res.columns else 0
    n_lof = int(df_res["lof_flag"].sum()) if "lof_flag" in df_res.columns else 0
    n_consensus = int((df_res["concordance"] == "Consensus Anomaly").sum()) if "concordance" in df_res.columns else 0
    n_normal = int((df_res["concordance"] == "Normal Baseline").sum()) if "concordance" in df_res.columns else total_obs

    # ZONE 1: STATUS
    with st.container(border=True):
        zone_eyebrow("STATUS")
        label, cls, why = overall_status(df_res)
        status_banner(cls, label, why)

    # ZONE 2: KEY NUMBERS
    with st.container(border=True):
        zone_eyebrow("KEY NUMBERS")
        kpi_grid([
            kpi_card("Readings checked", f"{total_obs:,}", "Rows analysed", "plain",
                     "Number of readings (rows) scored by the models."),
            kpi_card("Flagged by both", f"{n_consensus}", f"{n_consensus/total_obs*100:.1f}% of readings", "red",
                     "Readings that both models consider unusual. Highest confidence."),
            kpi_card("Isolation Forest", f"{n_iforest}", f"{n_iforest/total_obs*100:.1f}% flagged", "amber",
                     "Readings flagged by Isolation Forest, which catches readings that are extreme overall."),
            kpi_card("LOF", f"{n_lof}", f"{n_lof/total_obs*100:.1f}% flagged", "amber",
                     "Readings flagged by Local Outlier Factor, which catches readings that sit far from similar days."),
            kpi_card("Normal", f"{n_normal}", f"{n_normal/total_obs*100:.1f}% of readings", "plain",
                     "Readings neither model flagged."),
        ])

    # ZONE 3: ANALYSIS (Hero chart 8/12 + Right rail 4/12)
    with st.container(border=True):
        zone_eyebrow("ANALYSIS")
        col_chart, col_rail = st.columns([8, 4])

        with col_chart:
            x = np.arange(1, total_obs + 1)
            cons_mask = (df_res["concordance"] == "Consensus Anomaly").values
            fig_timeline = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.14,
                subplot_titles=("Isolation Forest score", "LOF score"),
            )
            fig_timeline.add_trace(go.Scatter(
                x=x, y=df_res["iforest_score"], mode="lines", name="iForest score",
                line=dict(color=C_ACCENT, width=2), customdata=df_res["Date"],
                hovertemplate="%{customdata}<br>iForest %{y:.3f}<extra></extra>"), row=1, col=1)
            fig_timeline.add_trace(go.Scatter(
                x=x[cons_mask], y=df_res.loc[cons_mask, "iforest_score"], mode="markers",
                name="Flagged by both models",
                marker=dict(color=C_ANOMALY, size=8, symbol="diamond", line=dict(color="#FFFFFF", width=1)),
                customdata=df_res.loc[cons_mask, "Date"],
                hovertemplate="%{customdata}<br>iForest %{y:.3f}<extra></extra>"), row=1, col=1)
            fig_timeline.add_trace(go.Scatter(
                x=x, y=df_res["lof_score"], mode="lines", name="LOF score",
                line=dict(color=C_INFO, width=2), customdata=df_res["Date"],
                hovertemplate="%{customdata}<br>LOF %{y:.2f}<extra></extra>"), row=2, col=1)
            fig_timeline.add_trace(go.Scatter(
                x=x[cons_mask], y=df_res.loc[cons_mask, "lof_score"], mode="markers",
                name="Flagged by both (LOF)", showlegend=False,
                marker=dict(color=C_ANOMALY, size=8, symbol="diamond", line=dict(color="#FFFFFF", width=1)),
                customdata=df_res.loc[cons_mask, "Date"],
                hovertemplate="%{customdata}<br>LOF %{y:.2f}<extra></extra>"), row=2, col=1)

            apply_plot_theme(fig_timeline, 360)
            fig_timeline.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0, font=dict(size=11, color=C_TEXT_MUTED)),
                margin=dict(l=55, r=15, t=55, b=40)
            )
            fig_timeline.layout.annotations[0].update(y=1.00, font=dict(size=11, color=C_TEXT_MUTED), x=0, xanchor="left")
            fig_timeline.layout.annotations[1].update(font=dict(size=11, color=C_TEXT_MUTED), x=0, xanchor="left")
            fig_timeline.update_xaxes(title_text="Reading number (date order)", row=2, col=1)
            fig_timeline.update_yaxes(title_text="iForest score", row=1, col=1)
            fig_timeline.update_yaxes(title_text="LOF score", row=2, col=1)
            st.plotly_chart(fig_timeline, width="stretch", config={"displaylogo": False})

        with col_rail:
            # Key Card
            st.markdown(
                f"""
                <div class="rail-card">
                    <div class="rail-title"><span class="chip-dot"></span>Legend & Key</div>
                    <div class="key-list">
                        <div class="key-item"><div class="key-diamond-red"></div><span><b>Flagged by both models</b>: Strongest consensus anomaly</span></div>
                        <div class="key-item"><div class="key-circle-amber"></div><span><b>One model flag</b>: Borderline outlier (iForest or LOF)</span></div>
                        <div class="key-item"><div class="key-circle-blue"></div><span><b>Normal baseline</b>: Operates within historical envelope</span></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Plant at a Glance Card (Latest Raw Values)
            latest = df_res.iloc[-1]
            qe = f"{latest.get('Q-E', 0):,.0f} m³/d" if 'Q-E' in latest else "—"
            sse = f"{latest.get('SS-E', 0):.0f} mg/L" if 'SS-E' in latest else "—"
            ssp = f"{latest.get('SS-P', 0):.0f} mg/L" if 'SS-P' in latest else "—"
            sedp = f"{latest.get('SED-P', 0):.1f} mL/L" if 'SED-P' in latest else "—"
            ssd = f"{latest.get('SS-D', 0):.0f} mg/L" if 'SS-D' in latest else "—"
            sss = f"{latest.get('SS-S', 0):.0f} mg/L" if 'SS-S' in latest else "—"
            seds = f"{latest.get('SED-S', 0):.2f} mL/L" if 'SED-S' in latest else "—"

            st.markdown(
                f"""
                <div class="rail-card">
                    <div class="rail-title"><span class="chip-dot"></span>Plant at a glance (Latest: {latest.get('Date', 'N/A')})</div>
                    <div class="plant-strip">
                        <div class="stage-tile"><span class="stage-tile-name">Stage 1: Influent</span><span class="stage-tile-vals">Q: {qe} | SS: {sse}</span></div>
                        <div class="stage-tile"><span class="stage-tile-name">Stage 2: Primary</span><span class="stage-tile-vals">SS: {ssp} | Sed: {sedp}</span></div>
                        <div class="stage-tile"><span class="stage-tile-name">Stage 3: Aeration</span><span class="stage-tile-vals">SS: {ssd}</span></div>
                        <div class="stage-tile"><span class="stage-tile-name">Stage 4: Effluent</span><span class="stage-tile-vals">SS: {sss} | Sed: {seds}</span></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # ZONE 4: DETAILS & REPLAY
    with st.container(border=True):
        zone_eyebrow("DETAILS & EXPLORATION")
        tab_replay, tab_top, tab_works = st.tabs(["Live replay", "Top 10 most unusual readings", "How MIZU works"])

        with tab_replay:
            replay_block(df_res)

        with tab_top:
            st.caption("Ranked by Isolation Forest score. Open Flagged Readings for the complete table.")
            if int((df_res["concordance"] != "Normal Baseline").sum()) == 0:
                st.success("No unusual readings found in this data.")
            else:
                st.dataframe(flagged_table(df_res, limit=10), hide_index=True, width="stretch")

        with tab_works:
            st.markdown(
                """
                <div class="step-flow">
                    <div class="step-card">
                        <div class="step-num">Step 1</div>
                        <div class="step-title">Ingest & Verify</div>
                        <div class="step-desc">38 continuous sensors mapped across all 4 plant stages. Missing values strictly audited.</div>
                    </div>
                    <div class="step-card">
                        <div class="step-num">Step 2</div>
                        <div class="step-title">Dual Model Scoring</div>
                        <div class="step-desc">Isolation Forest (global tree isolation) + Novelty LOF (local density k=20).</div>
                    </div>
                    <div class="step-card">
                        <div class="step-num">Step 3</div>
                        <div class="step-title">Regime Clustering</div>
                        <div class="step-desc">K-Means partitions operations into 4 distinct physical and biochemical regimes.</div>
                    </div>
                    <div class="step-card">
                        <div class="step-num">Step 4</div>
                        <div class="step-title">Feature Ablation</div>
                        <div class="step-desc">Mathematical sensitivity attribution reveals exactly which sensors caused any flag.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


def page_flagged():
    page_header(
        "Flagged readings",
        "Concordance analysis between Isolation Forest and Local Outlier Factor models.",
        how_to_read_text=(
            "- **Concordance**: Where both models flag an event, confidence is highest.\n"
            "- **Thresholds**: Dotted lines mark the operational decision boundaries (iForest: 0.55, LOF: 1.5).\n"
            "- **Inspection**: Select any reading from the scatter or dropdown to inspect its exact stage telemetry."
        )
    )

    if not st.session_state.get("is_ml_permitted", True):
        status_banner("warn", "Scoring Skipped", ABSTAIN_NOTE)
        zone_eyebrow("EXTRACTED DATA")
        st.dataframe(display_df(st.session_state["active_results"].head(30)), width="stretch")
        return

    df_res = st.session_state["active_results"]
    cat_counts = df_res["concordance"].value_counts()
    n_both = cat_counts.get("Consensus Anomaly", 0)
    n_iforest_only = cat_counts.get("iForest Only", 0)
    n_lof_only = cat_counts.get("LOF Only", 0)
    union_flags = n_both + n_iforest_only + n_lof_only
    jaccard = (n_both / union_flags) if union_flags > 0 else 1.0

    # ZONE: KEY NUMBERS
    with st.container(border=True):
        zone_eyebrow("KEY NUMBERS")
        kpi_grid([
            kpi_card("Flagged by both", f"{n_both}", f"{n_both/len(df_res)*100:.1f}% of readings", "red", "Consensus anomaly flags"),
            kpi_card("Isolation Forest only", f"{n_iforest_only}", "Global extreme outliers", "amber", "Flagged by Isolation Forest, normal under LOF"),
            kpi_card("LOF only", f"{n_lof_only}", "Local density outliers", "amber", "Flagged by LOF, normal under Isolation Forest"),
            kpi_card("Agreement", f"{jaccard*100:.1f}%", "Jaccard similarity index", "plain", "Concordance ratio between both detector engines"),
        ])

    # ZONE: ANALYSIS
    with st.container(border=True):
        zone_eyebrow("ANALYSIS")
        col_scat, col_desc = st.columns([8, 4])

        with col_scat:
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
            fig_scatter.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0)))
            fig_scatter.add_vline(x=0.55, line_width=1, line_dash="dash", line_color=C_ACCENT,
                                  annotation_text="iForest boundary (0.55)", annotation_font_size=11,
                                  annotation_position="top right")
            fig_scatter.add_hline(y=1.5, line_width=1, line_dash="dash", line_color=C_ACCENT,
                                  annotation_text="LOF boundary (1.50)", annotation_font_size=11,
                                  annotation_position="bottom right")
            apply_plot_theme(fig_scatter, 440)

            clicked_pos = None
            try:
                event = st.plotly_chart(
                    fig_scatter, width="stretch", key="concordance_scatter",
                    on_select="rerun", selection_mode="points", config={"displaylogo": False}
                )
                pts = event["selection"]["points"] if event and event.get("selection") else []
                if pts:
                    clicked_pos = int(pts[0]["customdata"][0])
            except Exception:
                st.plotly_chart(fig_scatter, width="stretch", config={"displaylogo": False})

        with col_desc:
            st.markdown(
                f"""
                <div class="rail-card">
                    <div class="rail-title"><span class="chip-dot"></span>Boundary Interpretation</div>
                    <div class="key-list">
                        <div class="key-item">
                            <span class="dot bad"></span>
                            <span><b>Quadrant 4 (Upper Right)</b>: Exceeds both iForest (0.55) and LOF (1.50). Severe plant events.</span>
                        </div>
                        <div class="key-item">
                            <span class="dot warn"></span>
                            <span><b>Quadrant 2 (Bottom Right)</b>: High iForest score alone. Volumetric storm surges or global load shocks.</span>
                        </div>
                        <div class="key-item">
                            <span class="dot warn"></span>
                            <span><b>Quadrant 3 (Upper Left)</b>: High LOF score alone. Localized parameter deviations far from 20 nearest neighbours.</span>
                        </div>
                        <div class="key-item">
                            <span class="dot ok"></span>
                            <span><b>Quadrant 1 (Lower Left)</b>: Baseline normal operations. Neither model triggers an alert.</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # ZONE: DETAILS
    with st.container(border=True):
        zone_eyebrow("DETAILS")
        flagged_dates = list(df_res.loc[df_res["concordance"] != "Normal Baseline", "Date"])
        picked = st.selectbox(
            "Pick a flagged reading to inspect details",
            options=flagged_dates, index=None, placeholder="Choose a date (e.g. D-13/3/90)",
            key="inspect_date",
        )
        if picked is not None:
            matches = np.where(df_res["Date"].values == picked)[0]
            if len(matches):
                clicked_pos = int(matches[0])

        if clicked_pos is not None:
            inspect_reading(df_res, clicked_pos)
            st.markdown("---")

        st.markdown("##### All flagged readings table")
        if union_flags == 0:
            st.success("No unusual readings found in this data.")
        else:
            st.dataframe(flagged_table(df_res), hide_index=True, width="stretch")

        with st.expander("Why do the models sometimes disagree?"):
            col_agree, col_disagree = st.columns(2)
            with col_agree:
                st.markdown(
                    "**Flagged by both (for example D-13/3/90)**\n\n"
                    "- A severe breakdown that is extreme overall and also unlike similar days.\n"
                    "- Example: a clarifier failure pushed settleable solids out (SED-S) to 3.5 mL/L, "
                    "against a historical median of 0.02 mL/L.\n"
                    "- Both models flag these independently, so confidence is highest."
                )
            with col_disagree:
                st.markdown(
                    "**Only one model flags it (for example D-12/8/90 vs D-7/2/91)**\n\n"
                    "- **Isolation Forest only (D-12/8/90)**: global extremes such as a storm surge "
                    "(Q-E above 50,000 m3/day).\n"
                    "- **LOF only (D-7/2/91)**: a reading far from its 20 nearest similar days, such as a zinc spike "
                    "(ZN-E = 10.0 mg/L), even if it is not extreme overall."
                )


def page_patterns():
    page_header(
        "Patterns",
        "Operational regime clustering (K=4) projected onto the 2D principal component subspace.",
        how_to_read_text=(
            "- **PCA Projection**: Compresses all 38 continuous sensor streams into two principal axes preserving 81.8% of variance.\n"
            "- **Regimes (Colors)**: Colors distinguish the 4 K-Means clusters representing distinct biological clarification states.\n"
            "- **Status (Symbols)**: Diamonds indicate consensus anomalies; circles represent baseline operations."
        )
    )

    if not st.session_state.get("is_ml_permitted", True):
        status_banner("warn", "Scoring Skipped", ABSTAIN_NOTE)
        zone_eyebrow("EXTRACTED DATA")
        st.dataframe(display_df(st.session_state["active_results"].head(30)), width="stretch")
        return

    df_res = with_status(st.session_state["active_results"])

    # ZONE: ANALYSIS
    with st.container(border=True):
        zone_eyebrow("ANALYSIS")
        col_pca, col_summary = st.columns([8, 4])

        with col_pca:
            hover_cols = [c for c in ["Date", "status", "SED-S", "Q-E", "ZN-E"] if c in df_res.columns]

            # Single clean legend per regime via legendgroup
            fig_pca = go.Figure()

            # Symbol mapping by status
            symbol_map = {
                "Flagged by both models": "diamond",
                "Isolation Forest only": "triangle-up",
                "LOF only": "triangle-down",
                "Normal": "circle",
            }

            for c_id in sorted(REGIME_METADATA.keys()):
                regime_name = REGIME_METADATA[c_id]["name"]
                color = REGIME_METADATA[c_id]["color"]
                sub_c = df_res[df_res["cluster"] == c_id]

                first_trace = True
                for stat, sym in symbol_map.items():
                    sub_stat = sub_c[sub_c["status"] == stat]
                    if not sub_stat.empty:
                        fig_pca.add_trace(go.Scatter(
                            x=sub_stat["PC1"],
                            y=sub_stat["PC2"],
                            mode="markers",
                            name=REGIME_METADATA[c_id]["short"],
                            legendgroup=f"regime_{c_id}",
                            showlegend=first_trace,
                            marker=dict(
                                color=color,
                                symbol=sym,
                                size=9 if sym == "diamond" else 8,
                                opacity=0.85,
                                line=dict(color="#FFFFFF", width=1 if sym == "diamond" else 0)
                            ),
                            customdata=sub_stat[["Date", "status", "regime_short"]].values,
                            hovertemplate="<b>%{customdata[0]}</b><br>Regime: %{customdata[2]}<br>Status: %{customdata[1]}<br>PC1: %{x:.2f}, PC2: %{y:.2f}<extra></extra>"
                        ))
                        first_trace = False

            apply_plot_theme(fig_pca, 480)
            fig_pca.update_xaxes(title="Axis 1 (77.2% of variation)")
            fig_pca.update_yaxes(title="Axis 2 (4.6% of variation)")
            st.plotly_chart(fig_pca, width="stretch", config={"displaylogo": False})
            st.caption("Marker symbols: ◆ Flagged by both models | ▲ iForest only | ▼ LOF only | ● Normal baseline")

        with col_summary:
            st.markdown(
                """
                <div class="rail-card">
                    <div class="rail-title"><span class="chip-dot"></span>What the two axes mean</div>
                    <div class="key-list">
                        <div class="key-item"><span><b>Axis 1 (77.2%)</b>: Governed by clarification efficiency and effluent solids (SED-S, RD-SED-G). High values indicate physical clarification failure.</span></div>
                        <div class="key-item"><span><b>Axis 2 (4.6%)</b>: Governed by raw influent solids loading (SS-E, SS-P). Spikes represent volumetric storm shocks.</span></div>
                        <div class="key-item"><span><b>Variance Retained</b>: Combined 2D subspace captures <b>81.83%</b> of total plant variability without distortion.</span></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # ZONE: DETAILS
    with st.container(border=True):
        zone_eyebrow("DAYS IN EACH OPERATIONAL REGIME")
        cluster_counts = df_res["cluster"].value_counts().to_dict()
        cols = st.columns(4)
        for i, col in enumerate(cols):
            meta = REGIME_METADATA[i]
            count = int(cluster_counts.get(i, 0))
            pct = count / len(df_res) * 100
            with col:
                st.markdown(
                    f'<div class="kpi-card" style="border-top: 3px solid {meta["color"]}; min-height: 120px;">'
                    f'<div class="kpi-label">{meta["short"]}</div>'
                    f'<div class="kpi-value" style="font-size: 1.5rem; color: {meta["color"]};">{count} days</div>'
                    f'<div class="kpi-sub" style="margin-bottom: 6px;">{pct:.1f}% of historical records</div>'
                    f'<div style="font-size: 0.82rem; color: {C_TEXT_MUTED}; line-height: 1.4;">{meta["description"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )


def page_why_flagged():
    page_header(
        "Why flagged",
        "Mathematical feature ablation: measures score drop when resetting each sensor to historical median.",
        how_to_read_text=(
            "- **Feature Ablation**: For each sensor stream, the reading is replaced with its historical plant median.\n"
            "- **Score Reduction**: A larger score drop means the model was more sensitive to that specific sensor.\n"
            "- **Color Coding**: Bars are colored by treatment stage (Influent, Primary, Secondary, Effluent, Efficiency)."
        )
    )

    if not st.session_state.get("is_ml_permitted", True):
        status_banner("warn", "Scoring Skipped", ABSTAIN_NOTE)
        zone_eyebrow("EXTRACTED DATA")
        st.dataframe(display_df(st.session_state["active_results"].head(30)), width="stretch")
        return

    _, _, _, _, _, explainer = load_all_models()
    df_res = st.session_state["active_results"]
    scaled_df = st.session_state["active_scaled"]
    raw_df = st.session_state["active_raw"]
    anomalies_only = df_res[df_res["concordance"] != "Normal Baseline"]

    # ZONE: CONTROLS
    with st.container(border=True):
        zone_eyebrow("CONTROLS")
        col_date, col_model, col_topn = st.columns([6, 3.5, 2.5], vertical_alignment="center")

        with col_date:
            date_options = list(anomalies_only["Date"]) if not anomalies_only.empty else list(df_res["Date"])
            if st.session_state.get("explain_date_sel") not in date_options:
                st.session_state.pop("explain_date_sel", None)

            def format_date_option(d):
                row = df_res[df_res["Date"] == d].iloc[0]
                return f"{d} | {STATUS_LABELS.get(row['concordance'], row['concordance'])} | iForest: {row['iforest_score']:.3f}, LOF: {row['lof_score']:.2f}"

            selected_date = st.selectbox(
                "Reading to explain",
                options=date_options,
                format_func=format_date_option,
                key="explain_date_sel",
            )

        with col_model:
            model_choice = st.radio(
                "Detection model",
                options=["iforest", "lof"],
                format_func=lambda m: "Isolation Forest" if m == "iforest" else "Local Outlier Factor (LOF)",
                horizontal=True
            )

        with col_topn:
            top_n = st.slider("Sensors to display", min_value=3, max_value=8, value=5)

    if selected_date:
        idx = df_res[df_res["Date"] == selected_date].index[0]
        sample_scaled = scaled_df.loc[idx:idx]
        sample_raw = raw_df.loc[idx] if idx < len(raw_df) else None

        with st.spinner("Computing mathematical sensor sensitivity..."):
            top_df, metadata = explainer.explain_sample(
                sample=sample_scaled,
                raw_row=sample_raw,
                sample_id=selected_date,
                model_type=model_choice,
                top_n=top_n
            )

        # ZONE: ANALYSIS (Bar chart 8/12 + Method note 4/12)
        with st.container(border=True):
            zone_eyebrow("ANALYSIS")
            col_bar, col_card = st.columns([8, 4])

            with col_bar:
                calc_height = int(56 * top_n + 140)
                max_drop = float(top_df["score_reduction"].max())
                fig_bar = px.bar(
                    top_df,
                    x="score_reduction",
                    y="sensor_name",
                    orientation="h",
                    color="stage",
                    text="score_reduction",
                    color_discrete_map=STAGE_COLOR_MAP,
                    labels={
                        "score_reduction": "Score drop when set to normal",
                        "sensor_name": "Sensor",
                        "stage": "Plant stage"
                    },
                )
                fig_bar.update_traces(
                    texttemplate="%{text:.4f}",
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate="<b>%{y}</b><br>Stage: %{customdata[0]}<br>Score drop: %{x:.6f}<br>Raw: %{customdata[1]} %{customdata[2]}<extra></extra>",
                    customdata=top_df[["stage", "raw_value", "unit"]].values
                )
                apply_plot_theme(fig_bar, calc_height)
                fig_bar.update_layout(
                    yaxis=dict(autorange="reversed"),
                    xaxis=dict(range=[0, max_drop * 1.18], title="Score drop when reset to plant median")
                )
                st.plotly_chart(fig_bar, width="stretch", config={"displaylogo": False})

            with col_card:
                st.markdown(
                    f"""
                    <div class="rail-card">
                        <div class="rail-title"><span class="chip-dot"></span>Attribution Summary</div>
                        <div style="font-size: 0.875rem; line-height: 1.55; color: {C_TEXT};">
                            Setting <b>{metadata['top_feature']}</b> (<code>{metadata['top_feature_code']}</code>) to its historical median
                            reduces the <b>{metadata['model_name']}</b> anomaly score by <b>{metadata['top_score_reduction']:.6f}</b>.
                            <br><br>
                            This indicates that this sensor parameter drove the highest mathematical deviation for reading <b>{selected_date}</b>.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # ZONE: DETAILS
        with st.container(border=True):
            zone_eyebrow("SENSOR ATTRIBUTION AUDIT")
            display_attribution = top_df[[
                "feature_code", "sensor_name", "stage", "raw_value", "unit",
                "scaled_value", "original_score", "ablated_score", "score_reduction"
            ]].copy()
            display_attribution = display_attribution.rename(columns={
                "feature_code": "Code", "sensor_name": "Sensor", "stage": "Stage",
                "raw_value": "Reading", "unit": "Unit", "scaled_value": "Deviation (IQR)",
                "original_score": "Score", "ablated_score": "Score if median",
                "score_reduction": "Score reduction",
            })
            st.dataframe(
                display_attribution.style.format({
                    "Reading": "{:.2f}",
                    "Deviation (IQR)": "{:+.3f}",
                    "Score": "{:.4f}",
                    "Score if median": "{:.4f}",
                    "Score reduction": "{:.6f}"
                }),
                hide_index=True,
                width="stretch"
            )


def page_upload():
    page_header(
        "Upload data",
        "Multi-format ingestion pipeline for CSV, Excel, and PDF telemetry logs.",
        how_to_read_text=(
            "- **38 Sensor Contract**: All 38 plant sensors must be present or mapped before ML models will score.\n"
            "- **Strict Abstention**: Incomplete reports are parsed and displayed, but ML inference is safely skipped.\n"
            "- **Audit Tables**: Inspect detected unit conversions, stage mappings, and unrecognised fields."
        )
    )

    preprocessor, iforest, lof, kmeans, pca, _ = load_all_models()
    ingestion_engine = MultiFormatIngestionEngine()

    # ZONE: UPLOAD
    with st.container(border=True):
        zone_eyebrow("DATA INGESTION")
        uploaded_file = st.file_uploader(
            "Upload water-treatment data file (CSV, Excel, PDF)",
            type=["csv", "xlsx", "xls", "pdf"],
            help="Supports CSV tables, Excel spreadsheets (horizontal and vertical), and digital PDF tables or certificates."
        )

        if uploaded_file is not None:
            outcome = None
            extracted = None
            report = None
            results_up = None
            scaled_up = None
            upload_bytes = uploaded_file.getvalue()
            upload_key = hashlib.sha256(upload_bytes).hexdigest()

            # Streamlit reruns the page after every widget interaction. Process a file
            # only when it changes so confirmed ambiguous-column mappings are retained.
            if st.session_state.get("upload_key") != upload_key:
                try:
                    with st.status("Reading file and matching sensors...", expanded=False) as status:
                        extracted = ingestion_engine.ingest_file(io.BytesIO(upload_bytes), uploaded_file.name)
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
                    outcome = "exception"
                    st.error(f"Error processing file: {str(e)}")

                if outcome == "error":
                    st.error(report.error_message)
                    if report.file_type == "scanned_pdf":
                        st.info(
                            "This appears to be a scanned raster PDF. Optical character recognition (OCR) is not supported "
                            "in this container environment. Export the document as a text-based PDF, CSV, or Excel file."
                        )
                elif outcome in ("complete", "incomplete"):
                    st.session_state["upload_key"] = upload_key
                    st.session_state["upload_filename"] = uploaded_file.name
                    st.session_state["active_report"] = report
                    st.session_state["extracted_raw"] = extracted.raw_df
                    st.session_state["extracted_payload"] = extracted

                    if outcome == "complete":
                        st.success(
                            f"All 38 sensors matched ({report.coverage_pct}% coverage) from {report.file_type.upper()} "
                            f"across {report.total_rows} reading(s). Models executed successfully."
                        )
                        st.session_state["active_results"] = results_up
                        st.session_state["active_scaled"] = scaled_up
                        # Attribution requires canonical, unit-normalized readings.
                        st.session_state["active_raw"] = extracted.standardized_df
                        st.session_state["active_name"] = f"Uploaded File: {uploaded_file.name} ({report.total_rows} rows)"
                        st.session_state["data_source"] = "uploaded"
                        st.session_state["is_ml_permitted"] = True
                    else:
                        st.warning(
                            f"Found {report.detected_canonical_count} of 38 sensors ({report.coverage_pct}% coverage)."
                        )
                        status_banner("warn", "Scoring Skipped",
                                      "The models require all 38 continuous sensor streams covering the full treatment cycle. "
                                      "Synthesizing missing values would bias anomaly scoring. Review missing sensors below.")
                        st.session_state["active_results"] = extracted.standardized_df if extracted.standardized_df is not None else extracted.raw_df
                        st.session_state["active_raw"] = extracted.standardized_df if extracted.standardized_df is not None else extracted.raw_df
                        st.session_state["active_name"] = f"Uploaded (Incomplete): {uploaded_file.name} ({report.detected_canonical_count}/38 features)"
                        st.session_state["data_source"] = "uploaded_incomplete"
                        st.session_state["is_ml_permitted"] = False

    st.markdown("---")

    # ZONE: DATA AUDIT & REPORT
    with st.container(border=True):
        zone_eyebrow("DATA INSPECTION")
        df_active = st.session_state["active_results"]
        report_active: Optional[IngestionReport] = st.session_state.get("active_report", None)

        st.markdown(f"##### Data in use: {st.session_state['active_name']}")

        tab_preview, tab_audit, tab_stages, tab_download = st.tabs([
            "Data preview", "Quality report", "By plant stage", "Download"
        ])

        with tab_preview:
            st.dataframe(display_df(df_active.head(50)), width="stretch")

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

                if report_active.missing_features:
                    st.markdown("##### Missing sensors by plant stage")
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
                            missing_details.append(f"{code} ({meta['name']}, {meta['unit']})")
                        st.markdown(f"- **{stage_title}** ({len(missing_list)} missing): {', '.join(missing_details)}")

                if report_active.ambiguous_features:
                    st.markdown("##### Columns requiring manual stage confirmation")
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
                        submitted = st.form_submit_button("Confirm mappings and score")
                        if submitted and resolutions:
                            cur_payload = st.session_state.get("extracted_payload", None)
                            if cur_payload is not None:
                                updated_extracted = ingestion_engine.apply_user_mappings(
                                    cur_payload,
                                    resolutions,
                                    filename=st.session_state.get("upload_filename", "uploaded_file")
                                )
                                st.session_state["extracted_payload"] = updated_extracted
                                st.session_state["active_report"] = updated_extracted.report
                                if updated_extracted.report.is_complete_38:
                                    st.success("All 38 sensors confirmed. Running models...")
                                    res_ml, sc_ml, _ = run_mizu_inference(
                                        updated_extracted.standardized_df, preprocessor, iforest, lof, kmeans, pca
                                    )
                                    st.session_state["active_results"] = res_ml
                                    st.session_state["active_scaled"] = sc_ml
                                    st.session_state["active_raw"] = updated_extracted.standardized_df
                                    st.session_state["data_source"] = "uploaded"
                                    st.session_state["is_ml_permitted"] = True
                                else:
                                    st.session_state["active_results"] = updated_extracted.standardized_df
                                    st.session_state["active_raw"] = updated_extracted.standardized_df
                                    st.session_state["data_source"] = "uploaded_incomplete"
                                    st.session_state["is_ml_permitted"] = False
                                st.rerun()

                if report_active.unrecognised_features:
                    st.markdown("##### Columns ignored (Non-sensor parameters)")
                    st.info(f"Ignored: {', '.join([f'{u}' for u in report_active.unrecognised_features])}")

                if report_active.unit_warnings:
                    st.markdown("##### Unit conversions applied")
                    for w in report_active.unit_warnings:
                        st.caption(f"- {w}")

                if report_active.mapped_features:
                    st.markdown("##### Matched sensor audit table")
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
                st.caption(
                    "Gaps in the baseline dataset are filled via pre-fitted median imputation. "
                    "Biochemical oxygen demand (BOD) has higher missingness due to standard 5-day incubation test duration."
                )
                missing_summary = get_missing_summary(st.session_state["active_raw"])
                st.dataframe(missing_summary.head(15), width="stretch")

        with tab_stages:
            stages = get_feature_groups()
            selected_stage = st.selectbox(
                "Filter by treatment stage:",
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
            st.dataframe(display_df(df_active[stage_cols].head(30)), width="stretch")

        with tab_download:
            st.caption("Export standardized dataset (mapped canonical sensor codes and normalized units).")
            csv_bytes = df_active.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download standardized CSV",
                data=csv_bytes,
                file_name="mizu_standardized_telemetry.csv",
                mime="text/csv",
                width="stretch"
            )


def _page_about_legacy():
    page_header(
        "About MIZU",
        "Architecture, algorithmic methodology, regulatory boundaries, and engineering limits.",
        how_to_read_text=(
            "- **Decision-support**: MIZU is a secondary operational screening tool, not an automatic controller.\n"
            "- **Limits**: Does not certify drinking-water potability or test for biological pathogens.\n"
            "- **Methodology**: Combines tree-based global isolation, k-NN local density, K-Means, and PCA."
        )
    )

    # 4-Step Infographic Flow Card
    zone_eyebrow("SYSTEM ARCHITECTURE")
    st.markdown(
        """
        <div class="step-flow">
            <div class="step-card">
                <div class="step-num">Step 1: Ingestion</div>
                <div class="step-title">38-Sensor Validation</div>
                <div class="step-desc">Extracts multi-format CSV/XLSX/PDF, converts engineering units, and validates all 4 stages.</div>
            </div>
            <div class="step-card">
                <div class="step-num">Step 2: Scoring</div>
                <div class="step-title">Dual Unsupervised ML</div>
                <div class="step-desc">Isolation Forest (contamination 3.0%) and Novelty LOF (k=20) evaluate multidimensional deviance.</div>
            </div>
            <div class="step-card">
                <div class="step-num">Step 3: Grouping</div>
                <div class="step-title">Regime Clustering</div>
                <div class="step-desc">K-Means partitions operations into 4 operational regimes mapped to a 2D PCA subspace.</div>
            </div>
            <div class="step-card">
                <div class="step-num">Step 4: Attribution</div>
                <div class="step-title">Feature Ablation</div>
                <div class="step-desc">Systematic ablation quantifies individual sensor contribution to anomaly scores.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    tab_how, tab_why, tab_limits, tab_terms = st.tabs([
        "How it works", "Design choices", "Limits", "Glossary"
    ])

    with tab_how:
        st.markdown("##### End-to-end Processing Pipeline")
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
            "It does not detect Cryptosporidium, Giardia, Legionella or Norovirus.\n"
            "3. **Human-in-the-loop.** Every flag needs a physical sample and lab confirmation by a certified operator "
            "before any physical process change."
        )
        st.markdown(
            f"""
            <div class="notice">
                <b>Regulatory and engineering notice:</b> MIZU is an unsupervised decision-support tool for secondary operational screening.
                It does not trigger automated physical action in the treatment plant. Scores show mathematical deviation from baseline
                operating conditions and must be verified by accredited laboratory tests.
            </div>
            """,
            unsafe_allow_html=True
        )

    with tab_terms:
        st.markdown(
            "- **Anomaly**: a reading that deviates significantly from baseline plant behaviour.\n"
            "- **Isolation Forest**: finds readings that require few partitions to isolate (extreme overall).\n"
            "- **LOF (Local Outlier Factor)**: finds readings with substantially lower density than their k=20 nearest neighbours.\n"
            "- **Flagged by both models**: consensus anomaly detection; highest confidence alert signal.\n"
            "- **Regime**: typical operating states identified via unsupervised K-Means clustering.\n"
            "- **PCA**: 2D dimensionality reduction capturing 81.83% of multidimensional variance.\n"
            "- **Feature ablation**: single-feature median substitution to quantify sensitivity drop.\n"
            "- **IQR**: interquartile range (75th minus 25th percentile); robust against heavy tails."
        )


def page_about():
    """Short, plain-language description of the product and its limits."""
    page_header(
        "About MIZU",
        "A decision-support dashboard for reviewing wastewater-treatment telemetry.",
        how_to_read_text="Use MIZU to review unusual operating patterns; verify any decision with plant records and laboratory testing.",
    )

    with st.container(border=True):
        zone_eyebrow("WHAT MIZU DOES")
        st.markdown(
            "MIZU checks 38 water-treatment measurements together and highlights readings that differ from normal plant behaviour. "
            "It combines two anomaly checks, groups similar operating conditions, and shows which measurements contributed to a flag."
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("##### 1. Upload or use baseline data")
            st.caption("Review CSV, Excel, or compatible report data after sensor names and units are checked.")
        with c2:
            st.markdown("##### 2. Review unusual readings")
            st.caption("Compare model flags, operating patterns, and the sensors that most influenced each result.")
        with c3:
            st.markdown("##### 3. Confirm before acting")
            st.caption("Use the dashboard as a screening aid alongside operator review and accredited laboratory checks.")

    with st.container(border=True):
        zone_eyebrow("IMPORTANT LIMITS")
        st.markdown(
            "- MIZU supports wastewater operations; it does not certify drinking-water safety.\n"
            "- It does not detect biological pathogens.\n"
            "- A flag indicates an unusual measurement pattern, not a confirmed process failure."
        )


def page_documentation():
    """Operational reference for the actual MIZU ingestion and scoring workflow."""
    page_header(
        "Documentation",
        "A practical guide to how MIZU checks, explains, and presents treatment-plant telemetry.",
        how_to_read_text="This guide describes the live MIZU workflow. It does not use generated or sample measurements.",
    )

    pdf_path = project_root / "assets" / "mizu-documentation-guide.pdf"
    if pdf_path.exists():
        st.download_button(
            "Download documentation as PDF",
            data=pdf_path.read_bytes(),
            file_name="mizu-documentation-guide.pdf",
            mime="application/pdf",
            icon=":material/picture_as_pdf:",
        )
    else:
        st.caption("The documentation PDF is not available in this deployment.")

    with st.container(border=True):
        zone_eyebrow("HOW MIZU WORKS")
        st.markdown(
            "**1. Ingest and standardise.** MIZU reads CSV, Excel, and supported PDF report formats. It maps recognised sensor names, records any unit conversion, and shows fields it could not identify.\n\n"
            "**2. Check completeness.** The fitted models require all 38 sensor features. If a required field is missing or ambiguous, MIZU shows the extracted measurements and audit report, but deliberately skips scoring.\n\n"
            "**3. Prepare values.** Complete records are processed with the existing fitted median imputer and RobustScaler. Uploading a file does not retrain the models or change their thresholds.\n\n"
            "**4. Detect unusual patterns.** Isolation Forest looks for readings that are unusual across the whole sensor set. Novelty LOF looks for readings that differ from nearby baseline operating conditions.\n\n"
            "**5. Add context and explain.** K-Means assigns an operating regime, PCA provides a two-dimensional exploration view, and feature ablation estimates which sensors most affected a selected anomaly score."
        )

    with st.container(border=True):
        zone_eyebrow("READING RESULTS")
        st.markdown(
            "A reading flagged by both models is the strongest agreement signal. A reading flagged by one model is a weaker signal that still deserves context. "
            "A normal result only means that the measurements resemble the fitted baseline; it is not a regulatory pass/fail decision. Confirm significant events with operator records, samples, and laboratory testing."
        )


def page_settings():
    """Settings page providing operational baseline parameters and restoration control."""
    page_header(
        "Settings",
        "System analysis settings and operational parameters.",
    )
    with st.container(border=True):
        zone_eyebrow("ANALYSIS SETTINGS")
        st.caption("These are the fixed settings used by the fitted baseline models.")
        st.markdown("**Isolation Forest**")
        st.caption("Contamination rate: 3.0% (about 16 baseline events)")
        st.markdown("**Local Outlier Factor**")
        st.caption("Neighbourhood: 20 readings")
        st.markdown("**Operating patterns**")
        st.caption("K-Means: 4 regimes · PCA retained variance: 81.83%")
        if st.session_state.get("data_source", "baseline") != "baseline":
            st.markdown("---")
            if st.button("Restore baseline data", icon=":material/restart_alt:"):
                raw_baseline = load_baseline_raw_data()
                preprocessor, iforest, lof, kmeans, pca, _ = load_all_models()
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


# =============================================================================
# 6. ROUTER ENTRYPOINT & LEFT SIDEBAR NAVIGATION
# =============================================================================

def main():
    st.set_page_config(
        page_title="MIZU | Intelligent Water Systems Monitoring",
        page_icon="assets/mizu_logo.svg",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Inject the light operational UI layer.
    css_path = project_root / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

    # Initialize model artifacts and baseline dataset
    preprocessor, iforest, lof, kmeans, pca, _ = load_all_models()

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

    # -------------------------------------------------------------------------
    # Multipage Page Definitions
    # -------------------------------------------------------------------------
    page_overview_ref = st.Page(
        page_overview, title="Overview", default=True, url_path="overview"
    )
    page_flagged_ref = st.Page(
        page_flagged, title="Flagged Readings", url_path="flagged"
    )
    page_patterns_ref = st.Page(
        page_patterns, title="Patterns", url_path="patterns"
    )
    page_why_flagged_ref = st.Page(
        page_why_flagged, title="Why Flagged", url_path="why-flagged"
    )
    page_upload_ref = st.Page(
        page_upload, title="Upload Data", url_path="upload"
    )
    page_about_ref = st.Page(
        page_about, title="About", url_path="about"
    )
    page_documentation_ref = st.Page(
        page_documentation, title="Documentation", url_path="documentation"
    )
    page_settings_ref = st.Page(
        page_settings, title="Settings", url_path="settings"
    )

    pages = {
        "MONITOR": [
            page_overview_ref,
            page_flagged_ref,
            page_patterns_ref,
            page_why_flagged_ref,
        ],
        "": [
            page_upload_ref,
            page_documentation_ref,
            page_about_ref,
        ],
        "SETTINGS": [
            page_settings_ref,
        ],
    }

    # Left sidebar multipage router
    pg = st.navigation(pages, position="sidebar")

    # -------------------------------------------------------------------------
    # Sidebar Header (Logo + Brand + Tagline)
    # -------------------------------------------------------------------------
    logo_path = project_root / "assets" / "mizu_logo.svg"
    logo_data = base64.b64encode(logo_path.read_bytes()).decode("ascii") if logo_path.exists() else ""

    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-brand">
                <img class="mizu-logo" src="data:image/svg+xml;base64,{logo_data}" alt="MIZU logo" />
                <div class="brand-text-block">
                    <span class="brand-name">MIZU</span>
                    <span class="brand-tagline">Water systems monitoring</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # -------------------------------------------------------------------------
    # Execute Active Page (Isolated Multipage Run)
    # -------------------------------------------------------------------------
    st.markdown('<div class="page-enter">', unsafe_allow_html=True)
    pg.run()
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
