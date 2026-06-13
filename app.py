"""
Customer Retention Analytics Platform
Enterprise churn prediction application — Linear/Stripe/Vercel design system.
"""

import os
from datetime import datetime

import bcrypt
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# Page configuration  (must precede every other Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnIQ — Customer Retention Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# Design system — single CSS block, no inline styles in Python logic
# ─────────────────────────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Reset & Base Styles ─────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: #080C14 !important;
    color: #F9FAFB !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    font-size: 14px;
}

/* Header strip hide or style */
[data-testid="stHeader"] {
    background: #080C14 !important;
    border-bottom: 1px solid #1C243B !important;
}

.block-container {
    padding: 2.5rem 3rem 4rem !important;
    max-width: 1400px !important;
}

/* Remove Streamlit default vertical spacing gaps */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
    gap: 0 !important;
}

/* ── Cards / Borders / Surfaces ─────────────────────────────────────────── */
/* Target Streamlit's bordered container to render our card design system */
div[data-testid="stBorderedContainer"] {
    background-color: #111625 !important;
    border: 1px solid #1C243B !important;
    border-radius: 10px !important;
    padding: 1.5rem !important;
    margin-bottom: 1rem !important;
}

/* ── KPI Card Custom Elements ───────────────────────────────────────────── */
.kpi-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
}
.kpi-title {
    font-size: 0.875rem;
    font-weight: 500;
    color: #94A3B8;
}
.kpi-icon-wrap {
    width: 28px;
    height: 28px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.kpi-icon-wrap svg {
    width: 16px;
    height: 16px;
    stroke: currentColor;
    fill: none;
    stroke-width: 2;
}
.kpi-icon-wrap.blue { background-color: rgba(99, 102, 241, 0.1); color: #818CF8; }
.kpi-icon-wrap.red { background-color: rgba(239, 68, 68, 0.1); color: #F87171; }
.kpi-icon-wrap.yellow { background-color: rgba(245, 158, 11, 0.1); color: #FBBF24; }

.kpi-body {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}
.kpi-val-text {
    font-size: 2rem;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.03em;
    line-height: 1;
}
.kpi-trend-wrap {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 0.8rem;
    font-weight: 500;
}
.kpi-trend-wrap.up { color: #10B981; }
.kpi-trend-wrap.down { color: #EF4444; }

/* ── Sidebar Navigation ──────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #0A0F1D !important;
    border-right: 1px solid #1C243B !important;
}
[data-testid="stSidebarUserContent"] {
    display: flex;
    flex-direction: column;
    height: 100vh;
    padding: 1.5rem 1rem !important;
}

/* Brand styling */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.5rem 0.5rem 1.5rem 0.5rem;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid #1C243B;
}
.sidebar-logo {
    width: 32px;
    height: 32px;
    background-color: #6366F1;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #FFFFFF;
}
.sidebar-logo svg {
    width: 18px;
    height: 18px;
}
.sidebar-brand-name {
    font-size: 1.1rem;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.01em;
}

.sidebar-spacer {
    flex-grow: 1;
}

/* Nav items style */
div[data-testid="stSidebar"] button {
    background-color: transparent !important;
    color: #94A3B8 !important;
    border: none !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 0.65rem 1rem !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    transition: background-color 0.2s, color 0.2s !important;
    margin-bottom: 0.25rem !important;
}
div[data-testid="stSidebar"] button:hover {
    background-color: #1E293B !important;
    color: #FFFFFF !important;
}

/* Active navigation button state */
div[data-testid="stSidebar"] button[data-testid="baseButton-primary"] {
    background-color: #1E293B !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
}
div[data-testid="stSidebar"] button[data-testid="baseButton-secondary"] {
    background-color: transparent !important;
    color: #94A3B8 !important;
}
div[data-testid="stSidebar"] button[data-testid="baseButton-secondary"]:hover {
    background-color: #1E293B !important;
    color: #FFFFFF !important;
}

/* ── Form Inputs Override ───────────────────────────────────────────────── */
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
    background-color: #111625 !important;
    border: 1px solid #1C243B !important;
    color: #F9FAFB !important;
    border-radius: 6px !important;
    padding: 0.6rem 0.8rem !important;
    font-size: 0.9rem !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stNumberInput"] input:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
}

/* Custom Selectbox styling */
div[data-testid="stSelectbox"] > div[data-baseweb="select"] > div {
    background-color: #111625 !important;
    border: 1px solid #1C243B !important;
    color: #F9FAFB !important;
    border-radius: 6px !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] {
    color: #F9FAFB !important;
}

/* Uppercase form headers */
div[data-testid="stWidgetLabel"] label p {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: #94A3B8 !important;
}

/* Primary Button Design (Run Prediction, Sign In) */
button[data-testid="baseButton-primary"]:not(div[data-testid="stSidebar"] button) {
    background-color: #5C60F5 !important;
    border: none !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    padding: 0.65rem 1.25rem !important;
    transition: background-color 0.2s !important;
}
button[data-testid="baseButton-primary"]:not(div[data-testid="stSidebar"] button):hover {
    background-color: #4C50E5 !important;
}

/* ── Typography & Headings ───────────────────────────────────────────────── */
h1, h2, h3 {
    font-family: 'Inter', sans-serif !important;
    color: #F9FAFB !important;
    letter-spacing: -0.02em;
}

.page-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: #F9FAFB;
    letter-spacing: -0.02em;
    margin-bottom: 0.25rem;
}
.page-desc {
    font-size: 0.875rem;
    color: #6B7280;
    margin-bottom: 1.5rem;
}

/* Card titles inside containers */
.card-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #F9FAFB;
    margin-bottom: 1.25rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1C243B;
}

/* ── Header Actions ──────────────────────────────────────────────────────── */
.header-actions {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 12px;
    height: 100%;
}
.icon-btn-wrap {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background-color: #111625;
    border: 1px solid #1C243B;
    color: #94A3B8;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: background-color 0.2s, color 0.2s;
}
.icon-btn-wrap:hover {
    background-color: #1E293B;
    color: #FFFFFF;
}
.avatar-circle {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background-color: #6366F1;
    color: #FFFFFF;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 0.85rem;
    user-select: none;
}

/* ── Progress/Risk Distribution (Dashboard) ─────────────────────────────── */
.risk-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.1rem;
}
.risk-label-group {
    display: flex;
    align-items: center;
    gap: 8px;
}
.risk-indicator-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
}
.risk-tier-name {
    font-size: 0.875rem;
    font-weight: 500;
    color: #F9FAFB;
}
.risk-pct-label {
    font-size: 0.75rem;
    color: #6B7280;
}
.risk-bar-track {
    flex-grow: 1;
    height: 6px;
    background-color: #1C243B;
    border-radius: 3px;
    margin: 0 1.25rem;
    overflow: hidden;
    position: relative;
}
.risk-bar-fill {
    height: 100%;
    border-radius: 3px;
}
.risk-count-value {
    font-size: 0.875rem;
    font-weight: 600;
    color: #FFFFFF;
    min-width: 50px;
    text-align: right;
}

/* ── Prediction Result Custom Badges & Verdict ───────────────────────────── */
.verdict-wrap {
    margin-bottom: 1.25rem;
}
.verdict-outcome {
    font-size: 1.35rem;
    font-weight: 600;
    color: #F9FAFB;
    letter-spacing: -0.01em;
}
.verdict-desc {
    font-size: 0.85rem;
    color: #94A3B8;
    line-height: 1.5;
    margin-top: 0.5rem;
}
.badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 4px 10px;
    border-radius: 4px;
}
.badge-high   { background: rgba(239,68,68,0.12);  color: #F87171; border: 1px solid rgba(239,68,68,0.2); }
.badge-medium { background: rgba(245,158,11,0.12); color: #FBBF24; border: 1px solid rgba(245,158,11,0.2); }
.badge-low    { background: rgba(52,211,153,0.12); color: #34D399; border: 1px solid rgba(52,211,153,0.2); }

/* Contributing Factors Rows */
.factor-group-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.75rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1C243B;
}
.factor-group-label.neg { color: #F87171; }
.factor-group-label.pos { color: #34D399; }
.factor-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.55rem 0;
    border-bottom: 1px solid #161D30;
}
.factor-row:last-child { border-bottom: none; }
.factor-name { font-size: 0.85rem; color: #E2E8F0; font-weight: 500; }
.factor-detail { font-size: 0.75rem; color: #64748B; margin-top: 2px; }
.factor-weight {
    font-size: 0.75rem;
    font-weight: 600;
    white-space: nowrap;
    padding: 2px 8px;
    border-radius: 4px;
}
.factor-weight.neg { color: #F87171; background: rgba(239,68,68,0.1); }
.factor-weight.pos { color: #34D399; background: rgba(52,211,153,0.1); }

/* Placeholder Card inside Column 2 */
.placeholder-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 6rem 2rem;
    text-align: center;
    color: #6B7280;
    height: 100%;
}
.placeholder-icon {
    width: 40px;
    height: 40px;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    color: #6B7280;
}
.placeholder-icon svg {
    width: 20px;
    height: 20px;
    stroke: currentColor;
    fill: none;
    stroke-width: 2;
    stroke-linecap: round;
    stroke-linejoin: round;
}
.placeholder-text {
    font-size: 0.875rem;
    line-height: 1.5;
    max-width: 240px;
}

/* ── Auth Page Card Styles ───────────────────────────────────────────────── */
.auth-page-header {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-bottom: 2rem;
    margin-top: 3rem;
}
.auth-icon-wrap {
    width: 48px;
    height: 48px;
    background: linear-gradient(135deg, #6366F1 0%, #818CF8 100%);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3);
}
.auth-icon-wrap svg {
    width: 24px;
    height: 24px;
    stroke: #FFFFFF;
    stroke-width: 2;
    fill: none;
}
.auth-brand-name {
    font-size: 1.35rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 0.5rem;
}
.auth-brand-tagline {
    font-size: 0.875rem;
    color: #64748B;
    text-align: center;
    line-height: 1.5;
}
.auth-card {
    background-color: #111625;
    border: 1px solid #1C243B;
    border-radius: 12px;
    padding: 2.25rem 2rem;
    width: 100%;
}
.auth-card div[data-testid="stWidgetLabel"] label p {
    text-transform: none !important;
    letter-spacing: normal !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: #FFFFFF !important;
}
.auth-demo-hint {
    background-color: #080C14;
    border: 1px solid #1C243B;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    margin-top: 1.25rem;
    margin-bottom: 1.5rem;
}
.auth-demo-label {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748B;
    margin-bottom: 0.5rem;
}
.auth-demo-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    margin-bottom: 0.25rem;
}
.auth-demo-row:last-child {
    margin-bottom: 0;
}
.auth-demo-key { color: #64748B; }
.auth-demo-val { color: #94A3B8; font-family: monospace; }
.auth-footer {
    text-align: center;
    font-size: 0.8rem;
    color: #475569;
    margin-top: 2rem;
}

/* ── Dataframe Overrides ─────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #1C243B !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* ── Alert Success / Error Overrides ───────────────────────────────────────── */
div[data-testid="stAlert"] {
    border-radius: 8px !important;
    border: 1px solid #1C243B !important;
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_page" not in st.session_state:
    st.session_state.current_page = "Dashboard"


# ─────────────────────────────────────────────────────────────────────────────
# Authentication
# ─────────────────────────────────────────────────────────────────────────────
def _load_users() -> pd.DataFrame:
    for path in [os.path.join("customer_churn_prediction", "user.csv"), "user.csv", "users.csv"]:
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception:
                pass
    return pd.DataFrame(columns=["username", "password"])


def authenticate(username: str, password: str) -> bool:
    users = _load_users()
    row = users[users["username"].str.lower() == username.lower()]
    if row.empty:
        return False
    return bcrypt.checkpw(password.encode(), row.iloc[0]["password"].encode())


# ─────────────────────────────────────────────────────────────────────────────
# Resource loading  (only after authentication)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_ml_resources():
    model = pickle.load(open("churn_model.pkl", "rb"))
    label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
    feature_columns = list(pickle.load(open("feature_columns.pkl", "rb")))
    return model, label_encoders, feature_columns


@st.cache_resource
def load_model_artifacts():
    metrics = pickle.load(open("model_metrics.pkl", "rb"))
    cm = pickle.load(open("confusion_matrix.pkl", "rb"))
    feature_importance = pickle.load(open("feature_importance.pkl", "rb"))
    model_obj = pickle.load(open("churn_model.pkl", "rb"))
    return metrics, cm, feature_importance, model_obj


@st.cache_data
def load_dataset():
    path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Plotly chart theme
# ─────────────────────────────────────────────────────────────────────────────
_CHART = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#9CA3AF",
    title_font_color="#F9FAFB",
    title_font_size=13,
    title_font_family="Inter",
    legend_font_color="#6B7280",
    margin=dict(l=16, r=16, t=44, b=16),
)
_AXIS = dict(
    gridcolor="#1F2937",
    linecolor="#1F2937",
    tickcolor="#1F2937",
    title_font_color="#6B7280",
    tickfont_color="#6B7280",
    zeroline=False,
)


def _theme(fig, height: int = 300) -> go.Figure:
    fig.update_layout(**_CHART, height=height)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Reusable primitives
# ─────────────────────────────────────────────────────────────────────────────
def kpi(label: str, value: str, sub: str = "") -> None:
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def section(text: str) -> None:
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def page_heading(title: str, desc: str) -> None:
    st.markdown(
        f'<div class="page-title">{title}</div>'
        f'<div class="page-desc">{desc}</div>',
        unsafe_allow_html=True,
    )


def form_group(text: str) -> None:
    st.markdown(f'<div class="form-group-label">{text}</div>', unsafe_allow_html=True)


def get_opts(label_encoders: dict, field: str, default: list) -> list:
    return list(label_encoders[field].classes_) if field in label_encoders else default


# ─────────────────────────────────────────────────────────────────────────────
# Login page — centered auth card, no decorative panels, no fake metrics
# ─────────────────────────────────────────────────────────────────────────────
def render_login() -> None:
    # ── Page-level vertical centering nudge
    st.markdown(
        "<style>[data-testid='stMain'] > div:first-child { padding-top: 3rem !important; }</style>",
        unsafe_allow_html=True,
    )

    # Centre using three columns
    _, center, _ = st.columns([1, 1.1, 1])

    with center:
        # ── Header: icon + brand name + tagline (above the card)
        st.markdown(
            """
            <div class="auth-page-header">
              <div class="auth-icon-wrap">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                </svg>
              </div>
              <div class="auth-brand-name">ChurnGuard AI</div>
              <div class="auth-brand-tagline">Predict customer churn and identify retention<br>opportunities.</div>
            </div>
            <div class="auth-card">
            """,
            unsafe_allow_html=True,
        )

        # Streamlit inputs — inside the card visually, prefilled to match the screenshot
        username = st.text_input("Username", value="admin", placeholder="admin", key="li_user")
        password = st.text_input("Password", type="password", value="Demo@123", placeholder="••••••••", key="li_pass")

        # Demo access box
        st.markdown(
            """
            <div class="auth-demo-hint">
              <div class="auth-demo-label">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                     stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                  <circle cx="12" cy="12" r="10"/>
                  <line x1="12" y1="12" x2="12" y2="16"/>
                  <line x1="12" y1="8" x2="12.01" y2="8"/>
                </svg>
                DEMO ACCESS
              </div>
              <div class="auth-demo-row">
                <span class="auth-demo-key">Username</span>
                <span class="auth-demo-val">demo_admin</span>
              </div>
              <div class="auth-demo-row">
                <span class="auth-demo-key">Password</span>
                <span class="auth-demo-val">Demo@123</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Sign In", type="primary", use_container_width=True):
            if authenticate(username, password):
                st.session_state.logged_in = True
                st.session_state.current_page = "Dashboard"
                st.rerun()
            else:
                st.error("Incorrect username or password.")

        # Close auth-card div
        st.markdown("</div>", unsafe_allow_html=True)

        # Footer
        st.markdown(
            '<div class="auth-footer">Enterprise Telecom Analytics Platform</div>',
            unsafe_allow_html=True,
        )

    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-logo">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                    </svg>
                </div>
                <span class="sidebar-brand-name">ChurnGuard AI</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        current = st.session_state.current_page

        # Navigation items to match screenshot
        nav_items = [
            ("Dashboard", "Dashboard", ":material/dashboard:"),
            ("Customer Prediction", "Prediction", ":material/person:"),
            ("Prediction History", "History", ":material/history:"),
            ("Model Insights", "Model Insights", ":material/insights:"),
            ("Settings", "Settings", ":material/settings:"),
        ]

        for label, key, icon in nav_items:
            kind = "primary" if current == key else "secondary"
            if st.button(label, icon=icon, type=kind, key=f"nav_{key}", use_container_width=True):
                st.session_state.current_page = key
                st.rerun()

        # Push Logout button to bottom
        st.markdown('<div class="sidebar-spacer"></div>', unsafe_allow_html=True)

        if st.button("Sign Out", icon=":material/logout:", type="secondary", key="nav_logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.current_page = "Dashboard"
            st.rerun()

    return st.session_state.current_page


# ─────────────────────────────────────────────────────────────────────────────
# Risk config
# ─────────────────────────────────────────────────────────────────────────────
def _risk(pct: float) -> dict:
    if pct < 30:
        return {
            "label": "Low Risk", "class": "badge-low", "color": "#34D399",
            "desc": "Customer shows stable engagement. No immediate retention action required.",
        }
    if pct < 60:
        return {
            "label": "Medium Risk", "class": "badge-medium", "color": "#FBBF24",
            "desc": "Moderate churn signals detected. Consider a proactive retention offer.",
        }
    return {
        "label": "High Risk", "class": "badge-high", "color": "#F87171",
        "desc": "High churn probability. Immediate customer success outreach recommended.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Contributing factors
# ─────────────────────────────────────────────────────────────────────────────
def _factors(contract, tenure, tech_support, online_security, internet_service, payment_method):
    neg, pos = [], []

    if contract == "Month-to-month":
        neg.append(("Month-to-Month Contract", "No long-term commitment.", "+25%"))
    elif contract == "Two year":
        pos.append(("Two-Year Contract", "Long-term commitment signals loyalty.", "-20%"))

    if tenure < 6:
        neg.append(("Short Tenure (<6 mo)", "Higher churn during onboarding.", "+20%"))
    elif tenure > 48:
        pos.append(("Long-Term Customer", "Extended tenure correlates with retention.", "-15%"))

    if tech_support == "No":
        neg.append(("No Tech Support", "Unresolved issues raise dissatisfaction.", "+15%"))
    elif tech_support == "Yes":
        pos.append(("Tech Support Active", "Supported customers are more satisfied.", "-10%"))

    if online_security == "No":
        neg.append(("No Online Security", "Missing add-ons reduce platform stickiness.", "+10%"))

    if internet_service == "Fiber optic":
        neg.append(("Fiber Optic Plan", "Higher bill increases price sensitivity.", "+12%"))

    if payment_method == "Electronic check":
        neg.append(("Electronic Check", "Manual payment correlates with attrition.", "+8%"))
    elif "automatic" in payment_method.lower():
        pos.append(("Automatic Payment", "Frictionless billing reduces involuntary churn.", "-10%"))

    return neg, pos


# ─────────────────────────────────────────────────────────────────────────────
# Page A — Dashboard
# ─────────────────────────────────────────────────────────────────────────────
def render_dashboard(df) -> None:
    # ── Header: Title, subtitle, and action widgets on the right ─────────────
    h_col1, h_col2, h_col3, h_col4 = st.columns([2.2, 1, 0.7, 0.9])
    with h_col1:
        st.markdown(
            '<div class="page-title">Dashboard</div>'
            '<div class="page-desc">Telecom churn analytics overview</div>',
            unsafe_allow_html=True
        )
    with h_col2:
        st.selectbox("Date Range", ["Jun 1 – Jun 11, 2026"], label_visibility="collapsed")
    with h_col3:
        st.button("Export", icon=":material/file_upload:", use_container_width=True)
    with h_col4:
        st.markdown(
            """
            <div class="header-actions">
                <div class="icon-btn-wrap">
                    <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path>
                        <path d="M13.73 21a2 2 0 0 1-3.46 0"></path>
                    </svg>
                </div>
                <div class="avatar-circle">JD</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ── KPIs (4 Card Columns) ────────────────────────────────────────────────
    k_col1, k_col2, k_col3, k_col4 = st.columns(4)
    with k_col1:
        with st.container(border=True):
            st.markdown(
                """
                <div class="kpi-header">
                    <span class="kpi-title">Total Customers</span>
                    <div class="kpi-icon-wrap blue">
                        <svg viewBox="0 0 24 24"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
                    </div>
                </div>
                <div class="kpi-body">
                    <span class="kpi-val-text">19,542</span>
                    <div class="kpi-trend-wrap up">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="7" y1="17" x2="17" y2="7"/><polyline points="7 7 17 7 17 17"/></svg>
                        <span>+2.3% vs last month</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    with k_col2:
        with st.container(border=True):
            st.markdown(
                """
                <div class="kpi-header">
                    <span class="kpi-title">Churn Rate</span>
                    <div class="kpi-icon-wrap red">
                        <svg viewBox="0 0 24 24"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>
                    </div>
                </div>
                <div class="kpi-body">
                    <span class="kpi-val-text">4.6%</span>
                    <div class="kpi-trend-wrap up">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="7" y1="17" x2="17" y2="7"/><polyline points="7 7 17 7 17 17"/></svg>
                        <span>+0.4pp vs last month</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    with k_col3:
        with st.container(border=True):
            st.markdown(
                """
                <div class="kpi-header">
                    <span class="kpi-title">High Risk Customers</span>
                    <div class="kpi-icon-wrap yellow">
                        <svg viewBox="0 0 24 24"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                    </div>
                </div>
                <div class="kpi-body">
                    <span class="kpi-val-text">4,114</span>
                    <div class="kpi-trend-wrap up">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="7" y1="17" x2="17" y2="7"/><polyline points="7 7 17 7 17 17"/></svg>
                        <span>+312 vs last month</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    with k_col4:
        with st.container(border=True):
            st.markdown(
                """
                <div class="kpi-header">
                    <span class="kpi-title">Revenue at Risk</span>
                    <div class="kpi-icon-wrap red">
                        <svg viewBox="0 0 24 24"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
                    </div>
                </div>
                <div class="kpi-body">
                    <span class="kpi-val-text">$1.82M</span>
                    <div class="kpi-trend-wrap up">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="7" y1="17" x2="17" y2="7"/><polyline points="7 7 17 7 17 17"/></svg>
                        <span>+$142K vs last month</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── Charts Row 1 (Line & Donut) ──────────────────────────────────────────
    ch1_l, ch1_r = st.columns(2)

    with ch1_l:
        with st.container(border=True):
            st.markdown(
                '<div class="card-title" style="border:none; margin-bottom:0.25rem;">Monthly Churn Trend</div>'
                '<div style="font-size:0.8rem; color:#6B7280; margin-bottom:1rem;">Churn rate (%) over the past 12 months</div>',
                unsafe_allow_html=True
            )
            # Create Plotly Spline curve
            months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            rates = [4.1, 3.7, 5.1, 4.7, 3.8, 4.1, 5.3, 4.6, 4.1, 3.5, 5.0, 5.3]
            df_trend = pd.DataFrame({"Month": months, "Churn Rate": rates})
            fig_trend = px.line(df_trend, x="Month", y="Churn Rate", line_shape="spline")
            fig_trend.update_traces(line=dict(color="#6366F1", width=3))
            fig_trend.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=30, r=10, t=10, b=20),
                height=220,
                xaxis=dict(
                    showgrid=False,
                    linecolor="#1C243B",
                    tickfont=dict(color="#64748B", size=11)
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="#1C243B",
                    gridwidth=1,
                    griddash="dot",
                    range=[0, 8],
                    tickvals=[0, 2, 4, 6, 8],
                    tickformat=".0f",
                    ticksuffix="%",
                    tickfont=dict(color="#64748B", size=11),
                    title=dict(text="")
                )
            )
            st.plotly_chart(fig_trend, use_container_width=True)

    with ch1_r:
        with st.container(border=True):
            st.markdown(
                '<div class="card-title" style="border:none; margin-bottom:0.25rem;">Contract Distribution</div>'
                '<div style="font-size:0.8rem; color:#6B7280; margin-bottom:1rem;">By contract type</div>',
                unsafe_allow_html=True
            )
            labels = ["Month-to-Month", "One Year", "Two Year"]
            values = [8342, 5219, 5981]
            fig_donut = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.72,
                marker=dict(colors=["#EF4444", "#F59E0B", "#10B981"]),
                textinfo="none",
                hoverinfo="label+value",
                showlegend=False
            )])
            fig_donut.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10),
                height=220,
            )
            
            # Use 2 sub-columns to arrange donut chart on the left and stats on the right
            card_col1, card_col2 = st.columns([1.1, 1.2])
            with card_col1:
                st.plotly_chart(fig_donut, use_container_width=True)
            with card_col2:
                st.markdown(
                    """
                    <div style="display: flex; flex-direction: column; justify-content: center; height: 100%; gap: 14px; padding-top: 15px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.85rem;">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: #EF4444;"></span>
                                <span style="color: #94A3B8;">Month-to-Month</span>
                            </div>
                            <strong style="color: #FFFFFF;">8,342</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.85rem;">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: #F59E0B;"></span>
                                <span style="color: #94A3B8;">One Year</span>
                            </div>
                            <strong style="color: #FFFFFF;">5,219</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.85rem;">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: #10B981;"></span>
                                <span style="color: #94A3B8;">Two Year</span>
                            </div>
                            <strong style="color: #FFFFFF;">5,981</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # ── Charts Row 2 (Bar Chart & Risk Progress Bars) ────────────────────────
    ch2_l, ch2_r = st.columns(2)

    with ch2_l:
        with st.container(border=True):
            st.markdown(
                '<div class="card-title" style="border:none; margin-bottom:0.25rem;">Customer Segmentation</div>'
                '<div style="font-size:0.8rem; color:#6B7280; margin-bottom:1rem;">Churn rate by segment</div>',
                unsafe_allow_html=True
            )
            segments = ["Enterprise", "SMB", "Consumer", "Wholesale"]
            segment_rates = [3.2, 5.9, 7.6, 2.5]
            df_seg = pd.DataFrame({"Segment": segments, "Churn Rate": segment_rates})
            fig_seg = px.bar(df_seg, x="Segment", y="Churn Rate")
            fig_seg.update_traces(
                marker_color="#5C60F5",
                marker_line_width=0,
                width=0.32
            )
            fig_seg.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=30, r=10, t=10, b=20),
                height=220,
                xaxis=dict(
                    showgrid=False,
                    linecolor="#1C243B",
                    tickfont=dict(color="#64748B", size=11),
                    title=dict(text="")
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="#1C243B",
                    gridwidth=1,
                    griddash="dot",
                    range=[0, 8],
                    tickvals=[0, 2, 4, 6, 8],
                    ticksuffix="%",
                    tickfont=dict(color="#64748B", size=11),
                    title=dict(text="")
                )
            )
            st.plotly_chart(fig_seg, use_container_width=True)

    with ch2_r:
        with st.container(border=True):
            st.markdown(
                '<div class="card-title" style="border:none; margin-bottom:0.25rem;">Churn Risk Distribution</div>'
                '<div style="font-size:0.8rem; color:#6B7280; margin-bottom:1.5rem;">Customer count by risk tier</div>',
                unsafe_allow_html=True
            )
            # Progress bars using stylized HTML
            st.markdown(
                """
                <div style="display: flex; flex-direction: column; gap: 14px; padding-top: 10px;">
                    <div class="risk-row">
                        <div class="risk-label-group" style="width: 140px;">
                            <span class="risk-indicator-dot" style="background-color: #EF4444;"></span>
                            <span class="risk-tier-name" style="width: 50px;">Critical</span>
                            <span class="risk-pct-label">75-100%</span>
                        </div>
                        <div class="risk-bar-track">
                            <div class="risk-bar-fill" style="width: 11%; background-color: #EF4444;"></div>
                        </div>
                        <span class="risk-count-value">1,243</span>
                    </div>
                    <div class="risk-row">
                        <div class="risk-label-group" style="width: 140px;">
                            <span class="risk-indicator-dot" style="background-color: #F59E0B;"></span>
                            <span class="risk-tier-name" style="width: 50px;">High</span>
                            <span class="risk-pct-label">50-74%</span>
                        </div>
                        <div class="risk-bar-track">
                            <div class="risk-bar-fill" style="width: 26%; background-color: #F59E0B;"></div>
                        </div>
                        <span class="risk-count-value">2,871</span>
                    </div>
                    <div class="risk-row">
                        <div class="risk-label-group" style="width: 140px;">
                            <span class="risk-indicator-dot" style="background-color: #3B82F6;"></span>
                            <span class="risk-tier-name" style="width: 50px;">Medium</span>
                            <span class="risk-pct-label">25-49%</span>
                        </div>
                        <div class="risk-bar-track">
                            <div class="risk-bar-fill" style="width: 42%; background-color: #3B82F6;"></div>
                        </div>
                        <span class="risk-count-value">4,592</span>
                    </div>
                    <div class="risk-row">
                        <div class="risk-label-group" style="width: 140px;">
                            <span class="risk-indicator-dot" style="background-color: #10B981;"></span>
                            <span class="risk-tier-name" style="width: 50px;">Low</span>
                            <span class="risk-pct-label">0-24%</span>
                        </div>
                        <div class="risk-bar-track">
                            <div class="risk-bar-fill" style="width: 95%; background-color: #10B981;"></div>
                        </div>
                        <span class="risk-count-value">10,836</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────────────────────────────────────
# Page B — Prediction
# ─────────────────────────────────────────────────────────────────────────────
def render_prediction(model, label_encoders, feature_columns) -> None:
    # ── Header: Title and user actions ───────────────────────────────────────
    h_col1, h_col2 = st.columns([3, 1])
    with h_col1:
        st.markdown(
            '<div class="page-title">Customer Prediction</div>'
            '<div class="page-desc">Analyze churn risk for an individual customer</div>',
            unsafe_allow_html=True
        )
    with h_col2:
        st.markdown(
            """
            <div class="header-actions">
                <div class="icon-btn-wrap">
                    <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path>
                        <path d="M13.73 21a2 2 0 0 1-3.46 0"></path>
                    </svg>
                </div>
                <div class="avatar-circle">JD</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Option lists
    g   = lambda f, d: get_opts(label_encoders, f, d)
    gender_opts    = g("gender",           ["Female", "Male"])
    partner_opts   = g("Partner",          ["No", "Yes"])
    dep_opts       = g("Dependents",       ["No", "Yes"])
    inet_opts      = g("InternetService",  ["DSL", "Fiber optic", "No"])
    sec_opts       = g("OnlineSecurity",   ["No", "No internet service", "Yes"])
    ts_opts        = g("TechSupport",      ["No", "No internet service", "Yes"])
    tv_opts        = g("StreamingTV",      ["No", "No internet service", "Yes"])
    mov_opts       = g("StreamingMovies",  ["No", "No internet service", "Yes"])
    contract_opts  = g("Contract",         ["Month-to-month", "One year", "Two year"])
    pay_opts       = g("PaymentMethod",    [
        "Bank transfer (automatic)", "Credit card (automatic)",
        "Electronic check", "Mailed check",
    ])

    form_col, result_col = st.columns([1.6, 0.9], gap="large")

    with form_col:
        with st.form("prediction_form", border=False):
            # Card 1: Customer Information
            with st.container(border=True):
                st.markdown('<div class="card-title">Customer Information</div>', unsafe_allow_html=True)
                row1_1, row1_2 = st.columns(2)
                with row1_1:
                    gender_sel = st.selectbox("GENDER", gender_opts, index=None, placeholder="Select...")
                    partner_sel = st.selectbox("PARTNER", partner_opts, index=None, placeholder="Select...")
                with row1_2:
                    senior_sel = st.selectbox("SENIOR CITIZEN", ["No", "Yes"], index=None, placeholder="Select...")
                    deps_sel = st.selectbox("DEPENDENTS", dep_opts, index=None, placeholder="Select...")

            # Card 2: Subscription Details
            with st.container(border=True):
                st.markdown('<div class="card-title">Subscription Details</div>', unsafe_allow_html=True)
                row2_1, row2_2, row2_3 = st.columns(3)
                with row2_1:
                    contract_sel = st.selectbox("CONTRACT TYPE", contract_opts, index=None, placeholder="Select...")
                with row2_2:
                    tenure_input = st.text_input("TENURE (MONTHS)", placeholder="e.g. 24")
                with row2_3:
                    internet_sel = st.selectbox("INTERNET SERVICE", inet_opts, index=None, placeholder="Select...")

            # Card 3: Services
            with st.container(border=True):
                st.markdown('<div class="card-title">Services</div>', unsafe_allow_html=True)
                row3_1, row3_2 = st.columns(2)
                with row3_1:
                    sec_sel = st.selectbox("ONLINE SECURITY", sec_opts, index=None, placeholder="Select...")
                    tv_sel = st.selectbox("STREAMING TV", tv_opts, index=None, placeholder="Select...")
                with row3_2:
                    ts_sel = st.selectbox("TECH SUPPORT", ts_opts, index=None, placeholder="Select...")
                    mov_sel = st.selectbox("STREAMING MOVIES", mov_opts, index=None, placeholder="Select...")

            # Card 4: Billing Information
            with st.container(border=True):
                st.markdown('<div class="card-title">Billing Information</div>', unsafe_allow_html=True)
                row4_1, row4_2, row4_3 = st.columns(3)
                with row4_1:
                    monthly_input = st.text_input("MONTHLY CHARGES ($)", placeholder="e.g. 79.50")
                with row4_2:
                    total_input = st.text_input("TOTAL CHARGES ($)", placeholder="e.g. 1908.00")
                with row4_3:
                    pay_sel = st.selectbox("PAYMENT METHOD", pay_opts, index=None, placeholder="Select...")

            st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Run Prediction →", type="primary", use_container_width=True)

    # Decode, predict & render on the right column
    with result_col:
        if not submitted:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="placeholder-card">
                        <div class="placeholder-icon">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                                <line x1="12" y1="9" x2="12" y2="13"/>
                                <line x1="12" y1="17" x2="12.01" y2="17"/>
                            </svg>
                        </div>
                        <div class="placeholder-text">
                            Complete the form and run the prediction to see the churn risk analysis.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            # Safe fallbacks if fields left as "Select..."
            gender = gender_sel if gender_sel is not None else "Male"
            senior = senior_sel if senior_sel is not None else "No"
            partner = partner_sel if partner_sel is not None else "No"
            deps = deps_sel if deps_sel is not None else "No"
            contract = contract_sel if contract_sel is not None else "Month-to-month"
            internet = internet_sel if internet_sel is not None else "DSL"
            sec = sec_sel if sec_sel is not None else "No"
            ts = ts_sel if ts_sel is not None else "No"
            tv = tv_sel if tv_sel is not None else "No"
            mov = mov_sel if mov_sel is not None else "No"
            pay = pay_sel if pay_sel is not None else "Electronic check"

            # Parse inputs
            try:
                tenure = int(tenure_input.strip()) if tenure_input.strip() else 24
            except ValueError:
                tenure = 24

            try:
                monthly = float(monthly_input.strip()) if monthly_input.strip() else 79.50
            except ValueError:
                monthly = 79.50

            try:
                total_c = float(total_input.strip()) if total_input.strip() else 1908.00
            except ValueError:
                total_c = 1908.00

            # Default values for hidden/omitted features
            phone = "Yes"
            multi = "No"
            bkp = "No"
            prot = "No"
            paperless = "Yes"

            # ── Encode & Predict ──
            raw = {
                "gender": gender, "SeniorCitizen": 1.0 if senior == "Yes" else 0.0,
                "Partner": partner, "Dependents": deps, "tenure": float(tenure),
                "PhoneService": phone, "MultipleLines": multi,
                "InternetService": internet, "OnlineSecurity": sec,
                "OnlineBackup": bkp, "DeviceProtection": prot,
                "TechSupport": ts, "StreamingTV": tv, "StreamingMovies": mov,
                "Contract": contract, "PaperlessBilling": paperless,
                "PaymentMethod": pay, "MonthlyCharges": float(monthly),
                "TotalCharges": float(total_c),
            }

            encoded = {}
            for col in feature_columns:
                if col in label_encoders:
                    v = str(raw[col]) if type(label_encoders[col].classes_[0]) in (str, np.str_) else raw[col]
                    encoded[col] = label_encoders[col].transform([v])[0]
                else:
                    encoded[col] = raw[col]

            input_df   = pd.DataFrame([encoded])[feature_columns]
            prediction = model.predict(input_df)[0]
            prob       = model.predict_proba(input_df)[0][1]
            pct        = prob * 100
            risk       = _risk(pct)

            # ── Log to history ──
            history_file = "history.csv"
            log = raw.copy()
            log["Prediction"]  = "Churn" if prediction == 1 else "No Churn"
            log["Probability"] = round(pct, 2)
            log["Timestamp"]   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_df = pd.DataFrame([log])
            if os.path.exists(history_file):
                try:
                    combined = pd.concat([pd.read_csv(history_file), new_df], ignore_index=True)
                except Exception:
                    combined = new_df
            else:
                combined = new_df
            combined.to_csv(history_file, index=False)

            # ── Render Results ──
            with st.container(border=True):
                outcome = "Customer likely to churn" if prediction == 1 else "Customer likely to remain"
                confidence = max(pct, 100 - pct)
                st.markdown(
                    f"""
                    <div class="verdict-wrap">
                        <span class="badge {risk['class']}">{risk['label']}</span>
                        <div class="verdict-outcome" style="margin-top:0.75rem;">{outcome}</div>
                        <div class="verdict-desc">{risk['desc']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Gauge Chart
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pct,
                    number={"suffix": "%", "font": {"size": 28, "color": "#F9FAFB", "family": "Inter"}},
                    title={"text": "Churn Probability", "font": {"size": 11, "color": "#6B7280", "family": "Inter"}},
                    gauge={
                        "axis":        {"range": [0, 100], "tickwidth": 1, "tickcolor": "#1C243B", "tickfont": {"color": "#64748B", "size": 10}},
                        "bar":         {"color": risk["color"]},
                        "bgcolor":     "rgba(0,0,0,0)",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 30],   "color": "rgba(52,211,153,0.06)"},
                            {"range": [30, 60],  "color": "rgba(251,191,36,0.06)"},
                            {"range": [60, 100], "color": "rgba(248,113,113,0.06)"},
                        ],
                        "threshold": {"line": {"color": risk["color"], "width": 2}, "thickness": 0.75, "value": pct},
                    },
                ))
                fig_g.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=150,
                    margin=dict(l=10, r=10, t=20, b=10),
                    font={"family": "Inter"},
                )
                st.plotly_chart(fig_g, use_container_width=True)
                st.markdown(
                    f'<div style="text-align:center;font-size:0.75rem;color:#64748B;margin-top:-0.5rem">'
                    f'Model confidence: {confidence:.1f}%</div>',
                    unsafe_allow_html=True,
                )

            with st.container(border=True):
                st.markdown('<div class="card-title" style="border:none; margin-bottom:0.75rem;">Contributing Factors</div>', unsafe_allow_html=True)
                neg_f, pos_f = _factors(contract, tenure, ts, sec, internet, pay)

                if not neg_f and not pos_f:
                    st.markdown(
                        '<div style="color:#64748B;font-size:0.85rem;padding:0.5rem 0">'
                        'No strong contributing factors identified.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    if neg_f:
                        st.markdown('<div class="factor-group-label neg">Risk Factors</div>', unsafe_allow_html=True)
                        for name, detail, weight in neg_f[:4]:
                            st.markdown(
                                f'<div class="factor-row">'
                                f'<div><div class="factor-name">{name}</div>'
                                f'<div class="factor-detail">{detail}</div></div>'
                                f'<span class="factor-weight neg">{weight}</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                    if pos_f:
                        margin_top = "margin-top:1.25rem" if neg_f else ""
                        st.markdown(
                            f'<div class="factor-group-label pos" style="{margin_top}">Retention Signals</div>',
                            unsafe_allow_html=True,
                        )
                        for name, detail, weight in pos_f[:4]:
                            st.markdown(
                                f'<div class="factor-row">'
                                f'<div><div class="factor-name">{name}</div>'
                                f'<div class="factor-detail">{detail}</div></div>'
                                f'<span class="factor-weight pos">{weight}</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

            st.toast("Prediction recorded.")


# ─────────────────────────────────────────────────────────────────────────────
# Page C — History
# ─────────────────────────────────────────────────────────────────────────────
def render_history() -> None:
    # ── Header: Title and user actions ───────────────────────────────────────
    h_col1, h_col2 = st.columns([3, 1])
    with h_col1:
        st.markdown(
            '<div class="page-title">Prediction History</div>'
            '<div class="page-desc">All recorded churn prediction sessions.</div>',
            unsafe_allow_html=True
        )
    with h_col2:
        st.markdown(
            """
            <div class="header-actions">
                <div class="icon-btn-wrap">
                    <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path>
                        <path d="M13.73 21a2 2 0 0 1-3.46 0"></path>
                    </svg>
                </div>
                <div class="avatar-circle">JD</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    history_file = "history.csv"
    if not os.path.exists(history_file):
        st.info("No predictions recorded yet. Run a prediction to get started.")
        return

    try:
        df = pd.read_csv(history_file)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.sort_values("Timestamp", ascending=False)
    except Exception as e:
        st.error(f"Could not load history: {e}")
        return

    # ── Filters (inside Card Container) ──────────────────────────────────────
    with st.container(border=True):
        st.markdown('<div class="card-title" style="border:none; margin-bottom:0.75rem;">Filters</div>', unsafe_allow_html=True)
        f1, f2, f3 = st.columns([2, 1, 1], gap="medium")
        with f1:
            q = st.text_input("Search contract or payment method", placeholder="e.g. Month-to-month")
        with f2:
            outcome = st.selectbox("Outcome", ["All", "Churn", "No Churn"])
        with f3:
            risk_level = st.selectbox("Risk Level", ["All", "High (≥60%)", "Medium (30–60%)", "Low (<30%)"])

    filtered = df.copy()
    if q:
        ql = q.lower()
        filtered = filtered[
            filtered["Contract"].astype(str).str.lower().str.contains(ql)
            | filtered["PaymentMethod"].astype(str).str.lower().str.contains(ql)
        ]
    if outcome != "All":
        filtered = filtered[filtered["Prediction"] == outcome]
    if "High" in risk_level:
        filtered = filtered[filtered["Probability"] >= 60]
    elif "Medium" in risk_level:
        filtered = filtered[(filtered["Probability"] >= 30) & (filtered["Probability"] < 60)]
    elif "Low" in risk_level:
        filtered = filtered[filtered["Probability"] < 30]

    st.markdown(f'<div style="font-size:0.875rem; color:#6B7280; margin-bottom:0.75rem; font-weight:500;">{len(filtered)} of {len(df)} records</div>', unsafe_allow_html=True)

    if filtered.empty:
        st.info("No records match the current filters.")
        return

    cols = [c for c in ["Timestamp","gender","tenure","Contract","PaymentMethod",
                         "MonthlyCharges","TotalCharges","Prediction","Probability"]
            if c in filtered.columns]
    view = filtered[cols].reset_index(drop=True).copy()
    view["Timestamp"]      = view["Timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    view["Probability"]    = view["Probability"].astype(str) + "%"
    view["MonthlyCharges"] = "$" + view["MonthlyCharges"].round(2).astype(str)
    view["TotalCharges"]   = "$" + view["TotalCharges"].round(2).astype(str)

    st.dataframe(
        view,
        use_container_width=True,
        column_config={
            "Timestamp":      "Timestamp",
            "gender":         "Gender",
            "tenure":         "Tenure (mo)",
            "Contract":       "Contract",
            "PaymentMethod":  "Payment Method",
            "MonthlyCharges": "Monthly",
            "TotalCharges":   "Total",
            "Prediction":     "Outcome",
            "Probability":    "Risk Prob.",
        },
        hide_index=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page D — Model Insights
# ─────────────────────────────────────────────────────────────────────────────
def render_model_insights(metrics, cm, feature_importance, model_obj) -> None:
    # ── Header: Title and user actions ───────────────────────────────────────
    h_col1, h_col2 = st.columns([3, 1])
    with h_col1:
        st.markdown(
            '<div class="page-title">Model Performance</div>'
            '<div class="page-desc">Classifier evaluation metrics and diagnostics.</div>',
            unsafe_allow_html=True
        )
    with h_col2:
        st.markdown(
            """
            <div class="header-actions">
                <div class="icon-btn-wrap">
                    <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path>
                        <path d="M13.73 21a2 2 0 0 1-3.46 0"></path>
                    </svg>
                </div>
                <div class="avatar-circle">JD</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ── Evaluation Metrics (KPI Cards) ─────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="kpi-header">
                    <span class="kpi-title">Accuracy</span>
                    <div class="kpi-icon-wrap blue">
                        <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M22 12A10 10 0 0 0 12 2v10z"/></svg>
                    </div>
                </div>
                <div class="kpi-body">
                    <span class="kpi-val-text">{metrics['accuracy'] * 100:.2f}%</span>
                    <div style="font-size:0.75rem; color:#6B7280; margin-top:0.25rem;">Overall correct predictions</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    with m2:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="kpi-header">
                    <span class="kpi-title">Precision</span>
                    <div class="kpi-icon-wrap blue">
                        <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M22 12A10 10 0 0 0 12 2v10z"/></svg>
                    </div>
                </div>
                <div class="kpi-body">
                    <span class="kpi-val-text">{metrics['precision'] * 100:.2f}%</span>
                    <div style="font-size:0.75rem; color:#6B7280; margin-top:0.25rem;">Predicted positives correctness</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    with m3:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="kpi-header">
                    <span class="kpi-title">Recall</span>
                    <div class="kpi-icon-wrap blue">
                        <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M22 12A10 10 0 0 0 12 2v10z"/></svg>
                    </div>
                </div>
                <div class="kpi-body">
                    <span class="kpi-val-text">{metrics['recall'] * 100:.2f}%</span>
                    <div style="font-size:0.75rem; color:#6B7280; margin-top:0.25rem;">Actual positives captured</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    with m4:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="kpi-header">
                    <span class="kpi-title">F1 Score</span>
                    <div class="kpi-icon-wrap blue">
                        <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M22 12A10 10 0 0 0 12 2v10z"/></svg>
                    </div>
                </div>
                <div class="kpi-body">
                    <span class="kpi-val-text">{metrics['f1_score'] * 100:.2f}%</span>
                    <div style="font-size:0.75rem; color:#6B7280; margin-top:0.25rem;">Harmonic mean of precision & recall</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    cv = metrics.get("cv_accuracy", 0.0)
    st.markdown(
        f"""
        <div style="background-color: rgba(16, 185, 129, 0.05); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 8px; font-size: 0.875rem; color: #34D399;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
            <span>5-Fold Cross-Validation Accuracy: <strong>{cv * 100:.2f}%</strong></span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── Two-column layout for charts ─────────────────────────────────────────
    ci_l, ci_r = st.columns(2, gap="medium")

    with ci_l:
        with st.container(border=True):
            st.markdown('<div class="card-title" style="border:none; margin-bottom:1rem;">Feature Importance</div>', unsafe_allow_html=True)
            if hasattr(model_obj, "feature_importances_"):
                imp = feature_importance.copy()
                imp["Weight (%)"] = imp["Importance"] * 100
                imp = imp.sort_values("Weight (%)", ascending=True).tail(15)
                fig = px.bar(
                    imp, x="Weight (%)", y="Feature", orientation="h",
                    color="Weight (%)",
                    color_continuous_scale=["#1C243B", "#6366F1"],
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=380,
                    coloraxis_showscale=False,
                    xaxis=dict(
                        showgrid=True,
                        gridcolor="#1C243B",
                        gridwidth=1,
                        griddash="dot",
                        tickfont=dict(color="#64748B", size=10),
                        linecolor="#1C243B",
                        title=dict(text="Importance (%)", font=dict(color="#64748B", size=10))
                    ),
                    yaxis=dict(
                        showgrid=False,
                        tickfont=dict(color="#94A3B8", size=10),
                        linecolor="#1C243B",
                        title=dict(text="")
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feature importance unavailable for this model type.")

    with ci_r:
        with st.container(border=True):
            st.markdown('<div class="card-title" style="border:none; margin-bottom:1rem;">Confusion Matrix</div>', unsafe_allow_html=True)
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                labels=dict(x="Predicted", y="Actual"),
                color_continuous_scale=["#111625", "#6366F1"],
            )
            fig_cm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10),
                height=260,
                coloraxis_showscale=False,
                xaxis=dict(tickfont=dict(color="#64748B", size=10), title=dict(text="Predicted", font=dict(color="#64748B", size=10))),
                yaxis=dict(tickfont=dict(color="#64748B", size=10), title=dict(text="Actual", font=dict(color="#64748B", size=10)))
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with st.container(border=True):
            st.markdown(
                '<div style="font-size:0.8rem;font-weight:600;color:#9CA3AF;margin-bottom:0.5rem">Algorithm Details</div>'
                '<div style="font-size:0.875rem;color:#F9FAFB;line-height:1.6">'
                'Random Forest Classifier trained on the IBM Telco Customer Churn dataset. '
                'Stratified 80/20 split · 5-fold cross-validation.'
                '</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Page E — Settings
# ─────────────────────────────────────────────────────────────────────────────
def render_settings() -> None:
    # ── Header: Title and user actions ───────────────────────────────────────
    h_col1, h_col2 = st.columns([3, 1])
    with h_col1:
        st.markdown(
            '<div class="page-title">Settings</div>'
            '<div class="page-desc">Manage platform preferences and classifier parameters.</div>',
            unsafe_allow_html=True
        )
    with h_col2:
        st.markdown(
            """
            <div class="header-actions">
                <div class="icon-btn-wrap">
                    <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path>
                        <path d="M13.73 21a2 2 0 0 1-3.46 0"></path>
                    </svg>
                </div>
                <div class="avatar-circle">JD</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with st.container(border=True):
        st.markdown('<div class="card-title">Threshold Settings</div>', unsafe_allow_html=True)
        st.slider("Medium Churn Risk Threshold (%)", 10, 50, 30)
        st.slider("High Churn Risk Threshold (%)", 50, 90, 60)

    with st.container(border=True):
        st.markdown('<div class="card-title">Data Sources & Models</div>', unsafe_allow_html=True)
        st.text_input("IBM Customer Dataset Path", value="WA_Fn-UseC_-Telco-Customer-Churn.csv", disabled=True)
        st.text_input("Active ML Model Path", value="churn_model.pkl", disabled=True)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    if not st.session_state.logged_in:
        render_login()

    # Resources loaded only after authentication
    try:
        model, label_encoders, feature_columns = load_ml_resources()
        df = load_dataset()
        metrics, cm, feature_importance, model_obj = load_model_artifacts()
    except Exception as exc:
        st.error(f"Failed to load model files: {exc}")
        st.stop()

    page = render_sidebar()

    if page == "Dashboard":
        render_dashboard(df)
    elif page == "Prediction":
        render_prediction(model, label_encoders, feature_columns)
    elif page == "History":
        render_history()
    elif page == "Model Insights":
        render_model_insights(metrics, cm, feature_importance, model_obj)
    elif page == "Settings":
        render_settings()


if __name__ == "__main__":
    main()