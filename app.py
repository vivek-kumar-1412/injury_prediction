"""
Football Injury Prediction System
Swiss International Typographic Style UI
"""
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
from datetime import datetime

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Football Injury Prediction System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# SESSION STATE — prediction history
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------------------------
# CSS — Swiss Design System
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');

:root {
    --bg: #0a0a0f;
    --bg-secondary: #12121a;
    --bg-card: rgba(255,255,255,0.04);
    --fg: #e8e8f0;
    --fg-muted: #8888a0;
    --accent: #6c5ce7;
    --accent-secondary: #a29bfe;
    --danger: #ff6b6b;
    --success: #51cf66;
    --gradient-1: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 50%, #74b9ff 100%);
    --gradient-2: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    --glass: rgba(255,255,255,0.05);
    --glass-border: rgba(255,255,255,0.08);
    --glass-hover: rgba(255,255,255,0.1);
    --unit: 8px;
    --radius: 16px;
    --radius-sm: 8px;
    --font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    font-family: var(--font) !important;
    background: var(--bg) !important;
    color: var(--fg) !important;
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; top: -50%; left: -50%; width: 200%; height: 200%;
    background: radial-gradient(circle at 20% 20%, rgba(108,92,231,0.08) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(116,185,255,0.06) 0%, transparent 50%);
    z-index: 0; pointer-events: none;
    animation: bgShift 20s ease-in-out infinite alternate;
}

@keyframes bgShift {
    0% { transform: translate(0, 0); }
    100% { transform: translate(3%, 3%); }
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 20px rgba(108,92,231,0.2); }
    50% { box-shadow: 0 0 40px rgba(108,92,231,0.4); }
}

#MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {
    display: none !important;
}

.block-container {
    padding-top: calc(var(--unit)*5) !important;
    padding-bottom: calc(var(--unit)*10) !important;
    max-width: 1200px !important;
    position: relative; z-index: 1;
}

h1,h2,h3,h4,h5,h6,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    font-family: var(--font) !important;
    text-transform: uppercase;
    letter-spacing: -0.02em;
    font-weight: 800 !important;
    color: var(--fg) !important;
    line-height: 1 !important;
}

p, span, label, div { font-family: var(--font) !important; }

/* --- Inputs --- */
[data-testid="stSlider"] label,
[data-testid="stNumberInput"] label,
[data-testid="stTextInput"] label {
    font-family: var(--font) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    font-weight: 600 !important;
    font-size: 11px !important;
    color: var(--fg-muted) !important;
}
[data-testid="stSlider"] [role="slider"] {
    background: var(--accent) !important;
    border-radius: 50% !important;
    box-shadow: 0 0 12px rgba(108,92,231,0.5) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="progressbar"] {
    background: var(--gradient-1) !important;
}
[data-testid="stTextInput"] input {
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--glass-border) !important;
    background: var(--glass) !important;
    color: var(--fg) !important;
    font-family: var(--font) !important;
    font-weight: 500 !important;
    font-size: 15px !important;
    padding: calc(var(--unit)*1.5) calc(var(--unit)*2) !important;
    backdrop-filter: blur(12px) !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(108,92,231,0.2) !important;
}

/* --- Buttons --- */
[data-testid="stButton"]>button, .stButton>button {
    background: var(--gradient-1) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    font-weight: 700 !important;
    font-family: var(--font) !important;
    font-size: 13px !important;
    padding: calc(var(--unit)*2) calc(var(--unit)*4) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease, opacity 0.2s ease !important;
    width: 100% !important;
    min-height: 52px !important;
    box-shadow: 0 4px 20px rgba(108,92,231,0.3) !important;
}
[data-testid="stButton"]>button:hover, .stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(108,92,231,0.5) !important;
    opacity: 0.95 !important;
}
[data-testid="stButton"]>button:active, .stButton>button:active {
    transform: translateY(0) !important;
}

[data-testid="stHorizontalBlock"] { gap: calc(var(--unit)*3) !important; }

/* --- Tabs --- */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: calc(var(--unit)*1) !important;
    border-bottom: 1px solid var(--glass-border) !important;
    padding-bottom: 0 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: var(--font) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-weight: 600 !important;
    font-size: 11px !important;
    border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
    border: 1px solid transparent !important;
    padding: calc(var(--unit)*1.5) calc(var(--unit)*3) !important;
    color: var(--fg-muted) !important;
    background: transparent !important;
    transition: all 0.2s ease !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    color: var(--fg) !important;
    background: var(--glass) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
    background: var(--glass) !important;
    color: var(--accent-secondary) !important;
    border-color: var(--glass-border) !important;
    border-bottom-color: transparent !important;
}

/* --- Dividers --- */
hr {
    border: none !important;
    border-top: 1px solid var(--glass-border) !important;
    margin: calc(var(--unit)*4) 0 !important;
}

/* --- Metrics --- */
[data-testid="stMetric"] {
    border: 1px solid var(--glass-border);
    border-radius: var(--radius) !important;
    padding: calc(var(--unit)*3);
    background: var(--glass);
    backdrop-filter: blur(12px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}
[data-testid="stMetric"] label {
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    font-weight: 600 !important;
    font-size: 10px !important;
    color: var(--fg-muted) !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-weight: 800 !important;
    font-size: 26px !important;
    color: var(--fg) !important;
}

/* --- Selectbox --- */
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: var(--glass) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--fg) !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius-sm) !important;
}

/* --- Animations --- */
.stSlider, .stTextInput, .stSelectbox, .stButton, .stMetric, .stTabs {
    animation: fadeInUp 0.5s ease-out both;
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}

@media (max-width: 768px) {
    .block-container {
        padding-left: calc(var(--unit)*2) !important;
        padding-right: calc(var(--unit)*2) !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "injury_model.pkl")

@st.cache_resource
def load_model():
    """Load the pre-trained injury prediction model."""
    return joblib.load(MODEL_PATH)

model = load_model()

FEATURE_NAMES = ["Age","BMI","Total Distance","Sprint Count","Acceleration Load",
                 "ACWR","Yo-Yo Score","Jump Height","Previous Injuries","Minutes Played"]

SAMPLE_PROFILES = {
    "-- Select --": None,
    "Young Midfielder (Low Risk)": [22, 22.5, 10.2, 28, 120, 0.85, 20.5, 42, 0, 75],
    "Veteran Striker (High Risk)":  [34, 26.8, 12.5, 52, 260, 1.75, 15.0, 30, 5, 88],
    "Average Player":               [27, 24.0, 9.0, 35, 180, 1.10, 18.0, 38, 2, 60],
}

# Helper: section label
def section_label(num, text):
    st.markdown(f'<p style="font-size:11px;text-transform:uppercase;letter-spacing:0.25em;'
                f'font-weight:600;margin:0 0 12px 0;font-family:Inter,sans-serif;'
                f'background:linear-gradient(135deg,#6c5ce7,#a29bfe);'
                f'-webkit-background-clip:text;-webkit-text-fill-color:transparent">'
                f'{num}. {text}</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown("""
<div style="border-bottom:1px solid rgba(255,255,255,0.08);padding-bottom:32px;margin-bottom:40px;
    animation:fadeInUp 0.6s ease-out both">
    <p style="font-size:11px;text-transform:uppercase;letter-spacing:0.25em;font-weight:600;
       margin:0 0 12px 0;font-family:Inter,sans-serif;
       background:linear-gradient(135deg,#6c5ce7,#a29bfe);
       -webkit-background-clip:text;-webkit-text-fill-color:transparent">01. System</p>
    <h1 style="font-size:clamp(2.2rem,5vw,4.2rem);margin:0;line-height:1;font-weight:900;
       font-family:Inter,sans-serif;text-transform:uppercase;letter-spacing:-0.03em;
       background:linear-gradient(135deg,#e8e8f0 0%,#a29bfe 60%,#74b9ff 100%);
       -webkit-background-clip:text;-webkit-text-fill-color:transparent">
       Football<br>Injury<br>Prediction</h1>
    <p style="font-size:13px;letter-spacing:0.05em;font-weight:400;
       margin:20px 0 0 0;max-width:480px;line-height:1.7;font-family:Inter,sans-serif;
       color:#8888a0">
       Machine-learning driven risk assessment.<br>
       10 biometric inputs &middot; Binary classification &middot; Instant result.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PLAYER NAME + PROFILE SELECTOR
# ---------------------------------------------------------------------------
section_label("02", "PLAYER IDENTITY")
name_col, profile_col = st.columns([1, 1], gap="large")
with name_col:
    player_name = st.text_input("PLAYER NAME", value="", placeholder="Enter player name...")
with profile_col:
    profile_choice = st.selectbox("AUTOFILL PROFILE", options=list(SAMPLE_PROFILES.keys()), label_visibility="visible")

defaults = SAMPLE_PROFILES.get(profile_choice)
st.markdown("---")

# ---------------------------------------------------------------------------
# INPUT SLIDERS
# ---------------------------------------------------------------------------
section_label("03", "INPUT PARAMETERS")

FEATURE_CONFIG = [
    ("AGE",18.0,40.0,25.0,1.0,"Player age in years"),
    ("BMI",18.0,32.0,23.5,0.1,"Body Mass Index (kg/m^2)"),
    ("TOTAL DISTANCE",4.0,15.0,9.5,0.1,"Distance covered per match (km)"),
    ("SPRINT COUNT",5.0,70.0,30.0,1.0,"Number of sprints per match"),
    ("ACCELERATION LOAD",40.0,350.0,160.0,5.0,"Cumulative acceleration load (AU)"),
    ("ACWR",0.4,2.2,1.05,0.05,"Acute : Chronic Workload Ratio"),
    ("YO-YO SCORE",13.0,24.0,18.5,0.5,"Yo-Yo Intermittent Recovery level"),
    ("JUMP HEIGHT",20.0,60.0,40.0,0.5,"Counter-movement jump height (cm)"),
    ("PREVIOUS INJURIES",0.0,8.0,1.0,1.0,"Total prior injury count"),
    ("MINUTES PLAYED",0.0,90.0,65.0,1.0,"Average minutes per match"),
]

inputs = []
col_left, col_right = st.columns(2, gap="large")
for idx, (label, mn, mx, default_val, step, help_text) in enumerate(FEATURE_CONFIG):
    col = col_left if idx < 5 else col_right
    val = defaults[idx] if defaults else default_val
    with col:
        v = st.slider(label, min_value=mn, max_value=mx, value=float(val), step=step, help=help_text)
        inputs.append(v)

st.markdown("---")

# ---------------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------------
section_label("04", "ANALYSIS")

predict_col, _ = st.columns([1, 2])
with predict_col:
    predict_clicked = st.button("RUN PREDICTION")

if predict_clicked:
    display_name = player_name.strip() if player_name.strip() else "UNKNOWN PLAYER"
    X_input = np.array(inputs).reshape(1, -1)
    prediction = model.predict(X_input)[0]
    probabilities = model.predict_proba(X_input)[0]
    risk_prob = probabilities[1]
    safe_prob = probabilities[0]
    is_high_risk = prediction == 1
    label = "HIGH RISK" if is_high_risk else "LOW RISK"

    # Save to history
    st.session_state.history.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "player": display_name,
        "risk": label,
        "probability": round(risk_prob * 100, 1),
        "inputs": list(inputs),
    })

    # ── Result banner ──────────────────────────────────────────────
    banner_gradient = "linear-gradient(135deg,rgba(255,107,107,0.15),rgba(238,90,36,0.1))" if is_high_risk else "linear-gradient(135deg,rgba(81,207,102,0.12),rgba(108,92,231,0.08))"
    border_color = "rgba(255,107,107,0.3)" if is_high_risk else "rgba(81,207,102,0.3)"
    label_gradient = "linear-gradient(135deg,#ff6b6b,#ee5a24)" if is_high_risk else "linear-gradient(135deg,#51cf66,#74b9ff)"
    icon_char = "⚠" if is_high_risk else "✓"

    st.markdown(f"""
    <div style="background:{banner_gradient};border:1px solid {border_color};border-radius:16px;
        padding:48px;margin:24px 0;position:relative;backdrop-filter:blur(16px);
        animation:fadeInUp 0.4s ease-out both;overflow:hidden">
        <div style="position:absolute;top:16px;right:24px;font-size:64px;opacity:0.15;
            line-height:1">{icon_char}</div>
        <p style="font-size:11px;text-transform:uppercase;letter-spacing:0.3em;font-weight:600;
            margin:0 0 4px 0;color:#8888a0;font-family:Inter,sans-serif">Player</p>
        <p style="font-size:clamp(1.2rem,3vw,2rem);font-weight:800;margin:0 0 20px 0;
            letter-spacing:0.02em;font-family:Inter,sans-serif;text-transform:uppercase;
            color:#e8e8f0">{display_name}</p>
        <p style="font-size:11px;text-transform:uppercase;letter-spacing:0.3em;font-weight:600;
            margin:0 0 8px 0;color:#8888a0;font-family:Inter,sans-serif">Injury Risk Assessment</p>
        <p style="font-size:clamp(2.5rem,7vw,5rem);font-weight:900;margin:0;line-height:0.95;
            letter-spacing:-0.03em;font-family:Inter,sans-serif;text-transform:uppercase;
            background:{label_gradient};-webkit-background-clip:text;
            -webkit-text-fill-color:transparent">{label}</p>
        <p style="font-size:18px;font-weight:700;margin:20px 0 0 0;font-family:Inter,sans-serif;
            letter-spacing:0.05em;color:#e8e8f0">Probability: {risk_prob*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics row ────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("INJURY PROBABILITY", f"{risk_prob*100:.1f}%")
    m2.metric("SAFE PROBABILITY", f"{safe_prob*100:.1f}%")
    m3.metric("CLASSIFICATION", label)
    st.markdown("---")

    # ── Graphs section ─────────────────────────────────────────────
    section_label("05", "PLAYER ANALYSIS")
    tab_radar, tab_importance, tab_inputs = st.tabs(["PLAYER RADAR", "FEATURE IMPORTANCE", "INPUT SUMMARY"])

    # -- Radar chart: normalise inputs to 0-100 scale
    with tab_radar:
        mins = [c[1] for c in FEATURE_CONFIG]
        maxs = [c[2] for c in FEATURE_CONFIG]
        normalised = [((inputs[i] - mins[i]) / (maxs[i] - mins[i])) * 100 for i in range(10)]
        radar_labels = [c[0] for c in FEATURE_CONFIG]

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(
            r=normalised + [normalised[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            fillcolor="rgba(108,92,231,0.15)",
            line=dict(color="#a29bfe", width=2),
            marker=dict(size=6, color="#6c5ce7"),
            name=display_name,
        ))
        fig_r.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], showticklabels=False,
                                gridcolor="rgba(255,255,255,0.06)"),
                angularaxis=dict(tickfont=dict(family="Inter", size=10, color="#8888a0"),
                                 gridcolor="rgba(255,255,255,0.06)"),
                bgcolor="rgba(255,255,255,0.02)",
            ),
            paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
            margin=dict(l=60, r=60, t=40, b=40), height=420,
            font=dict(family="Inter", color="#e8e8f0"),
        )
        st.plotly_chart(fig_r, use_container_width=True)

    # -- Feature importance
    with tab_importance:
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        sorted_names = [FEATURE_NAMES[i] for i in sorted_idx]
        sorted_vals = importances[sorted_idx]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=sorted_vals, y=sorted_names, orientation="h",
            marker=dict(color=["#6c5ce7" if i == 0 else "#a29bfe" for i in range(len(sorted_vals))],
                        line=dict(width=0)),
            text=[f"{v:.3f}" for v in sorted_vals], textposition="outside",
            textfont=dict(family="Inter", size=12, color="#e8e8f0"),
        ))
        fig.update_layout(
            font=dict(family="Inter", size=12, color="#e8e8f0"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=40, t=24, b=0), height=400,
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=True,
                       zerolinecolor="rgba(255,255,255,0.15)", zerolinewidth=1,
                       title=dict(text="IMPORTANCE SCORE", font=dict(size=10, family="Inter", color="#8888a0"))),
            yaxis=dict(autorange="reversed", tickfont=dict(size=11, family="Inter", color="#8888a0")),
            bargap=0.3,
        )
        st.plotly_chart(fig, use_container_width=True)

    # -- Input summary table
    with tab_inputs:
        df_inputs = pd.DataFrame({"Feature": FEATURE_NAMES, "Value": [f"{v:.2f}" for v in inputs]})
        st.dataframe(df_inputs, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# PREDICTION HISTORY
# ---------------------------------------------------------------------------
if st.session_state.history:
    st.markdown("---")
    section_label("06", "PREDICTION HISTORY")

    hist = st.session_state.history
    df_hist = pd.DataFrame([{
        "Time": h["timestamp"], "Player": h["player"],
        "Risk": h["risk"], "Probability (%)": h["probability"],
    } for h in hist])

    tab_table, tab_trend = st.tabs(["HISTORY TABLE", "PROBABILITY TREND"])

    with tab_table:
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

    with tab_trend:
        fig_t = go.Figure()
        colors = ["#ff6b6b" if h["risk"] == "HIGH RISK" else "#51cf66" for h in hist]
        labels = [f'{h["player"][:12]}' for h in hist]
        fig_t.add_trace(go.Scatter(
            x=list(range(1, len(hist) + 1)), y=[h["probability"] for h in hist],
            mode="lines+markers+text", text=[f'{h["probability"]}%' for h in hist],
            textposition="top center", textfont=dict(family="Inter", size=11, color="#e8e8f0"),
            line=dict(color="#a29bfe", width=2),
            marker=dict(size=10, color=colors, line=dict(width=2, color="rgba(255,255,255,0.2)")),
        ))
        # Danger threshold line
        fig_t.add_hline(y=50, line_dash="dash", line_color="#ff6b6b", line_width=1,
                        annotation_text="50% THRESHOLD", annotation_font=dict(size=10, color="#ff6b6b", family="Inter"))
        fig_t.update_layout(
            font=dict(family="Inter", size=12, color="#e8e8f0"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=20, t=30, b=40), height=350,
            xaxis=dict(title="PREDICTION #", tickvals=list(range(1, len(hist) + 1)),
                       ticktext=labels, tickfont=dict(size=10, color="#8888a0"),
                       gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="RISK PROBABILITY (%)", range=[0, 105],
                       gridcolor="rgba(255,255,255,0.06)", tickfont=dict(size=10, color="#8888a0")),
            showlegend=False,
        )
        st.plotly_chart(fig_t, use_container_width=True)

    # Clear history button
    clear_col, _ = st.columns([1, 3])
    with clear_col:
        if st.button("CLEAR HISTORY"):
            st.session_state.history = []
            st.rerun()

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown("""
<div style="border-top:1px solid rgba(255,255,255,0.08);margin-top:48px;padding-top:24px;display:flex;
    justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:16px">
    <p style="font-size:10px;text-transform:uppercase;letter-spacing:0.2em;font-weight:600;
       margin:0;font-family:Inter,sans-serif;color:#8888a0">Football Injury Prediction System</p>
    <p style="font-size:10px;text-transform:uppercase;letter-spacing:0.2em;font-weight:400;
       margin:0;font-family:Inter,sans-serif;color:#8888a0;opacity:0.5">
       Machine Learning &middot; Gradient Boosting &middot; 10 Features</p>
</div>
""", unsafe_allow_html=True)
