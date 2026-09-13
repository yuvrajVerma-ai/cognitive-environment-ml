import streamlit as st
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import predict_reaction_time

st.set_page_config(
    page_title="Cognitive Environment Predictor",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
<style>
    .big-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #9CA3AF;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .result-card {
        background-color: #1C2333;
        border: 1px solid #2E3648;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .rt-number {
        font-size: 3rem;
        font-weight: 800;
        color: #F96167;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

badge_colors = {
    "Baseline": "#2E7D32",
    "Light":    "#F9A825",
    "Noise":    "#1565C0",
    "Both":     "#C62828"
}

st.markdown('<div class="big-title">🧠 Cognitive Environment Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Predicts reaction time on an N-back cognitive task from light and noise conditions. '
    'Model: Random Forest Regressor.</div>',
    unsafe_allow_html=True
)

with st.expander("ℹ️ About this data (click to read)"):
    st.write(
        "Trained on a literature-informed **synthetic** dataset. The real IMPACT "
        "dataset (Markov et al., 2024) was request-only and not accessible within "
        "the project timeline. Effect directions (noise raises reaction time; light "
        "has a U-shaped effect around ~500 lux) are based on published research. "
        "See README.md for full disclosure."
    )

st.subheader("Environmental Conditions")

st.write("**Quick presets:**")
preset_cols = st.columns(4)
preset_values = {
    "Baseline": (480, 38),
    "Light":    (1100, 38),
    "Noise":    (480, 68),
    "Both":     (1100, 68),
}

if "light_lux" not in st.session_state:
    st.session_state.light_lux = 500
    st.session_state.noise_db = 40

for col, (label, (lux, db)) in zip(preset_cols, preset_values.items()):
    if col.button(label, use_container_width=True):
        st.session_state.light_lux = lux
        st.session_state.noise_db = db

col1, col2 = st.columns(2)
with col1:
    light_lux = st.slider("Light condition (lux)", min_value=350, max_value=1400,
                           value=st.session_state.light_lux, step=10, key="light_lux")
with col2:
    noise_db = st.slider("Noise condition (dB)", min_value=25, max_value=85,
                          value=st.session_state.noise_db, step=1, key="noise_db")

n_back_level = st.selectbox("Cognitive task difficulty (N-back level)", options=[1, 2], index=0)

predict_clicked = st.button("PREDICT", type="primary", use_container_width=True)

if predict_clicked:
    predicted_rt, inferred_condition = predict_reaction_time(
        light_lux=light_lux,
        noise_db=noise_db,
        n_back_level=n_back_level,
        model_path=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "best_model_random_forest.pkl"
        )
    )

    color = badge_colors.get(inferred_condition, "#555")

    st.markdown(f"""
    <div class="result-card">
        <div style="color:#9CA3AF; font-size:0.9rem;">PREDICTED REACTION TIME</div>
        <div class="rt-number">{predicted_rt} ms</div>
        <span class="badge" style="background-color:{color}; color:white;">
            Condition: {inferred_condition}
        </span>
        <div style="margin-top:0.75rem; color:#9CA3AF; font-size:0.85rem;">
            Model: Random Forest Regressor (R² = 0.846 on held-out test data)
        </div>
    </div>
    """, unsafe_allow_html=True)
    # --- Comparison bar chart against typical condition averages ---
    import matplotlib.pyplot as plt

    baseline_avgs = {"Baseline": 590, "Light": 891, "Noise": 675, "Both": 993}
    labels = list(baseline_avgs.keys()) + ["Your Prediction"]
    values = list(baseline_avgs.values()) + [predicted_rt]
    colors = ["#4A90D9", "#4A90D9", "#4A90D9", "#4A90D9", "#F96167"]

    st.write("**How this compares to typical condition averages (training data):**")

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")
    bars = ax.bar(labels, values, color=colors, width=0.6)
    ax.set_ylabel("Reaction Time (ms)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#2E3648")
    ax.bar_label(bars, fmt='%.0f', color="white", padding=3)
    plt.tight_layout()
    st.pyplot(fig)
