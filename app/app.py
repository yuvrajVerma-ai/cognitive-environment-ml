import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import predict_reaction_time

st.set_page_config(page_title="Cognitive Environment Predictor", page_icon=None)

st.title("COGNITIVE ENVIRONMENT PREDICTOR")
st.write(
    "Predicts reaction time on an N-back cognitive task, given light and "
    "noise conditions. Model: Random Forest Regressor."
)
st.caption(
    "Note: trained on a literature-informed synthetic dataset (real IMPACT "
    "dataset was not accessible within the project timeline)."
)

st.subheader("Environmental Conditions")

light_lux = st.slider("Light condition (lux)", min_value=350, max_value=1400, value=500, step=10)
noise_db = st.slider("Noise condition (dB)", min_value=25, max_value=85, value=40, step=1)
n_back_level = st.selectbox("Cognitive task difficulty (N-back level)", options=[1, 2], index=0)

if st.button("PREDICT"):
    predicted_rt, inferred_condition = predict_reaction_time(
        light_lux=light_lux,
        noise_db=noise_db,
        n_back_level=n_back_level,
        model_path=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "best_model_random_forest.pkl"
        )
    )

    st.subheader("Prediction")
    st.metric(label="Predicted reaction time", value=f"{predicted_rt} ms")
    st.write(f"Inferred context state: **{inferred_condition}**")
    st.write("Model: Random Forest Regressor (best-performing model, see model_comparison.csv)")
