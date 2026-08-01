"""Streamlit interface for predicting startup success."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "Models" / "startup_success_model.joblib"
METADATA_PATH = ROOT / "Models" / "startup_success_model_metadata.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        # This allows a fresh deployment to start successfully even when the
        # generated artifact was not committed. Subsequent runs load the file.
        from src.train_model import train_and_export

        train_and_export()
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata


def main() -> None:
    st.set_page_config(page_title="Startup Success Predictor", page_icon="🚀")
    st.title("🚀 Startup Success Predictor")
    st.write("Enter a startup's details to estimate its probability of success.")

    try:
        model, metadata = load_assets()
    except Exception as error:
        st.error(str(error))
        st.stop()

    with st.form("prediction_form"):
        left, right = st.columns(2)
        with left:
            company_age = st.number_input("Company age (years)", min_value=0, max_value=150, value=3, step=1)
            funding = st.number_input("Total funding (USD)", min_value=0.0, value=1000000.0, step=10000.0, format="%.2f")
            funding_rounds = st.number_input("Funding rounds", min_value=0, max_value=100, value=1, step=1)
        with right:
            industry = st.selectbox("Industry", metadata["industries"])
            country = st.selectbox("Country", metadata["top_countries"] + ["Other"])
        submitted = st.form_submit_button("Predict success probability", type="primary")

    if submitted:
        input_data = pd.DataFrame(
            [{
                "company_age": company_age,
                "Funding_log": float(np.log1p(funding)),
                "funding_rounds": funding_rounds,
                "Industry_clean": industry,
                "country_code": country,
            }]
        )
        probability = float(model.predict_proba(input_data)[0, 1])
        prediction = int(model.predict(input_data)[0])

        st.subheader("Prediction")
        st.metric("Estimated probability of success", f"{probability:.1%}")
        if prediction == 1:
            st.success("Model classification: likely successful")
        else:
            st.warning("Model classification: higher risk / not likely successful")
        st.caption(
            "This is a statistical estimate from historical data, not investment advice. "
            "The model defines success as an IPO, acquisition, or an operating company at least five years old."
        )

    with st.expander("Model details"):
        metrics = metadata["test_metrics"]
        st.write("Random Forest model evaluated on a held-out 20% test set.")
        st.json(metrics)


if __name__ == "__main__":
    main()
