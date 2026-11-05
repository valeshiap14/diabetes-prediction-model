"""Diabetes Prediction Studio — portfolio demonstration, not clinical software.

Launch from repository root: streamlit run app/main.py
The dashboard refuses to invent predictions or evaluation statistics when
training artifacts have not yet been created by the notebook.
"""
from __future__ import annotations

import sys
from pathlib import Path
import json

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core import make_patient_row, load_metadata, predict, metric_rows
from app.explain import explain_prediction

MODEL_PATH = ROOT / "output" / "best_diabetes_pipeline.joblib"
META_PATH = ROOT / "output" / "model_metadata.json"

st.set_page_config(
    page_title="Diabetes Prediction Studio | ML Research",
    layout="wide", initial_sidebar_state="expanded",
)
st.markdown("""
<style>
.block-container {padding-top: 1.8rem; max-width: 1280px;}
[data-testid="stMetric"] {background: #f3f7fa; border: 1px solid #dde8ef; border-radius: 14px; padding: 15px;}
@media (prefers-color-scheme: dark) {[data-testid="stMetric"] {background: #172332; border-color:#344459;}}
.hero {padding: 1.3rem 1.5rem; border-radius: 18px; background: linear-gradient(125deg,#083c5c,#126e7f); color:white; margin-bottom: 1.1rem;}
.hero h1 {color:white; font-size:2rem; margin:0 0 0.35rem 0;}
.hero p {color:#e5f4f6; margin:0;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_artifacts(model_path: str, meta_path: str, model_mtime: float, meta_mtime: float):
    """Only load a trusted model built locally by this project.

    Joblib uses pickle semantics: never load untrusted downloaded artifacts.
    """
    meta = load_metadata(meta_path)
    pipeline = joblib.load(model_path)
    if not hasattr(pipeline, "predict_proba") or not hasattr(pipeline, "named_steps"):
        raise ValueError("Artifact is not a compatible sklearn/imblearn pipeline")
    return pipeline, meta

@st.cache_data(show_spinner=False)
def load_reference_metrics():
    # This deliberately reads no patient-level dataset or confidential records.
    return None

st.markdown('<div class="hero"><h1>Diabetes Prediction Studio</h1><p>Explainable machine learning • reproducible evaluation • research demonstration</p></div>', unsafe_allow_html=True)
st.caption("Educational portfolio application only. Not a medical device; not a diagnosis or an estimate of clinical risk. Use synthetic records only.")

ready = MODEL_PATH.is_file() and META_PATH.is_file()
model = metadata = None
if ready:
    try:
        model, metadata = load_artifacts(str(MODEL_PATH), str(META_PATH), MODEL_PATH.stat().st_mtime, META_PATH.stat().st_mtime)
    except Exception as exc:
        st.error(f"Saved model could not be loaded: {exc}")
        ready = False
else:
    st.warning("**Model not trained yet.** Run `diabetes_prediction_improved.ipynb` with your dataset to create `output/best_diabetes_pipeline.joblib` and `output/model_metadata.json`. The app will not manufacture predictions or performance scores.")

with st.sidebar:
    st.subheader("Project controls")
    st.write("**Research status:** " + ("Model loaded" if ready else "Training required"))
    if ready:
        st.metric("Selected threshold", f"{float(metadata['decision_threshold']):.3f}")
        st.caption(f"Model: {metadata.get('model', 'Not recorded')}")
        st.caption(f"CV folds: {metadata.get('cv_folds', 'Not recorded')}")
    st.divider()
    st.info("Model output represents a dataset-label classification, not a clinical diagnosis. High HbA1c and glucose are already related to diagnostic assessment.")
    st.caption("For demonstrations, do not enter identifiable real patient information.")

prediction_tab, explain_tab, performance_tab, monitor_tab, about_tab = st.tabs([
    "Prediction", "Explainability", "Performance", "Monitoring", "About"
])

with prediction_tab:
    st.subheader("Explore a fictional patient record")
    st.write("Values are prefilled as a synthetic example. The app uses your fitted model only when you select **Run research prediction**.")
    with st.form("patient_form"):
        a, b = st.columns(2)
        with a:
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=52)
            bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, value=31.8, step=0.1)
            a1c = st.number_input("HbA1c (%)", min_value=3.0, max_value=20.0, value=6.7, step=0.1)
            glucose = st.number_input("Blood glucose (mg/dL)", min_value=40, max_value=500, value=180)
        with b:
            gender = st.selectbox("Gender (as recorded in dataset)", ["Female", "Male", "Other"])
            smoking = st.selectbox("Smoking history", ["never", "former", "current", "not current", "ever", "No Info"])
            hypertension = st.selectbox("Hypertension history", ["No", "Yes"])
            heart_disease = st.selectbox("Heart disease history", ["No", "Yes"])
        submitted = st.form_submit_button("Run research prediction", type="primary", disabled=not ready, use_container_width=True)
    if submitted and ready:
        inputs = {
            "age": age, "bmi": bmi, "HbA1c_level": a1c, "blood_glucose_level": glucose,
            "gender": gender, "smoking_history": smoking,
            "hypertension": int(hypertension == "Yes"), "heart_disease": int(heart_disease == "Yes"),
        }
        try:
            row = make_patient_row(inputs)
            result = predict(model, row, float(metadata["decision_threshold"]))
            st.session_state["last_row"] = row
            st.session_state["last_prediction"] = result
            # Session-only counts, no stored record values or identifiers.
            st.session_state["n_predictions"] = st.session_state.get("n_predictions", 0) + 1
            st.session_state["n_positive"] = st.session_state.get("n_positive", 0) + result.label
        except Exception as exc:
            st.error(f"Unable to generate a prediction: {exc}")
    if "last_prediction" in st.session_state and ready:
        result = st.session_state["last_prediction"]
        st.divider()
        left, right = st.columns([1, 1.2])
        with left:
            st.metric("Model probability for dataset label 1", f"{result.probability:.1%}")
            st.metric("Research classification", "Positive class (1)" if result.label else "Negative class (0)")
            st.caption(f"Frozen validation-selected threshold: {result.threshold:.3f}")
        with right:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=result.probability * 100,
                number={"suffix": "%", "font": {"size": 36}},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#087e8b"},
                       "threshold": {"line": {"color": "#db7a32", "width": 4},
                                     "value": result.threshold * 100}},
                title={"text": "Algorithmic score (not calibrated clinical risk)"},
            ))
            fig.update_layout(height=310, margin=dict(l=20, r=20, t=55, b=20))
            st.plotly_chart(fig, use_container_width=True)
        st.info("This output describes a statistical classification against this dataset's labels. It is not a diagnostic opinion, and the probability may not be clinically calibrated.")

with explain_tab:
    st.subheader("Why did the model classify this record?")
    if not ready or "last_row" not in st.session_state:
        st.info("Train the model and run a fictional prediction to view SHAP explanations.")
    else:
        try:
            explanation = explain_prediction(model, st.session_state["last_row"])
            top = explanation.head(12).sort_values("Contribution")
            fig = go.Figure(go.Bar(
                x=top["Contribution"], y=top["Feature"], orientation="h",
                marker_color=["#087e8b" if x >= 0 else "#e38b53" for x in top["Contribution"]],
            ))
            fig.update_layout(height=465, margin=dict(l=20, r=15, t=20, b=15),
                              xaxis_title="SHAP contribution (model output units)", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Positive/negative SHAP contributions describe the model's output direction, not changes in diabetes risk caused by a feature. Values may be on a log-odds scale.")
            with st.expander("Show feature contributions as a table"):
                st.dataframe(explanation, hide_index=True, use_container_width=True)
        except Exception as exc:
            st.warning(f"SHAP explanation unavailable for this saved model or installed SHAP version: {exc}")
            st.info("The prediction is still available. SHAP explanations are optional and do not influence the model's output.")

with performance_tab:
    st.subheader("Verified holdout evaluation")
    if not ready:
        st.info("No current-run metrics are available. Train the model before viewing evaluation statistics.")
    else:
        metrics = metric_rows(metadata)
        if metrics:
            cols = st.columns(3)
            for i, (key, value) in enumerate(metrics):
                fmt = f"{value:.1%}" if key not in ("Brier score (lower is better)",) else f"{value:.4f}"
                with cols[i % 3]:
                    st.metric(key, fmt)
            st.caption("Figures are loaded from the saved `model_metadata.json` created by the improved training notebook; they are not live clinical performance metrics.")
            with st.expander("Evaluation provenance"):
                st.json({k: metadata.get(k) for k in ["model", "selection_metric", "cv_folds", "dataset_rows_after_cleaning", "train_rows", "holdout_rows", "validation_recall_target_research_only"]})
        else:
            st.warning("No holdout metrics are recorded in this artifact's metadata.")
        st.info("The previous PDF's 96.85% accuracy is from the *original* experiment. Only metrics saved by the revised training notebook are displayed here.")

with monitor_tab:
    st.subheader("Local session monitoring")
    st.caption("This view intentionally records only aggregate counts for the current browser session; it does not store patient records.")
    num = int(st.session_state.get("n_predictions", 0))
    positives = int(st.session_state.get("n_positive", 0))
    x, y, z = st.columns(3)
    x.metric("Session predictions", num)
    y.metric("Positive classifications", positives)
    z.metric("Negative classifications", num - positives)
    if num:
        fig = go.Figure(go.Pie(labels=["Positive", "Negative"], values=[positives, num-positives], hole=.65,
                               marker_colors=["#db7a32", "#087e8b"]))
        fig.update_layout(height=290, margin=dict(l=10, r=10, t=15, b=15), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    st.warning("**Data drift is not measured yet.** Proper drift analysis needs a defined reference distribution, sufficient live input data, privacy controls and a monitoring policy. A few manually entered examples are not evidence of drift or predictive performance.")

with about_tab:
    st.subheader("Research methodology and limitations")
    st.markdown("""
- **Problem:** classify a diabetes label using demographic and clinical features, including HbA1c and blood glucose.
- **Training:** three candidate algorithms, cross-validation on development data, model selection using mean ROC-AUC, threshold selection on validation data, final holdout evaluation.
- **Interpretability:** SHAP is provided for supported fitted tree models. SHAP explains model attribution, not causation.
- **Scope:** synthetic records only. The project has not been clinically or externally validated and should not guide treatment.
- **Provenance:** add the dataset's source, license, label definition, collection setting and sampling limitations before distributing the dataset.
- **Privacy:** do not deploy an application that invites submission of real patient data without an appropriate privacy and security assessment.
""")
    st.caption("Source: the improved training notebook and its current serialized output, when available.")
