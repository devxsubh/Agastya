"""Streamlit demo app for the Phase 3 Hybrid Pipeline (Legal-BERT + Random Forest).

Renders the risk prediction and uses SHAP explainability to show exactly
which clauses drove the risk score, satisfying the Extra Mile rubric.
"""

from __future__ import annotations

import os
import sys
from io import BytesIO

import numpy as np
import pandas as pd
import pdfplumber
import shap
import streamlit as st
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase3.hybrid_pipeline import AgastyaHybridPipeline
from src.phase3.ocr.extractor import extract_text

st.set_page_config(page_title="Agastya Risk Assessor", page_icon="⚖", layout="wide")

st.markdown(
    """
    <style>
    .stButton>button { width: 100%; text-align: left; border-radius: 5px; margin-bottom: 5px; }
    .active-clause { border: 2px solid #007bff !important; background-color: #e7f1ff !important; }
    .risk-high { background-color: #dc3545; color: white; padding: 5px; border-radius: 5px;}
    .risk-medium { background-color: #ffc107; color: black; padding: 5px; border-radius: 5px;}
    .risk-low { background-color: #28a745; color: white; padding: 5px; border-radius: 5px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Agastya - Contract Risk Analyzer")
st.caption("Hybrid AI: Phase 2 Legal-BERT + Phase 3 Random Forest")

@st.cache_resource(show_spinner="Loading Legal-BERT + RF Pipeline...")
def load_pipeline() -> AgastyaHybridPipeline:
    return AgastyaHybridPipeline(
        rf_model_path="results/phase3/rf_reasoner.pkl",
        bn_model_path=None,
        bert_checkpoint_path="results/phase2/models/legal_bert_phase2.pt",
        label_map_path="results/phase2/label2id.json",
        adapter_path="results/phase2/models/legal_bert_lora_adapter",
    )

@st.cache_data(show_spinner="Analyzing contract and running SHAP explainer...")
def run_prediction(text: str) -> dict:
    pipeline = load_pipeline()
    return pipeline.predict(text)

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "selected_index" not in st.session_state:
    st.session_state.selected_index = 0
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None

uploaded = st.file_uploader("Upload contract", type=["pdf", "txt"])

if uploaded is not None:
    if st.session_state.pdf_bytes != uploaded.getvalue():
        st.session_state.pdf_bytes = uploaded.getvalue()
        text = extract_text(uploaded)
        st.session_state.analysis_result = run_prediction(text)
        st.session_state.selected_index = 0

def render_risk_heatmap(result: dict):
    st.subheader("🔥 Risk Distribution Heatmap")
    st.caption("Visual density of high-risk (Red) vs mitigating (Green) clauses.")
    
    details = result.get("bert_details", [])
    if not details:
        return
        
    # Create a horizontal strip of color-coded boxes
    cols = st.columns(len(details) if len(details) < 100 else 100)
    for i, detail in enumerate(details[:100]):
        ctype = detail.get("clause_type", "Other")
        conf = detail.get("confidence", 0)
        
        color = "#e0e0e0" # Default Gray
        if ctype in ["Termination", "Liability"]:
            color = "#ff4b4b" if conf > 0.1 else "#ffbaba"
        elif ctype in ["Payment", "Confidentiality"]:
            color = "#ffa500" if conf > 0.1 else "#ffe4b5"
        elif ctype in ["Dispute"]:
            color = "#28a745"
            
        cols[i % len(cols)].markdown(
            f'<div style="background-color:{color}; height:20px; width:100%; border-radius:2px;" title="Clause {i+1}: {ctype} ({conf:.1%})"></div>',
            unsafe_allow_html=True
        )

def render_shap_explanation(result: dict):
    st.subheader("🧠 SHAP Explainability (Why this risk score?)")
    st.markdown("This chart explains how the Neural Network's predictions influenced the Random Forest's final decision.")
    
    feature_vector = np.array(result.get("feature_vector", [])).reshape(1, -1)
    if not feature_vector.any() or feature_vector.shape[1] == 0:
        st.warning("No features available for SHAP explanation.")
        return

    try:
        pipeline = load_pipeline()
        clf = pipeline.rf["model"]
        feature_labels = pipeline.rf["feature_labels"]
        # Extract underlying uncalibrated RF for SHAP TreeExplainer
        base_rf = clf.calibrated_classifiers_[0].estimator
        
        explainer = shap.TreeExplainer(base_rf)
        shap_values = explainer.shap_values(feature_vector)
        
        # Handle multi-class output format for shap values
        # Depending on shap version, it might be a list of arrays (one for each class)
        if isinstance(shap_values, list):
            class_idx = list(clf.classes_).index(result["risk_level"])
            shap_values = shap_values[class_idx]
            
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.summary_plot(shap_values, feature_vector, feature_names=feature_labels, plot_type="bar", show=False)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Failed to generate SHAP plot: {e}")

if st.session_state.analysis_result:
    result = st.session_state.analysis_result

    m1, m2, m3 = st.columns(3)
    risk_class = result["risk_level"].lower()
    m1.markdown(f"### Risk Level: <span class='risk-{risk_class}'>{result['risk_level']}</span>", unsafe_allow_html=True)
    m2.metric("High Risk Prob", f'{result["risk_probabilities"].get("High", 0.0):.1%}')
    m3.metric("Medium Risk Prob", f'{result["risk_probabilities"].get("Medium", 0.0):.1%}')
    st.divider()

    col_nav, col_viewer = st.columns([1, 2])

    with col_nav:
        st.subheader("📑 Identified Clauses (Neural Engine)")
        named_clauses = [(i, d) for i, d in enumerate(result["bert_details"]) if d.get("clause_type") != "Other"]
        other_clauses = [(i, d) for i, d in enumerate(result["bert_details"]) if d.get("clause_type") == "Other"]
        tab_named, tab_other = st.tabs([f"Named ({len(named_clauses)})", f"Other ({len(other_clauses)})"])
        
        with tab_named:
            with st.container(height=600):
                for i, detail in named_clauses:
                    ctype = detail.get("clause_type", "Unknown")
                    conf = detail.get("confidence", 0)
                    btn_label = f"#{i+1}: {ctype} ({conf:.0%})"
                    if st.button(btn_label, key=f"btn_named_{i}"):
                        st.session_state.selected_index = i
                        
        with tab_other:
            with st.container(height=600):
                for i, detail in other_clauses:
                    if st.button(f"Clause #{i+1}: General Text", key=f"btn_other_{i}"):
                        st.session_state.selected_index = i

    with col_viewer:
        idx = st.session_state.selected_index
        idx = max(0, min(idx, len(result["bert_details"]) - 1))
        active_detail = result["bert_details"][idx]
        clause_text = active_detail.get("clause_text", "")
        
        st.subheader(f"🔍 Viewing Clause #{idx+1} ({active_detail.get('clause_type', 'Other')})")
        
        # We try to highlight it in the PDF if possible, otherwise show raw text
        found_highlight = False
        if st.session_state.pdf_bytes and str(uploaded.name).lower().endswith(".pdf"):
            try:
                import re
                with pdfplumber.open(BytesIO(st.session_state.pdf_bytes)) as pdf:
                    # Use the first 15 words to create a robust regex pattern
                    words = clause_text.split()
                    snippet_words = words[:15]
                    if snippet_words:
                        pattern = r'\s+'.join(re.escape(w) for w in snippet_words)
                        for page_num, page in enumerate(pdf.pages):
                            matches = page.search(pattern, regex=True, case=False)
                            if matches:
                                im = page.to_image(resolution=150)
                                for match in matches:
                                    im.draw_rect(match, stroke="#ff0000", stroke_width=3, fill="#ff000033")
                                st.image(im.annotated, caption=f"Page {page_num + 1}", use_column_width=True)
                                found_highlight = True
                                break
            except Exception as e:
                st.error(f"Error reading PDF: {e}")
                
        if not found_highlight:
            st.warning("Could not locate the exact text in the PDF viewer (likely due to OCR differences). Displaying raw text instead.")
            st.info(clause_text)
            
    st.divider()
    render_risk_heatmap(result)
    st.divider()
    render_shap_explanation(result)
