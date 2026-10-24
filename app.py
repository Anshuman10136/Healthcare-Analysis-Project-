"""
NeuroExplain AI: Interactive Clinical Decision Support & Explainability Dashboard
for Cognitive Impairment Prediction using OASIS-1 Clinical and Neuroimaging Features.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap

from src.config import (
    MODELS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    CLASS_NAMES,
    CDR_MAPPING
)
from src.data_loader import load_clinical_cohort
from src.feature_engineering import engineer_oasis_features
from src.explainability import ShapExplainerManager

# Page Configuration
st.set_page_config(
    page_title="NeuroExplain AI | OASIS-1 Cognitive Impairment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1e3d59;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4b6584;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f7f9fc;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1e3d59;
        margin-bottom: 10px;
    }
    .badge-normal {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 10px;
        border-radius: 5px;
        font-weight: 600;
    }
    .badge-impaired {
        background-color: #f8d7da;
        color: #721c24;
        padding: 4px 10px;
        border-radius: 5px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load clinical cohort and engineered features."""
    df_raw = load_clinical_cohort()
    df_eng = engineer_oasis_features(df_raw)
    return df_raw, df_eng


@st.cache_resource
def load_model_pipeline():
    """Load serialized pipeline, models, and metadata."""
    pipeline = joblib.load(MODELS_DIR / "best_model_pipeline.joblib")
    champion_model = joblib.load(MODELS_DIR / "champion_model.joblib")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    
    with open(MODELS_DIR / "feature_names.json", "r") as f:
        feature_names = json.load(f)
        
    with open(MODELS_DIR / "optuna_best_params.json", "r") as f:
        optuna_params = json.load(f)
        
    benchmark_df = pd.read_csv(REPORTS_DIR / "benchmark_summary.csv")
    return pipeline, champion_model, preprocessor, feature_names, optuna_params, benchmark_df


df_raw, df_eng = load_data()
pipeline, champion_model, preprocessor, selected_features, optuna_params, benchmark_df = load_model_pipeline()

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/brain.png", width=70)
st.sidebar.title("NeuroExplain AI")
st.sidebar.caption("Multimodal Machine Learning & SHAP Decision Support")
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Cohort:** OASIS-1 Cross-Sectional MRI")
st.sidebar.markdown(f"**Sample Size:** N = {len(df_raw)} clinically assessed")
st.sidebar.markdown(f"**Impairment Prevalence:** {df_raw['Impaired'].mean()*100:.1f}%")
st.sidebar.markdown(f"**Champion Model:** Optuna LightGBM (AUC: 0.919)")
st.sidebar.markdown("---")
st.sidebar.info("💡 **Clinical Relevance:** Early identification of Mild Cognitive Impairment (MCI) and Alzheimer's disease using non-invasive structural MRI volumetry and bedside cognitive testing.")

# Main Header
st.markdown('<div class="main-title">🧠 NeuroExplain AI: Cognitive Impairment Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Explainable Machine Learning integrating Clinical Demographics, Cognitive Testing (MMSE), and Neuroimaging Biomarkers (OASIS-1)</div>', unsafe_allow_html=True)

# Top KPI row
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric(label="Evaluated Cohort", value=f"{len(df_raw)} Patients", delta="OASIS-1 Baseline")
with kpi2:
    st.metric(label="Champion 5-Fold ROC-AUC", value="0.919 ± 0.035", delta="LightGBM")
with kpi3:
    st.metric(label="Diagnostic Specificity", value="89.6%", delta="Controlled false alarms")
with kpi4:
    st.metric(label="Precision-Recall AUC", value="0.909 ± 0.042", delta="PR-AUC")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Cohort & Neuroimaging Explorer",
    "🧬 Feature Engineering & Selection",
    "🚀 Model Benchmarking & Optuna Tuning",
    "🧠 Global SHAP Interpretability",
    "🩺 Patient Diagnostic Simulator (Local SHAP)"
])

# ==============================================================================
# TAB 1: COHORT EXPLORER
# ==============================================================================
with tab1:
    st.subheader("OASIS-1 Clinical and Neuroimaging Cohort Characterization")
    st.markdown("""
    The **Open Access Series of Imaging Studies (OASIS-1)** cross-sectional collection contains MRI sessions
    from healthy individuals and subjects with very mild to moderate Alzheimer's Disease (AD).
    """)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### Clinical Dementia Rating (CDR) Distribution")
        cdr_counts = df_raw["CDR"].value_counts().reset_index()
        cdr_counts.columns = ["CDR", "Count"]
        cdr_counts["Stage"] = cdr_counts["CDR"].map(CDR_MAPPING)
        fig_cdr = px.bar(
            cdr_counts, x="Stage", y="Count", color="Stage",
            text="Count", color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_cdr.update_traces(textposition='outside')
        fig_cdr.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_cdr, use_container_width=True)

    with col2:
        st.markdown("#### Multi-dimensional Interactive Scatter")
        fig_scatter = px.scatter(
            df_eng,
            x="nWBV",
            y="MMSE",
            color="CDR_Label",
            size="Age",
            hover_data=["ID", "Educ", "SES", "eTIV"],
            labels={
                "nWBV": "Normalized Whole Brain Volume (nWBV)",
                "MMSE": "Mini-Mental State Exam Score (MMSE)",
                "CDR_Label": "Diagnosis"
            },
            color_discrete_sequence=["#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
        )
        fig_scatter.update_layout(height=350)
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Diagnostic Stratification Visualizations")
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        dist_path = FIGURES_DIR / "eda_distributions.png"
        if os.path.exists(dist_path):
            st.image(str(dist_path), caption="Feature Distributions: Normal (Green) vs Impaired (Red)", use_container_width=True)
    with col_img2:
        corr_path = FIGURES_DIR / "eda_correlations.png"
        if os.path.exists(corr_path):
            st.image(str(corr_path), caption="Pearson Correlation Heatmap (Clinical + Volumetric)", use_container_width=True)

# ==============================================================================
# TAB 2: FEATURE ENGINEERING & SELECTION
# ==============================================================================
with tab2:
    st.subheader("Domain-Specific Feature Engineering & Consensus Selection")
    st.markdown("""
    To bridge clinical neurology with neuroimaging volumetry, we engineered 7 domain-informed interaction features:
    """)

    feat_desc = [
        {"Feature": "Brain_Atrophy_Index", "Formula": "1.0 - nWBV", "Clinical Significance": "Direct measure of cerebral parenchyma loss relative to total intracranial capacity."},
        {"Feature": "Estimated_Brain_Volume", "Formula": "eTIV × nWBV", "Clinical Significance": "Approximation of absolute brain parenchymal volume in cubic millimeters (mm³)."},
        {"Feature": "Cranial_Size_Ratio", "Formula": "ASF × eTIV", "Clinical Significance": "Atlas scaling factor calibration reflecting head size normalisation across subjects."},
        {"Feature": "Age_Adjusted_Atrophy", "Formula": "Age × (1.0 - nWBV)", "Clinical Significance": "Pathological neurodegeneration accelerating beyond typical chronological senescence."},
        {"Feature": "MMSE_Deficit", "Formula": "30.0 - MMSE", "Clinical Significance": "Absolute cognitive deviation from ceiling neurocognitive performance."},
        {"Feature": "Cognitive_Reserve_Index", "Formula": "Educ / SES", "Clinical Significance": "Educational and occupational buffer mitigating clinical expression of pathology."},
        {"Feature": "Neuro_Cognitive_Ratio", "Formula": "nWBV × MMSE", "Clinical Significance": "Coupled functional-structural biomarker quantifying combined cognitive & anatomical integrity."}
    ]
    st.table(pd.DataFrame(feat_desc))

    st.markdown("---")
    st.markdown("#### Feature Selection Methodology Comparison (Mutual Information, LASSO, RF, RFECV)")
    fs_path = FIGURES_DIR / "feature_selection_comparison.png"
    if os.path.exists(fs_path):
        st.image(str(fs_path), caption="Consensus Feature Importance & Recursive Feature Elimination (RFECV) Curve", use_container_width=True)

    st.markdown(f"**Identified Optimal Feature Set ({len(selected_features)} features):** `{', '.join(selected_features)}`")

# ==============================================================================
# TAB 3: MODEL BENCHMARKING & OPTUNA TUNING
# ==============================================================================
with tab3:
    st.subheader("Model Benchmarking on Stratified 5-Fold Cross Validation")
    st.markdown("""
    All models were evaluated using rigorous out-of-fold cross-validation with fold-isolated preprocessing
    (imputation and scaling) to eliminate data leakage.
    """)

    # Benchmark Table
    st.dataframe(
        benchmark_df[["Model", "ROC_AUC", "PR_AUC", "Balanced_Acc", "Sensitivity", "Specificity", "F1_Score", "MCC"]],
        use_container_width=True
    )

    col_curv1, col_curv2 = st.columns(2)
    with col_curv1:
        roc_path = FIGURES_DIR / "roc_pr_curves_comparison.png"
        if os.path.exists(roc_path):
            st.image(str(roc_path), caption="ROC and Precision-Recall Curves across candidate models", use_container_width=True)
    with col_curv2:
        cm_path = FIGURES_DIR / "confusion_matrices.png"
        if os.path.exists(cm_path):
            st.image(str(cm_path), caption="Out-of-Fold Normalized Confusion Matrices", use_container_width=True)

    st.markdown("---")
    st.markdown("#### Optuna Bayesian Hyperparameter Optimization Results")
    st.markdown("Optuna Tree-Structured Parzen Estimator (TPE) discovered the following optimal parameters:")
    
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        with st.expander("🌲 Best LightGBM Parameters", expanded=True):
            st.json(optuna_params.get("LightGBM", {}))
        with st.expander("⚡ Best XGBoost Parameters"):
            st.json(optuna_params.get("XGBoost", {}))
    with col_opt2:
        with st.expander("🌳 Best Random Forest Parameters"):
            st.json(optuna_params.get("RandomForest", {}))
        with st.expander("🎯 Best SVM Parameters"):
            st.json(optuna_params.get("SVM", {}))

# ==============================================================================
# TAB 4: GLOBAL SHAP INTERPRETABILITY
# ==============================================================================
with tab4:
    st.subheader("Global Explainable AI (SHAP) Interpretation")
    st.markdown("""
    **SHapley Additive exPlanations (SHAP)** attributes game-theoretic Shapley values to each biomarker,
    explaining not just *which* features matter, but *how* their values push risk up or down.
    """)

    col_sh1, col_sh2 = st.columns([1.2, 1])
    with col_sh1:
        st.markdown("#### Global SHAP Summary (Beeswarm Plot)")
        st.markdown("Points colored **red** indicate high feature values; **blue** indicate low values. Points to the right increase impairment probability.")
        beeswarm_path = FIGURES_DIR / "shap_summary_beeswarm.png"
        if os.path.exists(beeswarm_path):
            st.image(str(beeswarm_path), use_container_width=True)
    with col_sh2:
        st.markdown("#### Global Mean Absolute SHAP Ranking")
        st.markdown("Average magnitude of each feature's impact across the entire OASIS-1 patient cohort.")
        bar_path = FIGURES_DIR / "shap_bar_importance.png"
        if os.path.exists(bar_path):
            st.image(str(bar_path), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Non-Linear Biomarker Dependence & Risk Thresholds")
    dep_cols = st.columns(3)
    with dep_cols[0]:
        dep_mmse = FIGURES_DIR / "shap_dependence_MMSE.png"
        if os.path.exists(dep_mmse):
            st.image(str(dep_mmse), caption="MMSE vs SHAP Risk Contribution", use_container_width=True)
    with dep_cols[1]:
        dep_nwbv = FIGURES_DIR / "shap_dependence_nWBV.png"
        if os.path.exists(dep_nwbv):
            st.image(str(dep_nwbv), caption="nWBV vs SHAP Risk Contribution", use_container_width=True)
    with dep_cols[2]:
        dep_age = FIGURES_DIR / "shap_dependence_Age.png"
        if os.path.exists(dep_age):
            st.image(str(dep_age), caption="Age vs SHAP Risk Contribution", use_container_width=True)

# ==============================================================================
# TAB 5: PATIENT CLINICAL SIMULATOR & LOCAL SHAP
# ==============================================================================
with tab5:
    st.subheader("Interactive Patient Clinical Decision Support (CDSS)")
    st.markdown("""
    Simulate a patient presentation or select an existing OASIS-1 subject to view real-time diagnostic probabilities,
    local SHAP waterfall decompositions, and natural language clinical insights.
    """)

    sim_mode = st.radio("Patient Selection Mode:", ["Select from OASIS-1 Cohort", "Custom Clinician Input"], horizontal=True)

    if sim_mode == "Select from OASIS-1 Cohort":
        patient_options = df_raw["ID"].tolist()
        selected_id = st.selectbox("Select Patient Subject ID:", patient_options, index=2) # OAS1_0003
        pat_raw_row = df_raw[df_raw["ID"] == selected_id].iloc[0]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.write(f"**Age:** {pat_raw_row['Age']} yrs")
        c2.write(f"**Sex:** {'Male' if pat_raw_row['M/F'] == 'M' else 'Female'}")
        c3.write(f"**MMSE:** {pat_raw_row['MMSE']} / 30")
        c4.write(f"**True Diagnosis:** {pat_raw_row['CDR_Label']}")
        
        # Extract features for this patient
        pat_eng_row = df_eng[df_eng["ID"] == selected_id].iloc[0]
        input_data = pd.DataFrame([pat_eng_row[selected_features]])
        true_cdr = pat_raw_row["CDR_Label"]
        subject_label = selected_id

    else:
        st.markdown("#### Patient Demographics, Cognitive Testing & MRI Volumetry")
        col_in1, col_in2, col_in3 = st.columns(3)
        with col_in1:
            in_age = st.slider("Patient Age (years)", 60, 95, 74)
            in_sex = st.selectbox("Patient Sex", ["Female", "Male"])
            in_educ = st.slider("Education Level (1: Less than HS, 5: Post-Grad)", 1.0, 5.0, 3.0, 1.0)
        with col_in2:
            in_ses = st.slider("Socioeconomic Status (1: Highest, 5: Lowest)", 1.0, 5.0, 2.0, 1.0)
            in_mmse = st.slider("MMSE Score (0 - 30)", 10.0, 30.0, 26.0, 1.0)
            in_etiv = st.slider("eTIV (Total Intracranial Volume, mm³)", 1100, 1900, 1450, 10)
        with col_in3:
            in_nwbv = st.slider("nWBV (Normalized Whole Brain Volume)", 0.650, 0.850, 0.725, 0.005)
            in_asf = st.slider("ASF (Atlas Scaling Factor)", 0.900, 1.600, 1.210, 0.01)

        # Build custom dictionary
        raw_custom = pd.DataFrame([{
            "ID": "CUSTOM_PATIENT",
            "M/F": "M" if in_sex == "Male" else "F",
            "Age": in_age,
            "Educ": in_educ,
            "SES": in_ses,
            "MMSE": in_mmse,
            "eTIV": in_etiv,
            "nWBV": in_nwbv,
            "ASF": in_asf
        }])
        eng_custom = engineer_oasis_features(raw_custom)
        input_data = eng_custom[selected_features]
        true_cdr = "Unlabeled / Simulated"
        subject_label = "Simulated Patient"

    st.markdown("---")
    # Predict with Pipeline
    pred_prob = float(pipeline.predict_proba(input_data)[0, 1])
    is_impaired = pred_prob >= 0.5

    res_col1, res_col2 = st.columns([1, 1.5])
    with res_col1:
        st.markdown("#### Diagnostic Risk Assessment")
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Probability of Cognitive Impairment (%)", 'font': {'size': 18}},
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#d62728" if is_impaired else "#2ca02c"},
                'steps': [
                    {'range': [0, 35], 'color': "#e8f5e9"},
                    {'range': [35, 65], 'color': "#fff3e0"},
                    {'range': [65, 100], 'color': "#ffebee"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        gauge_fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(gauge_fig, use_container_width=True)

        status_class = "badge-impaired" if is_impaired else "badge-normal"
        status_text = "HIGH RISK: Cognitive Impairment Predicted (CDR ≥ 0.5)" if is_impaired else "LOW RISK: Cognitively Normal (CDR 0.0)"
        st.markdown(f"<div style='text-align: center;'><span class='{status_class}'>{status_text}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; color: gray; margin-top: 5px;'>Reference Diagnosis: <b>{true_cdr}</b></p>", unsafe_allow_html=True)

    with res_col2:
        st.markdown("#### Real-time SHAP Waterfall Explanation")
        # Compute exact SHAP values for this input
        proc_input = preprocessor.transform(input_data)
        explainer = shap.TreeExplainer(champion_model)
        shap_res = explainer(proc_input)
        
        # Binary positive class
        if len(shap_res.shape) == 3 and shap_res.shape[2] == 2:
            shap_res_single = shap_res[0, :, 1]
        else:
            shap_res_single = shap_res[0]
            
        shap_res_single.feature_names = selected_features
        shap_res_single.data = np.round(input_data.values[0], 2)

        fig_waterfall, ax_wf = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(shap_res_single, max_display=8, show=False)
        plt.title(f"SHAP Biomarker Breakdown: {subject_label}", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_waterfall)
        plt.close()

    # Clinical EMR Summary Card
    st.markdown("---")
    st.markdown("#### Automated Clinician Electronic Health Record (EHR) Summary Card")
    
    # Extract top pushing drivers
    effects = []
    for fn, sv, dv in zip(selected_features, shap_res_single.values, input_data.values[0]):
        effects.append({"feature": fn, "shap": sv, "val": dv})
    effects_sorted = sorted(effects, key=lambda x: abs(x["shap"]), reverse=True)
    drivers = [e for e in effects_sorted if e["shap"] > 0][:3]
    protective = [e for e in effects_sorted if e["shap"] < 0][:2]

    narrative = f"""
    > **Patient Evaluation Card ({subject_label}):**
    > - **Predicted Cognitive Profile:** **{'Cognitively Impaired / MCI' if is_impaired else 'Cognitively Intact / Normal'}** (Model Confidence: **{pred_prob*100:.1f}%**).
    > - **Primary Biomarkers Elevating Impairment Risk:** {', '.join([f"**{d['feature']}** ({d['val']:.2f}, SHAP: +{d['shap']:.2f})" for d in drivers]) if drivers else "None"}.
    > - **Protective Compensatory Reserve Factors:** {', '.join([f"**{p['feature']}** ({p['val']:.2f}, SHAP: {p['shap']:.2f})" for p in protective]) if protective else "None detected"}.
    > - **Clinical Action Recommendation:** {'Refer to multidisciplinary memory clinic for comprehensive neuropsychological assessment and PET amyloid biomarker confirmation.' if is_impaired else 'Standard longitudinal surveillance; repeat annual cognitive screening.'}
    """
    st.markdown(narrative)
