# 🧠 NeuroExplain AI: Explainable Machine Learning for Cognitive Impairment Prediction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/App-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![SHAP Explainability](https://img.shields.io/badge/Explainability-SHAP%20v0.52-brightgreen.svg)](https://github.com/slundberg/shap)
[![Optuna Tuned](https://img.shields.io/badge/Optimization-Optuna%20Bayesian-blueviolet.svg)](https://optuna.org/)

An explainable, optimized multimodal machine learning framework integrating clinical demographics, bedside cognitive assessments (MMSE), and structural MRI volumetry from the **OASIS-1** (Open Access Series of Imaging Studies) dataset. The pipeline implements domain-driven feature engineering, multi-criteria consensus feature selection, Bayesian hyperparameter optimization via **Optuna**, and game-theoretic **SHAP (SHapley Additive exPlanations)** interpretability at both global cohort and local patient levels.

---

## 📌 Table of Contents
1. [Clinical Background & Problem Formulation](#-clinical-background--problem-formulation)
2. [Dataset Overview (OASIS-1)](#-dataset-overview-oasis-1)
3. [Project Architecture](#-project-architecture)
4. [Domain-Specific Feature Engineering](#-domain-specific-feature-engineering)
5. [Feature Selection Methodology](#-feature-selection-methodology)
6. [Bayesian Hyperparameter Optimization (Optuna)](#-bayesian-hyperparameter-optimization-optuna)
7. [Cross-Validation Model Benchmark Results](#-cross-validation-model-benchmark-results)
8. [Explainable AI (XAI) & SHAP Interpretability](#-explainable-ai-xai--shap-interpretability)
   - [Global Cohort Explanations (Beeswarm & Feature Ranking)](#global-cohort-explanations)
   - [Biomarker Dependence & Risk Thresholds](#biomarker-dependence--risk-thresholds)
   - [Local Patient Decision Waterfall Case Studies](#local-patient-decision-waterfall-case-studies)
9. [Interactive Streamlit Clinical Decision Platform](#-interactive-streamlit-clinical-decision-platform)
10. [Installation & Reproduction Guide](#-installation--reproduction-guide)

---

## 🩺 Clinical Background & Problem Formulation

Cognitive impairment—spanning Mild Cognitive Impairment (MCI) to Alzheimer's Disease (AD)—poses a critical diagnostic challenge in global healthcare. Early diagnosis affords early intervention, lifestyle modification, and clinical trial stratification. 

Traditional clinical screening relies heavily on bedside cognitive questionnaires (e.g., Mini-Mental State Examination, MMSE), which are susceptible to education and cultural biases. Conversely, magnetic resonance imaging (MRI) captures structural macro-anatomical neurodegeneration (such as cerebral atrophy and ventricular enlargement) but can be costly and challenging to interpret quantitatively.

This project formulates **Cognitive Impairment Prediction** as a supervised multimodal machine learning problem:
$$\hat{y} = f(\mathbf{x}_{\text{clinical}}, \mathbf{x}_{\text{MRI}}, \mathbf{x}_{\text{engineered}})$$
where the ground-truth target is derived from the **Clinical Dementia Rating (CDR)**:
- **Normal / Cognitively Intact ($y = 0$):** $\text{CDR} = 0.0$ ($n = 135$)
- **Cognitively Impaired ($y = 1$):** $\text{CDR} \ge 0.5$ ($n = 100$: 70 Very Mild/MCI, 28 Mild AD, 2 Moderate AD)

---

## 📊 Dataset Overview (OASIS-1)

The cross-sectional **OASIS-1** dataset (`oasis_cross-sectional.csv`) includes $N = 436$ total participants. In the cohort, individuals aged 18–59 were young healthy volunteers who did not undergo the standardized dementia clinical evaluation protocol. Following established clinical neuroimaging standards, our primary cohort is restricted to the **$N = 235$ clinically evaluated individuals** with documented CDR scores:

| Demographic / Clinical Parameter | Cohort Value ($N = 235$) | Normal ($CDR = 0.0$, $n=135$) | Impaired ($CDR \ge 0.5$, $n=100$) |
|---|---|---|---|
| **Age (years)** | $72.3 \pm 12.1$ | $69.1 \pm 12.3$ | $76.8 \pm 10.3$ |
| **Sex (Female / Male)** | 156 F / 79 M (66.4% F) | 99 F / 36 M (73.3% F) | 57 F / 43 M (57.0% F) |
| **MMSE Score (0 - 30)** | $27.1 \pm 3.7$ | $29.1 \pm 1.2$ | $24.4 \pm 4.1$ |
| **Normalized Whole Brain Volume (nWBV)** | $0.749 \pm 0.048$ | $0.769 \pm 0.044$ | $0.722 \pm 0.038$ |
| **Estimated Total Intracranial Volume (eTIV)** | $1460.6 \pm 175.8\text{ mm}^3$ | $1441.5 \pm 167.3\text{ mm}^3$ | $1486.5 \pm 184.8\text{ mm}^3$ |
| **Atlas Scaling Factor (ASF)** | $1.218 \pm 0.138$ | $1.231 \pm 0.134$ | $1.200 \pm 0.142$ |
| **Socioeconomic Status (SES)** | Median: 2.0 (19 imputed) | Median: 2.0 | Median: 3.0 |

---

## 🏗 Project Architecture

```
oasis_cognitive_ml/
├── data/
│   ├── oasis_cross-sectional.csv         # Raw OASIS-1 dataset (auto-downloaded)
│   └── oasis_features_engineered.csv     # Extracted & engineered multimodal features
├── src/
│   ├── __init__.py
│   ├── config.py                        # Paths, hyperparameter ranges, clinical schemas
│   ├── data_loader.py                   # Ingestion, validation, clinical cohort filter
│   ├── feature_engineering.py           # Domain-informed neuroimaging & interaction metrics
│   ├── feature_selection.py             # MI, LASSO, Tree Gini, and Stratified RFECV
│   ├── models.py                        # Factory for LightGBM, XGBoost, RF, SVM, MLP, LogReg
│   ├── hyperparameter_tuning.py        # Bayesian optimization engine using Optuna
│   ├── evaluation.py                    # Stratified 5-Fold OOF metrics and curve generator
│   └── explainability.py                # SHAP manager (Beeswarm, Bar, Dependence, Waterfall)
├── models_saved/
│   ├── best_model_pipeline.joblib       # End-to-end serialized inference pipeline
│   ├── champion_model.joblib            # Trained Optuna LightGBM model
│   ├── preprocessor.joblib              # Imputer & scaler preprocessor
│   ├── feature_names.json               # Selected optimal feature names
│   └── optuna_best_params.json          # Best parameters discovered via Optuna
├── reports/
│   ├── benchmark_summary.csv            # 5-fold cross-validation performance table
│   ├── model_benchmark_report.json      # Full fold-by-fold metric serialization
│   └── figures/
│       ├── eda_distributions.png        # Feature distributions by impairment status
│       ├── eda_correlations.png         # Pearson correlation heatmap
│       ├── feature_selection_comparison.png # Consensus ranking & RFECV curve
│       ├── roc_pr_curves_comparison.png # Comparative ROC & PR curves
│       ├── confusion_matrices.png       # Normalized out-of-fold confusion matrices
│       ├── shap_summary_beeswarm.png    # Global SHAP beeswarm plot
│       ├── shap_bar_importance.png      # Global mean absolute SHAP value ranking
│       ├── shap_dependence_MMSE.png     # MMSE risk dependence plot
│       ├── shap_dependence_nWBV.png     # nWBV atrophy dependence plot
│       ├── shap_dependence_Age.png      # Age non-linear risk dependence plot
│       ├── shap_waterfall_normal_case.png # Patient OAS1_0001 local explanation
│       └── shap_waterfall_impaired_case.png # Patient OAS1_0003 local explanation
├── app.py                               # Production-grade Streamlit clinical interactive CDSS
├── main.py                              # End-to-end training and evaluation script
├── requirements.txt                     # Project dependencies
└── README.md                            # Comprehensive documentation
```

---

## 🧬 Domain-Specific Feature Engineering

To capture subtle physiological interactions, we engineered seven domain-specific features derived from neuroimaging and clinical biomarkers:

1. **Brain Atrophy Index ($\text{BAI}$):**
   $$\text{BAI} = 1.0 - \text{nWBV}$$
   *Direct estimate of cerebral parenchyma lost to ventriculomegaly and sulcal widening.*

2. **Estimated Absolute Brain Parenchymal Volume ($\text{EBV}$):**
   $$\text{EBV} = \text{eTIV} \times \text{nWBV}\quad(\text{in mm}^3)$$
   *Recovers absolute cerebral tissue volume, differentiating global cranial size from localized shrinkage.*

3. **Cranial Vault Size Scaling Ratio ($\text{CSR}$):**
   $$\text{CSR} = \text{ASF} \times \text{eTIV}$$
   *Stereotaxic atlas calibration verifying cranial vault spatial transformation.*

4. **Age-Adjusted Pathological Atrophy Interaction ($\text{AAAI}$):**
   $$\text{AAAI} = \text{Age} \times (1.0 - \text{nWBV})$$
   *Decouples physiological chronological senescence from accelerated disease-mediated neurodegeneration.*

5. **MMSE Cognitive Deficit Index ($\text{MCDI}$):**
   $$\text{MCDI} = 30.0 - \text{MMSE}$$
   *Measures the absolute functional deficit from ceiling neurocognitive performance.*

6. **Cognitive Reserve Index ($\text{CRI}$):**
   $$\text{CRI} = \frac{\text{Educ}}{\text{SES} + 0.1}$$
   *Combines education years with socioeconomic status as a proxy for neural cognitive reserve.*

7. **Multimodal Neuro-Cognitive Coupling ($\text{NCC}$):**
   $$\text{NCC} = \text{nWBV} \times \text{MMSE}$$
   *Coupled functional-structural biomarker: captures cases where structural reserve preserves cognition, or where low scores coincide with severe structural loss.*

---

## 🔍 Feature Selection Methodology

To avoid collinearity and prevent overfitting on clinical sample sizes, we implemented a multi-criteria consensus feature selection protocol:
1. **Mutual Information (Information Gain):** Quantifies non-linear dependency between biomarkers and cognitive impairment.
2. **L1-penalized LASSO:** Shrinks redundant linear predictors to zero via elasticnet regularization.
3. **Random Forest Ensemble Importance:** Computes Mean Decrease in Impurity (Gini importance).
4. **Recursive Feature Elimination with Cross-Validation (RFECV):** Evaluated over 5 stratified folds optimizing ROC-AUC.

**Optimal Selected Biomarker Subset (12 features):**
`Neuro_Cognitive_Ratio`, `MMSE_Deficit`, `MMSE`, `Brain_Atrophy_Index`, `Age_Adjusted_Atrophy`, `nWBV`, `Age`, `Cognitive_Reserve_Index`, `Estimated_Brain_Volume`, `eTIV`, `ASF`, `Cranial_Size_Ratio`.

---

## ⚡ Bayesian Hyperparameter Optimization (Optuna)

We utilized **Optuna** with the **Tree-structured Parzen Estimator (TPE)** sampler across 35 trials per candidate architecture, evaluating strictly via Stratified 5-Fold Cross Validation.

Key optimized parameters:
- **LightGBM:** `n_estimators`: 100, `num_leaves`: 20, `max_depth`: 4, `learning_rate`: 0.046, `subsample`: 0.7, `colsample_bytree`: 0.7, `reg_alpha`: 0.003, `reg_lambda`: 0.176.
- **XGBoost:** `n_estimators`: 125, `max_depth`: 3, `learning_rate`: 0.051, `subsample`: 0.8, `colsample_bytree`: 0.7, `reg_alpha`: 0.012, `reg_lambda`: 1.25.
- **Random Forest:** `n_estimators`: 175, `max_depth`: 4, `min_samples_split`: 4, `min_samples_leaf`: 2, `max_features`: 'sqrt'.
- **Support Vector Machine (SVM):** `C`: 1.84, `gamma`: 'scale', `kernel`: 'rbf'.

---

## 📈 Cross-Validation Model Benchmark Results

Out-of-fold cross-validation performance across 5 stratified folds (with fold-isolated imputation and scaling):

| Model Architecture | 5-Fold ROC-AUC | Precision-Recall AUC (PR-AUC) | Balanced Accuracy | Sensitivity (Recall) | Specificity | F1-Score | Matthews Corr. (MCC) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🏆 **Optuna LightGBM** | **0.919 ± 0.035** | **0.909 ± 0.042** | 0.808 ± 0.051 | 0.720 ± 0.115 | **0.896 ± 0.092** | 0.772 ± 0.067 | **0.641 ± 0.098** |
| 🥈 **Optuna XGBoost** | 0.913 ± 0.027 | 0.906 ± 0.032 | 0.812 ± 0.059 | **0.780 ± 0.097** | 0.844 ± 0.103 | **0.784 ± 0.065** | 0.633 ± 0.121 |
| 🥉 **Optuna RandomForest** | 0.911 ± 0.029 | 0.903 ± 0.030 | 0.812 ± 0.065 | 0.720 ± 0.144 | **0.904 ± 0.056** | 0.773 ± 0.086 | 0.647 ± 0.111 |
| 📊 **Logistic Regression** | 0.901 ± 0.033 | 0.898 ± 0.036 | **0.816 ± 0.056** | 0.750 ± 0.141 | 0.881 ± 0.061 | 0.780 ± 0.075 | 0.647 ± 0.092 |
| 🎯 **Optuna SVM (RBF)** | 0.901 ± 0.039 | 0.895 ± 0.043 | 0.794 ± 0.063 | 0.700 ± 0.127 | 0.889 ± 0.064 | 0.753 ± 0.084 | 0.610 ± 0.115 |
| 🧠 **Baseline MLP** | 0.901 ± 0.028 | 0.894 ± 0.033 | 0.796 ± 0.060 | 0.740 ± 0.082 | 0.852 ± 0.087 | 0.763 ± 0.066 | 0.602 ± 0.121 |

---

## 🧠 Explainable AI (XAI) & SHAP Interpretability

### Global Cohort Explanations
Using `shap.TreeExplainer`, Shapley values quantify the exact positive and negative contribution of each biomarker:
- **Global Beeswarm Plot:** Demonstrates that low `Neuro_Cognitive_Ratio` and high `MMSE_Deficit` are the primary drivers shifting predictions toward cognitive impairment. Lower `nWBV` (cerebral atrophy) consistently pushes SHAP values positively toward dementia.
- **Global Mean Absolute SHAP Ranking:** Confirms that multi-modal interaction features (`Neuro_Cognitive_Ratio`, `MMSE_Deficit`, `Age_Adjusted_Atrophy`) provide stronger discriminative signal than any single raw feature alone.

### Biomarker Dependence & Risk Thresholds
- **nWBV Atrophy Threshold:** Across the cohort, SHAP values remain neutral or protective when $\text{nWBV} > 0.76$. Once $\text{nWBV} \le 0.73$, SHAP contributions sharply escalate into positive risk territory.
- **MMSE Threshold:** Scores $\ge 29$ provide strong negative SHAP values (protective against impairment prediction). Scores $\le 26$ exhibit a steep non-linear surge in positive log-odds.

### Local Patient Decision Waterfall Case Studies

#### Case 1: Healthy Control (`OAS1_0001_MR1`) — Non-Demented ($CDR = 0.0$)
- **Model Estimated Risk:** **37.5%** (Classified Normal)
- **Primary Protective Drivers:** Perfect cognitive performance ($\text{MMSE} = 29.0$, $\text{MMSE\_Deficit} = 1.0$, $\text{SHAP} = -0.37$), coupled with balanced neuro-cognitive interaction ($\text{SHAP} = -0.26$).
- **Risk Mitigators:** Compensated for slight age-associated atrophy ($74\text{ yrs}$, $\text{nWBV} = 0.743$).

#### Case 2: Impaired Patient (`OAS1_0003_MR1`) — Very Mild Impairment / MCI ($CDR = 0.5$)
- **Model Estimated Risk:** **87.2%** (Classified Impaired)
- **Primary Risk Drivers:** Marked reduction in neuro-cognitive coupling ($\text{SHAP} = +1.63$), severe accelerated brain atrophy ($\text{nWBV} = 0.708$, $\text{AAAI} = 21.32$, $\text{SHAP} = +0.33$).
- **Clinical Conclusion:** Structural brain atrophy combined with mild MMSE reduction ($27.0$) triggered a high-confidence cognitive impairment alert.

---

## 🖥 Interactive Streamlit Clinical Decision Platform

The included Streamlit web application (`app.py`) provides a multi-tab clinical dashboard:
1. **📊 Cohort & Neuroimaging Explorer:** Interactive 3D and 2D scatter plots of nWBV vs MMSE vs Age, interactive CDR stage breakdowns, and correlation matrices.
2. **🧬 Feature Engineering & Selection Matrix:** Formula reference table, consensus ranking, and RFECV curves.
3. **🚀 Model Benchmarking & Optuna Tuning:** Side-by-side ROC, PR, Confusion Matrices, and hyperparameter inspector.
4. **🧠 Global SHAP Interpretability:** Interactive Beeswarm, Bar importance, and non-linear dependence plots.
5. **🩺 Patient Diagnostic Simulator (Local SHAP):** Clinicians can select real OASIS-1 patients or input custom values (Age, MMSE, nWBV, eTIV), receive an instant risk probability gauge, generate a live SHAP waterfall plot, and export an EHR-ready diagnostic narrative card.

---

## 🚀 Installation & Reproduction Guide

### 1. Clone & Set Up Environment
```bash
# Clone the repository
git clone https://github.com/your-username/oasis-cognitive-impairment-ml.git
cd "HA project"

# Install dependencies
python -m pip install -r requirements.txt
```

### 2. Run the End-to-End ML Pipeline
```bash
# Ingests data, engineers features, tunes with Optuna, evaluates models, and saves SHAP figures
python main.py
```

### 3. Launch the Interactive Streamlit App
```bash
streamlit run app.py
```
*Open your web browser at `http://localhost:8501` to access the clinical decision support interface.*

---

## 📜 Citation & Acknowledgments
- **Dataset:** OASIS-1 (Open Access Series of Imaging Studies), Washington University School of Medicine, Harvard University, Neuroinformatics Research Group. [https://www.oasis-brains.org/#oasis1](https://www.oasis-brains.org/#oasis1)
- **Marcus, D. S. et al.** (2007). *Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI Data in Young, Middle Aged, Nondemented, and Demented Older Adults.* Journal of Cognitive Neuroscience, 19(9), 1498-1507.
- **Lundberg, S. M., & Lee, S. I.** (2017). *A Unified Approach to Interpreting Model Predictions.* Advances in Neural Information Processing Systems (NeurIPS 30).
- **Akiba, T. et al.** (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework.* KDD 2019.
