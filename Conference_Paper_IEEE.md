# Explainable Machine Learning for Prediction of Cognitive Impairment Using Clinical and Neuroimaging Features

**Anshuman Padole**  
*Department of Computer Science & Engineering*  
*Affiliation / University Name*  
*City, Country*  
*anshuman.padole@example.edu*

**Dev Dhapodkar**  
*Department of Computer Science & Engineering*  
*Affiliation / University Name*  
*City, Country*  
*dev.dhapodkar@example.edu*

---

### Abstract
Early and accurate detection of cognitive impairment, spanning Mild Cognitive Impairment (MCI) and Alzheimer's Disease (AD), is crucial for timely therapeutic interventions and patient care planning. However, conventional diagnostic workflows face trade-offs between subjective, culturally sensitive cognitive assessments and resource-intensive, high-dimensional neuroimaging modalities. Moreover, black-box machine learning approaches frequently encounter reluctance among clinicians due to the absence of mechanistic interpretability. In this work, we propose an explainable and optimized multimodal machine learning framework that seamlessly fuses clinical demographics, bedside cognitive assessments (Mini-Mental State Examination, MMSE), and structural Magnetic Resonance Imaging (MRI) volumetric features from the Open Access Series of Imaging Studies (OASIS-1) cross-sectional cohort. We engineer seven biologically grounded domain features capturing non-linear interactions between chronological senescence, parenchymal atrophy, and cognitive reserve. A multi-criteria consensus feature selection strategy combining Mutual Information, L1-penalized LASSO, and Recursive Feature Elimination with Cross-Validation (RFECV) isolates a parsimonious twelve-biomarker signature. Hyperparameter optimization across candidate architectures (LightGBM, XGBoost, Random Forest, Support Vector Machines, Multi-Layer Perceptrons, and Logistic Regression) is conducted using Bayesian optimization via the Tree-structured Parzen Estimator (TPE) algorithm in Optuna over Stratified 5-Fold Cross-Validation. The champion Optuna-tuned LightGBM model achieves an Out-of-Fold Receiver Operating Characteristic Area Under Curve (ROC-AUC) of 0.919 ± 0.035, a Precision-Recall AUC (PR-AUC) of 0.909 ± 0.042, an accuracy of 82.1%, and a clinical specificity of 89.6%, outperforming standalone linear baselines and complex ensembles. To enable clinical transparency, game-theoretic SHapley Additive exPlanations (SHAP) are integrated at both global cohort and local patient levels. SHAP global analyses identify coupled neuro-cognitive integrity and normalized whole-brain volume as the primary drivers of impairment, revealing non-linear risk transition thresholds at MMSE ≤ 26 and normalized whole brain volume ≤ 0.73. Local waterfall visualizations and an automated electronic health record clinical summary card are deployed inside an interactive clinical decision support web platform. This framework bridges the gap between high-precision machine learning and interpretable, actionable bedside neurology.

**Keywords**—Cognitive Impairment, Alzheimer's Disease, OASIS-1, Explainable Artificial Intelligence (XAI), SHAP, Bayesian Optimization, Optuna, Neuroimaging Biomarkers, Machine Learning.

---

## I. Introduction

Dementia, primarily driven by Alzheimer's Disease (AD) and its prodromal precursor Mild Cognitive Impairment (MCI), represents one of the most critical public health imperatives of the twenty-first century, affecting over 55 million individuals worldwide [1]. The insidious onset of neurodegeneration often precedes overt clinical symptomatology by more than a decade. Consequently, establishing computational frameworks capable of identifying cognitive impairment during its early stages is essential for disease-modifying therapies, lifestyle modifications, and enrichment of prospective clinical trials [2].

In routine memory clinic practice, screening predominantly relies on psychometric questionnaires such as the Mini-Mental State Examination (MMSE) and the Clinical Dementia Rating (CDR) scale [3]. While bedside cognitive screening is rapid and low-cost, scores are heavily confounded by educational background, socioeconomic disparities, linguistic proficiency, and examiner subjectivity [4]. Conversely, structural Magnetic Resonance Imaging (MRI) offers non-invasive, objective biomarkers of macrostructural brain deterioration, including cerebral cortical thinning, temporal lobe atrophy, and ventricular enlargement [5]. Nevertheless, visual evaluation of MRI scans by radiologists is qualitative, labor-intensive, and prone to intra- and inter-observer variability, while automated volumetric extraction generates high-dimensional metrics whose multivariate non-linear interactions remain difficult to parse heuristically.

Although advanced machine learning (ML) models, including gradient-boosted decision trees and deep neural networks, have shown substantial promise in biomedical classification [6], their adoption in high-stakes clinical decision-making remains hindered by their "black-box" opacity. In neurology, clinicians cannot safely rely on a probabilistic risk output without understanding the underlying mechanistic rationale: *Why is a given patient classified as impaired? Is the prediction driven by irreversible cerebral atrophy, acute cognitive deficits, or normal age-related senescence?*

To overcome these foundational challenges, this paper presents an end-to-end, optimized, and fully explainable multimodal machine learning framework designed for the cross-sectional OASIS-1 cohort. Our principal contributions are summarized as follows:

1. **Multimodal Domain-Informed Feature Engineering:** We design and integrate seven physiologically grounded interaction metrics combining structural MRI volumetry (Normalized Whole Brain Volume, Estimated Total Intracranial Volume, Atlas Scaling Factor) with clinical markers (MMSE, Education, Socioeconomic Status, Age), systematically accounting for physiological brain reserve and chronological versus pathological atrophy.
2. **Multi-Criteria Consensus Feature Selection:** We formulate an objective feature selection pipeline integrating Information Gain (Mutual Information), sparse L1-regularization (LASSO), Random Forest Gini impurity, and Stratified Recursive Feature Elimination with Cross-Validation (RFECV) to eliminate collinearity and prevent data dimensionality traps.
3. **Bayesian Hyperparameter Optimization:** We implement Tree-structured Parzen Estimator (TPE) Bayesian optimization via Optuna across six candidate classifiers, strictly enclosed within Stratified 5-Fold Cross-Validation loops to eliminate data leakage.
4. **Dual-Level Explainable AI (XAI) via SHAP:** We integrate game-theoretic SHapley Additive exPlanations (SHAP) to establish both global population-level feature importance dynamics and patient-specific local decision waterfall breakdowns, identifying explicit biomarker risk thresholds for clinical practice.
5. **Interactive Clinical Decision Support Deployment:** We construct a production-ready interactive diagnostic web platform (NeuroExplain AI) featuring interactive parameter manipulation, automated Electronic Health Record (EHR) narrative report generation, and real-time waterfall visual decomposition.

---

## II. Related Work

The application of computational intelligence to neurodegenerative disease diagnosis has evolved across distinct paradigms over the past two decades. Early studies utilized unimodal approaches, relying either exclusively on psychometric cognitive scores [3] or raw volumetric measurements extracted via tools such as FreeSurfer or SPM [7]. Unimodal clinical models frequently demonstrated high sensitivity but compromised specificity, misclassifying healthy older adults with lower educational attainment as impaired. On the other hand, purely MRI-derived morphometric models frequently underperformed on subtle MCI stages where macrostructural tissue loss remained borderline [8].

Recent investigations have increasingly favored multimodal fusion, combining neuroimaging, clinical evaluations, and genetic risk alleles (e.g., APOE-ε4) [9]. For example, Marcus et al. [10] established the foundational utility of the OASIS repository for cross-sectional morphometric benchmarking. Subsequent studies deployed Support Vector Machines (SVM), Random Forests, and Multi-Layer Perceptrons on raw OASIS parameters, reporting classification accuracies between 75% and 88% [11], [12]. However, many existing pipelines suffer from critical methodological shortcomings:
- **Data Leakage in Preprocessing:** Numerous published benchmarks perform normalization, imputation, or feature selection over the entire dataset prior to cross-validation partitioning, artificially inflating reported performance metrics [13].
- **Neglect of Interaction Biomarkers:** Most literature inputs raw parameters directly into tree ensembles without explicitly modeling non-linear physiological interactions such as the interplay between chronological age and parenchymal volume loss [14].
- **Lack of Local Decision Interpretability:** While several studies present global feature importances, very few offer local, instance-level explanations that clarify why an individual patient was categorized into a specific diagnostic stratum.

Our work explicitly addresses these gaps by embedding leak-free cross-validation architectures, domain-engineered interaction terms, and local game-theoretic Shapley attributions into an integrated framework.

---

## III. Dataset and Cohort Formulation

### A. OASIS-1 Cross-Sectional Cohort
The Open Access Series of Imaging Studies (OASIS-1) dataset consists of a cross-sectional collection of 436 subjects aged 18 to 96 years [10]. For each subject, three to four individual T1-weighted magnetization-prepared rapid gradient-echo (MP-RAGE) scans were acquired on a 1.5-T Siemens Vision scanner within a single imaging session.

In the OASIS-1 distribution, young adults aged 18–59 ($N = 201$) represent cognitively normal volunteers who did not undergo the comprehensive dementia evaluation protocol; hence, clinical scales including MMSE and CDR were not administered to this sub-cohort. To maintain clinical rigor and prevent synthetic distortion from imputing unadministered cognitive tests, our study restricts the primary analytical cohort to the **$N = 235$ clinically evaluated individuals** aged 33 to 96 years.

```
OASIS-1 Full Repository (N = 436)
 │
 ├── Unassessed Young Controls (Age 18-59, CDR unadministered) ──> Excluded (n = 201)
 │
 └── Clinically Evaluated Cohort ──> Included (N = 235)
      ├── Cognitively Intact / Normal (CDR = 0.0) ──> n = 135 (57.4%)
      └── Cognitively Impaired (CDR ≥ 0.5)        ──> n = 100 (42.6%)
           ├── Very Mild Impairment / MCI (CDR = 0.5) ──> n = 70
           ├── Mild Dementia (CDR = 1.0)              ──> n = 28
           └── Moderate Dementia (CDR = 2.0)          ──> n = 2
```

### B. Diagnostic Target Definition
The target variable is derived from the Clinical Dementia Rating (CDR), a global clinical staging instrument:
- **Normal / Cognitively Intact ($y = 0$):** $\text{CDR} = 0.0$ ($n = 135$, 57.4%)
- **Cognitively Impaired ($y = 1$):** $\text{CDR} \ge 0.5$ ($n = 100$, 42.6%)

The demographic and neuroimaging characteristics of the clinically evaluated cohort are detailed in Table I.

```
TABLE I
BASELINE COHORT DEMOGRAPHICS AND NEUROIMAGING CHARACTERISTICS
======================================================================================
Characteristic                    Full Cohort (N=235)   Normal (n=135)   Impaired (n=100)
--------------------------------------------------------------------------------------
Age (years, Mean ± SD)                72.3 ± 12.1        69.1 ± 12.3       76.8 ± 10.3
Sex (Female / Male)                   156 / 79            99 / 36           57 / 43
Education Level (Educ, 1 to 5)         3.2 ± 1.3          3.4 ± 1.3         2.9 ± 1.3
Socioeconomic Status (SES, 1 to 5)     2.5 ± 1.1          2.3 ± 1.1         2.8 ± 1.1
Mini-Mental State Exam (MMSE)         27.1 ± 3.7         29.1 ± 1.2        24.4 ± 4.1
Normalized Whole Brain Volume (nWBV)  0.749 ± 0.048      0.769 ± 0.044     0.722 ± 0.038
Estimated Total Intracranial Vol (eTIV) 1460.6 ± 175.8  1441.5 ± 167.3   1486.5 ± 184.8
Atlas Scaling Factor (ASF)            1.218 ± 0.138      1.231 ± 0.134     1.200 ± 0.142
======================================================================================
```

---

## IV. Methodology

The overall architecture of the proposed explainable multimodal framework is depicted in Fig. 1.

```
+----------------------------------------------------------------------------------+
|                           PROPOSED FRAMEWORK PIPELINE                            |
+----------------------------------------------------------------------------------+
  [ OASIS-1 Data ] ──> [ Clinical Filtering (N=235) ] ──> [ Preprocessing Pipeline ]
                                                                  │
  +───────────────────────────────────────────────────────────────+
  │
  ▼
[ Multimodal Feature Engineering ] ──> 7 Domain Interaction Biomarkers Engineered
  │
  ▼
[ Multi-Criteria Selection ]       ──> Consensus Ranking (MI + LASSO + RFECV) ──> Top 12
  │
  ▼
[ Bayesian Optimization ]          ──> Optuna (TPE Sampler) over Stratified 5-Fold CV
  │
  ▼
[ Model Benchmarking ]             ──> LightGBM, XGBoost, RF, SVM, MLP, LogReg
  │
  ▼
[ SHAP Explainability Engine ]     ──> Global Beeswarm / Bar / 2D Dependence Plots
  │                                    Local Patient Decision Waterfall Decompositions
  ▼
[ Interactive CDSS Platform ]      ──> Streamlit UI with Live Inference & EHR Reports
```
*Fig. 1. Architectural schematic of the proposed explainable machine learning pipeline.*

### A. Multimodal Feature Engineering
To bridge macrostructural neuroimaging with functional clinical observations, we formulated seven domain-specific interaction biomarkers:

1. **Brain Atrophy Index ($\text{BAI}$):** Quantifies total proportional intracranial tissue loss:
$$\text{BAI} = 1.0 - \text{nWBV} \tag{1}$$

2. **Estimated Parenchymal Brain Volume ($\text{EBV}$):** Approximates absolute parenchymal volume in cubic millimeters, dissociating localized atrophy from baseline head geometry:
$$\text{EBV} = \text{eTIV} \times \text{nWBV} \tag{2}$$

3. **Cranial Size Ratio ($\text{CSR}$):** Calibrates stereotaxic atlas transformation consistency:
$$\text{CSR} = \text{ASF} \times \text{eTIV} \tag{3}$$

4. **Age-Adjusted Pathological Atrophy Interaction ($\text{AAAI}$):** Distinguishes accelerated disease-mediated atrophy from expected physiological senescence:
$$\text{AAAI} = \text{Age} \times (1.0 - \text{nWBV}) \tag{4}$$

5. **MMSE Cognitive Deficit Index ($\text{MCDI}$):** Quantifies functional degradation relative to maximum neurocognitive performance:
$$\text{MCDI} = 30.0 - \text{MMSE} \tag{5}$$

6. **Cognitive Reserve Index ($\text{CRI}$):** Approximates neurocognitive resilience combining educational attainment ($\text{Educ} \in [1, 5]$) and socioeconomic status ($\text{SES} \in [1, 5]$):
$$\text{CRI} = \frac{\text{Educ}}{\text{SES} + 0.1} \tag{6}$$

7. **Multimodal Neuro-Cognitive Coupling ($\text{NCC}$):** Models structural-functional concordance:
$$\text{NCC} = \text{nWBV} \times \text{MMSE} \tag{7}$$

### B. Preprocessing and Leak-Free Pipeline
All data transformations are encapsulated inside strict cross-validation folds. Missing entries in socioeconomic status ($n = 19$, 8.1%) are imputed using median estimation derived exclusively from the training partition of each fold. Continuous variables are scaled using standard $z$-score normalization:
$$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}} \tag{8}$$
Uninformative variables exhibiting zero variance (e.g., Handedness, which is 100% right-handed) or 100% missingness (Delay, unpopulated in cross-sectional sessions) are eliminated.

### C. Multi-Criteria Feature Selection
To prevent feature redundancy and mitigate the curse of dimensionality, we employ a multi-criteria consensus feature ranking protocol:
1. **Mutual Information ($I(X; Y)$):** Measures general non-linear statistical dependence:
$$I(X; Y) = \sum_{y \in Y} \sum_{x \in X} p(x, y) \log \frac{p(x, y)}{p(x)p(y)} \tag{9}$$
2. **L1-Penalized LASSO:** Imposes an $\ell_1$ penalty to drive uninformative coefficients to zero:
$$\min_{\mathbf{w}} \left\{ -\sum_{i=1}^N \log P(y_i | \mathbf{x}_i; \mathbf{w}) + \lambda \|\mathbf{w}\|_1 \right\} \tag{10}$$
3. **Random Forest Gini Importance:** Measures mean decrease in node impurity.
4. **Stratified RFECV:** Employs recursive backward elimination evaluating 5-fold cross-validated ROC-AUC.

The consensus scoring algorithm yielded twelve optimal biomarkers: `Neuro_Cognitive_Ratio`, `MMSE_Deficit`, `MMSE`, `Brain_Atrophy_Index`, `Age_Adjusted_Atrophy`, `nWBV`, `Age`, `Cognitive_Reserve_Index`, `Estimated_Brain_Volume`, `eTIV`, `ASF`, and `Cranial_Size_Ratio`.

### D. Bayesian Hyperparameter Optimization
To maximize clinical predictive efficacy, we deployed Bayesian optimization utilizing the Tree-structured Parzen Estimator (TPE) algorithm implemented in Optuna [15]. Unlike brute-force grid search, TPE models the probability distribution of hyperparameters conditioned on past evaluation outcomes:
$$p(\theta | y) = \begin{cases} \ell(\theta), & \text{if } y < y^* \\ g(\theta), & \text{if } y \ge y^* \end{cases} \tag{11}$$
where $y^*$ is the objective quantile threshold (optimizing 5-fold Stratified ROC-AUC), and $\theta$ denotes the parameter vector. Optimization was executed across 35 trials per model architecture.

---

## V. Experimental Results and Discussion

### A. Candidate Model Benchmark
We conducted comprehensive out-of-fold benchmarking across six diverse classifier paradigms:
1. **Optuna LightGBM:** Fast gradient-boosted decision trees utilizing histogram-based binning.
2. **Optuna XGBoost:** Scalable second-order gradient boosted decision trees.
3. **Optuna Random Forest:** Bagged ensemble of decorrelated decision trees.
4. **Optuna Support Vector Classifier (SVM):** Non-linear kernel classification with Radial Basis Function (RBF).
5. **Logistic Regression (ElasticNet):** Regularized linear baseline.
6. **Multi-Layer Perceptron (MLP):** Feedforward artificial neural network ($64 \times 32$ hidden units).

Out-of-fold performance across all five folds is reported in Table II.

```
TABLE II
OUT-OF-FOLD 5-FOLD CROSS-VALIDATION PERFORMANCE BENCHMARK
===========================================================================================================
Model Architecture       ROC-AUC          PR-AUC       Balanced Acc    Sensitivity    Specificity    F1-Score
-----------------------------------------------------------------------------------------------------------
Optuna LightGBM      0.919 ± 0.035    0.909 ± 0.042   0.808 ± 0.051  0.720 ± 0.115  0.896 ± 0.092  0.772 ± 0.067
Optuna XGBoost       0.913 ± 0.027    0.906 ± 0.032   0.812 ± 0.059  0.780 ± 0.097  0.844 ± 0.103  0.784 ± 0.065
Optuna RandomForest  0.911 ± 0.029    0.903 ± 0.030   0.812 ± 0.065  0.720 ± 0.144  0.904 ± 0.056  0.773 ± 0.086
Logistic Regression  0.901 ± 0.033    0.898 ± 0.036   0.816 ± 0.056  0.750 ± 0.141  0.881 ± 0.061  0.780 ± 0.075
Optuna SVM (RBF)     0.901 ± 0.039    0.895 ± 0.043   0.794 ± 0.063  0.700 ± 0.127  0.889 ± 0.064  0.753 ± 0.084
Baseline MLP         0.901 ± 0.028    0.894 ± 0.033   0.796 ± 0.060  0.740 ± 0.082  0.852 ± 0.087  0.763 ± 0.066
===========================================================================================================
```

### B. Performance Analysis
As illustrated in Table II, **Optuna LightGBM** achieved the highest discriminative power, recording an Out-of-Fold ROC-AUC of **0.919 ± 0.035** and a PR-AUC of **0.909 ± 0.042**. In clinical diagnostic environments, false positives impose substantial emotional and economic burdens through unnecessary invasive lumbar punctures or amyloid-PET scans; LightGBM demonstrated a high clinical specificity of **89.6%**, with Random Forest achieving **90.4%**. Concurrently, Optuna XGBoost exhibited superior sensitivity (**78.0%**), demonstrating balanced detection of very mild MCI cases.

The optimal hyperparameters selected via Bayesian search for the champion LightGBM model were: `n_estimators = 100`, `num_leaves = 20`, `max_depth = 4`, `learning_rate = 0.046`, `subsample = 0.70`, `colsample_bytree = 0.70`, `reg_alpha = 0.003`, and `reg_lambda = 0.176`.

---

## VI. Explainable AI (XAI) Analysis Using SHAP

To establish clinical interpretability, we compute exact Shapley values using `shap.TreeExplainer` rooted in cooperative game theory [16]:
$$f(x) = \phi_0 + \sum_{i=1}^M \phi_i(x) \tag{12}$$
where $\phi_0$ represents the base expected value across the training population, and $\phi_i(x)$ denotes the additive attribution of biomarker $i$.

### A. Global Feature Importance and Attribution
Global SHAP attribution reveals that the engineered interaction biomarker `Neuro_Cognitive_Ratio` ($\text{nWBV} \times \text{MMSE}$) is the single most dominant predictor across the population, followed by `MMSE_Deficit` and raw `MMSE`. Macrostructural neuroimaging measures—specifically `Brain_Atrophy_Index` and `nWBV`—consistently rank as the top morphological determinants. In the SHAP beeswarm distribution, low values of `nWBV` (manifesting cerebral atrophy) generate pronounced positive Shapley attributions, actively driving model log-odds toward cognitive impairment.

### B. Non-Linear Biomarker Tipping Points
SHAP dependence plots uncover critical physiological thresholds:
- **Whole Brain Volume Threshold:** For subjects with $\text{nWBV} > 0.76$, the Shapley attribution remains consistently negative (protective against an impairment diagnosis). However, once $\text{nWBV} \le 0.73$, the attribution sharply shifts into positive risk territory, accelerating in magnitude.
- **MMSE Cognitive Threshold:** While MMSE scores of 29–30 provide strong negative attributions, scores falling below 26 exhibit a steep, non-linear escalation in positive SHAP values.

```
       SHAP Dependence: nWBV vs. Risk
  +0.8 |                      * (High Impairment Risk)
       |                    *
  +0.4 |                  *
SHAP   | -------------- * ------------------------ (Baseline = 0)
Value  |              *
  -0.4 |            *      * (Protective Zone)
       +--------+---------+---------+--------+
              0.68      0.72      0.76     0.80
                    nWBV (Normalized Volume)
```
*Fig. 2. Conceptual representation of the non-linear risk tipping point observed in SHAP dependence for normalized whole-brain volume.*

### C. Patient-Level Decision Waterfall Case Studies

To validate local clinical decision support, we examine two contrasting patient cases from the OASIS-1 cohort:

#### 1) Case Study A: Healthy Control (Subject `OAS1_0001_MR1`)
- **Clinical Profile:** Female, 74 years old, $\text{MMSE} = 29.0$, $\text{nWBV} = 0.743$, $\text{CDR} = 0.0$.
- **Model Output:** Estimated Impairment Risk = **37.5%** (Correctly categorized as Cognitively Normal).
- **Shapley Decomposition:** The base population log-odds is shifted downward primarily by high cognitive performance (`MMSE_Deficit = 1.0`, $\phi = -0.37$) and preserved neuro-cognitive coupling (`Neuro_Cognitive_Ratio = 21.55`, $\phi = -0.26$). These protective factors successfully counterbalanced the mild positive risk contribution associated with advanced age ($74\text{ years}$, $\phi = +0.16$).

#### 2) Case Study B: Very Mild Impairment / MCI (Subject `OAS1_0003_MR1`)
- **Clinical Profile:** Female, 73 years old, $\text{MMSE} = 27.0$, $\text{nWBV} = 0.708$, $\text{CDR} = 0.5$.
- **Model Output:** Estimated Impairment Risk = **87.2%** (Correctly categorized as Impaired).
- **Shapley Decomposition:** Although an MMSE of 27.0 might heuristically appear borderline in basic screening, the model detects severe structural-functional decoupling. The primary risk driver is the severely diminished `Neuro_Cognitive_Ratio` ($19.12$, $\phi = +1.63$), accompanied by marked cerebral atrophy (`Age_Adjusted_Atrophy = 21.32`, $\phi = +0.33$). The confluence of accelerated parenchymal shrinkage and subtle cognitive deficit triggered an accurate, high-confidence warning.

---

## VII. Interactive Decision Support Deployment

To translate our algorithmic contributions into bedside utility, we deployed the trained pipeline into **NeuroExplain AI**, an interactive Streamlit application. The platform features:
1. **Interactive Population Analytics:** Visualizes multi-dimensional scatter matrices ($3\text{D}/2\text{D}$) correlating volumetric loss, age, and MMSE across CDR stages.
2. **Dynamic Risk Simulator:** Allows neurologists to adjust patient parameters via interactive sliders or import existing OASIS subject IDs.
3. **Real-Time SHAP Waterfall Rendering:** Generates on-the-fly local waterfall visualizations detailing the specific physiological drivers behind each prediction.
4. **Automated EHR Narrative Generation:** Translates complex mathematical Shapley values into standardized, clinician-friendly electronic medical record summary cards specifying primary risk drivers, protective compensatory factors, and suggested diagnostic follow-ups.

---

## VIII. Conclusion and Future Work

In this paper, we introduced an explainable, optimized multimodal machine learning framework for predicting cognitive impairment in the OASIS-1 cohort. By engineering biologically grounded interaction features between structural MRI volumetry and cognitive evaluations, executing multi-criteria consensus feature selection, and leveraging Optuna Bayesian hyperparameter optimization, the champion LightGBM model achieved a 5-fold cross-validated ROC-AUC of **0.919** and a specificity of **89.6%**. The integration of SHAP game-theoretic explainability resolved the black-box dilemma by illuminating both global population tipping points and patient-specific decision waterfall trajectories.

Future extensions of this research will focus on:
1. Validating the framework on longitudinal cohorts (e.g., OASIS-2 and ADNI) to forecast progression from MCI to active Alzheimer's Dementia over 24- to 48-month horizons.
2. Integrating regional subcortical segmentations (hippocampal volume, entorhinal cortical thickness) with plasma fluid biomarkers (p-tau217, neurofilament light chain).
3. Implementing conformal prediction intervals to provide rigorous statistical confidence guarantees for clinical risk estimates.

---

## Acknowledgment
The authors would like to acknowledge Washington University in St. Louis and the Open Access Series of Imaging Studies (OASIS) initiative for making the neuroimaging repository publicly available for neuroinformatics research.

---

## References

[1] World Health Organization, *Global status report on the public health response to dementia*, Geneva: World Health Organization, 2021.

[2] R. Petersen et al., "Mild cognitive impairment: clinical characterization and outcome," *Arch. Neurol.*, vol. 56, no. 3, pp. 303–308, Mar. 1999.

[3] M. F. Folstein, S. E. Folstein, and P. R. McHugh, "'Mini-mental state': a practical method for grading the cognitive state of patients for the clinician," *J. Psychiatr. Res.*, vol. 12, no. 3, pp. 189–198, Nov. 1975.

[4] J. C. Morris, "The Clinical Dementia Rating (CDR): current version and scoring rules," *Neurology*, vol. 43, no. 11, pp. 2412–2414, Nov. 1993.

[5] C. R. Jack et al., "Brain atrophy rates in normal aging, mild cognitive impairment, and Alzheimer's disease," *Neurology*, vol. 65, no. 8, pp. 1227–1233, Oct. 2005.

[6] Y. LeCun, Y. Bengio, and G. Hinton, "Deep learning," *Nature*, vol. 521, no. 7553, pp. 436–444, May 2015.

[7] B. Fischl, "FreeSurfer," *NeuroImage*, vol. 62, no. 2, pp. 774–781, Aug. 2012.

[8] S. Rathore, M. Habes, M. A. Iftikhar, H. Shacklett, and C. Davatzikos, "A review on neuroimaging-based classification of Alzheimer's disease and mild cognitive impairment," *NeuroImage*, vol. 155, pp. 530–548, Jul. 2017.

[9] D. Shen, G. Wu, and H. I. Suk, "Deep learning in medical image analysis," *Annu. Rev. Biomed. Eng.*, vol. 19, pp. 221–248, Jun. 2017.

[10] D. S. Marcus et al., "Open Access Series of Imaging Studies (OASIS): cross-sectional MRI data in young, middle aged, nondemented, and demented older adults," *J. Cogn. Neurosci.*, vol. 19, no. 9, pp. 1498–1507, Sep. 2007.

[11] F. J. Martinez-Murcia et al., "Studying the manifold structure of Alzheimer's disease: a deep learning approach using volumetric MRI," *IEEE J. Biomed. Health Inform.*, vol. 24, no. 1, pp. 17–26, Jan. 2020.

[12] A. Gupta, M. Ayhan, and A. Maida, "Natural image bases to represent neuroimaging data," in *Proc. 30th Int. Conf. Mach. Learn. (ICML)*, 2013, pp. 987–994.

[13] M. A. Little et al., "Using and reporting machine learning in medical research: a dual review," *BMC Med.*, vol. 22, no. 1, p. 10, Jan. 2024.

[14] G. B. Frisoni et al., "The clinical use of structural MRI in Alzheimer disease," *Nat. Rev. Neurol.*, vol. 6, no. 2, pp. 67–77, Feb. 2010.

[15] T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, "Optuna: a next-generation hyperparameter optimization framework," in *Proc. 25th ACM SIGKDD Int. Conf. Knowl. Discov. Data Min. (KDD)*, 2019, pp. 2623–2631.

[16] S. M. Lundberg and S. I. Lee, "A unified approach to interpreting model predictions," in *Adv. Neural Inf. Process. Syst. (NeurIPS 30)*, 2017, pp. 4765–4774.
