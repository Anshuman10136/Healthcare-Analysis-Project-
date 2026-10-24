"""
Main Execution Script for OASIS-1 Explainable Machine Learning Pipeline.
Executes End-to-End Workflow:
1. Data Ingestion & Clinical Cohort Validation
2. Neuroimaging & Clinical Feature Engineering
3. Feature Selection & Consensus Ranking
4. Bayesian Hyperparameter Optimization (Optuna)
5. Out-of-Fold Cross-Validation Model Benchmarking
6. Global & Local SHAP Interpretability Analysis
7. Model Artifact Serialization
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    MODELS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    RANDOM_STATE,
    CLASS_NAMES
)
from src.data_loader import load_clinical_cohort, get_cohort_summary
from src.feature_engineering import prepare_data, engineer_oasis_features
from src.feature_selection import FeatureSelector
from src.models import create_model, get_default_models
from src.hyperparameter_tuning import HyperparameterOptimizer
from src.evaluation import ModelEvaluator
from src.explainability import ShapExplainerManager
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def generate_eda_figures(df: pd.DataFrame) -> None:
    """Generate and save publication-quality exploratory data analysis plots."""
    print("\n[EDA] Generating Exploratory Data Analysis Figures...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Feature distributions stratified by Cognitive Impairment
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    palette = {0: "#2ca02c", 1: "#d62728"}

    # nWBV
    sns.kdeplot(data=df, x="nWBV", hue="Impaired", fill=True, palette=palette, common_norm=False, ax=axes[0, 0], alpha=0.4)
    axes[0, 0].set_title("Normalized Whole Brain Volume (nWBV) Distribution", fontweight="bold")
    axes[0, 0].set_xlabel("nWBV (Brain Volume / Total Intracranial Volume)")

    # MMSE
    sns.kdeplot(data=df, x="MMSE", hue="Impaired", fill=True, palette=palette, common_norm=False, ax=axes[0, 1], alpha=0.4)
    axes[0, 1].set_title("Mini-Mental State Exam (MMSE) Distribution", fontweight="bold")
    axes[0, 1].set_xlabel("MMSE Cognitive Score (0 - 30)")

    # Age
    sns.kdeplot(data=df, x="Age", hue="Impaired", fill=True, palette=palette, common_norm=False, ax=axes[1, 0], alpha=0.4)
    axes[1, 0].set_title("Age Distribution by Impairment Status", fontweight="bold")
    axes[1, 0].set_xlabel("Age (years)")

    # eTIV
    sns.kdeplot(data=df, x="eTIV", hue="Impaired", fill=True, palette=palette, common_norm=False, ax=axes[1, 1], alpha=0.4)
    axes[1, 1].set_title("Estimated Total Intracranial Volume (eTIV)", fontweight="bold")
    axes[1, 1].set_xlabel("eTIV (mm³)")

    for ax in axes.flat:
        if ax.get_legend():
            ax.get_legend().get_texts()[0].set_text("Normal (CDR 0)")
            ax.get_legend().get_texts()[1].set_text("Impaired (CDR > 0)")

    plt.tight_layout()
    eda_dist_path = str(FIGURES_DIR / "eda_distributions.png")
    fig.savefig(eda_dist_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved distribution plots to {eda_dist_path}")

    # 2. Correlation Matrix
    numeric_cols = ["Age", "Educ", "SES", "MMSE", "eTIV", "nWBV", "ASF", "Impaired"]
    corr_df = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title("Clinical & Neuroimaging Correlation Matrix (OASIS-1)", fontweight="bold", pad=12)
    plt.tight_layout()
    eda_corr_path = str(FIGURES_DIR / "eda_correlations.png")
    fig.savefig(eda_corr_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved correlation heatmap to {eda_corr_path}")


def run_pipeline() -> None:
    """Execute complete end-to-end explainable machine learning pipeline."""
    print("=" * 75)
    print(" EXPLAINABLE ML FOR COGNITIVE IMPAIRMENT PREDICTION (OASIS-1)")
    print("=" * 75)

    # 1. Load Data
    print("\n[Step 1/7] Ingesting OASIS-1 Clinical Cohort...")
    df = load_clinical_cohort()
    summary = get_cohort_summary(df)
    print(f"  Total Subjects: {summary['Total_Subjects']} "
          f"(Normal: {summary['Normal_Subjects']}, Impaired: {summary['Impaired_Subjects']})")
    print(f"  Age: {summary['Age_Mean_Std']}, MMSE: {summary['MMSE_Mean_Std']}, nWBV: {summary['nWBV_Mean_Std']}")

    # 2. Exploratory Visualizations
    generate_eda_figures(df)

    # 3. Feature Engineering
    print("\n[Step 2/7] Engineering Domain-Informed Clinical & Neuroimaging Features...")
    X, y, all_feature_names, patient_ids = prepare_data(df, feature_set="multimodal_all")
    print(f"  Constructed feature matrix: {X.shape[0]} subjects × {X.shape[1]} multimodal features.")
    print(f"  Features: {all_feature_names}")

    # 4. Feature Selection
    print("\n[Step 3/7] Performing Multi-Criteria Feature Selection (MI, LASSO, RF, RFECV)...")
    selector = FeatureSelector(random_state=RANDOM_STATE)
    ranking_df = selector.run_consensus_selection(X, y)
    selector.plot_feature_selection_summary()
    
    # Select top features identified by consensus and RFECV
    selected_features = ranking_df[ranking_df["In_RFECV_Optimal"]].index.tolist()
    if len(selected_features) < 6:
        selected_features = ranking_df.head(8).index.tolist()
    print(f"  [OK] Optimal Selected Features ({len(selected_features)}): {selected_features}")
    
    X_selected = X[selected_features].copy()

    # 5. Bayesian Hyperparameter Optimization (Optuna)
    print("\n[Step 4/7] Bayesian Hyperparameter Optimization with Optuna (Stratified 5-Fold CV)...")
    optimizer = HyperparameterOptimizer(n_trials=35, timeout=120, random_state=RANDOM_STATE)
    tuning_results = optimizer.tune_all_models(
        X_selected, y, models=["LightGBM", "XGBoost", "RandomForest", "SVM"]
    )

    # 6. Model Benchmarking & Evaluation
    print("\n[Step 5/7] Comprehensive Model Benchmarking on Stratified 5-Fold Cross Validation...")
    candidate_models = {
        "Optuna_LightGBM": create_model("LightGBM", tuning_results["LightGBM"]["best_params"]),
        "Optuna_XGBoost": create_model("XGBoost", tuning_results["XGBoost"]["best_params"]),
        "Optuna_RandomForest": create_model("RandomForest", tuning_results["RandomForest"]["best_params"]),
        "Optuna_SVM": create_model("SVM", tuning_results["SVM"]["best_params"]),
        "Baseline_LogisticRegression": create_model("LogisticRegression"),
        "Baseline_MLP": create_model("MLP")
    }

    evaluator = ModelEvaluator(n_splits=5, random_state=RANDOM_STATE)
    benchmark_df = evaluator.benchmark_models(candidate_models, X_selected, y)
    print("\n--- Model Benchmark Results (5-Fold Cross Validation) ---")
    print(benchmark_df.to_string(index=False))

    # Save benchmark curves
    evaluator.plot_roc_pr_curves(y)
    evaluator.plot_confusion_matrices(y)

    # 7. Select Champion Model and Train Full Preprocessor Pipeline
    print("\n[Step 6/7] Training Champion Model & Persisting Artifacts...")
    best_model_name = benchmark_df.iloc[0]["Model"]
    print(f"  Champion Model Selected: {best_model_name}")

    champion_raw_model = candidate_models[best_model_name]
    
    # Build complete end-to-end sklearn pipeline for inference
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    X_proc = preprocessor.fit_transform(X_selected)
    champion_raw_model.fit(X_proc, y)

    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", champion_raw_model)
    ])

    # Save artifacts
    joblib.dump(full_pipeline, MODELS_DIR / "best_model_pipeline.joblib")
    joblib.dump(champion_raw_model, MODELS_DIR / "champion_model.joblib")
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    
    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump(selected_features, f, indent=4)

    with open(REPORTS_DIR / "model_benchmark_report.json", "w") as f:
        json.dump(evaluator.results, f, indent=4)

    print(f"  [OK] Serialized model artifacts to {MODELS_DIR}")

    # 8. SHAP Explainability Engine
    print("\n[Step 7/7] Computing SHAP Global & Local Interpretability Visualizations...")
    # For tree explainability, choose a tree model (e.g. LightGBM or XGBoost or RF)
    explain_model = champion_raw_model
    if not hasattr(explain_model, "predict_proba"):
        explain_model = candidate_models["Optuna_LightGBM"]
        explain_model.fit(X_proc, y)

    shap_mgr = ShapExplainerManager(
        model=explain_model,
        X_train=X_proc,
        feature_names=selected_features,
        model_name=best_model_name
    )

    # Global explanations
    shap_mgr.plot_summary_beeswarm()
    shap_mgr.plot_bar_importance()

    # Dependence plots for top clinical and MRI biomarkers
    for feat in ["MMSE", "nWBV", "Age"]:
        if feat in selected_features:
            shap_mgr.plot_dependence(feat)

    # Local patient-level explanations (Example normal patient and impaired patient)
    normal_indices = np.where(y.values == 0)[0]
    impaired_indices = np.where(y.values == 1)[0]
    
    if len(normal_indices) > 0:
        norm_idx = int(normal_indices[0])
        shap_mgr.plot_patient_waterfall(
            norm_idx, save_path=str(FIGURES_DIR / "shap_waterfall_normal_case.png")
        )
        card_norm = shap_mgr.generate_clinical_narrative_card(
            norm_idx, patient_id_label=str(patient_ids.iloc[norm_idx]), true_label="Normal (CDR 0)"
        )
        print(f"\n[Case 1: Normal Subject {card_norm['patient_id']}]")
        print(f"  {card_norm['clinical_summary']}")

    if len(impaired_indices) > 0:
        imp_idx = int(impaired_indices[0])
        shap_mgr.plot_patient_waterfall(
            imp_idx, save_path=str(FIGURES_DIR / "shap_waterfall_impaired_case.png")
        )
        card_imp = shap_mgr.generate_clinical_narrative_card(
            imp_idx, patient_id_label=str(patient_ids.iloc[imp_idx]), true_label="Impaired (CDR > 0)"
        )
        print(f"\n[Case 2: Impaired Subject {card_imp['patient_id']}]")
        print(f"  {card_imp['clinical_summary']}")

    print("\n" + "=" * 75)
    print(" PIPELINE EXECUTION COMPLETE!")
    print(f" Generated Reports: {REPORTS_DIR}")
    print(f" Saved Model Pipeline: {MODELS_DIR / 'best_model_pipeline.joblib'}")
    print("=" * 75)


if __name__ == "__main__":
    run_pipeline()
