#!/usr/bin/env python3

import os
import sys
import time
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from lightgbm import LGBMClassifier

def run_shap_pipeline(drug_name):
    print(f"Loading DepMap dataset to run SHAP pipeline explainer for {drug_name}...")
    dataset = pd.read_csv("input/combined_DepMap_21Q3.csv")
    num_gene = 17651
    X = dataset.iloc[:, 1:num_gene+1]
    
    if drug_name not in dataset.columns:
        print(f"Error: Drug {drug_name} not found in target columns.")
        sys.exit(1)
        
    y = dataset[drug_name]
    
    print("Splitting data into training and validation sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Constructing and fitting the FULL Pipeline (Scaler -> SVD -> LGBM)...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svd', TruncatedSVD(n_components=100, random_state=42)),
        ('lgbm', LGBMClassifier(n_estimators=150, class_weight='balanced', learning_rate=0.05, n_jobs=-1, random_state=7, verbose=-1))
    ])
    
    pipeline.fit(X_train, y_train)
    
    print("\nExtracting background and test datasets (CRITICAL FOR MEMORY)...")
    # Using shap.kmeans to drastically condense the massive X_train down to just 10 representative background rows
    print("Condensing X_train into 10 background clusters using K-Means...")
    X_train_summary = shap.kmeans(X_train, 10)
    
    # Selecting exactly 5 random rows for the test dataset
    X_test_subset = X_test.sample(n=5, random_state=42)
    
    print("\nInitializing shap.KernelExplainer...")
    # The Explainer needs to predict probabilties for class 1 (positive responsive class)
    def predict_function(X_data):
        return pipeline.predict_proba(X_data)[:, 1]

    # Initialize a model-agnostic KernelExplainer passing the entire pipeline prediction logic
    explainer = shap.KernelExplainer(predict_function, X_train_summary)
    
    print("\nCalculating SHAP values for the 5-row test dataset (this will take a moment)...")
    start_time = time.time()
    
    # Suppress output logs of KernelExplainer progress bar since it loops explicitly
    shap_values = explainer.shap_values(X_test_subset)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"SHAP calculation for just 5 rows took: {elapsed_time:.2f} seconds.")
    
    # Estimate time for a standard 500-row test dataset
    est_500 = (elapsed_time / 5.0) * 500.0
    
    print(f"\n=================================================================")
    print(f"⚠️ WARNING: COMPUTATIONAL TIME ESTIMATE ⚠️")
    print(f"Running this KernelExplainer on a full 500-row test dataset")
    print(f"would theoretically take approximately {est_500/60:.2f} minutes ({est_500/3600:.2f} hours).")
    print(f"")
    print(f"Model-agnostic explainers passing data through 17k raw features")
    print(f"are extremely computationally expensive compared to native TreeExplainers!")
    print(f"=================================================================\n")
    
    print("Generating SHAP summary plot on RAW gene space...")
    os.makedirs("output/shap_plots", exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_subset, feature_names=X.columns, show=False)
    
    output_path = "output/shap_plots/pipeline_shap_raw.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Successfully saved plot to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 shap_pipeline_explainer.py [DRUG_NAME]")
        target_drug = "BRD-A30403214-001-03-3::0.25::HTS"
        print(f"No drug specified. Defaulting to: {target_drug}")
    else:
        target_drug = sys.argv[1]
    
    run_shap_pipeline(target_drug)
