#!/usr/bin/env python3

import sys
import os
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 shap_svd_mapper.py <DRUG_NAME>")
        sys.exit(1)
        
    drug_name = sys.argv[1]
    
    # Target paths
    model_path = f"output/models/LightGBM_SVD_{drug_name}_leakproof.joblib"
    svd_path = f"output/models/LightGBM_SVD_transformer_{drug_name}.joblib"
    scaler_path = f"output/models/LightGBM_SVD_scaler_{drug_name}.joblib"
    
    # Validations
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Did you run the LightGBM_SVD step?")
        sys.exit(1)
    if not os.path.exists(svd_path):
        print(f"Error: SVD transformer not found at {svd_path}.")
        sys.exit(1)
    if not os.path.exists(scaler_path):
        print(f"Error: Scaler not found at {scaler_path}.")
        sys.exit(1)

    print("Loading models, transformers, and scalers...")
    lgbm_model = joblib.load(model_path)
    svd_transformer = joblib.load(svd_path)
    scaler = joblib.load(scaler_path)

    print("Loading DepMap dataset and sampling 200 rows to save memory...")
    dataset = pd.read_csv("input/combined_DepMap_21Q3.csv")
    
    num_gene = 17651
    # Full dataset features natively
    X_full = dataset.iloc[:, 1:num_gene+1]
    
    # 1. Take a random sample of 200 rows
    X_sample = X_full.sample(n=200, random_state=42)
    gene_names = X_sample.columns.tolist()
    
    # 2. Transform the 200 rows using scaler then SVD
    print("Scaling and applying SVD transformations onto the sample...")
    X_scaled = scaler.transform(X_sample)
    X_svd = svd_transformer.transform(X_scaled)
    
    # 3. Calculate SHAP values on LightGBM tree (Outputs 200x100 matrix)
    print("Calculating native SHAP arrays across SVD components...")
    explainer = shap.TreeExplainer(lgbm_model)
    shap_vals_svd = explainer.shap_values(X_svd)
    
    # LightGBM handles class arrays natively via shap values lists
    if isinstance(shap_vals_svd, list):
        shap_vals_svd = shap_vals_svd[1] # Map structurally to the positive classifier
        
    print(f" -> Output tree matrix shape: {np.array(shap_vals_svd).shape}")
    
    components = svd_transformer.components_ # 100x17651
    print(f" -> Input SVD matrix shape: {components.shape}")
    
    # 4. Perform Dot Product (200x100) dot (100x17651)
    print("Mathematically aligning backpropagation maps onto raw feature bounds...")
    true_shap_vals = np.dot(shap_vals_svd, components)
    
    # 5. Output mapped summary plot
    print(f" -> Transposed true SHAP matrix shape: {true_shap_vals.shape}")
    print("Rendering SHAP summary plot silently...")
    plt.figure()
    shap.summary_plot(true_shap_vals, X_sample, feature_names=gene_names, show=False)
    
    os.makedirs("output/shap_plots", exist_ok=True)
    out_file = f"output/shap_plots/true_svd_mapped_shap_{drug_name}.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully processed script and saved {out_file}!")

if __name__ == "__main__":
    main()
