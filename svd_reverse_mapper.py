#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import joblib

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 svd_reverse_mapper.py <DRUG_NAME>")
        sys.exit(1)
        
    drug_name = sys.argv[1]
    
    # Paths
    model_path = f"output/models/LightGBM_SVD_{drug_name}_leakproof.joblib"
    svd_path = f"output/models/LightGBM_SVD_transformer_{drug_name}.joblib"
    
    # 1. Load the leak-proof LightGBM model and extract feature importances
    try:
        lgbm_model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}.")
        print("Make sure you have trained the model using LightGBM_SVD.py first.")
        sys.exit(1)
        
    importances = lgbm_model.feature_importances_
    
    # 2. Load the fitted TruncatedSVD transformer and extract components matrix
    try:
        svd_transformer = joblib.load(svd_path)
    except FileNotFoundError:
        print(f"Error: SVD transformer not found at {svd_path}.")
        sys.exit(1)
        
    components = svd_transformer.components_ # shape should be (100, 17651)
    
    if len(importances) != components.shape[0]:
        print(f"Warning: LightGBM features ({len(importances)}) != SVD components ({components.shape[0]}).")
        print("This could happen if VarianceThreshold removed some SVD components during training.")
        
    # 3. Perform a dot product between importances and the absolute values of SVD components_ matrix
    print(f"Calculating reverse mapping for {drug_name}...")
    gene_scores = np.dot(importances, np.abs(components))
    
    # 4. Map the 17,651 scores back to DepMap gene names
    print("Loading dataset headers to map gene names...")
    # Read only the header row to save memory and time
    dataset_headers = pd.read_csv("input/combined_DepMap_21Q3.csv", nrows=0)
    
    num_gene = 17651
    gene_names = dataset_headers.columns[1:num_gene+1]
    
    if len(gene_scores) != len(gene_names):
        print(f"Error: Resulting dot-product shape ({len(gene_scores)}) does not match gene columns shape ({len(gene_names)}).")
        sys.exit(1)
        
    mapping_df = pd.DataFrame({
        "Gene_Name": gene_names,
        "Reverse_Importance_Score": gene_scores
    })
    
    # Sort descending
    mapping_df_sorted = mapping_df.sort_values(by="Reverse_Importance_Score", ascending=False).reset_index(drop=True)
    
    # 5. Print the top 20 genes
    print("\n" + "="*60)
    print(f"Top 20 Genes Driving the LightGBM SVD Model for {drug_name}")
    print("="*60)
    for i in range(20):
        gene_name = mapping_df_sorted.loc[i, "Gene_Name"]
        score = mapping_df_sorted.loc[i, "Reverse_Importance_Score"]
        print(f"{i+1:2d}. {gene_name:<35} {score:.4f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
