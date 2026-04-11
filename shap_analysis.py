import os
import glob
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

import sys

def main():
    if len(sys.argv) > 1:
        drug_name = sys.argv[1]
        model_path = f"output/models/LightGBM_{drug_name}_leakproof.joblib"
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}.")
            return
    else:
        # Find a model
        model_files = glob.glob("output/models/LightGBM_*_leakproof.joblib")
        if not model_files:
            print("No LightGBM models found in output/models/.")
            # Fallback to output/ in case it hasn't been moved yet
            model_files = glob.glob("output/LightGBM_*_leakproof.joblib")
            if not model_files:
                print("No LightGBM models found.")
                return

        model_path = model_files[0]
        # Extract drug name
        filename = os.path.basename(model_path)
        drug_name = filename.replace("LightGBM_", "").replace("_leakproof.joblib", "")
    
    print(f"Using model: {model_path}")
    print(f"Drug name: {drug_name}")
    
    # Load model
    lgbm = joblib.load(model_path)
    
    # Load data
    print("Loading dataset...")
    dataset = pd.read_csv("input/combined_DepMap_21Q3.csv")
    num_gene = 17651
    X = dataset.iloc[:, 1:num_gene+1]
    drug_y_all = dataset.iloc[:, -4686:]
    
    if drug_name not in drug_y_all.columns:
        print(f"Error: Drug {drug_name} not found in dataset columns.")
        return
        
    y = drug_y_all[drug_name]
    
    # Recreate the train/test split to get the test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20, stratify=y)
    
    # Apply the same variance threshold
    vt = VarianceThreshold(threshold=0.0)
    vt.fit(X_train)
    X_test_filtered = vt.transform(X_test)
    
    # Create DataFrame to keep feature names for SHAP
    feature_names = vt.get_feature_names_out(X.columns)
    X_test_df = pd.DataFrame(X_test_filtered, columns=feature_names)
    
    # Sample 500 rows to save memory
    sample_size = min(500, len(X_test_df))
    X_sample = X_test_df.sample(sample_size, random_state=42)
    
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(lgbm)
    shap_values = explainer.shap_values(X_sample)
    
    # In some versions of SHAP with LightGBM binary classification, 
    # shap_values is a list of [negative_class_values, positive_class_values]
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values
        
    print("Generating SHAP summary plot...")
    plt.figure()
    shap.summary_plot(shap_values_to_plot, X_sample, show=False)
    
    os.makedirs("output/shap_plots", exist_ok=True)
    plot_path = f"output/shap_plots/shap_summary_{drug_name}.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved SHAP summary plot to {plot_path}")

if __name__ == "__main__":
    main()
