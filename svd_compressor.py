import pandas as pd
import numpy as np
import joblib
import os
from sklearn.decomposition import TruncatedSVD

def main():
    print("Loading the full DepMap dataset...")
    # Load dataset
    dataset = pd.read_csv("input/combined_DepMap_21Q3.csv")
    
    num_gene = 17651
    # Extract gene expression data (columns 1 to 17651)
    X = dataset.iloc[:, 1:num_gene+1]
    
    # Do NOT scale the data per instructions
    
    # Initialize TruncatedSVD with 100 components
    print("Fitting TruncatedSVD(n_components=100)...")
    svd = TruncatedSVD(n_components=100, random_state=42)
    
    # Fit and transform
    X_transformed = svd.fit_transform(X)
    
    # Print the total explained variance ratio
    explained_variance_ratio_sum = np.sum(svd.explained_variance_ratio_)
    print(f"Total Explained Variance Ratio (100 components): {explained_variance_ratio_sum:.4f}")
    
    # Save the fitted SVD transformer object
    os.makedirs("output/models", exist_ok=True)
    transformer_path = "output/models/svd_transformer.joblib"
    joblib.dump(svd, transformer_path)
    print(f"Saved SVD transformer to {transformer_path}")
    
    # Save the transformed data
    # Create column names for the SVD components
    svd_cols = [f"SVD_Comp_{i+1}" for i in range(100)]
    df_transformed = pd.DataFrame(X_transformed, columns=svd_cols)
    
    # Re-attach the ID column (index 0) and the drug targets (from num_gene+1 to end)
    df_labels = dataset.iloc[:, num_gene+1:]
    id_col = dataset.iloc[:, 0]
    
    print("Constructing the compressed dataset...")
    df_final = pd.concat([id_col, df_transformed, df_labels], axis=1)
    
    # Save as CSV named svd_compressed_data.csv
    out_csv = "input/svd_compressed_data.csv"
    print(f"Saving compressed data to {out_csv}...")
    df_final.to_csv(out_csv, index=False)
    print("SVD compression and export complete!")

if __name__ == "__main__":
    main()
