#!/usr/bin/env python3

import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline # CRITICAL: imblearn's pipeline, not sklearn's

# 1. Load Data
print("Loading DepMap dataset...")
dataset = pd.read_csv("input/combined_DepMap_21Q3.csv")
num_gene = 17651

X = dataset.iloc[:, 1:num_gene+1]
drug_y_all = dataset.iloc[:, -4686:]

def run_rf_corrected(drug):
    print(f"Targeting Drug: {drug}")
    y = drug_y_all[drug]

    # 2. Train/Test Split FIRST (Fixing the Data Leakage)
    # stratify=y ensures the 80/20 split maintains the same ratio of effective/ineffective cases
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20, stratify=y)

    # 3. Apply Variance Threshold (Feature Selection)
    # Drops genes whose expression doesn't change at all (variance = 0)
    vt = VarianceThreshold(threshold=0.0)
    X_train_filtered = vt.fit_transform(X_train)
    X_test_filtered = vt.transform(X_test) # Apply the exact same filter to the test set

    # 4. Oversample ONLY the Training Data
    ros = RandomOverSampler(random_state=72)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train_filtered, y_train)

    # 5. Train the Model (Optimized for Apple Silicon)
    # n_jobs=-1 tells the script to use all available CPU cores on your machine
    rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=7) 
    rfc.fit(X_train_resampled, y_train_resampled)

    # Save model
    os.makedirs("output/models", exist_ok=True)
    joblib.dump(rfc, f"output/models/RandomForest_{drug}_leakproof.joblib")

    # 6. Evaluate on the UNTOUCHED Test Data
    y_pred = rfc.predict(X_test_filtered)

    print("\n--- Model Evaluation ---")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    model_report = classification_report(y_test, y_pred, output_dict=True, labels=np.unique(y_pred))
    os.makedirs("output/reports", exist_ok=True)
    pd.DataFrame(model_report).transpose().to_csv(f"output/reports/classification_report_{drug}_leakproof.csv")

    # 7. Cross-Validation (Safely built with an imblearn Pipeline)
    # This prevents data leakage across the 5 different testing folds
    print("\nRunning Cross-Validation (This may take a minute)...")
    pipeline = Pipeline([
        ('vt', VarianceThreshold(threshold=0.0)),
        ('ros', RandomOverSampler(random_state=72)),
        ('rfc', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=7))
    ])

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    cv_results = cross_validate(pipeline, X, y, cv=kfold, scoring=['accuracy', 'precision', 'recall', 'f1'], n_jobs=-1)
    os.makedirs("output/cv_results", exist_ok=True)
    pd.DataFrame(cv_results).to_csv(f"output/cv_results/cv_results_{drug}_leakproof.csv")

    print("\nPipeline Complete. Results saved to /output")

if __name__ == "__main__":
    # Checks if a drug name was passed in the terminal command
    if len(sys.argv) < 2:
        print("Error: Please provide a drug name. Example: python RandomForest_LeakProof.py Paclitaxel")
        sys.exit(1)
        
    starttime = time.time()
    drug_name = sys.argv[1]
    run_rf_corrected(drug_name)
    endtime = time.time()
    
    with open('time_log_leakproof.txt', 'a') as f:
        f.write(f"{drug_name}\t{endtime-starttime}s\n")