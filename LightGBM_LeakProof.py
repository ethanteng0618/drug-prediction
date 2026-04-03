#!/usr/bin/env python3

import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline # Standard sklearn pipeline is safe now!

# 1. Load Data
print("Loading DepMap dataset...")
dataset = pd.read_csv("input/combined_DepMap_21Q3.csv")
num_gene = 17651

X = dataset.iloc[:, 1:num_gene+1]
drug_y_all = dataset.iloc[:, -4686:]

def run_lgbm(drug):
    print(f"Targeting Drug: {drug} with LightGBM")
    y = drug_y_all[drug]

    # 2. Train/Test Split FIRST
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20, stratify=y)

    # 3. Apply Variance Threshold
    vt = VarianceThreshold(threshold=0.0)
    X_train_filtered = vt.fit_transform(X_train)
    X_test_filtered = vt.transform(X_test)

    # 4. Train LightGBM (Notice: No OverSampler needed!)
    # class_weight='balanced' handles the rare cases automatically
    # n_jobs=-1 uses all your Mac's CPU cores
    lgbm = LGBMClassifier(
        n_estimators=150, 
        class_weight='balanced', 
        learning_rate=0.05,
        n_jobs=-1, 
        random_state=7,
        verbose=-1 # Turns off annoying LightGBM warning logs
    ) 
    
    print("Training model...")
    lgbm.fit(X_train_filtered, y_train)

    # Save model
    os.makedirs("output", exist_ok=True)
    joblib.dump(lgbm, f"output/LightGBM_{drug}_leakproof.joblib")

    # 5. Evaluate on the UNTOUCHED Test Data
    y_pred = lgbm.predict(X_test_filtered)

    print("\n--- Model Evaluation ---")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    model_report = classification_report(y_test, y_pred, output_dict=True, labels=np.unique(y_pred))
    pd.DataFrame(model_report).transpose().to_csv(f"output/classification_report_{drug}_LGBM.csv")

    # 6. Cross-Validation (Using standard Pipeline since we dropped the OverSampler)
    print("\nRunning Cross-Validation...")
    pipeline = Pipeline([
        ('vt', VarianceThreshold(threshold=0.0)),
        ('lgbm', LGBMClassifier(n_estimators=150, class_weight='balanced', learning_rate=0.05, n_jobs=-1, random_state=7, verbose=-1))
    ])

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    cv_results = cross_validate(pipeline, X, y, cv=kfold, scoring=['accuracy', 'precision', 'recall', 'f1'], n_jobs=1)
    pd.DataFrame(cv_results).to_csv(f"output/cv_results_{drug}_LGBM.csv")

    print("\nPipeline Complete. Results saved to /output")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Please provide a drug name. Example: python LightGBM_LeakProof.py Paclitaxel")
        sys.exit(1)
        
    starttime = time.time()
    drug_name = sys.argv[1]
    run_lgbm(drug_name)
    endtime = time.time()
    
    with open('time_log_LGBM.txt', 'a') as f:
        f.write(f"{drug_name}\t{endtime-starttime}s\n")