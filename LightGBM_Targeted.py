#!/usr/bin/env python3

import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit, cross_validate
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin

class ThresholdWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, threshold=0.40):
        self.estimator = estimator
        self.threshold = threshold
    def fit(self, X, y, **kwargs):
        self.estimator.fit(X, y, **kwargs)
        self.classes_ = self.estimator.classes_
        return self
    def predict(self, X):
        probs = self.estimator.predict_proba(X)
        return self.classes_[(probs[:, 1] >= self.threshold).astype(int)]
    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

def run_targeted_lgbm(drug):
    print(f"Targeting Drug: {drug} with Targeted Biological Filtering (Top 200 Genes)")
    
    print("Loading DepMap dataset...")
    dataset = pd.read_csv("input/combined_DepMap_21Q3.csv")
    num_gene = 17651
    
    X = dataset.iloc[:, 1:num_gene+1]
    
    target_cols = dataset.iloc[:, -4686:]
    if drug not in target_cols.columns:
        print(f"Error: Drug '{drug}' not found in the dataset targets.")
        sys.exit(1)
        
    y = target_cols[drug]
    cell_col = dataset.columns[0]
    groups = dataset[cell_col]
    
    print("\n=========================================")
    print("   TARGETED AI PIPELINE (UN-LEAKED)      ")
    print("=========================================")
    print("Executing Grouped Splitting to seal leakage boundaries...")
    print("Extracting explicitly targeted features dynamically...")

    # We use a base LightGBM to identify the Top 200 Biological Features natively without SVD blurring.
    feature_selector = SelectFromModel(
        LGBMClassifier(n_estimators=100, class_weight='balanced', learning_rate=0.05, n_jobs=4, random_state=42, verbose=-1),
        max_features=200,
        threshold=-np.inf
    )
    
    pipeline_clean = Pipeline([
        ('scaler', StandardScaler()),
        ('vt', VarianceThreshold(threshold=0.0)),
        ('feature_selector', feature_selector),
        ('lgbm', ThresholdWrapper(LGBMClassifier(
            n_estimators=100, 
            max_depth=5,
            min_child_samples=5,
            class_weight='balanced', 
            learning_rate=0.03, 
            n_jobs=4, 
            random_state=7, 
            verbose=-1
        ), threshold=0.40))
    ])
    
    kfold_group = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=7)
    
    print("Running Cross-Validation across targeted features...")
    cv_results = cross_validate(pipeline_clean, X, y, groups=groups, cv=kfold_group, scoring=['precision', 'recall'], n_jobs=-1)
    
    avg_precision = np.mean(cv_results['test_precision'])
    avg_recall = np.mean(cv_results['test_recall'])
    print(f"[TARGETED] Average Precision: {avg_precision:.4f}")
    print(f"[TARGETED] Average Recall:    {avg_recall:.4f}")
    
    os.makedirs("output/cv_results", exist_ok=True)
    pd.DataFrame(cv_results).to_csv(f"output/cv_results/cv_results_Targeted_{drug}_LGBM.csv")
    
    print("\nTraining final targeted baseline model...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=20)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    print("Isolating pure predictive biological features...")
    pipeline_clean.fit(X_train, y_train)
    
    # Extract the names of the top 200 genes dynamically for the user:
    vt_mask = pipeline_clean.named_steps['vt'].get_support()
    X_train_vt_cols = X_train.columns[vt_mask]
    fs_mask = pipeline_clean.named_steps['feature_selector'].get_support()
    top_genes = X_train_vt_cols[fs_mask]
    
    print("\n*** Top 10 High-Vulnerability Biological Genes Mapped ***")
    print(", ".join(top_genes[:10].to_list()) + " ... (Total Extracted: 200)")

    y_pred = pipeline_clean.predict(X_test)
    
    print("\n--- Targeted Model Test Evaluation ---")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    model_report = classification_report(y_test, y_pred, output_dict=True, labels=np.unique(y_pred))
    
    os.makedirs("output/reports", exist_ok=True)
    os.makedirs("output/models", exist_ok=True)
    
    # Generic naming convention universally matching the inputted drug
    report_path = f"output/reports/classification_report_Targeted_{drug}.csv"
    model_path = f"output/models/LightGBM_Targeted_{drug}_leakproof.joblib"
    
    pd.DataFrame(model_report).transpose().to_csv(report_path)
    joblib.dump(pipeline_clean, model_path)
    
    print(f"\nPipeline Complete. Outputs strictly saved to:")
    print(report_path)
    print(model_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Please provide a drug name. Example: python LightGBM_Targeted.py Paclitaxel")
        sys.exit(1)
        
    starttime = time.time()
    drug_name = sys.argv[1]
    run_targeted_lgbm(drug_name)
    endtime = time.time()
    
    with open('time_log_LGBM.txt', 'a') as f:
        f.write(f"Targeted_{drug_name}\t{endtime-starttime}s\n")
