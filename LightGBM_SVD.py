#!/usr/bin/env python3

import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline # Standard sklearn pipeline is safe now!
from sklearn.decomposition import TruncatedSVD
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

# 1. Load Data
print("Loading DepMap dataset...")
dataset = pd.read_csv("input/combined_DepMap_21Q3.csv")
num_gene = 17651

X = dataset.iloc[:, 1:num_gene+1]
drug_y_all = dataset.iloc[:, -4686:]

def run_lgbm(drug):
    print(f"Targeting Drug: {drug} with LightGBM (100 SVD Features)")
    
    print("\n=========================================")
    print("      PHASE 1: LEAKED PIPELINE (OLD)     ")
    print("=========================================")
    y_leaked = drug_y_all[drug]
    X_leaked = X
    
    pipeline_leaked = Pipeline([
        ('scaler', StandardScaler()),
        ('svd', TruncatedSVD(n_components=100, random_state=42)),
        ('vt', VarianceThreshold(threshold=0.0)),
        ('lgbm', ThresholdWrapper(LGBMClassifier(n_estimators=150, class_weight='balanced', learning_rate=0.05, n_jobs=-1, random_state=7, verbose=-1), threshold=0.40))
    ])
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    cv_results_leaked = cross_validate(pipeline_leaked, X_leaked, y_leaked, cv=kfold, scoring=['precision', 'recall'], n_jobs=1)
    print(f"[LEAKED] Average Precision: {np.mean(cv_results_leaked['test_precision']):.4f}")
    print(f"[LEAKED] Average Recall:    {np.mean(cv_results_leaked['test_recall']):.4f}")
    
    print("\n=========================================")
    print("     PHASE 2: CLEAN PIPELINE (NEW)       ")
    print("=========================================")
    print("Executing Grouped Splitting to retain all technical replicates while sealing leakage...")
    
    cell_col = dataset.columns[0]
    groups = dataset[cell_col]
    
    pipeline_clean = Pipeline([
        ('scaler', StandardScaler()),
        ('svd', TruncatedSVD(n_components=100, random_state=42)),
        ('vt', VarianceThreshold(threshold=0.0)),
        ('lgbm', ThresholdWrapper(LGBMClassifier(
            n_estimators=100, 
            max_depth=5,
            min_child_samples=5,
            class_weight='balanced', 
            learning_rate=0.03, 
            n_jobs=-1, 
            random_state=7, 
            verbose=-1
        ), threshold=0.40))
    ])
    
    kfold_group = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=7)
    cv_results_clean = cross_validate(pipeline_clean, X, y_leaked, groups=groups, cv=kfold_group, scoring=['precision', 'recall'], n_jobs=1)
    print(f"[CLEAN] Average Precision: {np.mean(cv_results_clean['test_precision']):.4f}")
    print(f"[CLEAN] Average Recall:    {np.mean(cv_results_clean['test_recall']):.4f}")
    
    print("\nTraining final un-leaked grouped model...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=20)
    train_idx, test_idx = next(gss.split(X, y_leaked, groups=groups))
    
    X_train_c, X_test_c = X.iloc[train_idx], X.iloc[test_idx]
    y_train_c, y_test_c = y_leaked.iloc[train_idx], y_leaked.iloc[test_idx]
    
    print("Fitting Scaler/SVD on explicit training bounds...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_c)
    X_test_scaled = scaler.transform(X_test_c)
    
    svd = TruncatedSVD(n_components=100, random_state=42)
    X_train_svd = svd.fit_transform(X_train_scaled)
    X_test_svd = svd.transform(X_test_scaled)
    
    vt = VarianceThreshold(threshold=0.0)
    X_train_filtered = vt.fit_transform(X_train_svd)
    X_test_filtered = vt.transform(X_test_svd)
    
    lgbm_clean = ThresholdWrapper(LGBMClassifier(
        n_estimators=100, max_depth=5, min_child_samples=5, class_weight='balanced', learning_rate=0.03, n_jobs=-1, random_state=7, verbose=-1
    ), threshold=0.40)
    
    print("Training cleanly bounded model...")
    lgbm_clean.fit(X_train_filtered, y_train_c)
    
    y_pred = lgbm_clean.predict(X_test_filtered)
    print("\n--- Un-Leaked Model Test Evaluation ---")
    print("Confusion Matrix:\n", confusion_matrix(y_test_c, y_pred))
    
    model_report = classification_report(y_test_c, y_pred, output_dict=True, labels=np.unique(y_pred))
    os.makedirs("output/reports", exist_ok=True)
    pd.DataFrame(model_report).transpose().to_csv(f"output/reports/classification_report_SVD_{drug}_LGBM.csv")
    
    os.makedirs("output/models", exist_ok=True)
    joblib.dump(scaler, f"output/models/LightGBM_SVD_scaler_{drug}.joblib")
    joblib.dump(svd, f"output/models/LightGBM_SVD_transformer_{drug}.joblib")
    joblib.dump(lgbm_clean, f"output/models/LightGBM_SVD_{drug}_leakproof.joblib")
    
    print("Pipeline Complete. Results saved to /output")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Please provide a drug name. Example: python LightGBM_SVD.py Paclitaxel")
        sys.exit(1)
        
    starttime = time.time()
    drug_name = sys.argv[1]
    run_lgbm(drug_name)
    endtime = time.time()
    
    with open('time_log_LGBM.txt', 'a') as f:
        f.write(f"SVD_{drug_name}\t{endtime-starttime}s\n")
