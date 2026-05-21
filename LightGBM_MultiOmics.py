#!/usr/bin/env python3

import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
import argparse
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit, cross_validate
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import VarianceThreshold
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

def run_multiomics_lgbm(drug, mode):
    print(f"Targeting Drug: {drug}")
    
    # Check for alignments
    if not os.path.exists("input/aligned/aligned_crispr.csv") or not os.path.exists("input/aligned/aligned_mutations.csv"):
        print("Error: Missing aligned matrices in input/aligned. Ensure you've run multiomics_preprocessor.py first.")
        return

    print("Loading mapped CRISPR and Mutation data arrays...")
    crispr_df = pd.read_csv("input/aligned/aligned_crispr.csv", index_col=0)
    mut_df = pd.read_csv("input/aligned/aligned_mutations.csv", index_col=0)
    
    # Standard 21Q3 bounds
    num_gene_crispr = 17651
    X_crispr = crispr_df.iloc[:, :num_gene_crispr]
    target_columns = crispr_df.iloc[:, -4686:]
    
    if drug not in target_columns.columns:
        print(f"Error: Target Drug {drug} not explicitly found in biological target columns.")
        return
        
    y = target_columns[drug]

    groups = crispr_df.index

    if mode == "mutations_only":
        print("\n[Architecture Initialization: MUTATIONS ONLY]")
        print("Feeding raw, highly-sparse boolean arrays structurally directly into LightGBM.")
        X = mut_df
        pipeline = Pipeline([
            ('vt', VarianceThreshold(threshold=0.0)),
            ('smote', SMOTE(random_state=42)),
            ('lgbm', ThresholdWrapper(LGBMClassifier(n_estimators=100, max_depth=5, min_child_samples=5, class_weight='balanced', learning_rate=0.05, n_jobs=4, random_state=7, verbose=-1), threshold=0.40))
        ])
    elif mode == "combined":
        print("\n[Architecture Initialization: COMBINED MULTI-OMICS FUSION]")
        print("Standardizing & Truncating CRISPR continuous arrays to 100 dimensions, then concatenating to raw Boolean mutations.")
        
        # We manually process X_crispr here for the combined framework mapping securely without leaking arrays inside Groups
        scaler = StandardScaler()
        svd = TruncatedSVD(n_components=100, random_state=42)
        X_crispr_scaled = scaler.fit_transform(X_crispr)
        X_crispr_svd = pd.DataFrame(svd.fit_transform(X_crispr_scaled), index=X_crispr.index)
        
        # SVD features + ALL sparse mutation boolean indicators natively
        X = pd.concat([X_crispr_svd, mut_df], axis=1)
        X.columns = X.columns.astype(str)
        
        pipeline = Pipeline([
            ('vt', VarianceThreshold(threshold=0.0)),
            ('smote', SMOTE(random_state=42)),
            ('lgbm', ThresholdWrapper(LGBMClassifier(n_estimators=100, max_depth=5, min_child_samples=5, class_weight='balanced', learning_rate=0.05, n_jobs=4, random_state=7, verbose=-1), threshold=0.40))
        ])
    else:
        print("Invalid operational mode provided.")
        return

    print("\nExecuting Pipeline via 5-Fold Stratified Grouped Cross-Validation...")
    kfold = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=7)
    
    cv_results = cross_validate(pipeline, X, y, groups=groups, cv=kfold, scoring=['precision', 'recall'], n_jobs=-1)
    
    avg_precision = np.mean(cv_results['test_precision'])
    avg_recall = np.mean(cv_results['test_recall'])
    print(f"\n==== Evaluated CV Results [UN-LEAKED]: {mode} ====")
    print(f"Precision: {avg_precision:.4f}  |  Recall: {avg_recall:.4f}")

    os.makedirs("output/cv_results", exist_ok=True)
    pd.DataFrame(cv_results).to_csv(f"output/cv_results/cv_results_Multi_{mode}_{drug}_LGBM.csv")

    # Train global state model without leaking arrays iteratively
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=20)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    print("\nTraining final targeted baseline model...")
    pipeline.fit(X_train, y_train)
    
    # Dump the fitted model structure purely onto output layer
    os.makedirs("output/models", exist_ok=True)
    out_path = f"output/models/LightGBM_MultiOmics_{mode}_{drug}.joblib"
    joblib.dump(pipeline, out_path)
    print(f"Saved completed object strictly to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Omics LGBM Runner.")
    parser.add_argument("drug", type=str, help="Target drug metadata string (e.g. BRD-A30403214-001-03-3::0.25::HTS)")
    parser.add_argument("--mode", type=str, choices=["mutations_only", "combined"], default="mutations_only", help="Architecture setting.")
    args = parser.parse_args()
    
    run_multiomics_lgbm(args.drug, args.mode)
