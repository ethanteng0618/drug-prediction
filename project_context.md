# UMD IHC Drug Prediction Project Context

## Overview
This project predicts drug response using the `combined_DepMap_21Q3.csv` dataset. The feature set comprises 17,651 gene expression columns, and the target vectors are drug sensitivity scores. 

## Data Architecture
- **Input Data**: `input/combined_DepMap_21Q3.csv`
- **Features (X)**: Columns 1 through 17651 (exclusive of the 0th index column).
- **Targets (Y)**: Columns from -4686 to the end. These are drug responses.

## Key Scripts
- **`LightGBM_LeakProof.py`**: Trains an `LGBMClassifier` using a specific drug. Applies a `VarianceThreshold(threshold=0.0)` for feature selection to remove zero-variance features before passing to LightGBM. Includes cross-validation via `StratifiedKFold`. 
  *Note*: `cross_validate` was modified to use `n_jobs=1` to prevent macOS deadlocks when LightGBM runs OpenMP threads simultaneously.
- **`RandomForest_LeakProof.py`**: A random forest counterpart to `LightGBM_LeakProof.py` for comparative baseline models.
- **`dashboard.py`**: A Streamlit application built to visualize LightGBM Feature Importances for the tested drug `BRD-K62008436-001-22-1::2.5::HTS`.

## Open Issues / Notes
- The model files (`.joblib`) are saved in the `output/` directory.
- `classification_report` and `cv_results` CSVs are also stored in `output/` for each drug.
- If nested parallelization deadlocks occur during cross-validation with other algorithms on a Mac, ensure `scikit-learn` uses `n_jobs=1`.

## Quick Start Commands

To train a LightGBM model for a given drug:
```bash
python LightGBM_LeakProof.py [DRUG_NAME]
```

To run the visualization dashboard:
```bash
streamlit run dashboard.py
```
