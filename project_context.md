# UMD IHC Drug Prediction Project Context

## Overview
This project predicts drug response using the `combined_DepMap_21Q3.csv` dataset. The feature set comprises 17,651 gene expression columns, and the target vectors are drug sensitivity scores. 

## Data Architecture
- **Input Data**: `input/combined_DepMap_21Q3.csv`
- **Features (X)**: Columns 1 through 17651 (exclusive of the 0th index column).
- **Targets (Y)**: Columns from -4686 to the end. These are drug responses.

## Key Scripts
- **`LightGBM_LeakProof.py`**: Trains an `LGBMClassifier` securely against data leakage natively using a standard scaler (for original unextrapolated targets) alongside `VarianceThreshold(threshold=0.0)`. Evaluates cross-validation via `StratifiedKFold`. 
  *Note*: `cross_validate` operates with `n_jobs=1` to prevent macOS deadlocks over native OpenMP thread conflicts.
- **`RandomForest_LeakProof.py`**: A random forest counterpart to `LightGBM_LeakProof.py` utilized for comparative baseline structural models.
- **`LightGBM_SVD.py`**: A secondary predictive architectural path that reduces complex 17k+ genetic expressions into natively scalable structural bounds passing `StandardScaler` limits over `TruncatedSVD(n_components=100)`. Extensively leak-proofed by manually securing fit properties across explicitly designated `X_train` bindings before passing dynamically onto the CV Pipeline.
- **`svd_compressor.py`**: Original data transformation loop yielding `input/svd_compressed_data.csv` (No longer the strict functional dependency, maintained for logic validation purposes).
- **`svd_reverse_mapper.py`**: Mathematically reverses array limits from standard `LightGBM_SVD` importances via dot product with the stored SVD `components_` matrix absolute array, returning actionable top 20 native Gene predictions in string formats natively from 100 aggregated structural nodes.
- **`shap_analysis.py`**: SHAP explainer configuration that natively parses `LightGBM_LeakProof` validation subsets and outputs beeswarm plots silently formatted via `matplotlib`.
- **`shap_svd_mapper.py`**: Reconstructs SHAP visual validations across SVD parameters by scaling SHAP derivations matrix-wise `(200x100) * (100x17651)` directly mapping real world feature weights cleanly into final `shap.summary_plot` visuals matching baseline graphs.
- **`dashboard.py`**: A Streamlit application built to visualize LightGBM Feature Importances dynamically. Features interactive configurations transitioning transparently between **Original (Raw Genes)** architectures securely and **SVD Extrapolated** operations seamlessly reverse mapping SVD matrices mathematically mid-render for UI graphing.

## Output Structure (`output/`)
The workspace leverages distinct subdirectories mapping predictive states precisely:
- `output/models/`: Joblib `.joblib` objects storing discrete trained classifiers, VarianceThresholds, StandardScalers, and SVD Transformers.
- `output/reports/`: Extracted binary/multiclass standard classification prediction reports (`.csv`).
- `output/cv_results/`: Stratified 5-iteration matrices logging average Recalls, Precisions, F1, and Accuracies conditionally (`.csv`).
- `output/shap_plots/`: Exported `.png` artifacts visualizing explicit TreeExplainer targets.

## Quick Start Commands

To train standard architectural constraints naturally:
```bash
python3 LightGBM_LeakProof.py "[DRUG_NAME]"
```

To leverage explicit SVD architectural parameters structurally:
```bash
python3 LightGBM_SVD.py "[DRUG_NAME]"
```

To review model operations mathematically via scaled plotting mapping:
```bash
python3 shap_svd_mapper.py "[DRUG_NAME]"
```

To run the visualization dashboard securely:
```bash
streamlit run dashboard.py
```
