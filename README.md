# UM IHC Drug Response Prediction Pipeline

This repository contains a machine learning pipeline designed to predict drug response based on transcriptomic and feature data. The models account for data leakage issues, compress high-dimensional feature spaces via SVD, and provide interpretability mapping back to the uncompressed feature space via SHAP.

## Project Structure

### Data & Compression
- **`svd_compressor.py`**: Handles dimensionality reduction (Singular Value Decomposition) to compress sparse, high-dimensional gene expression data into meaningful components. Outputs deterministic transformations that avoid test-set data leakage.
- **`input/`**: Directory containing target metadata, drug target sets, and compressed outputs. Note that large gene expression datasets (like DepMap) are placed here but excluded from version control.

### ML Models
- **`LightGBM_LeakProof.py` & `RandomForest_LeakProof.py`**: Original uncompressed models leveraging strictly separated train/test splits to calculate cross-validated metrics without target leakage.
- **`LightGBM_SVD.py`**: Trains LightGBM models over the SVD-compressed data space. Reintegrates the SVD transformer in the pipeline context.
- **`dashboard.py`**: Streamlit visualization dashboard providing an interface for selecting targets and reading model interpretations and performance metrics.

### Model Interpretability (SHAP)
- **`shap_analysis.py`**: Computes SHAP (SHapley Additive exPlanations) values for target models using `TreeExplainer`.
- **`shap_svd_mapper.py` & `svd_reverse_mapper.py`**: Takes the computed SHAP values on the compressed latent components and projects them back onto the original high-dimensional feature space. This is critical to linking model logic back to biological insight (e.g. mapping components back to specific genes). 

### Reporting Outputs
- **`output/models/`**: Joblib dumps for trained models, data scalers, and the SVD transformers trained across experimental splits. 
- **`output/cv_results/`** & **`output/reports/`**: Raw CSV artifacts mapping split-level model cross-validation performances and global classification reports, generated during the automated training loop.
- **`output/shap_plots/`**: Visualization artifacts (e.g. dot summaries) tracing SHAP predictions.

## Setup & Installation

1. Create a python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Provide the initial DepMap matrices in the `input/` directory:
   - `combined_DepMap_21Q3.csv`
   - `combined_DepMap_21Q3_CCLE_expression.csv`
   - `combined_DepMap_21Q3_druggable.csv`

## Usage

**Running dimensionality compression:**
```bash
python svd_compressor.py
```

**Training Models (SVD or LeakProof variants):**
```bash
python LightGBM_SVD.py
```
*(Produces serialized models and `classification_report.csv` in `output/`)*

**Generating Interpretability Matrices:**
```bash
python shap_analysis.py
```

**Running the Interactive Dashboard:**
```bash
streamlit run dashboard.py
```
