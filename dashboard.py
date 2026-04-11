import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_selection import VarianceThreshold
import os
import glob

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drug Prediction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0a0a0f;
    border-right: 1px solid #1e1e2e;
}
[data-testid="stSidebar"] * { color: #8b8b9a !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.82rem; font-weight: 500; }

/* Main */
.main .block-container { padding-top: 2rem; padding-bottom: 3rem; background: #f9f9fb; }

/* Page title */
.page-title { font-size: 1.5rem; font-weight: 600; color: #111; margin-bottom: 4px; }
.page-sub   { font-size: 0.88rem; color: #666; margin-bottom: 24px; }

/* Stat cards */
.stat-row { display: flex; gap: 16px; margin-bottom: 24px; }
.stat-card {
    flex: 1;
    background: #fff;
    border: 1px solid #e8e8ee;
    border-radius: 10px;
    padding: 18px 20px;
}
.stat-card .slabel { font-size: 0.72rem; font-weight: 600; color: #888; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; }
.stat-card .sval   { font-size: 1.6rem; font-weight: 600; color: #111; letter-spacing: -0.5px; }
.stat-card .ssub   { font-size: 0.78rem; color: #999; margin-top: 3px; }

/* Divider */
.divider { border: none; border-top: 1px solid #ebebf0; margin: 24px 0; }

/* Section label */
.section-label { font-size: 0.8rem; font-weight: 600; color: #444; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 12px; }

/* Sidebar label */
.sb-label { font-size: 0.68rem; font-weight: 600; color: #444 !important; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 2px; }
.sb-val   { font-size: 1.5rem; font-weight: 600; color: #fff !important; }
.sb-sub   { font-size: 0.7rem; color: #555 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Data Helpers ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data_and_features():
    with open("features.txt", "r") as f:
        features = [line.strip() for line in f.readlines()]
    return features

@st.cache_data(show_spinner=False)
def load_metadata():
    try:
        df20 = pd.read_csv("input/prism-repurposing-20q2-secondary-screen-replicate-treatment-info.csv")
        df20['core_id'] = df20['broad_id'].astype(str).apply(
            lambda x: "-".join(x.split("-")[:2]) if pd.notnull(x) and "-" in x else str(x)
        )
        df20 = df20.drop_duplicates(subset=['core_id']).set_index('core_id')
        
        df24 = pd.read_csv("input/Repurposing_Public_24Q2_Extended_Primary_Compound_List.csv")
        df24['core_id'] = df24['IDs'].astype(str).apply(
            lambda x: "-".join(x.replace("BRD:", "").split("-")[:2]) if pd.notnull(x) and "-" in x else str(x)
        )
        df24 = df24.drop_duplicates(subset=['core_id']).set_index('core_id')
        
        df24_m = pd.read_csv("input/Repurposing_Public_24Q2_Treatment_Meta_Data.csv")
        df24_m['core_id'] = df24_m['broad_id'].astype(str).apply(
            lambda x: "-".join(x.split("-")[:2]) if pd.notnull(x) and "-" in x else str(x)
        )
        df24_m = df24_m.drop_duplicates(subset=['core_id']).set_index('core_id')
        
        meta_dict = {}
        all_ids = set(df20.index.tolist() + df24.index.tolist() + df24_m.index.tolist())
        
        for c in all_ids:
            if pd.isna(c) or c == 'nan': continue
            
            in_20 = c in df20.index
            in_24 = (c in df24.index) or (c in df24_m.index)
            
            tag = "Valid 2020 & 2024" if (in_24 and in_20) else "Valid 2024" if in_24 else "Valid 2020" if in_20 else ""
                
            name = np.nan
            moa = np.nan
            phase = np.nan
            ind = np.nan
            
            if in_24:
                if c in df24.index:
                    name = df24.loc[c, 'Drug.Name']
                    moa = df24.loc[c, 'MOA']
                if pd.isna(name) and c in df24_m.index:
                    name = df24_m.loc[c, 'name']
                    
            if pd.isna(name) and in_20: name = df20.loc[c, 'name']
            if pd.isna(moa) and in_20: moa = df20.loc[c, 'moa']
            if pd.isna(phase) and in_20: phase = df20.loc[c, 'phase']
            if pd.isna(ind) and in_20: ind = df20.loc[c, 'disease.area']
                
            meta_dict[c] = {
                'name': name if pd.notnull(name) else "Unknown",
                'moa': moa if pd.notnull(moa) else "Unknown MOA",
                'phase': phase if pd.notnull(phase) else "Unknown Phase",
                'disease.area': ind if pd.notnull(ind) else "Unknown Indication",
                'tag': tag
            }
            
        return pd.DataFrame.from_dict(meta_dict, orient='index')
    except Exception as e:
        return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def load_model(drug_name, struct="Original"):
    if struct == "Original":
        return joblib.load(f"output/models/LightGBM_{drug_name}_leakproof.joblib")
    else:
        return joblib.load(f"output/models/LightGBM_SVD_{drug_name}_leakproof.joblib")

def get_available_drugs(struct="Original"):
    if struct == "Original":
        files = glob.glob("output/models/LightGBM_*_leakproof.joblib")
        drugs = [os.path.basename(f).replace("LightGBM_", "").replace("_leakproof.joblib", "") for f in files if "SVD" not in os.path.basename(f)]
    else:
        files = glob.glob("output/models/LightGBM_SVD_*_leakproof.joblib")
        drugs = [os.path.basename(f).replace("LightGBM_SVD_", "").replace("_leakproof.joblib", "") for f in files]
    return sorted(drugs)

def load_cv_results(struct="Original"):
    results = []
    if struct == "Original":
        files = glob.glob("output/cv_results/cv_results_*_LGBM.csv")
        files = [f for f in files if "SVD" not in os.path.basename(f)]
        for f in files:
            drug = os.path.basename(f).replace("cv_results_", "").replace("_LGBM.csv", "")
            df = pd.read_csv(f)
            if 'test_accuracy' in df.columns:
                results.append({"Drug": drug, "Accuracy": round(df['test_accuracy'].mean(), 4), "Precision": round(df['test_precision'].mean(), 4), "Recall": round(df['test_recall'].mean(), 4), "F1": round(df['test_f1'].mean(), 4)})
    else:
        files = glob.glob("output/cv_results/cv_results_SVD_*_LGBM.csv")
        for f in files:
            drug = os.path.basename(f).replace("cv_results_SVD_", "").replace("_LGBM.csv", "")
            df = pd.read_csv(f)
            if 'test_accuracy' in df.columns:
                results.append({"Drug": drug, "Accuracy": round(df['test_accuracy'].mean(), 4), "Precision": round(df['test_precision'].mean(), 4), "Recall": round(df['test_recall'].mean(), 4), "F1": round(df['test_f1'].mean(), 4)})
    return results

metadata_df = load_metadata()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("###  Drug Prediction")
    st.markdown("---")
    str_choice = st.radio("Model Architecture", ["Original (Raw Genes)", "SVD Extrapolated"])
    ms_key = "Original" if "Original" in str_choice else "SVD"
    st.markdown("---")
    page = st.radio("Analysis", ["Drug Response Drivers", "Cross-Validation Analysis"], label_visibility="collapsed")
    st.markdown("---")
    
    available_drugs = get_available_drugs(ms_key)
    
    if not available_drugs:
        st.error(f"No trained models found for {ms_key} Architecture.")
        st.stop()
        
    st.markdown(f'<div class="sb-label">Models trained</div><div class="sb-val">{len(available_drugs)}</div><div class="sb-sub">{ms_key} LightGBM</div>', unsafe_allow_html=True)


# ─── Page 1: Drug Response Drivers ───────────────────────────────────────────
if page == "Drug Response Drivers":
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown('<div class="page-title">Drug Response Drivers</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-sub">Gene feature importances from trained LightGBM models</div>', unsafe_allow_html=True)
    with col_r:
        def format_drug_label(d_str):
            core_id = d_str.split("-")[0] + "-" + d_str.split("-")[1] if "-" in d_str else d_str
            if not metadata_df.empty and core_id in metadata_df.index:
                info = metadata_df.loc[core_id]
                if isinstance(info, pd.DataFrame):
                    info = info.iloc[0]
                n = info['name']
                if pd.notnull(n):
                    return f"{str(n).capitalize()} ({d_str})"
            return d_str
            
        selected_drug = st.selectbox("Drug", available_drugs, format_func=format_drug_label, label_visibility="collapsed")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    core_id = selected_drug.split("-")[0] + "-" + selected_drug.split("-")[1] if "-" in selected_drug else selected_drug
    if not metadata_df.empty and core_id in metadata_df.index:
        info = metadata_df.loc[core_id]
        if isinstance(info, pd.DataFrame):
            info = info.iloc[0]
        meta_name = info['name'] if pd.notnull(info['name']) else "Unknown"
        meta_moa = info['moa'] if pd.notnull(info['moa']) else "Unknown MOA"
        meta_phase = info['phase'] if pd.notnull(info['phase']) else "Unknown Phase"
        meta_ind = info['disease.area'] if pd.notnull(info['disease.area']) else "Unknown Indication"
        meta_tag = info['tag'] if 'tag' in info and pd.notnull(info['tag']) else ""
        st.markdown(f"""
        <div style="background:#fff; border:1px solid #e8e8ee; border-radius:10px; padding:16px 20px; margin-bottom:24px;">
            <div style="color:#111; font-weight:600; font-size:1.1rem; margin-bottom:4px; text-transform:capitalize;">
                {meta_name} 
                <span style="font-size:0.8rem; color:#888; font-weight:500; text-transform:none;">({meta_phase})</span>
                <span style="font-size:0.65rem; color:#fff; background:#3730a3; padding:3px 6px; border-radius:4px; margin-left:8px; vertical-align:middle; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; display:{'inline-block' if meta_tag else 'none'};">{meta_tag}</span>
            </div>
            <div style="font-size:0.85rem; color:#555;"><b>MOA:</b> {meta_moa} &nbsp;|&nbsp; <b>Disease Area:</b> {meta_ind}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No metadata record found in PRISM 20Q2 or 24Q2 Databases for this associated experimental drug.")

    with st.spinner("Loading features..."):
        surviving_features = load_data_and_features()
    with st.spinner("Loading model..."):
        model = load_model(selected_drug, ms_key)

    if ms_key == "Original":
        importances = model.feature_importances_
    else:
        svd_transformer = joblib.load(f"output/models/LightGBM_SVD_transformer_{selected_drug}.joblib")
        importances = np.dot(model.feature_importances_, np.abs(svd_transformer.components_))

    if len(importances) != len(surviving_features):
        st.error(f"Feature mismatch: map shape {len(importances)} vs VT {len(surviving_features)}.")
    else:
        df_imp = pd.DataFrame({"Gene": surviving_features, "Importance": importances})
        df_top20 = df_imp.sort_values("Importance", ascending=False).head(20)

        total_imp = df_imp["Importance"].sum()
        top5_share = df_top20.head(5)["Importance"].sum() / total_imp * 100 if total_imp > 0 else 0

        # Stat cards via HTML
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-card">
                <div class="slabel">Top Gene</div>
                <div class="sval" style="font-size:1.1rem;">{df_top20.iloc[0]['Gene']}</div>
                <div class="ssub">Highest importance score</div>
            </div>
            <div class="stat-card">
                <div class="slabel">Top Score</div>
                <div class="sval">{df_top20.iloc[0]['Importance']:,.0f}</div>
                <div class="ssub">Importance units</div>
            </div>
            <div class="stat-card">
                <div class="slabel">Features</div>
                <div class="sval">{len(surviving_features):,}</div>
                <div class="ssub">Post variance filter</div>
            </div>
            <div class="stat-card">
                <div class="slabel">Top-5 Share</div>
                <div class="sval">{top5_share:.1f}%</div>
                <div class="ssub">Of total importance</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">Top 20 Gene Importances</div>', unsafe_allow_html=True)

        df_chart = df_top20.sort_values("Importance", ascending=True)
        fig = go.Figure(go.Bar(
            x=df_chart["Importance"],
            y=df_chart["Gene"],
            orientation="h",
            marker=dict(
                color=df_chart["Importance"],
                colorscale=[[0, "#dde3ff"], [1, "#3730a3"]],
                line=dict(width=0)
            ),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:,.0f}<extra></extra>"
        ))
        fig.update_layout(
            plot_bgcolor="#12121f",
            paper_bgcolor="#12121f",
            height=480,
            margin=dict(l=10, r=20, t=10, b=30),
            xaxis=dict(showgrid=True, gridcolor="#2a2a3f", zeroline=False,
                       tickfont=dict(size=11, color="#aaa"), title_font=dict(color="#aaa"),
                       title="Importance Score"),
            yaxis=dict(tickfont=dict(size=12, color="#e0e0e0"), title=""),
            font=dict(family="Inter"),
            hoverlabel=dict(bgcolor="#3730a3", font_color="#fff", font_size=12)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Raw Scores — Top 20</div>', unsafe_allow_html=True)
        st.dataframe(
            df_top20.reset_index(drop=True),
            use_container_width=True,
            height=300
        )


# ─── Page 2: Cross-Validation Analysis ───────────────────────────────────────
elif page == "Cross-Validation Analysis":
    st.markdown('<div class="page-title">Cross-Validation Analysis</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">Mean metrics from 5-fold stratified CV across all {ms_key} models</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    results = load_cv_results(ms_key)

    if not results:
        st.warning("No CV results found. Run LightGBM_LeakProof.py on a drug first.")
    else:
        df = pd.DataFrame(results)

        def get_drug_name(drug_str):
            core_id = drug_str.split("-")[0] + "-" + drug_str.split("-")[1] if "-" in drug_str else drug_str
            if not metadata_df.empty and core_id in metadata_df.index:
                info = metadata_df.loc[core_id]
                if isinstance(info, pd.DataFrame):
                    info = info.iloc[0]
                n = info['name']
                return f"{str(n).capitalize()} ({drug_str})" if pd.notnull(n) else drug_str
            return drug_str

        df['Drug'] = df['Drug'].apply(get_drug_name)

        # Stat cards
        best = {m: df.loc[df[m].idxmax()] for m in ["Accuracy", "Precision", "Recall", "F1"]}
        labels = {"Accuracy": "Best Accuracy", "Precision": "Best Precision", "Recall": "Best Recall", "F1": "Best F1"}
        cards_html = '<div class="stat-row">'
        for m, lbl in labels.items():
            row = best[m]
            short = row["Drug"].split("::")[0]
            cards_html += f"""
            <div class="stat-card">
                <div class="slabel">{lbl}</div>
                <div class="sval">{row[m]:.1%}</div>
                <div class="ssub">{short}</div>
            </div>"""
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)

        # Metric bar chart
        col_l, col_r = st.columns([2, 1])
        with col_r:
            metric = st.selectbox("Metric", ["Accuracy", "Precision", "Recall", "F1"])
        with col_l:
            st.markdown(f'<div class="section-label">{metric} by Drug</div>', unsafe_allow_html=True)

        df_sorted = df.sort_values(metric, ascending=False)

        bar_colors = ["#3730a3" if i == 0 else "#c7d2fe" for i in range(len(df_sorted))]
        fig2 = go.Figure(go.Bar(
            x=df_sorted["Drug"],
            y=df_sorted[metric],
            marker=dict(color=bar_colors, line=dict(width=0)),
            hovertemplate="<b>%{x}</b><br>" + metric + ": %{y:.4f}<extra></extra>"
        ))
        fig2.update_layout(
            plot_bgcolor="#12121f",
            paper_bgcolor="#12121f",
            height=360,
            margin=dict(l=10, r=10, t=10, b=100),
            xaxis=dict(tickangle=-30, tickfont=dict(size=9, color="#aaa"), showgrid=False, title=""),
            yaxis=dict(range=[0, 1.05], showgrid=True, gridcolor="#2a2a3f", zeroline=False,
                       tickfont=dict(size=11, color="#aaa"), title=metric,
                       title_font=dict(color="#aaa")),
            font=dict(family="Inter"),
            hoverlabel=dict(bgcolor="#3730a3", font_color="#fff", font_size=12)
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # Radar chart
        st.markdown('<div class="section-label">All Metrics — Radar View</div>', unsafe_allow_html=True)
        metrics_cols = ["Accuracy", "Precision", "Recall", "F1"]
        palette = ["#3730a3", "#7c3aed", "#0891b2", "#059669", "#d97706", "#dc2626"]

        fig3 = go.Figure()
        for i, row in df.iterrows():
            fig3.add_trace(go.Scatterpolar(
                r=[row[m] for m in metrics_cols] + [row[metrics_cols[0]]],
                theta=metrics_cols + [metrics_cols[0]],
                fill="toself",
                fillcolor=palette[i % len(palette)],
                opacity=0.15,
                name=row["Drug"].split("::")[0],
                line=dict(color=palette[i % len(palette)], width=2),
                hovertemplate=row["Drug"] + "<br>%{theta}: %{r:.4f}<extra></extra>"
            ))
        fig3.update_layout(
            polar=dict(
                bgcolor="#12121f",
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="#2a2a3f",
                                tickfont=dict(size=9, color="#aaa"),
                                linecolor="#2a2a3f"),
                angularaxis=dict(tickfont=dict(size=12, color="#e0e0e0", family="Inter"),
                                 linecolor="#2a2a3f", gridcolor="#2a2a3f")
            ),
            paper_bgcolor="#12121f",
            legend=dict(font=dict(size=11, family="Inter", color="#e0e0e0"),
                        bgcolor="#1e1e30", bordercolor="#2a2a3f", borderwidth=1),
            height=420,
            margin=dict(l=40, r=160, t=20, b=20),
            font=dict(family="Inter"),
            hoverlabel=dict(bgcolor="#3730a3", font_color="#fff", font_size=12)
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Summary Table</div>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, height=250)
