#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os

def build_aligned_dataset():
    print("Loading baseline CRISPR/Expression dataset...")
    crispr_df = pd.read_csv("input/combined_DepMap_21Q3.csv")
    crispr_idx_col = crispr_df.columns[0]
    
    print("Loading new Damaging Mutations boolean dataset...")
    try:
        mut_df = pd.read_csv("input/CCLE_mutations_bool_damaging.csv")
        mut_idx_col = mut_df.columns[0]
    except FileNotFoundError:
        print("\nERROR: The file 'input/CCLE_mutations_bool_damaging.csv' is physically missing!")
        print("Please ensure it was placed in the input folder.")
        return

    print("Aligning row indices...")
    try:
        sample_info = pd.read_csv("input/sample_info.csv")
        mapper = dict(zip(sample_info['stripped_cell_line_name'], sample_info['DepMap_ID']))
        mapper.update(dict(zip(sample_info['cell_line_name'], sample_info['DepMap_ID'])))
        
        if not str(crispr_df[crispr_idx_col].iloc[0]).startswith('ACH-'):
            crispr_df[crispr_idx_col] = crispr_df[crispr_idx_col].map(mapper).fillna(crispr_df[crispr_idx_col])
        if not str(mut_df[mut_idx_col].iloc[0]).startswith('ACH-'):
            mut_df[mut_idx_col] = mut_df[mut_idx_col].map(mapper).fillna(mut_df[mut_idx_col])
    except FileNotFoundError:
        print("Warning: sample_info.csv not found, proceeding without ID mapping.")

    crispr_df.set_index(crispr_idx_col, inplace=True)
    mut_df.set_index(mut_idx_col, inplace=True)
    
    if mut_df.index.duplicated().any():
        print("Handling duplicate index entries in Mutation data by taking the max (logical OR)...")
        mut_df = mut_df.groupby(mut_df.index).max()

    shared_cell_lines = list(set(crispr_df.index) & set(mut_df.index))
    print(f"Successfully identified {len(shared_cell_lines)} overlapping cell line domains.")

    if len(shared_cell_lines) == 0:
        print("Error: No overlapping indices found. Cell line mapping logic needs manual intervention.")
        return

    # Retain all physical tech replicates cleanly aligned to corresponding mutated boolean maps natively
    crispr_aligned = crispr_df[crispr_df.index.isin(shared_cell_lines)]
    mut_aligned = mut_df.loc[crispr_aligned.index]

    os.makedirs("input/aligned", exist_ok=True)
    print("Exporting aligned CRISPR matrices...")
    crispr_aligned.to_csv("input/aligned/aligned_crispr.csv")
    
    print("Exporting tightly bounded Mutation matrices...")
    mut_aligned.to_csv("input/aligned/aligned_mutations.csv")
    
    print("Preprocessing successful -> Data cleanly localized into 'input/aligned/'.")

if __name__ == '__main__':
    build_aligned_dataset()
