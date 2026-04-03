import pandas as pd

def find_common_drugs():
    print("Loading datasets...")
    meta = pd.read_csv("input/prism-repurposing-20q2-secondary-screen-replicate-treatment-info.csv")
    
    # Read only the headers of the massive dataset to get columns
    dataset = pd.read_csv("input/combined_DepMap_21Q3.csv", nrows=0)
    drug_cols = dataset.columns[-4686:]
    
    print(f"Total dataset drug columns: {len(drug_cols)}")
    
    meta['core_id'] = meta['broad_id'].astype(str).apply(
        lambda x: "-".join(x.split("-")[:2]) if pd.notnull(x) and "-" in x else str(x)
    )
    meta_core_ids = set(meta['core_id'].dropna())
    
    found = []
    
    for col in drug_cols:
        core_id = col.split('-')[0] + '-' + col.split('-')[1] if '-' in col else col
        if core_id in meta_core_ids:
            found.append(col)
            if len(found) >= 2: # Stop early since we only need 2
                break
                
    for f in found:
        print(f"MATCH: {f}")

if __name__ == '__main__':
    find_common_drugs()
