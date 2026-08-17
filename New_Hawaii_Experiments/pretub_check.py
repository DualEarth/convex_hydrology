import os
import re
from pathlib import Path

import yaml
import torch
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — loaded from config.yaml (section: pretub_check)
# ─────────────────────────────────────────────────────────────────────────────
_cfg = yaml.safe_load(open(Path(__file__).with_name("config.yaml")))["pretub_check"]

TEST_DIR      = _cfg["test_dir"]
REF_BASIN_ID  = _cfg["reference_basin_id"]   # The "Anchor" basin
PERT_BASIN_ID = _cfg["perturbed_basin_id"]   # The Perturbed version
OUTPUT_FILE   = _cfg["output_file"]

FILE_PATTERN  = r'c_n_reduced_epoch\d+_basin_([^\.]+)\.pt$'

def clean_id(s):
    s = str(s).strip()
    return re.sub(r'(\.0+|_0)$', '', s)

# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def load_basin_tensor(path):
    """Loads and flattens the 7D state tensor."""
    t = torch.load(path, map_location="cpu")
    # Handle both [Time, Dim] and [Batch, Time, Dim]
    arr = t.reshape(-1, t.shape[-1]).numpy()
    return arr[~np.isnan(arr).any(axis=1)]

def main():
    # 1. Identify files
    all_files = os.listdir(TEST_DIR)
    basin_map = {}
    for f in all_files:
        m = re.search(FILE_PATTERN, f)
        if m:
            bid = clean_id(m.group(1))
            basin_map[bid] = os.path.join(TEST_DIR, f)

    if REF_BASIN_ID not in basin_map:
        print(f"Error: Reference basin {REF_BASIN_ID} not found in {TEST_DIR}")
        return

    # 2. Load Reference Basin and fit a KDTree for speed
    print(f"Loading Reference Basin: {REF_BASIN_ID}...")
    ref_pts = load_basin_tensor(basin_map[REF_BASIN_ID])
    # n_neighbors=1 because we just want the closest point in the reference manifold
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(ref_pts)

    # 3. Iterate through all basins and compute distance to Reference
    results = []
    print("Computing distances to reference manifold...")
    
    for bid, path in tqdm(basin_map.items()):
        try:
            target_pts = load_basin_tensor(path)
            
            # For every point in target, find distance to closest point in reference
            distances, _ = nn.kneighbors(target_pts)
            
            results.append({
                "basin_id": bid,
                "is_perturbed": (bid == PERT_BASIN_ID),
                "mean_dist": np.mean(distances),
                "median_dist": np.median(distances),
                "min_dist": np.min(distances),
                "max_dist": np.max(distances),
                "sum_dist": np.sum(distances), # Total distance across all timesteps
                "n_timesteps": len(target_pts)
            })
        except Exception as e:
            print(f"Skipping {bid}: {e}")

    # 4. Process and Save
    df = pd.DataFrame(results).sort_values("mean_dist")
    
    # Calculate "Proximity Rank" (1 = closest to the reference)
    df['rank'] = df['mean_dist'].rank(method='min')
    
    print("\n--- Top 5 Closest Basins to Reference ---")
    print(df[['basin_id', 'mean_dist', 'rank', 'is_perturbed']].head(50).to_string(index=False))
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nFull results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
