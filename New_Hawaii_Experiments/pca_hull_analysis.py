"""
pca_hull_analysis.py  (Hawaii / parquet)
────────────────────────────────────────
1. Loads last-epoch non-reduced train + validation .pt files
2. Fits PCA to reduce to PCA_DIMS proxy space
3. Subsamples reference for tractable convex hull construction
4. Projects test basin states into same PCA space
5. Computes hull + Mahalanobis distances (GPU accelerated)
6. Bootstraps Spearman r vs NSE
7. Produces 4 scatter plots with bootstrapped LOESS band + binned trend

Run paths and thresholds come from config.yaml (section: pca_hull_analysis)
in this same folder — edit that file to point at a new training run,
nothing in this script needs to change.
"""

import os
import re
import time
import warnings
from pathlib import Path

import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from scipy.spatial import ConvexHull, KDTree, Delaunay

from utils import load_nse, bootstrap_spearman, bootstrap_loess, binned_trend


# CONFIG — loaded from config.yaml (section: pca_hull_analysis)

_cfg = yaml.safe_load(open(Path(__file__).with_name("config.yaml")))["pca_hull_analysis"]

TRAIN_STATE_DIR = _cfg["train_state_dir"]
VAL_STATE_DIR   = _cfg["val_state_dir"]
TEST_STATE_DIR  = _cfg["test_state_dir"]
NSE_FILE        = _cfg["nse_file"]
NSE_BASIN_COL   = _cfg["nse_basin_col"]
NSE_SCORE_COL   = _cfg["nse_score_col"]
OUTPUT_DIR      = _cfg["output_dir"]

TEST_PATTERN    = _cfg["test_pattern"]   # non-reduced per-basin state files

PCA_DIMS        = _cfg["pca_dims"]       # keep ≤ 6 for tractable hull
MAX_HULL_PTS    = _cfg["max_hull_pts"]   # subsample reference before hull construction
MAHAL_THRESHOLD = _cfg["mahal_threshold"]
NSE_MIN_FILTER  = _cfg["nse_min_filter"]
N_BOOTSTRAP     = _cfg["n_bootstrap"]
CI_LEVEL        = _cfg["ci_level"]
N_BINS          = _cfg["n_bins"]
RNG_SEED        = _cfg["rng_seed"]



# LOAD
def find_last_epoch(folder, pattern):
    matches = []
    for fname in os.listdir(folder):
        m = re.search(pattern, fname)
        if m:
            matches.append((int(m.group(1)), fname))
    return max(matches, key=lambda x: x[0])[1] if matches else None


def load_last_epoch(folder, pattern):
    fname = find_last_epoch(folder, pattern)
    if fname is None:
        print(f"  [WARN] No file matching '{pattern}' in {folder}")
        return None
    path = os.path.join(folder, fname)
    t    = torch.load(path, map_location="cpu")
    arr  = t.reshape(-1, t.shape[-1]).numpy()
    arr  = arr[~np.isnan(arr).any(axis=1)]
    print(f"  Loaded {fname}: {arr.shape}")
    return arr


def load_test_basins(folder, pattern):
    basin_files = {}
    for fname in os.listdir(folder):
        m = re.search(pattern, fname)
        if not m:
            continue
        basin_id = m.group(1)
        ep       = int(re.search(r'epoch(\d+)', fname).group(1))
        if basin_id not in basin_files or ep > basin_files[basin_id][0]:
            basin_files[basin_id] = (ep, fname)

    basin_data = {}
    for basin_id, (_, fname) in basin_files.items():
        try:
            t   = torch.load(os.path.join(folder, fname), map_location="cpu")
            arr = t.reshape(-1, t.shape[-1]).numpy()
            arr = arr[~np.isnan(arr).any(axis=1)]
            basin_data[basin_id] = arr
        except Exception as e:
            print(f"  [WARN] Skipping {fname}: {e}")

    print(f"  Loaded {len(basin_data)} test basins.")
    return basin_data

# PCA
def fit_pca_and_plot(ref_data, n_components, output_dir):
    n        = min(256, ref_data.shape[1])
    pca_full = PCA(n_components=n).fit(ref_data)
    cumvar   = np.cumsum(pca_full.explained_variance_ratio_)
    k95      = np.argmax(cumvar >= 0.95) + 1
    print(f"  Components for 95% variance: {k95}")

    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, n + 1), 1 - cumvar)
    plt.axvline(n_components, color="red",   linestyle="--", label=f"{n_components}D (used)")
    plt.axvline(k95,          color="green", linestyle="--", label=f"{k95}D (95% var)")
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Information Loss (1 - cumulative variance)")
    plt.title("PCA Information Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "pca_information_loss.png"), dpi=300)
    plt.close()
    print("  Saved: pca_information_loss.png")

    pca = PCA(n_components=n_components).fit(ref_data)
    print(f"  PCA {n_components}D explains "
          f"{np.sum(pca.explained_variance_ratio_)*100:.1f}% of variance.")
    return pca



# HULL + DISTANCES

def subsample(data, max_pts, seed=42):
    if len(data) <= max_pts:
        return data
    idx = np.random.default_rng(seed).choice(len(data), size=max_pts, replace=False)
    print(f"  Subsampled {len(data):,} → {max_pts:,} points for hull construction.")
    return data[idx]


def build_hull(data):
    try:
        hull     = ConvexHull(data)
        vertices = data[hull.vertices]
        delaunay = Delaunay(vertices)
        tree     = KDTree(vertices)
        print(f"  Hull built: {len(hull.vertices)} vertices in {data.shape[1]}D.")
        return hull, delaunay, tree, vertices
    except Exception as e:
        print(f"  [WARN] Hull failed: {e}")
        return None, None, None, None


def compute_distances(test_points, ref_data, delaunay, tree):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_t = torch.tensor(test_points, dtype=torch.float32, device=device)
    ref_t  = torch.tensor(ref_data,    dtype=torch.float32, device=device)

    # Hull distance (GPU nearest vertex, CPU inside check)
    if tree is not None:
        verts_t   = torch.tensor(tree.data, dtype=torch.float32, device=device)
        batch     = 10000
        hull_dist = []
        for i in range(0, len(test_t), batch):
            b    = test_t[i:i+batch]
            diff = b.unsqueeze(1) - verts_t.unsqueeze(0)
            d    = torch.sqrt((diff**2).sum(2)).min(1).values
            hull_dist.append(d.cpu().numpy())
        hull_dist = np.concatenate(hull_dist)
        if delaunay is not None:
            hull_dist[delaunay.find_simplex(test_points) >= 0] = 0.0
    else:
        hull_dist = np.zeros(len(test_points))

    # Centroid distance (GPU) 
    centroid  = ref_t.mean(0)
    cent_dist = []
    for i in range(0, len(test_t), 10000):
        cent_dist.append(
            torch.norm(test_t[i:i+10000] - centroid, dim=1).cpu().numpy()
        )
    cent_dist = np.concatenate(cent_dist)

    # Mahalanobis (GPU) 
    cov  = LedoitWolf().fit(ref_data)
    VI   = torch.tensor(cov.get_precision(), dtype=torch.float32, device=device)
    loc  = torch.tensor(cov.location_,       dtype=torch.float32, device=device)
    mah  = []
    for i in range(0, len(test_t), 10000):
        d = test_t[i:i+10000] - loc
        mah.append(torch.sqrt((d @ VI * d).sum(1).clamp(0)).cpu().numpy())
    mahal_dist = np.concatenate(mah)

    return hull_dist, cent_dist, mahal_dist


def compute_basin_metrics(basin_data_pca, ref_pca, delaunay, tree, mahal_threshold):
    all_points = np.vstack(list(basin_data_pca.values()))
    hull_dist, cent_dist, mahal_dist = compute_distances(
        all_points, ref_pca, delaunay, tree
    )

    rows = []
    idx  = 0
    for basin_id, pts in basin_data_pca.items():
        n      = len(pts)
        hd     = hull_dist[idx:idx+n]
        cd     = cent_dist[idx:idx+n]
        md     = mahal_dist[idx:idx+n]
        extrap = md > mahal_threshold
        rows.append({
            "basin_id":                              basin_id,
            "n_points":                              n,
            "sum_distance_convex_hull":              float(hd.sum()),
            "sum_distance_hull_center":              float(cd.sum()),
            "sum_mahalanobis_distance":              float(md.sum()),
            "sum_mahalanobis_distance_extrapolated": float(md[extrap].sum()),
        })
        idx += n

    return pd.DataFrame(rows)



# PLOT

def plot_xy(x, y, output_dir, x_col, xlabel, boot, loess_result, bins,
            ylabel="NSE"):
    x_grid, obs_loess, lo, hi = loess_result

    plt.rcParams.update({"font.size": 12})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                             gridspec_kw={"width_ratios": [2, 1]})

    # Left — scatter + LOESS band
    ax = axes[0]
    sc = ax.scatter(x, y, c=y, cmap="viridis", s=40, alpha=0.6, zorder=3)
    ax.fill_between(x_grid, lo, hi, alpha=0.25, color="steelblue",
                    label=f"LOESS {boot['ci_level']}% CI")
    ax.plot(x_grid, obs_loess, color="steelblue", lw=2.5, label="LOESS")
    ax.set_xscale("log")
    plt.colorbar(sc, ax=ax).set_label(ylabel, fontsize=12)
    ax.text(0.97, 0.97,
            f"Spearman r = {boot['spearman_r']:.2f}\n"
            f"{boot['ci_level']}% CI  [{boot['ci_lo']:.2f}, {boot['ci_hi']:.2f}]\n"
            f"P(r < 0) = {boot['p_negative']:.2f}\n"
            f"n = {boot['n']}  |  {boot['n_boot']} bootstrap",
            transform=ax.transAxes, fontsize=10, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"PCA {PCA_DIMS}D  |  NSE > {NSE_MIN_FILTER}  |  n={boot['n']}",
                 fontsize=10, color="gray")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="lower left")

    # Right — binned trend
    ax2 = axes[1]
    ax2.errorbar(bins["x_centre"], bins["mean"], yerr=bins["std"],
                 fmt="o-", color="steelblue", capsize=5,
                 elinewidth=1.5, lw=2, markersize=7)
    for _, row in bins.iterrows():
        ax2.annotate(f"n={int(row['count'])}",
                     (row["x_centre"], row["mean"] + (row["std"] or 0) + 0.01),
                     fontsize=9, ha="center", color="gray")
    ax2.set_xscale("log")
    ax2.set_xlabel(xlabel, fontsize=12)
    ax2.set_ylabel(ylabel, fontsize=12)
    ax2.set_title(f"Binned trend ({N_BINS} quantile bins)", fontsize=10, color="gray")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fname = f"NSE_vs_{x_col}_pca{PCA_DIMS}d.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close()
    print(f"  Saved: {fname}")


# MAIN

def main():
    t0 = time.perf_counter()

    # Load last epoch train + validation
    print("\nLoading train states (last epoch)...")
    train_raw = load_last_epoch(TRAIN_STATE_DIR, r'c_n_epoch(\d+)\.pt$')

    print("\nLoading validation states (last epoch)...")
    val_raw = load_last_epoch(VAL_STATE_DIR, r'c_n_epoch(\d+)_validation\.pt$')

    ref_raw = np.vstack([train_raw, val_raw]) if val_raw is not None else train_raw
    print(f"\nReference (train+val): {ref_raw.shape}")

    # Fit PCA 
    print(f"\nFitting PCA ({PCA_DIMS}D)...")
    pca     = fit_pca_and_plot(ref_raw, PCA_DIMS, OUTPUT_DIR)
    ref_pca = pca.transform(ref_raw)

    # Build hull on subsampled reference 
    print("\nBuilding convex hull...")
    hull_data            = subsample(ref_pca, MAX_HULL_PTS, RNG_SEED)
    hull, delaunay, tree, _ = build_hull(hull_data)

    # Load + project test basins
    print("\nLoading test basin states...")
    basin_data_raw = load_test_basins(TEST_STATE_DIR, TEST_PATTERN)
    if not basin_data_raw:
        print("[ERROR] No test basins loaded.")
        return

    basin_data_pca = {bid: pca.transform(pts)
                      for bid, pts in basin_data_raw.items()}

    # Compute metrics
    print("\nComputing per-basin metrics...")
    df = compute_basin_metrics(
        basin_data_pca, ref_pca, delaunay, tree, MAHAL_THRESHOLD
    )

    # Attach NSE + filter
    nse_map  = load_nse(NSE_FILE, NSE_BASIN_COL, NSE_SCORE_COL)
    df["NSE"] = df["basin_id"].map(lambda b: float(nse_map.get(str(b), np.nan)))
    df        = df[df["NSE"] > NSE_MIN_FILTER].copy()
    n_extrap  = (df["sum_distance_convex_hull"] > 0).sum()
    print(f"Basins: {len(df)} total  |  {n_extrap} outside hull")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_DIR, "basin_metrics_pca.csv"), index=False)

    # 4 plots
    plot_specs = [
        ("sum_distance_convex_hull",
         "Sum of Extrapolated Dist. From Convex Hull (log)"),
        ("sum_distance_hull_center",
         "Sum Distance from Geometric Hull Center (log)"),
        ("sum_mahalanobis_distance_extrapolated",
         "Sum of Mahalanobis Dist. for Extrapolated States (log)"),
        ("sum_mahalanobis_distance",
         "Sum of Mahalanobis Dist. (log)"),
    ]

    sns.set_theme(style="whitegrid")
    boot_rows = []

    for x_col, xlabel in plot_specs:
        print(f"\n── {x_col} ──")
        sub = df[["NSE", x_col]].dropna()
        sub = sub[sub[x_col] > 0]
        if len(sub) < 10:
            print(f"  [WARN] Only {len(sub)} non-zero points, skipping.")
            continue

        x, y = sub[x_col].values, sub["NSE"].values
        print(f"  Bootstrapping ({len(x)} basins)...")
        boot         = bootstrap_spearman(x, y, N_BOOTSTRAP, CI_LEVEL, RNG_SEED)
        loess_result = bootstrap_loess(x, y, 500, CI_LEVEL, RNG_SEED)
        bins         = binned_trend(x, y, N_BINS)

        plot_xy(x, y, OUTPUT_DIR, x_col, xlabel, boot, loess_result, bins)

        print(f"  r={boot['spearman_r']:.3f} "
              f"[{boot['ci_lo']:.3f}, {boot['ci_hi']:.3f}]  "
              f"P(r<0)={boot['p_negative']:.3f}  n={len(x)}")

        boot_rows.append({
            "distance_metric": x_col,
            "pca_dims":        PCA_DIMS,
            "hull_subsample":  MAX_HULL_PTS,
            "n_basins":        len(x),
            "spearman_r":      round(boot["spearman_r"], 3),
            "spearman_ci_lo":  round(boot["ci_lo"],       3),
            "spearman_ci_hi":  round(boot["ci_hi"],       3),
            "p_negative":      round(boot["p_negative"],  3),
        })

    if boot_rows:
        tbl = pd.DataFrame(boot_rows)
        out = os.path.join(OUTPUT_DIR, "bootstrap_summary_pca.csv")
        tbl.to_csv(out, index=False)
        print(f"\nBootstrap summary:\n{tbl.to_string(index=False)}")

    print(f"\nTotal runtime: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
