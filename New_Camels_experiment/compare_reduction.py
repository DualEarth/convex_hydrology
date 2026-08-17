"""
compare_reduction.py
────────────────────
Compares two dimensionality reduction approaches on the same 64D cell states:
  A) Neural Reducer 10D  (learned autoencoder)
  B) PCA 10D             (linear projection fitted here)

For each basin computes:
  - Trustworthiness & Continuity  (neighbourhood preservation, k=10)
  - Kruskal Stress                (distance distortion)
  - Pairwise distance Spearman r  (rank preservation)
  - Procrustes disparity          (shape preservation)
  - Variance explained            (PCA only)
  - Reconstruction MSE            (neural reducer only)

Outputs:
  - per_basin_comparison.csv
  - summary_comparison.csv
  - comparison_boxplots.png
  - comparison_vs_nse.png
"""

import os
import re
import warnings
from pathlib import Path

import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from scipy.spatial.distance import pdist
from scipy.spatial import procrustes
from scipy.stats import spearmanr


# CONFIG — loaded from config.yaml (section: compare_reduction)
_cfg = yaml.safe_load(open(Path(__file__).with_name("config.yaml")))["compare_reduction"]

TEST_DIR        = _cfg["test_dir"]
NSE_FILE        = _cfg["nse_file"]
NSE_BASIN_COL   = _cfg["nse_basin_col"]
NSE_SCORE_COL   = _cfg["nse_score_col"]
OUTPUT_DIR      = _cfg["output_dir"]

# File patterns
ORIG_PATTERN    = _cfg["orig_pattern"]      # non-reduced original state dim
REDUCED_PATTERN = _cfg["reduced_pattern"]   # neural-reducer output dim

K_NEIGHBOURS    = _cfg["k_neighbours"]  # for trustworthiness / continuity
MAX_POINTS      = _cfg["max_points"]    # max points per basin (subsampled for speed)
REDUCED_DIMS    = _cfg["reduced_dims"]  # must match both neural reducer and PCA output


# LOAD
def load_basin_files(folder, pattern):
    """Load last epoch per basin. Returns {basin_id: array}."""
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

    print(f"  Loaded {len(basin_data)} basins  (pattern: {pattern})")
    return basin_data


def load_nse(path, basin_col, score_col):
    df = pd.read_csv(path)
    df[basin_col] = df[basin_col].astype(str)
    return df.set_index(basin_col)[score_col].to_dict()


# PCA — fit on ALL original basins, transform per basin
def fit_pca_global(orig_data, n_components):
    """
    Fit PCA on all original states concatenated, then return
    per-basin projections and variance explained.
    """
    all_pts  = np.vstack(list(orig_data.values()))
    pca      = PCA(n_components=n_components).fit(all_pts)
    var_expl = float(np.sum(pca.explained_variance_ratio_))

    # Full variance curve
    n_full   = min(all_pts.shape[1], all_pts.shape[0])
    pca_full = PCA(n_components=n_full).fit(all_pts)
    cumvar   = np.cumsum(pca_full.explained_variance_ratio_)
    k95      = int(np.argmax(cumvar >= 0.95)) + 1

    print(f"  PCA {n_components}D explains {var_expl*100:.1f}% of variance  "
          f"(95% needs {k95} components)")

    pca_data = {bid: pca.transform(pts) for bid, pts in orig_data.items()}
    return pca, pca_data, var_expl, cumvar


def plot_pca_variance(cumvar, n_used, output_dir):
    n = len(cumvar)
    k95 = int(np.argmax(cumvar >= 0.95)) + 1
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, n + 1), 1 - cumvar, color="steelblue")
    plt.axvline(n_used, color="red",   linestyle="--", label=f"{n_used}D (used)")
    plt.axvline(k95,    color="green", linestyle="--", label=f"{k95}D (95% var)")
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Information Loss (1 - cumulative variance)")
    plt.title("PCA Information Loss — 64D Cell States")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "pca_information_loss.png"), dpi=300)
    plt.close()
    print("  Saved: pca_information_loss.png")


# PRESERVATION METRICS
def subsample_aligned(orig, reduced, max_pts, seed=42):
    n = min(len(orig), len(reduced), max_pts)
    if n < min(len(orig), len(reduced)):
        idx = np.random.default_rng(seed).choice(
            min(len(orig), len(reduced)), size=n, replace=False
        )
        return orig[idx], reduced[idx]
    n_min = min(len(orig), len(reduced))
    return orig[:n_min], reduced[:n_min]


def kruskal_stress(orig, reduced):
    d_orig    = pdist(orig,    metric="euclidean")
    d_reduced = pdist(reduced, metric="euclidean")
    scale     = d_orig.std() / (d_reduced.std() + 1e-12)
    d_reduced = d_reduced * scale
    return float(np.sqrt(
        np.sum((d_orig - d_reduced)**2) / (np.sum(d_orig**2) + 1e-12)
    ))


def pairwise_spearman(orig, reduced):
    d_orig    = pdist(orig,    metric="euclidean")
    d_reduced = pdist(reduced, metric="euclidean")
    r, _      = spearmanr(d_orig, d_reduced)
    return float(r)


def procrustes_distance(orig, reduced):
    orig_proj = PCA(n_components=reduced.shape[1]).fit_transform(orig) \
                if orig.shape[1] > reduced.shape[1] else orig
    try:
        _, _, disparity = procrustes(orig_proj, reduced)
        return float(disparity)
    except Exception:
        return np.nan


def get_metrics(orig, reduced, k):
    if len(orig) < k + 2:
        return None
    t    = float(trustworthiness(orig, reduced, n_neighbors=k, metric="euclidean"))
    c    = float(trustworthiness(reduced, orig, n_neighbors=k, metric="euclidean"))
    st   = kruskal_stress(orig, reduced)
    pw   = pairwise_spearman(orig, reduced)
    proc = procrustes_distance(orig, reduced)
    return {
        "trustworthiness":   t,
        "continuity":        c,
        "stress":            st,
        "pairwise_spearman": pw,
        "procrustes":        proc,
        "n_points":          len(orig),
    }


def reconstruction_mse(orig, reduced):
    """
    MSE of reconstructing orig from reduced using a simple linear decoder
    (least squares). This gives a fair lower-bound estimate of information loss.
    """
    # Fit linear decoder: reduced @ W ≈ orig
    W, _, _, _ = np.linalg.lstsq(reduced, orig, rcond=None)
    recon      = reduced @ W
    return float(np.mean((orig - recon)**2))


# PLOTS
METRIC_LABELS = {
    "trustworthiness":   "Trustworthiness",
    "continuity":        "Continuity",
    "stress":            "Kruskal Stress",
    "pairwise_spearman": "Pairwise Spearman r",
    "procrustes":        "Procrustes Disparity",
    "recon_mse":         "Reconstruction MSE",
}


def plot_boxplots(df, output_dir):
    metrics = ["trustworthiness", "continuity", "stress",
               "pairwise_spearman", "procrustes", "recon_mse"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        col_nr  = f"{metric}_neural"
        col_pca = f"{metric}_pca"

        plot_data = []
        if col_nr in df.columns:
            vals = df[col_nr].dropna()
            plot_data.extend([{"Method": "Neural 7D", "value": v} for v in vals])
        if col_pca in df.columns:
            vals = df[col_pca].dropna()
            plot_data.extend([{"Method": "PCA 7D", "value": v} for v in vals])

        if not plot_data:
            ax.set_visible(False)
            continue

        pdf = pd.DataFrame(plot_data)
        sns.boxplot(data=pdf, x="Method", y="value", ax=ax,
                    palette={"Neural 7D": "steelblue", "PCA 7D": "tomato"},
                    width=0.5)

        # Annotate median
        for i, method in enumerate(["Neural 7D", "PCA 7D"]):
            med = pdf[pdf["Method"] == method]["value"].median()
            ax.text(i, med, f" {med:.3f}", va="center", fontsize=9, color="black")

        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle(
        f"Neural Reducer 7D vs PCA 7D — Preservation Metrics\n"
        f"(from 64D cell states, k={K_NEIGHBOURS}, n≤{MAX_POINTS} pts/basin)",
        fontsize=13
    )
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "comparison_boxplots.png"), dpi=300)
    plt.close()
    print("  Saved: comparison_boxplots.png")


def plot_vs_nse(df, output_dir):
    metrics = ["trustworthiness", "continuity", "stress",
               "pairwise_spearman", "procrustes"]

    fig, axes = plt.subplots(2, len(metrics), figsize=(20, 8),
                             sharey="row")

    for col_idx, metric in enumerate(metrics):
        for row_idx, (method, color) in enumerate(
            [("neural", "steelblue"), ("pca", "tomato")]
        ):
            ax  = axes[row_idx][col_idx]
            col = f"{metric}_{method}"

            if col not in df.columns or "NSE" not in df.columns:
                ax.set_visible(False)
                continue

            sub = df[["NSE", col]].dropna()
            if sub.empty:
                continue

            ax.scatter(sub[col], sub["NSE"], c=sub["NSE"],
                       cmap="viridis", s=25, alpha=0.6)

            try:
                sns.regplot(data=sub, x=col, y="NSE", lowess=True,
                            scatter=False, ax=ax,
                            line_kws={"color": color, "lw": 2}, ci=None)
            except Exception:
                pass

            r, _ = spearmanr(sub[col], sub["NSE"])
            ax.set_title(
                f"{'Neural' if method == 'neural' else 'PCA'} {REDUCED_DIMS}D\n"
                f"{METRIC_LABELS.get(metric, metric)}\nr={r:.2f}",
                fontsize=9
            )
            ax.set_xlabel("")
            ax.set_ylabel("NSE" if col_idx == 0 else "", fontsize=10)
            ax.grid(True, alpha=0.3)

    plt.suptitle("Preservation Metrics vs NSE — Neural 7D (top) vs PCA 7D (bottom)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_vs_nse.png"), dpi=300)
    plt.close()
    print("  Saved: comparison_vs_nse.png")


# MAIN
def main():
    print("\nLoading original (64D) states...")
    orig_data = load_basin_files(TEST_DIR, ORIG_PATTERN)

    print("\nLoading neural reducer (7D) states...")
    neural_data = load_basin_files(TEST_DIR, REDUCED_PATTERN)

    # Common basins across all three
    common = sorted(set(orig_data.keys()) & set(neural_data.keys()))
    print(f"\nCommon basins: {len(common)}")

    # Fit PCA globally on all original states
    print("\nFitting PCA 7D on all original states...")
    pca, pca_data, var_expl, cumvar = fit_pca_global(
        {b: orig_data[b] for b in common}, REDUCED_DIMS
    )
    plot_pca_variance(cumvar, REDUCED_DIMS, OUTPUT_DIR)

    # Load NSE
    nse_map = load_nse(NSE_FILE, NSE_BASIN_COL, NSE_SCORE_COL)

    # Per-basin metrics
    rows = []
    for i, basin_id in enumerate(common):
        orig    = orig_data[basin_id]
        neural  = neural_data[basin_id]
        pca_pts = pca_data[basin_id]

        # Align lengths
        n       = min(len(orig), len(neural), len(pca_pts))
        orig    = orig[:n]
        neural  = neural[:n]
        pca_pts = pca_pts[:n]

        # Subsample for speed
        orig_s,   neural_s  = subsample_aligned(orig, neural,  MAX_POINTS)
        orig_s2,  pca_s     = subsample_aligned(orig, pca_pts, MAX_POINTS)

        m_neural = get_metrics(orig_s,  neural_s, K_NEIGHBOURS)
        m_pca    = get_metrics(orig_s2, pca_s,    K_NEIGHBOURS)

        if m_neural is None or m_pca is None:
            continue

        # Reconstruction MSE
        mse_neural = reconstruction_mse(orig_s,  neural_s)
        mse_pca    = reconstruction_mse(orig_s2, pca_s)

        row = {"basin_id": basin_id,
               "NSE":      float(nse_map.get(str(basin_id), np.nan))}

        for key, val in m_neural.items():
            row[f"{key}_neural"] = val
        for key, val in m_pca.items():
            row[f"{key}_pca"] = val

        row["recon_mse_neural"] = mse_neural
        row["recon_mse_pca"]    = mse_pca
        rows.append(row)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(common)} basins...")

    df = pd.DataFrame(rows)
    print(f"\nComputed metrics for {len(df)} basins.")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_DIR, "per_basin_comparison.csv"), index=False)
    print("  Saved: per_basin_comparison.csv")

    # Summary table
    metrics = ["trustworthiness", "continuity", "stress",
               "pairwise_spearman", "procrustes", "recon_mse"]
    summary_rows = []
    for metric in metrics:
        cn = f"{metric}_neural"
        cp = f"{metric}_pca"
        if cn not in df.columns and cp not in df.columns:
            continue
        summary_rows.append({
            "metric":         metric,
            "neural_mean":    round(df[cn].mean(),   4) if cn in df.columns else np.nan,
            "neural_std":     round(df[cn].std(),    4) if cn in df.columns else np.nan,
            "pca_mean":       round(df[cp].mean(),   4) if cp in df.columns else np.nan,
            "pca_std":        round(df[cp].std(),    4) if cp in df.columns else np.nan,
            "pca_var_expl":   round(var_expl, 4)    if "pca" in metric or True else np.nan,
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(OUTPUT_DIR, "summary_comparison.csv"), index=False)

    print(f"\n{'─'*70}")
    print(f"  PCA {REDUCED_DIMS}D variance explained: {var_expl*100:.1f}%")
    print(f"{'─'*70}")
    print(summary[["metric", "neural_mean", "neural_std",
                   "pca_mean", "pca_std"]].to_string(index=False))
    print(f"{'─'*70}")

    # Plots
    sns.set_theme(style="whitegrid")
    plot_boxplots(df, OUTPUT_DIR)
    plot_vs_nse(df, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
