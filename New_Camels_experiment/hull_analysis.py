"""
hull_analysis.py  (CAMELS_US)
──────────────────────────────
Canonical raw-7D (autoencoder-reduced) convex-hull / Mahalanobis extrapolation
analysis. Addresses reviewer comments on statistical robustness:
  1. Bootstrap CI on Spearman r (robust to tail outliers)
  2. Bootstrapped LOESS band instead of polynomial (no overfitting)
  3. Binned trend with mean ± std error bars (shows trend holds across full range)

Run paths and thresholds come from config.yaml (section: hull_analysis) in
this same folder — edit that file to point at a new training run, nothing
in this script needs to change.

Usage:
    python hull_analysis.py              # plots only
    python hull_analysis.py --viz basin  # + 3D view coloured by basin
    python hull_analysis.py --viz nse    # + 3D view coloured by NSE
"""

import os
import re
import time
import argparse
import warnings
from pathlib import Path

import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns

from scipy.spatial import ConvexHull, KDTree, Delaunay
from sklearn.covariance import LedoitWolf

from utils import bootstrap_spearman, bootstrap_loess, binned_trend

# CONFIG — loaded from config.yaml (section: hull_analysis)
_cfg = yaml.safe_load(open(Path(__file__).with_name("config.yaml")))["hull_analysis"]

TRAIN_FILE      = _cfg["train_file"]
TEST_DIR        = _cfg["test_dir"]
NSE_FILE        = _cfg["nse_file"]
NSE_BASIN_COL   = _cfg["nse_basin_col"]
NSE_SCORE_COL   = _cfg["nse_score_col"]
OUTPUT_DIR      = _cfg["output_dir"]

FILE_PATTERN    = _cfg["file_pattern"]
MAHAL_THRESHOLD = _cfg["mahal_threshold"]
MAX_HULL_PTS    = _cfg["max_hull_pts"]        # subsample train points before ConvexHull
NSE_MIN_FILTER  = _cfg["nse_min_filter"]      # basins with NSE > this value
N_BOOTSTRAP     = _cfg["n_bootstrap"]
CI_LEVEL        = _cfg["ci_level"]
N_BINS          = _cfg["n_bins"]              # quantile bins for binned trend plot
RNG_SEED        = _cfg["rng_seed"]

# Optional: highlight a reference/perturbed basin pair on the scatter plots
# (used for perturbation-sensitivity checks, see pretub_check.py). Leave both
# as null in config.yaml to disable — no highlighting.
REFERENCE_BASIN = _cfg["reference_basin"]
PERTURBED_BASIN = _cfg["perturbed_basin"]


# DATA LOADING
def load_train(path):
    data = torch.load(path, map_location="cpu").numpy()
    data = data[~np.isnan(data).any(axis=1)]
    print(f"Train: {data.shape[0]} points, dim {data.shape[1]}")
    return data


def load_test(test_dir, pattern, train_dim):
    basin_data = {}
    found = 0
    for fname in os.listdir(test_dir):
        if not re.search(pattern, fname):
            continue
        found += 1
        m = re.search(pattern, fname)
        basin_id = m.group(1) if m else None
        if basin_id is None:
            continue
        try:
            raw   = torch.load(os.path.join(test_dir, fname), map_location="cpu").numpy()
            clean = raw[~np.isnan(raw).any(axis=1)]
            if clean.shape[1] != train_dim:
                print(f"[WARN] Skipping {fname}: dim mismatch ({clean.shape[1]} vs {train_dim})")
                continue
            basin_data[basin_id] = clean
        except Exception as e:
            print(f"[WARN] Skipping {fname}: {e}")
    print(f"Test: {len(basin_data)} basins from {found} files.")
    return basin_data


def load_nse(nse_file, basin_col, score_col):
    try:
        df = pd.read_csv(nse_file)
        df[basin_col] = df[basin_col].astype(str)
        return df.set_index(basin_col)[score_col].to_dict()
    except Exception as e:
        print(f"[WARN] Could not load NSE file: {e}")
        return {}


# GEOMETRY
def subsample_for_hull(data, max_pts, seed=42):
    if len(data) <= max_pts:
        return data
    idx = np.random.default_rng(seed).choice(len(data), size=max_pts, replace=False)
    return data[idx]


def build_hull(train_data):
    hull_data = subsample_for_hull(train_data, MAX_HULL_PTS, RNG_SEED)
    print(f"Building convex hull on {hull_data.shape[0]} of {train_data.shape[0]} train points "
          f"(subsampled for tractability).")
    try:
        hull = ConvexHull(hull_data)
        tree = KDTree(hull_data[hull.vertices])
        print("Training convex hull built.")
        return hull, tree
    except Exception as e:
        print(f"[WARN] Could not build hull: {e}")
        return None, None


def dist_to_hull_batch(points, hull, tree, batch=2000):
    
    dists, _ = tree.query(points)
    A, b     = hull.equations[:, :-1], hull.equations[:, -1]
    inside   = np.empty(len(points), dtype=bool)
    for i in range(0, len(points), batch):
        inside[i:i+batch] = np.all(points[i:i+batch] @ A.T + b <= 1e-7, axis=1)
    dists[inside] = 0.0
    return dists


def compute_mahalanobis(train_data, points):
    cov  = LedoitWolf().fit(train_data)
    VI   = cov.get_precision()
    diff = points - cov.location_
    sq   = np.sum(diff @ VI * diff, axis=1)
    return np.sqrt(np.clip(sq, 0, None))


# PER-BASIN METRICS
def compute_per_basin_metrics(basin_data, train_data, hull,
                               tree, centroid, mahal_threshold):
    all_points = np.vstack(list(basin_data.values()))

    print("Computing Mahalanobis distances...")
    all_mahal = compute_mahalanobis(train_data, all_points)

    print("Computing hull and centroid distances...")
    all_hull_dist = (dist_to_hull_batch(all_points.copy(), hull, tree)
                     if hull is not None else np.zeros(len(all_points)))
    all_centroid_dist = np.linalg.norm(all_points - centroid, axis=1)

    vol_rng = np.random.default_rng(RNG_SEED)
    vol_cap = 500  # 7D ConvexHull cost explodes with point count (~17s at n=3652); subsample

    rows = []
    idx  = 0
    for basin_id, pts in basin_data.items():
        n          = len(pts)
        mahal      = all_mahal[idx:idx+n]
        hull_dist  = all_hull_dist[idx:idx+n]
        cent_dist  = all_centroid_dist[idx:idx+n]
        is_extrap  = mahal > mahal_threshold

        vol = 0.0
        if n >= 4:
            try:
                vol_pts = pts if n <= vol_cap else pts[vol_rng.choice(n, size=vol_cap, replace=False)]
                vol = ConvexHull(vol_pts).volume
            except Exception:
                pass

        rows.append({
            "basin_id":                              basin_id,
            "n_points":                              n,
            "sum_distance_convex_hull":              float(hull_dist.sum()),
            "sum_distance_hull_center":              float(cent_dist.sum()),
            "sum_mahalanobis_distance":              float(mahal.sum()),
            "sum_mahalanobis_distance_extrapolated": float(mahal[is_extrap].sum()),
            "mean_mahal":                            float(mahal.mean()),
            "n_anomalies":                           int(is_extrap.sum()),
            "basin_spread_volume":                   vol,
        })
        idx += n

    df = pd.DataFrame(rows)
    print(f"Computed metrics for {len(df)} basins.")
    return df




# PLOT
def _get_special_xy(basin_df, basin_id, x_col):
    """Look up a single basin's (x, y) point for scatter highlighting."""
    if basin_df is None or basin_id is None:
        return None, None
    mask = basin_df["basin_id"].astype(str) == str(basin_id)
    if not mask.any():
        return None, None
    row = basin_df[mask].iloc[0]
    return row[x_col], row["NSE"]


def plot_xy(x_all, y_all, x_log, output_dir, x_col, xlabel,
            boot_sp, loess_result, bin_stats, ylabel="NSE", basin_df=None):
    """
    x_all  — full x values (including zeros, for bootstrap)
    x_log  — x values > 0 (for log-scale scatter and LOESS)
    y_all  — NSE for x_all
    y_log  — NSE for x_log
    basin_df — optional DataFrame with columns [basin_id, NSE, x_col], used
               only to highlight REFERENCE_BASIN / PERTURBED_BASIN if set.
    """
    mask  = x_all > 0
    x_pos = x_all[mask]
    y_pos = y_all[mask]

    x_grid, obs_loess, lo, hi = loess_result

    plt.rcParams.update({"font.size": 12})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                             gridspec_kw={"width_ratios": [2, 1]})

    # Left panel: scatter + bootstrapped LOESS band
    ax = axes[0]
    sc = ax.scatter(x_pos, y_pos, c=y_pos, cmap="viridis",
                    s=40, alpha=0.6, zorder=3, label="test basin")

    # Bootstrapped LOESS CI band
    ax.fill_between(x_grid, lo, hi, alpha=0.25, color="steelblue",
                    label=f"LOESS {boot_sp['ci_level']}% CI (bootstrap)")
    ax.plot(x_grid, obs_loess, color="steelblue", lw=2.5, label="LOESS (observed)")

    rx, ry = _get_special_xy(basin_df, REFERENCE_BASIN, x_col)
    if rx is not None:
        ax.scatter([rx], [ry], s=100, color="blue", zorder=8,
                   label=f"Reference ({REFERENCE_BASIN})")
    px, py = _get_special_xy(basin_df, PERTURBED_BASIN, x_col)
    if px is not None:
        ax.scatter([px], [py], s=100, color="red", zorder=9,
                   label=f"Perturbed ({PERTURBED_BASIN})")

    ax.set_xscale("log")
    plt.colorbar(sc, ax=ax).set_label(ylabel, fontsize=12)

    # Stats annotation
    stats_text = (
        f"Spearman r = {boot_sp['spearman_r']:.2f}\n"
        f"{boot_sp['ci_level']}% CI  "
        f"[{boot_sp['ci_lo']:.2f}, {boot_sp['ci_hi']:.2f}]\n"
        f"P(r < 0) = {boot_sp['p_negative']:.2f}\n"
        f"n = {boot_sp['n']} basins  |  {boot_sp['n_boot']} bootstrap"
    )
    ax.text(0.97, 0.97, stats_text,
            transform=ax.transAxes, fontsize=10,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"NSE > {NSE_MIN_FILTER}  |  n = {boot_sp['n']} basins",
                 fontsize=10, color="gray")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="lower left")

    # Right panel: binned mean ± std
    ax2 = axes[1]
    ax2.errorbar(
        bin_stats["x_centre"], bin_stats["mean"],
        yerr=bin_stats["std"],
        fmt="o-", color="steelblue", capsize=5,
        elinewidth=1.5, linewidth=2, markersize=7,
        label="Mean NSE ± std"
    )
    # Annotate n per bin
    for _, row in bin_stats.iterrows():
        ax2.annotate(f"n={int(row['count'])}",
                     (row["x_centre"], row["mean"] + (row["std"] or 0) + 0.01),
                     fontsize=9, ha="center", color="gray")

    ax2.set_xscale("log")
    ax2.set_xlabel(xlabel, fontsize=12)
    ax2.set_ylabel(ylabel, fontsize=12)
    ax2.set_title(f"Binned trend ({N_BINS} quantile bins)", fontsize=10, color="gray")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fname = f"NSE_vs_{x_col}_robust.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close()
    print(f"  Saved: {fname}")


# 3D VIZ
def visualise_3d(train_data, basin_data, nse_map, hull, color_mode):
    try:
        import pyvista as pv
        import matplotlib.colors as mcolors
        import matplotlib.cm as cm
    except ImportError:
        print("[WARN] pyvista not available, skipping 3D visualisation.")
        return

    all_points    = np.vstack(list(basin_data.values()))
    all_basin_ids = np.concatenate([
        np.full(len(pts), bid) for bid, pts in basin_data.items()
    ])

    pv.global_theme.font.size = 14
    pv.set_plot_theme("document")
    plotter = pv.Plotter(window_size=(1200, 900))
    legend  = []

    if hull is not None:
        faces = np.hstack([[3, *f] for f in hull.simplices])
        mesh  = pv.PolyData(train_data)
        mesh.faces = faces
        plotter.add_mesh(mesh, color="brown", opacity=0.25, show_edges=True)
        legend.append(["Convex Hull (Train)", "brown"])

    plotter.add_points(train_data, color="blue", point_size=4,
                       render_points_as_spheres=True)
    legend.append(["Train Data", "blue"])

    unique_basins = list(basin_data.keys())

    if color_mode == "basin":
        cmap_c = plt.get_cmap("tab20").colors
        for i, basin in enumerate(unique_basins):
            c = cmap_c[i % len(cmap_c)]
            plotter.add_points(basin_data[basin], color=c,
                               point_size=7, render_points_as_spheres=True)
            legend.append([f"Basin {basin}", c])
        title = "Cell States — Coloured by Basin ID"
    else:
        valid = [float(v) for v in nse_map.values() if not np.isnan(float(v))]
        vmin, vmax = (min(valid), max(valid)) if valid else (-1.0, 1.0)
        cmap_fn = cm.get_cmap("RdYlGn")
        norm    = mcolors.Normalize(vmin=vmin, vmax=vmax)
        for basin in unique_basins:
            nv  = nse_map.get(str(basin), np.nan)
            col = (0.5, 0.5, 0.5) if np.isnan(float(nv)) \
                  else cmap_fn(norm(float(nv)))[:3]
            plotter.add_points(basin_data[basin], color=col,
                               point_size=7, render_points_as_spheres=True)
        nse_vals     = np.array([nse_map.get(str(b), np.nan)
                                 for b in all_basin_ids], dtype=float)
        cloud        = pv.PolyData(all_points)
        cloud["NSE"] = nse_vals
        plotter.add_mesh(cloud, scalars="NSE", cmap="RdYlGn",
                         clim=[vmin, vmax], point_size=0, show_scalar_bar=True,
                         scalar_bar_args={"title": "NSE", "vertical": True})
        title = "Cell States — Coloured by NSE Score"

    plotter.add_legend(labels=legend, bcolor="white", border=True,
                       size=(0.18, 0.12))
    plotter.show(title=title)


# MAIN
def main(viz_mode=None):
    t0 = time.perf_counter()

    # Load
    train_data = load_train(TRAIN_FILE)
    nse_map    = load_nse(NSE_FILE, NSE_BASIN_COL, NSE_SCORE_COL)
    basin_data = load_test(TEST_DIR, FILE_PATTERN, train_data.shape[1])
    if not basin_data:
        print("[ERROR] No test data loaded.")
        return

    # Geometry
    hull, tree = build_hull(train_data)
    centroid = train_data.mean(axis=0)

    # Metrics in memory
    df = compute_per_basin_metrics(
        basin_data, train_data, hull, tree, centroid, MAHAL_THRESHOLD
    )

    # Attach NSE and filter
    df["NSE"] = df["basin_id"].map(lambda b: float(nse_map.get(str(b), np.nan)))
    df = df[df["NSE"] > NSE_MIN_FILTER].copy()
    print(f"Basins after NSE > {NSE_MIN_FILTER} filter: {len(df)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_DIR, "basin_metrics.csv"), index=False)

    # Plot specs
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

        sub = df[["basin_id", "NSE", x_col]].dropna()
        if len(sub) < 10:
            print(f"  [WARN] Only {len(sub)} points, skipping.")
            continue

        x_all = sub[x_col].values
        y_all = sub["NSE"].values

        # Bootstrap Spearman on ALL basins (including zero-distance ones)
        print(f"  Bootstrapping Spearman r ({len(x_all)} basins)...")
        boot_sp = bootstrap_spearman(x_all, y_all,
                                     n_boot=N_BOOTSTRAP, ci=CI_LEVEL, seed=RNG_SEED)

        # LOESS band — only on x > 0 for log scale
        mask  = x_all > 0
        x_pos = x_all[mask]
        y_pos = y_all[mask]

        if len(x_pos) < 10:
            print(f"  [WARN] Too few non-zero points for LOESS, skipping.")
            continue

        print(f"  Bootstrapping LOESS band ({len(x_pos)} non-zero points)...")
        loess_result = bootstrap_loess(x_pos, y_pos,
                                       n_boot=500, ci=CI_LEVEL, seed=RNG_SEED)

        # Binned trend on non-zero x
        bins = binned_trend(x_pos, y_pos, n_bins=N_BINS)

        # Plot
        plot_xy(x_all, y_all, x_pos, OUTPUT_DIR, x_col, xlabel,
                boot_sp, loess_result, bins, basin_df=sub)

        print(f"  Spearman r={boot_sp['spearman_r']:.3f} "
              f"[{boot_sp['ci_lo']:.3f}, {boot_sp['ci_hi']:.3f}]  "
              f"P(r<0)={boot_sp['p_negative']:.3f}")

        boot_rows.append({
            "distance_metric":     x_col,
            "n_basins_total":      len(x_all),
            "n_basins_nonzero":    int(mask.sum()),
            "spearman_r":          round(boot_sp["spearman_r"],  3),
            "spearman_ci_lo":      round(boot_sp["ci_lo"],        3),
            "spearman_ci_hi":      round(boot_sp["ci_hi"],        3),
            "p_negative":          round(boot_sp["p_negative"],   3),
        })

    # Summary
    if boot_rows:
        tbl = pd.DataFrame(boot_rows)
        out = os.path.join(OUTPUT_DIR, "bootstrap_summary.csv")
        tbl.to_csv(out, index=False)
        print(f"\nBootstrap summary:\n{tbl.to_string(index=False)}")
        print(f"Saved to {out}")

    if viz_mode:
        visualise_3d(train_data, basin_data, nse_map, hull, viz_mode)

    print(f"\nTotal runtime: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--viz", choices=["basin", "nse"], default=None,
                        help="Optional 3D visualisation colour mode.")
    args = parser.parse_args()
    main(viz_mode=args.viz)
