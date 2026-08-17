"""
combined_cv_analysis.py
────────────────────────────────────────────────────────────────────────────────
Loads two basin_metrics_pca.csv files from a 50-50 cross-validation,
concatenates them (each basin appears once with its correct-fold NSE),
then runs bootstrapped Spearman correlation and LOESS plots for all
four distance metrics across all basins.
"""

import os
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess

# CONFIG — loaded from config.yaml (section: combined_cv_analysis)

_cfg = yaml.safe_load(open(Path(__file__).with_name("config.yaml")))["combined_cv_analysis"]

CSV_FOLD_1   = _cfg["csv_fold_1"]
CSV_FOLD_2   = _cfg["csv_fold_2"]
OUTPUT_DIR   = _cfg["output_dir"]

NSE_MIN_FILTER  = _cfg["nse_min_filter"]
N_BOOTSTRAP     = _cfg["n_bootstrap"]
CI_LEVEL        = _cfg["ci_level"]
N_BINS          = _cfg["n_bins"]
RNG_SEED        = _cfg["rng_seed"]
PCA_DIMS        = _cfg["pca_dims"]  

PLOT_SPECS = [
    ("sum_distance_convex_hull",
     "Sum of Extrapolated Dist. From Convex Hull (log)"),
    ("sum_distance_hull_center",
     "Sum Distance from Geometric Hull Center (log)"),
    ("sum_mahalanobis_distance",
     "Sum of Mahalanobis Dist. (log)"),
    ("sum_mahalanobis_distance_extrapolated",
     "Sum of Mahalanobis Dist. for Extrapolated States (log)"),
]


# LOAD & COMBINE

def load_and_combine(csv1: str, csv2: str) -> pd.DataFrame:
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    print(f"Fold 1: {len(df1)} basins  |  Fold 2: {len(df2)} basins")

    # Sanity check — no basin should appear in both folds
    overlap = set(df1["basin_id"].astype(str)) & set(df2["basin_id"].astype(str))
    if overlap:
        print(f"  [WARN] {len(overlap)} basin(s) appear in both folds — "
              f"check your CSV files. Overlapping: {list(overlap)[:5]}")

    combined = pd.concat([df1, df2], ignore_index=True)
    combined["basin_id"] = combined["basin_id"].astype(str)
    print(f"Combined: {len(combined)} basins total")
    return combined



# BOOTSTRAP

def bootstrap_spearman(x, y, n_boot=1500, ci=95, seed=42):
    rng   = np.random.default_rng(seed)
    boots = np.full(n_boot, np.nan)
    for i in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        try:
            boots[i], _ = spearmanr(x[idx], y[idx])
        except Exception:
            pass
    alpha  = (100 - ci) / 2
    valid  = boots[~np.isnan(boots)]
    obs, _ = spearmanr(x, y)
    return {
        "spearman_r": obs,
        "ci_lo":      np.percentile(valid, alpha),
        "ci_hi":      np.percentile(valid, 100 - alpha),
        "p_negative": float(np.mean(valid < 0)),
        "n":          len(x),
        "n_boot":     n_boot,
        "ci_level":   ci,
    }


def bootstrap_loess(x, y, n_boot=500, ci=95, seed=42, frac=0.4, n_grid=100):
    rng    = np.random.default_rng(seed)
    log_x  = np.log10(np.clip(x, 1e-12, None))
    x_grid = np.linspace(log_x.min(), log_x.max(), n_grid)
    curves = np.full((n_boot, n_grid), np.nan)
    for i in range(n_boot):
        idx   = rng.integers(0, len(x), size=len(x))
        lx    = log_x[idx]
        yb    = y[idx]
        order = np.argsort(lx)
        try:
            sm = lowess(yb[order], lx[order], frac=frac, return_sorted=True)
            curves[i] = np.interp(x_grid, sm[:, 0], sm[:, 1])
        except Exception:
            pass
    alpha      = (100 - ci) / 2
    lo         = np.nanpercentile(curves, alpha,       axis=0)
    hi         = np.nanpercentile(curves, 100 - alpha, axis=0)
    order      = np.argsort(log_x)
    obs_sm     = lowess(y[order], log_x[order], frac=frac, return_sorted=True)
    obs_interp = np.interp(x_grid, obs_sm[:, 0], obs_sm[:, 1])
    return 10**x_grid, obs_interp, lo, hi


def binned_trend(x, y, n_bins=5):
    df        = pd.DataFrame({"x": x, "y": y})
    df["bin"] = pd.qcut(df["x"], q=n_bins, duplicates="drop")
    stats     = df.groupby("bin", observed=True)["y"].agg(
        mean="mean", std="std", count="count"
    ).reset_index()
    stats["x_centre"] = stats["bin"].apply(lambda b: b.mid)
    return stats



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
            f"n = {boot['n']}  |  {boot['n_boot']} bootstrap\n"
            f"[50-50 CV combined]",
            transform=ax.transAxes, fontsize=10, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(
        f"PCA {PCA_DIMS}D  |  NSE > {NSE_MIN_FILTER}  |  n={boot['n']}  "
        f"|  50-50 CV all basins",
        fontsize=10, color="gray"
    )
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
    fname = f"NSE_vs_{x_col}_combined_cv.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close()
    print(f"  Saved: {fname}")



# MAIN

    # Load and combine both folds
    df = load_and_combine(CSV_FOLD_1, CSV_FOLD_2)

    # Filter by NSE
    df = df[df["NSE"] > NSE_MIN_FILTER].copy()
    print(f"After NSE > {NSE_MIN_FILTER} filter: {len(df)} basins")

    n_extrap = (df["sum_distance_convex_hull"] > 0).sum()
    print(f"Basins outside convex hull: {n_extrap}")

    # Save combined CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    combined_csv = os.path.join(OUTPUT_DIR, "basin_metrics_combined_cv.csv")
    df.to_csv(combined_csv, index=False)
    print(f"Combined CSV saved: {combined_csv}")

    # Run plots
    sns.set_theme(style="whitegrid")
    boot_rows = []

    for x_col, xlabel in PLOT_SPECS:
        print(f"\n── {x_col} ──")
        sub = df[["NSE", x_col]].dropna()
        sub = sub[sub[x_col] > 0]

        if len(sub) < 10:
            print(f"  [WARN] Only {len(sub)} non-zero points — skipping.")
            continue

        x, y = sub[x_col].values, sub["NSE"].values
        print(f"  Bootstrapping Spearman ({len(x)} basins, {N_BOOTSTRAP} iterations)...")
        boot         = bootstrap_spearman(x, y, N_BOOTSTRAP, CI_LEVEL, RNG_SEED)
        loess_result = bootstrap_loess(x, y, 500, CI_LEVEL, RNG_SEED)
        bins         = binned_trend(x, y, N_BINS)

        plot_xy(x, y, OUTPUT_DIR, x_col, xlabel, boot, loess_result, bins)

        print(f"  r={boot['spearman_r']:.3f} "
              f"[{boot['ci_lo']:.3f}, {boot['ci_hi']:.3f}]  "
              f"P(r<0)={boot['p_negative']:.3f}  n={len(x)}")

        boot_rows.append({
            "distance_metric":  x_col,
            "pca_dims":         PCA_DIMS,
            "n_basins":         len(x),
            "spearman_r":       round(boot["spearman_r"], 3),
            "spearman_ci_lo":   round(boot["ci_lo"],      3),
            "spearman_ci_hi":   round(boot["ci_hi"],      3),
            "p_negative":       round(boot["p_negative"], 3),
        })

    if boot_rows:
        summary = pd.DataFrame(boot_rows)
        out     = os.path.join(OUTPUT_DIR, "bootstrap_summary_combined_cv.csv")
        summary.to_csv(out, index=False)
        print(f"\nBootstrap summary saved: {out}")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
