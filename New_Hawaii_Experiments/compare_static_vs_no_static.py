"""
compare_static_vs_no_static.py
───────────────────────────────
Plots distance-from-training-distribution vs NSE (and other metrics) for two
model runs:
  - Model trained WITH    static attributes
  - Model trained WITHOUT static attributes

Each (distance metric x model-metric) pair gets one figure with:
  - Scatter points (two colours)
  - LOESS trend line + bootstrap CI band (two colours)
  - Spearman r annotation for each model
  - Shared x-axis (log scale)

Input: two basin_metrics.csv files produced by hull_analysis.py
       (columns: basin_id, sum_distance_convex_hull, sum_distance_hull_center,
                 sum_mahalanobis_distance, sum_mahalanobis_distance_extrapolated,
                 mean_mahal, n_anomalies, basin_spread_volume, NSE, ...)
"""

import os
import warnings
from pathlib import Path

import yaml
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from utils import bootstrap_spearman, bootstrap_loess

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — loaded from config.yaml (section: compare_static_vs_no_static)
# ─────────────────────────────────────────────────────────────────────────────
_cfg = yaml.safe_load(open(Path(__file__).with_name("config.yaml")))["compare_static_vs_no_static"]

CSV_WITH_STATIC    = _cfg["csv_with_static"]
CSV_NO_STATIC      = _cfg["csv_no_static"]

OUTPUT_DIR         = _cfg["output_dir"]

# The three distance metrics hull_analysis.py computes — each gets its own set
# of figures (one per model-metric below), so this produces len(DIST_COLS)
# times as many plots as the single-metric version used to.
DIST_COLS = {
    "sum_distance_convex_hull": "Sum Distance from Convex Hull Boundary (log)",
    "sum_distance_hull_center": "Sum Distance from Hull Centroid (log)",
    "sum_mahalanobis_distance": "Sum Mahalanobis Distance (log)",
}

# Which metrics to plot — None = auto-detect all *_NSE / *_RMSE / NSE / RMSE cols
METRIC_COLS        = None

# Filter: only keep basins where the NSE column > this value (set 0 to keep all)
NSE_MIN_FILTER     = _cfg["nse_min_filter"]

# LOESS smoothing fraction (0.3–0.6 works well for ~100 basins)
LOESS_FRAC         = _cfg["loess_frac"]

# Bootstrap settings
N_BOOTSTRAP        = _cfg["n_bootstrap"]
CI_LEVEL           = _cfg["ci_level"]
RNG_SEED           = _cfg["rng_seed"]

# Plot style
COLOR_WITH    = "#2196F3"   # blue  — with static
COLOR_WITHOUT = "#F44336"   # red   — without static
ALPHA_SCATTER = 0.55
ALPHA_CI      = 0.18
POINT_SIZE    = 45

LABEL_WITH    = "With static attributes"
LABEL_WITHOUT = "Without static attributes"
# ─────────────────────────────────────────────────────────────────────────────


# ── Helpers ───────────────────────────────────────────────────────────────────
# bootstrap_loess now lives in hull_analysis_common.py (identical body; every
# call site here already passes frac explicitly, so behaviour is unchanged).

def spearman_ci(x, y, n_boot=1000, ci=95, seed=42):
    """Thin wrapper around the shared bootstrap_spearman, kept for the
    (obs, lo, hi) tuple shape this script's call sites expect."""
    result = bootstrap_spearman(x, y, n_boot=n_boot, ci=ci, seed=seed)
    return result["spearman_r"], result["ci_lo"], result["ci_hi"]


def prepare_data(df, x_col, metric_col, nse_filter_col, nse_min):
    """Filter to positive extrapolation distance and minimum NSE."""
    cols = [x_col, metric_col]
    if nse_filter_col and nse_filter_col != metric_col:
        cols.append(nse_filter_col)
    sub = df[cols].dropna()
    sub = sub[sub[x_col] > 0]
    if nse_filter_col and nse_filter_col in sub.columns:
        sub = sub[sub[nse_filter_col] > nse_min]
    return sub[x_col].values, sub[metric_col].values


def annotation_text(r, lo, hi, n, label):
    return (f"{label}\n"
            f"  r = {r:.3f}  [{lo:.3f}, {hi:.3f}]\n"
            f"  n = {n}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading CSVs...")
    df_with    = pd.read_csv(CSV_WITH_STATIC)
    df_without = pd.read_csv(CSV_NO_STATIC)

    # Auto-detect metric columns from whichever file has more
    global METRIC_COLS
    if METRIC_COLS is None:
        candidates = set(df_with.columns) | set(df_without.columns)
        METRIC_COLS = [
            c for c in candidates
            if c.endswith("_NSE") or c.endswith("_RMSE")
            or c in ("NSE", "RMSE")
        ]
        METRIC_COLS = sorted(METRIC_COLS)

    print(f"Metrics to plot: {METRIC_COLS}")

    # Find NSE column for filtering (prefer first *_NSE column)
    nse_cols_w  = [c for c in df_with.columns    if c.endswith("_NSE") or c == "NSE"]
    nse_cols_wo = [c for c in df_without.columns if c.endswith("_NSE") or c == "NSE"]
    nse_filter_w  = nse_cols_w[0]  if nse_cols_w  else None
    nse_filter_wo = nse_cols_wo[0] if nse_cols_wo else None

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for x_col, x_label in DIST_COLS.items():
        for metric_col in METRIC_COLS:
            print(f"\n── Plotting: {metric_col} vs {x_col} ──")

            # Prepare each dataset
            have_w = metric_col in df_with.columns
            have_wo = metric_col in df_without.columns

            if not have_w and not have_wo:
                print(f"  [SKIP] '{metric_col}' not found in either CSV.")
                continue

            datasets = []
            if have_w:
                x_w, y_w = prepare_data(df_with, x_col, metric_col, nse_filter_w, NSE_MIN_FILTER)
                if len(x_w) >= 10:
                    datasets.append((x_w, y_w, COLOR_WITH, LABEL_WITH))
                else:
                    print(f"  [WARN] WITH-static: only {len(x_w)} points, skipping.")

            if have_wo:
                x_wo, y_wo = prepare_data(df_without, x_col, metric_col, nse_filter_wo, NSE_MIN_FILTER)
                if len(x_wo) >= 10:
                    datasets.append((x_wo, y_wo, COLOR_WITHOUT, LABEL_WITHOUT))
                else:
                    print(f"  [WARN] WITHOUT-static: only {len(x_wo)} points, skipping.")

            if not datasets:
                continue

            # ── Figure ────────────────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(11, 7))

            annotation_blocks = []

            for x, y, color, label in datasets:
                # Scatter
                ax.scatter(x, y, color=color, s=POINT_SIZE, alpha=ALPHA_SCATTER,
                           zorder=3, label=f"{label}  (n={len(x)})")

                # LOESS + CI
                x_grid, obs_loess, lo, hi = bootstrap_loess(
                    x, y, n_boot=N_BOOTSTRAP, ci=CI_LEVEL,
                    seed=RNG_SEED, frac=LOESS_FRAC
                )
                ax.fill_between(x_grid, lo, hi, color=color, alpha=ALPHA_CI, zorder=2)
                ax.plot(x_grid, obs_loess, color=color, lw=2.5, zorder=4)

                # Spearman
                r, r_lo, r_hi = spearman_ci(x, y, N_BOOTSTRAP, CI_LEVEL, RNG_SEED)
                print(f"  {label}: r={r:.3f} [{r_lo:.3f}, {r_hi:.3f}]  n={len(x)}")
                annotation_blocks.append(
                    annotation_text(r, r_lo, r_hi, len(x), label)
                )

            # Combined annotation box
            ax.text(
                0.97, 0.97,
                "\n\n".join(annotation_blocks),
                transform=ax.transAxes, fontsize=10,
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.88, ec="lightgrey"),
            )

            # Axes formatting
            ax.set_xscale("log")
            ylabel = metric_col.replace("_", " ")
            ax.set_xlabel(x_label, fontsize=13)
            ax.set_ylabel(ylabel, fontsize=13)
            ax.set_title(
                f"{ylabel}  vs  {x_label}\n"
                f"With vs Without Static Attributes",
                fontsize=13,
            )
            ax.grid(True, alpha=0.3)

            # Legend: scatter markers + LOESS lines
            legend_handles = []
            for x, y, color, label in datasets:
                legend_handles.append(
                    mlines.Line2D([], [], color=color, marker='o', linestyle='-',
                                  markersize=7, alpha=0.85,
                                  label=f"{label}  (n={len(x)})")
                )
            ax.legend(handles=legend_handles, fontsize=11,
                      loc="lower left", framealpha=0.85)

            plt.tight_layout()
            fname = f"{metric_col}_vs_{x_col}_with_vs_without_static.png"
            fpath = os.path.join(OUTPUT_DIR, fname)
            plt.savefig(fpath, dpi=180, bbox_inches="tight")
            plt.close()
            print(f"  Saved → {fpath}")

    print(f"\nAll plots written to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
