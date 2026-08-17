"""
utils.py
────────────────────────────────────────────────────────────────────────────
Shared helpers for this folder's hull/Mahalanobis extrapolation analysis
scripts (hull_analysis.py, pca_hull_analysis.py, compare_reduction.py,
combined_cv_analysis.py). Kept identical to the CAMELS-side copy in spirit —
each experiment folder owns its own copy so the folders stay self-contained,
per the project's "separate scripts, shared per-folder utils" convention.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess


def load_nse(path, basin_col, score_col):
    df = pd.read_csv(path)
    df[basin_col] = df[basin_col].astype(str)
    return df.set_index(basin_col)[score_col].to_dict()


def bootstrap_spearman(x, y, n_boot=1000, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.full(n_boot, np.nan)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            boots[i], _ = spearmanr(x[idx], y[idx])
        except Exception:
            pass
    alpha = (100 - ci) / 2
    valid = boots[~np.isnan(boots)]
    obs_r, _ = spearmanr(x, y)
    return {
        "spearman_r": obs_r,
        "ci_lo": np.percentile(valid, alpha),
        "ci_hi": np.percentile(valid, 100 - alpha),
        "p_negative": float(np.mean(valid < 0)),
        "n": n,
        "n_boot": n_boot,
        "ci_level": ci,
    }


def bootstrap_loess(x, y, n_boot=500, ci=95, seed=42, frac=0.4, n_grid=100):
    rng = np.random.default_rng(seed)
    log_x = np.log10(np.clip(x, 1e-12, None))
    x_grid = np.linspace(log_x.min(), log_x.max(), n_grid)
    curves = np.full((n_boot, n_grid), np.nan)
    for i in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        lx, yb = log_x[idx], y[idx]
        order = np.argsort(lx)
        try:
            sm = lowess(yb[order], lx[order], frac=frac, return_sorted=True)
            curves[i] = np.interp(x_grid, sm[:, 0], sm[:, 1])
        except Exception:
            pass
    alpha = (100 - ci) / 2
    lo = np.nanpercentile(curves, alpha, axis=0)
    hi = np.nanpercentile(curves, 100 - alpha, axis=0)
    order = np.argsort(log_x)
    obs_sm = lowess(y[order], log_x[order], frac=frac, return_sorted=True)
    obs_interp = np.interp(x_grid, obs_sm[:, 0], obs_sm[:, 1])
    return 10**x_grid, obs_interp, lo, hi


def binned_trend(x, y, n_bins=5):
    """Divide x into n_bins quantile bins; return bin centres, mean, std, count."""
    df = pd.DataFrame({"x": x, "y": y})
    df["bin"] = pd.qcut(df["x"], q=n_bins, duplicates="drop")
    stats = df.groupby("bin", observed=True)["y"].agg(
        mean="mean", std="std", count="count"
    ).reset_index()
    stats["x_centre"] = stats["bin"].apply(lambda b: b.mid)
    return stats
