"""
seven_time.py
──────────────
Compares two models (e.g. with vs without static attributes) by plotting:
  - Normalized rolling average NNSE across all basins [0, 1]
    (8-step window × 3h = 1 day effective window)
  - Per-timestep hull extrapolation distance [0, 1] computed directly from
    reduced state .pt files

Normalization:
  - Both NNSE and hull distance normalized to [0, 1]
  - Both metrics use shared global min/max across both models
  - Hull distance min fixed at 0 (inside hull = 0, outside = positive)

Alignment fix:
  With predict_last_n=10, states are saved once per sequence.
  NNSE is subsampled every PREDICT_LAST_N steps to align correctly.

NNSE = 1 / (2 - NSE)
"""

import os
import re
import pickle
import warnings
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from scipy.spatial import ConvexHull

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — loaded from config.yaml (section: seven_time)
# ─────────────────────────────────────────────────────────────────────────────
_cfg = yaml.safe_load(open(Path(__file__).with_name("config.yaml")))["seven_time"]
_m1, _m2 = _cfg["model_1"], _cfg["model_2"]

PICKLE1   = _m1["pickle"]
TRAIN_PT1 = _m1["train_pt"]
VAL_PT1   = _m1["val_pt"]
TEST_PT1  = _m1["test_pt"]
LABEL1    = _m1["label"]

# Model 2 (e.g. without static attributes)
PICKLE2   = _m2["pickle"]
TRAIN_PT2 = _m2["train_pt"]
VAL_PT2   = _m2["val_pt"]
TEST_PT2  = _m2["test_pt"]
LABEL2    = _m2["label"]


OUTPUT_PATH = _cfg["output_path"]

# Rolling NNSE window — 8 × 3h = 1 day
WINDOW = _cfg["window"]

# Must match predict_last_n in your config yaml
PREDICT_LAST_N = _cfg["predict_last_n"]

# Optional: exclude basins where mean NNSE < this (None = keep all)
NNSE_MIN_MEAN = _cfg["nnse_min_mean"]

# Filename patterns
TRAIN_PATTERN = r'c_n_reduced_epoch(\d+)\.pt$'
VAL_PATTERN   = r'c_n_reduced_epoch(\d+)_validation\.pt$'
TEST_PATTERN  = r'c_n_reduced_epoch(\d+)_basin_([^\.]+)\.pt$'

# Hull distance chunk size — lower if RAM is tight
FACE_CHUNK = 100_000


# ─────────────────────────────────────────────────────────────────────────────
# BASIN ID NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────
def clean_id(s):
    s = str(s).strip()
    s = re.sub(r'\.0+$', '', s)
    s = re.sub(r'_0$',   '', s)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# NSE / NNSE
# ─────────────────────────────────────────────────────────────────────────────
def calculate_nse(obs, sim):
    obs  = np.asarray(obs, dtype=float)
    sim  = np.asarray(sim, dtype=float)
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    obs, sim = obs[mask], sim[mask]
    if len(obs) == 0:
        return np.nan
    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom == 0:
        return np.nan
    return 1 - np.sum((sim - obs) ** 2) / denom


def nse_to_nnse(nse):
    """NNSE = 1 / (2 - NSE). Bounded [0, 1]."""
    return 1.0 / (2.0 - nse)


def rolling_nnse_series(obs, sim, window):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    n   = len(obs)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        nse = calculate_nse(obs[i - window + 1:i + 1],
                            sim[i - window + 1:i + 1])
        if not np.isnan(nse):
            out[i] = nse_to_nnse(nse)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# .PT FILE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def find_last_epoch_file(folder, pattern):
    matches = []
    for fname in os.listdir(folder):
        m = re.search(pattern, fname)
        if m:
            matches.append((int(m.group(1)), fname))
    return max(matches, key=lambda x: x[0])[1] if matches else None


def load_flat(folder, pattern):
    fname = find_last_epoch_file(folder, pattern)
    if fname is None:
        print(f"  [WARN] No file matching '{pattern}' in {folder}")
        return None
    t   = torch.load(os.path.join(folder, fname), map_location='cpu')
    arr = t.reshape(-1, t.shape[-1]).numpy().astype(np.float64)
    arr = arr[~np.isnan(arr).any(axis=1)]
    print(f"  Loaded {fname}: {arr.shape}")
    return arr


def load_test_basin_states(test_dir):
    basin_files = {}
    for fname in os.listdir(test_dir):
        m = re.search(TEST_PATTERN, fname)
        if not m:
            continue
        epoch    = int(m.group(1))
        basin_id = clean_id(m.group(2))
        if basin_id not in basin_files or epoch > basin_files[basin_id][0]:
            basin_files[basin_id] = (epoch, fname)

    result = {}
    for basin_id, (_, fname) in basin_files.items():
        t   = torch.load(os.path.join(test_dir, fname), map_location='cpu')
        arr = t.reshape(-1, t.shape[-1]).numpy().astype(np.float64)
        result[basin_id] = arr
    print(f"  Loaded {len(result)} test basin state files.")
    if result:
        print(f"  Sample state IDs: {list(result.keys())[:3]}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# HULL CONSTRUCTION + DISTANCE
# ─────────────────────────────────────────────────────────────────────────────
def build_hull(train_dir, val_dir):
    print("  Building convex hull from train + val states...")
    train  = load_flat(train_dir, TRAIN_PATTERN)
    val    = load_flat(val_dir,   VAL_PATTERN)
    arrays = [x for x in [train, val] if x is not None]
    if not arrays:
        raise RuntimeError("No train/val .pt files found.")
    ref = np.vstack(arrays)
    print(f"  Reference shape: {ref.shape}")
    hull    = ConvexHull(ref)
    normals = hull.equations[:, :-1]
    offsets = hull.equations[:, -1]
    print(f"  Hull: {len(hull.simplices):,} faces, "
          f"{len(hull.vertices):,} vertices.")
    return normals, offsets


def hull_distance_per_timestep(states, normals, offsets):
    N          = len(states)
    F          = len(offsets)
    max_signed = np.full(N, -np.inf)
    for start in range(0, F, FACE_CHUNK):
        end   = min(start + FACE_CHUNK, F)
        chunk = states @ normals[start:end].T + offsets[start:end]
        np.maximum(max_signed, chunk.max(axis=1), out=max_signed)
    return np.where(max_signed > 0, max_signed, 0.0)


def compute_hull_distances(states, normals, offsets):
    nan_mask = np.isnan(states).any(axis=1)
    out      = np.full(len(states), np.nan)
    valid    = states[~nan_mask]
    if len(valid) > 0:
        out[~nan_mask] = hull_distance_per_timestep(valid, normals, offsets)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PER-BASIN PICKLE PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────
def process_basin_pickle(basin_tuple, window):
    basin_id, entry = basin_tuple
    try:
        xr_obj = entry["3h"]["xr"]
        if isinstance(xr_obj, xr.DataArray):
            xr_obj = xr_obj.to_dataset(name="data_var")
        df = xr_obj.to_dataframe().reset_index()

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        elif "time" in df.columns:
            df["date"] = pd.to_datetime(df["time"])
        else:
            return None

        var_bases = set()
        for col in df.columns:
            m = re.match(r"(.+)_obs$", col) or re.match(r"(.+)_sim$", col)
            if m:
                var_bases.add(m.group(1))

        nnse_arrays = []
        for var in var_bases:
            obs_col = f"{var}_obs"
            sim_col = f"{var}_sim"
            if obs_col in df.columns and sim_col in df.columns:
                arr = rolling_nnse_series(df[obs_col].values,
                                          df[sim_col].values, window)
                if not np.all(np.isnan(arr)):
                    nnse_arrays.append(arr)

        if not nnse_arrays:
            return None

        nnse_mean = np.nanmean(np.vstack(nnse_arrays), axis=0)

        if NNSE_MIN_MEAN is not None and np.nanmean(nnse_mean) < NNSE_MIN_MEAN:
            return None

        return pd.DataFrame({
            "date":              df["date"].values,
            "rolling_nnse_mean": nnse_mean,
            "basin_id":          clean_id(basin_id),
        })

    except Exception as e:
        print(f"  [WARN] Basin {basin_id}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZE [0, 1]
# ─────────────────────────────────────────────────────────────────────────────
def normalize_0_1(series, lo, hi):
    """Normalize series to [0, 1] using provided global lo and hi."""
    series = np.asarray(series, dtype=float)
    if hi == lo:
        return np.zeros_like(series)
    return (series - lo) / (hi - lo)


# ─────────────────────────────────────────────────────────────────────────────
# PROCESS ONE MODEL
# ─────────────────────────────────────────────────────────────────────────────
def process_model(pickle_path, train_dir, val_dir, test_dir, label, window):
    print(f"\n{'═' * 60}")
    print(f"Processing: {label}")

    normals, offsets = build_hull(train_dir, val_dir)

    print("  Loading test basin reduced states...")
    basin_states = load_test_basin_states(test_dir)

    print(f"  Loading pickle and computing rolling NNSE (window={window})...")
    with open(pickle_path, "rb") as f:
        results = pickle.load(f)
    print(f"  Basins in pickle: {len(results)}")

    func = partial(process_basin_pickle, window=window)
    with Pool(processes=cpu_count()) as pool:
        nnse_list = list(tqdm(
            pool.imap(func, list(results.items())),
            total=len(results),
            desc=f"  NNSE [{label[:25]}]"
        ))

    nnse_dfs = {}
    for d in nnse_list:
        if d is not None:
            bid = d["basin_id"].iloc[0]
            nnse_dfs[bid] = d
    print(f"  Valid NNSE basins: {len(nnse_dfs)}")
    if nnse_dfs:
        print(f"  Sample NNSE IDs:  {list(nnse_dfs.keys())[:3]}")

    common_basins = set(nnse_dfs.keys()) & set(basin_states.keys())
    print(f"  Basins with both NNSE and states: {len(common_basins)}")
    if len(common_basins) == 0:
        print("  [ERROR] No basin ID overlap — check ID formats:")
        print(f"    NNSE IDs  (first 5): {list(nnse_dfs.keys())[:5]}")
        print(f"    State IDs (first 5): {list(basin_states.keys())[:5]}")
        raise RuntimeError(f"No basin ID overlap for {label}.")

    print(f"  Aligning NNSE (subsampled every {PREDICT_LAST_N} steps) "
          f"with hull distances...")
    aligned_frames = []

    for basin_id in tqdm(common_basins, desc="  Aligning", ncols=72):
        df_nnse = nnse_dfs[basin_id].copy()
        states  = basin_states[basin_id]

        # Subsample NNSE to match state frequency
        df_nnse_sub = df_nnse.iloc[PREDICT_LAST_N - 1::PREDICT_LAST_N].reset_index(drop=True)

        dists = compute_hull_distances(states, normals, offsets)

        n = min(len(df_nnse_sub), len(dists))
        if len(df_nnse_sub) != len(dists):
            print(f"  [WARN] Basin {basin_id}: subsampled NNSE={len(df_nnse_sub)}, "
                  f"states={len(dists)} — trimming to {n}")

        df_aligned             = df_nnse_sub.iloc[:n].copy()
        df_aligned["distance"] = dists[:n]
        aligned_frames.append(df_aligned)

    if not aligned_frames:
        raise RuntimeError(f"No aligned basins for {label}")

    df_all     = pd.concat(aligned_frames, ignore_index=True)
    df_by_date = (df_all
                  .groupby("date")[["rolling_nnse_mean", "distance"]]
                  .mean()
                  .reset_index()
                  .sort_values("date"))

    print(f"  Date range:  {df_by_date['date'].min()} – "
          f"{df_by_date['date'].max()}")
    print(f"  NNSE range:  {df_by_date['rolling_nnse_mean'].min():.3f} – "
          f"{df_by_date['rolling_nnse_mean'].max():.3f}")
    print(f"  Dist range:  {df_by_date['distance'].min():.3f} – "
          f"{df_by_date['distance'].max():.3f}")

    return df_by_date


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    df1 = process_model(PICKLE1, TRAIN_PT1, VAL_PT1, TEST_PT1, LABEL1, WINDOW)
    df2 = process_model(PICKLE2, TRAIN_PT2, VAL_PT2, TEST_PT2, LABEL2, WINDOW)

    # ── Global normalization to [0, 1] across both models ─────────────────────

    # NNSE: shared global min/max across both models
    nnse_all = np.concatenate([df1["rolling_nnse_mean"].values,
                                df2["rolling_nnse_mean"].values])
    nnse_min = np.nanmin(nnse_all)
    nnse_max = np.nanmax(nnse_all)

    # Hull distance: min fixed at 0, shared global max across both models
    ext_all = np.concatenate([df1["distance"].values,
                               df2["distance"].values])
    ext_min = 0.0                  # inside hull always maps to 0
    ext_max = np.nanmax(ext_all)   # shared global max across both models

    print(f"\nGlobal normalization ranges:")
    print(f"  NNSE:      [{nnse_min:.3f}, {nnse_max:.3f}]")
    print(f"  Hull dist: [{ext_min:.3f}, {ext_max:.3f}]")

    df1["nnse_norm"] = normalize_0_1(df1["rolling_nnse_mean"], nnse_min, nnse_max)
    df2["nnse_norm"] = normalize_0_1(df2["rolling_nnse_mean"], nnse_min, nnse_max)
    df1["ext_norm"]  = normalize_0_1(df1["distance"], ext_min, ext_max)
    df2["ext_norm"]  = normalize_0_1(df2["distance"], ext_min, ext_max)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(16, 7))
    ax2 = ax1.twinx()
    LW  = 1.5

    ax1.plot(df1["date"], df1["nnse_norm"],
             color="black",     lw=LW, linestyle="-",
             label=f"{LABEL1} (NNSE)")
    ax2.plot(df1["date"], df1["ext_norm"],
             color="red",       lw=LW, linestyle="--",
             label=f"{LABEL1} (Hull dist)")

    ax1.plot(df2["date"], df2["nnse_norm"],
             color="steelblue", lw=LW, linestyle="-",
             label=f"{LABEL2} (NNSE)")
    ax2.plot(df2["date"], df2["ext_norm"],
             color="orange",    lw=LW, linestyle="--",
             label=f"{LABEL2} (Hull dist)")

    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("Normalized Rolling Avg NNSE [0, 1]",
                   fontsize=12, color="black")
    ax2.set_ylabel("Normalized Hull Extrapolation Distance [0, 1]",
                   fontsize=12, color="red")
    ax1.set_title(
        f"Normalized Rolling Avg NNSE vs Hull Extrapolation Distance  "
        f"({WINDOW} steps × 3h = 1-day window, averaged across basins)",
        fontsize=12
    )

    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=11)

    ax1.tick_params(axis="both", labelsize=11)
    ax2.tick_params(axis="both", labelsize=11)
    ax1.grid(True, alpha=0.3, linestyle=":")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300)
    plt.close()
    print(f"\nPlot saved: {OUTPUT_PATH}")