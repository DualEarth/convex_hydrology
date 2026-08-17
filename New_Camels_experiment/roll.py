"""
rolling_nnse_reduced_states.py

Two-run version:
  - Run A  : owns basins in PICKLE_A  (BASINS_RUN_A)
  - Run B  : owns basins in PICKLE_B  (BASINS_RUN_B)
  - Each run has its own train / val / test state directories.
  - LedoitWolf covariance is fitted SEPARATELY per run.
  - Raw Mahalanobis distances are computed per basin using its run's covariance.
  - Normalisation (p1 / p99) is derived from ALL 4 selected basins POOLED,
    so both runs share the same scale.
  - One combined metrics CSV supplies NSE for all basins.
"""

import os
import re
import pickle
import shutil
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import torch

from multiprocessing import Pool
from functools import partial
from sklearn.covariance import LedoitWolf

# CONFIG — loaded from config.yaml (section: roll)
_cfg = yaml.safe_load(open(Path(__file__).with_name("config.yaml")))["roll"]

# Run A
RUN_A = dict(
    train_state_dir = _cfg["run_a"]["train_state_dir"],
    val_state_dir   = _cfg["run_a"]["val_state_dir"],
    test_state_dir  = _cfg["run_a"]["test_state_dir"],
    pickle_path     = _cfg["run_a"]["pickle_path"],
    basins          = _cfg["run_a"]["basins"],   # basins that live in this pickle
)

# Run B
RUN_B = dict(
    train_state_dir = _cfg["run_b"]["train_state_dir"],
    val_state_dir   = _cfg["run_b"]["val_state_dir"],
    test_state_dir  = _cfg["run_b"]["test_state_dir"],
    pickle_path     = _cfg["run_b"]["pickle_path"],
    basins          = _cfg["run_b"]["basins"],   # basins that live in this pickle
)

# Shared
SELECTED_BASINS = RUN_A['basins'] + RUN_B['basins']   # all 4

NSE_CSV_PATH  = _cfg["nse_csv_path"]
NSE_BASIN_COL = _cfg["nse_basin_col"]
NSE_SCORE_COL = _cfg["nse_score_col"]

OUTPUT_DIR = _cfg["output_dir"]
WINDOW     = _cfg["window"]
N_CORES    = _cfg["n_cores"]

# File-name patterns (shared across both runs)
TRAIN_PATTERN = r'c_n_reduced_epoch(\d+)\.pt$'
VAL_PATTERN   = r'c_n_reduced_epoch(\d+)_validation\.pt$'
TEST_PATTERN  = r'c_n_reduced_epoch(\d+)_basin_([^\.]+)\.pt$'

# Per-basin colours / labels for plotting
BASIN_COLORS = {
    '11143000': '#E41A1C',
    '13011900': '#377EB8',
    '10348850': '#4DAF4A',
    '12189500': '#FF7F00',
}
BASIN_LABELS = {
    '11143000': 'Basin 11143000 (Run A)',
    '13011900': 'Basin 13011900 (Run A)',
    '10348850': 'Basin 10348850 (Run B)',
    '12189500': 'Basin 12189500 (Run B)',
}

# UTILITIES

def load_nse_from_csv(csv_path, basin_col, score_col):
    print("\nLoading NSE from CSV...")
    df = pd.read_csv(csv_path)
    df[basin_col] = df[basin_col].astype(str).str.strip()
    nse_map = df.groupby(basin_col)[score_col].mean().to_dict()
    print(f"  Loaded NSE for {len(nse_map)} basins.")
    return nse_map


def match_nse(basin_id: str, nse_map: dict) -> float:
    """Zero-padding-tolerant NSE lookup."""
    if basin_id in nse_map:
        return nse_map[basin_id]
    stripped = basin_id.lstrip('0')
    if stripped in nse_map:
        return nse_map[stripped]
    padded = basin_id.zfill(8)
    if padded in nse_map:
        return nse_map[padded]
    for key, val in nse_map.items():
        if key.lstrip('0') == stripped:
            return val
    return np.nan


def rolling_nse(obs: pd.Series, sim: pd.Series, window: int,
                obs_mean: float) -> pd.Series:
    """
    Rolling NSE with denominator anchored to the full test-period observed mean
    so the skill baseline is stable across all windows.
    """
    values = []
    for i in range(len(obs) - window + 1):
        o_win = obs.iloc[i:i + window]
        s_win = sim.iloc[i:i + window]
        if o_win.isna().any() or s_win.isna().any():
            values.append(np.nan)
            continue
        ss_res = ((o_win - s_win) ** 2).sum()
        ss_tot = ((o_win - obs_mean) ** 2).sum()
        if ss_tot < 1e-10 * (obs_mean ** 2 + 1e-12):
            values.append(np.nan)
            continue
        values.append(1.0 - ss_res / ss_tot)
    return pd.Series(values, index=obs.index[window - 1:])


# PICKLE → ROLLING NNSE

def process_single_basin(basin_tuple, window: int):
    basin_id, entry = basin_tuple
    result = []
    try:
        ds = entry['1D']['xr']
        if isinstance(ds, xr.DataArray):
            ds = ds.to_dataset(name='data_var')
        df = ds.to_dataframe().reset_index()
        df['basin_id'] = str(basin_id)

        if 'date' in df.columns:
            df['time'] = pd.to_datetime(df['date'])
        elif 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        else:
            df['time'] = pd.date_range(start='1980-01-01', periods=len(df), freq='D')

        var_bases = set()
        for col in df.columns:
            m = re.match(r"(.+)_obs$", col) or re.match(r"(.+)_sim$", col)
            if m:
                var_bases.add(m.group(1))

        for var in var_bases:
            obs_col, sim_col = f"{var}_obs", f"{var}_sim"
            if obs_col in df.columns and sim_col in df.columns:
                obs_series = df[obs_col]
                obs_mean   = obs_series.dropna().mean()
                r_nse      = rolling_nse(obs_series, df[sim_col], window, obs_mean)
                if not r_nse.empty:
                    result.append(pd.DataFrame({
                        'time':        df['time'].iloc[window - 1: window - 1 + len(r_nse)],
                        'rolling_nse': r_nse.values,
                        'basin_id':    str(basin_id),
                        'variable':    var,
                    }))
    except Exception as e:
        print(f"  [WARN] Failed basin {basin_id}: {e}")
    return pd.concat(result) if result else None


def load_rolling_nnse_from_pickle(pickle_path, basin_ids, window, n_cores):
    """Load rolling NNSE for a specific list of basin_ids from one pickle."""
    print(f"\n  Loading pickle: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    basin_items = [(k, v) for k, v in data.items() if str(k) in basin_ids]
    if not basin_items:
        raise ValueError(f"None of {basin_ids} found in {pickle_path}.")

    worker = partial(process_single_basin, window=window)
    with Pool(processes=n_cores) as pool:
        results = pool.map(worker, basin_items)

    df = pd.concat([r for r in results if r is not None], ignore_index=True)
    raw_nnse       = 1.0 / (2.0 - df['rolling_nse'])
    clipped        = raw_nnse.clip(0.0, 1.0)
    n_low          = (raw_nnse < 0.0).sum()
    n_high         = (raw_nnse > 1.0).sum()
    print(f"    NNSE clipping: {n_low} below 0 → 0,  {n_high} above 1 → 1.")
    df['NNSE'] = clipped
    return df


def load_all_rolling_nnse(window, n_cores):
    """Combine NNSE from both pickles."""
    print("\nComputing rolling NNSE from both pickles...")
    df_a = load_rolling_nnse_from_pickle(
        RUN_A['pickle_path'], RUN_A['basins'], window, n_cores)
    df_b = load_rolling_nnse_from_pickle(
        RUN_B['pickle_path'], RUN_B['basins'], window, n_cores)
    df   = pd.concat([df_a, df_b], ignore_index=True)
    print(f"  Total NNSE rows: {len(df)}  |  basins: {sorted(df['basin_id'].unique())}")
    return df


# REDUCED STATES → MAHALANOBIS

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
    arr = t.reshape(-1, t.shape[-1]).numpy()
    arr = arr[~np.isnan(arr).any(axis=1)]
    print(f"  Loaded {fname}: {arr.shape}")
    return arr


def fit_mahalanobis_for_run(run_cfg, label):
    """Fit LedoitWolf covariance on the train+val states of one run."""
    print(f"\nFitting Mahalanobis covariance for {label} ...")
    train = load_flat(run_cfg['train_state_dir'], TRAIN_PATTERN)
    val   = load_flat(run_cfg['val_state_dir'],   VAL_PATTERN)
    arrays = [x for x in [train, val] if x is not None]
    if not arrays:
        raise RuntimeError(f"No train/val state files found for {label}.")
    ref = np.vstack(arrays)
    print(f"  Reference shape: {ref.shape}")
    cov = LedoitWolf().fit(ref)
    print(f"  LedoitWolf covariance fitted for {label}.")
    return cov


def extract_basin_dates_from_pickle(pickle_path):
    """Return {basin_id: np.ndarray of datetime64} for all basins in a pickle."""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    basin_dates = {}
    for basin_id, entry in data.items():
        try:
            ds = entry['1D']['xr']
            if isinstance(ds, xr.DataArray):
                ds = ds.to_dataset(name='data_var')
            df = ds.to_dataframe().reset_index()
            if 'date' in df.columns:
                dates = pd.to_datetime(df['date']).values
            elif 'time' in df.columns:
                dates = pd.to_datetime(df['time']).values
            else:
                dates = pd.date_range(start='1980-01-01', periods=len(df), freq='D').values
            basin_dates[str(basin_id)] = dates
        except Exception as e:
            print(f"  [WARN] Could not extract dates for basin {basin_id}: {e}")
    return basin_dates


def load_selected_basin_states(run_cfg, cov, basin_dates, label):
    """
    Load test-state .pt files for the basins belonging to one run,
    compute raw Mahalanobis distances, and return per-basin DataFrames.

    Returns
    -------
    list of DataFrames  with columns: basin_id, time, mahal_dist
    """
    print(f"\nLoading test states for {label} basins ...")
    test_dir   = run_cfg['test_state_dir']
    target_ids = set(run_cfg['basins'])

    # find last-epoch file per basin
    basin_files = {}
    for fname in os.listdir(test_dir):
        m = re.search(TEST_PATTERN, fname)
        if not m:
            continue
        epoch, basin_id = int(m.group(1)), m.group(2)
        if basin_id not in target_ids:
            continue
        if basin_id not in basin_files or epoch > basin_files[basin_id][0]:
            basin_files[basin_id] = (epoch, fname)

    VI  = cov.get_precision()
    loc = cov.location_

    frames = []
    for basin_id, (_, fname) in basin_files.items():
        t      = torch.load(os.path.join(test_dir, fname), map_location='cpu')
        states = t.reshape(-1, t.shape[-1]).numpy()

        dates = basin_dates.get(basin_id)
        if dates is None:
            print(f"  [WARN] No pickle dates for basin {basin_id} – skipping.")
            continue

        # trim warmup rows if states longer than dates
        if len(states) > len(dates):
            states = states[len(states) - len(dates):]
        elif len(states) < len(dates):
            print(f"  [WARN] Basin {basin_id}: states ({len(states)}) < "
                  f"dates ({len(dates)}) – skipping.")
            continue

        nan_rows = np.isnan(states).any(axis=1)
        mah      = np.full(len(states), np.nan)
        d        = states[~nan_rows] - loc
        mah[~nan_rows] = np.sqrt((d @ VI * d).sum(axis=1).clip(0))

        frames.append(pd.DataFrame({
            'basin_id':  basin_id,
            'time':      pd.to_datetime(dates).normalize(),
            'mahal_dist': mah,
        }))
        print(f"  {basin_id}: {(~nan_rows).sum()} valid timesteps, "
              f"mean dist = {np.nanmean(mah):.2f}")

    return frames


def compute_joint_normalization(all_frames):
    """
    Pool raw Mahalanobis distances from all 4 selected basins and derive
    a shared p1 / p99 normalisation range.
    """
    df_all = pd.concat(all_frames, ignore_index=True)
    valid  = df_all['mahal_dist'].dropna()

    p01  = np.percentile(valid, 1)
    p99  = np.percentile(valid, 99)
    pmax = valid.max()

    print(f"\n  Joint Mahalanobis distribution (all 4 selected basins, "
          f"{len(valid):,} timesteps):")
    print(f"    p1   = {p01:.2f}")
    print(f"    p50  = {np.percentile(valid, 50):.2f}  (median)")
    print(f"    p99  = {p99:.2f}  ← normalisation ceiling")
    print(f"    max  = {pmax:.2f}")
    if pmax > 3 * p99:
        print(f"    [!] Outlier alert: max > 3× p99.")

    norm_min, norm_max = p01, p99
    if norm_max == norm_min:
        df_all['mahal_norm'] = 0.0
    else:
        df_all['mahal_norm'] = (
            (df_all['mahal_dist'] - norm_min) / (norm_max - norm_min)
        ).clip(0.0, 1.0)

    basin_sums = (
        df_all.groupby('basin_id')['mahal_dist']
        .sum()
        .sort_values(ascending=False)
    )
    print(f"\n  Most novel basin: {basin_sums.index[0]} "
          f"(sum = {basin_sums.iloc[0]:.2f})")

    mahal_lookup = {
        bid: grp[['time', 'mahal_dist', 'mahal_norm']].reset_index(drop=True)
        for bid, grp in df_all.groupby('basin_id')
    }
    return mahal_lookup, basin_sums


# MERGE

def build_merged_df(df_nnse, mahal_lookup):
    print("\nMerging NNSE with jointly-normalised Mahalanobis distances...")
    frames = []
    for basin_id in SELECTED_BASINS:
        if basin_id not in mahal_lookup:
            print(f"  [WARN] Basin {basin_id} not in mahal_lookup – skipping.")
            continue

        df_mah_b = mahal_lookup[basin_id].copy()
        df_mah_b['time'] = pd.to_datetime(df_mah_b['time']).dt.normalize()

        df_b = df_nnse[df_nnse['basin_id'] == basin_id].copy()
        df_b['time'] = pd.to_datetime(df_b['time']).dt.normalize()

        merged = pd.merge(df_b, df_mah_b, on='time', how='inner')

        if merged.empty:
            print(f"  [WARN] Basin {basin_id}: no overlapping dates.")
        else:
            print(f"  Basin {basin_id}: {len(merged)} aligned timesteps "
                  f"({merged['time'].min().date()} → {merged['time'].max().date()})")
        frames.append(merged)

    return pd.concat(frames, ignore_index=True)


# PLOTTING

NNSE_COLOR  = '#E41A1C'   # red  – NNSE curve
MAHAL_COLOR = '#377EB8'   # blue – Mahalanobis curve


def plot_individual(df_merged, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for var in df_merged['variable'].unique():
        for basin_id in sorted(df_merged['basin_id'].unique()):
            sub = df_merged[
                (df_merged['basin_id'] == basin_id) &
                (df_merged['variable'] == var)
            ].copy()
            if sub.empty:
                continue

            full_nse    = sub['full_nse'].iloc[0]
            nse_str     = f"{full_nse:.3f}" if not np.isnan(full_nse) else "N/A"
            basin_label = BASIN_LABELS.get(basin_id, f'Basin {basin_id}')

            fig, ax = plt.subplots(figsize=(16, 8))
            ax.plot(sub['time'], sub['NNSE'],
                    color=NNSE_COLOR,  lw=2.5, linestyle='-',  label='NNSE')
            ax.plot(sub['time'], sub['mahal_norm'],
                    color=MAHAL_COLOR, lw=2.5, linestyle='-',  label='Normalised Mahalanobis Distance')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.tick_params(axis='both', labelsize=13)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate()
            ax.set_title(f'{basin_label}  |  NSE = {nse_str}', fontsize=12)
            ax.legend(fontsize=13, loc='upper right')
            plt.tight_layout()

            safe_var = re.sub(r'[^A-Za-z0-9_.]+', '_', var)
            path = os.path.join(output_dir, f'basin_{basin_id}_{safe_var}.png')
            plt.savefig(path, dpi=300)
            plt.close(fig)
            print(f"  Saved: {path}")


def plot_combined(df_merged, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Each basin gets a distinct linestyle so they remain distinguishable
    # even though colour now encodes metric type (red=NNSE, blue=Mahalanobis).
    LINESTYLES = ['-', '--', '-.', ':']

    for var in df_merged['variable'].unique():
        plt.figure(figsize=(16, 8))
        nse_handles, mah_handles = [], []

        for idx, basin_id in enumerate(sorted(df_merged['basin_id'].unique())):
            sub = df_merged[
                (df_merged['basin_id'] == basin_id) &
                (df_merged['variable'] == var)
            ]
            if sub.empty:
                continue

            ls       = LINESTYLES[idx % len(LINESTYLES)]
            label    = BASIN_LABELS.get(basin_id, basin_id)
            full_nse = sub['full_nse'].iloc[0]
            nse_str  = f"{full_nse:.3f}" if not np.isnan(full_nse) else "N/A"

            plt.plot(sub['time'], sub['NNSE'],
                     color=NNSE_COLOR,  lw=1.5, linestyle=ls, alpha=0.9)
            plt.plot(sub['time'], sub['mahal_norm'],
                     color=MAHAL_COLOR, lw=1.5, linestyle=ls, alpha=0.9)

            nse_handles.append(mlines.Line2D(
                [], [], color=NNSE_COLOR, linestyle=ls,
                label=f'{label} (NNSE | NSE={nse_str})'))
            mah_handles.append(mlines.Line2D(
                [], [], color=MAHAL_COLOR, linestyle=ls,
                label=f'{label} (Mahalanobis)'))

        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.ylim(0, 1)
        plt.grid(True, linestyle=':')
        plt.tick_params(axis='both', labelsize=13)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()
        plt.legend(handles=nse_handles + mah_handles,
                   title='Trends by Basin', loc='upper right',
                   fontsize=13, title_fontsize=12)
        plt.tight_layout()

        safe_var = re.sub(r'[^A-Za-z0-9_.]+', '_', var)
        path = os.path.join(output_dir, f'combined_{safe_var}.png')
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"  Saved: {path}")



# Three basins shown in the 3-panel comparison plot
THREE_PANEL_BASINS = ['11143000', '12189500', '13011900']

def plot_three_panel(df_merged, output_dir):
    """
    Vertical 3-panel figure comparing NNSE (red) and normalised Mahalanobis
    distance (blue) for THREE_PANEL_BASINS.
 
    Layout rules:
      - One shared legend in the top panel only.
      - No panel titles; basin label + NSE shown as a text annotation instead.
      - X-axis tick labels (years only) shown on the bottom panel only,
        rotated 45 degrees.
      - All upper panels have tick marks but no tick labels on x-axis.
      - Figure width and font sizes match the individual/combined plots.
    """
    os.makedirs(output_dir, exist_ok=True)
 
    panel_h = 6.5 * 0.45   # same aspect as individual plots
    fig_w   = 6.5
    n       = len(THREE_PANEL_BASINS)
 
    for var in df_merged["variable"].unique():
 
        fig, axes = plt.subplots(n, 1, figsize=(6.5, 0.45 * 6.5 * n), sharex=False)
 
        legend_handles = [
            mlines.Line2D([], [], color=NNSE_COLOR,  lw=1.1, linestyle="-",
                          label="NNSE"),
            mlines.Line2D([], [], color=MAHAL_COLOR, lw=1.1, linestyle="-",
                          label="Normalised Mahalanobis Distance"),
        ]
 
        # collect global time extent so all panels share the same x range
        all_times  = []
        panel_data = []
        for basin_id in THREE_PANEL_BASINS:
            sub = df_merged[
                (df_merged["basin_id"] == basin_id) &
                (df_merged["variable"] == var)
            ].copy()
            panel_data.append((basin_id, sub))
            if not sub.empty:
                all_times.extend([sub["time"].min(), sub["time"].max()])
 
        x_min = min(all_times) if all_times else None
        x_max = max(all_times) if all_times else None
 
        for row, (basin_id, sub) in enumerate(panel_data):
            ax = axes[row]
 
            if not sub.empty:
                ax.plot(sub["time"], sub["NNSE"],
                        color=NNSE_COLOR,  lw=1.1, linestyle="-")
                ax.plot(sub["time"], sub["mahal_norm"],
                        color=MAHAL_COLOR, lw=1.1, linestyle="-")
 
            ax.set_ylim(0, 1)
            ax.set_yticks(np.arange(0, 1.01, 0.25))
            ax.set_ylabel("Value", fontsize=10)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.tick_params(axis="y", labelsize=10)
 
            # basin label + NSE as text inside panel (top-left corner)
            basin_label = BASIN_LABELS.get(basin_id, f"Basin {basin_id}")

            # a/b/c panel label (top-left, outside the text annotation)
            ax.text(-0.07, 1.02, f"({chr(ord('a') + row)})",
                    transform=ax.transAxes,
                    fontsize=11, fontweight="bold", va="bottom", ha="left")
 
            if x_min is not None:
                ax.set_xlim(x_min, x_max)
 
            # year ticks on every panel; labels only on the bottom one
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
 
            if row < n - 1:
                ax.tick_params(axis="x", labelbottom=False, length=4)
            else:
                ax.tick_params(axis="x", labelsize=10, rotation=45)
                for tick in ax.get_xticklabels():
                    tick.set_ha("right")
                ax.set_xlabel("Year", fontsize=10)
 
            # legend only on the top panel
            if row == 0:
                ax.legend(handles=legend_handles, fontsize=10,
                          loc="upper right", framealpha=0.85)
 
        plt.tight_layout(h_pad=0.15)
 
        safe_var = re.sub(r"[^A-Za-z0-9_.]+", "_", var)
        path = os.path.join(output_dir, f"three_panel_{safe_var}.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")
# MAIN

def main():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        print(f"Cleared old output directory: {OUTPUT_DIR}")

    # 1. NSE lookup (single combined CSV)
    nse_map = load_nse_from_csv(NSE_CSV_PATH, NSE_BASIN_COL, NSE_SCORE_COL)

    # 2. Fit one covariance per run
    cov_a = fit_mahalanobis_for_run(RUN_A, label='Run A')
    cov_b = fit_mahalanobis_for_run(RUN_B, label='Run B')

    # 3. Extract dates from each pickle (needed to align state tensors)
    dates_a = extract_basin_dates_from_pickle(RUN_A['pickle_path'])
    dates_b = extract_basin_dates_from_pickle(RUN_B['pickle_path'])

    # 4. Load test states + compute raw Mahalanobis per run
    frames_a = load_selected_basin_states(RUN_A, cov_a, dates_a, label='Run A')
    frames_b = load_selected_basin_states(RUN_B, cov_b, dates_b, label='Run B')

    if not frames_a and not frames_b:
        print("[ERROR] No test basin states loaded for either run.")
        return

    # 5. Joint normalisation across all 4 selected basins
    mahal_lookup, basin_sums = compute_joint_normalization(frames_a + frames_b)

    print(f"\n{'='*60}")
    print(f"  Most novel basin: {basin_sums.index[0]}  "
          f"(sum raw dist = {basin_sums.iloc[0]:.2f})")
    print(f"{'='*60}\n")

    # 6. Rolling NNSE from both pickles
    df_nnse = load_all_rolling_nnse(WINDOW, N_CORES)

    # 7. Merge
    df_merged = build_merged_df(df_nnse, mahal_lookup)

    # 8. Attach full-period NSE from CSV
    df_merged['full_nse'] = df_merged['basin_id'].apply(
        lambda bid: match_nse(bid, nse_map))
    missing = df_merged[df_merged['full_nse'].isna()]['basin_id'].unique()
    if len(missing):
        print(f"  [WARN] NSE not found for: {missing.tolist()}")

    # 9. Save outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_out = os.path.join(OUTPUT_DIR, 'merged_nnse_mahalanobis.csv')
    df_merged.to_csv(csv_out, index=False)
    print(f"\nMerged data saved: {csv_out}")

    rank_out = os.path.join(OUTPUT_DIR, 'basin_novelty_ranking.csv')
    basin_sums.reset_index().rename(
        columns={'mahal_dist': 'sum_mahal_dist'}
    ).to_csv(rank_out, index=False)
    print(f"Basin novelty ranking saved: {rank_out}")

    # 10. Plots
    print("\nPlotting individual basin plots...")
    plot_individual(df_merged, os.path.join(OUTPUT_DIR, 'individual'))

    print("\nPlotting combined basin plots...")
    plot_combined(df_merged, os.path.join(OUTPUT_DIR, 'combined'))

    print("\nPlotting three-panel comparison plot...")
    plot_three_panel(df_merged, os.path.join(OUTPUT_DIR, 'three_panel'))

    print("\nDone.")


if __name__ == "__main__":
    main()
