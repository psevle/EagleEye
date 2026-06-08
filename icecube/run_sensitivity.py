#!/usr/bin/env python3
"""
EagleEye sensitivity analysis for IceCube GW230808 follow-up.

Pipeline:
  1. Load hitspool background (tscore >= 0.55) + signal simulation
  2. Drop redundant features, apply StandardScaler fitted on background
  3. Null trials: random 20/80 splits of background through EagleEye
  4. Injection trials: same splits + N_inj simulated signal events appended to Y
  5. Power curve -> sensitivity at TARGET_POWER detection efficiency

Run:
  python run_sensitivity.py [NuE|NuMu|NuTau]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eagleeye'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from IPython.utils.capture import capture_output

import EagleEye
from utils_EE import compute_the_null, partitioning_function

# ── Configuration ─────────────────────────────────────────────────────────────

H5_DIR         = os.path.join(os.path.dirname(__file__), 'h5files')
TSCORE_CUT     = 0.55

# Metadata columns to drop before any processing
DROP_META      = ['Run', 'Event', 'SubEvent', 'SubEventStream', 'exists']

# Redundant features identified from correlation analysis (|r| > 0.7 with another feature)
DROP_FEATS     = ['CoGw_score', 'ztyhp', 'Nhits']

FRAC_Y         = 0.2      # fraction of background in on-region (Y)
K_M            = 50       # EagleEye max neighbourhood size
P_EXT          = 1e-3     # cluster p-value threshold
N_JOBS         = 8

N_NULL         = 200      # background-only trials for null distribution
N_INJ_TRIALS   = 100      # trials per injection strength
N_INJ_GRID     = [5, 10, 20, 50, 100, 200]   # signal events injected per trial
ALPHA          = 0.05     # false-positive rate
TARGET_POWER   = 0.50     # detection efficiency at which sensitivity is quoted

SIGNAL_FLAVOUR = sys.argv[1] if len(sys.argv) > 1 else 'NuE'

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_features(path, tscore_min=None):
    df = pd.read_hdf(path, key='Bifrost_variables')
    df = df.drop(columns=[c for c in DROP_META  if c in df.columns])
    df = df.drop(columns=[c for c in DROP_FEATS if c in df.columns])
    if tscore_min is not None:
        df = df[df['tscore'] >= tscore_min].reset_index(drop=True)
    return df


def get_T(EE_book):
    """Total repechaged events across all Y-overdensity clusters."""
    over = EE_book['Y_OVER_clusters']
    return sum(len(over[i]['Repechaged']) for i in over)


def run_eagleeye(X, Y, stats_null):
    with capture_output():
        result_dict, _ = EagleEye.Soar(
            X, Y, K_M=K_M, p_ext=P_EXT, n_jobs=N_JOBS,
            stats_null=stats_null, result_dict_in={}, do_IDE=True)
        clusters = partitioning_function(X, Y, result_dict, p_ext=P_EXT)
        EE_book  = EagleEye.Repechage(X, Y, result_dict, clusters, p_ext=P_EXT)
    return get_T(EE_book)


def random_split(arr, frac_y=FRAC_Y):
    """Random 20/80 split into Y (on-region) and X (off-region)."""
    idx  = np.random.permutation(len(arr))
    cut  = int(len(arr) * frac_y)
    return arr[idx[cut:]], arr[idx[:cut]]   # X, Y


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    assert SIGNAL_FLAVOUR in ('NuE', 'NuMu', 'NuTau', 'NuMix'), \
        "Signal flavour must be one of: NuE, NuMu, NuTau, NuMix"

    # ── 1. Load and preprocess ────────────────────────────────────────────────
    print("Loading data...")
    df_bg  = load_features(f'{H5_DIR}/hitspool_gw230808.h5', tscore_min=TSCORE_CUT)
    flavour_names = ['NuE', 'NuMu', 'NuTau'] if SIGNAL_FLAVOUR == 'NuMix' else [SIGNAL_FLAVOUR]
    sig_dfs = [load_features(f'{H5_DIR}/genie_{fl}.h5') for fl in flavour_names]

    print(f"  Background (tscore >= {TSCORE_CUT}): {len(df_bg):,} events")
    for fl, df in zip(flavour_names, sig_dfs):
        print(f"  Signal ({fl}): {len(df):,} events")
    print(f"  Features ({len(df_bg.columns)}): {df_bg.columns.tolist()}")

    # Fit scaler on background only
    scaler   = StandardScaler().fit(df_bg.values)
    bg_arr   = scaler.transform(df_bg.values).astype(np.float64)
    sig_arrs = [scaler.transform(df.values).astype(np.float64) for df in sig_dfs]

    N_total = len(bg_arr)
    N_Y     = int(N_total * FRAC_Y)
    N_X     = N_total - N_Y
    p       = FRAC_Y

    print(f"\n  N_total={N_total:,}  N_Y={N_Y:,}  N_X={N_X:,}")
    print(f"  K_M={K_M}  p_ext={P_EXT}  features={bg_arr.shape[1]}")

    # ── 2. Precompute null statistics for all needed p values ─────────────────
    # When N_inj signal events are added to Y, p = n2/(n1+n2) shifts slightly.
    # Soar() looks up stats_null[p] exactly, so we must precompute each one.
    print(f"\nPrecomputing coin-flip nulls for all injection p values...")
    stats_null = {}
    all_p = {N_Y / (N_Y + N_X)}   # p for null trials
    for N_inj in N_INJ_GRID:
        all_p.add((N_Y + N_inj) / (N_Y + N_X + N_inj))
    for p_val in sorted(all_p):
        print(f"  p={p_val:.6f} ...", end=' ', flush=True)
        stats_null.update(compute_the_null(p=p_val, K_M=K_M))
        print("done")

    # ── 3. Null trials ────────────────────────────────────────────────────────
    print(f"\nRunning {N_NULL} background-only null trials...")
    T_null = []
    for _ in tqdm(range(N_NULL), desc='Null'):
        X, Y = random_split(bg_arr)
        T_null.append(run_eagleeye(X, Y, stats_null))
    T_null = np.array(T_null)

    print(f"  T_null: mean={T_null.mean():.2f}  std={T_null.std():.2f}  "
          f"max={T_null.max()}  zeros={np.sum(T_null == 0)}/{N_NULL}")

    T_crit = np.quantile(T_null, 1 - ALPHA)
    print(f"  T_crit (alpha={ALPHA}): {T_crit:.1f}")

    # ── 4. Injection trials ───────────────────────────────────────────────────
    print(f"\nRunning injection trials over N_inj = {N_INJ_GRID}...")
    T_inj_dict = {}
    for N_inj in N_INJ_GRID:
        T_inj = []
        for _ in tqdm(range(N_INJ_TRIALS), desc=f'  N_inj={N_inj:4d}'):
            X, Y     = random_split(bg_arr)
            n_fl   = len(sig_arrs)
            counts = [N_inj // n_fl] * n_fl
            for i in range(N_inj % n_fl): counts[i] += 1
            sig_samp = np.vstack([arr[np.random.choice(len(arr), n, replace=True)]
                                  for arr, n in zip(sig_arrs, counts)])
            Y_inj    = np.vstack([Y, sig_samp])
            T_inj.append(run_eagleeye(X, Y_inj, stats_null))
        T_inj_dict[N_inj] = np.array(T_inj)

    # ── 5. Power curve and sensitivity ───────────────────────────────────────
    power_values = [np.mean(T_inj_dict[N] > T_crit) for N in N_INJ_GRID]

    try:
        f           = interp1d(power_values, N_INJ_GRID,
                               bounds_error=False, fill_value='extrapolate')
        sensitivity = float(f(TARGET_POWER))
    except Exception:
        sensitivity = np.nan

    print(f"\n{'N_inj':>8}  {'Power':>8}")
    for N, pw in zip(N_INJ_GRID, power_values):
        marker = ' <--' if abs(pw - TARGET_POWER) == min(abs(p - TARGET_POWER) for p in power_values) else ''
        print(f"{N:>8}  {pw:>8.3f}{marker}")
    print(f"\nSensitivity at {int(TARGET_POWER*100)}% power: {sensitivity:.1f} signal events")

    # ── 6. Plots ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    all_T  = np.concatenate([T_null] + list(T_inj_dict.values()))
    bins   = np.arange(0, all_T.max() + 2) - 0.5

    axes[0].hist(T_null, bins=bins, density=True, alpha=0.6,
                 color='steelblue', label='Background only')
    cmap = plt.cm.Oranges(np.linspace(0.4, 0.9, len(N_INJ_GRID)))
    for (N, T_inj), colour in zip(T_inj_dict.items(), cmap):
        axes[0].hist(T_inj, bins=bins, density=True, alpha=0.4,
                     color=colour, label=f'N_inj={N}')
    axes[0].axvline(T_crit, color='red', ls='--',
                    label=f'T_crit (α={ALPHA}) = {T_crit:.1f}')
    axes[0].set_xlabel('T  (repechaged events)')
    axes[0].set_ylabel('Density')
    axes[0].legend(fontsize=7)
    axes[0].set_title('Null vs injected distributions')

    axes[1].plot(N_INJ_GRID, power_values, 'o-', color='steelblue')
    axes[1].axhline(TARGET_POWER, color='red', ls='--',
                    label=f'{int(TARGET_POWER*100)}% power')
    if np.isfinite(sensitivity):
        axes[1].axvline(sensitivity, color='green', ls='--',
                        label=f'Sensitivity: {sensitivity:.1f} events')
    axes[1].set_xlabel('N injected signal events')
    axes[1].set_ylabel('Detection power')
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].set_title(f'Power curve  —  {SIGNAL_FLAVOUR}  '
                      f'(K_M={K_M}, {bg_arr.shape[1]}D)')

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__),
               f'sensitivity_{SIGNAL_FLAVOUR}_KM{K_M}'
               f'_Nnull{N_NULL}_Ninj{N_INJ_TRIALS}.pdf')
    plt.savefig(out)
    print(f"\nPlot saved to {out}")


if __name__ == '__main__':
    main()
