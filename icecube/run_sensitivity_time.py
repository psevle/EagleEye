#!/usr/bin/env python3
"""
Model-matched time-aware EagleEye sensitivity for IceCube GW230808.

Time is treated as a 14th feature with model-matched scaling:

    t_feature = t_sec / SIG_HALF_WIN

Signal events (injected at t ~ Uniform(-SIG_HALF_WIN, +SIG_HALF_WIN)) always
span ±1 in the time feature.  Background spans ±(510 / SIG_HALF_WIN), which
is large for short-duration models and shrinks toward ±1 for very long ones.

The StandardScaler is fitted on the 13 physics features ONLY.  Time is NOT
passed through StandardScaler — its raw model-matched scale is preserved.
This means the KNN metric naturally weights time more for short-duration
models (tight cluster relative to the spread-out background) and less for
long-duration models (signal and background both span ±1).

Why this works
--------------
For τ = 10 s  : background time spans ±51,  signal spans ±1 → time dominates
For τ = 100 s : background time spans ±5.1, signal spans ±1 → time/physics balanced
For τ = 200 s : background time spans ±2.6, signal spans ±1 → physics dominates

The random 20/80 X/Y split keeps T_null low regardless of τ because background
X and Y events have the same time distribution — no artificial boundary effect.

Usage:
  python run_sensitivity_time.py [NuE|NuMu|NuTau|NuMix] [sig_half_win_seconds]
  e.g.  python run_sensitivity_time.py NuMix 10
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

DROP_META      = ['Run', 'Event', 'SubEvent', 'SubEventStream', 'exists']
DROP_FEATS     = ['CoGw_score', 'ztyhp', 'Nhits']

PHYSICS_FEATURES = ['CoG_score', 'CoGx', 'CoGy', 'CoGz', 'Nchannel', 'Qtot',
                    'cbdt', 'duration', 'frequency', 'sbdt', 'tscore',
                    'xyhp', 'ztxhp']

T_BG           = 510.0    # dataset half-span in seconds

# Model duration parameter: signal injected uniform in ±SIG_HALF_WIN around t=0
# t_feature = t_sec / SIG_HALF_WIN  →  signal spans ±1, background spans ±(T_BG/SIG_HALF_WIN)
SIG_HALF_WIN   = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0

FRAC_Y         = 0.2
K_M            = 50
P_EXT          = 1e-3
N_JOBS         = 8

N_NULL         = 200
N_INJ_TRIALS   = 100
N_INJ_GRID     = [5, 10, 20, 50, 75, 100, 150, 200]
ALPHA          = 0.05
TARGET_POWER   = 0.50

SIGNAL_FLAVOUR = sys.argv[1] if len(sys.argv) > 1 else 'NuE'

# ── Data loading ───────────────────────────────────────────────────────────────

def load_background():
    """Returns (t_sec, phys_raw) — time and unscaled physics separately."""
    df_bif = pd.read_hdf(f'{H5_DIR}/hitspool_gw230808.h5', key='Bifrost_variables')
    df_hs  = pd.read_hdf(f'{H5_DIR}/hitspool_gw230808.h5', key='HSEventHeader')

    t_mjd  = df_hs['time_start_mjd'].values
    t_sec  = (t_mjd - t_mjd.mean()) * 86400.0

    mask   = df_bif['tscore'].values >= TSCORE_CUT
    t_sec  = t_sec[mask]

    df_bif = df_bif[mask].reset_index(drop=True)
    df_bif = df_bif.drop(columns=[c for c in DROP_META + DROP_FEATS
                                   if c in df_bif.columns])
    return t_sec, df_bif[PHYSICS_FEATURES].values


def load_signal():
    """Returns list of physics arrays, one per flavour."""
    flavours = ['NuE', 'NuMu', 'NuTau'] if SIGNAL_FLAVOUR == 'NuMix' else [SIGNAL_FLAVOUR]
    arrays = []
    for fl in flavours:
        df = pd.read_hdf(f'{H5_DIR}/genie_{fl}.h5', key='Bifrost_variables')
        df = df.drop(columns=[c for c in DROP_META + DROP_FEATS if c in df.columns])
        arrays.append(df[PHYSICS_FEATURES].values)
    return arrays


def build_features(t_sec, phys_scaled):
    """Concatenate model-matched time feature with scaled physics."""
    t_feat = (t_sec / SIG_HALF_WIN).reshape(-1, 1)   # signal → ±1, bg → ±(T_BG/τ)
    return np.hstack([t_feat, phys_scaled])


# ── EagleEye helpers ──────────────────────────────────────────────────────────

def get_T(EE_book):
    over = EE_book['Y_OVER_clusters']
    return sum(len(over[i]['Repechaged']) for i in over)


def run_eagleeye(X, Y, stats_null):
    p_actual  = len(Y) / (len(X) + len(Y))
    nearest_p = min(stats_null.keys(), key=lambda k: abs(k - p_actual))
    nearest_q = min(stats_null.keys(), key=lambda k: abs(k - (1 - p_actual)))
    sn = {p_actual: stats_null[nearest_p], (1 - p_actual): stats_null[nearest_q]}
    with capture_output():
        result_dict, _ = EagleEye.Soar(
            X, Y, K_M=K_M, p_ext=P_EXT, n_jobs=N_JOBS,
            stats_null=sn, result_dict_in={}, do_IDE=True)
        clusters = partitioning_function(X, Y, result_dict, p_ext=P_EXT)
        EE_book  = EagleEye.Repechage(X, Y, result_dict, clusters, p_ext=P_EXT)
    return get_T(EE_book)


def random_split(arr, frac_y=FRAC_Y):
    idx = np.random.permutation(len(arr))
    cut = int(len(arr) * frac_y)
    return arr[idx[cut:]], arr[idx[:cut]]   # X, Y


def make_signal_events(sig_arrays, N_inj, phys_scaler):
    """Sample N_inj signal events equally from each flavour."""
    n_fl   = len(sig_arrays)
    counts = [N_inj // n_fl] * n_fl
    for i in range(N_inj % n_fl):
        counts[i] += 1
    phys_parts = [arr[np.random.choice(len(arr), n, replace=True)]
                  for arr, n in zip(sig_arrays, counts)]
    phys_scaled = phys_scaler.transform(np.vstack(phys_parts))

    t_sig  = np.random.uniform(-SIG_HALF_WIN, SIG_HALF_WIN, N_inj)
    t_feat = (t_sig / SIG_HALF_WIN).reshape(-1, 1)   # always ±1
    return np.hstack([t_feat, phys_scaled])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    tau_label = f'τ={2*SIG_HALF_WIN:.0f}s'
    bg_ratio  = T_BG / SIG_HALF_WIN
    print(f"Model-matched EagleEye  |  14D  |  {tau_label}  "
          f"(bg time spans ±{bg_ratio:.1f} × signal)  p_ext={P_EXT}  K_M={K_M}")

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("\nLoading data...")
    t_sec, bg_phys_raw = load_background()
    sig_arrays         = load_signal()

    print(f"  Background (tscore>={TSCORE_CUT}): {len(t_sec):,} events")
    flavour_names = ['NuE','NuMu','NuTau'] if SIGNAL_FLAVOUR=='NuMix' else [SIGNAL_FLAVOUR]
    for fl, arr in zip(flavour_names, sig_arrays):
        print(f"  Signal ({fl}): {len(arr):,} events")

    # ── 2. Scale physics only; time kept at model-matched scale ───────────────
    phys_scaler     = StandardScaler().fit(bg_phys_raw)
    bg_phys_scaled  = phys_scaler.transform(bg_phys_raw)
    bg_scaled       = build_features(t_sec, bg_phys_scaled)

    N_total = len(bg_scaled)
    N_Y     = int(N_total * FRAC_Y)
    N_X     = N_total - N_Y
    print(f"\n  N_total={N_total:,}  N_Y={N_Y:,}  N_X={N_X:,}")
    print(f"  Time feature: signal ±1,  background ±{bg_ratio:.1f}")

    # ── 3. Coin-flip nulls ────────────────────────────────────────────────────
    p_ref = N_Y / (N_Y + N_X)
    print(f"\nPrecomputing coin-flip nulls...")
    all_p = {p_ref}
    for N_inj in N_INJ_GRID:
        all_p.add((N_Y + N_inj) / (N_Y + N_X + N_inj))
    stats_null = {}
    for p_val in sorted(all_p):
        print(f"  p={p_val:.6f} ...", end=' ', flush=True)
        stats_null.update(compute_the_null(p=p_val, K_M=K_M))
        print("done")

    # ── 4. Null trials ────────────────────────────────────────────────────────
    print(f"\nRunning {N_NULL} null trials...")
    T_null = [run_eagleeye(*random_split(bg_scaled), stats_null)
              for _ in tqdm(range(N_NULL), desc='Null')]
    T_null = np.array(T_null)

    print(f"  T_null: mean={T_null.mean():.2f}  std={T_null.std():.2f}  "
          f"max={T_null.max()}  zeros={np.sum(T_null==0)}/{N_NULL}")
    T_crit = np.quantile(T_null, 1 - ALPHA)
    print(f"  T_crit (alpha={ALPHA}): {T_crit:.1f}")

    # ── 5. Injection trials ───────────────────────────────────────────────────
    print(f"\nRunning injection trials over N_inj = {N_INJ_GRID}...")
    T_inj_dict = {}
    for N_inj in N_INJ_GRID:
        T_inj = []
        for _ in tqdm(range(N_INJ_TRIALS), desc=f'  N_inj={N_inj:4d}'):
            X, Y      = random_split(bg_scaled)
            sig       = make_signal_events(sig_arrays, N_inj, phys_scaler)
            Y_inj     = np.vstack([Y, sig])
            T_inj.append(run_eagleeye(X, Y_inj, stats_null))
        T_inj_dict[N_inj] = np.array(T_inj)

    # ── 6. Power curve ────────────────────────────────────────────────────────
    power_values = [np.mean(T_inj_dict[N] > T_crit) for N in N_INJ_GRID]
    try:
        f           = interp1d(power_values, N_INJ_GRID,
                               bounds_error=False, fill_value='extrapolate')
        sensitivity = float(f(TARGET_POWER))
    except Exception:
        sensitivity = np.nan

    print(f"\n{'N_inj':>8}  {'Power':>8}")
    for N, pw in zip(N_INJ_GRID, power_values):
        print(f"{N:>8}  {pw:>8.3f}")
    print(f"\nSensitivity at {int(TARGET_POWER*100)}% power: {sensitivity:.1f} signal events")

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    all_T = np.concatenate([T_null] + list(T_inj_dict.values()))
    bins  = np.arange(0, all_T.max() + 2) - 0.5

    axes[0].hist(T_null, bins=bins, density=True, alpha=0.6,
                 color='steelblue', label='Background only')
    cmap = plt.cm.Oranges(np.linspace(0.4, 0.9, len(N_INJ_GRID)))
    for (N, T_inj), colour in zip(T_inj_dict.items(), cmap):
        axes[0].hist(T_inj, bins=bins, density=True, alpha=0.4,
                     color=colour, label=f'N_inj={N}')
    axes[0].axvline(T_crit, color='red', ls='--',
                    label=f'T_crit={T_crit:.1f}')
    axes[0].set_xlabel('T  (repechaged events)')
    axes[0].set_ylabel('Density')
    axes[0].legend(fontsize=7)
    axes[0].set_title(f'Null vs injected  |  {tau_label}  (bg ±{bg_ratio:.1f}×)')

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
    axes[1].set_title(f'{SIGNAL_FLAVOUR}  |  {tau_label}  |  K_M={K_M}')

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__),
                       f'sensitivity_{SIGNAL_FLAVOUR}_tau{int(2*SIG_HALF_WIN)}s'
                       f'_KM{K_M}_pext{P_EXT:.0e}.pdf')
    plt.savefig(out)
    print(f"\nPlot saved to {out}")


if __name__ == '__main__':
    main()
