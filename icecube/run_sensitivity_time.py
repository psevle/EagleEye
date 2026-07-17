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

# Test statistics computed from each EagleEye run (all from the same run, no
# extra cost). Larger = more signal-like for all of them.
#   T_rep   : total repechaged Y events across clusters (original statistic)
#   N_put   : number of putative flagged test points |Y^+|
#   Ups_max : max anomaly score Upsilon_i over test points (continuous,
#             defined in every trial -> no zero-inflation at the null median)
#   Lam_max : hottest-cluster Lambda_alpha = S_hat/sqrt(B_hat), paper Eq. 9
STATS       = ['T_rep', 'N_put', 'Ups_max', 'Lam_max']
STAT_LABELS = {
    'T_rep':   'T (repechaged count)',
    'N_put':   r'$|Y^+|$ (putative count)',
    'Ups_max': r'max $\Upsilon_i$',
    'Lam_max': r'max$_\alpha\ \Lambda_\alpha$',
}


def compute_stats(result_dict, EE_book, p_ext=P_EXT):
    """Extract all per-trial test statistics from one EagleEye run."""
    over = EE_book['Y_OVER_clusters']

    T_rep   = sum(len(over[i]['Repechaged']) for i in over)
    N_put   = len(result_dict['Y^+'])
    Ups_max = float(np.max(result_dict['Upsilon_i_Y']))

    # Hottest-cluster Lambda_alpha (Eq. 9 of arXiv:2503.23927):
    #   B_hat_alpha = |Y_alpha^inj| * (n_Y - |Yhat^+|) / (n_X - |Xhat^+|)
    #   Lambda_alpha = (|Y_alpha^anom| - B_hat_alpha) / sqrt(B_hat_alpha)
    nY   = len(result_dict['Upsilon_i_Y'])
    nX   = len(result_dict['Upsilon_i_X'])
    Yhat = len(result_dict['Y_Pruned'][p_ext]) if result_dict['Y_Pruned'] else 0
    Xhat = len(result_dict['X_Pruned'][p_ext]) if result_dict['X_Pruned'] else 0
    scale = (nY - Yhat) / (nX - Xhat)

    Lam_max = 0.0
    for i in over:
        S_obs = len(over[i]['Repechaged'])
        if S_obs == 0:
            continue
        # floor the injected count at 1 so B_hat > 0 (keeps Lambda finite and
        # monotone when no injected reference point lands in the cluster)
        B_hat = max(len(over[i]['Background']), 1) * scale
        Lam_max = max(Lam_max, (S_obs - B_hat) / np.sqrt(B_hat))

    return {'T_rep': T_rep, 'N_put': N_put,
            'Ups_max': Ups_max, 'Lam_max': Lam_max}


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
    return compute_stats(result_dict, EE_book)


def random_split(arr, frac_y=FRAC_Y):
    idx = np.random.permutation(len(arr))
    cut = int(len(arr) * frac_y)
    return arr[idx[cut:]], arr[idx[:cut]]   # X, Y


def sensitivity_from_trials(t_null, t_inj_dict, threshold_q, target_power):
    """
    Generic pseudo-experiment sensitivity:
      threshold = threshold_q quantile of the null distribution
      power(N)  = fraction of injection trials with T > threshold
      sensitivity = N_inj at target_power (interpolated)

    Standard IceCube convention: threshold_q=0.5 (median), target_power=0.9.
    """
    thresh = np.quantile(t_null, threshold_q)
    Ns     = sorted(t_inj_dict)
    power  = [np.mean(t_inj_dict[N] > thresh) for N in Ns]
    try:
        f    = interp1d(power, Ns, bounds_error=False, fill_value='extrapolate')
        sens = float(f(target_power))
    except Exception:
        sens = np.nan
    # null exceedance actually achieved at the threshold (ties/discreteness
    # can push it well below 1 - threshold_q, making the test conservative
    # or, for zero-inflated statistics, degenerate)
    null_exc = float(np.mean(t_null > thresh))
    return thresh, power, sens, null_exc


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
    T_null = {s: [] for s in STATS}
    for _ in tqdm(range(N_NULL), desc='Null'):
        trial = run_eagleeye(*random_split(bg_scaled), stats_null)
        for s in STATS:
            T_null[s].append(trial[s])
    T_null = {s: np.array(v) for s, v in T_null.items()}

    for s in STATS:
        v = T_null[s]
        print(f"  {s:>8} null: mean={v.mean():.2f}  std={v.std():.2f}  "
              f"median={np.median(v):.2f}  max={v.max():.2f}  "
              f"zeros={np.sum(v == 0)}/{N_NULL}")

    # ── 5. Injection trials ───────────────────────────────────────────────────
    print(f"\nRunning injection trials over N_inj = {N_INJ_GRID}...")
    T_inj = {s: {} for s in STATS}
    for N_inj in N_INJ_GRID:
        trials = {s: [] for s in STATS}
        for _ in tqdm(range(N_INJ_TRIALS), desc=f'  N_inj={N_inj:4d}'):
            X, Y  = random_split(bg_scaled)
            sig   = make_signal_events(sig_arrays, N_inj, phys_scaler)
            trial = run_eagleeye(X, np.vstack([Y, sig]), stats_null)
            for s in STATS:
                trials[s].append(trial[s])
        for s in STATS:
            T_inj[s][N_inj] = np.array(trials[s])

    # ── 6. Sensitivity per statistic, both conventions ────────────────────────
    # 'standard': threshold = median of null, sensitivity quoted at 90% power
    # 'previous': threshold = (1-ALPHA) null quantile, quoted at TARGET_POWER
    results = {}
    for s in STATS:
        thr_std, pow_std, sens_std, exc_std = sensitivity_from_trials(
            T_null[s], T_inj[s], threshold_q=0.5, target_power=0.9)
        thr_old, pow_old, sens_old, _ = sensitivity_from_trials(
            T_null[s], T_inj[s], threshold_q=1 - ALPHA, target_power=TARGET_POWER)
        results[s] = dict(thr_std=thr_std, pow_std=pow_std, sens_std=sens_std,
                          exc_std=exc_std, thr_old=thr_old, pow_old=pow_old,
                          sens_old=sens_old)

    print(f"\n{'Statistic':>10}  {'med-null thr':>12}  {'P(null>thr)':>11}  "
          f"{'N@90% (std)':>11}  {'N@50% (a=0.05)':>14}")
    for s in STATS:
        r = results[s]
        flag = '  [degenerate null median!]' if r['exc_std'] < 0.25 else ''
        print(f"{s:>10}  {r['thr_std']:>12.2f}  {r['exc_std']:>11.3f}  "
              f"{r['sens_std']:>11.1f}  {r['sens_old']:>14.1f}{flag}")

    # ── 7. Plots: per-statistic distributions + power curves ─────────────────
    fig, axes = plt.subplots(2, len(STATS), figsize=(4.2 * len(STATS), 8))
    cmap = plt.cm.Oranges(np.linspace(0.4, 0.9, len(N_INJ_GRID)))

    for j, s in enumerate(STATS):
        r = results[s]
        ax = axes[0, j]
        all_T = np.concatenate([T_null[s]] + list(T_inj[s].values()))
        if s in ('T_rep', 'N_put'):
            bins = np.arange(0, all_T.max() + 2) - 0.5
        else:
            bins = np.linspace(all_T.min(), all_T.max(), 35)
        ax.hist(T_null[s], bins=bins, density=True, alpha=0.6,
                color='steelblue', label='Background only')
        for (N, vals), colour in zip(T_inj[s].items(), cmap):
            ax.hist(vals, bins=bins, density=True, alpha=0.4, color=colour,
                    label=f'N_inj={N}' if j == 0 else None)
        ax.axvline(r['thr_std'], color='red', ls='--',
                   label=f'median null = {r["thr_std"]:.1f}')
        ax.set_xlabel(STAT_LABELS[s])
        if j == 0:
            ax.set_ylabel('Density')
        ax.legend(fontsize=6)
        ax.set_title(STAT_LABELS[s], fontsize=10)

        ax = axes[1, j]
        ax.plot(N_INJ_GRID, r['pow_std'], 'o-', color='steelblue',
                label='median-null / 90% (standard)')
        ax.plot(N_INJ_GRID, r['pow_old'], 's--', color='grey', alpha=0.7,
                label=f'α={ALPHA} / {int(TARGET_POWER*100)}% (previous)')
        ax.axhline(0.9, color='red', ls=':', lw=1)
        if np.isfinite(r['sens_std']):
            ax.axvline(r['sens_std'], color='green', ls='--',
                       label=f'N@90%: {r["sens_std"]:.1f}')
        ax.set_xlabel('N injected signal events')
        if j == 0:
            ax.set_ylabel('Detection power')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=6)

    fig.suptitle(f'{SIGNAL_FLAVOUR}  |  {tau_label}  |  K_M={K_M}  '
                 f'p_ext={P_EXT:.0e}  |  {N_NULL} null / {N_INJ_TRIALS} inj trials',
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    base = (f'sensitivity_multistat_{SIGNAL_FLAVOUR}_tau{int(2*SIG_HALF_WIN)}s'
            f'_KM{K_M}_pext{P_EXT:.0e}')
    out = os.path.join(os.path.dirname(__file__), base + '.pdf')
    plt.savefig(out)
    print(f"\nPlot saved to {out}")

    # Save raw trial distributions so conventions/statistics can be re-analysed
    # without rerunning the (expensive) pseudo-experiments.
    npz = {f'null_{s}': T_null[s] for s in STATS}
    for s in STATS:
        for N, vals in T_inj[s].items():
            npz[f'inj_{s}_{N}'] = vals
    out_npz = os.path.join(os.path.dirname(__file__), base + '.npz')
    np.savez(out_npz, **npz)
    print(f"Raw trial statistics saved to {out_npz}")


if __name__ == '__main__':
    main()
