#!/usr/bin/env python3
"""
Window-based (on/off) EagleEye sensitivity for IceCube GW230808.

This is the honest on/off geometry: the test sample Y is the set of events
inside a time window of width W, and the reference X is everything outside it.
The window duration therefore controls the size ratio N_Y / N_X directly, which
is exactly the coin-flip baseline p_hat = N_Y / (N_X + N_Y) of EagleEye's
Bernoulli null.  Shrinking the window shrinks p_hat, so a fixed number of
injected signal events becomes a progressively more improbable local clump of
Y-labels — sensitivity (in events) should improve for shorter windows.

Design choices (the four points discussed):

  1. NO time feature.  With a time-based on/off split, X and Y already occupy
     disjoint time ranges; adding time as a coordinate would globally separate
     the two clouds (tripping EagleEye's Condition_3 guard).  We use the 13
     physics features only — the time information is carried entirely by the
     split.

  2. K_M floor.  KSTAR_RANGE = range(20, K_M) in the library, so K_M must be
     > 21, and the paper recommends K_M <= 0.05 * min(N_X, N_Y) → N_Y >~ 20*K_M.
     Validity is never at risk (sensitivity is calibrated from real-data null
     trials), but interpretability degrades for very short windows; we warn.

  3. Per-trial p_hat.  N_Y fluctuates window-to-window, so the exact coin-flip
     null p shifts trial to trial.  We precompute the null on a grid of p values
     bracketing the observed range and let run_eagleeye() pick the nearest.

  4. Counting baseline.  T = N_on (raw event count in the window passing a hard
     tscore cut) is precisely the standard on/off counting analysis.  It is
     computed from the SAME pseudo-experiments as the EagleEye statistics, so
     all sensitivities are directly comparable.

tscore handling:
  EagleEye sees the full, uncut point cloud (--ee-tscore-cut, default 0.0).
  The counting baseline applies its own hard cut (--count-tscore-cut, default
  0.55) to approach a background-free regime, mimicking the standard analysis.
  Signal simulations are already cut at tscore >= 0.55, so injected signal
  events that pass the counting cut are counted consistently.

Usage:
  python run_sensitivity_window.py [NuE|NuMu|NuTau|NuMix] --window 50
  python run_sensitivity_window.py NuMix --window 20 --max-seconds 300   # quick
"""

import sys
import os
import argparse

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

# ── Fixed configuration ─────────────────────────────────────────────────────────

H5_DIR     = os.path.join(os.path.dirname(__file__), 'h5files')
DROP_META  = ['Run', 'Event', 'SubEvent', 'SubEventStream', 'exists']
DROP_FEATS = ['CoGw_score', 'ztyhp', 'Nhits']

PHYSICS_FEATURES = ['CoG_score', 'CoGx', 'CoGy', 'CoGz', 'Nchannel', 'Qtot',
                    'cbdt', 'duration', 'frequency', 'sbdt', 'tscore',
                    'xyhp', 'ztxhp']
TSCORE_IDX = PHYSICS_FEATURES.index('tscore')

# Test statistics, all oriented so larger = more signal-like.
#   N_on    : raw count of in-window events passing the counting cut
#             (the standard on/off counting analysis baseline)
#   T_rep   : total repechaged Y events across clusters (EagleEye, original)
#   N_put   : number of putative flagged test points |Y^+|
#   Ups_max : max anomaly score Upsilon_i over test points (continuous)
#   Lam_max : hottest-cluster Lambda_alpha = S_hat/sqrt(B_hat), paper Eq. 9
STATS       = ['N_on', 'T_rep', 'N_put', 'Ups_max', 'Lam_max']
STAT_LABELS = {
    'N_on':    r'$N_{\rm on}$ (counting baseline)',
    'T_rep':   'T (repechaged count)',
    'N_put':   r'$|Y^+|$ (putative count)',
    'Ups_max': r'max $\Upsilon_i$',
    'Lam_max': r'max$_\alpha\ \Lambda_\alpha$',
}
EE_STATS = ['T_rep', 'N_put', 'Ups_max', 'Lam_max']   # statistics from EagleEye


# ── Data loading ────────────────────────────────────────────────────────────────

def load_background(ee_tscore_cut, max_seconds):
    """
    Returns (t_sec, phys_raw, tscore) for the full (uncut) hitspool sample,
    optionally restricted to events with tscore >= ee_tscore_cut and to the
    first `max_seconds` of livetime (rate-preserving span restriction).
    """
    df_bif = pd.read_hdf(f'{H5_DIR}/hitspool_gw230808.h5', key='Bifrost_variables')
    df_hs  = pd.read_hdf(f'{H5_DIR}/hitspool_gw230808.h5', key='HSEventHeader')

    t_mjd = df_hs['time_start_mjd'].values
    t_sec = (t_mjd - t_mjd.min()) * 86400.0

    df_bif = df_bif.drop(columns=[c for c in DROP_META + DROP_FEATS
                                   if c in df_bif.columns])
    tscore = df_bif['tscore'].values

    mask = tscore >= ee_tscore_cut
    if max_seconds is not None:
        mask &= (t_sec <= t_sec.min() + max_seconds)

    t_sec  = t_sec[mask]
    phys   = df_bif[PHYSICS_FEATURES].values[mask]
    tscore = tscore[mask]

    order  = np.argsort(t_sec)
    return t_sec[order], phys[order], tscore[order]


def load_signal(flavour):
    """Returns list of physics arrays (one per flavour), tscore already >= 0.55."""
    flavours = ['NuE', 'NuMu', 'NuTau'] if flavour == 'NuMix' else [flavour]
    arrays = []
    for fl in flavours:
        df = pd.read_hdf(f'{H5_DIR}/genie_{fl}.h5', key='Bifrost_variables')
        df = df.drop(columns=[c for c in DROP_META + DROP_FEATS if c in df.columns])
        arrays.append(df[PHYSICS_FEATURES].values)
    return arrays


# ── EagleEye statistics ─────────────────────────────────────────────────────────

def compute_ee_stats(result_dict, EE_book, p_ext):
    """Extract the four EagleEye test statistics from one Soar/Repechage run."""
    over = EE_book['Y_OVER_clusters']

    T_rep   = sum(len(over[i]['Repechaged']) for i in over)
    N_put   = len(result_dict['Y^+'])
    Ups_max = float(np.max(result_dict['Upsilon_i_Y']))

    # Hottest-cluster Lambda_alpha (Eq. 9 of arXiv:2503.23927):
    #   B_hat_alpha = |background_alpha| * (n_Y - |Yhat^+|) / (n_X - |Xhat^+|)
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
        # floor the background count at 1 so B_hat > 0 (keeps Lambda finite and
        # monotone when no injected reference point lands in the cluster)
        B_hat = max(len(over[i]['Background']), 1) * scale
        Lam_max = max(Lam_max, (S_obs - B_hat) / np.sqrt(B_hat))

    return {'T_rep': T_rep, 'N_put': N_put,
            'Ups_max': Ups_max, 'Lam_max': Lam_max}


def run_eagleeye(X, Y, stats_null, K_M, p_ext, n_jobs):
    """Run Soar → partitioning → Repechage with the nearest precomputed null."""
    p_actual  = len(Y) / (len(X) + len(Y))
    nearest_p = min(stats_null.keys(), key=lambda k: abs(k - p_actual))
    nearest_q = min(stats_null.keys(), key=lambda k: abs(k - (1 - p_actual)))
    sn = {p_actual: stats_null[nearest_p], (1 - p_actual): stats_null[nearest_q]}
    with capture_output():
        result_dict, _ = EagleEye.Soar(
            X, Y, K_M=K_M, p_ext=p_ext, n_jobs=n_jobs,
            stats_null=sn, result_dict_in={}, do_IDE=True)
        clusters = partitioning_function(X, Y, result_dict, p_ext=p_ext)
        EE_book  = EagleEye.Repechage(X, Y, result_dict, clusters, p_ext=p_ext)
    return compute_ee_stats(result_dict, EE_book, p_ext)


# ── On/off windowing ─────────────────────────────────────────────────────────────

def draw_window(t_sec, window, K_M, rng):
    """
    Pick a random window of width `window` seconds and return boolean masks
    (in_window, out_window).  Retries until both sides have >= K_M events.
    """
    t_lo, t_hi = t_sec.min(), t_sec.max()
    for _ in range(1000):
        t0 = rng.uniform(t_lo, t_hi - window)
        on = (t_sec >= t0) & (t_sec < t0 + window)
        n_on = on.sum()
        if n_on >= K_M and (len(t_sec) - n_on) >= K_M:
            return on, ~on
    raise RuntimeError(f"Could not place a window of {window}s with >= {K_M} "
                       f"events on each side; try a larger window or K_M.")


def make_signal(sig_arrays, N_inj, phys_scaler, rng):
    """
    Sample N_inj signal events equally across flavours.
    Returns (scaled_features, tscore_values) so the counting cut can be applied.
    """
    n_fl   = len(sig_arrays)
    counts = [N_inj // n_fl] * n_fl
    for i in range(N_inj % n_fl):
        counts[i] += 1
    parts = [arr[rng.integers(0, len(arr), n)] for arr, n in zip(sig_arrays, counts)]
    raw   = np.vstack(parts)
    return phys_scaler.transform(raw), raw[:, TSCORE_IDX]


# ── Sensitivity ──────────────────────────────────────────────────────────────────

def sensitivity_from_trials(t_null, t_inj_dict, threshold_q, target_power):
    """
    threshold = threshold_q quantile of null; power(N) = P(T_inj > threshold);
    sensitivity = N_inj at target_power (interpolated on the monotonic part).
    Standard on/off convention: threshold_q=0.5 (median), target_power=0.9.
    """
    thresh = np.quantile(t_null, threshold_q)
    Ns     = sorted(t_inj_dict)
    power  = [np.mean(t_inj_dict[N] > thresh) for N in Ns]

    sens = np.nan
    p_arr = np.asarray(power)
    if np.any(p_arr >= target_power) and np.any(p_arr < target_power):
        # interpolate on the cumulative-max power to guard against non-monotone
        # wiggles from finite trial statistics
        p_mono = np.maximum.accumulate(p_arr)
        try:
            f = interp1d(p_mono, Ns, bounds_error=False, fill_value='extrapolate')
            sens = float(f(target_power))
        except Exception:
            sens = np.nan
    elif np.all(p_arr >= target_power):
        sens = float(min(Ns))          # already saturated at the smallest grid point
    null_exc = float(np.mean(t_null > thresh))
    return thresh, power, sens, null_exc


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('flavour', nargs='?', default='NuMix',
                    choices=['NuE', 'NuMu', 'NuTau', 'NuMix'])
    ap.add_argument('--window', type=float, default=50.0,
                    help='on-region window width in seconds (default 50)')
    ap.add_argument('--ee-tscore-cut', type=float, default=0.0,
                    help='tscore cut for the EagleEye cloud (default 0 = uncut)')
    ap.add_argument('--count-tscore-cut', type=float, default=0.55,
                    help='hard tscore cut for the N_on counting baseline (default 0.55)')
    ap.add_argument('--K_M', type=int, default=50)
    ap.add_argument('--p_ext', type=float, default=1e-3)
    ap.add_argument('--n-jobs', type=int, default=8)
    ap.add_argument('--n-null', type=int, default=200)
    ap.add_argument('--n-inj-trials', type=int, default=100)
    ap.add_argument('--n-inj-grid', type=str, default='5,10,20,50,75,100,150,200')
    ap.add_argument('--max-seconds', type=float, default=None,
                    help='restrict to the first N seconds of livetime (speed/testing)')
    ap.add_argument('--alpha', type=float, default=0.05,
                    help='previous-convention false-positive rate (default 0.05)')
    ap.add_argument('--target-power', type=float, default=0.5,
                    help='previous-convention target power (default 0.5)')
    ap.add_argument('--seed', type=int, default=0)
    return ap.parse_args()


def main():
    a = parse_args()
    rng = np.random.default_rng(a.seed)
    N_inj_grid = [int(x) for x in a.n_inj_grid.split(',')]

    print(f"Window on/off EagleEye  |  {a.flavour}  |  W={a.window:.0f}s  "
          f"|  K_M={a.K_M}  p_ext={a.p_ext:.0e}")

    # ── 1. Load (uncut for EagleEye) ──────────────────────────────────────────
    print("\nLoading data...")
    t_sec, bg_phys, bg_tscore = load_background(a.ee_tscore_cut, a.max_seconds)
    sig_arrays                = load_signal(a.flavour)

    span = t_sec.max() - t_sec.min()
    rate = len(t_sec) / span
    print(f"  EagleEye cloud (tscore>={a.ee_tscore_cut}): {len(t_sec):,} events "
          f"over {span:.0f}s  ({rate:.1f} Hz)")
    print(f"  Counting baseline cut: tscore>={a.count_tscore_cut} "
          f"({np.mean(bg_tscore >= a.count_tscore_cut)*100:.1f}% of cloud)")
    flavour_names = ['NuE', 'NuMu', 'NuTau'] if a.flavour == 'NuMix' else [a.flavour]
    for fl, arr in zip(flavour_names, sig_arrays):
        print(f"  Signal ({fl}): {len(arr):,} events")

    # ── 2. Scale physics on the background cloud ──────────────────────────────
    phys_scaler = StandardScaler().fit(bg_phys)
    bg_scaled   = phys_scaler.transform(bg_phys)

    # ── 3. Characterise the window: N_Y distribution → p grid ────────────────
    print(f"\n[W={a.window:.0f}s] Phase 1/4: characterising window placement "
          f"(200 draws)...", flush=True)
    n_on_samples = []
    for _ in tqdm(range(200), desc=f'[W={a.window:.0f}s] window draws',
                  mininterval=1.0):
        on, _ = draw_window(t_sec, a.window, a.K_M, rng)
        n_on_samples.append(on.sum())
    n_on_samples = np.array(n_on_samples)
    N_total = len(t_sec)
    p_lo = n_on_samples.min() / N_total
    p_hi = (n_on_samples.max() + max(N_inj_grid)) / (N_total + max(N_inj_grid))
    print(f"\n  Window N_Y: mean={n_on_samples.mean():.0f} "
          f"[{n_on_samples.min()}–{n_on_samples.max()}]  "
          f"p_hat ≈ {n_on_samples.mean()/N_total:.4f}")
    if n_on_samples.min() < 20 * a.K_M:
        print(f"  ! warning: min N_Y={n_on_samples.min()} < 20*K_M={20*a.K_M}; "
              f"K_M={a.K_M} exceeds the paper's 0.05*min(N_X,N_Y) guidance "
              f"(scores still calibrated by null trials, but treat as exploratory)")

    # ── Precompute coin-flip nulls on a p grid bracketing the observed range ──
    print(f"\n[W={a.window:.0f}s] Phase 2/4: precomputing coin-flip nulls "
          f"(6 p-values, 1e6 sequences each)...", flush=True)
    p_grid = np.linspace(p_lo, p_hi, 6)
    stats_null = {}
    for p_val in tqdm(p_grid, desc=f'[W={a.window:.0f}s] null precompute',
                      mininterval=1.0):
        stats_null.update(compute_the_null(p=float(p_val), K_M=a.K_M))

    # ── 4. Null trials ────────────────────────────────────────────────────────
    print(f"\n[W={a.window:.0f}s] Phase 3/4: {a.n_null} background-only null "
          f"trials...", flush=True)
    T_null = {s: [] for s in STATS}
    pbar = tqdm(range(a.n_null), desc=f'[W={a.window:.0f}s] null', mininterval=1.0)
    for _ in pbar:
        on, off = draw_window(t_sec, a.window, a.K_M, rng)
        Y, X    = bg_scaled[on], bg_scaled[off]
        ee = run_eagleeye(X, Y, stats_null, a.K_M, a.p_ext, a.n_jobs)
        for s in EE_STATS:
            T_null[s].append(ee[s])
        T_null['N_on'].append(int(np.sum(bg_tscore[on] >= a.count_tscore_cut)))
        pbar.set_postfix(N_on=f"{np.median(T_null['N_on']):.0f}",
                         Ups=f"{np.median(T_null['Ups_max']):.1f}",
                         Nput=f"{np.median(T_null['N_put']):.0f}")
    T_null = {s: np.array(v, dtype=float) for s, v in T_null.items()}

    print(f"[W={a.window:.0f}s] null-trial summary:", flush=True)
    for s in STATS:
        v = T_null[s]
        print(f"  {s:>8} null: mean={v.mean():.2f}  std={v.std():.2f}  "
              f"median={np.median(v):.2f}  max={v.max():.2f}  "
              f"zeros={int(np.sum(v == 0))}/{a.n_null}", flush=True)

    # Median-null thresholds (standard convention) — used for live power readout
    null_med = {s: float(np.median(T_null[s])) for s in STATS}

    # ── 5. Injection trials ───────────────────────────────────────────────────
    print(f"\n[W={a.window:.0f}s] Phase 4/4: injection trials over "
          f"N_inj = {N_inj_grid}  ({a.n_inj_trials} trials each)", flush=True)
    print(f"[W={a.window:.0f}s] live power = fraction of trials exceeding the "
          f"median-null threshold (standard convention, target 0.90)", flush=True)
    T_inj = {s: {} for s in STATS}
    for gi, N_inj in enumerate(N_inj_grid):
        trials = {s: [] for s in STATS}
        pbar = tqdm(range(a.n_inj_trials),
                    desc=f'[W={a.window:.0f}s] N_inj={N_inj:4d} '
                         f'({gi+1}/{len(N_inj_grid)})', mininterval=1.0)
        for _ in pbar:
            on, off    = draw_window(t_sec, a.window, a.K_M, rng)
            Y_bg, X    = bg_scaled[on], bg_scaled[off]
            sig, sig_t = make_signal(sig_arrays, N_inj, phys_scaler, rng)
            Y          = np.vstack([Y_bg, sig])
            ee = run_eagleeye(X, Y, stats_null, a.K_M, a.p_ext, a.n_jobs)
            for s in EE_STATS:
                trials[s].append(ee[s])
            n_on = int(np.sum(bg_tscore[on] >= a.count_tscore_cut)) \
                 + int(np.sum(sig_t >= a.count_tscore_cut))
            trials['N_on'].append(n_on)
            # running detection power vs the median-null threshold
            pbar.set_postfix(
                Non=f"{np.mean(np.array(trials['N_on'])  > null_med['N_on']):.2f}",
                Ups=f"{np.mean(np.array(trials['Ups_max'])> null_med['Ups_max']):.2f}",
                Nput=f"{np.mean(np.array(trials['N_put']) > null_med['N_put']):.2f}")
        for s in STATS:
            T_inj[s][N_inj] = np.array(trials[s], dtype=float)
        # live power-curve line: one row per N_inj as it completes
        pl = "  ".join(f"{s}={np.mean(T_inj[s][N_inj] > null_med[s]):.2f}"
                       for s in STATS)
        print(f"[W={a.window:.0f}s] >>> N_inj={N_inj:4d} power(>med-null):  {pl}",
              flush=True)

    # ── 6. Sensitivity per statistic, both conventions ────────────────────────
    results = {}
    for s in STATS:
        thr_std, pow_std, sens_std, exc_std = sensitivity_from_trials(
            T_null[s], T_inj[s], threshold_q=0.5, target_power=0.9)
        thr_old, pow_old, sens_old, _ = sensitivity_from_trials(
            T_null[s], T_inj[s], threshold_q=1 - a.alpha, target_power=a.target_power)
        results[s] = dict(thr_std=thr_std, pow_std=pow_std, sens_std=sens_std,
                          exc_std=exc_std, thr_old=thr_old, pow_old=pow_old,
                          sens_old=sens_old)

    # A statistic is "degenerate" when its null is effectively a point mass
    # (no resolving power): P(null > median) collapses toward 0.  This happens
    # for the Repechage statistics (T_rep, Lam_max) at small windows where the
    # on-region is too sparse for any cluster to form, so they are exactly 0 for
    # both null and injected trials.  Such stats carry no sensitivity and are
    # reported as N/A rather than as a misleading numeric N@90%.
    degenerate = {s: results[s]['exc_std'] < 0.25 for s in STATS}

    print(f"\n{'Statistic':>10}  {'med-null thr':>12}  {'P(null>thr)':>11}  "
          f"{'N@90% (std)':>11}  {'N@'+str(int(a.target_power*100))+'% (a='+str(a.alpha)+')':>16}")
    for s in STATS:
        r = results[s]
        flag = '  [degenerate null — N/A]' if degenerate[s] else ''
        n90 = '       N/A' if degenerate[s] else f'{r["sens_std"]:>11.1f}'
        n50 = '             N/A' if degenerate[s] else f'{r["sens_old"]:>16.1f}'
        print(f"{s:>10}  {r['thr_std']:>12.2f}  {r['exc_std']:>11.3f}  "
              f"{n90}  {n50}{flag}")

    base = (f"sensitivity_window_{a.flavour}_W{int(a.window)}s"
            f"_KM{a.K_M}_pext{a.p_ext:.0e}")
    if a.ee_tscore_cut > 0:
        base += f"_eecut{a.ee_tscore_cut:g}"

    # ── 7a. Per-statistic distributions + power curves ────────────────────────
    fig, axes = plt.subplots(2, len(STATS), figsize=(3.6 * len(STATS), 8))
    cmap = plt.cm.Oranges(np.linspace(0.4, 0.9, len(N_inj_grid)))
    for j, s in enumerate(STATS):
        r = results[s]
        ax = axes[0, j]
        all_T = np.concatenate([T_null[s]] + list(T_inj[s].values()))
        if s in ('N_on', 'T_rep', 'N_put'):
            bins = np.arange(0, all_T.max() + 2) - 0.5
        else:
            bins = np.linspace(all_T.min(), all_T.max(), 35)
        ax.hist(T_null[s], bins=bins, density=True, alpha=0.6,
                color='steelblue', label='Background only')
        for (N, vals), colour in zip(T_inj[s].items(), cmap):
            ax.hist(vals, bins=bins, density=True, alpha=0.4, color=colour,
                    label=f'N_inj={N}' if j == 0 else None)
        ax.axvline(r['thr_std'], color='red', ls='--',
                   label=f'median null={r["thr_std"]:.1f}')
        ax.set_xlabel(STAT_LABELS[s])
        if j == 0:
            ax.set_ylabel('Density')
        ax.legend(fontsize=6)
        ax.set_title(STAT_LABELS[s], fontsize=9)

        ax = axes[1, j]
        ax.plot(N_inj_grid, r['pow_std'], 'o-', color='steelblue',
                label='median-null / 90% (standard)')
        ax.plot(N_inj_grid, r['pow_old'], 's--', color='grey', alpha=0.7,
                label=f'α={a.alpha} / {int(a.target_power*100)}% (previous)')
        ax.axhline(0.9, color='red', ls=':', lw=1)
        if degenerate[s]:
            ax.text(0.5, 0.5, 'degenerate null\n(N/A at this window)',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=9, color='firebrick', fontweight='bold')
        elif np.isfinite(r['sens_std']):
            ax.axvline(r['sens_std'], color='green', ls='--',
                       label=f'N@90%={r["sens_std"]:.1f}')
        ax.set_xlabel('N injected signal events')
        if j == 0:
            ax.set_ylabel('Detection power')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=6)

    fig.suptitle(f'{a.flavour}  |  W={a.window:.0f}s  (p_hat≈{n_on_samples.mean()/N_total:.3f})  '
                 f'|  K_M={a.K_M}  p_ext={a.p_ext:.0e}  '
                 f'|  {a.n_null} null / {a.n_inj_trials} inj trials',
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(os.path.dirname(__file__), base + '.pdf')
    plt.savefig(out)
    print(f"\nPlot saved to {out}")

    # ── 7b. Raw trials saved for re-analysis (conventions, windows, scans) ────
    npz = {f'null_{s}': T_null[s] for s in STATS}
    for s in STATS:
        for N, vals in T_inj[s].items():
            npz[f'inj_{s}_{N}'] = vals
    npz['meta'] = np.array([a.window, n_on_samples.mean(), N_total, rate,
                            a.K_M, a.p_ext, a.count_tscore_cut, a.ee_tscore_cut])
    npz['N_inj_grid'] = np.array(N_inj_grid)
    # Persist the degeneracy verdict (aligned to `stats`) so downstream analysis
    # renders N/A instead of reading the raw point-mass zeros as real numbers.
    npz['stats'] = np.array(STATS)
    npz['degenerate'] = np.array([degenerate[s] for s in STATS])
    out_npz = os.path.join(os.path.dirname(__file__), base + '.npz')
    np.savez(out_npz, **npz)
    print(f"Raw trial statistics saved to {out_npz}")
    print("\nSummary (N events for 50%-power, standard convention) — "
          "compare EagleEye statistics against the N_on counting baseline.")


if __name__ == '__main__':
    main()
