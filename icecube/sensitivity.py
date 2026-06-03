"""
EagleEye sensitivity analysis for IceCube subthreshold event selection.

Assumes the following are already defined in the calling notebook:
    scaler, reducer, stats_null, K_M, p_ext, n_jobs, N_Y, N_X
    df_hitspool, df (background features), df_nue / df_numu / df_nutau (signal)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm
from IPython.utils.capture import capture_output

import EagleEye
from utils_EE import partitioning_function


# ── Test statistic ────────────────────────────────────────────────────────────

def get_T(EE_book):
    """Total number of repechaged Y events across all Y-overdensity clusters."""
    over_clusters = EE_book["Y_OVER_clusters"]
    return sum(len(over_clusters[i]["Repechaged"]) for i in over_clusters)


# ── Core pipeline ─────────────────────────────────────────────────────────────

def run_eagleeye(X, Y, stats_null, K_M, p_ext, n_jobs):
    """Run Soar → partitioning → Repechage and return (result_dict, EE_book)."""
    with capture_output():
        result_dict, _ = EagleEye.Soar(
            X, Y, K_M=K_M, p_ext=p_ext, n_jobs=n_jobs,
            stats_null=stats_null, result_dict_in={}, do_IDE=True,
        )
        clusters = partitioning_function(X, Y, result_dict, p_ext=p_ext)
        EE_book  = EagleEye.Repechage(X, Y, result_dict, clusters, p_ext=p_ext)
    return result_dict, EE_book


# ── Random on/off sampling ────────────────────────────────────────────────────

def sample_XY(df_hitspool, df_features, N_Y, N_X, scaler, reducer):
    """
    Draw a random time-slice of fractional width N_Y/N_total as the on-region,
    subsample to exactly N_Y and N_X events, and project into UMAP space.
    """
    t     = df_hitspool["time_seconds"].values
    width = (t.max() - t.min()) * (N_Y / len(t))

    while True:
        t0      = np.random.uniform(t.min(), t.max() - width)
        on_mask = (t >= t0) & (t < t0 + width)
        on_idx  = np.where(on_mask)[0]
        off_idx = np.where(~on_mask)[0]
        if len(on_idx) >= N_Y and len(off_idx) >= N_X:
            break

    Y_idx = np.random.choice(on_idx,  N_Y, replace=False)
    X_idx = np.random.choice(off_idx, N_X, replace=False)

    X = reducer.transform(scaler.transform(np.array(df_features.iloc[X_idx])))
    Y = reducer.transform(scaler.transform(np.array(df_features.iloc[Y_idx])))
    return X, Y


# ── Null trials ───────────────────────────────────────────────────────────────

def run_null_trials(df_hitspool, df_features, N_Y, N_X, scaler, reducer,
                    stats_null, K_M, p_ext, n_jobs, N_trials=500):
    """
    Run N_trials background-only trials and return the null distribution of T.
    """
    T_null = []
    for _ in tqdm(range(N_trials), desc="Null trials"):
        X, Y = sample_XY(df_hitspool, df_features, N_Y, N_X, scaler, reducer)
        _, EE_book = run_eagleeye(X, Y, stats_null, K_M, p_ext, n_jobs)
        T_null.append(get_T(EE_book))
    return np.array(T_null)


# ── Injection trials ──────────────────────────────────────────────────────────

def run_injection_trials(df_hitspool, df_features, df_signal, N_inj,
                          N_Y, N_X, scaler, reducer,
                          stats_null, K_M, p_ext, n_jobs, N_trials=200):
    """
    Run N_trials signal-injection trials with N_inj events drawn from df_signal
    appended to the on-region Y. Returns the distribution of T under injection.

    df_signal should already have the same columns as df_features
    (i.e. Bifrost_variables with header columns dropped).
    """
    T_inj = []
    for _ in tqdm(range(N_trials), desc=f"Injection N_inj={N_inj}"):
        X, Y = sample_XY(df_hitspool, df_features, N_Y, N_X, scaler, reducer)

        sig_raw = np.array(df_signal.sample(N_inj, replace=True))
        Y_sig   = reducer.transform(scaler.transform(sig_raw))
        Y_inj   = np.vstack([Y, Y_sig])

        _, EE_book = run_eagleeye(X, Y_inj, stats_null, K_M, p_ext, n_jobs)
        T_inj.append(get_T(EE_book))
    return np.array(T_inj)


# ── Power curve and sensitivity ───────────────────────────────────────────────

def compute_sensitivity(T_null, T_inj_dict, alpha=0.05, target_power=0.5):
    """
    Parameters
    ----------
    T_null      : array of T values from background-only trials
    T_inj_dict  : {N_inj: array of T values} from injection trials
    alpha       : false positive rate (e.g. 0.05 for 95% CL)
    target_power: detection efficiency at which to quote sensitivity (default 0.5)

    Returns
    -------
    T_crit        : detection threshold
    N_inj_values  : sorted injection strengths
    power_values  : detection power at each injection strength
    sensitivity_N : N_inj at target_power (interpolated)
    """
    T_crit       = np.quantile(T_null, 1 - alpha)
    N_inj_values = sorted(T_inj_dict.keys())
    power_values = [np.mean(T_inj_dict[N] > T_crit) for N in N_inj_values]

    try:
        f             = interp1d(power_values, N_inj_values,
                                 bounds_error=False, fill_value="extrapolate")
        sensitivity_N = float(f(target_power))
    except Exception:
        sensitivity_N = np.nan

    return T_crit, N_inj_values, power_values, sensitivity_N


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_sensitivity(T_null, T_inj_dict, T_crit, N_inj_values, power_values,
                     sensitivity_N, alpha=0.05, target_power=0.5):

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: null and injection T distributions
    bins = np.linspace(0, max(max(v) for v in T_inj_dict.values()) + 5, 40)
    axes[0].hist(T_null, bins=bins, density=True, alpha=0.6,
                 color="steelblue", label="Background only")
    for N in N_inj_values:
        axes[0].hist(T_inj_dict[N], bins=bins, density=True, alpha=0.3,
                     label=f"N_inj = {N}")
    axes[0].axvline(T_crit, color="red", linestyle="--",
                    label=f"T_crit (α={alpha}): {T_crit:.1f}")
    axes[0].set_xlabel("T  (repechaged events)")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Null vs signal distributions")

    # Right: power curve
    axes[1].plot(N_inj_values, power_values, "o-", color="steelblue")
    axes[1].axhline(target_power, color="red", linestyle="--",
                    label=f"{int(target_power*100)}% power")
    if np.isfinite(sensitivity_N):
        axes[1].axvline(sensitivity_N, color="green", linestyle="--",
                        label=f"Sensitivity: {sensitivity_N:.1f} events")
    axes[1].set_xlabel("N injected signal events")
    axes[1].set_ylabel("Detection power")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Power curve")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    print(f"Detection threshold T_crit (α={alpha}):      {T_crit:.1f} events")
    print(f"Sensitivity at {int(target_power*100)}% power:               {sensitivity_N:.1f} signal events")
    print(f"Flux sensitivity = {sensitivity_N:.1f} / (A_eff [cm²] × T_obs [s])  cm⁻² s⁻¹")


# ── Main entry point ──────────────────────────────────────────────────────────

def run_sensitivity_analysis(
    df_hitspool, df_features, df_signal,
    scaler, reducer, stats_null,
    N_Y, N_X, K_M, p_ext, n_jobs,
    N_inj_grid,
    N_null_trials=500,
    N_inj_trials=200,
    alpha=0.05,
    target_power=0.5,
):
    """
    Full sensitivity pipeline.

    Parameters
    ----------
    df_signal   : signal simulation dataframe (e.g. df_nue, df_numu, df_nutau)
                  with the same columns as df_features
    N_inj_grid  : list of injection strengths to scan, e.g. [5, 10, 20, 50, 100]
    N_null_trials, N_inj_trials : number of pseudo-experiments per point
    alpha       : false positive rate
    target_power: detection efficiency at which to quote sensitivity

    Returns
    -------
    results dict with keys: T_null, T_inj_dict, T_crit, power_values, sensitivity_N
    """
    print("=" * 60)
    print("Step 1/2: Null trials")
    print("=" * 60)
    T_null = run_null_trials(
        df_hitspool, df_features, N_Y, N_X,
        scaler, reducer, stats_null, K_M, p_ext, n_jobs,
        N_trials=N_null_trials,
    )

    print("\n" + "=" * 60)
    print("Step 2/2: Signal injection trials")
    print("=" * 60)
    T_inj_dict = {}
    for N_inj in N_inj_grid:
        T_inj_dict[N_inj] = run_injection_trials(
            df_hitspool, df_features, df_signal, N_inj,
            N_Y, N_X, scaler, reducer, stats_null, K_M, p_ext, n_jobs,
            N_trials=N_inj_trials,
        )

    T_crit, N_inj_values, power_values, sensitivity_N = compute_sensitivity(
        T_null, T_inj_dict, alpha=alpha, target_power=target_power,
    )

    plot_sensitivity(
        T_null, T_inj_dict, T_crit, N_inj_values, power_values,
        sensitivity_N, alpha=alpha, target_power=target_power,
    )

    return {
        "T_null":        T_null,
        "T_inj_dict":    T_inj_dict,
        "T_crit":        T_crit,
        "N_inj_values":  N_inj_values,
        "power_values":  power_values,
        "sensitivity_N": sensitivity_N,
    }
