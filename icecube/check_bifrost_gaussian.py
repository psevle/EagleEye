import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import umap, argparse

module_path = "../eagleeye"
import sys
sys.path.append(module_path)
import EagleEye
from utils_EE import compute_the_null, partitioning_function

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA

import contextlib
import warnings
warnings.filterwarnings('ignore')

colours = {
    "blue": "#6495ec",
    "green": "#93d4c0",
    "pink": "eeadd5",
    "lightblue": "#afbcda",
    "red": "#dc143c",
}

def snr_estimator_Y(result_dict, EE_book):
    over_clusters  = EE_book["Y_OVER_clusters"]
    under_clusters = EE_book["X_OVER_clusters"]

    lenSo = len(np.concatenate([over_clusters[i]['Repechaged'] for i in over_clusters]))
    lenBo = len(np.concatenate([over_clusters[i]['Background'] for i in over_clusters]))
    lenWo = len(np.concatenate([over_clusters[i]['Pruned']     for i in over_clusters]))
    lenWu = len(np.concatenate([under_clusters[i]['Pruned']    for i in under_clusters]))
    n1 = len(result_dict['Upsilon_i_Y'])
    n2 = len(result_dict['Upsilon_i_X'])

    B_hat = lenBo * (n2 - lenWo) / (n1 - lenWu)
    S_hat = lenSo - B_hat

    if B_hat <= 0:
        return np.nan
    return S_hat / np.sqrt(B_hat)

def snr_estimator_X(result_dict, EE_book):
    over_clusters  = EE_book["X_OVER_clusters"]
    under_clusters = EE_book["Y_OVER_clusters"]

    lenSo = len(np.concatenate([over_clusters[i]['Repechaged'] for i in over_clusters]))
    lenBo = len(np.concatenate([over_clusters[i]['Background'] for i in over_clusters]))
    lenWo = len(np.concatenate([over_clusters[i]['Pruned']     for i in over_clusters]))
    lenWu = len(np.concatenate([under_clusters[i]['Pruned']    for i in under_clusters]))
    n1 = len(result_dict['Upsilon_i_X'])
    n2 = len(result_dict['Upsilon_i_Y'])

    B_hat = lenBo * (n2 - lenWo) / (n1 - lenWu)
    S_hat = lenSo - B_hat

    if B_hat <= 0:
        return np.nan
    return S_hat / np.sqrt(B_hat)

def one_background_trial(df_hitspool, df_features, N_Y, N_X, scaler, reducer, K_M=200, p_ext=1e-5, n_jobs=8, stats_null=None, pca=True):
    # Random time-slice of the same width as the usual on-region
    t = df_hitspool["time_seconds"].values
    width = (t.max() - t.min()) * (N_Y / len(t))
    
    while True:
        t0 = np.random.uniform(t.min(), t.max() - width)
        on_mask  = (t >= t0) & (t < t0 + width)
        on_idx   = np.where(on_mask)[0]
        off_idx  = np.where(~on_mask)[0]
        if len(on_idx) >= N_Y and len(off_idx) >= N_X:
            break
    
    Y_idx = np.random.choice(on_idx,  N_Y, replace=False)
    X_idx = np.random.choice(off_idx, N_X, replace=False)

    X_raw = np.array(df_features.iloc[X_idx])
    Y_raw = np.array(df_features.iloc[Y_idx])

    if pca:
        X = reducer.transform(scaler.transform(X_raw))
        Y = reducer.transform(scaler.transform(Y_raw))
    else:
        X = scaler.transform(X_raw)
        Y = scaler.transform(Y_raw)
    
    p = len(Y) / (len(X) + len(Y))
    if stats_null is None:
        stats_null = compute_the_null(p=p, K_M=K_M)
    
    with contextlib.redirect_stdout(None):
        result_dict, stats_null = EagleEye.Soar(X, Y, p_ext=p_ext, n_jobs=n_jobs, stats_null=stats_null, result_dict_in={}, do_IDE=True)
        clusters = partitioning_function(X, Y, result_dict, p_ext=p_ext)
        EE_book = EagleEye.Repechage(X, Y, result_dict, clusters, p_ext=p_ext)
    
    return {
        "Y": snr_estimator_Y(result_dict, EE_book),
        "X": snr_estimator_X(result_dict, EE_book)
    }

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--K_M", required=True, type=int)
    parser.add_argument("--p_ext", required=False, type=float, default=1e-2)
    parser.add_argument("--n_jobs", required=False, type=int, default=1)
    parser.add_argument("--N_trials", required=False, type=int, default=1)
    parser.add_argument("--pca", required=False, action="store_true")
    parser.add_argument("--split", required=False, type=float, default=0.2)
    parser.add_argument("--outfile", required=True, type=str)
    args = parser.parse_args()

    df_hitspool = pd.read_hdf("h5files/hitspool_gw230808.h5", key="HSEventHeader")[:20_000]
    df_hitspool["time_seconds"] = (df_hitspool["time_start_utc_daq"] - min(df_hitspool["time_start_utc_daq"]) ) / 1e10
    
    mid = np.mean(df_hitspool["time_seconds"])
    dt = ( max(df_hitspool["time_seconds"]) - min(df_hitspool["time_seconds"]) ) * args.split
    dt_min, dt_max = mid - dt/2, mid + dt/2
    df_on = df_hitspool[ (df_hitspool["time_seconds"] >= dt_min) & (df_hitspool["time_seconds"] <= dt_max) ]
    df_off = df_hitspool[ (df_hitspool["time_seconds"] < dt_min) | (df_hitspool["time_seconds"] > dt_max) ]
    
    df = pd.read_hdf("h5files/hitspool_gw230808.h5", key="Bifrost_variables")[:20_000]
    df = df.drop(columns=['Run', 'Event', 'SubEvent', 'SubEventStream', 'exists'])

    X_full_raw = np.array(df.iloc[df_off.index])
    scaler = StandardScaler().fit(X_full_raw)
    X_full_scaled = scaler.transform(X_full_raw)

    if args.pca:
        print("Fitting PCA...")
        reducer = PCA(n_components=3, random_state=42)
        reducer.fit(X_full_scaled)
        print("Finished")
    else:
        reducer = None

    N_Y = len(df_on)
    N_X = len(df_off)
    p_fixed = N_Y / (N_Y + N_X)

    print("Computing the null...")
    stats_null = compute_the_null(p=p_fixed, K_M=args.K_M)
    print("Finished")

    X_raw = np.array(df.iloc[df_off.index])
    Y_raw = np.array(df.iloc[df_on.index])

    if args.pca:
        X = reducer.transform(scaler.transform(X_raw))
        Y = reducer.transform(scaler.transform(Y_raw))
    else:
        X = scaler.transform(X_raw)
        Y = scaler.transform(Y_raw)

    print(f"Running EagleEye {args.N_trials} times")
    Lambda_alpha = [one_background_trial(df_hitspool, df, N_Y, N_X, scaler=scaler, reducer=reducer, K_M=args.K_M, p_ext=args.p_ext, n_jobs=args.n_jobs, stats_null=stats_null, pca=args.pca) for _ in tqdm(range(args.N_trials))]
    print("Finished")

    snr_Y = np.array([t["Y"] for t in Lambda_alpha])
    snr_X = np.array([t["X"] for t in Lambda_alpha])
    
    mu_Y, sigma_Y = snr_Y.mean(), snr_Y.std()
    mu_X, sigma_X = snr_X.mean(), snr_X.std()

    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    s, p_norm = stats.normaltest(snr_Y)
    axes[0].hist(snr_Y, bins=30, density=True, color=colours["blue"], label=f"N={len(df)}, K_M={args.K_M}")
    x = np.linspace(np.nanmin(snr_Y), np.nanmax(snr_Y), 300)
    mu, sigma = np.nanmean(snr_Y), np.nanstd(snr_Y)
    axes[0].plot(x, stats.norm.pdf(x, mu, sigma), "k-", label=f"p-val={p_norm:.3f}")
    axes[0].set_xlabel(r"$\Lambda_\alpha$")
    
    stats.probplot(snr_Y, dist="norm", plot=axes[1])
    axes[1].set_title("")
    
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(args.outfile)
